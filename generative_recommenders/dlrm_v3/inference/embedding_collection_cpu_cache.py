#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单机版 CPU+GPU cache EmbeddingCollection。
特性：
1) 所有表权重常驻 CPU。
2) 可指定 GPU cache 总预算（bytes），按表均分容量。
3) 支持在初始化或运行时预加载外部提供的 hot ids（按容量截断）。
4) 访问路径：优先查 GPU cache，未命中则从 CPU 加载并按 LRU 置换。
5) 提供简单的 cache 统计接口，便于 benchmark。
"""

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

try:
    # 复用 torchrec 的 EmbeddingConfig / DataType 定义，方便与现有配置对接
    from torchrec.modules.embedding_configs import EmbeddingConfig, DataType  # type: ignore
except Exception:  # pragma: no cover - 兜底，若依赖不可用则定义最小结构
    @dataclass
    class EmbeddingConfig:  # type: ignore
        num_embeddings: int
        embedding_dim: int
        name: str
        data_type: str = "fp16"
        feature_names: Optional[List[str]] = None

    class DataType:  # type: ignore
        FP16 = "fp16"
        BF16 = "bf16"
        FP32 = "fp32"

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # type: ignore
except Exception:  # pragma: no cover
    KeyedJaggedTensor = None  # type: ignore


def _dtype_from_cfg(cfg: EmbeddingConfig) -> torch.dtype:
    raw = getattr(cfg, "data_type", None)
    token = getattr(raw, "name", raw) if raw is not None else "fp16"
    text = str(token).lower()
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _bytes_per_row(cfg: EmbeddingConfig) -> int:
    dtype = _dtype_from_cfg(cfg)
    return cfg.embedding_dim * torch.tensor([], dtype=dtype).element_size()


class _TableCache:
    """每张表的 LRU cache 状态。"""

    def __init__(self, capacity: int, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        self.capacity = max(int(capacity), 0)
        self.device = device
        self.dtype = dtype
        self.weight: Optional[torch.Tensor] = (
            torch.empty(self.capacity, dim, device=device, dtype=dtype) if self.capacity > 0 else None
        )
        self.id_to_slot: "OrderedDict[int, int]" = OrderedDict()
        self.free_slots: List[int] = list(range(self.capacity))
        self.slot_to_id: Optional[torch.Tensor] = (
            torch.full((self.capacity,), -1, device=device, dtype=torch.long) if self.capacity > 0 else None
        )

    def current_bytes(self, row_bytes: int) -> int:
        return len(self.id_to_slot) * row_bytes


class CPUGPUCachedEmbeddingCollection(nn.Module):
    """
    简化版 EmbeddingCollection：CPU 存储 + GPU LRU cache。
    - 输入：KeyedJaggedTensor（单机），feature_name -> table 通过 EmbeddingConfig.feature_names 映射。
    - 输出：dict[key] = pooled embedding (sum pooling)。
    """

    def __init__(
        self,
        table_configs: Dict[str, EmbeddingConfig],
        device: torch.device,
        cache_budget_bytes: Optional[int] = None,
        hot_ids: Optional[Dict[str, torch.Tensor]] = None,
        prebuilt_cpu_weights: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if KeyedJaggedTensor is None:
            raise ImportError("torchrec.sparse.jagged_tensor.KeyedJaggedTensor is required for this module.")
        self.device = device
        self.verbose = verbose
        self.table_configs = table_configs
        self.feature_to_table: Dict[str, str] = {}
        for name, cfg in table_configs.items():
            for feat in getattr(cfg, "feature_names", []) or []:
                self.feature_to_table[feat] = name

        # CPU 权重（可共享）
        self.cpu_weights: Dict[str, torch.Tensor] = {}
        self.row_bytes: Dict[str, int] = {}
        for name, cfg in table_configs.items():
            dtype = _dtype_from_cfg(cfg)
            if prebuilt_cpu_weights and name in prebuilt_cpu_weights:
                emb = prebuilt_cpu_weights[name]
                # 简单防御：确保在 CPU，形状一致
                if emb.device.type != "cpu" or emb.shape != (cfg.num_embeddings, cfg.embedding_dim):
                    raise ValueError(f"prebuilt_cpu_weights[{name}] 形状或设备不匹配")
                emb = emb.to(torch.device("cpu"))
                emb.requires_grad_(False)
            else:
                emb = torch.empty(cfg.num_embeddings, cfg.embedding_dim, device=torch.device("cpu"), dtype=dtype)
                emb.requires_grad_(False)
            self.cpu_weights[name] = emb
            self.row_bytes[name] = _bytes_per_row(cfg)

        # GPU cache 容量（按表均分）
        total_budget = cache_budget_bytes if cache_budget_bytes is None else max(int(cache_budget_bytes), 0)
        per_table_budget = None
        if total_budget is not None:
            per_table_budget = total_budget // max(len(table_configs), 1)
        self.caches: Dict[str, _TableCache] = {}
        for name, cfg in table_configs.items():
            row_bytes = max(self.row_bytes[name], 1)
            capacity_rows = 0
            if per_table_budget is not None:
                capacity_rows = per_table_budget // row_bytes
            self.caches[name] = _TableCache(capacity_rows, cfg.embedding_dim, device, _dtype_from_cfg(cfg))

        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._page_ins = 0
        self._usage_samples: List[Tuple[str, int]] = []

        # 预加载 hot ids
        if hot_ids:
            self.preload_hot_ids(hot_ids)
        print(
            "[cache] init done | tables={} cache_budget={} bytes (per-table rows: {})".format(
                len(self.caches),
                cache_budget_bytes if cache_budget_bytes is not None else "unlimited",
                {k: v.capacity for k, v in self.caches.items()},
            )
        )

    # --------------- cache 内部方法 --------------- #
    def _cache_insert(self, table: str, idx: int) -> Optional[int]:
        cache = self.caches[table]
        if cache.capacity <= 0 or cache.weight is None:
            return None
        if idx in cache.id_to_slot:
            cache.id_to_slot.move_to_end(idx, last=True)
            return cache.id_to_slot[idx]
        # 申请槽位
        if cache.free_slots:
            slot = cache.free_slots.pop()
        else:
            evict_id, slot = cache.id_to_slot.popitem(last=False)
            if cache.slot_to_id is not None:
                cache.slot_to_id[slot] = -1
            self._evictions += 1
        cache.id_to_slot[idx] = slot
        if cache.slot_to_id is not None:
            cache.slot_to_id[slot] = idx
        # 加载权重
        self._page_ins += 1
        cache.weight[slot].copy_(self.cpu_weights[table][idx].to(device=self.device, non_blocking=True))
        return slot

    def _lookup_table(self, table: str, ids: torch.Tensor) -> torch.Tensor:
        cfg = self.table_configs[table]
        cache = self.caches[table]
        dim = cfg.embedding_dim
        out = torch.empty((ids.numel(), dim), device=self.device, dtype=_dtype_from_cfg(cfg))

        # 逐 id 处理（Python LRU 管理，方便演示）
        ids_list = ids.tolist()
        for pos, idx in enumerate(ids_list):
            slot = cache.id_to_slot.get(idx)
            if slot is not None and cache.weight is not None:
                cache.id_to_slot.move_to_end(idx, last=True)
                out[pos] = cache.weight[slot]
                self._hits += 1
                continue
            self._misses += 1
            slot = self._cache_insert(table, idx)
            if slot is not None and cache.weight is not None:
                out[pos] = cache.weight[slot]
            else:
                out[pos] = self.cpu_weights[table][idx].to(self.device)
        return out

    # --------------- 公共接口 --------------- #
    def preload_hot_ids(self, hot_ids: Dict[str, torch.Tensor]) -> None:
        """根据外部提供的 hot ids 预热 GPU cache。"""
        for table, ids in hot_ids.items():
            if table not in self.caches:
                continue
            cache = self.caches[table]
            if cache.capacity <= 0:
                continue
            unique_ids = torch.unique(ids).cpu().tolist()
            for idx in unique_ids:
                if len(cache.id_to_slot) >= cache.capacity and idx in cache.id_to_slot:
                    continue
                if len(cache.id_to_slot) >= cache.capacity and idx not in cache.id_to_slot:
                    # 如果满了，后续插入会触发 LRU 置换
                    pass
                self._cache_insert(table, int(idx))
            print(f"[cache] preload | table={table} loaded={min(len(unique_ids), cache.capacity)} / {cache.capacity}")

    def forward(self, batch: "KeyedJaggedTensor") -> Dict[str, torch.Tensor]:
        keys = list(batch.keys()) if isinstance(batch.keys(), (list, tuple)) else list(batch.keys())
        if self.verbose:
            total_vals = int(batch.values().numel()) if hasattr(batch, "values") else -1
            print(f"[cache] forward | keys={len(keys)} total_values={total_vals}")
        outputs: Dict[str, torch.Tensor] = {}
        for key in keys:
            table = self.feature_to_table.get(key)
            if table is None:
                continue
            values = batch[key].values()
            lengths = batch[key].lengths()
            if values is None or lengths is None:
                continue
            emb = self._lookup_table(table, values)
            # sum pooling
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).to(torch.long)  # type: ignore
            pooled = torch.zeros((lengths.numel(), emb.shape[1]), device=self.device, dtype=emb.dtype)
            for i in range(lengths.numel()):
                start = int(offsets[i].item())
                end = int(offsets[i + 1].item())
                if end > start:
                    pooled[i] = emb[start:end].sum(dim=0)
            outputs[key] = pooled
        return outputs

    def record_cache_usage(self) -> None:
        for name, cache in self.caches.items():
            self._usage_samples.append((name, cache.current_bytes(self.row_bytes[name])))

    def cache_stats(self) -> Dict[str, float]:
        if self._usage_samples:
            per_table_bytes: Dict[str, List[int]] = {}
            for name, used in self._usage_samples:
                per_table_bytes.setdefault(name, []).append(used)
            avg_mb = sum(sum(v) / len(v) for v in per_table_bytes.values()) / max(len(per_table_bytes), 1) / (
                1024 ** 2
            )
            max_mb = max(max(v) for v in per_table_bytes.values()) / (1024 ** 2)
        else:
            avg_mb = 0.0
            max_mb = 0.0
        total_bytes = sum(cache.current_bytes(self.row_bytes[name]) for name, cache in self.caches.items())
        return {
            "avg_mb": avg_mb,
            "max_mb": max_mb,
            "bytes": int(total_bytes),
            "evictions": int(self._evictions),
            "page_ins": int(self._page_ins),
            "hits": int(self._hits),
            "misses": int(self._misses),
        }


def load_hot_ids_from_json(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """支持 JSON {table: [ids]}，便于外部指定 hot embedding。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, torch.Tensor] = {}
    for name, ids in data.items():
        if isinstance(ids, list):
            tensor = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            tensor = torch.as_tensor(ids, dtype=torch.long, device=device)
        result[name] = tensor
    return result


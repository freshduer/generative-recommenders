#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark embedding throughput under GPU embedding-cache budgets."""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

PROJECT_ROOT = os.path.abspath(
    "/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference"
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from custom_sharding import CustomEmbeddingCollection, ShardingType  # noqa: E402
from torchrec.distributed.planner.types import ParameterConstraints  # noqa: E402
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType  # noqa: E402
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # noqa: E402


FEATURE_GROUPS: Dict[str, Dict[str, object]] = {
    "scalars": {
        "names": ["v", "d"],
        "avg_len": 1,
        "variance": 0,
    },
    "history": {
        "names": [
            "uih_post_id",
            "uih_action_time",
            "uih_weight",
            "uih_owner_id",
            "uih_watchtime",
            "uih_surface_type",
            "uih_video_length",
            "viewer_id",
            "dummy_contexual",
        ],
        "avg_len": 1400,
        "variance": 100,
    },
    "items": {
        "names": [
            "item_post_id",
            "item_owner_id",
            "item_surface_type",
            "item_video_length",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ],
        "avg_len": 128,
        "variance": 10,
    },
}


@dataclass
class BenchConfig:
    batch_size: int = 32
    warmup_steps: int = 5
    measure_steps: int = 50
    num_embeddings: int = 60_000_000
    embedding_dim: int = 256
    hot_ratio: float = 0.10
    access_ratio: float = 0.90
    prefill_ratio: float = 0.10
    seed: int = 1337
    prebuilt_chunk_rows: int = 5_000_000
    prebuilt_pin_memory: bool = False
    resample_per_step: bool = False


@dataclass
class BenchmarkLimit:
    label: str
    bytes: Optional[int]


def _parse_limit(value: str) -> BenchmarkLimit:
    text = value.strip().lower()
    if text in {"none", "unlimited", "inf", "infinite"}:
        return BenchmarkLimit(label="unlimited", bytes=None)
    units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            number = float(text[: -len(suffix)])
            return BenchmarkLimit(label=f"{number:g}{suffix.upper()}", bytes=int(number * multiplier))
    if text.endswith("b"):
        number = float(text[:-1])
        return BenchmarkLimit(label=f"{number:g}B", bytes=int(number))
    if text.endswith("k"):
        number = float(text[:-1])
        return BenchmarkLimit(label=f"{number:g}KB", bytes=int(number * 1024))
    if text.endswith("m"):
        number = float(text[:-1])
        return BenchmarkLimit(label=f"{number:g}MB", bytes=int(number * 1024**2))
    if text.endswith("g"):
        number = float(text[:-1])
        return BenchmarkLimit(label=f"{number:g}GB", bytes=int(number * 1024**3))
    if not value.replace(".", "", 1).isdigit():
        raise ValueError(f"Could not parse limit value: {value}")
    number = float(text)
    return BenchmarkLimit(label=f"{number:g}MB", bytes=int(number * 1024**2))


def _format_limit(limit: BenchmarkLimit) -> str:
    return limit.label


def _dtype_num_bytes(data_type: object) -> int:
    mapping = {
        getattr(DataType, "FP16", None): 2,
        getattr(DataType, "BF16", None): 2,
        getattr(DataType, "FP32", None): 4,
        getattr(DataType, "FP64", None): 8,
    }
    for key, size in mapping.items():
        if key is not None and data_type == key:
            return size
    if isinstance(data_type, str):
        lowered = data_type.lower()
        if lowered in {"fp16", "float16", "half"}:
            return 2
        if lowered in {"bf16", "bfloat16"}:
            return 2
        if lowered in {"fp64", "float64", "double"}:
            return 8
    return 4


def _estimate_table_bytes(tables: Dict[str, EmbeddingConfig]) -> Dict[str, int]:
    estimates: Dict[str, int] = {}
    for name, cfg in tables.items():
        dtype_size = _dtype_num_bytes(getattr(cfg, "data_type", None))
        estimates[name] = cfg.num_embeddings * cfg.embedding_dim * dtype_size
    return estimates


def _cuda_mem_info(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    try:
        free, total = torch.cuda.mem_get_info(idx)
    except (AttributeError, RuntimeError, TypeError):
        props = torch.cuda.get_device_properties(idx)
        total = getattr(props, "total_memory", 0)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        free = max(total - max(allocated, reserved), 0)
    return int(free), int(total)


def _is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _maybe_init_dist(backend: str, local_rank: int) -> Tuple[bool, int, int]:
    if _is_dist_available():
        return True, dist.get_rank(), dist.get_world_size()
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    should_init = world_size_env > 1 or os.environ.get("MASTER_ADDR") is not None
    if should_init:
        print("[benchmark] initializing torch.distributed...")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        try:
            dist.init_process_group(backend=backend, device_id=local_rank)
        except TypeError:
            dist.init_process_group(backend=backend)
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def _log_device_memory(tag: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(
        f"[benchmark] {tag} | cuda_allocated={allocated / (1024 ** 2):.2f} MB "
        f"reserved={reserved / (1024 ** 2):.2f} MB"
    )


def _build_tables_config(cfg: BenchConfig) -> Dict[str, EmbeddingConfig]:
    return {
        "table_post_id": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_post_id",
            data_type=DataType.FP16,
            feature_names=["uih_post_id", "item_post_id"],
        ),
        "table_user_id": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_user_id",
            data_type=DataType.FP16,
            feature_names=["uih_owner_id", "item_owner_id", "viewer_id"],
        ),
        "table_video_meta": EmbeddingConfig(
            num_embeddings=max(1, cfg.num_embeddings // 10),
            embedding_dim=cfg.embedding_dim,
            name="table_video_meta",
            data_type=DataType.FP16,
            feature_names=[
                "uih_surface_type",
                "item_surface_type",
                "uih_video_length",
                "item_video_length",
            ],
        ),
        "table_context_stats": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_context_stats",
            data_type=DataType.FP16,
            feature_names=[
                "v",
                "d",
                "dummy_contexual",
                "uih_action_time",
                "uih_weight",
                "uih_watchtime",
                "item_action_weight",
                "item_target_watchtime",
                "item_query_time",
            ],
        ),
        "table_user_behavior": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_user_behavior",
            data_type=DataType.FP16,
            feature_names=[
                "user_shortterm_clk",
                "user_shortterm_watch",
                "user_history_like",
                "user_recent_interactions"
            ],
        ),
        "table_item_popularity": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_item_popularity",
            data_type=DataType.FP16,
            feature_names=[
                "item_popularity_long",
                "item_popularity_short",
            ],
        ),
        "table_item_topic": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_item_topic",
            data_type=DataType.FP16,
            feature_names=["item_topic_id"],
        ),
        "table_item_quality": EmbeddingConfig(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            name="table_item_quality",
            data_type=DataType.FP16,
            feature_names=[
                "item_quality_score",
                "item_safeness_score",
            ],
        ),
    }


def _feature_vocab_sizes(tables: Dict[str, EmbeddingConfig]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for table_name, cfg in tables.items():
        for feat in cfg.feature_names:
            mapping[feat] = cfg.num_embeddings
    return mapping


def _feature_to_table(tables: Dict[str, EmbeddingConfig]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for table_name, cfg in tables.items():
        for feat in cfg.feature_names:
            mapping[feat] = table_name
    return mapping


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    return gen


def generate_skewed_indices(
    num_embeddings: int,
    total_count: int,
    device: torch.device,
    cfg: BenchConfig,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if total_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    num_hot = max(1, int(num_embeddings * cfg.hot_ratio))
    rand = torch.rand(total_count, device=device, generator=generator)
    is_hot = rand < cfg.access_ratio
    num_hot_access = int(is_hot.sum().item())
    num_cold_access = total_count - num_hot_access
    indices = torch.empty(total_count, dtype=torch.long, device=device)
    if num_hot_access > 0:
        indices[is_hot] = torch.randint(0, num_hot, (num_hot_access,), device=device, generator=generator)
    if num_cold_access > 0:
        indices[~is_hot] = torch.randint(
            num_hot,
            num_embeddings,
            (num_cold_access,),
            device=device,
            generator=generator,
        )
    return indices


def get_kjt_batch(
    batch_size: int,
    device: torch.device,
    feature_vocab_sizes: Dict[str, int],
    cfg: BenchConfig,
    generator: Optional[torch.Generator] = None,
) -> KeyedJaggedTensor:
    all_keys: List[str] = []
    all_lengths: List[torch.Tensor] = []
    all_values: List[torch.Tensor] = []
    for spec in FEATURE_GROUPS.values():
        base_len = int(spec["avg_len"])  # type: ignore[index]
        var = int(spec["variance"])  # type: ignore[index]
        for feat_name in spec["names"]:  # type: ignore[index]
            all_keys.append(feat_name)
            if var > 0:
                low = max(1, base_len - var)
                high = base_len + var
                lengths = torch.randint(
                    low,
                    high,
                    (batch_size,),
                    dtype=torch.int32,
                    device=device,
                    generator=generator,
                )
            else:
                lengths = torch.full((batch_size,), base_len, dtype=torch.int32, device=device)
            total_vals = int(lengths.sum().item())
            vocab_size = feature_vocab_sizes.get(feat_name, cfg.num_embeddings)
            values = generate_skewed_indices(vocab_size, total_vals, device, cfg, generator)
            all_lengths.append(lengths)
            all_values.append(values)
    final_lengths = torch.cat(all_lengths)
    final_values = torch.cat(all_values)
    kjt = KeyedJaggedTensor.from_lengths_sync(keys=all_keys, values=final_values, lengths=final_lengths)
    return kjt.to(device)


def _compute_latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(q: float) -> float:
        if n == 1:
            return sorted_lat[0]
        rank = (n - 1) * q / 100.0
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        weight = rank - low
        if high >= n:
            high = n - 1
        if low == high:
            return sorted_lat[low]
        return sorted_lat[low] * (1.0 - weight) + sorted_lat[high] * weight

    avg_ms = sum(sorted_lat) / n * 1000.0
    return {
        "avg_ms": avg_ms,
        "p50_ms": percentile(50.0) * 1000.0,
        "p95_ms": percentile(95.0) * 1000.0,
        "p99_ms": percentile(99.0) * 1000.0,
    }


def _reduce_cache_stats(stats_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        stats_list = []
    defaults = {
        "avg_mb": 0.0,
        "max_mb": 0.0,
        "bytes": 0,
        "evictions": 0,
        "page_ins": 0,
        "hits": 0,
        "misses": 0,
    }
    normalized = [stat if stat else defaults for stat in stats_list] or [defaults]
    avg_mb = sum(stat.get("avg_mb", 0.0) for stat in normalized) / len(normalized)
    max_mb = max(stat.get("max_mb", 0.0) for stat in normalized)
    return {
        "avg_mb": avg_mb,
        "max_mb": max_mb,
        "bytes": sum(stat.get("bytes", 0) for stat in normalized),
        "evictions": int(sum(stat.get("evictions", 0) for stat in normalized)),
        "page_ins": int(sum(stat.get("page_ins", 0) for stat in normalized)),
        "hits": int(sum(stat.get("hits", 0) for stat in normalized)),
        "misses": int(sum(stat.get("misses", 0) for stat in normalized)),
    }


def _prefill_range_map(
    tables: Dict[str, EmbeddingConfig],
    ratio: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    mapping: Dict[str, torch.Tensor] = {}
    if ratio <= 0.0:
        return mapping
    for table_name, cfg in tables.items():
        limit = max(1, int(cfg.num_embeddings * ratio))
        limit = min(limit, cfg.num_embeddings)
        mapping[table_name] = torch.arange(0, limit, device=device, dtype=torch.long)
    return mapping


def _prefill_from_batch(
    batch: KeyedJaggedTensor,
    tables: Dict[str, EmbeddingConfig],
    ratio: float,
    device: torch.device,
    max_ids_per_table: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    if ratio <= 0.0:
        return {}
    feature_to_table = _feature_to_table(tables)
    per_table_values: Dict[str, List[torch.Tensor]] = {name: [] for name in tables.keys()}
    keys = batch.keys() if isinstance(batch.keys(), list) else list(batch.keys())
    for key in keys:
        table_name = feature_to_table.get(key)
        if table_name is None:
            continue
        values = batch[key].values()
        if values is None or values.numel() == 0:
            continue
        per_table_values[table_name].append(values)
    mapping: Dict[str, torch.Tensor] = {}
    for table_name, tensors in per_table_values.items():
        if not tensors:
            continue
        merged = torch.cat(tensors)
        if merged.numel() == 0:
            continue
        unique, counts = torch.unique(merged, sorted=False, return_counts=True)
        desired = max(1, int(tables[table_name].num_embeddings * ratio))
        if max_ids_per_table is not None:
            cap = max_ids_per_table.get(table_name)
            if cap is not None:
                desired = min(desired, cap)
        desired = min(desired, unique.numel())
        if desired <= 0:
            continue
        if desired < unique.numel():
            # Pick the most frequent IDs observed in the synthetic batch.
            # topk keeps indices unsorted; sort for deterministic preload order.
            topk = torch.topk(counts.to(torch.float32), desired).indices
            selected = unique[topk]
            selected = torch.sort(selected).values
        else:
            selected = torch.sort(unique).values
        mapping[table_name] = selected.to(device=device, dtype=torch.long)
    return mapping


def _expand_prefill_to_budget(
    prefill: Dict[str, torch.Tensor],
    tables: Dict[str, EmbeddingConfig],
    per_table_cap: Optional[Dict[str, int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not per_table_cap:
        return prefill
    expanded: Dict[str, torch.Tensor] = dict(prefill)
    for table_name, cap in per_table_cap.items():
        cfg = tables.get(table_name)
        if cfg is None:
            continue
        target = int(cap or 0)
        if target <= 0:
            expanded.pop(table_name, None)
            continue
        target = min(target, cfg.num_embeddings)
        current = expanded.get(table_name)
        if current is None or current.numel() == 0:
            expanded[table_name] = torch.arange(0, target, device=device, dtype=torch.long)
            continue
        if current.device != device:
            current = current.to(device)
        unique_current = torch.unique(current, sorted=True)
        if unique_current.numel() >= target:
            expanded[table_name] = unique_current[:target]
            continue
        needed = target - unique_current.numel()
        extras: List[torch.Tensor] = []
        cursor = 0
        chunk_size = max(1, min(target, 1_048_576))
        max_id = cfg.num_embeddings
        # Fill the remaining slots with the lowest unused IDs while keeping existing hot IDs.
        while needed > 0 and cursor < max_id:
            chunk_end = min(cursor + chunk_size, max_id)
            chunk = torch.arange(cursor, chunk_end, device=device, dtype=torch.long)
            if unique_current.numel() > 0:
                start_idx = int(torch.searchsorted(unique_current, unique_current.new_tensor(cursor), right=False).item())
                end_idx = int(torch.searchsorted(unique_current, unique_current.new_tensor(chunk_end), right=False).item())
                if end_idx > start_idx:
                    taken = unique_current[start_idx:end_idx] - cursor
                    mask = torch.ones(chunk.shape[0], dtype=torch.bool, device=device)
                    mask[taken.to(torch.long)] = False
                    chunk = chunk[mask]
            if chunk.numel() == 0:
                cursor = chunk_end
                continue
            take = min(needed, chunk.numel())
            extras.append(chunk[:take])
            needed -= take
            cursor = chunk_end
        if extras:
            extra_ids = torch.cat(extras)
            combined = torch.cat((unique_current, extra_ids))
            expanded[table_name] = torch.sort(combined).values[:target]
        else:
            expanded[table_name] = unique_current
    return expanded


class NoInitEmbedding(torch.nn.Embedding):
    """Embedding layer that skips the default weight initialization."""

    def reset_parameters(self) -> None:  # type: ignore[override]
        # Leave weights uninitialized to avoid the heavy uniform_ fill.
        return


def _touch_embedding_chunks(
    weight: torch.Tensor,
    chunk_rows: int,
    table_name: str,
    rank: int,
) -> None:
    if chunk_rows <= 0 or chunk_rows >= weight.shape[0]:
        return
    total_rows = weight.shape[0]
    total_chunks = math.ceil(total_rows / chunk_rows)
    log_step = max(1, total_chunks // 5)
    for chunk_idx, start in enumerate(range(0, total_rows, chunk_rows), 1):
        end = min(start + chunk_rows, total_rows)
        _ = weight[start:end]
        if rank == 0 and (
            chunk_idx == 1
            or chunk_idx == total_chunks
            or chunk_idx % log_step == 0
        ):
            print(
                f"[benchmark] prebuilt chunk | table={table_name} "
                f"chunk={chunk_idx}/{total_chunks} rows={end - start}"
            )


def _build_prebuilt_embeddings(
    tables_config: Dict[str, EmbeddingConfig],
    device: torch.device,
    sharding: ShardingType,
    rank: int,
    cfg: BenchConfig,
) -> Dict[str, torch.nn.Embedding]:
    prebuilt: Dict[str, torch.nn.Embedding] = {}
    total_tables = len(tables_config)
    for idx, (name, table_cfg) in enumerate(tables_config.items(), 1):
        if sharding != ShardingType.ROW_WISE:
            continue
        raw_dtype = getattr(table_cfg, "data_type", DataType.FP16)
        if hasattr(raw_dtype, "name"):
            dtype_token = getattr(raw_dtype, "name")
        else:
            dtype_token = str(raw_dtype)
        if str(dtype_token).lower() in {"fp16", "half", "float16"}:
            target_dtype = torch.float16
        elif str(dtype_token).lower() in {"bf16", "bfloat16"}:
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32
        emb = NoInitEmbedding(
            table_cfg.num_embeddings,
            table_cfg.embedding_dim,
            device=torch.device("cpu"),
            dtype=target_dtype,
        )
        emb.weight.requires_grad_(False)
        weight = emb.weight.detach()
        if getattr(cfg, "prebuilt_pin_memory", False) and torch.cuda.is_available():
            weight = weight.pin_memory()
        emb.weight = torch.nn.Parameter(weight, requires_grad=False)
        _touch_embedding_chunks(
            emb.weight,
            max(0, getattr(cfg, "prebuilt_chunk_rows", 0)),
            name,
            rank,
        )
        prebuilt[name] = emb
        if rank == 0 and idx <= total_tables:
            print(
                f"[benchmark] prebuilt embeddings | table={name} ({idx}/{total_tables}) "
                f"rows={table_cfg.num_embeddings} dim={table_cfg.embedding_dim}"
            )
    return prebuilt


def benchmark_limit(
    cfg: BenchConfig,
    limit: BenchmarkLimit,
    tables_config: Dict[str, EmbeddingConfig],
    feature_vocab_sizes: Dict[str, int],
    device: torch.device,
    sharding: ShardingType,
    prebuilt_tables: Optional[Dict[str, torch.nn.Embedding]] = None,
    enable_cache: bool = True,
) -> Optional[Dict[str, float]]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dist_enabled = True
    else:
        rank = 0
        world_size = 1
        dist_enabled = False
    if rank == 0:
        if enable_cache:
            print(f"[benchmark] === limit={limit.label} | sharding={sharding.value} | world_size={world_size} ===")
        else:
            print(f"[benchmark] === CPU baseline (cache disabled) | world_size={world_size} ===")
        _log_device_memory("before_model", device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.empty_cache()
    table_size_bytes = _estimate_table_bytes(tables_config)
    row_size_bytes = {
        name: tables_config[name].embedding_dim
        * _dtype_num_bytes(getattr(tables_config[name], "data_type", None))
        for name in tables_config
    }
    effective_limit_bytes: Optional[int] = limit.bytes
    per_table_prefill_cap: Optional[Dict[str, int]] = None
    if device.type == "cuda":
        free_mem, total_mem = _cuda_mem_info(device)
        safety_margin = max(int(total_mem * 0.1), 512 * 1024**2)
        if limit.bytes is None:
            required_bytes = sum(table_size_bytes.values())
            available = max(0, free_mem - safety_margin)
            if required_bytes > available:
                if rank == 0:
                    print(
                        f"[benchmark] skipping limit={limit.label} | requires {required_bytes / (1024 ** 3):.2f} GB "
                        f"but only {available / (1024 ** 3):.2f} GB free after safety margin"
                    )
                return None
        else:
            available_for_cache = max(0, free_mem - safety_margin)
            if available_for_cache <= 0:
                if rank == 0:
                    print(
                        f"[benchmark] skipping limit={limit.label} | insufficient free GPU memory ({free_mem / (1024 ** 3):.2f} GB)"
                    )
                return None
            if limit.bytes > available_for_cache:
                if rank == 0:
                    print(
                        f"[benchmark] clipping budget from {limit.bytes / (1024 ** 3):.2f} GB "
                        f"to {available_for_cache / (1024 ** 3):.2f} GB based on free memory"
                    )
                effective_limit_bytes = available_for_cache
            table_count = max(len(tables_config), 1)
            per_table_budget = effective_limit_bytes // table_count if effective_limit_bytes is not None else 0
            per_table_prefill_cap = {}
            for name, table_cfg in tables_config.items():
                row_bytes = table_cfg.embedding_dim * _dtype_num_bytes(getattr(table_cfg, "data_type", None))
                if row_bytes <= 0:
                    per_table_prefill_cap[name] = table_cfg.num_embeddings
                else:
                    cap = per_table_budget // row_bytes if per_table_budget > 0 else 0
                    per_table_prefill_cap[name] = min(cap, table_cfg.num_embeddings) if cap > 0 else 0
    if effective_limit_bytes is not None and effective_limit_bytes <= 0:
        effective_limit_bytes = None
    constraints = {name: ParameterConstraints(sharding_types=[sharding.value]) for name in tables_config.keys()}
    if rank == 0:
        print("[benchmark] initializing embedding collection (allocating CPU weights & GPU caches)...")
    model = CustomEmbeddingCollection(
        table_config=tables_config,
        constraints=constraints,
        device=device,
        embedding_budget_bytes=effective_limit_bytes,
        prebuilt_tables=prebuilt_tables,
        cache_enabled=enable_cache,
    )
    if rank == 0:
        print("[benchmark] embedding collection ready")
        print("[benchmark] constructing synthetic batch...")
    generator = _torch_generator(device, cfg.seed + rank)

    def sample_batch() -> KeyedJaggedTensor:
        return get_kjt_batch(cfg.batch_size, device, feature_vocab_sizes, cfg, generator)

    base_batch = sample_batch()
    if rank == 0:
        print("[benchmark] synthetic batch ready")
        kjt_keys = base_batch.keys()
        key_count = len(kjt_keys) if isinstance(kjt_keys, list) else len(list(kjt_keys))
        total_values = int(base_batch.values().numel()) if hasattr(base_batch, "values") else 0
        print(
            f"[benchmark] batch ready | keys={key_count} total_values={total_values} "
            f"batch_size={cfg.batch_size}"
        )
    if (
        effective_limit_bytes is not None
        and effective_limit_bytes > 0
        and device.type == "cuda"
        and enable_cache
    ):
        if rank == 0:
            print(f"[benchmark] preloading hot ids ratio={cfg.prefill_ratio}")
        preload_start = time.perf_counter()
        observed_prefill = _prefill_from_batch(
            base_batch,
            tables_config,
            cfg.prefill_ratio,
            device,
            max_ids_per_table=per_table_prefill_cap,
        )
        if not observed_prefill:
            observed_prefill = _prefill_range_map(tables_config, cfg.prefill_ratio, device)
            if rank == 0:
                print("[benchmark] preload_fallback=range_map (no ids from batch)")
        observed_prefill = _expand_prefill_to_budget(
            observed_prefill,
            tables_config,
            per_table_prefill_cap,
            device,
        )
        if rank == 0 and per_table_prefill_cap:
            print("[benchmark] preload adjusted to fill cache budget per table")
        preload_counts = {name: tensor.numel() for name, tensor in observed_prefill.items()}
        for table_name, ids in observed_prefill.items():
            row_bytes = max(row_size_bytes.get(table_name, cfg.embedding_dim * 2), 1)
            target_chunk_bytes = 128 * 1024**2
            chunk_size = max(1, min(ids.numel(), target_chunk_bytes // row_bytes))
            splits = ids.split(chunk_size)
            num_chunks = len(splits)
            milestones = {1, num_chunks}
            step = max(1, num_chunks // 5)
            for m in range(step, num_chunks, step):
                milestones.add(m)
            processed = 0
            for idx, chunk in enumerate(splits, 1):
                if chunk.numel() == 0:
                    continue
                model.preload_hot_ids({table_name: chunk.contiguous()})
                processed += chunk.numel()
                if rank == 0:
                    if idx in milestones or processed >= preload_counts[table_name]:
                        pct = processed / max(preload_counts[table_name], 1)
                        print(
                            f"[benchmark] preload progress | table={table_name} "
                            f"chunk={idx}/{num_chunks} loaded={processed}/{preload_counts[table_name]} ({pct:.1%})"
                        )
        if rank == 0:
            preload_elapsed = time.perf_counter() - preload_start
            preload_bytes = {
                name: preload_counts.get(name, 0) * row_size_bytes.get(name, 0)
                for name in preload_counts
            }
            total_ids = sum(preload_counts.values())
            total_mb = sum(preload_bytes.values()) / (1024 ** 2)
            counts_str = ", ".join(
                f"{name}:{count} (~{preload_bytes.get(name, 0) / (1024 ** 2):.2f} MB)"
                for name, count in preload_counts.items()
            ) or "none"
            print(
                f"[benchmark] preload complete | ids={total_ids} (~{total_mb:.2f} MB) "
                f"elapsed={preload_elapsed:.3f}s | {counts_str}"
            )
            _log_device_memory("after_preload", device)
        del observed_prefill
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with torch.no_grad():
        warmup_start = time.perf_counter()
        for step in range(cfg.warmup_steps):
            if rank == 0:
                print(f"[benchmark] warmup step {step + 1}/{cfg.warmup_steps}")
            current_batch = sample_batch() if cfg.resample_per_step else base_batch
            model(current_batch)
        if rank == 0:
            print(f"[benchmark] warmup elapsed={time.perf_counter() - warmup_start:.3f}s")
            _log_device_memory("after_warmup", device)
            warmup_stats = model.cache_stats()
            print(f"[benchmark] cache stats after warmup={warmup_stats}")
    if dist_enabled:
        dist.barrier()
    latencies: List[float] = []
    with torch.no_grad():
        for step in range(cfg.measure_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            current_batch = sample_batch() if cfg.resample_per_step else base_batch
            t0 = time.perf_counter()
            model(current_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            model.record_cache_usage()
            if rank == 0 and ((step + 1) % max(1, cfg.measure_steps // 5) == 0 or step == 0):
                print(
                    f"[benchmark] measure step {step + 1}/{cfg.measure_steps} "
                    f"latency_ms={latencies[-1] * 1000.0:.3f}"
                )
    stats = model.cache_stats()
    if rank == 0:
        print(f"[benchmark] local cache stats={stats}")
        _log_device_memory("after_measure", device)
    local_elapsed = sum(latencies)
    if dist_enabled:
        gathered_latencies = [None] * world_size if rank == 0 else None
        gathered_stats = [None] * world_size if rank == 0 else None
        gathered_elapsed = [None] * world_size if rank == 0 else None
        dist.gather_object(latencies, gathered_latencies, dst=0)
        dist.gather_object(stats, gathered_stats, dst=0)
        dist.gather_object(local_elapsed, gathered_elapsed, dst=0)
    else:
        gathered_latencies = [latencies]
        gathered_stats = [stats]
        gathered_elapsed = [local_elapsed]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist_enabled:
        dist.barrier()
    if rank != 0:
        return None
    merged_latencies: List[float] = []
    for item in gathered_latencies:
        merged_latencies.extend(item or [])
    lat_stats = _compute_latency_stats(merged_latencies)
    global_elapsed = max(val or 0.0 for val in gathered_elapsed)
    total_requests = cfg.batch_size * cfg.measure_steps * world_size
    throughput = total_requests / max(global_elapsed, 1e-9)
    cache_stats = _reduce_cache_stats([entry or {} for entry in gathered_stats])
    return {
        "limit": _format_limit(limit),
        "effective_limit_mb": 0.0 if effective_limit_bytes is None else effective_limit_bytes / (1024 ** 2),
        "throughput": throughput,
        "avg_gpu_mb": cache_stats["avg_mb"],
        "max_gpu_mb": cache_stats["max_mb"],
        "evictions": cache_stats["evictions"],
        "page_ins": cache_stats["page_ins"],
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "lat_p50": lat_stats["p50_ms"],
        "lat_p95": lat_stats["p95_ms"],
        "lat_p99": lat_stats["p99_ms"],
        "lat_avg": lat_stats["avg_ms"],
    }


def run_benchmark(
    cfg: BenchConfig,
    limits: Sequence[BenchmarkLimit],
    sharding: ShardingType,
    include_cpu_baseline: bool = False,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist_enabled, rank, world_size = _maybe_init_dist(backend, local_rank)
    if not dist_enabled:
        print("[benchmark] Running in single-process mode (no torch.distributed backend).")
    tables_config = _build_tables_config(cfg)
    feature_vocab_sizes = _feature_vocab_sizes(tables_config)
    prebuilt_tables: Optional[Dict[str, torch.nn.Embedding]] = None
    if sharding == ShardingType.ROW_WISE:
        if rank == 0:
            print("[benchmark] preparing shared CPU embeddings once for all limits...")
        prebuilt_tables = _build_prebuilt_embeddings(tables_config, device, sharding, rank, cfg)
    results: List[Dict[str, float]] = []
    for limit in limits:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        summary = benchmark_limit(
            cfg,
            limit,
            tables_config,
            feature_vocab_sizes,
            device,
            sharding,
            prebuilt_tables=prebuilt_tables,
        )
        if rank == 0 and summary is not None:
            results.append(summary)
    if include_cpu_baseline:
        cpu_limit = BenchmarkLimit(label="CPU", bytes=0)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        cpu_summary = benchmark_limit(
            cfg,
            cpu_limit,
            tables_config,
            feature_vocab_sizes,
            device,
            sharding,
            prebuilt_tables=prebuilt_tables,
            enable_cache=False,
        )
        if rank == 0 and cpu_summary is not None:
            results.append(cpu_summary)
    if rank == 0:
        print("==== Embedding Cache Budget Benchmark ====")
        print(f"world_size: {world_size}, device: {device}, sharding: {sharding.value}")
        print(
            "tables: {num} | embed_dim: {dim} | batch: {batch} | warmup: {warmup} | measure: {measure}".format(
                num=len(tables_config),
                dim=cfg.embedding_dim,
                batch=cfg.batch_size,
                warmup=cfg.warmup_steps,
                measure=cfg.measure_steps,
            )
        )
        header = (
            f"{'limit':>12} {'effective_mb':>14} {'throughput (samples/s)':>24} {'avg_gpu_mb':>12} {'max_gpu_mb':>12} "
            f"{'evict':>8} {'page_in':>10} {'hits':>10} {'misses':>10} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10}"
        )
        print(header)
        for row in results:
            print(
                f"{row['limit']:>12} {row['effective_limit_mb']:>14.2f} {row['throughput']:>24.2f} {row['avg_gpu_mb']:>12.2f} {row['max_gpu_mb']:>12.2f} "
                f"{row['evictions']:>8} {row['page_ins']:>10} {row['hits']:>10} {row['misses']:>10} "
                f"{row['lat_p50']:>10.2f} {row['lat_p95']:>10.2f} {row['lat_p99']:>10.2f}"
            )
        print("-- Latency Averages (ms) --")
        for row in results:
            print(f"{row['limit']:>12} avg={row['lat_avg']:.2f}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> Tuple[BenchConfig, List[BenchmarkLimit], ShardingType, bool]:
    parser = argparse.ArgumentParser(
        description="Benchmark CustomEmbeddingCollection under GPU embedding cache limits",
    )
    parser.add_argument("--batch-size", type=int, default=BenchConfig.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=BenchConfig.warmup_steps)
    parser.add_argument("--measure-steps", type=int, default=BenchConfig.measure_steps)
    parser.add_argument("--num-embeddings", type=int, default=BenchConfig.num_embeddings)
    parser.add_argument("--embedding-dim", type=int, default=BenchConfig.embedding_dim)
    parser.add_argument("--hot-ratio", type=float, default=BenchConfig.hot_ratio)
    parser.add_argument("--access-ratio", type=float, default=BenchConfig.access_ratio)
    parser.add_argument("--prefill-ratio", type=float, default=BenchConfig.prefill_ratio)
    parser.add_argument("--seed", type=int, default=BenchConfig.seed)
    parser.add_argument(
        "--resample-per-step",
        action="store_true",
        default=BenchConfig.resample_per_step,
        help="Regenerate synthetic batches for every warmup/measure iteration to emulate streaming traffic.",
    )
    parser.add_argument(
        "--include-cpu-baseline",
        action="store_true",
        default=False,
        help="Run an extra measurement with the GPU cache disabled (pure CPU embedding lookups).",
    )
    parser.add_argument(
        "--prebuilt-chunk-rows",
        type=int,
        default=BenchConfig.prebuilt_chunk_rows,
        help="Chunk size (rows) when touching CPU embedding weights; 0 disables chunked progress.",
    )
    parser.add_argument(
        "--prebuilt-pin-memory",
        action="store_true",
        default=BenchConfig.prebuilt_pin_memory,
        help="Pin CPU embedding weights for faster H2D copies (uses extra pinned RAM).",
    )
    parser.add_argument(
        "--sharding",
        type=str,
        default=ShardingType.ROW_WISE.value,
        choices=[mode.value for mode in ShardingType],
        help="Sharding mode for embedding tables.",
    )
    parser.add_argument(
        "--limit",
        dest="limits",
        action="append",
        default=None,
        help="GPU embedding budget (e.g. 10GB, 4096MB, none). Repeat to benchmark multiple limits.",
    )
    args = parser.parse_args()
    cfg = BenchConfig(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        hot_ratio=args.hot_ratio,
        access_ratio=args.access_ratio,
        prefill_ratio=args.prefill_ratio,
        seed=args.seed,
        prebuilt_chunk_rows=args.prebuilt_chunk_rows,
        prebuilt_pin_memory=args.prebuilt_pin_memory,
        resample_per_step=args.resample_per_step,
    )
    raw_limits = args.limits if args.limits is not None else ["none", "10GB", "6GB", "4GB"]
    limits: List[BenchmarkLimit] = []
    for value in raw_limits:
        try:
            limits.append(_parse_limit(value))
        except ValueError as exc:
            raise SystemExit(str(exc))
    sharding = ShardingType(args.sharding)
    return cfg, limits, sharding, args.include_cpu_baseline


def main() -> None:
    cfg, limits, sharding, include_cpu_baseline = parse_args()
    run_benchmark(cfg, limits, sharding, include_cpu_baseline=include_cpu_baseline)


if __name__ == "__main__":
    main()

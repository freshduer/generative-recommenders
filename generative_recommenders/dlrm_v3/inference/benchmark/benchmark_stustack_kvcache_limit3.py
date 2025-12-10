#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
import time
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STULayer, STULayerConfig, STUStack
import numpy as np

INT64_ELEMENT_SIZE = torch.tensor(0, dtype=torch.int64).element_size()


def _complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Return offsets [0, cumsum(lengths)]."""
    try:
        fbgemm_ops = getattr(torch.ops, "fbgemm", None)
        if fbgemm_ops is not None and hasattr(fbgemm_ops, "asynchronous_complete_cumsum"):
            return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except Exception:
        pass
    zero = torch.zeros(1, device=lengths.device, dtype=lengths.dtype)
    return torch.cat([zero, torch.cumsum(lengths, dim=0)])


@dataclass
class BenchConfig:
    num_layers: int = 8
    num_heads: int = 4
    embedding_dim: int = 256
    attention_dim: int = 64
    hidden_dim: int = 1024
    contextual_seq_len: int = 0
    use_group_norm: bool = False
    causal: bool = True
    kernel: str = "triton"
    dtype: str = "float16"
    device: Optional[str] = None
    batch_size: int = 16
    num_users: int = 288
    prefill_len: int = 1
    delta_size: int = 15000
    warmup_requests: int = 128
    measure_requests: int = 1024
    seed: int = 2025
    stats_interval: int = 200  # 每多少个请求重新统计一次热门用户


@dataclass
class BenchmarkLimit:
    label: str
    bytes: Optional[int]


@dataclass
class LayerCache:
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    offsets: Optional[torch.Tensor]
    max_len: int

    def nbytes(self) -> int:
        total = 0
        if self.k is not None:
            total += self.k.numel() * self.k.element_size()
        if self.v is not None:
            total += self.v.numel() * self.v.element_size()
        if self.offsets is not None:
            total += self.offsets.numel() * self.offsets.element_size()
        return total


@dataclass
class KVEntry:
    caches: List[LayerCache]
    last_access: float = 0.0  # Time of last access


class KVCacheManager:
    def __init__(self, limit_bytes: Optional[int], device: torch.device) -> None:
        self.limit_bytes = limit_bytes
        self.device = device
        self.entries: Dict[int, KVEntry] = {}
        self.hot_users: Set[int] = set()
        
        # Metrics
        self.current_bytes: int = 0
        self.peak_bytes: int = 0
        self.eviction_count: int = 0
        self.recompute_count: int = 0
        self.hit_count: int = 0
        self.total_access: int = 0

        # CUDA Reserve
        self.cuda_reserve_bytes: int = self._compute_cuda_reserve(device)

    def _compute_cuda_reserve(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            total_bytes = torch.cuda.get_device_properties(index).total_memory
        except Exception:
            total_bytes = 0
        # 预留 5% 显存防止 OOM
        return max(int(total_bytes * 0.05), 512 * 1024**2)

    def contains(self, user_id: int) -> bool:
        return user_id in self.entries

    def touch(self, user_id: int) -> None:
        if user_id in self.entries:
            self.entries[user_id].last_access = time.time()

    def update_hot_users(self, new_hot_users: Set[int]) -> None:
        old_hot = self.hot_users
        self.hot_users = new_hot_users
        no_longer_hot = old_hot - new_hot_users
        if no_longer_hot:
            print(f"[kv-manager] Hot set changed. Dropped {len(no_longer_hot)} users from hot status.")
        added_hot = new_hot_users - old_hot
        if added_hot:
            print(f"[kv-manager] Added {len(added_hot)} new users to hot set.")

    def capture_and_store(self, user_id: int, layers: Sequence[STULayer], protected_users: Optional[Set[int]] = None) -> None:
        """从 Layer 中捕获 KV Cache 并存储。protected_users 中的 ID 绝对不驱逐。"""
        # 1. 计算需要的空间
        needed_bytes = 0
        snapshots = []
        
        for layer in layers:
            k = layer.k_cache
            v = layer.v_cache
            offsets = layer.kv_caching_offsets
            
            if k is None or v is None or offsets is None:
                snapshots.append(None)
                continue
                
            start = int(offsets[0].item())
            end = int(offsets[1].item())
            length = end - start
            
            k_slice = k[start:end].detach().clone()
            v_slice = v[start:end].detach().clone()
            local_offsets = torch.tensor([0, length], dtype=torch.int64, device=self.device)
            
            size = k_slice.nbytes + v_slice.nbytes + local_offsets.nbytes
            needed_bytes += size
            snapshots.append((k_slice, v_slice, local_offsets, length))

        # 2. 确保空间 (Eviction Logic)
        # 这里的 protected_users 必须包含当前的 user_id
        protect_set = protected_users if protected_users is not None else set()
        protect_set.add(user_id)
        
        self._ensure_space(needed_bytes, protected_users=protect_set)

        # 3. 存储
        caches = []
        for snap in snapshots:
            if snap is None:
                caches.append(LayerCache(None, None, None, 0))
            else:
                caches.append(LayerCache(k=snap[0], v=snap[1], offsets=snap[2], max_len=snap[3]))
        
        self.entries[user_id] = KVEntry(caches=caches, last_access=time.time())
        self.current_bytes += needed_bytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def load_into_layers(self, user_ids: Sequence[int], layers: Sequence[STULayer]) -> List[int]:
        miss_ids = []
        hit_ids = []
        
        for uid in user_ids:
            if uid not in self.entries:
                miss_ids.append(uid)
            else:
                hit_ids.append(uid)
                self.touch(uid)

        # 统计口径改为在重算之前进行，避免因提前重算导致“全命中”假象。
        # 此处不再更新 hit/recompute/total 计数。

        if not hit_ids:
            for layer in layers:
                layer.reset_kv_cache()
            return miss_ids

        for layer_idx, layer in enumerate(layers):
            k_list = []
            v_list = []
            lengths = []
            max_len = 0
            
            for uid in hit_ids:
                cache = self.entries[uid].caches[layer_idx]
                if cache.k is not None:
                    k_list.append(cache.k)
                    v_list.append(cache.v)
                    lengths.append(cache.max_len)
                    max_len = max(max_len, cache.max_len)
                else:
                    lengths.append(0)

            if not k_list:
                layer.reset_kv_cache()
                continue
            
            combined_k = torch.cat(k_list, dim=0)
            combined_v = torch.cat(v_list, dim=0)
            
            offsets = [0]
            curr = 0
            for l in lengths:
                curr += l
                offsets.append(curr)
            
            layer.k_cache = combined_k
            layer.v_cache = combined_v
            layer.max_kv_caching_len = max_len
            layer.kv_caching_offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)

        return miss_ids

    def _ensure_space(self, needed_bytes: int, protected_users: Set[int]) -> None:
        """如果空间不足，踢人。绝对不踢 protected_users 中的人。"""
        if self.limit_bytes is None:
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                if free > needed_bytes + self.cuda_reserve_bytes:
                    return
            else:
                return

        limit = self.limit_bytes if self.limit_bytes else float('inf')
        
        while True:
            current_usage = self.current_bytes
            phys_free = float('inf')
            if self.device.type == "cuda":
                phys_free, _ = torch.cuda.mem_get_info(self.device)
            
            if (current_usage + needed_bytes <= limit) and (phys_free > needed_bytes + self.cuda_reserve_bytes):
                break
            
            victim = self._pick_victim(protected_users)
            if victim is None:
                # 确实没法踢了（全是 protected 或者 hot 且策略不允许踢 hot，或者单纯只有 protected）
                # 这里我们假设 Hot 不是 protected，可以被踢，但 protected 绝对不能被踢。
                # 如果连 protected 都占满了显存，那就是 Batch Size 太大了。
                print(f"[kv-manager] WARNING: Cache full. Usage: {current_usage/1024**2:.2f}MB, Needed: {needed_bytes/1024**2:.2f}MB. Cannot evict protected users.")
                break
                
            self._evict_user(victim)

    def _pick_victim(self, protected_users: Set[int]) -> Optional[int]:
        """选择驱逐对象：跳过 protected_users，优先踢非热门，再踢 LRU"""
        candidates = []
        for uid, entry in self.entries.items():
            if uid in protected_users:
                continue
            is_hot = uid in self.hot_users
            candidates.append((is_hot, entry.last_access, uid))
        
        if not candidates:
            return None
        
        # 排序：先踢非热门(False < True)，再踢最老的(last_access 小的在前)
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    def _evict_user(self, user_id: int) -> None:
        entry = self.entries.pop(user_id)
        freed_bytes = 0
        for cache in entry.caches:
            freed_bytes += cache.nbytes()
        self.current_bytes -= freed_bytes
        self.eviction_count += 1

class StuService:
    def __init__(
        self,
        cfg: BenchConfig,
        stu: STUStack,
        manager: KVCacheManager,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cfg = cfg
        self.stu = stu
        self.manager = manager
        self.device = device
        self.dtype = dtype
        self.layers: List[STULayer] = list(stu._stu_layers)
        
        self.prefill_lengths = torch.full((1,), cfg.prefill_len, device=device, dtype=torch.int64)
        self.prefill_offsets = _complete_cumsum(self.prefill_lengths)
        self.prefill_targets = torch.zeros_like(self.prefill_lengths)
        
        self.generators: List[torch.Generator] = []
        for idx in range(cfg.num_users):
            gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
            gen.manual_seed(cfg.seed + idx)
            self.generators.append(gen)

    def _reset_layers(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()

    def recompute_user(self, user_id: int, protected_users: Optional[Set[int]] = None) -> None:
        """重计算并存入 Cache，同时保护 protected_users 不被踢出"""
        self._reset_layers()
        generator = self.generators[user_id]
        history = torch.randn(
            self.cfg.prefill_len,
            self.cfg.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        
        self.stu(
            history,
            self.prefill_lengths,
            self.prefill_offsets,
            self.cfg.prefill_len,
            self.prefill_targets,
            self.cfg.prefill_len,
            self.prefill_lengths,
        )
        
        self.manager.capture_and_store(user_id, self.layers, protected_users=protected_users)

    def prefill_all_users(self) -> None:
        print(f"[kv-bench] Initial prefill for {self.cfg.num_users} users...")
        for user_id in range(self.cfg.num_users):
            # 这里的 protected_users 可以是 None，因为预热不关心是否踢出刚算好的
            self.recompute_user(user_id)
            if (user_id + 1) % 50 == 0:
                print(f"  Prefilled {user_id + 1}/{self.cfg.num_users}")

    def run_requests(
        self,
        request_ids: Sequence[int],
        collect_stats: bool = True,
    ) -> Dict[str, float]:
        total = len(request_ids)
        if total == 0:
            return {}
        
        print(f"[kv-bench] Starting processing {total} requests...", flush=True)
        freq_counter = Counter()
        stats_interval = self.cfg.stats_interval
        start_time = time.perf_counter()
        latencies = []
        # 分阶段计时累计（秒）
        total_recompute_time = 0.0
        total_load_time = 0.0
        total_forward_time = 0.0
        hit_batch_time = 0.0
        miss_batch_time = 0.0
        hit_batch_count = 0
        miss_batch_count = 0
        forward_hit_time = 0.0
        forward_miss_time = 0.0
        
        # 估算容量
        if self.manager.limit_bytes and self.manager.current_bytes > 0 and len(self.manager.entries) > 0:
            avg_size = self.manager.current_bytes / len(self.manager.entries)
            estimated_capacity = int(self.manager.limit_bytes / avg_size)
        else:
            estimated_capacity = 100 

        processed = 0
        batch_size = self.cfg.batch_size
        
        while processed < total:
            # 1. 热门用户统计
            if processed > 0 and processed % stats_interval == 0:
                print(f"--- [Stats Check at {processed}] ---", flush=True)
                top_k = freq_counter.most_common(estimated_capacity)
                hot_user_set = {uid for uid, _ in top_k}
                print(f"    Identify {len(hot_user_set)} hot users.", flush=True)
                self.manager.update_hot_users(hot_user_set)
                freq_counter.clear()
            
            # 2. 准备 Batch
            end_idx = min(processed + batch_size, total)
            current_batch = request_ids[processed : end_idx]
            freq_counter.update(current_batch)
            
            req_start = time.perf_counter()
            # print(f"[kv-bench] Batch starting at {processed}, size={len(current_batch)}", flush=True)
            self._reset_layers()
            
            # Step A & B: 检查 Miss 并 Recompute
            # 关键修复：构建 protected_set，包含当前 Batch 所有用户
            # 这样在计算第 2 个用户时，不会把第 1 个刚刚算好的用户踢掉
            batch_protection_set = set(current_batch)
            
            miss_ids = []
            for uid in current_batch:
                if not self.manager.contains(uid):
                    miss_ids.append(uid)
            had_miss = len(miss_ids) > 0
            # 在重算前统计真实命中/未命中情况
            if collect_stats:
                self.manager.total_access += len(current_batch)
                self.manager.hit_count += (len(current_batch) - len(miss_ids))
                self.manager.recompute_count += len(miss_ids)
            
            if miss_ids:
                recompute_start = time.perf_counter()
                for uid in miss_ids:
                    # 传入 protection set
                    self.recompute_user(uid, protected_users=batch_protection_set)
                recompute_end = time.perf_counter()
                total_recompute_time += (recompute_end - recompute_start)
                print(f"[kv-bench] Recomputed {len(miss_ids)} users in batch (time={((recompute_end - recompute_start)*1000):.2f} ms)", flush=True)
            
            # Step C: Load
            load_start = time.perf_counter()
            still_missing = self.manager.load_into_layers(current_batch, self.layers)
            load_end = time.perf_counter()
            total_load_time += (load_end - load_start)
            
            if still_missing:
                # 只有当 Cache 上限比 Batch Size 还小的时候才会触发这里
                # 或者物理显存实在不够了
                raise RuntimeError(
                    f"Cache too small! Batch size {len(current_batch)} > Cache Capacity. "
                    f"Missing users: {still_missing}. Try reducing --batch-size."
                )
            
            # Step D: Forward
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            deltas = []
            for uid in current_batch:
                deltas.append(torch.randn(
                    self.cfg.delta_size,
                    self.cfg.embedding_dim,
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.generators[uid]
                ))
            delta_input = torch.cat(deltas, dim=0)
            targets = torch.full((len(current_batch),), self.cfg.delta_size, device=self.device, dtype=torch.int64)
            
            fwd_start = time.perf_counter()
            self.stu.cached_forward(delta_input, targets)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            fwd_end = time.perf_counter()
            total_forward_time += (fwd_end - fwd_start)
            if had_miss:
                forward_miss_time += (fwd_end - fwd_start)
            else:
                forward_hit_time += (fwd_end - fwd_start)
            
            # 已在前向结束同步，这里无需重复同步
            
            req_end = time.perf_counter()
            # 统计一个 batch 的总耗时（不再除以 batch size）
            latencies.append(req_end - req_start)
            if had_miss:
                miss_batch_count += 1
                miss_batch_time += (req_end - req_start)
            else:
                hit_batch_count += 1
                hit_batch_time += (req_end - req_start)
            # 每批次结束打印简要统计
            print(
                f"[kv-bench] Batch done at {processed + len(current_batch)}: latency={latencies[-1]*1000:.2f} ms, "
                f"hits={self.manager.hit_count}, recomputes={self.manager.recompute_count}, evictions={self.manager.eviction_count}, "
                f"phase(ms): recompute={(total_recompute_time*1000):.1f} load={(total_load_time*1000):.1f} forward={(total_forward_time*1000):.1f}",
                flush=True,
            )

            processed += len(current_batch)
            
            if total >= 10 and processed % (total // 10) == 0:
                  hit_rate = (self.manager.hit_count / max(1, self.manager.total_access)) * 100
                  print(f"[Progress {processed}/{total}] Hit Rate: {hit_rate:.2f}% | Cache Usage: {self.manager.current_bytes/1024**2:.1f} MB", flush=True)

            # 大规模测量时每 1000 请求打印一次累计吞吐/延迟
            if processed % 1000 == 0:
                elapsed = time.perf_counter() - start_time
                avg_latency_ms = (sum(latencies) / max(1, len(latencies))) * 1000.0
                throughput = processed / max(elapsed, 1e-9)
                print(
                    f"[kv-bench] Milestone {processed}/{total}: avg_batch_latency={avg_latency_ms:.2f} ms, throughput={throughput:.2f} req/s, "
                    f"evictions={self.manager.eviction_count}",
                    flush=True,
                )

        total_time = time.perf_counter() - start_time
        print(f"[kv-bench] Completed {total} requests in {total_time:.2f}s", flush=True)
        throughput = total / max(total_time, 1e-9)
        
        return {
            "throughput": throughput,
            "avg_latency": (sum(latencies) / len(latencies) * 1000) if latencies else 0,  # avg batch latency (ms)
            "hit_rate": (self.manager.hit_count / max(1, self.manager.total_access)) * 100,
            "evictions": self.manager.eviction_count,
            "recomputes": self.manager.recompute_count,
            "p50_ms": (float(np.percentile([l*1000 for l in latencies], 50)) if latencies else float('nan')),
            "p95_ms": (float(np.percentile([l*1000 for l in latencies], 95)) if latencies else float('nan')),
            "p99_ms": (float(np.percentile([l*1000 for l in latencies], 99)) if latencies else float('nan')),
            "forward_total_ms": total_forward_time * 1000.0,
            "load_total_ms": total_load_time * 1000.0,
            "recompute_total_ms": total_recompute_time * 1000.0,
            "hit_batches": hit_batch_count,
            "miss_batches": miss_batch_count,
            "hit_batch_avg_ms": (hit_batch_time / hit_batch_count * 1000.0) if hit_batch_count else float('nan'),
            "miss_batch_avg_ms": (miss_batch_time / miss_batch_count * 1000.0) if miss_batch_count else float('nan'),
            "forward_hit_avg_ms": (forward_hit_time / hit_batch_count * 1000.0) if hit_batch_count else float('nan'),
            "forward_miss_avg_ms": (forward_miss_time / miss_batch_count * 1000.0) if miss_batch_count else float('nan'),
        }


def _pick_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _pick_kernel(name: str) -> HammerKernel:
    return HammerKernel.TRITON if name.lower() != "pytorch" else HammerKernel.PYTORCH


def _build_stu(cfg: BenchConfig, device: torch.device, kernel: HammerKernel) -> STUStack:
    layers: List[STULayer] = [
        STULayer(
            config=STULayerConfig(
                embedding_dim=cfg.embedding_dim,
                num_heads=cfg.num_heads,
                hidden_dim=cfg.hidden_dim,
                attention_dim=cfg.attention_dim,
                output_dropout_ratio=0.0,
                causal=cfg.causal,
                target_aware=True,
                max_attn_len=None,
                attn_alpha=None,
                use_group_norm=cfg.use_group_norm,
                recompute_normed_x=False,
                recompute_uvqk=False,
                recompute_y=False,
                sort_by_length=True,
                contextual_seq_len=cfg.contextual_seq_len,
            ),
            is_inference=True,
        )
        for _ in range(cfg.num_layers)
    ]
    stu = STUStack(stu_list=layers, is_inference=True).to(device)
    stu.eval()
    stu.recursive_setattr("_hammer_kernel", kernel)
    return stu


def _parse_limit(value: str) -> BenchmarkLimit:
    text = value.strip().lower()
    if text in {"none", "unlimited"}:
        return BenchmarkLimit(label="Unlimited", bytes=None)
    
    # Simple parse
    units = {"gb": 1024**3, "mb": 1024**2}
    multiplier = 1
    for u, m in units.items():
        if text.endswith(u):
            multiplier = m
            text = text[:-len(u)]
            break
    try:
        val = float(text)
        return BenchmarkLimit(label=value, bytes=int(val * multiplier))
    except:
        return BenchmarkLimit(label="Unlimited", bytes=None)


def run_benchmark(cfg: BenchConfig, limits: List[BenchmarkLimit]) -> None:
    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _pick_dtype(cfg.dtype)
    kernel = _pick_kernel(cfg.kernel)

    print(f"==== Benchmarking STU Cache Recompute Strategy ====")
    print(f"Device: {device}, Users: {cfg.num_users}, Limit Stats Interval: {cfg.stats_interval}", flush=True)

    set_dev_mode(False)
    # Stub logging
    STULayer.log_kv_cache_size = lambda self, prefix="": None

    torch.manual_seed(cfg.seed)

    base_stu = _build_stu(cfg, device, kernel)
    base_state = base_stu.state_dict()
    del base_stu

    # 生成请求序列：模拟 Zipf 分布来制造 "热门用户" 现象
    # 这样 Cache 才有意义
    print("[kv-bench] Generating Zipfian request trace to simulate hot users...", flush=True)
    request_gen = torch.Generator()
    request_gen.manual_seed(cfg.seed + 999)
    
    # 简单的 Zipf 模拟: 
    # 大部分请求集中在少数 id 上
    # probs = 1 / (rank ^ alpha)
    alpha = 1.2
    ranks = torch.arange(1, cfg.num_users + 1, dtype=torch.float64)
    probs = 1.0 / (ranks ** alpha)
    probs /= probs.sum()
    
    total_requests = cfg.warmup_requests + cfg.measure_requests
    # 使用 multinomial 采样
    request_sequence = torch.multinomial(probs, total_requests, replacement=True, generator=request_gen).tolist()
    
    # 故意在中间改变热门模式 (通过 Shuffle mapping) 
    # 以测试 "热门用户变化" 的逻辑
    mid_point = total_requests // 2
    part1 = request_sequence[:mid_point]
    part2_raw = request_sequence[mid_point:]
    
    # 简单的 ID 映射偏移，制造 Drifting
    shift = cfg.num_users // 2
    part2 = [(uid + shift) % cfg.num_users for uid in part2_raw]
    
    final_requests = part1 + part2
    print(f"[kv-bench] Trace generated. Hot users will shift at request {mid_point}.", flush=True)

    warmup_requests = final_requests[: cfg.warmup_requests]
    measure_requests = final_requests[cfg.warmup_requests :]

    overall_rows = []
    for limit in limits:
        print(f"\n[kv-bench] === Testing Limit: {limit.label} ===", flush=True)
        stu = _build_stu(cfg, device, kernel)
        stu.load_state_dict(base_state)
        
        manager = KVCacheManager(limit.bytes, device)
        service = StuService(cfg, stu, manager, device, dtype)

        with torch.inference_mode():
            # Initial prefill (optional, depends on real world, let's skip global prefill to test lazy recompute)
            # service.prefill_all_users() 
            
            if cfg.warmup_requests > 0:
                print(f"[kv-bench] Warmup ({cfg.warmup_requests} reqs)...", flush=True)
                service.run_requests(warmup_requests, collect_stats=False)
                
                # Reset metrics but keep cache state
                manager.eviction_count = 0
                manager.recompute_count = 0
                manager.hit_count = 0
                manager.total_access = 0

            print(f"[kv-bench] Measurement ({cfg.measure_requests} reqs)...", flush=True)
            res = service.run_requests(measure_requests, collect_stats=True)
            
            print("-" * 60, flush=True)
            print(f"Result for Limit {limit.label}:")
            print(f"  Throughput : {res['throughput']:.2f} req/s")
            print(f"  Avg Latency: {res['avg_latency']:.2f} ms")
            print(f"  Hit Rate   : {res['hit_rate']:.2f}%")
            print(f"  Evictions  : {res['evictions']}")
            print(f"  Recomputes : {res['recomputes']}")
            print(f"  Hit Batches: {res['hit_batches']} (avg {res['hit_batch_avg_ms']:.2f} ms)")
            print(f"  Miss Batches: {res['miss_batches']} (avg {res['miss_batch_avg_ms']:.2f} ms)")
            print(f"  Forward Hit Avg: {res['forward_hit_avg_ms']:.2f} ms | Forward Miss Avg: {res['forward_miss_avg_ms']:.2f} ms")
            print("-" * 60, flush=True)

            # Per-limit summary table for easier scanning
            avg_gpu_mb = manager.current_bytes / 1024**2
            max_gpu_mb = manager.peak_bytes / 1024**2
            p50_ms = res.get('p50_ms', float('nan'))
            p95_ms = res.get('p95_ms', float('nan'))
            p99_ms = res.get('p99_ms', float('nan'))
            # We don't have the raw latencies here; rely on avg as main metric.
            print("==== STUStack KV-Cache Budget Benchmark ====", flush=True)
            print(f"device: {device.type}, kernel: {kernel.name}, dtype: {dtype}", flush=True)
            print(
                f"layers: {cfg.num_layers}, heads: {cfg.num_heads}, embed: {cfg.embedding_dim}, attn: {cfg.attention_dim}, hidden: {cfg.hidden_dim}",
                flush=True,
            )
            print(
                f"users: {cfg.num_users}, batch: {cfg.batch_size}, prefill_len: {cfg.prefill_len}, delta_size: {cfg.delta_size}, warmup: {cfg.warmup_requests}, measure: {cfg.measure_requests}",
                flush=True,
            )
            print("-- Results --", flush=True)
            print("       limit throughput (req/s)     avg_gpu_mb   max_gpu_mb  evictions  recomputes     p50_ms     p95_ms     p99_ms", flush=True)
            limit_label = limit.label
            print(
                f"{limit_label:>10}{res['throughput']:>18.2f}{avg_gpu_mb:>16.2f}{max_gpu_mb:>13.2f}{manager.eviction_count:>11}{res['recomputes']:>11}{p50_ms:>11.2f}{p95_ms:>11.2f}{p99_ms:>11.2f}",
                flush=True,
            )
            print("-- Latency Averages (ms) --", flush=True)
            print(f"{limit_label:>10} avg={res['avg_latency']:.2f}", flush=True)

            # Accumulate for final overall table
            overall_rows.append({
                "limit": limit_label,
                "throughput": res['throughput'],
                "avg_gpu_mb": avg_gpu_mb,
                "max_gpu_mb": max_gpu_mb,
                "evictions": manager.eviction_count,
                "recomputes": res['recomputes'],
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "avg_ms": res['avg_latency'],
                "hit_batch_avg_ms": res['hit_batch_avg_ms'],
                "miss_batch_avg_ms": res['miss_batch_avg_ms'],
                "forward_hit_avg_ms": res['forward_hit_avg_ms'],
                "forward_miss_avg_ms": res['forward_miss_avg_ms'],
                "hit_batches": res['hit_batches'],
                "miss_batches": res['miss_batches'],
            })

        del service
        del manager
        del stu
        torch.cuda.empty_cache()

    # Print one overall table including all limits
    if overall_rows:
        print("==== STUStack KV-Cache Budget Benchmark ====", flush=True)
        print(f"device: {device}, kernel: {kernel.name}, dtype: {dtype}", flush=True)
        print(
            f"layers: {cfg.num_layers}, heads: {cfg.num_heads}, embed: {cfg.embedding_dim}, attn: {cfg.attention_dim}, hidden: {cfg.hidden_dim}",
            flush=True,
        )
        print(
            f"users: {cfg.num_users}, batch: {cfg.batch_size}, prefill_len: {cfg.prefill_len}, delta_size: {cfg.delta_size}, warmup: {cfg.warmup_requests}, measure: {cfg.measure_requests}",
            flush=True,
        )
        print("-- Results --", flush=True)
        print("       limit throughput (req/s)     avg_gpu_mb   max_gpu_mb  evictions  recomputes     p50_ms     p95_ms     p99_ms", flush=True)
        for row in overall_rows:
            print(
                f"{row['limit']:>10}{row['throughput']:>18.2f}{row['avg_gpu_mb']:>16.2f}{row['max_gpu_mb']:>13.2f}{row['evictions']:>11}{row['recomputes']:>11}{row['p50_ms']:>11.2f}{row['p95_ms']:>11.2f}{row['p99_ms']:>11.2f}",
                flush=True,
            )
        print("-- Latency Averages (ms) --", flush=True)
        for row in overall_rows:
            print(f"{row['limit']:>10} avg={row['avg_ms']:.2f}", flush=True)
        print("-- Hit/Miss Batch Breakdown --", flush=True)
        for row in overall_rows:
            print(
                f"{row['limit']:>10} hits={row['hit_batches']} avg={row['hit_batch_avg_ms']:.2f} | "
                f"misses={row['miss_batches']} avg={row['miss_batch_avg_ms']:.2f}",
                flush=True,
            )
        print("-- Forward Stage (ms) --", flush=True)
        for row in overall_rows:
            print(
                f"{row['limit']:>10} hit_fwd={row['forward_hit_avg_ms']:.2f} | miss_fwd={row['forward_miss_avg_ms']:.2f}",
                flush=True,
            )

def parse_args():
    parser = argparse.ArgumentParser()
    # Basic Configs
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--limit", action="append", help="e.g. 1GB, 512MB")
    parser.add_argument("--stats-interval", type=int, default=200, help="Interval to update hot users")
    parser.add_argument("--prefill-len", type=int, default=BenchConfig.prefill_len, help="Long history prefill length per user")
    parser.add_argument("--delta-size", type=int, default=BenchConfig.delta_size, help="Decode block size per request")
    parser.add_argument("--warmup-requests", type=int, default=BenchConfig.warmup_requests, help="Number of warmup requests before measurement")
    parser.add_argument("--seed", type=int, default=BenchConfig.seed, help="Random seed for generators")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device override")
    parser.add_argument("--kernel", type=str, default=BenchConfig.kernel, choices=["triton", "pytorch"], help="Kernel implementation")
    parser.add_argument("--dtype", type=str, default=BenchConfig.dtype, choices=["float32", "float16", "bfloat16"], help="Computation dtype")
    
    # ... existing args from original script ...
    # (Simplified for brevity, ensuring defaults work)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--measure-requests", type=int, default=1000)
    
    args, unknown = parser.parse_known_args()
    
    cfg = BenchConfig(
        num_users=args.num_users,
        batch_size=args.batch_size,
        measure_requests=args.measure_requests,
        stats_interval=args.stats_interval,
        prefill_len=args.prefill_len,
        delta_size=args.delta_size,
        warmup_requests=args.warmup_requests,
        seed=args.seed,
        device=args.device,
        kernel=args.kernel,
        dtype=args.dtype,
    )
    
    limits = []
    if args.limit:
        for l in args.limit:
            limits.append(_parse_limit(l))
    else:
        limits = [_parse_limit("2GB"), _parse_limit("1GB")] # Default test cases
        
    return cfg, limits

if __name__ == "__main__":
    cfg, limits = parse_args()
    run_benchmark(cfg, limits)
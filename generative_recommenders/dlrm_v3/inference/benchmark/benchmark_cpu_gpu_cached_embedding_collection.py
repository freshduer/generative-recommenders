#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 CPUGPUCachedEmbeddingCollection 的单机 benchmark。
特点：
- 所有 embedding 常驻 CPU, GPU 只做 cache, 支持 LRU。
- 支持外部 JSON 提供的 hot ids 预热；若未提供，则按 hot_ratio 取前 N id。
- 可用 --limit 重复指定多个 GPU cache 预算（同 run_inferenfce.slurm 中示例）。
运行示例：
torchrun --master-port 29501 --nproc_per_node=1 benchmark/benchmark_cpu_gpu_cached_embedding_collection.py \
  --batch-size 8 --warmup-steps 100 --measure-steps 100 --num-embeddings 60000000 \
  --embedding-dim 256 --hot-ratio 0.10 --prefill-ratio 0.10 \
  --limit 2GB --limit 5GB --limit 8GB --limit 10GB --limit 12GB --limit 15GB \
  --limit 20GB --limit 25GB --limit 30GB --limit 35GB --limit 40GB \
  --hot-ids-path /path/to/hot_ids.json
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = os.path.abspath(
    "/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference"
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from embedding_collection_cpu_cache import (  # noqa: E402
    CPUGPUCachedEmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    load_hot_ids_from_json,
)  # noqa: E402
from torchrec.modules.embedding_configs import DataType  # noqa: E402


def _parse_limit(text: str) -> Optional[int]:
    s = text.strip().lower()
    if s in {"none", "unlimited", "inf"}:
        return None
    units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suf, mul in units.items():
        if s.endswith(suf):
            return int(float(s[:- len(suf)]) * mul)
    if s.endswith("b") and s[:-1].replace(".", "", 1).isdigit():
        return int(float(s[:-1]))
    if s.replace(".", "", 1).isdigit():
        # 默认 MB
        return int(float(s) * 1024**2)
    raise ValueError(f"invalid limit: {text}")


def _dtype_num_bytes(dtype: object) -> int:
    mapping = {
        getattr(DataType, "FP16", None): 2,
        getattr(DataType, "BF16", None): 2,
        getattr(DataType, "FP32", None): 4,
        getattr(DataType, "FP64", None): 8,
    }
    for k, v in mapping.items():
        if k is not None and dtype == k:
            return v
    return 4


def build_tables(num_embeddings: int, dim: int) -> Dict[str, EmbeddingConfig]:
    return {
        "table_post_id": EmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=dim,
            name="table_post_id",
            data_type=DataType.FP16,
            feature_names=["uih_post_id", "item_post_id"],
        ),
        "table_user_id": EmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=dim,
            name="table_user_id",
            data_type=DataType.FP16,
            feature_names=["uih_owner_id", "item_owner_id", "viewer_id"],
        ),
        "table_context": EmbeddingConfig(
            num_embeddings=max(1, num_embeddings // 10),
            embedding_dim=dim,
            name="table_context",
            data_type=DataType.FP16,
            feature_names=["uih_surface_type", "item_surface_type", "uih_video_length", "item_video_length"],
        ),
    }


def _feature_vocab_sizes(tables: Dict[str, EmbeddingConfig]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for name, cfg in tables.items():
        for feat in getattr(cfg, "feature_names", []) or []:
            vocab[feat] = cfg.num_embeddings
    return vocab


def generate_skewed_indices(
    num_embeddings: int,
    total_count: int,
    hot_ratio: float,
    access_ratio: float,
    device: torch.device,
    gen: Optional[torch.Generator],
) -> torch.Tensor:
    if total_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    num_hot = max(1, int(num_embeddings * hot_ratio))
    rand = torch.rand(total_count, device=device, generator=gen)
    is_hot = rand < access_ratio
    hot_count = int(is_hot.sum().item())
    cold_count = total_count - hot_count
    out = torch.empty(total_count, dtype=torch.long, device=device)
    if hot_count > 0:
        out[is_hot] = torch.randint(0, num_hot, (hot_count,), device=device, generator=gen)
    if cold_count > 0:
        out[~is_hot] = torch.randint(num_hot, num_embeddings, (cold_count,), device=device, generator=gen)
    return out


def get_kjt_batch(
    batch_size: int,
    device: torch.device,
    feature_vocab_sizes: Dict[str, int],
    hot_ratio: float,
    access_ratio: float,
    gen: Optional[torch.Generator],
) -> KeyedJaggedTensor:
    # 配置每个用户总长度约15000
    # uih_post_id: 6000, item_post_id: 6000, 其他feature: 3000
    features = [
        ("uih_post_id", 6000, 500),      # 主要feature，约6000
        ("item_post_id", 6000, 500),    # 主要feature，约6000
        ("uih_owner_id", 1500, 100),    # 约1500
        ("item_owner_id", 500, 50),     # 约500
        ("viewer_id", 500, 50),         # 约500
        ("uih_surface_type", 200, 20),  # 约200
        ("item_surface_type", 200, 20), # 约200
    ]
    # 总计约: 6000 + 6000 + 1500 + 500 + 500 + 200 + 200 = 14900
    keys: List[str] = []
    lengths_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    for name, avg, var in features:
        keys.append(name)
        low = max(1, avg - var)
        high = avg + var
        lengths = torch.randint(low, high, (batch_size,), device=device, dtype=torch.int32, generator=gen)
        total = int(lengths.sum().item())
        vocab = feature_vocab_sizes.get(name, max(feature_vocab_sizes.values()))
        values = generate_skewed_indices(vocab, total, hot_ratio, access_ratio, device, gen)
        lengths_list.append(lengths)
        values_list.append(values)
    lengths = torch.cat(lengths_list)
    values = torch.cat(values_list)
    kjt = KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)
    return kjt.to(device)


def _lat_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    vals = sorted(latencies)
    n = len(vals)

    def pct(q: float) -> float:
        if n == 1:
            return vals[0]
        rank = (n - 1) * q / 100.0
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if hi >= n:
            hi = n - 1
        if lo == hi:
            return vals[lo]
        w = rank - lo
        return vals[lo] * (1 - w) + vals[hi] * w

    return {
        "avg_ms": sum(vals) / n * 1000,
        "p50_ms": pct(50) * 1000,
        "p95_ms": pct(95) * 1000,
        "p99_ms": pct(99) * 1000,
    }


def _build_hot_ids(
    tables: Dict[str, EmbeddingConfig],
    hot_ratio: float,
    device: torch.device,
    per_table_cap: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    hot: Dict[str, torch.Tensor] = {}
    for name, cfg in tables.items():
        base = max(1, int(cfg.num_embeddings * hot_ratio))
        cap = per_table_cap.get(name) if per_table_cap else None
        target = base if cap is None else min(base, cap)
        target = max(1, min(target, cfg.num_embeddings))
        hot[name] = torch.arange(0, target, device=device, dtype=torch.long)
    return hot


def run_one_limit(
    limit_bytes: Optional[int],
    cfg: argparse.Namespace,
    device: torch.device,
    tables: Dict[str, EmbeddingConfig],
    feature_vocab_sizes: Dict[str, int],
    hot_ids: Optional[Dict[str, torch.Tensor]],
    shared_cpu_weights: Optional[Dict[str, torch.Tensor]],
    log_interval: int,
) -> Dict[str, float]:
    print(f"[bench] === start limit={limit_bytes if limit_bytes is not None else 'unlimited'} ===")
    model = CPUGPUCachedEmbeddingCollection(
        table_configs=tables,
        device=device,
        cache_budget_bytes=limit_bytes,
        prebuilt_cpu_weights=shared_cpu_weights,
        verbose=True,
    )
    # 根据 cache 容量裁剪 hot ids，避免初始化过久
    per_table_cap = {name: cache.capacity for name, cache in model.caches.items()}
    effective_hot: Optional[Dict[str, torch.Tensor]] = None
    if hot_ids:
        effective_hot = {}
        for name, ids in hot_ids.items():
            cap = per_table_cap.get(name, 0)
            if cap <= 0:
                continue
            if ids.numel() > cap:
                effective_hot[name] = ids[:cap]
            else:
                effective_hot[name] = ids
    else:
        effective_hot = _build_hot_ids(tables, cfg.prefill_ratio, device, per_table_cap)
    if effective_hot:
        print("[bench] preload hot ids...")
        model.preload_hot_ids(effective_hot)
        print("[bench] preload done")

    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(cfg.seed)

    def sample_batch() -> KeyedJaggedTensor:
        return get_kjt_batch(
            cfg.batch_size,
            device,
            feature_vocab_sizes,
            cfg.hot_ratio,
            cfg.access_ratio,
            gen,
        )

    print("[bench] building base batch...")
    base_batch = sample_batch()
    if hasattr(base_batch, "values"):
        total_values = int(base_batch.values().numel())
        # 计算每个用户的平均长度
        per_user_avg_length = total_values / cfg.batch_size if cfg.batch_size > 0 else 0
        print(
            f"[bench] base batch ready | batch_size={cfg.batch_size} keys={len(base_batch.keys())} "
            f"total_values={total_values} per_user_avg_length={per_user_avg_length:.1f}"
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"[bench] warmup {cfg.warmup_steps} steps...")
    with torch.no_grad():
        for step in range(cfg.warmup_steps):
            current = sample_batch() if cfg.resample_per_step else base_batch
            model(current)
            if (step + 1) % max(1, log_interval) == 0 or step == 0:
                print(f"[bench] warmup step {step + 1}/{cfg.warmup_steps}")
    warmup_stats = model.cache_stats()
    print(f"[bench] warmup done | hits={warmup_stats['hits']} misses={warmup_stats['misses']}")
    latencies: List[float] = []
    print(f"[bench] measure {cfg.measure_steps} steps...")
    with torch.no_grad():
        for step in range(cfg.measure_steps):
            current = sample_batch() if cfg.resample_per_step else base_batch
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(current)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            model.record_cache_usage()
            if (step + 1) % max(1, log_interval) == 0 or step == 0:
                current_lat_ms = elapsed * 1000
                recent_latencies = latencies[-min(log_interval, len(latencies)):]
                recent_avg_ms = sum(recent_latencies) / len(recent_latencies) * 1000
                print(
                    f"[bench] measure step {step + 1}/{cfg.measure_steps} | "
                    f"current_latency={current_lat_ms:.2f}ms | "
                    f"recent_avg_latency={recent_avg_ms:.2f}ms"
                )
    stats = model.cache_stats()
    lat = _lat_stats(latencies)
    total_req = cfg.batch_size * cfg.measure_steps
    throughput = total_req / max(sum(latencies), 1e-9)
    limit_str = "unlimited" if limit_bytes is None else f"{limit_bytes / (1024 ** 3):.1f}GB"
    print(
        f"[bench] === limit={limit_str} done ==="
    )
    print(
        f"[bench] latency stats: avg={lat['avg_ms']:.2f}ms p50={lat['p50_ms']:.2f}ms "
        f"p95={lat['p95_ms']:.2f}ms p99={lat['p99_ms']:.2f}ms"
    )
    print(
        f"[bench] throughput={throughput:.2f} samples/s | "
        f"hits={stats['hits']} misses={stats['misses']} evict={stats['evictions']} page_in={stats['page_ins']} | "
        f"avg_gpu_mb={stats['avg_mb']:.2f} max_gpu_mb={stats['max_mb']:.2f}"
    )
    return {
        "limit": "unlimited" if limit_bytes is None else f"{limit_bytes / (1024 ** 3):.1f}GB",
        "throughput": throughput,
        "avg_gpu_mb": stats["avg_mb"],
        "max_gpu_mb": stats["max_mb"],
        "evictions": stats["evictions"],
        "page_ins": stats["page_ins"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "lat_p50": lat["p50_ms"],
        "lat_p95": lat["p95_ms"],
        "lat_p99": lat["p99_ms"],
        "lat_avg": lat["avg_ms"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU+GPU cache embedding collection benchmark (单卡).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--num-embeddings", type=int, default=60_000_000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hot-ratio", type=float, default=0.10, help="整体 hot id 比例，用于默认预热/采样。")
    parser.add_argument("--access-ratio", type=float, default=0.90, help="请求命中 hot 区间的概率。")
    parser.add_argument("--prefill-ratio", type=float, default=0.10, help="未提供 hot_ids 时预热占比。")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resample-per-step", action="store_true", help="每步重新生成 batch。")
    parser.add_argument("--log-interval", type=int, default=10, help="warmup/measure 日志步长。")
    parser.add_argument(
        "--hot-ids-path",
        type=str,
        default=None,
        help="外部 JSON 路径，形如 {\"table_name\": [ids...] }，用于预热。",
    )
    parser.add_argument(
        "--limit",
        dest="limits",
        action="append",
        default=None,
        help="GPU cache 预算（支持重复），如 5GB / 4096MB / none。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = build_tables(args.num_embeddings, args.embedding_dim)
    print(f"[bench] device={device} tables={list(tables.keys())}")
    feature_vocab_sizes = _feature_vocab_sizes(tables)
    # 解析 hot ids
    hot_ids: Optional[Dict[str, torch.Tensor]] = None
    if args.hot_ids_path:
        hot_ids = load_hot_ids_from_json(args.hot_ids_path, device)
        print("[bench] hot ids source=file")
    else:
        hot_ids = None  # 延后到每个 limit，用 cache 容量裁剪
        print("[bench] hot ids source=auto (per-limit capped)")

    # 共享 CPU 权重供多 limit 复用，避免重复分配
    shared_cpu_weights: Dict[str, torch.Tensor] = {}
    for name, cfg in tables.items():
        dtype = torch.float16 if str(getattr(cfg, 'data_type', 'fp16')).lower() in {"fp16", "half", "float16"} else torch.float32
        shared_cpu_weights[name] = torch.empty(cfg.num_embeddings, cfg.embedding_dim, device=torch.device("cpu"), dtype=dtype)
    print("[bench] built shared CPU embeddings for all limits")

    raw_limits = args.limits if args.limits is not None else ["none", "10GB", "5GB"]
    limit_values = [_parse_limit(v) for v in raw_limits]
    print(f"[bench] limits={limit_values}")

    results: List[Dict[str, float]] = []
    for lim in limit_values:
        res = run_one_limit(
            lim,
            args,
            device,
            tables,
            feature_vocab_sizes,
            hot_ids,
            shared_cpu_weights,
            log_interval=max(1, args.log_interval),
        )
        results.append(res)

    print("==== CPU+GPU Cache EmbeddingCollection Benchmark ====")
    print(f"device={device} tables={len(tables)} dim={args.embedding_dim} batch={args.batch_size}")
    header = (
        f"{'limit':>12} {'throughput(s/s)':>18} {'avg_gpu_mb':>12} {'max_gpu_mb':>12} "
        f"{'evict':>8} {'page_in':>10} {'hits':>10} {'misses':>10} "
        f"{'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10}"
    )
    print(header)
    for row in results:
        print(
            f"{row['limit']:>12} {row['throughput']:>18.2f} {row['avg_gpu_mb']:>12.2f} {row['max_gpu_mb']:>12.2f} "
            f"{row['evictions']:>8} {row['page_ins']:>10} {row['hits']:>10} {row['misses']:>10} "
            f"{row['lat_p50']:>10.2f} {row['lat_p95']:>10.2f} {row['lat_p99']:>10.2f}"
        )
    print("-- Latency Averages (ms) --")
    for row in results:
        print(f"{row['limit']:>12} avg={row['lat_avg']:.2f}")


if __name__ == "__main__":
    main()


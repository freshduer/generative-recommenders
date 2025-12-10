#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script that shares a single CPU prebuilt embedding set across many GPU cache budget
limits. For each limit we instantiate a CustomEmbeddingCollection with a per-limit GPU cache size
(derived from the byte budget) while reusing the same CPU embeddings.

Usage (example):
torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 benchmark/benchmark_embedding_cache_limit.py \
  --batch-size 8 --warmup-steps 5 --measure-steps 100 \
  --num-embeddings 60000000 --embedding-dim 256 \
  --limit 2GB --limit 5GB --limit 8GB --limit 10GB \
  --sharding row_wise --prebuilt-pin-memory --include-cpu-baseline --resample-per-step
"""
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

# add project path if you need - keep as in your environment
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# import your implementation - adapt module name if necessary
from custom_sharding2 import CustomEmbeddingCollection, ShardingType, ShardingUtils  # noqa: E402
try:
    from torchrec.modules.embedding_configs import EmbeddingConfig, DataType  # noqa: E402
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # noqa: E402
    from torchrec.distributed.planner.types import ParameterConstraints  # noqa: E402
except Exception:
    # Minimal stand-ins if torchrec not installed (only for script to parse)
    EmbeddingConfig = object
    DataType = object
    KeyedJaggedTensor = object
    ParameterConstraints = object

# ---------------------------
# Small config / helper utils
# ---------------------------
@dataclass
class BenchConfig:
    batch_size: int = 8
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
    if not value.replace(".", "", 1).isdigit():
        raise ValueError(f"Could not parse limit value: {value}")
    number = float(text)
    return BenchmarkLimit(label=f"{number:g}MB", bytes=int(number * 1024**2))

def _dtype_num_bytes(data_type: object) -> int:
    # best-effort
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

# Simple feature groups (small subset of your original)
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

def _build_tables_config(cfg: BenchConfig) -> Dict[str, "EmbeddingConfig"]:
    # Build a few tables similar to your original script
    def E(name, num, dim, fns):
        # create a lightweight wrapper if torchrec EmbeddingConfig is unavailable
        try:
            return EmbeddingConfig(num_embeddings=num, embedding_dim=dim, name=name, data_type=DataType.FP16, feature_names=fns)
        except Exception:
            class _Cfg:
                def __init__(self, name, num, dim, fns):
                    self.name = name
                    self.num_embeddings = num
                    self.embedding_dim = dim
                    self.data_type = getattr(DataType, "FP16", "fp16")
                    self.feature_names = fns
            return _Cfg(name, num, dim, fns)
    return {
        "table_post_id": E("table_post_id", cfg.num_embeddings, cfg.embedding_dim, ["uih_post_id", "item_post_id"]),
        "table_user_id": E("table_user_id", cfg.num_embeddings, cfg.embedding_dim, ["uih_owner_id", "item_owner_id", "viewer_id"]),
        "table_video_meta": E("table_video_meta", max(1, cfg.num_embeddings // 10), cfg.embedding_dim, ["uih_video_length", "item_video_length"]),
    }

def _feature_vocab_sizes(tables: Dict[str, "EmbeddingConfig"]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for table_name, cfg in tables.items():
        for feat in cfg.feature_names:
            mapping[feat] = cfg.num_embeddings
    return mapping
# ---------------------------
# Fast embed class: skip default initialization
# ---------------------------
class NoInitEmbedding(torch.nn.Embedding):
    """Embedding layer that skips the heavy default initialization."""
    def reset_parameters(self) -> None:  # type: ignore[override]
        # no-op: leave memory uninitialized until we actually need rows
        return
# ---------------------------
# Lazy initialization helper (only initialize rows we actually need)
# ---------------------------
def _ensure_init_rows(emb: torch.nn.Embedding, ids: torch.Tensor, low: float = -0.01, high: float = 0.01) -> None:
    """
    Ensure specific rows in a CPU embedding `emb` are initialized with uniform(low, high).
    `ids` must be on the same device as emb.weight (CPU).
    This avoids filling the whole table up-front.
    """
    if ids is None or ids.numel() == 0:
        return
    if ids.device != emb.weight.device:
        ids = ids.to(emb.weight.device)
    ids = ids.to(torch.long)
    # unique to avoid repeated writes
    ids_u = torch.unique(ids)
    if ids_u.numel() == 0:
        return
    # allocate a temp tensor and fill with uniform; dtype matches emb.weight
    # do it in one bulk op where possible
    rows = ids_u.numel()
    dim = emb.weight.shape[1]
    # create random block and scatter
    tmp = torch.empty((rows, dim), dtype=emb.weight.dtype, device=emb.weight.device)
    tmp.uniform_(low, high)
    emb.weight.data[ids_u] = tmp


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    # Create a generator bound to same device type as `device`.
    # Torch requires generator.device matches the device used in rand(..., generator=gen).
    try:
        if device.type == "cuda":
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            return gen
    except Exception:
        # older torch versions may not accept device string; fallback to cpu generator
        pass
    # CPU fallback
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen

def generate_skewed_indices(num_embeddings: int, total_count: int, device: torch.device, hot_ratio: float, access_ratio: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if total_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    num_hot = max(1, int(num_embeddings * hot_ratio))
    rand = torch.rand(total_count, device=device, generator=generator)
    is_hot = rand < access_ratio
    num_hot_access = int(is_hot.sum().item())
    num_cold_access = total_count - num_hot_access
    indices = torch.empty(total_count, dtype=torch.long, device=device)
    if num_hot_access > 0:
        indices[is_hot] = torch.randint(0, num_hot, (num_hot_access,), device=device, generator=generator)
    if num_cold_access > 0:
        indices[~is_hot] = torch.randint(num_hot, num_embeddings, (num_cold_access,), device=device, generator=generator)
    return indices

def get_kjt_batch(batch_size: int, device: torch.device, feature_vocab_sizes: Dict[str, int], cfg: BenchConfig, generator: Optional[torch.Generator] = None):
    """
    Minimal KeyedJaggedTensor generator — if real KeyedJaggedTensor exists, adapt accordingly.
    We create a dict-like object with .keys() and __getitem__ returning an object with .values() and .lengths().
    """
    # If real KeyedJaggedTensor class exists, use it
    if KeyedJaggedTensor is not object and hasattr(KeyedJaggedTensor, "from_lengths_sync"):
        all_keys = []
        all_lengths = []
        all_values = []
        for spec in FEATURE_GROUPS.values():
            base_len = int(spec["avg_len"])
            var = int(spec["variance"])
            for feat_name in spec["names"]:
                all_keys.append(feat_name)
                if var > 0:
                    low = max(1, base_len - var)
                    high = base_len + var
                    lengths = torch.randint(low, high, (batch_size,), dtype=torch.int32, device=device, generator=generator)
                else:
                    lengths = torch.full((batch_size,), base_len, dtype=torch.int32, device=device)
                total_vals = int(lengths.sum().item())
                vocab_size = feature_vocab_sizes.get(feat_name, cfg.num_embeddings)
                values = generate_skewed_indices(vocab_size, total_vals, device, cfg.hot_ratio, cfg.access_ratio, generator)
                all_lengths.append(lengths)
                all_values.append(values)
        final_lengths = torch.cat(all_lengths)
        final_values = torch.cat(all_values)
        kjt = KeyedJaggedTensor.from_lengths_sync(keys=all_keys, values=final_values, lengths=final_lengths)
        return kjt.to(device)
    # Otherwise build a tiny shim
    class _Field:
        def __init__(self, values):
            self._values = values
        def values(self):
            return self._values
    class _KJT:
        def __init__(self, mapping):
            self._mapping = mapping
        def keys(self):
            return list(self._mapping.keys())
        def __getitem__(self, k):
            return self._mapping[k]
        def to(self, device):
            # values already allocated on device
            return self
    mapping = {}
    for spec in FEATURE_GROUPS.values():
        base_len = int(spec["avg_len"])
        var = int(spec["variance"])
        for feat_name in spec["names"]:
            if var > 0:
                low = max(1, base_len - var)
                high = base_len + var
                lengths = torch.randint(low, high, (batch_size,), dtype=torch.int32, device=device, generator=generator)
            else:
                lengths = torch.full((batch_size,), base_len, dtype=torch.int32, device=device)
            total_vals = int(lengths.sum().item())
            vocab_size = feature_vocab_sizes.get(feat_name, cfg.num_embeddings)
            values = generate_skewed_indices(vocab_size, total_vals, device, cfg.hot_ratio, cfg.access_ratio, generator)
            mapping[feat_name] = _Field(values)
    return _KJT(mapping)

# ---------------------------
# Build prebuilt embeddings quickly (lazy)
# ---------------------------
def _build_prebuilt_embeddings(tables_config: Dict[str, "EmbeddingConfig"], cfg: BenchConfig, pin_memory: bool = False) -> Dict[str, torch.nn.Embedding]:
    """
    Build CPU prebuilt embeddings (one-time). Signature matches the call used in your script:
        _build_prebuilt_embeddings(tables_config, cfg, pin_memory=cfg.prebuilt_pin_memory)

    Behavior:
      - Allocate embeddings on CPU with NoInitEmbedding to avoid heavy full-table initialization.
      - Optionally pin memory for faster H2D copies.
      - Uses lazy init: actual random filling happens only for rows we later initialize (via _ensure_init_rows).
    """
    # pick rank/world_size from dist if available
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    prebuilt: Dict[str, torch.nn.Embedding] = {}
    for idx, (name, table_cfg) in enumerate(tables_config.items(), 1):
        # choose dtype
        try:
            raw_dtype = getattr(table_cfg, "data_type", DataType.FP16)
        except Exception:
            raw_dtype = getattr(DataType, "FP16", "fp16") if DataType is not object else "fp16"
        dtype_token = str(raw_dtype).lower() if raw_dtype is not None else "fp16"
        if "fp16" in dtype_token or "half" in dtype_token:
            target_dtype = torch.float16
        elif "bf16" in dtype_token or "bfloat" in dtype_token:
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

        # create uninitialized embedding on CPU (fast)
        try:
            emb = NoInitEmbedding(table_cfg.num_embeddings, table_cfg.embedding_dim, device=torch.device("cpu"), dtype=target_dtype)
        except Exception:
            # fallback to normal Embedding if NoInitEmbedding not present
            emb = torch.nn.Embedding(table_cfg.num_embeddings, table_cfg.embedding_dim, device=torch.device("cpu"), dtype=target_dtype)
        emb.weight.requires_grad_(False)

        # detach param and optionally pin memory to speed up subsequent H2D.
        weight = emb.weight.detach()
        if pin_memory and torch.cuda.is_available():
            try:
                weight = weight.pin_memory()
            except Exception:
                # pinning may fail for some dtypes / platforms; ignore
                pass
        emb.weight = torch.nn.Parameter(weight, requires_grad=False)

        prebuilt[name] = emb

        if rank == 0:
            print(f"[benchmark] prebuilt (lazy) embeddings | table={name} rows={table_cfg.num_embeddings} dim={table_cfg.embedding_dim} dtype={target_dtype} pin_memory={pin_memory}")

    return prebuilt



# ---------------------------
# Utilities for cache sizing
# ---------------------------
def _estimate_table_bytes(tables: Dict[str, "EmbeddingConfig"]) -> Dict[str, int]:
    estimates = {}
    for name, cfg in tables.items():
        dtype_size = _dtype_num_bytes(getattr(cfg, "data_type", None))
        estimates[name] = cfg.num_embeddings * cfg.embedding_dim * dtype_size
    return estimates

def _compute_cache_ratio_from_budget(limit_bytes: Optional[int], tables_config: Dict[str, "EmbeddingConfig"], device: torch.device, world_size: int, sharding: ShardingType) -> Dict[str, float]:
    """
    Return per-table cache_ratio (fraction of local rows) for given limit_bytes.
    If limit_bytes is None => full cache (ratio=1.0).
    We split the budget equally among tables, then compute how many rows per table that money can hold.
    For row-wise partitioning local_rows = partition range length (using ShardingUtils).
    """
    ratios = {}
    table_count = max(len(tables_config), 1)
    per_table_budget = None if limit_bytes is None else max(0, limit_bytes // table_count)
    for name, cfg in tables_config.items():
        row_bytes = cfg.embedding_dim * _dtype_num_bytes(getattr(cfg, "data_type", None))
        # estimate local rows for row-wise sharding (approximate via get_partition_bounds)
        if sharding == ShardingType.ROW_WISE:
            start, end = ShardingUtils.get_partition_bounds(sharding, cfg.num_embeddings, world_size, 0, name)
            # Note: using rank=0 local_rows; actual local_rows depends on actual rank, but ratio is same across ranks
            local_rows = max(1, end - start)
        else:
            # if replicated/table-wise: local_rows == total rows for the owner, but we make conservative estimate:
            local_rows = max(1, cfg.num_embeddings // max(1, world_size))
        if per_table_budget is None:
            ratios[name] = 1.0
        else:
            if row_bytes <= 0:
                ratios[name] = 0.0
            else:
                cache_rows = per_table_budget // row_bytes
                ratios[name] = float(min(max(cache_rows, 0), local_rows)) / float(local_rows)
        # clamp
        ratios[name] = max(0.0, min(1.0, ratios[name]))
    return ratios

# ---------------------------
# Manual preload fallback - make sure we copy from CPU prebuilt (and use initialized rows)
# ---------------------------
def _manual_preload_into_model(model: CustomEmbeddingCollection, table_name: str, ids: torch.Tensor, device: torch.device):
    """
    Copy given global ids from model.tables[table_name].weight_cpu -> weight_gpu slots (first-K)
    This expects that the CPU prebuilt weights (or model.tables[].weight_cpu) already contain valid values
    for those ids. We compute local indices then copy rows into GPU cache slots.
    """
    tbl = model.tables[table_name]
    # prefer model's CPU weight if present
    cpu_weight = getattr(tbl, "weight_cpu", None)
    if cpu_weight is None:
        # nothing to copy from
        return
    # convert global -> local indices on CPU device
    ids_cpu = ids.to(cpu_weight.weight.device).long()
    local_idx = ShardingUtils.global_to_local(ids_cpu, tbl).to(cpu_weight.weight.device).long()
    # if model has weight_gpu, copy as many rows as fit (we use simple first-K policy)
    if getattr(tbl, "weight_gpu", None) is None:
        return
    gpu_weight = tbl.weight_gpu.weight
    # ensure local_idx fits in cpu weight
    local_idx = local_idx % cpu_weight.weight.shape[0]
    k = local_idx.numel()
    if k == 0:
        return
    # if k > gpu slots, only copy up to gpu slots
    max_copy = min(k, gpu_weight.shape[0])
    src = cpu_weight.weight.data[local_idx[:max_copy]].to(gpu_weight.device, non_blocking=True)
    gpu_weight.data[:max_copy].copy_(src)

# ---------------------------
# Try preload: ensure corresponding CPU rows initialized before copying to GPU
# ---------------------------
def _try_preload(model: CustomEmbeddingCollection, prebuilt_tables: Dict[str, torch.nn.Embedding], observed_prefill: Dict[str, torch.Tensor], device: torch.device):
    """
    Best-effort load using available API; if we need to copy rows from CPU prebuilt to model,
    initialize only those rows (lazy) before copying.
    """
    # 1) try model API if exists
    if hasattr(model, "load_cpu_weights"):
        try:
            model.load_cpu_weights(prebuilt_tables)
            return
        except Exception:
            pass

    # 2) if model has preload_hot_ids API, ensure CPU rows are initialized, then call it
    if hasattr(model, "preload_hot_ids"):
        # ensure rows are initialized in prebuilt tables
        prefill_initialized = {}
        for tname, ids in (observed_prefill or {}).items():
            if ids is None or ids.numel() == 0:
                continue
            cpu_emb = prebuilt_tables.get(tname)
            if cpu_emb is None:
                continue
            # make sure ids are on CPU for initialization
            ids_cpu = ids.to(cpu_emb.weight.device)
            _ensure_init_rows(cpu_emb, ids_cpu)
            prefill_initialized[tname] = ids_cpu
        try:
            model.preload_hot_ids(prefill_initialized)
            return
        except Exception:
            # fall through to manual copying
            pass

    # 3) otherwise, do manual copy into model.tables: initialize CPU rows and copy subset to GPU cache slots
    for tname, ids in (observed_prefill or {}).items():
        if ids is None or ids.numel() == 0:
            continue
        # init CPU rows in prebuilt
        cpu_emb = prebuilt_tables.get(tname)
        if cpu_emb is not None:
            ids_cpu = ids.to(cpu_emb.weight.device)
            _ensure_init_rows(cpu_emb, ids_cpu)
        # now copy rows into model (fallback: _manual_preload_into_model)
        try:
            _manual_preload_into_model(model, tname, ids, device)
        except Exception:
            pass

# ---------------------------
# Benchmark loop
# ---------------------------
def run_benchmark(cfg: BenchConfig, limits: Sequence[BenchmarkLimit], sharding: ShardingType, include_cpu_baseline: bool = False):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    # init dist if requested
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if (world_size_env > 1 or os.environ.get("MASTER_ADDR")) and not (dist.is_available() and dist.is_initialized()):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29501"))
        try:
            dist.init_process_group(backend=backend, device_ids=[local_rank] if device.type == "cuda" else None)
        except Exception:
            try:
                dist.init_process_group(backend=backend)
            except Exception:
                pass
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    tables_config = _build_tables_config(cfg)
    feature_vocab_sizes = _feature_vocab_sizes(tables_config)
    table_size_bytes = _estimate_table_bytes(tables_config)

    # Build prebuilt CPU embeddings once and reuse across limits
    if rank == 0:
        print("[benchmark] building shared CPU prebuilt embeddings (one-time)...")
    prebuilt_tables = _build_prebuilt_embeddings(tables_config, cfg, pin_memory=cfg.prebuilt_pin_memory)
    if rank == 0:
        print("[benchmark] prebuilt embeddings ready")

    results = []
    generator = _torch_generator(device, cfg.seed + rank)

    def sample_batch():
        return get_kjt_batch(cfg.batch_size, device, feature_vocab_sizes, cfg, generator)

    base_batch = sample_batch()

    # For each limit create model with per-limit cache size and run benchmark
    for limit_idx, limit in enumerate(limits, 1):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(f"[benchmark] ===== Limit {limit_idx}/{len(limits)}: {limit.label} (bytes={limit.bytes}) =====")
        # compute per-table cache ratios
        if rank == 0:
            print(f"[benchmark] computing cache ratios for limit={limit.label}...")
        ratios = _compute_cache_ratio_from_budget(limit.bytes, tables_config, device, world_size, sharding)
        # use average ratio across tables as model-level cache_ratio (your model may accept per-table control)
        avg_ratio = sum(ratios.values()) / max(len(ratios), 1)
        # clamp to reasonable range
        avg_ratio = max(0.0, min(1.0, avg_ratio))
        if rank == 0:
            print(f"[benchmark] cache_ratio={avg_ratio:.4f}, creating model...")

        # instantiate model - try to pass prebuilt_tables if ctor supports it
        model = None
        try:
            # try constructor that accepts prebuilt_tables / embedding_budget_bytes / cache_enabled
            model = CustomEmbeddingCollection(
                table_config=tables_config,
                cache_ratio=avg_ratio,
                device=device,
                constraints={n: ParameterConstraints(sharding_types=[sharding.value]) for n in tables_config.keys()},
            )
        except TypeError:
            # fallback: try with more generic args
            try:
                model = CustomEmbeddingCollection(tables_config, device=device, cache_ratio=avg_ratio)
            except Exception as ex:
                raise RuntimeError(f"Failed to construct CustomEmbeddingCollection: {ex}")
        # load shared CPU prebuilt weights into model (best-effort)
        if rank == 0:
            print(f"[benchmark] loading prebuilt weights for limit={limit.label}...")
        try:
            _try_preload(model, prebuilt_tables, _prefill_from_batch_safe(base_batch, tables_config, cfg, device), device)
            if rank == 0:
                print(f"[benchmark] preload complete for limit={limit.label}")
        except Exception as e:
            # if preload fails, continue; model should still work (will look into CPU weights itself)
            if rank == 0:
                print(f"[benchmark] preload failed (will use on-demand): {e}")
            pass

        # warmup
        if rank == 0:
            print(f"[benchmark] warmup for limit={limit.label} ({cfg.warmup_steps} steps)...")
        with torch.no_grad():
            for step in range(cfg.warmup_steps):
                batch = sample_batch() if cfg.resample_per_step else base_batch
                _ = model(batch)
                if rank == 0 and (step + 1) % max(1, cfg.warmup_steps // 2) == 0:
                    print(f"[benchmark] warmup step {step+1}/{cfg.warmup_steps}")

        # measure
        latencies = []
        if rank == 0:
            print(f"[benchmark] measuring for limit={limit.label} ({cfg.measure_steps} steps)...")
        with torch.no_grad():
            for step in range(cfg.measure_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(sample_batch() if cfg.resample_per_step else base_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)
                # call model.record_cache_usage if present
                if hasattr(model, "record_cache_usage"):
                    try:
                        model.record_cache_usage()
                    except Exception:
                        pass
                # 每 10 步或最后一步打印进度
                if rank == 0 and ((step + 1) % 10 == 0 or step + 1 == cfg.measure_steps):
                    avg_lat_so_far = sum(latencies) / len(latencies) * 1000
                    print(f"[benchmark] measure step {step+1}/{cfg.measure_steps}, avg_lat={avg_lat_so_far:.2f}ms")
        # collect cache stats if available
        if rank == 0:
            print(f"[benchmark] collecting stats for limit={limit.label}...")
        stats = {}
        if hasattr(model, "cache_stats"):
            try:
                stats = model.cache_stats() or {}
            except Exception:
                stats = {}
        # aggregate
        if rank == 0:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            def perc(q):
                if n == 0: return 0.0
                r = (n - 1) * q / 100.0
                lo = int(math.floor(r)); hi = int(math.ceil(r)); w = r - lo
                if hi >= n: hi = n - 1
                if lo == hi: return lat_sorted[lo]
                return lat_sorted[lo] * (1 - w) + lat_sorted[hi] * w
            lat_stats = {
                "avg_ms": (sum(lat_sorted) / max(1, n)) * 1000.0,
                "p50_ms": perc(50.0) * 1000.0,
                "p95_ms": perc(95.0) * 1000.0,
                "p99_ms": perc(99.0) * 1000.0,
            }
            total_elapsed = sum(lat_sorted)
            total_requests = cfg.batch_size * cfg.measure_steps * world_size
            throughput = total_requests / max(total_elapsed, 1e-9)
            results.append({
                "limit": limit.label,
                "bytes": 0.0 if limit.bytes is None else limit.bytes / (1024 ** 2),
                "throughput": throughput,
                "lat_p50": lat_stats["p50_ms"],
                "lat_p95": lat_stats["p95_ms"],
                "lat_p99": lat_stats["p99_ms"],
                "lat_avg": lat_stats["avg_ms"],
                "cache_stats": stats,
                "cache_ratio_used": avg_ratio,
            })
            print(f"[benchmark] completed limit={limit.label}, throughput={throughput:.2f} req/s, p95={lat_stats['p95_ms']:.2f}ms")
        # cleanup
        if rank == 0:
            print(f"[benchmark] cleaning up model for limit={limit.label}...")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"[benchmark] finished limit {limit_idx}/{len(limits)}\n")

    # Optionally include pure CPU baseline
    if include_cpu_baseline:
        if rank == 0:
            print("[benchmark] running CPU baseline (cache disabled)")
        # create model with cache_ratio = 0
        model = None
        try:
            model = CustomEmbeddingCollection(table_config=tables_config, cache_ratio=0.0, device=device,
                                              constraints={n: ParameterConstraints(sharding_types=[sharding.value]) for n in tables_config.keys()})
        except Exception:
            model = CustomEmbeddingCollection(tables_config, cache_ratio=0.0, device=device)
        # run warmup + measure
        with torch.no_grad():
            for _ in range(cfg.warmup_steps):
                model(sample_batch())
            latencies = []
            for _ in range(cfg.measure_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(sample_batch())
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)
        if rank == 0:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            def perc(q):
                if n == 0: return 0.0
                r = (n - 1) * q / 100.0
                lo = int(math.floor(r)); hi = int(math.ceil(r)); w = r - lo
                if hi >= n: hi = n - 1
                if lo == hi: return lat_sorted[lo]
                return lat_sorted[lo] * (1 - w) + lat_sorted[hi] * w
            lat_stats = {
                "avg_ms": (sum(lat_sorted) / max(1, n)) * 1000.0,
                "p50_ms": perc(50.0) * 1000.0,
                "p95_ms": perc(95.0) * 1000.0,
                "p99_ms": perc(99.0) * 1000.0,
            }
            total_elapsed = sum(lat_sorted)
            throughput = (cfg.batch_size * cfg.measure_steps * world_size) / max(total_elapsed, 1e-9)
            results.append({
                "limit": "CPU",
                "bytes": 0.0,
                "throughput": throughput,
                "lat_p50": lat_stats["p50_ms"],
                "lat_p95": lat_stats["p95_ms"],
                "lat_p99": lat_stats["p99_ms"],
                "lat_avg": lat_stats["avg_ms"],
                "cache_stats": {},
                "cache_ratio_used": 0.0,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        # print results summary
        print("==== Embedding Cache Budget Benchmark Results ====")
        header = f"{'limit':>12} {'effective_mb':>12} {'throughput(s/s)':>20} {'avg_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'cache_ratio':>12}"
        print(header)
        for r in results:
            print(f"{r['limit']:>12} {r['bytes']:12.2f} {r['throughput']:20.2f} {r['lat_avg']:10.2f} {r['lat_p95']:10.2f} {r['lat_p99']:10.2f} {r['cache_ratio_used']:12.3f}")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ---------------------------
# Small helper to extract prefill set from a batch (safe)
# ---------------------------
def _prefill_from_batch_safe(batch, tables_config, cfg, device):
    """
    Return a mapping {table_name: tensor(ids)} extracted from the synthetic base batch.
    This is a minimal re-implementation of your larger prefill helper for convenience.
    """
    feature_to_table = {}
    for tname, cfgt in tables_config.items():
        for f in cfgt.feature_names:
            feature_to_table[f] = tname
    per_table = {name: [] for name in tables_config.keys()}
    keys = batch.keys() if hasattr(batch, "keys") else []
    for k in keys:
        tname = feature_to_table.get(k)
        if tname is None:
            continue
        vals = batch[k].values()
        if vals is None or vals.numel() == 0:
            continue
        per_table[tname].append(vals)
    mapping = {}
    for tname, arrs in per_table.items():
        if not arrs:
            continue
        concat = torch.cat(arrs)
        unique = torch.unique(concat)
        desired = max(1, int(tables_config[tname].num_embeddings * cfg.prefill_ratio))
        desired = min(desired, unique.numel())
        if desired <= 0:
            continue
        # pick top "desired" smallest ids (deterministic)
        selected = torch.sort(unique).values[:desired].to(device)
        mapping[tname] = selected
    return mapping

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark embedding cache budgets (shared CPU prebuilt, per-limit GPU cache)")
    parser.add_argument("--batch-size", type=int, default=BenchConfig.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=BenchConfig.warmup_steps)
    parser.add_argument("--measure-steps", type=int, default=BenchConfig.measure_steps)
    parser.add_argument("--num-embeddings", type=int, default=BenchConfig.num_embeddings)
    parser.add_argument("--embedding-dim", type=int, default=BenchConfig.embedding_dim)
    parser.add_argument("--prefill-ratio", type=float, default=BenchConfig.prefill_ratio)
    parser.add_argument("--seed", type=int, default=BenchConfig.seed)
    parser.add_argument("--prebuilt-pin-memory", action="store_true")
    parser.add_argument("--resample-per-step", action="store_true")
    parser.add_argument("--include-cpu-baseline", action="store_true")
    parser.add_argument("--sharding", type=str, default=ShardingType.ROW_WISE.value, choices=[m.value for m in ShardingType])
    parser.add_argument("--limit", dest="limits", action="append", default=None, help="e.g. 2GB, 10GB, none")
    args = parser.parse_args()
    cfg = BenchConfig(batch_size=args.batch_size, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
                      num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim, prefill_ratio=args.prefill_ratio,
                      seed=args.seed, prebuilt_pin_memory=args.prebuilt_pin_memory, resample_per_step=args.resample_per_step)
    raw_limits = args.limits if args.limits is not None else ["none", "10GB", "6GB", "4GB"]
    limits = []
    for v in raw_limits:
        limits.append(_parse_limit(v))
    sharding = ShardingType(args.sharding)
    return cfg, limits, sharding, args.include_cpu_baseline

def main():
    cfg, limits, sharding, include_cpu = parse_args()
    run_benchmark(cfg, limits, sharding, include_cpu_baseline=include_cpu)

if __name__ == "__main__":
    main()

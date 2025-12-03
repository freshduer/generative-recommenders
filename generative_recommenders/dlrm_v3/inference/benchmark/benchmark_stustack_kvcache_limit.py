#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STULayer, STULayerConfig, STUStack


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
    num_layers: int = 6
    num_heads: int = 4
    embedding_dim: int = 256
    attention_dim: int = 64
    hidden_dim: int = 128
    contextual_seq_len: int = 0
    use_group_norm: bool = False
    causal: bool = True
    kernel: str = "triton"
    dtype: str = "float16"
    device: Optional[str] = None
    num_users: int = 64
    prefill_len: int = 15000
    delta_size: int = 128
    warmup_requests: int = 128
    measure_requests: int = 1024
    seed: int = 2025


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
        return total

    def to(self, device: torch.device) -> None:
        if self.k is not None:
            self.k = self.k.to(device, non_blocking=True)
        if self.v is not None:
            self.v = self.v.to(device, non_blocking=True)
        if self.offsets is not None:
            self.offsets = self.offsets.to(device)


@dataclass
class KVEntry:
    caches: List[LayerCache]
    device: torch.device

    def gpu_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return sum(cache.nbytes() for cache in self.caches)


class KVCacheManager:
    def __init__(self, limit_bytes: Optional[int], device: torch.device) -> None:
        self.limit_bytes = limit_bytes
        self.device = device
        self.entries: Dict[int, KVEntry] = {}
        self.lru: OrderedDict[int, None] = OrderedDict()
        self.total_gpu_bytes_samples: float = 0.0
        self.usage_samples: int = 0
        self.max_gpu_bytes: int = 0
        self.num_evictions: int = 0
        self.num_page_ins: int = 0
        self._last_logged_bytes: Optional[int] = None

    def touch(self, user_id: int) -> None:
        self.lru.pop(user_id, None)
        self.lru[user_id] = None

    def capture(self, user_id: int, layers: Sequence[STULayer]) -> None:
        caches: List[LayerCache] = []
        cache_device: Optional[torch.device] = None
        for layer in layers:
            k_cache = layer.k_cache.detach() if layer.k_cache is not None else None
            v_cache = layer.v_cache.detach() if layer.v_cache is not None else None
            offsets = layer.kv_caching_offsets.detach() if layer.kv_caching_offsets is not None else None
            caches.append(
                LayerCache(
                    k=k_cache,
                    v=v_cache,
                    offsets=offsets,
                    max_len=layer.max_kv_caching_len,
                )
            )
            if k_cache is not None:
                cache_device = k_cache.device
        entry_device = cache_device if cache_device is not None else torch.device("cpu")
        self.entries[user_id] = KVEntry(caches=caches, device=entry_device)
        self.touch(user_id)
        for layer in layers:
            layer.reset_kv_cache()
        self._log_gpu_bytes("capture")

    def load_into_layers(self, user_id: int, layers: Sequence[STULayer], device: torch.device) -> None:
        entry = self.entries[user_id]
        if entry.device != device:
            self._move_entry_to_device(user_id, device)
        for layer, cache in zip(layers, entry.caches):
            if cache.k is None or cache.v is None:
                layer.reset_kv_cache()
                continue
            layer.k_cache = cache.k
            layer.v_cache = cache.v
            layer.max_kv_caching_len = cache.max_len
            layer.kv_caching_offsets = cache.offsets
        self.touch(user_id)

    def _move_entry_to_device(self, user_id: int, device: torch.device) -> None:
        entry = self.entries[user_id]
        if entry.device == device:
            return
        from_device = entry.device
        for cache in entry.caches:
            cache.to(device)
        entry.device = device
        if device.type == "cuda" and from_device.type != "cuda":
            self.num_page_ins += 1
            self._log_gpu_bytes("page_in")

    def _move_entry_to_cpu(self, user_id: int) -> bool:
        entry = self.entries[user_id]
        if entry.device.type == "cpu":
            return False
        cpu_device = torch.device("cpu")
        for cache in entry.caches:
            cache.to(cpu_device)
        entry.device = cpu_device
        self.num_evictions += 1
        self._log_gpu_bytes("evict")
        return True

    def gpu_bytes(self) -> int:
        return sum(entry.gpu_bytes() for entry in self.entries.values())

    def enforce_limit(self, active_user: Optional[int]) -> None:
        if self.limit_bytes is None:
            return
        while self.gpu_bytes() > self.limit_bytes:
            candidate: Optional[int] = None
            # Evict least-recently used users while keeping the active user resident.
            for user_id in self.lru.keys():
                if user_id == active_user:
                    continue
                entry = self.entries[user_id]
                if entry.device.type == "cuda":
                    candidate = user_id
                    break
            if candidate is None:
                break
            self._move_entry_to_cpu(candidate)

    def reset_metrics(self) -> None:
        self.total_gpu_bytes_samples = 0.0
        self.usage_samples = 0
        self.max_gpu_bytes = self.gpu_bytes()
        self.num_evictions = 0
        self.num_page_ins = 0

    def record_usage(self) -> None:
        bytes_now = self.gpu_bytes()
        self.total_gpu_bytes_samples += bytes_now
        self.usage_samples += 1
        if bytes_now > self.max_gpu_bytes:
            self.max_gpu_bytes = bytes_now
        self._log_gpu_bytes("usage")

    def average_gpu_mb(self) -> float:
        if self.usage_samples == 0:
            return self.gpu_bytes() / (1024**2)
        return (self.total_gpu_bytes_samples / self.usage_samples) / (1024**2)

    def _log_gpu_bytes(self, event: str) -> None:
        if self.device.type != "cuda":
            return
        current = self.gpu_bytes()
        if self._last_logged_bytes == current:
            return
        mb = current / (1024**2)
        limit_mb = None if self.limit_bytes is None else self.limit_bytes / (1024**2)
        limit_text = "unlimited" if limit_mb is None else f"{limit_mb:.2f} MB"
        print(f"[kv-cache][{event}] gpu_usage={mb:.2f} MB limit={limit_text}")
        self._last_logged_bytes = current


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
        self.delta_targets = torch.full((1,), cfg.delta_size, device=device, dtype=torch.int64)
        self.prefill_lengths = torch.full((1,), cfg.prefill_len, device=device, dtype=torch.int64)
        self.prefill_offsets = _complete_cumsum(self.prefill_lengths)
        self.prefill_targets = torch.zeros_like(self.prefill_lengths)
        self.generators: List[torch.Generator] = []
        for idx in range(cfg.num_users):
            if device.type == "cuda":
                gen = torch.Generator(device=device)
            else:
                gen = torch.Generator()
            gen.manual_seed(cfg.seed + idx)
            # Keep per-user RNG state so different limits replay identical traffic.
            self.generators.append(gen)

    def _reset_layers(self) -> None:
        for layer in self.layers:
            layer.reset_kv_cache()

    def prefill_user(self, user_id: int) -> None:
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
        self.manager.capture(user_id, self.layers)
        self.manager.enforce_limit(active_user=user_id)

    def prefill_all_users(self) -> None:
        for user_id in range(self.cfg.num_users):
            self.prefill_user(user_id)

    def decode_user(self, user_id: int) -> None:
        self._reset_layers()
        self.manager.load_into_layers(user_id, self.layers, self.device)
        self.manager.enforce_limit(active_user=user_id)
        generator = self.generators[user_id]
        delta = torch.randn(
            self.cfg.delta_size,
            self.cfg.embedding_dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.stu.cached_forward(delta, self.delta_targets)
        self.manager.capture(user_id, self.layers)
        self.manager.enforce_limit(active_user=user_id)

    def run_requests(
        self,
        request_ids: Sequence[int],
        collect_stats: bool,
        record_latencies: Optional[List[float]] = None,
    ) -> None:
        for user_id in request_ids:
            t0 = time.perf_counter() if record_latencies is not None else 0.0
            self.decode_user(int(user_id))
            if collect_stats:
                self.manager.record_usage()
            if record_latencies is not None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                record_latencies.append(time.perf_counter() - t0)


def _pick_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _pick_kernel(name: str) -> HammerKernel:
    return HammerKernel.TRITON if name.lower() != "pytorch" else HammerKernel.PYTORCH


def _select_device(arg: Optional[str]) -> torch.device:
    if arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


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
    if text in {"none", "unlimited", "inf", "infinite"}:
        return BenchmarkLimit(label="unlimited", bytes=None)
    units = {
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
    }
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            number = float(text[:- len(suffix)])
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
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", text):
        raise ValueError(f"Could not parse limit value: {value}")
    number = float(text)
    return BenchmarkLimit(label=f"{number:g}MB", bytes=int(number * 1024**2))


def _format_limit(limit: BenchmarkLimit) -> str:
    return limit.label


def _compute_latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {key: 0.0 for key in ["avg_ms", "p50_ms", "p95_ms", "p99_ms"]}
    n = len(latencies)
    sorted_lat = sorted(latencies)

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

    avg_ms = sum(latencies) / n * 1000.0
    mid = percentile(50.0) * 1000.0
    p95 = percentile(95.0) * 1000.0
    p99 = percentile(99.0) * 1000.0
    return {
        "avg_ms": avg_ms,
        "p50_ms": mid,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def run_benchmark(cfg: BenchConfig, limits: List[BenchmarkLimit]) -> None:
    device = _select_device(cfg.device)
    dtype = _pick_dtype(cfg.dtype)
    kernel = _pick_kernel(cfg.kernel)

    set_dev_mode(False)
    STULayer.log_kv_cache_size = lambda self, prefix="": None  # type: ignore

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    base_stu = _build_stu(cfg, device, kernel)
    base_state = base_stu.state_dict()
    del base_stu

    request_gen = torch.Generator()
    request_gen.manual_seed(cfg.seed * 17 + 7)
    total_requests = cfg.warmup_requests + cfg.measure_requests
    request_sequence = torch.randint(
        0,
        cfg.num_users,
        (total_requests,),
        generator=request_gen,
    ).tolist()
    warmup_requests = request_sequence[: cfg.warmup_requests]
    measure_requests = request_sequence[cfg.warmup_requests :]

    results = []

    for limit in limits:
        stu = _build_stu(cfg, device, kernel)
        stu.load_state_dict(base_state)
        manager = KVCacheManager(limit.bytes, device)
        service = StuService(cfg, stu, manager, device, dtype)

        with torch.inference_mode():
            service.prefill_all_users()
            if cfg.warmup_requests > 0:
                service.run_requests(warmup_requests, collect_stats=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            manager.reset_metrics()
            if cfg.measure_requests == 0:
                throughput = 0.0
                latencies: List[float] = []
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                latencies = []
                service.run_requests(
                    measure_requests,
                    collect_stats=True,
                    record_latencies=latencies,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                throughput = len(measure_requests) / max(elapsed, 1e-9)

        avg_gpu_mb = manager.average_gpu_mb()
        max_gpu_mb = manager.max_gpu_bytes / (1024**2)
        lat_stats = _compute_latency_stats(latencies)
        results.append(
            {
                "limit": _format_limit(limit),
                "throughput": throughput,
                "avg_gpu_mb": avg_gpu_mb,
                "max_gpu_mb": max_gpu_mb,
                "evictions": manager.num_evictions,
                "page_ins": manager.num_page_ins,
                "lat_p50": lat_stats["p50_ms"],
                "lat_p95": lat_stats["p95_ms"],
                "lat_p99": lat_stats["p99_ms"],
                "lat_avg": lat_stats["avg_ms"],
            }
        )

        del service
        del manager
        del stu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("==== STUStack KV-Cache Budget Benchmark ====")
    print(f"device: {device}, kernel: {kernel.name}, dtype: {dtype}")
    print(
        "layers: {layers}, heads: {heads}, embed: {embed}, attn: {attn}, hidden: {hidden}".format(
            layers=cfg.num_layers,
            heads=cfg.num_heads,
            embed=cfg.embedding_dim,
            attn=cfg.attention_dim,
            hidden=cfg.hidden_dim,
        )
    )
    print(
        "users: {users}, prefill_len: {prefill}, delta_size: {delta}, warmup: {warmup}, measure: {measure}".format(
            users=cfg.num_users,
            prefill=cfg.prefill_len,
            delta=cfg.delta_size,
            warmup=cfg.warmup_requests,
            measure=cfg.measure_requests,
        )
    )
    print("-- Results --")
    header = "{limit:>12} {throughput:>18} {avg_gpu:>14} {max_gpu:>12} {evict:>10} {page_in:>10} {p50:>10} {p95:>10} {p99:>10}".format(
        limit="limit",
        throughput="throughput (req/s)",
        avg_gpu="avg_gpu_mb",
        max_gpu="max_gpu_mb",
        evict="evictions",
        page_in="page_ins",
        p50="p50_ms",
        p95="p95_ms",
        p99="p99_ms",
    )
    print(header)
    for row in results:
        print(
            "{limit:>12} {throughput:>18.2f} {avg_gpu:>14.2f} {max_gpu:>12.2f} {evict:>10} {page_in:>10} {p50:>10.2f} {p95:>10.2f} {p99:>10.2f}".format(
                limit=row["limit"],
                throughput=row["throughput"],
                avg_gpu=row["avg_gpu_mb"],
                max_gpu=row["max_gpu_mb"],
                evict=row["evictions"],
                page_in=row["page_ins"],
                p50=row["lat_p50"],
                p95=row["lat_p95"],
                p99=row["lat_p99"],
            )
        )
    print("-- Latency Averages (ms) --")
    for row in results:
        print(f"{row['limit']:>12} avg={row['lat_avg']:.2f}")


def parse_args() -> Tuple[BenchConfig, List[BenchmarkLimit]]:
    parser = argparse.ArgumentParser(
        description="Benchmark STUStack KV-cache throughput under GPU memory budgets",
    )
    parser.add_argument("--num-layers", type=int, default=BenchConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=BenchConfig.num_heads)
    parser.add_argument("--embedding-dim", type=int, default=BenchConfig.embedding_dim)
    parser.add_argument("--attention-dim", type=int, default=BenchConfig.attention_dim)
    parser.add_argument("--hidden-dim", type=int, default=BenchConfig.hidden_dim)
    parser.add_argument("--contextual-seq-len", type=int, default=BenchConfig.contextual_seq_len)
    parser.add_argument("--use-group-norm", action="store_true", default=BenchConfig.use_group_norm)
    parser.add_argument("--causal", action="store_true", default=BenchConfig.causal)
    parser.add_argument("--kernel", type=str, default=BenchConfig.kernel, choices=["triton", "pytorch"])
    parser.add_argument("--dtype", type=str, default=BenchConfig.dtype, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--num-users", type=int, default=BenchConfig.num_users)
    parser.add_argument("--prefill-len", type=int, default=BenchConfig.prefill_len)
    parser.add_argument("--delta-size", type=int, default=BenchConfig.delta_size)
    parser.add_argument("--warmup-requests", type=int, default=BenchConfig.warmup_requests)
    parser.add_argument("--measure-requests", type=int, default=BenchConfig.measure_requests)
    parser.add_argument("--seed", type=int, default=BenchConfig.seed)
    parser.add_argument(
        "--limit",
        dest="limits",
        action="append",
        default=None,
        help="GPU KV-cache budget (e.g. 2048, 2GB, 512MB, none). Repeat to test multiple budgets.",
    )
    args = parser.parse_args()

    cfg = BenchConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        contextual_seq_len=args.contextual_seq_len,
        use_group_norm=args.use_group_norm,
        causal=args.causal,
        kernel=args.kernel,
        dtype=args.dtype,
        device=args.device,
        num_users=args.num_users,
        prefill_len=args.prefill_len,
        delta_size=args.delta_size,
        warmup_requests=args.warmup_requests,
        measure_requests=args.measure_requests,
        seed=args.seed,
    )

    raw_limits = args.limits if args.limits is not None else ["none", "2048", "1024"]
    limits = []
    for value in raw_limits:
        try:
            limits.append(_parse_limit(value))
        except ValueError as exc:
            raise SystemExit(str(exc))

    return cfg, limits


def main() -> None:
    cfg, limits = parse_args()
    run_benchmark(cfg, limits)


if __name__ == "__main__":
    main()

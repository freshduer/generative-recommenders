#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.jagged_tensors import split_2D_jagged


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


def _split_prime_delta_simple(values: torch.Tensor, offsets_left: torch.Tensor, delta_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split flattened jagged tensor into prime and delta parts."""
    B = offsets_left.numel() - 1
    D = values.shape[1]
    prime_chunks = []
    delta_chunks = []
    for b in range(B):
        start = int(offsets_left[b].item())
        end = int(offsets_left[b + 1].item())
        assert end >= start + delta_size, "delta_size exceeds per-batch length"
        split = end - delta_size
        prime_chunks.append(values[start:split, :])
        delta_chunks.append(values[split:end, :])
    prime_values = torch.cat(prime_chunks, dim=0) if prime_chunks else values.new_zeros((0, D))
    delta_values = torch.cat(delta_chunks, dim=0) if delta_chunks else values.new_zeros((0, D))
    return prime_values, delta_values


@dataclass
class BenchConfig:
    batch_size: int = 16
    num_layers: int = 8
    num_heads: int = 4
    embedding_dim: int = 256
    attention_dim: int = 64
    hidden_dim: int = 128
    max_uih_len: int = 15000  # long sequence
    contextual_seq_len: int = 0
    delta_size: int = 64  # block decode size
    kernel: str = "triton"
    dtype: str = "float16"
    device: Optional[str] = None
    warmup: int = 2
    iters: int = 20
    seed: int = 2025
    use_group_norm: bool = False
    causal: bool = True


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


def _build_stu(cfg: BenchConfig, device: torch.device, dtype: torch.dtype, kernel: HammerKernel) -> STUStack:
    layers: List[STU] = [
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


def _rand_inputs(cfg: BenchConfig, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Generate a single long sequence per batch for testing long-user-sequences."""
    batch_size = cfg.batch_size
    delta_size = cfg.delta_size
    contextual_seq_len = cfg.contextual_seq_len
    max_uih_len = cfg.max_uih_len

    # 每个用户序列长度 = max_uih_len + contextual_seq_len
    x_lengths = torch.full((batch_size,), max_uih_len + contextual_seq_len, device=device)
    x_offsets = _complete_cumsum(x_lengths)
    total_seq_len = int(x_offsets[-1].item())
    x = torch.randn(total_seq_len, cfg.embedding_dim, device=device, dtype=dtype)

    # num_targets = delta_size
    num_targets = torch.full((batch_size,), delta_size, device=device)

    return x, x_lengths, x_offsets, num_targets, int(total_seq_len), delta_size, int(total_seq_len)


def _estimate_kv_bytes(stu: STUStack) -> int:
    total_bytes = 0
    for layer in stu._stu_layers:
        k = getattr(layer, "k_cache", None)
        v = getattr(layer, "v_cache", None)
        if k is None or v is None:
            continue
        total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total_bytes


def run_benchmark(cfg: BenchConfig) -> None:
    torch.manual_seed(cfg.seed)
    set_dev_mode(True)
    device = _select_device(cfg.device)
    dtype = _pick_dtype(cfg.dtype)
    kernel = _pick_kernel(cfg.kernel)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    stu = _build_stu(cfg, device, dtype, kernel)
    x, x_lengths, x_offsets, num_targets, max_seq_len, delta_size, _ = _rand_inputs(cfg, device, dtype)

    # prime / delta split
    prime_lengths = x_lengths - delta_size
    prime_offsets = _complete_cumsum(prime_lengths)

    try:
        prime_x, delta_x = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=x,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=kernel,
        )
    except AttributeError:
        prime_x, delta_x = _split_prime_delta_simple(x, x_offsets, delta_size)

    # warmup
    for _ in range(cfg.warmup):
        _ = stu(prime_x, prime_lengths, prime_offsets, max_seq_len, num_targets - delta_size, max_seq_len - delta_size, x_lengths - delta_size)
        _ = stu.cached_forward(delta_x, num_targets)
        for layer in stu._stu_layers:
            layer.reset_kv_cache()

    torch.cuda.synchronize() if device.type == "cuda" else None

    # prefill timing
    t0 = time.perf_counter()
    _ = stu(prime_x, prime_lengths, prime_offsets, max_seq_len, num_targets - delta_size, max_seq_len - delta_size, x_lengths - delta_size)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_prefill = time.perf_counter() - t0

    # decode timing
    t_decode = 0.0
    for _ in range(cfg.iters):
        t1 = time.perf_counter()
        _ = stu.cached_forward(delta_x, num_targets)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_decode += time.perf_counter() - t1
    t_decode_avg = t_decode / max(cfg.iters, 1)

    kv_bytes = _estimate_kv_bytes(stu)

    # numerical check
    with torch.inference_mode():
        ref_y = stu(x, x_lengths, x_offsets, max_seq_len, num_targets)
        try:
            _, ref_delta_y = split_2D_jagged(
                max_seq_len=max_seq_len,
                values=ref_y,
                max_len_left=None,
                max_len_right=delta_size,
                offsets_left=prime_offsets,
                offsets_right=None,
                kernel=kernel,
            )
        except AttributeError:
            _, ref_delta_y = _split_prime_delta_simple(ref_y, x_offsets, delta_size)
        delta_y = stu.cached_forward(delta_x, num_targets)
        try:
            atol = 5e-3 if dtype == torch.bfloat16 else 0.0
            torch.testing.assert_close(ref_delta_y, delta_y, rtol=1e-2, atol=atol)
            err = 0.0
        except AssertionError:
            err = float((ref_delta_y - delta_y).abs().max().item())

    # full forward timing
    for _ in range(cfg.warmup):
        _ = stu(x, x_lengths, x_offsets, max_seq_len, num_targets)
    torch.cuda.synchronize() if device.type == "cuda" else None

    base_time = 0.0
    for _ in range(cfg.iters):
        t1 = time.perf_counter()
        _ = stu(x, x_lengths, x_offsets, max_seq_len, num_targets)
        torch.cuda.synchronize() if device.type == "cuda" else None
        base_time += time.perf_counter() - t1
    base_avg = base_time / max(cfg.iters, 1)

    # print results
    total_tokens = int(x_offsets[-1].item())
    prime_tokens = int(prime_offsets[-1].item())
    delta_tokens = int((x_offsets[-1] - prime_offsets[-1]).item())

    def fmt_bytes(n: int) -> str:
        return f"{n/1024/1024:.2f} MB" if n else "0 MB"

    print("==== STUStack KV-Cache Benchmark ====")
    print(f"device: {device}, kernel: {kernel.name}, dtype: {dtype}")
    print(f"layers: {cfg.num_layers}, heads: {cfg.num_heads}, embed: {cfg.embedding_dim}, attn: {cfg.attention_dim}, hidden: {cfg.hidden_dim}")
    print(f"batch: {cfg.batch_size}, max_uih_len: {cfg.max_uih_len}, contextual_seq_len: {cfg.contextual_seq_len}")
    print(f"tokens: total={total_tokens}, prime={prime_tokens}, delta={delta_tokens}")
    print(f"kv cache size (sum over layers): {fmt_bytes(kv_bytes)}")
    print("-- Timing (averaged over iters) --")
    print(f"prefill_once: {t_prefill*1000:.2f} ms")
    print(f"decode_cached: {t_decode_avg*1000:.2f} ms per block (size {delta_size})")
    print(f"forward_full: {base_avg*1000:.2f} ms per full sequence")
    spd = base_avg / max(t_decode_avg, 1e-9)
    print(f"speedup (forward_full / decode_cached): {spd:.2f}x")
    print(f"delta output max abs error vs full forward: {err:.6f}")


def parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Benchmark STUStack KV-cache reuse for block decoding")
    p.add_argument("--batch-size", type=int, default=BenchConfig.batch_size)
    p.add_argument("--num-layers", type=int, default=BenchConfig.num_layers)
    p.add_argument("--num-heads", type=int, default=BenchConfig.num_heads)
    p.add_argument("--embedding-dim", type=int, default=BenchConfig.embedding_dim)
    p.add_argument("--attention-dim", type=int, default=BenchConfig.attention_dim)
    p.add_argument("--hidden-dim", type=int, default=BenchConfig.hidden_dim)
    p.add_argument("--max-uih-len", type=int, default=BenchConfig.max_uih_len)
    p.add_argument("--contextual-seq-len", type=int, default=BenchConfig.contextual_seq_len)
    p.add_argument("--delta-size", type=int, default=BenchConfig.delta_size)
    p.add_argument("--kernel", type=str, default=BenchConfig.kernel, choices=["triton", "pytorch"])
    p.add_argument("--dtype", type=str, default=BenchConfig.dtype, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--warmup", type=int, default=BenchConfig.warmup)
    p.add_argument("--iters", type=int, default=BenchConfig.iters)
    p.add_argument("--seed", type=int, default=BenchConfig.seed)
    p.add_argument("--use-group-norm", action="store_true", default=BenchConfig.use_group_norm)
    p.add_argument("--causal", action="store_true", default=True)
    args = p.parse_args()
    return BenchConfig(
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        max_uih_len=args.max_uih_len,
        contextual_seq_len=args.contextual_seq_len,
        delta_size=args.delta_size,
        kernel=args.kernel,
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        use_group_norm=args.use_group_norm,
        causal=args.causal,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_benchmark(cfg)

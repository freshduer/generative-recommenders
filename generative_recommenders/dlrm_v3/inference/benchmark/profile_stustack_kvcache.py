#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.profiler import (  # type: ignore[attr-defined]
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.jagged_tensors import split_2D_jagged


def _complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    try:
        fbgemm_ops = getattr(torch.ops, "fbgemm", None)
        if fbgemm_ops is not None and hasattr(fbgemm_ops, "asynchronous_complete_cumsum"):
            return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except Exception:
        pass
    zero = torch.zeros(1, device=lengths.device, dtype=lengths.dtype)
    return torch.cat([zero, torch.cumsum(lengths, dim=0)])


def _split_prime_delta_simple(
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    delta_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = offsets_left.numel() - 1
    dim = values.shape[1]
    prime_chunks: List[torch.Tensor] = []
    delta_chunks: List[torch.Tensor] = []
    for b in range(batch_size):
        start = int(offsets_left[b].item())
        end = int(offsets_left[b + 1].item())
        if end < start + delta_size:
            raise ValueError("delta_size exceeds per-batch length")
        split = end - delta_size
        prime_chunks.append(values[start:split, :])
        delta_chunks.append(values[split:end, :])
    prime_values = torch.cat(prime_chunks, dim=0) if prime_chunks else values.new_zeros((0, dim))
    delta_values = torch.cat(delta_chunks, dim=0) if delta_chunks else values.new_zeros((0, dim))
    return prime_values, delta_values


def _pick_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in ("float16", "fp16", "half"):
        return torch.float16
    if lowered in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _pick_kernel(name: str) -> HammerKernel:
    return HammerKernel.TRITON if name.lower() != "pytorch" else HammerKernel.PYTORCH


def _select_device(arg: Optional[str]) -> torch.device:
    if arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _build_stu(
    num_layers: int,
    cfg: "ProfileConfig",
    device: torch.device,
    dtype: torch.dtype,
    kernel: HammerKernel,
) -> STUStack:
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
        for _ in range(num_layers)
    ]
    stack = STUStack(stu_list=layers, is_inference=True).to(device)
    stack.eval()
    stack.recursive_setattr("_hammer_kernel", kernel)
    return stack


def _rand_inputs(
    cfg: "ProfileConfig",
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    batch_size = cfg.batch_size
    delta_size = cfg.delta_size
    contextual_seq_len = cfg.contextual_seq_len
    max_uih_len = cfg.max_uih_len

    x_lengths = torch.full((batch_size,), max_uih_len + contextual_seq_len, device=device)
    x_offsets = _complete_cumsum(x_lengths)
    total_seq_len = int(x_offsets[-1].item())
    x = torch.randn(total_seq_len, cfg.embedding_dim, device=device, dtype=dtype)
    num_targets = torch.full((batch_size,), delta_size, device=device)
    return x, x_lengths, x_offsets, num_targets, total_seq_len, delta_size


def _safe_split(
    max_seq_len: int,
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    delta_size: int,
    kernel: HammerKernel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=offsets_left,
            offsets_right=None,
            kernel=kernel,
        )
    except AttributeError:
        return _split_prime_delta_simple(values, offsets_left, delta_size)


def _parse_activities(spec: str) -> Sequence[ProfilerActivity]:
    choices: Dict[str, ProfilerActivity] = {
        "cpu": ProfilerActivity.CPU,
        "cuda": ProfilerActivity.CUDA,
    }
    parsed: List[ProfilerActivity] = []
    for token in spec.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in choices:
            raise ValueError(f"Unsupported profiler activity: {key}")
        parsed.append(choices[key])
    return parsed or [ProfilerActivity.CPU]


def _capture_cache_state(stu: STUStack) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]]:
    state: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]] = []
    for layer in stu._stu_layers:
        k = None if layer.k_cache is None else layer.k_cache.detach().clone()
        v = None if layer.v_cache is None else layer.v_cache.detach().clone()
        offsets = None if layer.kv_caching_offsets is None else layer.kv_caching_offsets.detach().clone()
        state.append((k, v, offsets, layer.max_kv_caching_len))
    return state


def _restore_cache_state(stu: STUStack, state: Sequence[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]]) -> None:
    for layer, (k, v, offsets, max_len) in zip(stu._stu_layers, state):
        layer.k_cache = None if k is None else k.detach().clone()
        layer.v_cache = None if v is None else v.detach().clone()
        layer.kv_caching_offsets = None if offsets is None else offsets.detach().clone()
        layer.max_kv_caching_len = max_len


@dataclass
class ProfileConfig:
    batch_size: int = 16
    num_layers: int = 8
    num_heads: int = 4
    embedding_dim: int = 256
    attention_dim: int = 64
    hidden_dim: int = 128
    max_uih_len: int = 2000
    contextual_seq_len: int = 0
    delta_size: int = 64
    kernel: str = "triton"
    dtype: str = "float16"
    device: Optional[str] = None
    seed: int = 2025
    use_group_norm: bool = False
    causal: bool = True
    phase: str = "both"
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 4
    repeat: int = 1
    activities: str = "cpu,cuda"
    logdir: str = "./profile_logs"
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    profiler_row_limit: int = 20
    metrics: Optional[str] = None


def run_profile(cfg: ProfileConfig) -> None:
    torch.manual_seed(cfg.seed)
    set_dev_mode(True)
    device = _select_device(cfg.device)
    dtype = _pick_dtype(cfg.dtype)
    kernel = _pick_kernel(cfg.kernel)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    stu = _build_stu(cfg.num_layers, cfg, device, dtype, kernel)
    x, x_lengths, x_offsets, num_targets, max_seq_len, delta_size = _rand_inputs(cfg, device, dtype)
    prime_lengths = x_lengths - delta_size
    prime_offsets = _complete_cumsum(prime_lengths)

    prime_x, delta_x = _safe_split(
        max_seq_len=max_seq_len,
        values=x,
        offsets_left=prime_offsets,
        delta_size=delta_size,
        kernel=kernel,
    )

    phases: List[str] = []
    lowered_phase = cfg.phase.lower()
    if lowered_phase in ("prefill", "both"):
        phases.append("prefill")
    if lowered_phase in ("decode", "both"):
        phases.append("decode")

    activities = _parse_activities(cfg.activities)
    sort_key = "self_cuda_time_total" if ProfilerActivity.CUDA in activities else "self_cpu_time_total"

    exp_config = None
    if cfg.metrics:
        try:
            metrics = [token.strip() for token in cfg.metrics.split(",") if token.strip()]
            if metrics:
                exp_config = torch.profiler._ExperimentalConfig(  # type: ignore[attr-defined]
                    profiler_metrics=metrics,
                )
        except AttributeError:
            exp_config = None

    prof_schedule = schedule(
        wait=cfg.wait_steps,
        warmup=cfg.warmup_steps,
        active=cfg.active_steps,
        repeat=cfg.repeat,
    )
    total_steps = (cfg.wait_steps + cfg.warmup_steps + cfg.active_steps) * cfg.repeat

    if device.type == "cuda":
        torch.cuda.synchronize()

    for phase in phases:
        trace_handler = None
        if cfg.logdir:
            phase_dir = Path(cfg.logdir) / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            trace_handler = tensorboard_trace_handler(str(phase_dir))

        @torch.inference_mode()
        def prefill_step() -> None:
            for layer in stu._stu_layers:
                layer.reset_kv_cache()
            stu(
                prime_x,
                prime_lengths,
                prime_offsets,
                max_seq_len,
                num_targets - delta_size,
                max_seq_len - delta_size,
                x_lengths - delta_size,
            )

        @torch.inference_mode()
        def decode_step(state: Sequence[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]]) -> None:
            _restore_cache_state(stu, state)
            stu.cached_forward(delta_x, num_targets)

        if phase == "prefill":
            with profile(
                activities=activities,
                schedule=prof_schedule,
                on_trace_ready=trace_handler,
                record_shapes=cfg.record_shapes,
                profile_memory=cfg.profile_memory,
                with_stack=cfg.with_stack,
                experimental_config=exp_config,
            ) as prof:
                for step_idx in range(total_steps):
                    prefill_step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    prof.step()
            print(f"==== Profiler ({phase}) ====")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=cfg.profiler_row_limit))
            continue

        if phase == "decode":
            prefill_step()
            cache_state = _capture_cache_state(stu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            with profile(
                activities=activities,
                schedule=prof_schedule,
                on_trace_ready=trace_handler,
                record_shapes=cfg.record_shapes,
                profile_memory=cfg.profile_memory,
                with_stack=cfg.with_stack,
                experimental_config=exp_config,
            ) as prof:
                for step_idx in range(total_steps):
                    decode_step(cache_state)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    prof.step()
            print(f"==== Profiler ({phase}) ====")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=cfg.profiler_row_limit))


def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(description="Profile STUStack KV-cache phases")
    parser.add_argument("--batch-size", type=int, default=ProfileConfig.batch_size)
    parser.add_argument("--num-layers", type=int, default=ProfileConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=ProfileConfig.num_heads)
    parser.add_argument("--embedding-dim", type=int, default=ProfileConfig.embedding_dim)
    parser.add_argument("--attention-dim", type=int, default=ProfileConfig.attention_dim)
    parser.add_argument("--hidden-dim", type=int, default=ProfileConfig.hidden_dim)
    parser.add_argument("--max-uih-len", type=int, default=ProfileConfig.max_uih_len)
    parser.add_argument("--contextual-seq-len", type=int, default=ProfileConfig.contextual_seq_len)
    parser.add_argument("--delta-size", type=int, default=ProfileConfig.delta_size)
    parser.add_argument("--kernel", type=str, default=ProfileConfig.kernel, choices=["triton", "pytorch"])
    parser.add_argument("--dtype", type=str, default=ProfileConfig.dtype, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=ProfileConfig.seed)
    parser.add_argument("--use-group-norm", action="store_true", default=ProfileConfig.use_group_norm)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--phase", type=str, default=ProfileConfig.phase, choices=["prefill", "decode", "both"])
    parser.add_argument("--wait-steps", type=int, default=ProfileConfig.wait_steps)
    parser.add_argument("--warmup-steps", type=int, default=ProfileConfig.warmup_steps)
    parser.add_argument("--active-steps", type=int, default=ProfileConfig.active_steps)
    parser.add_argument("--repeat", type=int, default=ProfileConfig.repeat)
    parser.add_argument("--activities", type=str, default=ProfileConfig.activities)
    parser.add_argument("--logdir", type=str, default=ProfileConfig.logdir)
    parser.add_argument("--record-shapes", action="store_true", default=ProfileConfig.record_shapes)
    parser.add_argument("--no-record-shapes", action="store_false", dest="record_shapes")
    parser.add_argument("--profile-memory", action="store_true", default=ProfileConfig.profile_memory)
    parser.add_argument("--no-profile-memory", action="store_false", dest="profile_memory")
    parser.add_argument("--with-stack", action="store_true", default=ProfileConfig.with_stack)
    parser.add_argument("--profiler-row-limit", type=int, default=ProfileConfig.profiler_row_limit)
    parser.add_argument("--metrics", type=str, default=ProfileConfig.metrics)
    args = parser.parse_args()
    return ProfileConfig(
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
        seed=args.seed,
        use_group_norm=args.use_group_norm,
        causal=args.causal,
        phase=args.phase,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        repeat=args.repeat,
        activities=args.activities,
        logdir=args.logdir,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        profiler_row_limit=args.profiler_row_limit,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    run_profile(parse_args())

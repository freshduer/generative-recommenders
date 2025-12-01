#!/usr/bin/env python3
"""Utility script to probe torch.ops.fbgemm.asynchronous_complete_cumsum."""

import argparse
import sys
from typing import Sequence

import torch
import fbgemm_gpu

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether torch.ops.fbgemm.asynchronous_complete_cumsum is available and run a quick correctness test."
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="*",
        help="Explicit sequence lengths to feed into the operator (default: random).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Number of random lengths to generate when --lengths is omitted.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Upper bound (inclusive) for randomly generated lengths.",
    )
    parser.add_argument(
        "--dtype",
        choices=["int32", "int64"],
        default="int32",
        help="Integer dtype to use for the lengths tensor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when generating lengths.",
    )
    return parser.parse_args()


def build_lengths(args: argparse.Namespace) -> torch.Tensor:
    if args.lengths:
        data: Sequence[int] = args.lengths
    else:
        gen = torch.Generator().manual_seed(args.seed)
        data = torch.randint(
            low=1,
            high=args.max_len + 1,
            size=(args.batch_size,),
            generator=gen,
        ).tolist()
    dtype = torch.int32 if args.dtype == "int32" else torch.int64
    return torch.tensor(data, dtype=dtype, device="cpu")


def has_async_cumsum() -> bool:
    fbgemm_ops = getattr(torch.ops, "fbgemm", None)
    return fbgemm_ops is not None and hasattr(fbgemm_ops, "asynchronous_complete_cumsum")


def reference_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(1, device=lengths.device, dtype=lengths.dtype)
    return torch.cat([zero, torch.cumsum(lengths, dim=0)])


def main() -> int:
    args = parse_args()
    lengths = build_lengths(args)

    print(f"lengths tensor: {lengths.tolist()} (dtype={lengths.dtype})")

    if not has_async_cumsum():
        print("torch.ops.fbgemm.asynchronous_complete_cumsum is NOT available in this build.")
        return 1

    print("torch.ops.fbgemm.asynchronous_complete_cumsum is available. Running test...")
    try:
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"Operator invocation failed with: {exc!r}")
        return 2

    expected = reference_complete_cumsum(lengths)
    max_abs_diff = float((result - expected).abs().max().item())
    matches = torch.equal(result, expected)

    print(f"operator output: {result.tolist()}")
    print(f"reference output: {expected.tolist()}")
    print(f"max abs diff: {max_abs_diff}")
    print(f"exact match: {matches}")

    if not matches:
        print("WARNING: operator output differs from reference cumsum result.")
        return 3

    print("Success: operator output matches reference result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

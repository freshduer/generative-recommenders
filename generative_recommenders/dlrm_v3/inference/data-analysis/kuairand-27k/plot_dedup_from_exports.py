#!/usr/bin/env python3
"""Plot user-level deduplication statistics from precomputed CSV exports."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception:  # pragma: no cover - plotting optional
    plt = None
    FuncFormatter = None


def _load_frames(interactions_path: Path, surplus_path: Path) -> pd.DataFrame:
    interactions = pd.read_csv(interactions_path)
    duplicates = pd.read_csv(surplus_path)

    required_interaction_cols = {"user_id", "interaction_count"}
    required_duplicate_cols = {"user_id", "duplicate_surplus"}
    if not required_interaction_cols.issubset(interactions.columns):
        missing = required_interaction_cols - set(interactions.columns)
        raise ValueError(
            f"Missing columns in interactions CSV: {sorted(missing)}"
        )
    if not required_duplicate_cols.issubset(duplicates.columns):
        missing = required_duplicate_cols - set(duplicates.columns)
        raise ValueError(
            f"Missing columns in duplicate-surplus CSV: {sorted(missing)}"
        )

    frame = interactions.merge(duplicates, on="user_id", how="left")
    frame["duplicate_surplus"] = frame["duplicate_surplus"].fillna(0).astype(int)
    frame["interaction_count"] = frame["interaction_count"].astype(int)
    frame["deduplicated_count"] = (
        frame["interaction_count"] - frame["duplicate_surplus"]
    ).clip(lower=0)
    with pd.option_context("mode.use_inf_as_na", True):
        frame["duplicate_ratio"] = (
            frame["duplicate_surplus"] / frame["interaction_count"].replace(0, pd.NA)
        ).fillna(0.0)
        frame["dedup_ratio"] = (
            frame["deduplicated_count"] / frame["interaction_count"].replace(0, pd.NA)
        ).fillna(0.0)
    return frame


def _print_summary(frame: pd.DataFrame) -> None:
    total_rows = int(frame["interaction_count"].sum())
    dedup_rows = int(frame["deduplicated_count"].sum())
    duplicate_loss = int(frame["duplicate_surplus"].sum())
    unique_users = frame.shape[0]
    nonzero_duplicates = int((frame["duplicate_surplus"] > 0).sum())
    percentile_targets: Tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)

    def _percentile(series: pd.Series) -> pd.Series:
        return series.quantile(percentile_targets)

    duplicate_ratio_pct = frame["duplicate_ratio"] * 100.0
    duplicate_user_share = (
        (nonzero_duplicates / unique_users) * 100.0 if unique_users else 0.0
    )
    percentile_values = _percentile(duplicate_ratio_pct)
    print(
        "Summary statistics",
        f"\n  Users: {unique_users}",
        f"\n  Total interactions: {total_rows:,}",
        f"\n  Deduplicated interactions: {dedup_rows:,}",
        f"\n  Duplicate surplus removed: {duplicate_loss:,}",
        f"\n  Users with duplicates: {nonzero_duplicates} ({duplicate_user_share:.2f}%):",
    )
    for q, value in zip(percentile_targets, percentile_values.to_numpy()):
        print(f"    p{int(q*100)} duplicate share: {value:.2f}%")
    print(
        f"  Mean duplicate share: {duplicate_ratio_pct.mean():.2f}%"
        f"\n  Median duplicate share: {duplicate_ratio_pct.median():.2f}%"
    )


def _maybe_plot(frame: pd.DataFrame, output: Path | None, show: bool) -> None:
    if output is None and not show:
        return
    if plt is None:
        raise RuntimeError("matplotlib is required to generate plots.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter, ax_hist = axes

    ax_scatter.scatter(
        frame["deduplicated_count"],
        frame["interaction_count"],
        s=8,
        alpha=0.35,
        edgecolors="none",
    )
    max_count = max(frame["interaction_count"].max(), frame["deduplicated_count"].max())
    ax_scatter.plot([0, max_count], [0, max_count], linestyle="--", color="red", linewidth=1.0, label="y = x")
    ax_scatter.set_xlabel("Deduplicated sequence length")
    ax_scatter.set_ylabel("Original sequence length")
    ax_scatter.set_title("Per-user sequence length before vs after dedup")
    ax_scatter.legend()
    ax_scatter.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    if FuncFormatter is not None:
        formatter = FuncFormatter(lambda val, _: f"{val:,.0f}")
        ax_scatter.xaxis.set_major_formatter(formatter)
        ax_scatter.yaxis.set_major_formatter(formatter)

    duplicate_ratio_pct = (frame["duplicate_ratio"] * 100.0).clip(lower=0)
    ax_hist.hist(duplicate_ratio_pct, bins=40, color="#1f77b4", edgecolor="white")
    ax_hist.set_xlabel("Duplicate share per user (%)")
    ax_hist.set_ylabel("User count")
    ax_hist.set_title("Distribution of duplicate share across users")
    ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactions-csv",
        type=Path,
        required=True,
        help="CSV with columns ['user_id','interaction_count'].",
    )
    parser.add_argument(
        "--duplicate-surplus-csv",
        type=Path,
        required=True,
        help="CSV with columns ['user_id','duplicate_surplus'].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot (PNG recommended).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively if matplotlib is available.",
    )
    args = parser.parse_args()

    frame = _load_frames(args.interactions_csv, args.duplicate_surplus_csv)
    _print_summary(frame)
    _maybe_plot(frame, args.output, args.show)


if __name__ == "__main__":
    main()

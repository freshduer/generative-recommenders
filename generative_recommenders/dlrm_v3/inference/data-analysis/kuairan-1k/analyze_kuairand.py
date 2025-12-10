#!/usr/bin/env python3
"""Utility to inspect KuaiRand-1K user/item statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception:  # pragma: no cover - plotting optional
    plt = None
    FuncFormatter = None


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _load_log_frames(
    data_dir: Path, show_progress: bool
) -> Tuple[pd.DataFrame, List[Path]]:
    """Load and concatenate all standard KuaiRand log CSV files."""
    log_paths = sorted(data_dir.glob("log_standard*.csv"))
    if not log_paths:
        msg = (
            "No KuaiRand log files found. Expected files like "
            "'log_standard_4_08_to_4_21_1k.csv'."
        )
        raise FileNotFoundError(msg)

    iterable = log_paths
    if show_progress and tqdm is not None:
        iterable = tqdm(log_paths, desc="Reading logs", unit="file")
    elif show_progress and tqdm is None:
        print("tqdm not installed; continuing without a file-level progress bar.")
        show_progress = False

    frames = []
    for csv_path in iterable:
        frame = pd.read_csv(csv_path)
        frame["__source_file"] = csv_path.name
        frames.append(frame)

    concatenated = pd.concat(frames, ignore_index=True)
    return concatenated, log_paths


def _summarise_series(
    series: pd.Series, percentiles: List[float]
) -> Dict[str, Optional[float]]:
    """Convert descriptive statistics to plain Python types."""
    stats = series.describe(percentiles=percentiles)
    summary: Dict[str, Optional[float]] = {}
    for key, value in stats.to_dict().items():
        summary[key] = None if pd.isna(value) else float(value)
    return summary


def analyse_kuairand(
    data_dir: Path,
    top_percent: float,
    top_k: int,
    show_progress: bool = False,
) -> Tuple[
    Dict[str, object],
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    float,
    List[int],
]:
    df, log_paths = _load_log_frames(data_dir, show_progress)

    if show_progress:
        print(f"Loaded {len(df):,} interactions from {len(log_paths)} log files.")

    required_cols = {"user_id", "video_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    total_rows = len(df)
    unique_users = df["user_id"].nunique()
    unique_items = df["video_id"].nunique()

    if show_progress:
        print("Computing per-user interaction counts...")
    user_interactions = df.groupby("user_id")["video_id"].size().sort_index()

    if show_progress:
        print("Computing per-user unique item counts...")
    per_user_unique_items = df.groupby("user_id")["video_id"].nunique().sort_index()

    if show_progress:
        print("Computing per-user duplicate surplus...")
    user_duplicate_surplus = (user_interactions - per_user_unique_items).sort_index()
    if show_progress:
        print("Computing per-item interaction counts...")
    item_popularity = (
        df.groupby("video_id")["user_id"].size().sort_values(ascending=False)
    )

    # Determine top-percent items and per-user engagement with them
    top_percent = max(0.0, min(100.0, top_percent))
    top_fraction = top_percent / 100.0
    if len(item_popularity) == 0 or top_fraction <= 0.0:
        top_item_ids: List[int] = []
        per_user_top_interactions = user_interactions.copy()
        per_user_top_interactions[:] = 0
        per_user_top_unique = per_user_unique_items.copy()
        per_user_top_unique[:] = 0
        top_share = 0.0
    else:
        top_count = max(1, int(round(top_fraction * len(item_popularity))))
        top_item_ids = item_popularity.head(top_count).index.tolist()
        top_mask = df["video_id"].isin(top_item_ids)
        per_user_top_interactions = (
            df[top_mask]
            .groupby("user_id")["video_id"]
            .size()
            .reindex(user_interactions.index, fill_value=0)
        )
        per_user_top_unique = (
            df[top_mask]
            .groupby("user_id")["video_id"]
            .nunique()
            .reindex(user_interactions.index, fill_value=0)
        )
        top_share = float(per_user_top_interactions.sum() / total_rows) if total_rows else 0.0

    per_user_top_ratio = per_user_top_interactions / user_interactions.replace(0, pd.NA)
    per_user_top_ratio = per_user_top_ratio.fillna(0.0)

    duplicate_rows = int(
        df.duplicated(subset=["user_id", "video_id", "time_ms"], keep=False).sum()
    ) if {"time_ms"}.issubset(df.columns) else int(
        df.duplicated(subset=["user_id", "video_id"], keep=False).sum()
    )

    top_items_records = [
        {"item_id": item_id, "interaction_count": int(count)}
        for item_id, count in item_popularity.head(top_k).items()
    ]

    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    summary = {
        "log_files": [path.name for path in log_paths],
        "total_interactions": int(total_rows),
        "unique_users": int(unique_users),
        "unique_items": int(unique_items),
        "users_with_multiple_interactions": int((user_interactions > 1).sum()),
        "repeated_interaction_surplus": int((user_interactions - 1).clip(lower=0).sum()),
        "duplicate_rows": duplicate_rows,
        "deduplicated_total_interactions": int(per_user_unique_items.sum()),
        "dedup_loss_total": int(user_duplicate_surplus.sum()),
        "top_percent": float(top_percent),
        "top_share": float(top_share),
        "user_interactions_summary": _summarise_series(user_interactions, percentiles),
        "per_user_unique_items_summary": _summarise_series(
            per_user_unique_items, percentiles
        ),
        "per_user_duplicate_surplus_summary": _summarise_series(
            user_duplicate_surplus, percentiles
        ),
        "item_popularity_summary": _summarise_series(item_popularity, percentiles),
        "per_user_top_item_interactions_summary": _summarise_series(
            per_user_top_interactions, percentiles
        ),
        "per_user_top_unique_items_summary": _summarise_series(
            per_user_top_unique, percentiles
        ),
        "per_user_top_item_ratio_summary": _summarise_series(
            per_user_top_ratio, percentiles
        ),
        "top_items": top_items_records,
    }
    return (
        summary,
        user_interactions,
        item_popularity,
        per_user_unique_items,
        user_duplicate_surplus,
        per_user_top_interactions,
        per_user_top_unique,
        per_user_top_ratio,
        top_share,
        top_item_ids,
    )


def write_csv(
    series: pd.Series, path: Optional[Path], column_names: Tuple[str, str]
) -> None:
    if path is None:
        return
    frame = series.sort_values(ascending=False).reset_index()
    frame.columns = list(column_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing KuaiRand log CSV files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write the high-level JSON summary.",
    )
    parser.add_argument(
        "--user-stats-csv",
        type=Path,
        help="Optional path to write per-user interaction counts.",
    )
    parser.add_argument(
        "--item-stats-csv",
        type=Path,
        help="Optional path to write per-item interaction counts.",
    )
    parser.add_argument(
        "--user-unique-items-csv",
        type=Path,
        help="Optional path to write per-user unique item counts.",
    )
    parser.add_argument(
        "--user-duplicate-surplus-csv",
        type=Path,
        help="Optional path to write per-user duplicate interaction surplus.",
    )
    parser.add_argument(
        "--user-top-item-csv",
        type=Path,
        help="Optional path to write per-user interactions with top-percent items.",
    )
    parser.add_argument(
        "--top-k-items",
        type=int,
        default=25,
        help="Number of most-interacted items to include in the printed summary.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10.0,
        help="Percentage of most popular items treated as 'top' for share stats.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress messages and use tqdm if available.",
    )
    parser.add_argument(
        "--duplicates-csv",
        type=Path,
        help=(
            "Optional path to write all row-level duplicates."
            " Duplicates are detected on ['user_id','video_id','time_ms']"
            " when 'time_ms' exists, otherwise on ['user_id','video_id']."
        ),
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        help=(
            "Optional JSON file capturing per-user stats, duplicate surplus, and top-item engagement."
        ),
    )
    parser.add_argument(
        "--user-dedup-plot",
        type=Path,
        help=(
            "Optional PNG to plot original vs deduplicated user sequence lengths."
            " Requires matplotlib."
        ),
    )

    args = parser.parse_args()

    default_data_dir = (
        Path(__file__).resolve().parents[1] / "data" / "KuaiRand-1K" / "data"
    )
    data_dir = (args.data_dir or default_data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist."
            " Please pass --data-dir to the folder containing the KuaiRand CSV files."
        )

    (
        summary,
        user_interactions,
        item_popularity,
        per_user_unique_items,
        user_duplicate_surplus,
        per_user_top_interactions,
        per_user_top_unique_items,
        per_user_top_ratio,
        top_share,
        top_item_ids,
    ) = analyse_kuairand(
        data_dir,
        args.top_percent,
        args.top_k_items,
        show_progress=args.show_progress,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"\nTop {args.top_percent:.1f}% items (n={len(top_item_ids)}) share of visits:"
        f" {top_share*100:.2f}%"
    )
    print(
        "Average repeated interactions per user (sequence length - unique items):"
        f" {user_duplicate_surplus.mean():.2f}"
    )
    print(
        f"Median share of top {args.top_percent:.1f}% items per user:"
        f" {per_user_top_ratio.median()*100:.2f}%"
    )

    if args.top_k_items > 0:
        print("\nTop items by interaction count:")
        top_items = item_popularity.head(args.top_k_items)
        for item_id, count in top_items.items():
            print(f"  {item_id}: {count}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    write_csv(user_interactions, args.user_stats_csv, ("user_id", "interaction_count"))
    write_csv(item_popularity, args.item_stats_csv, ("video_id", "interaction_count"))
    write_csv(
        per_user_unique_items,
        args.user_unique_items_csv,
        ("user_id", "unique_item_count"),
    )
    write_csv(
        user_duplicate_surplus,
        args.user_duplicate_surplus_csv,
        ("user_id", "duplicate_surplus"),
    )
    if args.user_top_item_csv is not None:
        if args.user_top_item_csv.suffix.lower() == ".csv":
            frame = pd.DataFrame(
                {
                    "user_id": user_interactions.index,
                    "top_item_interactions": per_user_top_interactions.astype(int),
                    "top_unique_items": per_user_top_unique_items.astype(int),
                    "top_item_ratio": per_user_top_ratio,
                }
            )
            frame.sort_values("top_item_interactions", ascending=False, inplace=True)
            args.user_top_item_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.user_top_item_csv, index=False)
        else:
            args.user_top_item_csv.parent.mkdir(parents=True, exist_ok=True)
            args.user_top_item_csv.write_text(
                json.dumps(
                    {
                        "per_user_top_item_interactions": per_user_top_interactions.to_dict(),
                        "per_user_top_unique_items": per_user_top_unique_items.to_dict(),
                        "per_user_top_item_ratio": per_user_top_ratio.to_dict(),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

    if args.user_dedup_plot is not None and plt is not None:
        args.user_dedup_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            per_user_unique_items,
            user_interactions,
            s=10,
            alpha=0.4,
            edgecolors="none",
        )
        max_val = max(user_interactions.max(), per_user_unique_items.max())
        ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.2, label="y = x")
        ax.set_xlabel("Deduplicated sequence length (unique items)")
        ax.set_ylabel("Original sequence length")
        ax.set_title("User sequence length before vs after deduplication (KuaiRand-1K)")
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        if FuncFormatter is not None:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        fig.tight_layout()
        fig.savefig(args.user_dedup_plot, dpi=200)
        plt.close(fig)

    if args.stats_json is not None:
        stats_payload = {
            "user_interactions": {
                str(uid): int(cnt) for uid, cnt in user_interactions.items()
            },
            "per_user_unique_items": {
                str(uid): int(cnt) for uid, cnt in per_user_unique_items.items()
            },
            "user_duplicate_surplus": {
                str(uid): int(cnt) for uid, cnt in user_duplicate_surplus.items()
            },
            "per_user_top_item_interactions": {
                str(uid): int(cnt) for uid, cnt in per_user_top_interactions.items()
            },
            "per_user_top_unique_items": {
                str(uid): int(cnt) for uid, cnt in per_user_top_unique_items.items()
            },
            "per_user_top_item_ratio": {
                str(uid): float(cnt) for uid, cnt in per_user_top_ratio.items()
            },
            "top_percent": float(args.top_percent),
            "top_share": float(top_share),
            "top_item_ids": [int(item) for item in top_item_ids],
            "total_interactions": int(summary["total_interactions"]),
            "deduplicated_total_interactions": int(summary["deduplicated_total_interactions"]),
            "dedup_loss_total": int(summary["dedup_loss_total"]),
        }
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False))

    # Export duplicates if requested: recompute duplicate mask using the same rules
    if args.duplicates_csv is not None:
        # Reload raw logs to construct duplicates table
        df_raw, _ = _load_log_frames(data_dir, show_progress=False)
        subset_cols = ["user_id", "video_id"]
        if "time_ms" in df_raw.columns:
            subset_cols.append("time_ms")
        dup_mask = df_raw.duplicated(subset=subset_cols, keep=False)
        dups = df_raw.loc[dup_mask].sort_values(subset_cols)
        args.duplicates_csv.parent.mkdir(parents=True, exist_ok=True)
        dups.to_csv(args.duplicates_csv, index=False)


if __name__ == "__main__":
    main()

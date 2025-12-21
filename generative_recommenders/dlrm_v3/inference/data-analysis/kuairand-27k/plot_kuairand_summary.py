#!/usr/bin/env python3
"""Plot summary figures from KuaiRand JSON report.

Reads a summary JSON (as produced by `analyze_kuairand.py`) and creates:
- Histogram of per-user interaction counts
- Histogram of per-user unique item counts
- Histogram of item popularity counts (log-scaled x-axis optional)
- Bar chart of top-K items by interaction count

Outputs PNGs into the chosen output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_hist(
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    bins: int,
    out_path: Path,
    use_log_x: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    if use_log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _plot_cdf(
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    use_log_x: bool = False,
) -> None:
    if not values:
        return
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    if use_log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(out_path)
    plt.close()

def _plot_top_items(top_items: List[Dict[str, Any]], out_path: Path) -> None:
    if not top_items:
        return
    labels = [str(item["item_id"]) for item in top_items]
    counts = [int(item["interaction_count"]) for item in top_items]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Top Items by Interaction Count")
    plt.xlabel("item_id")
    plt.ylabel("interaction_count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_item_popularity_cdf(
    counts: List[int], out_path: Path, annotate_lines: List[float]
) -> None:
    if not counts:
        return
    # Empirical CDF: sort counts and compute fraction of items <= x
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    y = [(i + 1) / n for i in range(n)]
    plt.figure(figsize=(8, 5))
    plt.plot(counts_sorted, y, drawstyle="steps-post")
    plt.title("Item Popularity CDF")
    plt.xlabel("interaction_count_per_item")
    plt.ylabel("fraction of items (<= x)")
    # Horizontal dashed lines (e.g., 0.9 and 0.99)
    for frac in annotate_lines:
        plt.axhline(frac, color="red", linestyle="--", linewidth=1)
        plt.text(
            counts_sorted[-1] * 0.02,
            frac + 0.005,
            f"{int(frac*100)}% items",
            color="red",
        )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _compute_top_item_share(counts: List[int], top_percent: float) -> float:
    """Return the share of total interactions covered by top X% items."""
    if not counts:
        return 0.0
    counts_sorted_desc = sorted(counts, reverse=True)
    n = len(counts_sorted_desc)
    k = max(1, int(round(n * top_percent)))
    top_sum = sum(counts_sorted_desc[:k])
    total = sum(counts_sorted_desc)
    return top_sum / total if total > 0 else 0.0


def _plot_item_share_curve(
    counts: List[int], out_path: Path, highlight_percents: List[float]
) -> None:
    """Plot cumulative interaction share vs item percentile (Pareto curve).

    - x-axis: fraction of items (sorted by count desc)
    - y-axis: fraction of interactions covered by top-x% items
    Adds vertical dashed lines at requested percents and labels the share.
    """
    if not counts:
        return
    counts_desc = sorted(counts, reverse=True)
    total = float(sum(counts_desc))
    n = len(counts_desc)
    cum = []
    running = 0.0
    for c in counts_desc:
        running += float(c)
        cum.append(running / total if total > 0 else 0.0)
    x = [(i + 1) / n for i in range(n)]

    plt.figure(figsize=(8, 5))
    plt.plot(x, cum)
    plt.title("Cumulative Interaction Share vs Item Percentile")
    plt.xlabel("fraction of items (top-x%)")
    plt.ylabel("fraction of interactions covered")

    for p in highlight_percents:
        k = max(1, int(round(n * p)))
        share = cum[k - 1]
        plt.axvline(p, color="red", linestyle="--", linewidth=1)
        plt.axhline(share, color="red", linestyle="--", linewidth=1)
        plt.text(
            min(p + 0.02, 0.98),
            min(share + 0.02, 0.98),
            f"Top {int(p*100)}% → {share*100:.1f}%",
            color="red",
        )

    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(out_path)
    plt.close()


def load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    return data


def extract_distribution_values(summary: Dict[str, Any], key: str) -> List[float]:
    # The summary JSON includes descriptive stats only; for plotting histograms
    # we approximate by sampling from percentiles when raw series aren't present.
    # However, since this JSON does not contain the raw series, we'll use a
    # synthetic expansion based on percentiles to visualize shape coarsely.
    # If raw CSVs are available, prefer plotting from series directly.
    stats = summary.get(key, {})
    count = int(stats.get("count", 0) or 0)
    if count == 0:
        return []

    # Percentiles present: 50%, 75%, 90%, 95%, 99%, plus min/max/mean.
    # We'll construct a coarse distribution using buckets.
    # Buckets: [min, 50%], (50%, 75%], (75%, 90%], (90%, 95%], (95%, 99%], (99%, max]
    # Assign approximate counts to each bucket proportionally.
    def val(name: str) -> float:
        v = stats.get(name)
        return float(v) if v is not None else 0.0

    v_min = val("min")
    v_p50 = val("50%")
    v_p75 = val("75%")
    v_p90 = val("90%")
    v_p95 = val("95%")
    v_p99 = val("99%")
    v_max = val("max")

    # Proportions per bucket
    props = [0.50, 0.25, 0.15, 0.05, 0.04, 0.01]
    bucket_bounds = [
        (v_min, v_p50),
        (v_p50, v_p75),
        (v_p75, v_p90),
        (v_p90, v_p95),
        (v_p95, v_p99),
        (v_p99, v_max if v_max >= v_p99 else v_p99),
    ]

    values: List[float] = []
    for prop, (lo, hi) in zip(props, bucket_bounds):
        n = max(1, int(round(count * prop)))
        if hi < lo:
            hi = lo
        if lo == hi:
            values.extend([lo] * n)
        else:
            # linearly spaced samples within the bucket
            step = (hi - lo) / max(1, n - 1)
            values.extend([lo + i * step for i in range(n)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            __file__
        ).resolve().parent / "reports" / "kuairand_summary.json",
        help="Path to the summary JSON produced by analyze_kuairand.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory to save output figures (PNGs).",
    )
    parser.add_argument(
        "--top-items-bar",
        action="store_true",
        help="If set, generate bar chart for top items.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--log-x-item-pop",
        action="store_true",
        help="Use logarithmic x-scale for item popularity histogram.",
    )
    parser.add_argument(
        "--item-stats-csv",
        type=Path,
        help="Optional path to per-item interaction counts CSV for exact CDF and share.",
    )
    parser.add_argument(
        "--cdf-item-pop",
        action="store_true",
        help="If set, also plot CDF for item popularity.",
    )

    args = parser.parse_args()

    _ensure_outdir(args.out_dir)
    summary = load_summary(args.summary_json)

    # Per-user interaction counts
    user_interactions_values = extract_distribution_values(
        summary, "user_interactions_summary"
    )
    _plot_hist(
        user_interactions_values,
        title="Per-User Interaction Counts (Approx)",
        xlabel="interaction_count",
        ylabel="users",
        bins=args.bins,
        out_path=args.out_dir / "hist_user_interactions.png",
        use_log_x=False,
    )

    # Per-user unique items
    user_unique_values = extract_distribution_values(
        summary, "per_user_unique_items_summary"
    )
    _plot_hist(
        user_unique_values,
        title="Per-User Unique Items (Approx)",
        xlabel="unique_item_count",
        ylabel="users",
        bins=args.bins,
        out_path=args.out_dir / "hist_user_unique_items.png",
        use_log_x=False,
    )

    # Item popularity histogram
    item_pop_values = extract_distribution_values(
        summary, "item_popularity_summary"
    )
    _plot_hist(
        item_pop_values,
        title="Item Popularity (Approx)",
        xlabel="interaction_count_per_item",
        ylabel="items",
        bins=args.bins,
        out_path=args.out_dir / "hist_item_popularity.png",
        use_log_x=args.log_x_item_pop,
    )

    # Item popularity CDF and top-share stats
    counts_for_cdf: List[int] = []
    if args.item_stats_csv and args.item_stats_csv.exists():
        df_items = pd.read_csv(args.item_stats_csv)
        # Expect columns: [video_id, interaction_count]
        counts_for_cdf = df_items["interaction_count"].astype(int).tolist()
    else:
        # Fall back to approximate values (floats), cast to int
        counts_for_cdf = [int(round(v)) for v in item_pop_values]

    # Compute and print top 1% and 10% item shares of interactions
    share_top_1 = _compute_top_item_share(counts_for_cdf, 0.01)
    share_top_10 = _compute_top_item_share(counts_for_cdf, 0.10)
    print(
        f"Top 1% items cover {share_top_1*100:.2f}% of interactions; "
        f"Top 10% items cover {share_top_10*100:.2f}% of interactions."
    )
    # Plot CDF for item popularity when requested
    if args.cdf_item_pop:
        _plot_item_popularity_cdf(
            counts_for_cdf,
            args.out_dir / "cdf_item_popularity.png",
            annotate_lines=[0.9, 0.99],
        )

    # Always generate the Pareto curve showing top-1%/10% coverage visually
    _plot_item_share_curve(
        counts_for_cdf,
        args.out_dir / "pareto_item_share.png",
        highlight_percents=[0.01, 0.10],
    )

    # Top items bar chart
    if args.top_items_bar:
        top_items = summary.get("top_items", [])
        _plot_top_items(top_items, args.out_dir / "bar_top_items.png")

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()

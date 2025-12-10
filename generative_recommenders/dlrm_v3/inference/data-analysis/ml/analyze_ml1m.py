#!/usr/bin/env python3
"""Analysis for ML-1M interactions: user sequence lengths, item popularity CDF,
and share of visits captured by top 10% most popular items.

This script expects SASRec-formatted CSVs produced by preprocessing, e.g.:
- `sasrec_format.csv` (two columns: `user_id`, `item_id`)
- or by-user files like `sasrec_format_by_user_train.csv` / `..._test.csv`

You can point `--data-dir` to a folder containing any of the above; the script
will load all matching CSVs and concatenate them.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception as e:  # pragma: no cover - plotting optional
    plt = None
    FuncFormatter = None

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
def _summarise_series(series: pd.Series, percentiles: List[float]) -> dict:
    stats = series.describe(percentiles=percentiles)
    summary = {}
    for key, value in stats.to_dict().items():
        summary[key] = None if pd.isna(value) else float(value)
    return summary



def _find_csvs(data_dir: Path) -> List[Path]:
    candidates = []
    # Common SASRec outputs
    for pattern in [
        "sasrec_format.csv",
        "sasrec_format_by_user_train.csv",
        "sasrec_format_by_user_test.csv",
    ]:
        candidates.extend(sorted(data_dir.glob(pattern)))
    # Fallback: any CSV inside the dir
    if not candidates:
        candidates = sorted(data_dir.glob("*.csv"))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _load_frames(paths: List[Path], show_progress: bool = False) -> pd.DataFrame:
    frames = []
    iterable = paths
    if show_progress and tqdm is not None:
        iterable = tqdm(paths, desc="Reading CSVs", unit="file")
    elif show_progress and tqdm is None:
        print("tqdm not installed; continuing without progress bar.")
        show_progress = False
    for p in iterable:
        # Prefer pyarrow engine for speed if available
        try:
            df = pd.read_csv(p, engine="pyarrow")
        except Exception:
            df = pd.read_csv(p)
        # Detect by-user aggregated format with sequence columns
        lower_cols = {c.lower(): c for c in df.columns}
        seq_col = None
        user_col = None
        # Common by-user sequence column names
        for c in df.columns:
            lc = c.lower()
            if lc in {"user", "user_id", "uid"}:
                user_col = c
            if lc in {"sequence_item_ids", "items", "item_ids"}:
                seq_col = c
        if seq_col is not None and user_col is not None:
            # Expand sequences into user-item interactions
            tmp = df[[user_col, seq_col]].rename(columns={user_col: "user_id", seq_col: "sequence_item_ids"}).copy()
            # Vectorized split on comma or whitespace, then explode
            # Handles NaN safely; result dtype is object
            split_series = tmp["sequence_item_ids"].astype(str).str.strip().replace({"nan": ""}).str.split(r"[,\s]+", regex=True)
            tmp = tmp.drop(columns=["sequence_item_ids"])  # free memory
            tmp["item_id"] = split_series
            if show_progress and tqdm is not None:
                exploded = []
                # manual explode with progress: iterate rows with lengths
                for _, row in tqdm(tmp.iterrows(), total=len(tmp), desc="Exploding sequences", unit="user"):
                    uid = row["user_id"]
                    items = row["item_id"] or []
                    for it in items:
                        exploded.append((uid, it))
                exploded = pd.DataFrame(exploded, columns=["user_id", "item_id"])
            else:
                exploded = tmp.explode("item_id")
            exploded = exploded.replace({"item_id": {"": pd.NA}}).dropna(subset=["item_id"])  # remove empties
            # Cast item_id to int if possible
            try:
                exploded["item_id"] = exploded["item_id"].astype("int64")
            except Exception:
                pass
            exploded["__source_file"] = p.name
            frames.append(exploded[["user_id", "item_id", "__source_file"]])
            continue

        # Otherwise expect flat interaction CSV with user/item columns
        item_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in {"user", "user_id", "uid"}:
                user_col = c
            if lc in {"item", "item_id", "iid", "video_id", "movie_id"}:
                item_col = c
        if user_col is None or item_col is None:
            raise ValueError(
                f"File {p} must contain user and item identifier columns, or a 'sequence_item_ids' column."
            )
        # Only keep needed columns to reduce memory
        df = df[[user_col, item_col]].rename(columns={user_col: "user_id", item_col: "item_id"})
        # Enforce integer dtypes when possible
        for col in ["user_id", "item_id"]:
            try:
                df[col] = df[col].astype("int64")
            except Exception:
                pass
        df["__source_file"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No usable CSV files found in data directory.")
    return pd.concat(frames, ignore_index=True)


def compute_stats(df: pd.DataFrame, top_pct: float = 0.10) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    float,
    List[int],
    pd.Series,
]:
    # Sequence length per user (number of interactions)
    user_seq_len = df.groupby("user_id")["item_id"].size().sort_index()

    # Unique item count per user (deduplicated sequence length)
    per_user_unique_items = df.groupby("user_id")["item_id"].nunique().sort_index()

    # Surplus interactions due to repeats within a user's sequence
    user_duplicate_surplus = (user_seq_len - per_user_unique_items).sort_index()

    # Item popularity: total interactions per item (sorted high → low)
    item_pop = df.groupby("item_id")["user_id"].size().sort_values(ascending=False)

    total_interactions = int(len(df))
    if total_interactions == 0:
        top_share = 0.0
        top_item_ids: List[int] = []
        per_user_top_interactions = user_seq_len.copy()
        per_user_top_interactions[:] = 0
        per_user_top_unique_items = per_user_unique_items.copy()
        per_user_top_unique_items[:] = 0
    else:
        top_pct = max(0.0, min(1.0, top_pct))
        if top_pct <= 0.0 or len(item_pop) == 0:
            top_item_ids = []
            per_user_top_interactions = user_seq_len.copy()
            per_user_top_interactions[:] = 0
            per_user_top_unique_items = per_user_unique_items.copy()
            per_user_top_unique_items[:] = 0
            top_share = 0.0
        else:
            k = max(1, int(round(top_pct * len(item_pop))))
            top_item_ids = item_pop.head(k).index.tolist()
            top_mask = df["item_id"].isin(top_item_ids)
            per_user_top_interactions = (
                df[top_mask]
                .groupby("user_id")["item_id"]
                .size()
                .reindex(user_seq_len.index, fill_value=0)
            )
            per_user_top_unique_items = (
                df[top_mask]
                .groupby("user_id")["item_id"]
                .nunique()
                .reindex(user_seq_len.index, fill_value=0)
            )
            top_share = float(per_user_top_interactions.sum() / total_interactions)

    return (
        user_seq_len,
        per_user_unique_items,
        user_duplicate_surplus,
        per_user_top_interactions,
        per_user_top_unique_items,
        top_share,
        top_item_ids,
        item_pop,
    )


def plot_user_seq_len(user_seq_len: pd.Series, out_path: Path | None) -> None:
    if plt is None or out_path is None:
        return
    plt.figure(figsize=(8, 5))
    # Use log-scaled bins for heavy-tailed sequences
    user_seq_len.hist(bins=50)
    plt.xlabel("User interaction sequence length")
    plt.ylabel("Count of users")
    plt.title("Distribution of user sequence lengths (ML-1M)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_item_pop_cdf(item_pop: pd.Series, out_path: Path | None, top_pct: float = 0.10) -> None:
    if plt is None or out_path is None:
        return
    plt.figure(figsize=(8, 5))
    # Sort ascending for a conventional CDF curve
    values = item_pop.sort_values(ascending=True).to_numpy()
    total = values.sum()
    if total == 0:
        total = 1  # avoid div by zero; empty plot
    cdf = values.cumsum() / total
    n = len(values)
    # X-axis as percentage of items (0-100%)
    x_pct = (pd.Series(range(1, n + 1)) / n * 100.0).to_numpy()
    plt.plot(x_pct, cdf, label="CDF of visits vs item %")

    # Annotate top (100*top_pct)% most popular items share of visits.
    # Since our curve increases from least→most popular, the share captured by top K% items
    # equals 1 - CDF at the (100 - K)% percentile.
    k_pct = top_pct * 100.0
    cutoff_pct = 100.0 - k_pct
    # Find index closest to cutoff percentile
    idx = max(0, min(n - 1, int(round(cutoff_pct / 100.0 * n)) - 1))
    cdf_at_cutoff = float(cdf[idx]) if n > 0 else 0.0
    share_top_k = 1.0 - cdf_at_cutoff

    # Draw guide lines
    plt.axvline(cutoff_pct, color="orange", linestyle="--", linewidth=1, label=f"{cutoff_pct:.0f}% items cutoff")
    plt.axhline(1.0 - share_top_k, color="green", linestyle="--", linewidth=1)

    # Text annotation
    plt.text(
        cutoff_pct,
        1.0 - share_top_k,
        f"Top {k_pct:.1f}% items share: {share_top_k*100:.2f}%",
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray", alpha=0.6),
    )

    plt.xlabel("Item percentage (sorted by popularity)")
    plt.ylabel("CDF of visits (%)")
    if FuncFormatter is not None:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    plt.title("Item popularity CDF (ML-1M)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with ML-1M SASRec CSVs.")
    parser.add_argument("--user-seqlen-plot", type=Path, help="Output PNG for user sequence length distribution.")
    parser.add_argument("--item-cdf-plot", type=Path, help="Output PNG for item popularity CDF.")
    parser.add_argument("--show-progress", action="store_true", help="Show progress bars during reading and exploding.")
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10.0,
        help="Percentage of most popular items to treat as 'top' for stats (default: 10).",
    )
    parser.add_argument(
        "--top-k-items",
        type=int,
        default=20,
        help="Number of top items to include in summary output list.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        help=(
            "Output JSON path storing user sequence lengths and item popularity. "
            "Format: { 'user_seq_len': [{user_id, seq_len}], 'item_popularity': [{item_id, interaction_count}] }"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help=(
            "Output JSON in the standard summary format (like KuaiRand). "
            "Includes totals, summaries, and top items."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    csvs = _find_csvs(data_dir)
    if not csvs:
        raise FileNotFoundError("No CSVs found. Expected SASRec-formatted files.")
    df = _load_frames(csvs, show_progress=args.show_progress)

    top_percent = max(0.0, min(100.0, args.top_percent))
    top_fraction = top_percent / 100.0

    (
        user_seq_len,
        per_user_unique_items,
        user_duplicate_surplus,
        per_user_top_interactions,
        per_user_top_unique_items,
        top_share,
        top_item_ids,
        item_pop,
    ) = compute_stats(df, top_pct=top_fraction)

    print(f"Total interactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    print(
        f"Top {top_percent:.1f}% items share of visits: {top_share:.4f}"
        f" ({top_share*100:.2f}%)"
    )
    avg_duplicate_surplus = user_duplicate_surplus.mean()
    print(
        f"Average repeated interactions per user (sequence length minus unique items):"
        f" {avg_duplicate_surplus:.2f}"
    )

    # Ratio of interactions spent on top items per user
    per_user_top_ratio = per_user_top_interactions / user_seq_len.replace(0, pd.NA)
    per_user_top_ratio = per_user_top_ratio.fillna(0.0)
    print(
        f"Median share of top {top_percent:.1f}% items per user:"
        f" {per_user_top_ratio.median()*100:.2f}%"
    )

    plot_user_seq_len(user_seq_len, args.user_seqlen_plot)
    plot_item_pop_cdf(item_pop, args.item_cdf_plot, top_pct=top_fraction)

    # Optional JSON export of raw stats
    if args.stats_json is not None:
        stats = {
            "user_seq_len": [
                {"user_id": int(uid), "seq_len": int(cnt)} for uid, cnt in user_seq_len.sort_values(ascending=False).items()
            ],
            "item_popularity": [
                {"item_id": int(iid), "interaction_count": int(cnt)} for iid, cnt in item_pop.sort_values(ascending=False).items()
            ],
            "per_user_unique_items": [
                {"user_id": int(uid), "unique_item_count": int(cnt)}
                for uid, cnt in per_user_unique_items.sort_values(ascending=False).items()
            ],
            "user_duplicate_surplus": [
                {"user_id": int(uid), "repeat_interactions": int(cnt)}
                for uid, cnt in user_duplicate_surplus.sort_values(ascending=False).items()
            ],
            "per_user_top_item_interactions": [
                {"user_id": int(uid), "top_item_interactions": int(cnt)}
                for uid, cnt in per_user_top_interactions.sort_values(ascending=False).items()
            ],
            "per_user_top_unique_items": [
                {"user_id": int(uid), "top_unique_items": int(cnt)}
                for uid, cnt in per_user_top_unique_items.sort_values(ascending=False).items()
            ],
            "per_user_top_item_ratio": [
                {"user_id": int(uid), "top_item_ratio": float(cnt)}
                for uid, cnt in per_user_top_ratio.sort_values(ascending=False).items()
            ],
            "top_share": float(top_share),
            "top_percent": float(top_percent),
            "top_item_ids": [int(item) for item in top_item_ids],
            "total_interactions": int(len(df)),
            "unique_users": int(df["user_id"].nunique()),
            "unique_items": int(df["item_id"].nunique()),
            "deduplicated_total_interactions": int(per_user_unique_items.sum()),
            "dedup_loss_total": int(user_duplicate_surplus.sum()),
        }
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        import json
        args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # Standard summary JSON (KuaiRand-like)
    if args.summary_json is not None:
        percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        top_k = max(1, args.top_k_items)
        top_items_records = [
            {"item_id": int(item_id), "interaction_count": int(count)}
            for item_id, count in item_pop.head(top_k).items()
        ]
        per_user_top_ratio_summary = _summarise_series(per_user_top_ratio, percentiles)
        summary = {
            "log_files": [p.name for p in csvs],
            "total_interactions": int(len(df)),
            "unique_users": int(df["user_id"].nunique()),
            "unique_items": int(df["item_id"].nunique()),
            "users_with_multiple_interactions": int((user_seq_len > 1).sum()),
            "repeated_interaction_surplus": int((user_seq_len - 1).clip(lower=0).sum()),
            "duplicate_rows": 0,
            "top_percent": float(top_percent),
            "top_share": float(top_share),
            "deduplicated_total_interactions": int(per_user_unique_items.sum()),
            "dedup_loss_total": int(user_duplicate_surplus.sum()),
            "user_interactions_summary": _summarise_series(user_seq_len, percentiles),
            "per_user_unique_items_summary": _summarise_series(per_user_unique_items, percentiles),
            "per_user_duplicate_surplus_summary": _summarise_series(user_duplicate_surplus, percentiles),
            "item_popularity_summary": _summarise_series(item_pop, percentiles),
            "per_user_top_item_interactions_summary": _summarise_series(
                per_user_top_interactions, percentiles
            ),
            "per_user_top_unique_items_summary": _summarise_series(
                per_user_top_unique_items, percentiles
            ),
            "per_user_top_item_ratio_summary": per_user_top_ratio_summary,
            "top_items": top_items_records,
        }
        # Ensure integer-like floats are rendered as ints where appropriate when later consumed
        import json
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

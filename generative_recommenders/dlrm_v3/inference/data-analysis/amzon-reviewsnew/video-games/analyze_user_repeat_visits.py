#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze user repeat visits in Amazon Reviews dataset
- Statistics on whether users repeat visits to the same items
- Frequency distribution of repeat visits per user
- Hot user analysis (top 10% users vs total visits)
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None
    FuncFormatter = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        return iterable

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
if HAS_MATPLOTLIB:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['axes.axisbelow'] = True


def parse_amazon_review_line(line: str) -> Optional[Dict]:
    """
    Parse a single line from Amazon Reviews JSONL file
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        record = json.loads(line)
        
        # Extract user_id
        user_id = record.get('user_id') or record.get('reviewerID')
        if not user_id:
            return None
        
        # Extract timestamp
        timestamp = None
        
        if 'timestamp' in record:
            ts_val = record['timestamp']
            if isinstance(ts_val, (int, float)):
                if ts_val > 1e12:  # Milliseconds
                    timestamp = int(ts_val) // 1000
                else:  # Seconds
                    timestamp = int(ts_val)
        elif 'unixReviewTime' in record:
            timestamp = int(record['unixReviewTime'])
        
        if timestamp is None:
            return None
        
        return {
            'user_id': user_id,
            'asin': record.get('asin', ''),
            'timestamp': timestamp,
            'datetime': pd.Timestamp.fromtimestamp(timestamp)
        }
    except (json.JSONDecodeError, Exception):
        return None


def load_chunk(lines: List[str]) -> List[Dict]:
    """Process a chunk of lines - used for multiprocessing"""
    records = []
    for line in lines:
        record = parse_amazon_review_line(line)
        if record:
            records.append(record)
    return records


def load_chunk_wrapper(args: Tuple[int, List[str]]) -> Tuple[int, List[Dict]]:
    """Wrapper function for multiprocessing"""
    chunk_idx, lines = args
    records = load_chunk(lines)
    return (chunk_idx, records)


def load_amazon_reviews_data(
    data_path: str,
    show_progress: bool = False,
    use_multiprocessing: bool = True,
    chunk_size: int = 100000,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Load Amazon Reviews dataset efficiently using chunking and optional multiprocessing
    """
    logger.info(f"Loading Amazon Reviews data from: {data_path}")
    start_time = time.time()
    
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Check if cached parquet file exists
    cache_dir = data_path_obj.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{data_path_obj.stem}_processed.parquet"
    
    if cache_file.exists():
        logger.info(f"Found cached parquet file: {cache_file}")
        logger.info("Loading from cache...")
        try:
            df = pd.read_parquet(cache_file)
            elapsed = time.time() - start_time
            logger.info(f"Data loaded from cache: {len(df):,} rows, elapsed {elapsed:.2f} seconds")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will reload from source")
    
    # Count total lines for progress bar
    total_lines = 0
    if HAS_TQDM and show_progress:
        logger.info("Counting total lines...")
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"Total lines: {total_lines:,}")
    
    all_records = []
    
    if use_multiprocessing and n_workers != 1:
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        logger.info(f"Using multiprocessing with {n_workers} workers...")
        
        chunks = []
        current_chunk = []
        chunk_idx = 0
        
        logger.info("Reading file and preparing chunks...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_chunk.append(line)
                if len(current_chunk) >= chunk_size:
                    chunks.append((chunk_idx, current_chunk))
                    current_chunk = []
                    chunk_idx += 1
        
        if current_chunk:
            chunks.append((chunk_idx, current_chunk))
        
        logger.info(f"Prepared {len(chunks)} chunks for parallel processing")
        
        if HAS_TQDM and show_progress:
            from tqdm import tqdm as tqdm_module
            with mp.Pool(processes=n_workers) as pool:
                results = []
                with tqdm_module(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
                    for result in pool.imap(load_chunk_wrapper, chunks):
                        results.append(result)
                        pbar.update(1)
        else:
            with mp.Pool(processes=n_workers) as pool:
                results = list(pool.imap(load_chunk_wrapper, chunks))
        
        results.sort(key=lambda x: x[0])
        for _, records in results:
            all_records.extend(records)
        
        logger.info(f"Multiprocessing complete: {len(all_records):,} valid records")
    else:
        logger.info("Using single-threaded processing...")
        with open(data_path, 'r', encoding='utf-8') as f:
            if HAS_TQDM and show_progress:
                f = tqdm(f, desc="Reading data", unit="lines", total=total_lines)
            
            for line in f:
                record = parse_amazon_review_line(line)
                if record:
                    all_records.append(record)
    
    if not all_records:
        raise ValueError("No valid records found in the data file")
    
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(all_records)
    
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    invalid_count = df['datetime'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid timestamp records, removing them")
        df = df.dropna(subset=['datetime'])
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save to cache
    if HAS_PYARROW:
        try:
            logger.info(f"Saving to cache: {cache_file}")
            df.to_parquet(cache_file, compression='snappy', index=False)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading complete: {len(df):,} rows")
    logger.info(f"Unique users: {df['user_id'].nunique():,}")
    logger.info(f"Unique products: {df['asin'].nunique():,}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    
    return df


def analyze_repeat_visits(df: pd.DataFrame, show_progress: bool = False) -> Dict:
    """
    Analyze repeat visits: whether users visit the same items multiple times
    """
    logger.info("Analyzing repeat visits...")
    start_time = time.time()
    
    # Count interactions per user-item pair
    if show_progress:
        logger.info("Counting interactions per user-item pair...")
    
    user_item_counts = df.groupby(['user_id', 'asin'], observed=True, sort=False).size().reset_index(name='visit_count')
    
    total_pairs = len(user_item_counts)
    
    # Identify repeat visits (visit_count > 1)
    repeat_pairs = user_item_counts[user_item_counts['visit_count'] > 1]
    num_repeat_pairs = len(repeat_pairs)
    
    # Users with repeat visits
    users_with_repeat = repeat_pairs['user_id'].nunique()
    total_users = df['user_id'].nunique()
    
    # Statistics
    stats = {
        'total_user_item_pairs': int(total_pairs),
        'repeat_visit_pairs': int(num_repeat_pairs),
        'repeat_visit_ratio': float(num_repeat_pairs / total_pairs) if total_pairs > 0 else 0.0,
        'total_users': int(total_users),
        'users_with_repeat_visits': int(users_with_repeat),
        'users_with_repeat_ratio': float(users_with_repeat / total_users) if total_users > 0 else 0.0,
        'avg_visits_per_pair': float(user_item_counts['visit_count'].mean()),
        'max_visits_per_pair': int(user_item_counts['visit_count'].max()),
        'total_interactions': int(len(df))
    }
    
    if num_repeat_pairs > 0:
        stats['avg_repeat_visits'] = float(repeat_pairs['visit_count'].mean())
    else:
        stats['avg_repeat_visits'] = 0.0
    
    elapsed = time.time() - start_time
    logger.info(f"Repeat visit analysis complete, elapsed {elapsed:.2f} seconds")
    
    return stats, user_item_counts


def analyze_user_repeat_frequency(df: pd.DataFrame, user_item_counts: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Analyze repeat visit frequency per user
    """
    logger.info("Analyzing user repeat visit frequency...")
    start_time = time.time()
    
    # For each user, count how many items they visited multiple times
    if show_progress:
        logger.info("Computing repeat visit frequency per user...")
    
    # Merge with user_item_counts to get visit counts
    repeat_pairs = user_item_counts[user_item_counts['visit_count'] > 1]
    
    # Count repeat items per user
    user_repeat_counts = repeat_pairs.groupby('user_id', observed=True, sort=False).size().reset_index(name='num_repeat_items')
    
    # Also count total items per user
    user_total_items = user_item_counts.groupby('user_id', observed=True, sort=False).size().reset_index(name='total_items')
    
    # Count total interactions per user
    user_total_interactions = df.groupby('user_id', observed=True, sort=False).size().reset_index(name='total_interactions')
    
    # Merge all statistics
    user_stats = user_total_items.merge(user_repeat_counts, on='user_id', how='left')
    user_stats = user_stats.merge(user_total_interactions, on='user_id', how='left')
    
    # Fill NaN for users with no repeat visits
    user_stats['num_repeat_items'] = user_stats['num_repeat_items'].fillna(0).astype(int)
    
    # Compute repeat visit ratio
    user_stats['repeat_item_ratio'] = user_stats['num_repeat_items'] / user_stats['total_items']
    
    elapsed = time.time() - start_time
    logger.info(f"User repeat frequency analysis complete, elapsed {elapsed:.2f} seconds")
    
    return user_stats


def analyze_hot_users(df: pd.DataFrame, top_percent: float = 0.1, show_progress: bool = False) -> Dict:
    """
    Analyze hot users: top X% users vs total visits
    """
    logger.info(f"Analyzing hot users (top {top_percent*100}%)...")
    start_time = time.time()
    
    # Count interactions per user
    if show_progress:
        logger.info("Counting interactions per user...")
    
    user_interaction_counts = df.groupby('user_id', observed=True, sort=False).size().reset_index(name='interaction_count')
    user_interaction_counts = user_interaction_counts.sort_values('interaction_count', ascending=False).reset_index(drop=True)
    
    total_users = len(user_interaction_counts)
    total_interactions = len(df)
    
    # Calculate top X% users
    num_top_users = max(1, int(total_users * top_percent))
    top_users = user_interaction_counts.head(num_top_users)
    
    top_users_interactions = top_users['interaction_count'].sum()
    top_users_ratio = top_users_interactions / total_interactions if total_interactions > 0 else 0.0
    
    stats = {
        'total_users': int(total_users),
        'total_interactions': int(total_interactions),
        'top_percent': float(top_percent),
        'top_user_count': int(num_top_users),
        'top_users_interactions': int(top_users_interactions),
        'top_users_ratio': float(top_users_ratio),
        'avg_interactions_per_user': float(user_interaction_counts['interaction_count'].mean()),
        'avg_interactions_top_users': float(top_users['interaction_count'].mean()),
        'median_interactions_per_user': float(user_interaction_counts['interaction_count'].median()),
        'min_interactions': int(user_interaction_counts['interaction_count'].min()),
        'max_interactions': int(user_interaction_counts['interaction_count'].max())
    }
    
    # Percentiles
    for p in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        stats[f'p{int(p*100)}'] = float(user_interaction_counts['interaction_count'].quantile(p))
    
    elapsed = time.time() - start_time
    logger.info(f"Hot user analysis complete, elapsed {elapsed:.2f} seconds")
    
    return stats, user_interaction_counts


def plot_repeat_visit_frequency_distribution(
    user_stats: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot repeat visit frequency distribution"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('User Repeat Visit Frequency Distribution', fontsize=16, fontweight='bold')
    
    # 1. Histogram of number of repeat items per user
    ax1 = axes[0, 0]
    repeat_items = user_stats['num_repeat_items']
    ax1.hist(repeat_items, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Repeat Visit Items per User', fontsize=10)
    ax1.set_ylabel('Number of Users', fontsize=10)
    ax1.set_title('Distribution of Repeat Visit Items per User', fontsize=12)
    ax1.axvline(repeat_items.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {repeat_items.mean():.2f}')
    ax1.axvline(repeat_items.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {repeat_items.median():.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF of repeat items
    ax2 = axes[0, 1]
    sorted_repeat = np.sort(repeat_items)
    y = np.arange(1, len(sorted_repeat) + 1) / len(sorted_repeat)
    ax2.plot(sorted_repeat, y * 100, linewidth=2, color='steelblue')
    ax2.set_xlabel('Number of Repeat Visit Items per User', fontsize=10)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=10)
    ax2.set_title('CDF of Repeat Visit Items per User', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Repeat item ratio distribution
    ax3 = axes[1, 0]
    repeat_ratio = user_stats['repeat_item_ratio']
    ax3.hist(repeat_ratio, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax3.set_xlabel('Repeat Item Ratio per User', fontsize=10)
    ax3.set_ylabel('Number of Users', fontsize=10)
    ax3.set_title('Distribution of Repeat Item Ratio', fontsize=12)
    ax3.axvline(repeat_ratio.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {repeat_ratio.mean():.3f}')
    ax3.axvline(repeat_ratio.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {repeat_ratio.median():.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter: total items vs repeat items
    ax4 = axes[1, 1]
    ax4.scatter(user_stats['total_items'], user_stats['num_repeat_items'], 
               alpha=0.3, s=10, color='steelblue')
    ax4.set_xlabel('Total Items per User', fontsize=10)
    ax4.set_ylabel('Number of Repeat Visit Items', fontsize=10)
    ax4.set_title('Total Items vs Repeat Visit Items', fontsize=12)
    ax4.grid(True, alpha=0.3)
    # Add diagonal line
    max_items = max(user_stats['total_items'].max(), user_stats['num_repeat_items'].max())
    ax4.plot([0, max_items], [0, max_items], 'r--', linewidth=2, alpha=0.5, label='y=x')
    ax4.legend()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved repeat visit frequency plot: {output_path}")


def plot_hot_users_analysis(
    user_interaction_counts: pd.DataFrame,
    hot_user_stats: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot hot users analysis"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Hot Users Analysis (Top Users vs Total Interactions)', fontsize=16, fontweight='bold')
    
    interaction_counts = user_interaction_counts['interaction_count'].values
    
    # 1. Histogram of interactions per user
    ax1 = axes[0, 0]
    ax1.hist(interaction_counts, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Interactions per User', fontsize=10)
    ax1.set_ylabel('Number of Users', fontsize=10)
    ax1.set_title('Distribution of Interactions per User', fontsize=12)
    mean_val = np.mean(interaction_counts)
    median_val = np.median(interaction_counts)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF of interactions
    ax2 = axes[0, 1]
    sorted_counts = np.sort(interaction_counts)
    y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(sorted_counts, y * 100, linewidth=2, color='steelblue')
    ax2.set_xlabel('Interactions per User', fontsize=10)
    ax2.set_ylabel('Cumulative Percentage of Users (%)', fontsize=10)
    ax2.set_title('CDF of Interactions per User', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # Mark top 10% point
    top_percent = hot_user_stats['top_percent']
    top_idx = int(len(sorted_counts) * (1 - top_percent))
    if top_idx < len(sorted_counts):
        ax2.axvline(sorted_counts[top_idx], color='red', linestyle='--', linewidth=2, 
                   label=f'Top {top_percent*100:.0f}% threshold')
        ax2.legend()
    
    # 3. Cumulative interactions (Lorenz curve style)
    ax3 = axes[1, 0]
    cumulative_users = np.arange(1, len(interaction_counts) + 1) / len(interaction_counts) * 100
    cumulative_interactions = np.cumsum(interaction_counts) / np.sum(interaction_counts) * 100
    ax3.plot(cumulative_users, cumulative_interactions, linewidth=2, color='steelblue', label='Actual')
    # Add diagonal line (perfect equality)
    ax3.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.5, label='Perfect Equality')
    # Mark top 10% point
    top_user_idx = int(len(interaction_counts) * top_percent)
    if top_user_idx > 0:
        top_user_ratio = cumulative_interactions[top_user_idx - 1]
        ax3.scatter([top_percent * 100], [top_user_ratio], s=200, color='red', 
                   zorder=5, label=f'Top {top_percent*100:.0f}% users: {top_user_ratio:.1f}% interactions')
    ax3.set_xlabel('Cumulative Percentage of Users (%)', fontsize=10)
    ax3.set_ylabel('Cumulative Percentage of Interactions (%)', fontsize=10)
    ax3.set_title(f'Lorenz Curve: Top {top_percent*100:.0f}% Users vs Total Interactions', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top users bar chart (top 20)
    ax4 = axes[1, 1]
    top_20 = user_interaction_counts.head(20)
    ax4.barh(range(len(top_20)), top_20['interaction_count'], color='steelblue', alpha=0.7)
    ax4.set_yticks(range(len(top_20)))
    ax4.set_yticklabels([f"User {i+1}" for i in range(len(top_20))], fontsize=8)
    ax4.set_xlabel('Number of Interactions', fontsize=10)
    ax4.set_title('Top 20 Most Active Users', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved hot users analysis plot: {output_path}")


def plot_interaction_frequency_curve(
    user_interaction_counts: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot interaction frequency curve (sorted by frequency)"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    interaction_counts = user_interaction_counts['interaction_count'].values
    sorted_counts = np.sort(interaction_counts)[::-1]  # Descending order
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('User Interaction Frequency Curve', fontsize=16, fontweight='bold')
    
    # 1. Linear scale
    ax1 = axes[0]
    x = np.arange(1, len(sorted_counts) + 1)
    ax1.plot(x, sorted_counts, linewidth=2, color='steelblue', alpha=0.8)
    ax1.set_xlabel('User Rank (sorted by interaction count)', fontsize=10)
    ax1.set_ylabel('Interaction Count', fontsize=10)
    ax1.set_title('Interaction Frequency Curve (Linear Scale)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Mark top 10% point
    top_10_idx = int(len(sorted_counts) * 0.1)
    if top_10_idx > 0:
        ax1.axvline(top_10_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Top 10% threshold ({top_10_idx} users)')
        ax1.scatter([top_10_idx], [sorted_counts[top_10_idx - 1]], 
                   s=100, color='red', zorder=5)
        ax1.legend()
    
    # 2. Log-log scale
    ax2 = axes[1]
    ax2.loglog(x, sorted_counts, linewidth=2, color='steelblue', alpha=0.8)
    ax2.set_xlabel('User Rank (log scale)', fontsize=10)
    ax2.set_ylabel('Interaction Count (log scale)', fontsize=10)
    ax2.set_title('Interaction Frequency Curve (Log-Log Scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Mark top 10% point
    if top_10_idx > 0:
        ax2.axvline(top_10_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Top 10% threshold')
        ax2.scatter([top_10_idx], [sorted_counts[top_10_idx - 1]], 
                   s=100, color='red', zorder=5)
        ax2.legend()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved interaction frequency curve: {output_path}")


def save_statistics_to_json(
    repeat_stats: Dict,
    hot_user_stats: Dict,
    output_path: Path
) -> None:
    """Save statistics to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'repeat_visit_statistics': repeat_stats,
        'hot_user_statistics': hot_user_stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved statistics to: {output_path}")


def print_statistics(repeat_stats: Dict, hot_user_stats: Dict):
    """Print statistics to console"""
    logger.info("\n" + "="*60)
    logger.info("Repeat Visit Statistics")
    logger.info("="*60)
    logger.info(f"Total user-item pairs: {repeat_stats['total_user_item_pairs']:,}")
    logger.info(f"Repeat visit pairs: {repeat_stats['repeat_visit_pairs']:,} ({repeat_stats['repeat_visit_ratio']*100:.2f}%)")
    logger.info(f"Total users: {repeat_stats['total_users']:,}")
    logger.info(f"Users with repeat visits: {repeat_stats['users_with_repeat_visits']:,} ({repeat_stats['users_with_repeat_ratio']*100:.2f}%)")
    logger.info(f"Average visits per pair: {repeat_stats['avg_visits_per_pair']:.2f}")
    logger.info(f"Average repeat visits: {repeat_stats['avg_repeat_visits']:.2f}")
    logger.info(f"Max visits per pair: {repeat_stats['max_visits_per_pair']}")
    
    logger.info("\n" + "="*60)
    logger.info("Hot User Statistics")
    logger.info("="*60)
    logger.info(f"Total users: {hot_user_stats['total_users']:,}")
    logger.info(f"Total interactions: {hot_user_stats['total_interactions']:,}")
    logger.info(f"Top {hot_user_stats['top_percent']*100:.0f}% users: {hot_user_stats['top_user_count']:,} users")
    logger.info(f"Top {hot_user_stats['top_percent']*100:.0f}% users interactions: {hot_user_stats['top_users_interactions']:,} ({hot_user_stats['top_users_ratio']*100:.2f}%)")
    logger.info(f"Average interactions per user: {hot_user_stats['avg_interactions_per_user']:.2f}")
    logger.info(f"Average interactions (top users): {hot_user_stats['avg_interactions_top_users']:.2f}")
    logger.info(f"Median interactions per user: {hot_user_stats['median_interactions_per_user']:.2f}")
    logger.info(f"P90: {hot_user_stats['p90']:.2f}")
    logger.info(f"P95: {hot_user_stats['p95']:.2f}")
    logger.info(f"P99: {hot_user_stats['p99']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze user repeat visits and hot users in Amazon Reviews dataset"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the Amazon Reviews JSONL file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: reports/ in script directory)'
    )
    parser.add_argument(
        '--top_percent',
        type=float,
        default=0.1,
        help='Top percentage of users to analyze (default: 0.1 for 10%%)'
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='Show progress bars'
    )
    parser.add_argument(
        '--no_multiprocessing',
        action='store_true',
        help='Disable multiprocessing for data loading'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=32,
        help='Number of worker processes (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    df = load_amazon_reviews_data(
        args.data_path,
        show_progress=args.show_progress,
        use_multiprocessing=not args.no_multiprocessing,
        n_workers=args.n_workers
    )
    
    # Analyze repeat visits
    repeat_stats, user_item_counts = analyze_repeat_visits(df, args.show_progress)
    
    # Analyze user repeat frequency
    user_stats = analyze_user_repeat_frequency(df, user_item_counts, args.show_progress)
    
    # Save user stats to CSV
    user_stats_file = output_dir / "user_repeat_frequency_stats.csv"
    user_stats.to_csv(user_stats_file, index=False)
    logger.info(f"Saved user stats to: {user_stats_file}")
    
    # Plot repeat visit frequency distribution
    repeat_freq_plot = output_dir / "repeat_visit_frequency_distribution.png"
    plot_repeat_visit_frequency_distribution(user_stats, repeat_freq_plot)
    
    # Analyze hot users
    hot_user_stats, user_interaction_counts = analyze_hot_users(
        df, top_percent=args.top_percent, show_progress=args.show_progress
    )
    
    # Plot hot users analysis
    hot_users_plot = output_dir / "hot_users_analysis.png"
    plot_hot_users_analysis(user_interaction_counts, hot_user_stats, hot_users_plot)
    
    # Plot interaction frequency curve
    freq_curve_plot = output_dir / "interaction_frequency_curve.png"
    plot_interaction_frequency_curve(user_interaction_counts, freq_curve_plot)
    
    # Save statistics
    stats_json = output_dir / "repeat_visit_statistics.json"
    save_statistics_to_json(repeat_stats, hot_user_stats, stats_json)
    
    # Print statistics
    print_statistics(repeat_stats, hot_user_stats)
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


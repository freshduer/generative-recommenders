#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze user repeat visit behavior and access concentration in Amazon Reviews dataset
- Analyze repeat visits (users visiting same items multiple times)
- Compute repeat visit frequency per user
- Plot frequency distribution
- Analyze long-tail distribution (top 10% users account for how much access)
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None

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
        
        # Extract user_id - try multiple field names
        user_id = record.get('user_id') or record.get('reviewerID')
        if not user_id:
            return None
        
        # Extract timestamp - try multiple formats
        timestamp = None
        
        # Try timestamp (milliseconds) - new format
        if 'timestamp' in record:
            ts_val = record['timestamp']
            if isinstance(ts_val, (int, float)):
                # Check if it's milliseconds (13 digits) or seconds (10 digits)
                if ts_val > 1e12:  # Milliseconds (13+ digits)
                    timestamp = int(ts_val) // 1000  # Convert to seconds
                else:  # Seconds (10 digits)
                    timestamp = int(ts_val)
        
        # Try unixReviewTime (seconds) - older format
        elif 'unixReviewTime' in record:
            timestamp = int(record['unixReviewTime'])
        
        # Try reviewTime string - fallback
        elif 'reviewTime' in record:
            try:
                review_time_str = record['reviewTime']
                dt = datetime.strptime(review_time_str, "%m %d, %Y")
                timestamp = int(dt.timestamp())
            except (ValueError, KeyError):
                return None
        
        if timestamp is None:
            return None
        
        return {
            'user_id': user_id,
            'asin': record.get('asin', ''),
            'timestamp': timestamp,
            'datetime': pd.Timestamp.fromtimestamp(timestamp)
        }
    except json.JSONDecodeError:
        return None
    except Exception:
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
    """Wrapper function for multiprocessing that returns chunk index and results"""
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
        # Use multiprocessing for faster parsing
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        logger.info(f"Using multiprocessing with {n_workers} workers...")
        
        # Read all lines into chunks
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
        
        # Add remaining lines
        if current_chunk:
            chunks.append((chunk_idx, current_chunk))
        
        logger.info(f"Prepared {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        if HAS_TQDM and show_progress:
            from tqdm import tqdm as tqdm_module
            with mp.Pool(processes=n_workers) as pool:
                results = []
                with tqdm_module(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
                    for result in pool.imap(load_chunk_wrapper, chunks):
                        results.append(result)
                        pbar.update(1)
                        if show_progress:
                            total_records = sum(len(r[1]) for r in results)
                            pbar.set_postfix({'records': f'{total_records:,}'})
        else:
            with mp.Pool(processes=n_workers) as pool:
                results = list(pool.imap(load_chunk_wrapper, chunks))
        
        # Sort results by chunk index and combine
        results.sort(key=lambda x: x[0])
        for _, records in results:
            all_records.extend(records)
        
        logger.info(f"Multiprocessing complete: {len(all_records):,} valid records")
    
    else:
        # Single-threaded processing
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
    
    # Convert to DataFrame
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Convert datetime column
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    # Remove invalid timestamps
    invalid_count = df['datetime'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid timestamp records, removing them")
        df = df.dropna(subset=['datetime'])
    
    # Sort by datetime
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
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Unique users: {df['user_id'].nunique():,}")
    logger.info(f"Unique products: {df['asin'].nunique():,}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    
    return df


def analyze_repeat_visits(df: pd.DataFrame, show_progress: bool = False) -> Dict:
    """
    Analyze repeat visit behavior: users visiting same items multiple times
    
    Returns:
        Dictionary with repeat visit statistics
    """
    logger.info("\n" + "="*60)
    logger.info("Analyzing Repeat Visit Behavior")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Count visits per user-item pair
    if show_progress:
        logger.info("Counting visits per user-item pair...")
    
    user_item_visits = df.groupby(['user_id', 'asin'], observed=True).size().reset_index(name='visit_count')
    
    total_pairs = len(user_item_visits)
    total_visits = len(df)
    
    # Identify repeat visits (visit_count > 1)
    repeat_pairs = user_item_visits[user_item_visits['visit_count'] > 1]
    num_repeat_pairs = len(repeat_pairs)
    num_single_visit_pairs = total_pairs - num_repeat_pairs
    
    repeat_pairs_pct = (num_repeat_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    # Users with repeat visits
    users_with_repeat = repeat_pairs['user_id'].nunique() if num_repeat_pairs > 0 else 0
    total_users = df['user_id'].nunique()
    users_with_repeat_pct = (users_with_repeat / total_users * 100) if total_users > 0 else 0
    
    # Average visits per pair
    avg_visits_per_pair = user_item_visits['visit_count'].mean()
    
    # Repeat visit statistics
    if num_repeat_pairs > 0:
        avg_repeat_visits = repeat_pairs['visit_count'].mean()
        max_repeat_visits = repeat_pairs['visit_count'].max()
        median_repeat_visits = repeat_pairs['visit_count'].median()
    else:
        avg_repeat_visits = 0.0
        max_repeat_visits = 0
        median_repeat_visits = 0.0
    
    # Total visits from repeat pairs
    total_repeat_visits = repeat_pairs['visit_count'].sum() if num_repeat_pairs > 0 else 0
    repeat_visits_pct = (total_repeat_visits / total_visits * 100) if total_visits > 0 else 0
    
    # Print statistics
    logger.info(f"Total user-item pairs: {total_pairs:,}")
    logger.info(f"Total visits: {total_visits:,}")
    logger.info(f"")
    logger.info(f"Repeat visit pairs (visit_count > 1): {num_repeat_pairs:,} ({repeat_pairs_pct:.2f}%)")
    logger.info(f"Single visit pairs: {num_single_visit_pairs:,} ({100-repeat_pairs_pct:.2f}%)")
    logger.info(f"")
    logger.info(f"Users with repeat visits: {users_with_repeat:,} / {total_users:,} ({users_with_repeat_pct:.2f}%)")
    logger.info(f"")
    logger.info(f"Average visits per pair: {avg_visits_per_pair:.2f}")
    if num_repeat_pairs > 0:
        logger.info(f"Average repeat visits per repeat pair: {avg_repeat_visits:.2f}")
        logger.info(f"Median repeat visits per repeat pair: {median_repeat_visits:.2f}")
        logger.info(f"Max repeat visits: {max_repeat_visits}")
    logger.info(f"")
    logger.info(f"Total visits from repeat pairs: {total_repeat_visits:,} / {total_visits:,} ({repeat_visits_pct:.2f}%)")
    
    # Visit count distribution
    if show_progress:
        logger.info("Computing visit count distribution...")
    
    visit_count_dist = user_item_visits['visit_count'].value_counts().sort_index()
    visit_count_stats = {}
    for visit_count, num_pairs in visit_count_dist.items():
        visit_count_stats[int(visit_count)] = {
            'num_pairs': int(num_pairs),
            'percentage': float(num_pairs / total_pairs * 100)
        }
    
    logger.info(f"\nVisit count distribution (top 10):")
    for visit_count in sorted(visit_count_stats.keys())[:10]:
        stats = visit_count_stats[visit_count]
        logger.info(f"  {visit_count} visit(s): {stats['num_pairs']:,} pairs ({stats['percentage']:.2f}%)")
    
    elapsed = time.time() - start_time
    logger.info(f"\nRepeat visit analysis complete, elapsed {elapsed:.2f} seconds")
    
    return {
        'total_pairs': int(total_pairs),
        'total_visits': int(total_visits),
        'num_repeat_pairs': int(num_repeat_pairs),
        'num_single_visit_pairs': int(num_single_visit_pairs),
        'repeat_pairs_pct': float(repeat_pairs_pct),
        'users_with_repeat': int(users_with_repeat),
        'total_users': int(total_users),
        'users_with_repeat_pct': float(users_with_repeat_pct),
        'avg_visits_per_pair': float(avg_visits_per_pair),
        'avg_repeat_visits': float(avg_repeat_visits),
        'median_repeat_visits': float(median_repeat_visits),
        'max_repeat_visits': int(max_repeat_visits),
        'total_repeat_visits': int(total_repeat_visits),
        'repeat_visits_pct': float(repeat_visits_pct),
        'visit_count_distribution': visit_count_stats
    }


def compute_user_repeat_frequency(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Compute repeat visit frequency for each user
    Frequency = number of repeat visits / total visits for that user
    
    Returns:
        DataFrame with user_id, total_visits, repeat_visits, repeat_frequency columns
    """
    logger.info("\n" + "="*60)
    logger.info("Computing User Repeat Visit Frequency")
    logger.info("="*60)
    
    start_time = time.time()
    
    if show_progress:
        logger.info("Counting visits per user...")
    
    # Total visits per user
    user_total_visits = df.groupby('user_id', observed=True).size().reset_index(name='total_visits')
    
    if show_progress:
        logger.info("Counting repeat visits per user...")
    
    # Count repeat visits per user (visits to items that user visited more than once)
    user_item_counts = df.groupby(['user_id', 'asin'], observed=True).size().reset_index(name='visit_count')
    repeat_user_items = user_item_counts[user_item_counts['visit_count'] > 1]
    
    if len(repeat_user_items) > 0:
        # For each user, sum up the repeat visits (visit_count - 1 for each repeat item)
        repeat_user_items['repeat_count'] = repeat_user_items['visit_count'] - 1
        user_repeat_visits = repeat_user_items.groupby('user_id', observed=True)['repeat_count'].sum().reset_index(name='repeat_visits')
    else:
        user_repeat_visits = pd.DataFrame(columns=['user_id', 'repeat_visits'])
    
    # Merge and compute frequency
    user_stats = user_total_visits.merge(
        user_repeat_visits,
        on='user_id',
        how='left'
    ).fillna({'repeat_visits': 0})
    
    user_stats['repeat_visits'] = user_stats['repeat_visits'].astype(int)
    user_stats['repeat_frequency'] = user_stats['repeat_visits'] / user_stats['total_visits']
    
    # Also compute unique items per user
    user_unique_items = df.groupby('user_id', observed=True)['asin'].nunique().reset_index(name='unique_items')
    user_stats = user_stats.merge(user_unique_items, on='user_id', how='left')
    
    # Compute average visits per item for each user
    user_stats['avg_visits_per_item'] = user_stats['total_visits'] / user_stats['unique_items']
    
    elapsed = time.time() - start_time
    logger.info(f"Computed repeat frequency for {len(user_stats):,} users, elapsed {elapsed:.2f} seconds")
    
    # Print statistics
    logger.info(f"\nRepeat frequency statistics:")
    logger.info(f"  Mean repeat frequency: {user_stats['repeat_frequency'].mean():.4f}")
    logger.info(f"  Median repeat frequency: {user_stats['repeat_frequency'].median():.4f}")
    logger.info(f"  Max repeat frequency: {user_stats['repeat_frequency'].max():.4f}")
    logger.info(f"  Users with repeat visits: {(user_stats['repeat_frequency'] > 0).sum():,} / {len(user_stats):,}")
    
    return user_stats.sort_values('total_visits', ascending=False).reset_index(drop=True)


def analyze_access_concentration(df: pd.DataFrame, user_stats: pd.DataFrame, show_progress: bool = False) -> Dict:
    """
    Analyze access concentration: top 10% users account for how much access
    
    Returns:
        Dictionary with access concentration statistics
    """
    logger.info("\n" + "="*60)
    logger.info("Analyzing Access Concentration (Long-tail Distribution)")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Sort users by total visits (descending)
    user_stats_sorted = user_stats.sort_values('total_visits', ascending=False).reset_index(drop=True)
    
    total_users = len(user_stats_sorted)
    total_visits = user_stats_sorted['total_visits'].sum()
    
    # Compute cumulative statistics
    user_stats_sorted['cumulative_visits'] = user_stats_sorted['total_visits'].cumsum()
    user_stats_sorted['cumulative_pct'] = (user_stats_sorted['cumulative_visits'] / total_visits * 100)
    user_stats_sorted['user_rank_pct'] = ((user_stats_sorted.index + 1) / total_users * 100)
    
    # Find top 10%, 5%, 1% users
    top_10_pct_idx = int(total_users * 0.1)
    top_5_pct_idx = int(total_users * 0.05)
    top_1_pct_idx = int(total_users * 0.01)
    
    top_10_pct_visits = user_stats_sorted.iloc[:top_10_pct_idx]['total_visits'].sum()
    top_5_pct_visits = user_stats_sorted.iloc[:top_5_pct_idx]['total_visits'].sum()
    top_1_pct_visits = user_stats_sorted.iloc[:top_1_pct_idx]['total_visits'].sum()
    
    top_10_pct_ratio = (top_10_pct_visits / total_visits * 100) if total_visits > 0 else 0
    top_5_pct_ratio = (top_5_pct_visits / total_visits * 100) if total_visits > 0 else 0
    top_1_pct_ratio = (top_1_pct_visits / total_visits * 100) if total_visits > 0 else 0
    
    # Print statistics
    logger.info(f"Total users: {total_users:,}")
    logger.info(f"Total visits: {total_visits:,}")
    logger.info(f"")
    logger.info(f"Top 1% users ({top_1_pct_idx:,} users): {top_1_pct_visits:,} visits ({top_1_pct_ratio:.2f}%)")
    logger.info(f"Top 5% users ({top_5_pct_idx:,} users): {top_5_pct_visits:,} visits ({top_5_pct_ratio:.2f}%)")
    logger.info(f"Top 10% users ({top_10_pct_idx:,} users): {top_10_pct_visits:,} visits ({top_10_pct_ratio:.2f}%)")
    logger.info(f"")
    logger.info(f"Bottom 90% users ({total_users - top_10_pct_idx:,} users): {total_visits - top_10_pct_visits:,} visits ({100-top_10_pct_ratio:.2f}%)")
    
    # Compute Gini coefficient (measure of inequality)
    # Sort visits in ascending order for Gini calculation
    visits_sorted = np.sort(user_stats_sorted['total_visits'].values)
    n = len(visits_sorted)
    cumsum = np.cumsum(visits_sorted)
    gini = (2 * np.sum((np.arange(1, n+1)) * visits_sorted)) / (n * cumsum[-1]) - (n + 1) / n
    
    logger.info(f"")
    logger.info(f"Gini coefficient: {gini:.4f} (0 = perfect equality, 1 = perfect inequality)")
    
    elapsed = time.time() - start_time
    logger.info(f"\nAccess concentration analysis complete, elapsed {elapsed:.2f} seconds")
    
    return {
        'total_users': int(total_users),
        'total_visits': int(total_visits),
        'top_1_pct_users': int(top_1_pct_idx),
        'top_1_pct_visits': int(top_1_pct_visits),
        'top_1_pct_ratio': float(top_1_pct_ratio),
        'top_5_pct_users': int(top_5_pct_idx),
        'top_5_pct_visits': int(top_5_pct_visits),
        'top_5_pct_ratio': float(top_5_pct_ratio),
        'top_10_pct_users': int(top_10_pct_idx),
        'top_10_pct_visits': int(top_10_pct_visits),
        'top_10_pct_ratio': float(top_10_pct_ratio),
        'gini_coefficient': float(gini),
        'cumulative_stats': user_stats_sorted[['user_id', 'total_visits', 'user_rank_pct', 'cumulative_pct']].to_dict('records')
    }


def plot_visit_count_distribution(
    repeat_stats: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot visit count distribution"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    visit_dist = repeat_stats['visit_count_distribution']
    
    visit_counts = sorted([k for k in visit_dist.keys() if k <= 20])  # Show up to 20 visits
    pair_counts = [visit_dist[k]['num_pairs'] for k in visit_counts]
    percentages = [visit_dist[k]['percentage'] for k in visit_counts]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Bar chart
    ax1 = axes[0]
    ax1.bar(visit_counts, pair_counts, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Visits per User-Item Pair', fontsize=12)
    ax1.set_ylabel('Number of Pairs', fontsize=12)
    ax1.set_title('Visit Count Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # 2. Percentage bar chart
    ax2 = axes[1]
    ax2.bar(visit_counts, percentages, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Number of Visits per User-Item Pair', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Visit Count Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visit count distribution plot: {output_path}")


def plot_repeat_frequency_distribution(
    user_stats: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Plot repeat frequency distribution per user"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('User Repeat Visit Frequency Distribution', fontsize=16, fontweight='bold')
    
    # 1. Repeat frequency histogram
    ax1 = axes[0, 0]
    ax1.hist(user_stats['repeat_frequency'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(user_stats['repeat_frequency'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {user_stats["repeat_frequency"].mean():.4f}')
    ax1.axvline(user_stats['repeat_frequency'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {user_stats["repeat_frequency"].median():.4f}')
    ax1.set_xlabel('Repeat Visit Frequency', fontsize=10)
    ax1.set_ylabel('Number of Users', fontsize=10)
    ax1.set_title('Repeat Frequency Distribution (Histogram)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Repeat frequency CDF
    ax2 = axes[0, 1]
    sorted_freq = np.sort(user_stats['repeat_frequency'])
    y = np.arange(1, len(sorted_freq) + 1) / len(sorted_freq)
    ax2.plot(sorted_freq, y * 100, linewidth=2, color='steelblue')
    ax2.axvline(user_stats['repeat_frequency'].median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: {user_stats["repeat_frequency"].median():.4f}')
    ax2.set_xlabel('Repeat Visit Frequency', fontsize=10)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=10)
    ax2.set_title('Repeat Frequency CDF', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Repeat frequency vs Total visits (scatter)
    ax3 = axes[1, 0]
    sample_size = min(10000, len(user_stats))
    sample_stats = user_stats.sample(n=sample_size, random_state=42) if sample_size < len(user_stats) else user_stats
    ax3.scatter(sample_stats['total_visits'], sample_stats['repeat_frequency'], 
                alpha=0.5, s=10, color='steelblue')
    ax3.set_xlabel('Total Visits per User', fontsize=10)
    ax3.set_ylabel('Repeat Visit Frequency', fontsize=10)
    ax3.set_title(f'Repeat Frequency vs Total Visits (sample: {sample_size:,} users)', fontsize=12)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_stats = {
        'Total Users': len(user_stats),
        'Mean Frequency': user_stats['repeat_frequency'].mean(),
        'Median Frequency': user_stats['repeat_frequency'].median(),
        'Std Frequency': user_stats['repeat_frequency'].std(),
        'Max Frequency': user_stats['repeat_frequency'].max(),
        'Users with Repeat': (user_stats['repeat_frequency'] > 0).sum(),
        'Users with Repeat %': (user_stats['repeat_frequency'] > 0).sum() / len(user_stats) * 100,
    }
    summary_text = "Summary Statistics:\n\n"
    for key, value in summary_stats.items():
        if isinstance(value, float):
            summary_text += f"{key}: {value:.4f}\n"
        else:
            summary_text += f"{key}: {value:,}\n"
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved repeat frequency distribution plot: {output_path}")


def plot_access_concentration(
    user_stats: pd.DataFrame,
    concentration_stats: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Plot access concentration (long-tail distribution)"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    # Prepare data
    user_stats_sorted = user_stats.sort_values('total_visits', ascending=False).reset_index(drop=True)
    total_users = len(user_stats_sorted)
    total_visits = user_stats_sorted['total_visits'].sum()
    
    user_stats_sorted['cumulative_visits'] = user_stats_sorted['total_visits'].cumsum()
    user_stats_sorted['cumulative_pct'] = (user_stats_sorted['cumulative_visits'] / total_visits * 100)
    user_stats_sorted['user_rank_pct'] = ((user_stats_sorted.index + 1) / total_users * 100)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Access Concentration Analysis (Long-tail Distribution)', fontsize=16, fontweight='bold')
    
    # 1. Cumulative access percentage (Lorenz curve)
    ax1 = axes[0, 0]
    # Perfect equality line
    ax1.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.5, label='Perfect Equality')
    # Actual curve
    ax1.plot(user_stats_sorted['user_rank_pct'], user_stats_sorted['cumulative_pct'], 
             linewidth=2, color='steelblue', label='Actual Distribution')
    # Mark top 10% point
    top_10_pct = concentration_stats['top_10_pct_ratio']
    ax1.plot([10, 10], [0, top_10_pct], 'r--', linewidth=1.5, alpha=0.7)
    ax1.plot([0, 10], [top_10_pct, top_10_pct], 'r--', linewidth=1.5, alpha=0.7, 
             label=f'Top 10%: {top_10_pct:.2f}%')
    ax1.set_xlabel('User Rank (Percentage)', fontsize=10)
    ax1.set_ylabel('Cumulative Access Percentage (%)', fontsize=10)
    ax1.set_title('Lorenz Curve (Access Concentration)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    # 2. Top users access distribution
    ax2 = axes[0, 1]
    top_n = min(100, len(user_stats_sorted))
    top_users = user_stats_sorted.head(top_n)
    ax2.bar(range(len(top_users)), top_users['total_visits'], alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel(f'User Rank (Top {top_n} Users)', fontsize=10)
    ax2.set_ylabel('Total Visits', fontsize=10)
    ax2.set_title(f'Top {top_n} Users Access Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Log-log plot of user visits distribution
    ax3 = axes[1, 0]
    visit_counts = user_stats_sorted['total_visits'].values
    rank = np.arange(1, len(visit_counts) + 1)
    ax3.loglog(rank, visit_counts, 'o', markersize=2, alpha=0.5, color='steelblue')
    ax3.set_xlabel('User Rank (Log Scale)', fontsize=10)
    ax3.set_ylabel('Total Visits (Log Scale)', fontsize=10)
    ax3.set_title('User Visits Distribution (Log-Log Plot)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "Access Concentration Statistics:\n\n"
    summary_text += f"Total Users: {concentration_stats['total_users']:,}\n"
    summary_text += f"Total Visits: {concentration_stats['total_visits']:,}\n\n"
    summary_text += f"Top 1% Users: {concentration_stats['top_1_pct_ratio']:.2f}%\n"
    summary_text += f"Top 5% Users: {concentration_stats['top_5_pct_ratio']:.2f}%\n"
    summary_text += f"Top 10% Users: {concentration_stats['top_10_pct_ratio']:.2f}%\n\n"
    summary_text += f"Gini Coefficient: {concentration_stats['gini_coefficient']:.4f}\n"
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved access concentration plot: {output_path}")


def save_results_to_json(results: Dict, output_path: Path) -> None:
    """Save analysis results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze user repeat visit behavior and access concentration"
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
    repeat_stats = analyze_repeat_visits(df, args.show_progress)
    
    # Compute user repeat frequency
    user_stats = compute_user_repeat_frequency(df, args.show_progress)
    
    # Analyze access concentration
    concentration_stats = analyze_access_concentration(df, user_stats, args.show_progress)
    
    # Plot visit count distribution
    visit_dist_plot = output_dir / "visit_count_distribution.png"
    plot_visit_count_distribution(repeat_stats, visit_dist_plot)
    
    # Plot repeat frequency distribution
    freq_dist_plot = output_dir / "repeat_frequency_distribution.png"
    plot_repeat_frequency_distribution(user_stats, freq_dist_plot)
    
    # Plot access concentration
    concentration_plot = output_dir / "access_concentration.png"
    plot_access_concentration(user_stats, concentration_stats, concentration_plot)
    
    # Save statistics to CSV
    user_stats_file = output_dir / "user_repeat_frequency_stats.csv"
    user_stats.to_csv(user_stats_file, index=False)
    logger.info(f"Saved user statistics to: {user_stats_file}")
    
    # Save results to JSON
    all_results = {
        'repeat_visit_statistics': repeat_stats,
        'access_concentration_statistics': concentration_stats,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    results_json = output_dir / "repeat_visit_analysis_results.json"
    save_results_to_json(all_results, results_json)
    
    logger.info("\n" + "="*60)
    logger.info("All analyses complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


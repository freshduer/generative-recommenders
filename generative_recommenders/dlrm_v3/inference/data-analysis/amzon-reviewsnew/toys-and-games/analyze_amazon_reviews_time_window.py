#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Amazon Reviews dataset user interaction sequence length distribution and active user count distribution
across different time windows (30min, 1h, 12h, 1d)
Optimized for large files using chunking, parallel processing, and efficient data structures
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
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
    
    Expected format: JSON object with fields like:
    - user_id: user identifier (or reviewerID for older format)
    - asin: product identifier
    - timestamp: Unix timestamp in milliseconds (or unixReviewTime in seconds for older format)
    - reviewTime: timestamp string (e.g., "01 1, 2014") - fallback option
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
                # Handle format like "01 1, 2014" or "01 15, 2014"
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
        # Silently skip invalid lines (avoid logger in multiprocessing)
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
    
    Args:
        data_path: Path to the JSONL file
        show_progress: Whether to show progress
        use_multiprocessing: Whether to use multiprocessing for parsing
        chunk_size: Number of lines to process at once
        n_workers: Number of worker processes (None = auto)
    
    Returns:
        DataFrame with columns: user_id, timestamp, datetime, asin
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
    
    # Test parsing first few lines to verify format
    logger.info("Testing data format with first few lines...")
    test_count = 0
    test_success = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test first 5 lines
                break
            test_count += 1
            record = parse_amazon_review_line(line)
            if record:
                test_success += 1
                logger.info(f"  Sample record: user_id={record['user_id'][:20]}..., timestamp={record['timestamp']}, datetime={record['datetime']}")
    
    if test_success == 0:
        logger.error("Failed to parse any test records! Please check data format.")
        logger.error("Expected fields: user_id (or reviewerID), timestamp (or unixReviewTime), asin")
        raise ValueError("Data format mismatch - cannot parse any records")
    
    logger.info(f"Format test: {test_success}/{test_count} records parsed successfully")
    
    all_records = []
    
    if use_multiprocessing and n_workers != 1:
        # Use multiprocessing for faster parsing
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid overhead
        
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
        
        logger.info(f"Multiprocessing complete: {len(all_records):,} valid records from {sum(len(chunk[1]) for chunk in chunks):,} total lines")
    
    else:
        # Single-threaded processing
        logger.info("Using single-threaded processing...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            if HAS_TQDM and show_progress:
                f = tqdm(f, desc="Reading data", unit="lines", total=total_lines)
            
            line_count = 0
            for line in f:
                line_count += 1
                record = parse_amazon_review_line(line)
                if record:
                    all_records.append(record)
                
                # Report progress every chunk_size lines
                if line_count % chunk_size == 0 and show_progress:
                    logger.info(f"Processed {line_count:,} lines, {len(all_records):,} valid records...")
            
            if show_progress:
                logger.info(f"Finished reading: {line_count:,} total lines, {len(all_records):,} valid records")
    
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


def create_time_window(datetime_series: pd.Series, window_type: str) -> pd.Series:
    """
    Create time window labels based on window type
    
    Args:
        datetime_series: datetime Series
        window_type: Time window type ('30min', '1h', '12h', '1d')
    
    Returns:
        Time window label Series
    """
    if window_type == "30min":
        return datetime_series.dt.floor('30min')
    elif window_type == "1h":
        return datetime_series.dt.floor('H')
    elif window_type == "12h":
        hours = datetime_series.dt.hour
        adjusted = datetime_series.dt.normalize() + pd.to_timedelta((hours // 12) * 12, unit='h')
        return adjusted
    elif window_type == "1d":
        return datetime_series.dt.date
    else:
        raise ValueError(f"Unsupported window type: {window_type}")


def compute_user_sequence_lengths_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute cumulative sequence lengths for each user by time window
    
    Optimized for large datasets using efficient grouping and numpy operations.
    
    Args:
        df: DataFrame with datetime column
        window_type: Time window type
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with time_window, user_id, sequence_length columns
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"Computing user sequence lengths for {window_type} windows...")
    
    # Create time window labels
    if show_progress:
        logger.info("Creating time window labels...")
    time_window = create_time_window(df["datetime"], window_type)
    
    # Use category type for faster grouping (especially for user_id)
    if show_progress:
        logger.info("Converting to category types for faster grouping...")
    
    # Convert to category to speed up grouping and reduce memory
    # Only convert if not already category and if it helps
    user_ids = df['user_id']
    if user_ids.dtype != 'category' and user_ids.nunique() < len(user_ids) * 0.5:
        user_ids = user_ids.astype('category')
    
    time_windows = time_window
    if time_windows.dtype != 'category':
        time_windows = time_windows.astype('category')
    
    # Compute interactions per window per user
    if show_progress:
        logger.info("Counting interactions per window per user...")
    
    # Use groupby with as_index=False for better performance
    temp_df = pd.DataFrame({
        'time_window': time_windows,
        'user_id': user_ids
    })
    
    # Groupby with as_index=False is faster for large datasets
    window_interactions = temp_df.groupby(["time_window", "user_id"], observed=True, sort=False, as_index=False).size()
    window_interactions.columns = ['time_window', 'user_id', 'window_count']
    
    if show_progress:
        logger.info(f"Found {len(window_interactions):,} unique (time_window, user_id) pairs")
        logger.info("Sorting for cumulative computation...")
    
    # Sort by user_id first, then time_window (more efficient for cumsum)
    window_interactions = window_interactions.sort_values(['user_id', 'time_window'], kind='mergesort')
    
    if show_progress:
        logger.info("Computing cumulative sequence lengths...")
    
    # Compute cumulative sum for each user using groupby.cumsum (very fast)
    window_interactions['sequence_length'] = window_interactions.groupby('user_id', observed=True, sort=False)['window_count'].cumsum()
    
    # Select only needed columns
    result = window_interactions[['time_window', 'user_id', 'sequence_length']].copy()
    
    # Convert back from category if needed
    if result['time_window'].dtype.name == 'category':
        result['time_window'] = result['time_window'].astype(time_window.dtype)
    if result['user_id'].dtype.name == 'category':
        result['user_id'] = result['user_id'].astype(str)
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Sequence length computation complete: {len(result):,} records, elapsed {elapsed:.2f} seconds")
    
    return result


def compute_active_users_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute active user count per time window
    
    Optimized for large datasets.
    
    Args:
        df: DataFrame with datetime column
        window_type: Time window type
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with time_window and active_user_count columns
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"Computing active users for {window_type} windows...")
    
    # Create time window labels
    time_window = create_time_window(df["datetime"], window_type)
    
    # Use drop_duplicates + value_counts for better performance on large datasets
    if show_progress:
        logger.info("Computing unique users per window...")
    
    # More efficient: drop duplicates first, then count
    unique_pairs = pd.DataFrame({
        'time_window': time_window.values,
        'user_id': df['user_id'].values
    }).drop_duplicates()
    
    # Count unique users per window
    active_users = unique_pairs['time_window'].value_counts().sort_index().reset_index()
    active_users.columns = ['time_window', 'active_user_count']
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Active user computation complete: {len(active_users)} windows, elapsed {elapsed:.2f} seconds")
    
    return active_users


def compute_window_statistics(
    df: pd.DataFrame,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute statistics for each time window
    
    Optimized to compute all statistics in a single pass where possible.
    
    Returns:
        DataFrame with statistics for each time window
    """
    start_time = time.time()
    if show_progress:
        unique_windows = df["time_window"].nunique()
        logger.info(f"Computing statistics for {unique_windows} time windows...")
    
    # Group by time window
    grouped = df.groupby("time_window", observed=True, sort=True)["sequence_length"]
    
    # Compute basic statistics - use agg() for better performance
    if show_progress:
        logger.info("Computing basic statistics...")
    
    # Use describe() + agg() for faster computation
    stats = grouped.agg(['count', 'mean', 'std', 'min', 'max', 'median']).reset_index()
    stats.columns = ['time_window', 'count', 'mean', 'std', 'min', 'max', 'median']
    
    # Ensure numeric types
    numeric_cols = ['count', 'mean', 'std', 'min', 'max', 'median']
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors='coerce').astype(float)
    
    # Compute variance
    stats['variance'] = stats['std'] ** 2
    
    # Fill NaN values
    stats['std'] = stats['std'].fillna(0.0).astype(float)
    stats['variance'] = stats['variance'].fillna(0.0).astype(float)
    
    # Compute percentiles
    if show_progress:
        logger.info("Computing percentiles...")
    
    all_percentiles = sorted(set(percentiles + [0.25, 0.5, 0.75]))
    
    # Set p50 as median
    if 0.5 in all_percentiles:
        stats['p50'] = stats['median']
    
    # Compute other percentiles
    percentiles_to_compute = sorted(set([p for p in all_percentiles if p != 0.5]))
    
    if percentiles_to_compute:
        if show_progress:
            logger.info(f"Computing {len(percentiles_to_compute)} percentiles...")
        # Compute percentiles one by one (more memory efficient for large datasets)
        for p in percentiles_to_compute:
            col_name = f"p{int(p*100)}"
            if col_name not in stats.columns:
                quantile_series = grouped.quantile(p)
                stats[col_name] = stats['time_window'].map(quantile_series).astype(float)
    
    # Ensure p25 and p75 exist
    if 'p25' not in stats.columns:
        p25_series = grouped.quantile(0.25)
        stats['p25'] = stats['time_window'].map(p25_series).astype(float)
    if 'p75' not in stats.columns:
        p75_series = grouped.quantile(0.75)
        stats['p75'] = stats['time_window'].map(p75_series).astype(float)
    
    # Compute IQR
    stats['p25'] = stats['p25'].astype(float)
    stats['p75'] = stats['p75'].astype(float)
    stats['iqr'] = stats['p75'] - stats['p25']
    
    # Compute skewness and kurtosis
    if show_progress:
        logger.info("Computing skewness and kurtosis...")
    
    try:
        skew_vals = grouped.skew().fillna(0.0)
        kurt_vals = grouped.apply(lambda x: x.kurtosis() if len(x) > 2 else 0.0).fillna(0.0)
        stats['skewness'] = stats['time_window'].map(skew_vals).fillna(0.0)
        stats['kurtosis'] = stats['time_window'].map(kurt_vals).fillna(0.0)
    except Exception as e:
        logger.warning(f"Error computing skewness/kurtosis: {e}, setting to 0")
        stats['skewness'] = 0.0
        stats['kurtosis'] = 0.0
    
    # Ensure all numeric columns are float
    numeric_cols = stats.select_dtypes(include=[np.number]).columns
    stats[numeric_cols] = stats[numeric_cols].astype(float)
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Statistics computation complete, elapsed {elapsed:.2f} seconds")
    
    return stats


def compute_active_users_statistics(
    active_users_df: pd.DataFrame,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute statistics for active user counts
    
    Returns:
        DataFrame with active user count statistics
    """
    start_time = time.time()
    if show_progress:
        logger.info("Computing active user count statistics...")
    
    stats = active_users_df.copy()
    
    # Compute rolling statistics
    window_size = min(10, len(stats) // 4)
    if window_size > 1:
        stats['active_user_count_mean_rolling'] = stats['active_user_count'].rolling(window=window_size, center=True).mean()
        stats['active_user_count_std_rolling'] = stats['active_user_count'].rolling(window=window_size, center=True).std()
    
    # Compute basic statistics
    active_user_counts = stats['active_user_count']
    stats['mean'] = active_user_counts.mean()
    stats['std'] = active_user_counts.std()
    stats['median'] = active_user_counts.median()
    stats['min'] = active_user_counts.min()
    stats['max'] = active_user_counts.max()
    stats['variance'] = active_user_counts.var()
    
    # Compute percentiles
    for p in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        col_name = f"p{int(p*100)}"
        stats[col_name] = active_user_counts.quantile(p)
    
    stats['iqr'] = stats['p75'] - stats['p25']
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Active user statistics complete, elapsed {elapsed:.2f} seconds")
    
    return stats


def plot_sequence_length_over_time(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot sequence length statistics over time (fluctuation chart)"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    # Ensure time_window is datetime type
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        if isinstance(window_stats["time_window"].iloc[0], (datetime, pd.Timestamp)):
            window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
        else:
            window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'User Interaction Sequence Length Distribution Statistics - {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    x = window_stats["time_window"]
    
    # 1. Mean, median, p90, p99 time series
    ax1 = axes[0, 0]
    ax1.plot(x, window_stats["mean"], label="Mean", marker='o', markersize=4, linewidth=2, alpha=0.8)
    ax1.plot(x, window_stats["median"], label="Median", marker='s', markersize=4, linewidth=2, alpha=0.8)
    if "p90" in window_stats.columns:
        ax1.plot(x, window_stats["p90"], label="P90", marker='^', markersize=4, linewidth=2, alpha=0.8, linestyle='--')
    if "p99" in window_stats.columns:
        ax1.plot(x, window_stats["p99"], label="P99", marker='d', markersize=4, linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_xlabel("Time Window", fontsize=10)
    ax1.set_ylabel("Sequence Length", fontsize=10)
    ax1.set_title("Sequence Length Statistics Over Time", fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Variance and std dev time series
    ax2 = axes[0, 1]
    ax2.plot(x, window_stats["variance"], label="Variance", marker='o', markersize=4, linewidth=2, alpha=0.8, color='orange')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, window_stats["std"], label="Std Dev", marker='s', markersize=4, linewidth=2, alpha=0.8, color='green')
    ax2.set_xlabel("Time Window", fontsize=10)
    ax2.set_ylabel("Variance", fontsize=10, color='orange')
    ax2_twin.set_ylabel("Std Dev", fontsize=10, color='green')
    ax2.set_title("Sequence Length Variance and Std Dev Over Time", fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    if mdates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Min and max time series
    ax3 = axes[1, 0]
    ax3.plot(x, window_stats["min"], label="Min", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    ax3.plot(x, window_stats["max"], label="Max", marker='s', markersize=4, linewidth=2, alpha=0.8, color='red')
    ax3.set_xlabel("Time Window", fontsize=10)
    ax3.set_ylabel("Sequence Length", fontsize=10)
    ax3.set_title("Sequence Length Min and Max Over Time", fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if mdates is not None:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. IQR time series
    ax4 = axes[1, 1]
    if "iqr" in window_stats.columns:
        ax4.plot(x, window_stats["iqr"], label="IQR (Interquartile Range)", marker='o', markersize=4, linewidth=2, alpha=0.8, color='purple')
    ax4.set_xlabel("Time Window", fontsize=10)
    ax4.set_ylabel("IQR", fontsize=10)
    ax4.set_title("Sequence Length IQR Over Time", fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    if mdates is not None:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_detailed_statistics(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Plot detailed statistics including all percentiles"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Detailed Sequence Length Statistics - {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    x = window_stats["time_window"]
    
    # 1. All percentiles
    ax1 = axes[0, 0]
    percentile_cols = [col for col in window_stats.columns if col.startswith('p') and col[1:].isdigit()]
    colors = plt.cm.tab10(np.linspace(0, 1, len(percentile_cols)))
    for col, color in zip(percentile_cols, colors):
        ax1.plot(x, window_stats[col], label=col.upper(), marker='o', markersize=3, linewidth=1.5, alpha=0.7, color=color)
    ax1.set_xlabel("Time Window", fontsize=10)
    ax1.set_ylabel("Sequence Length", fontsize=10)
    ax1.set_title("All Percentiles Over Time", fontsize=12)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Mean with confidence intervals (using std)
    ax2 = axes[0, 1]
    mean_vals = window_stats["mean"]
    std_vals = window_stats["std"]
    ax2.plot(x, mean_vals, label="Mean", marker='o', markersize=4, linewidth=2, alpha=0.8)
    ax2.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, label="±1 Std Dev")
    ax2.set_xlabel("Time Window", fontsize=10)
    ax2.set_ylabel("Sequence Length", fontsize=10)
    ax2.set_title("Mean with Standard Deviation Bands", fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    if mdates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Skewness and kurtosis
    ax3 = axes[1, 0]
    if "skewness" in window_stats.columns:
        ax3.plot(x, window_stats["skewness"], label="Skewness", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    if "kurtosis" in window_stats.columns:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x, window_stats["kurtosis"], label="Kurtosis", marker='s', markersize=4, linewidth=2, alpha=0.8, color='red')
        ax3_twin.set_ylabel("Kurtosis", fontsize=10, color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        ax3_twin.legend(loc='upper right', fontsize=9)
    ax3.set_xlabel("Time Window", fontsize=10)
    ax3.set_ylabel("Skewness", fontsize=10, color='blue')
    ax3.set_title("Skewness and Kurtosis Over Time", fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if mdates is not None:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Count of users per window
    ax4 = axes[1, 1]
    if "count" in window_stats.columns:
        ax4.plot(x, window_stats["count"], label="User Count", marker='o', markersize=4, linewidth=2, alpha=0.8, color='green')
    ax4.set_xlabel("Time Window", fontsize=10)
    ax4.set_ylabel("Number of Users", fontsize=10)
    ax4.set_title("Number of Users Per Time Window", fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    if mdates is not None:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Box plot data visualization (using quartiles)
    ax5 = axes[2, 0]
    if all(col in window_stats.columns for col in ['min', 'p25', 'median', 'p75', 'max']):
        # Create box plot-like visualization
        for i, (idx, row) in enumerate(window_stats.iterrows()):
            if i % max(1, len(window_stats) // 50) == 0:  # Sample to avoid overcrowding
                box_data = [row['min'], row['p25'], row['median'], row['p75'], row['max']]
                ax5.plot([i, i], [row['min'], row['max']], 'k-', linewidth=0.5, alpha=0.5)
                ax5.plot([i, i], [row['p25'], row['p75']], 'b-', linewidth=2, alpha=0.7)
                ax5.plot(i, row['median'], 'ro', markersize=3, alpha=0.7)
    ax5.set_xlabel("Time Window Index", fontsize=10)
    ax5.set_ylabel("Sequence Length", fontsize=10)
    ax5.set_title("Distribution Spread Over Time (Box Plot Style)", fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = axes[2, 1]
    ax6.axis('off')
    summary_stats = {
        'Overall Mean': window_stats['mean'].mean(),
        'Overall Std': window_stats['std'].mean(),
        'Overall Median': window_stats['median'].median(),
        'Overall P90': window_stats['p90'].mean() if 'p90' in window_stats.columns else None,
        'Overall P99': window_stats['p99'].mean() if 'p99' in window_stats.columns else None,
        'Overall Variance': window_stats['variance'].mean(),
    }
    summary_text = "Overall Statistics:\n\n"
    for key, value in summary_stats.items():
        if value is not None:
            summary_text += f"{key}: {value:.2f}\n"
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved detailed plot: {output_path}")


def plot_active_users_over_time(
    active_users_df: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot active user count over time"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    active_users_df = active_users_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(active_users_df["time_window"]):
        active_users_df["time_window"] = pd.to_datetime(active_users_df["time_window"])
    
    active_users_df = active_users_df.sort_values("time_window")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Active Users Over Time - {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    x = active_users_df["time_window"]
    y = active_users_df["active_user_count"]
    
    # 1. Active user count over time
    ax1 = axes[0, 0]
    ax1.plot(x, y, marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    ax1.set_xlabel("Time Window", fontsize=10)
    ax1.set_ylabel("Active User Count", fontsize=10)
    ax1.set_title("Active User Count Over Time", fontsize=12)
    ax1.grid(True, alpha=0.3)
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Active user count with rolling mean
    ax2 = axes[0, 1]
    ax2.plot(x, y, marker='o', markersize=3, linewidth=1, alpha=0.5, color='lightblue', label='Raw Data')
    if 'active_user_count_mean_rolling' in active_users_df.columns:
        ax2.plot(x, active_users_df['active_user_count_mean_rolling'], 
                marker='s', markersize=4, linewidth=2, alpha=0.8, color='red', label='Rolling Mean')
    ax2.set_xlabel("Time Window", fontsize=10)
    ax2.set_ylabel("Active User Count", fontsize=10)
    ax2.set_title("Active User Count with Rolling Mean", fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    if mdates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Statistics overlay
    ax3 = axes[1, 0]
    if 'mean' in active_users_df.columns:
        mean_val = active_users_df['mean'].iloc[0]
        ax3.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    if 'median' in active_users_df.columns:
        median_val = active_users_df['median'].iloc[0]
        ax3.axhline(y=median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
    if 'p90' in active_users_df.columns:
        p90_val = active_users_df['p90'].iloc[0]
        ax3.axhline(y=p90_val, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90_val:.0f}')
    if 'p99' in active_users_df.columns:
        p99_val = active_users_df['p99'].iloc[0]
        ax3.axhline(y=p99_val, color='purple', linestyle='--', linewidth=2, label=f'P99: {p99_val:.0f}')
    ax3.plot(x, y, marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    ax3.set_xlabel("Time Window", fontsize=10)
    ax3.set_ylabel("Active User Count", fontsize=10)
    ax3.set_title("Active User Count with Statistics Lines", fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if mdates is not None:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    if 'mean' in active_users_df.columns:
        mean_val = active_users_df['mean'].iloc[0]
        ax4.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    if 'median' in active_users_df.columns:
        median_val = active_users_df['median'].iloc[0]
        ax4.axvline(x=median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
    ax4.set_xlabel("Active User Count", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_title("Distribution of Active User Counts", fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved active users plot: {output_path}")


def save_statistics_to_file(
    stats_df: pd.DataFrame,
    output_path: Path,
    window_type: str,
    stats_type: str = "sequence_length"
) -> None:
    """Save statistics to CSV file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    logger.info(f"Saved {stats_type} statistics to: {output_path}")


def analyze_time_window(
    df: pd.DataFrame,
    window_type: str,
    output_dir: Path,
    show_progress: bool = False
) -> None:
    """Analyze a specific time window"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {window_type} time window")
    logger.info(f"{'='*60}")
    
    # Compute sequence lengths
    sequence_lengths = compute_user_sequence_lengths_by_window(df, window_type, show_progress)
    
    # Compute statistics
    window_stats = compute_window_statistics(sequence_lengths, show_progress=show_progress)
    
    # Save statistics
    stats_file = output_dir / f"sequence_length_stats_{window_type}.csv"
    save_statistics_to_file(window_stats, stats_file, window_type, "sequence_length")
    
    # Plot sequence length statistics
    plot_file = output_dir / f"sequence_length_over_time_{window_type}.png"
    plot_sequence_length_over_time(window_stats, plot_file, window_type)
    
    # Plot detailed statistics
    detailed_plot_file = output_dir / f"sequence_length_detailed_stats_{window_type}.png"
    plot_detailed_statistics(window_stats, detailed_plot_file, window_type)
    
    # Compute active users
    active_users = compute_active_users_by_window(df, window_type, show_progress)
    
    # Compute active user statistics
    active_users_stats = compute_active_users_statistics(active_users, show_progress)
    
    # Save active user statistics
    active_users_file = output_dir / f"active_users_stats_{window_type}.csv"
    save_statistics_to_file(active_users_stats, active_users_file, window_type, "active_users")
    
    # Plot active users
    active_users_plot_file = output_dir / f"active_users_over_time_{window_type}.png"
    plot_active_users_over_time(active_users_stats, active_users_plot_file, window_type)
    
    logger.info(f"Analysis complete for {window_type} window")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Amazon Reviews dataset across different time windows"
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
        '--window',
        type=str,
        choices=['30min', '1h', '12h', '1d'],
        help='Single time window to analyze'
    )
    parser.add_argument(
        '--all_windows',
        action='store_true',
        help='Analyze all time windows (30min, 1h, 12h, 1d)'
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
        help='Number of worker processes (default: auto)'
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
    
    # Determine which windows to analyze
    if args.all_windows:
        windows = ['30min', '1h', '12h', '1d']
    elif args.window:
        windows = [args.window]
    else:
        # Default: analyze all windows
        windows = ['30min', '1h', '12h', '1d']
        logger.info("No window specified, analyzing all windows by default")
    
    # Analyze each window
    for window_type in windows:
        try:
            analyze_time_window(df, window_type, output_dir, args.show_progress)
        except Exception as e:
            logger.error(f"Error analyzing {window_type} window: {e}", exc_info=True)
    
    logger.info("\n" + "="*60)
    logger.info("All analyses complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


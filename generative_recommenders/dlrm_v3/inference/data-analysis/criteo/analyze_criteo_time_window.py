#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Criteo dataset user interaction sequence length distribution and active user count distribution
across different time windows (30min, 1h, 12h, 1d)
Uses memory mapping for efficient loading of large files
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


def load_criteo_data(data_path: str, use_mmap: bool = True, show_progress: bool = False) -> pd.DataFrame:
    """
    Load Criteo dataset efficiently using memory mapping
    
    Args:
        data_path: Path to the data directory or processed npz file
        use_mmap: Whether to use memory mapping for large files
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with columns: user_id, timestamp, datetime
    """
    logger.info(f"Loading Criteo data from: {data_path}")
    start_time = time.time()
    
    data_path_obj = Path(data_path)
    
    # Check if it's a directory with day files or a single processed file
    if data_path_obj.is_dir():
        # First, try to find raw day files (without .npz extension)
        day_raw_files = sorted([f for f in data_path_obj.glob("day_*") if not f.name.endswith('.npz') and f.is_file()])
        
        if day_raw_files:
            logger.info(f"Found {len(day_raw_files)} raw day files, loading them...")
            return load_criteo_from_day_files(day_raw_files, show_progress)
        
        # If no raw files, look for processed npz files
        day_npz_files = sorted(data_path_obj.glob("day_*.npz"))
        if day_npz_files:
            logger.info(f"Found {len(day_npz_files)} npz day files, loading them...")
            return load_criteo_from_npz_files(day_npz_files, use_mmap, show_progress)
        
        # Look for terabyte_processed.npz
        processed_file = data_path_obj / "terabyte_processed.npz"
        if processed_file.exists():
            logger.info(f"Found processed file: {processed_file}")
            return load_criteo_from_npz_files([processed_file], use_mmap, show_progress)
        
        raise FileNotFoundError(f"No data files found in {data_path}")
    
    # If it's a file, try to load it
    elif data_path_obj.is_file():
        if data_path_obj.suffix == '.npz':
            return load_criteo_from_npz_files([data_path_obj], use_mmap, show_progress)
        else:
            # Assume it's a raw day file
            return load_criteo_from_day_files([data_path_obj], show_progress)
    
    raise FileNotFoundError(f"Data path not found: {data_path}")


def load_criteo_from_npz_files(npz_files: List[Path], use_mmap: bool = True, show_progress: bool = False) -> pd.DataFrame:
    """
    Load Criteo data from npz files using memory mapping
    
    Args:
        npz_files: List of npz file paths
        use_mmap: Whether to use memory mapping
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with user_id, timestamp, datetime columns
    """
    logger.info(f"Loading data from {len(npz_files)} npz files...")
    start_time = time.time()
    
    all_records = []
    chunk_size = 1000000  # Process 1M records at a time
    
    for file_idx, npz_file in enumerate(npz_files):
        logger.info(f"Processing npz file {file_idx + 1}/{len(npz_files)}: {npz_file.name}")
        
        try:
            # Load with memory mapping if requested
            if use_mmap:
                logger.info(f"  Loading with memory mapping...")
                data = np.load(str(npz_file), mmap_mode='r')
            else:
                data = np.load(str(npz_file), allow_pickle=True)
            
            available_keys = list(data.keys())
            logger.info(f"  Available keys: {available_keys}")
            
            # Try to extract data from common Criteo npz structure
            # Common structure: X_int (integer features), X_cat (categorical features), y (labels)
            # We'll use categorical features to create user_id and use array indices for timestamp
            
            if 'X_cat' in available_keys:
                X_cat = data['X_cat']
                num_samples = len(X_cat)
                logger.info(f"  Found X_cat with shape: {X_cat.shape}, {num_samples:,} samples")
                
                # Create user_id from first categorical feature (or hash multiple features)
                # Process in chunks to avoid memory issues
                chunk_records = []
                
                if show_progress and HAS_TQDM:
                    pbar = tqdm(total=num_samples, desc=f"Processing {npz_file.name}", unit="samples")
                else:
                    pbar = None
                
                for i in range(num_samples):
                    # Extract user_id from categorical features
                    if X_cat.ndim == 2:
                        # Multiple categorical features per sample
                        cat_features = X_cat[i]
                        # Use first few features to create user_id
                        user_id_str = '_'.join([str(int(cat_features[j])) for j in range(min(3, len(cat_features)))])
                        user_id = hash(user_id_str) % (2**31)
                    else:
                        # Single categorical feature
                        user_id = int(X_cat[i]) % (2**31)
                    
                    # Create timestamp: file index * large_offset + sample index
                    # Distribute samples across the day
                    base_timestamp = file_idx * 86400  # Each file represents a day
                    timestamp = base_timestamp + (i * 86400 // max(1, num_samples))
                    
                    chunk_records.append({
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'day': file_idx
                    })
                    
                    if pbar:
                        pbar.update(1)
                    
                    # Process in chunks
                    if len(chunk_records) >= chunk_size:
                        all_records.extend(chunk_records)
                        chunk_records = []
                        if show_progress:
                            logger.info(f"  Processed {len(all_records):,} records so far...")
                
                if chunk_records:
                    all_records.extend(chunk_records)
                
                if pbar:
                    pbar.close()
            
            else:
                # Try to find any array that might represent samples
                logger.warning(f"  X_cat not found, trying to infer structure from available keys...")
                # Use first available array as sample count
                sample_count = None
                for key in available_keys:
                    arr = data[key]
                    if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                        sample_count = len(arr)
                        logger.info(f"  Using {key} with shape {arr.shape} as sample reference")
                        break
                
                if sample_count is None:
                    logger.error(f"  Could not determine sample count from {available_keys}")
                    data.close()
                    continue
                
                # Create synthetic user_ids and timestamps
                chunk_records = []
                for i in range(sample_count):
                    user_id = i % 1000000  # Distribute users
                    base_timestamp = file_idx * 86400
                    timestamp = base_timestamp + (i * 86400 // max(1, sample_count))
                    
                    chunk_records.append({
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'day': file_idx
                    })
                    
                    if len(chunk_records) >= chunk_size:
                        all_records.extend(chunk_records)
                        chunk_records = []
            
            data.close()
            logger.info(f"  Completed {npz_file.name}: {len([r for r in all_records if r['day'] == file_idx]):,} records")
        
        except Exception as e:
            logger.error(f"Error processing {npz_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Convert to DataFrame
    if not all_records:
        logger.error("No records loaded from npz files")
        return pd.DataFrame()
    
    logger.info(f"Converting {len(all_records):,} records to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Convert timestamp to datetime
    reference_date = datetime(2014, 1, 1)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', origin=reference_date)
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading completed: {len(df):,} records in {elapsed:.2f} seconds")
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Unique users: {df['user_id'].nunique():,}")
    
    return df


def load_criteo_from_day_files(day_files: List[Path], show_progress: bool = False) -> pd.DataFrame:
    """
    Load Criteo data from raw day files
    
    Criteo day files format: Each line contains tab-separated values
    Format: <label> <integer_feature_1> ... <integer_feature_13> <categorical_feature_1> ... <categorical_feature_26>
    We need to extract user_id (usually categorical feature) and timestamp (if available)
    
    For Criteo Terabyte, we typically don't have explicit timestamps in the raw data.
    We'll use the day number and line number as a proxy for time ordering.
    """
    logger.info(f"Loading data from {len(day_files)} day files...")
    start_time = time.time()
    
    all_records = []
    chunk_size = 1000000  # Process 1M lines at a time
    
    for day_idx, day_file in enumerate(day_files):
        logger.info(f"Processing day file {day_idx + 1}/{len(day_files)}: {day_file.name}")
        
        try:
            # Read file in chunks
            chunk_records = []
            line_count = 0
            
            with open(day_file, 'r') as f:
                # Estimate total lines
                if show_progress and HAS_TQDM:
                    total_lines = sum(1 for _ in f)
                    f.seek(0)
                    pbar = tqdm(total=total_lines, desc=f"Reading {day_file.name}", unit="lines")
                else:
                    pbar = None
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    # Extract user_id from categorical features (typically last 26 features)
                    # User ID is often in one of the categorical features
                    # For analysis, we'll use the first categorical feature as user_id proxy
                    # Or we can hash multiple features to create a user_id
                    if len(parts) >= 14:  # At least label + 13 int + some cat features
                        # Use a combination of categorical features as user_id
                        # Hash the categorical features to create a user_id
                        cat_features = parts[14:] if len(parts) > 14 else []
                        if cat_features:
                            # Create user_id from hash of categorical features
                            user_id_str = '_'.join(cat_features[:3])  # Use first 3 cat features
                            user_id = hash(user_id_str) % (2**31)  # Convert to int32 range
                        else:
                            # Fallback: use line number as user_id
                            user_id = line_count % 1000000
                        
                        # Create timestamp: day number * 86400 + line offset (as seconds)
                        # This gives us a time ordering
                        base_timestamp = day_idx * 86400  # Each day is 86400 seconds
                        timestamp = base_timestamp + (line_count % 86400)  # Distribute within day
                        
                        chunk_records.append({
                            'user_id': user_id,
                            'timestamp': timestamp,
                            'day': day_idx
                        })
                        
                        line_count += 1
                        
                        if pbar:
                            pbar.update(1)
                        
                        # Process in chunks
                        if len(chunk_records) >= chunk_size:
                            all_records.extend(chunk_records)
                            chunk_records = []
                            if show_progress:
                                logger.info(f"  Processed {len(all_records):,} records so far...")
                
                if chunk_records:
                    all_records.extend(chunk_records)
                
                if pbar:
                    pbar.close()
                
                logger.info(f"  Completed {day_file.name}: {line_count:,} lines processed")
        
        except Exception as e:
            logger.error(f"Error processing {day_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    logger.info(f"Converting {len(all_records):,} records to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Convert timestamp to datetime
    # Criteo dataset doesn't have real timestamps, so we'll create a synthetic timeline
    # starting from a reference date
    reference_date = datetime(2014, 1, 1)  # Approximate Criteo dataset start date
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', origin=reference_date)
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading completed: {len(df):,} records in {elapsed:.2f} seconds")
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Unique users: {df['user_id'].nunique():,}")
    
    return df


def create_time_window(datetime_series: pd.Series, window_type: str) -> pd.Series:
    """
    Create time window labels for datetime series
    
    Args:
        datetime_series: Series of datetime values
        window_type: Type of time window ('30min', '1h', '12h', '1d')
    
    Returns:
        Series of time window labels (datetime)
    """
    if window_type == '30min':
        return datetime_series.dt.floor('30min')
    elif window_type == '1h':
        return datetime_series.dt.floor('1h')
    elif window_type == '12h':
        return datetime_series.dt.floor('12h')
    elif window_type == '1d':
        return datetime_series.dt.floor('1d')
    else:
        raise ValueError(f"Unsupported window_type: {window_type}")


def compute_user_sequence_lengths_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute cumulative sequence length for each user in each time window
    
    Args:
        df: DataFrame with datetime and user_id columns
        window_type: Type of time window
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with time_window, user_id, and sequence_length columns
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"Computing user sequence lengths for {window_type} time windows...")
    
    # Create time window labels
    time_window = create_time_window(df["datetime"], window_type)
    
    # Count interactions per user per window
    window_interactions = df.groupby([time_window, 'user_id'], observed=True, sort=False).size().reset_index(name='window_count')
    window_interactions.columns = ['time_window', 'user_id', 'window_count']
    
    # Sort by time window and user_id
    window_interactions = window_interactions.sort_values(['time_window', 'user_id'])
    
    if show_progress:
        logger.info("Computing cumulative sequence lengths...")
    
    # For each user, compute cumulative sum sorted by time window
    window_interactions['sequence_length'] = window_interactions.groupby('user_id', observed=True, sort=False)['window_count'].cumsum()
    
    result = window_interactions[['time_window', 'user_id', 'sequence_length']]
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Cumulative sequence length computation completed: {len(result):,} records in {elapsed:.2f} seconds")
    
    return result


def compute_active_users_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute number of active users per time window
    
    Args:
        df: DataFrame with datetime and user_id columns
        window_type: Type of time window
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with time_window and active_user_count columns
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"Computing active users for {window_type} time windows...")
    
    # Create time window labels
    time_window = create_time_window(df["datetime"], window_type)
    
    # Count unique users per window
    active_users = df.groupby(time_window, observed=True, sort=True)['user_id'].nunique().reset_index()
    active_users.columns = ['time_window', 'active_user_count']
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Active user count computation completed: {len(active_users)} time windows in {elapsed:.2f} seconds")
    
    return active_users


def compute_window_statistics(
    df: pd.DataFrame,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Compute statistical metrics for each time window
    
    Returns:
        DataFrame with statistical metrics per time window
    """
    start_time = time.time()
    if show_progress:
        logger.info("Computing window statistics...")
        unique_windows = df["time_window"].nunique()
        logger.info(f"Computing statistics for {unique_windows} time windows")
    
    # Group by time window and compute statistics
    grouped = df.groupby("time_window", observed=True, sort=True)["sequence_length"]
    
    # Compute basic statistics
    count_df = grouped.count().reset_index(name='count')
    mean_df = grouped.mean().reset_index(name='mean')
    std_df = grouped.std().reset_index(name='std')
    min_df = grouped.min().reset_index(name='min')
    max_df = grouped.max().reset_index(name='max')
    median_df = grouped.median().reset_index(name='median')
    
    # Merge all statistics
    stats = count_df.merge(mean_df, on='time_window')
    stats = stats.merge(std_df, on='time_window')
    stats = stats.merge(min_df, on='time_window')
    stats = stats.merge(max_df, on='time_window')
    stats = stats.merge(median_df, on='time_window')
    
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
        for p in percentiles_to_compute:
            col_name = f"p{int(p*100)}"
            if col_name not in stats.columns:
                quantile_series = grouped.quantile(p)
                stats[col_name] = stats['time_window'].map(quantile_series).astype(float)
    
    # Ensure p25 and p75 exist
    if 'p25' not in stats.columns:
        stats['p25'] = stats['time_window'].map(grouped.quantile(0.25)).astype(float)
    if 'p75' not in stats.columns:
        stats['p75'] = stats['time_window'].map(grouped.quantile(0.75)).astype(float)
    
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
        logger.warning(f"Error computing skewness and kurtosis: {e}, setting to 0")
        stats['skewness'] = 0.0
        stats['kurtosis'] = 0.0
    
    # Ensure all numeric columns are float
    numeric_cols = stats.select_dtypes(include=[np.number]).columns
    stats[numeric_cols] = stats[numeric_cols].astype(float)
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Statistics computation completed in {elapsed:.2f} seconds")
    
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
    
    # Compute rolling statistics (optional)
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
        logger.info(f"Active user statistics computation completed in {elapsed:.2f} seconds")
    
    return stats


def plot_sequence_length_over_time(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot sequence length statistics over time windows"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
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
    
    # 2. Variance and standard deviation
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
    
    # 3. Min and max
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
    
    # 4. IQR
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


def plot_detailed_statistics(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Plot detailed statistical information"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    x = window_stats["time_window"]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Detailed Statistical Analysis of User Interaction Sequence Length - {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    # 1. All percentiles comparison
    ax1 = axes[0, 0]
    percentiles = ["p50", "p75", "p90", "p95", "p99"]
    try:
        colors = plt.cm.tab10(np.linspace(0, 1, len(percentiles)))
    except AttributeError:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(percentiles)]
    for i, p in enumerate(percentiles):
        if p in window_stats.columns:
            ax1.plot(x, window_stats[p], label=p.upper(), marker='o', markersize=3, linewidth=1.5, alpha=0.8, color=colors[i % len(colors)])
    ax1.set_xlabel("Time Window", fontsize=10)
    ax1.set_ylabel("Sequence Length", fontsize=10)
    ax1.set_title("Sequence Length Percentiles Comparison", fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Skewness and kurtosis
    ax2 = axes[0, 1]
    if "skewness" in window_stats.columns:
        ax2.plot(x, window_stats["skewness"], label="Skewness", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    if "kurtosis" in window_stats.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, window_stats["kurtosis"], label="Kurtosis", marker='s', markersize=4, linewidth=2, alpha=0.8, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel("Time Window", fontsize=10)
    ax2.set_ylabel("Skewness", fontsize=10, color='blue')
    if "kurtosis" in window_stats.columns:
        ax2_twin.set_ylabel("Kurtosis", fontsize=10, color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2_twin.legend(loc='upper right', fontsize=9)
    ax2.set_title("Sequence Length Distribution Skewness and Kurtosis", fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    if mdates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. User count per window
    ax3 = axes[1, 0]
    ax3.plot(x, window_stats["count"], label="Active User Count", marker='o', markersize=4, linewidth=2, alpha=0.8, color='green')
    ax3.set_xlabel("Time Window", fontsize=10)
    ax3.set_ylabel("User Count", fontsize=10)
    ax3.set_title("Active User Count per Time Window", fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Mean ± std dev region
    ax4 = axes[1, 1]
    mean_vals = window_stats["mean"]
    std_vals = window_stats["std"]
    ax4.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color='blue', label='±1 Std Dev')
    ax4.plot(x, mean_vals, label="Mean", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    ax4.set_xlabel("Time Window", fontsize=10)
    ax4.set_ylabel("Sequence Length", fontsize=10)
    ax4.set_title("Mean ± Std Dev Region", fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    if mdates is not None:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Coefficient of variation
    ax5 = axes[2, 0]
    cv = window_stats["std"] / window_stats["mean"].replace(0, np.nan)
    ax5.plot(x, cv, label="Coefficient of Variation (CV)", marker='o', markersize=4, linewidth=2, alpha=0.8, color='orange')
    ax5.set_xlabel("Time Window", fontsize=10)
    ax5.set_ylabel("Coefficient of Variation", fontsize=10)
    ax5.set_title("Sequence Length Coefficient of Variation (Std Dev / Mean)", fontsize=12)
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    if mdates is not None:
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Box plot style summary
    ax6 = axes[2, 1]
    if "p25" in window_stats.columns and "p75" in window_stats.columns:
        ax6.fill_between(x, window_stats["p25"], window_stats["p75"], alpha=0.3, color='gray', label='IQR (P25-P75)')
        ax6.plot(x, window_stats["median"], label="Median", marker='o', markersize=4, linewidth=2, alpha=0.8, color='black')
    ax6.set_xlabel("Time Window", fontsize=10)
    ax6.set_ylabel("Sequence Length", fontsize=10)
    ax6.set_title("Sequence Length Distribution Spread (IQR)", fontsize=12)
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    if mdates is not None:
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_active_users_over_time(
    active_users_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot active user count statistics over time"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    active_users_stats = active_users_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(active_users_stats["time_window"]):
        active_users_stats["time_window"] = pd.to_datetime(active_users_stats["time_window"])
    
    active_users_stats = active_users_stats.sort_values("time_window")
    x = active_users_stats["time_window"]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Active User Count Distribution Statistics - {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    # 1. Active user count over time
    ax1 = axes[0, 0]
    ax1.plot(x, active_users_stats["active_user_count"], label="Active User Count", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    if "active_user_count_mean_rolling" in active_users_stats.columns:
        ax1.plot(x, active_users_stats["active_user_count_mean_rolling"], label="Rolling Mean", marker='s', markersize=3, linewidth=2, alpha=0.8, color='red', linestyle='--')
    ax1.set_xlabel("Time Window", fontsize=10)
    ax1.set_ylabel("Active User Count", fontsize=10)
    ax1.set_title("Active User Count Over Time", fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Statistics summary
    ax2 = axes[0, 1]
    if "mean" in active_users_stats.columns:
        ax2.axhline(y=active_users_stats["mean"].iloc[0], label=f"Mean: {active_users_stats['mean'].iloc[0]:,.0f}", 
                   color='green', linewidth=2, linestyle='-')
    if "median" in active_users_stats.columns:
        ax2.axhline(y=active_users_stats["median"].iloc[0], label=f"Median: {active_users_stats['median'].iloc[0]:,.0f}", 
                   color='orange', linewidth=2, linestyle='--')
    if "p90" in active_users_stats.columns:
        ax2.axhline(y=active_users_stats["p90"].iloc[0], label=f"P90: {active_users_stats['p90'].iloc[0]:,.0f}", 
                   color='red', linewidth=2, linestyle=':')
    if "p99" in active_users_stats.columns:
        ax2.axhline(y=active_users_stats["p99"].iloc[0], label=f"P99: {active_users_stats['p99'].iloc[0]:,.0f}", 
                   color='purple', linewidth=2, linestyle='-.')
    ax2.plot(x, active_users_stats["active_user_count"], marker='o', markersize=3, linewidth=1, alpha=0.6, color='blue')
    ax2.set_xlabel("Time Window", fontsize=10)
    ax2.set_ylabel("Active User Count", fontsize=10)
    ax2.set_title("Active User Count with Statistical Thresholds", fontsize=12)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Variance and std dev
    ax3 = axes[1, 0]
    if "variance" in active_users_stats.columns:
        ax3.axhline(y=active_users_stats["variance"].iloc[0], label=f"Variance: {active_users_stats['variance'].iloc[0]:,.0f}", 
                   color='orange', linewidth=2, linestyle='-')
    if "std" in active_users_stats.columns:
        ax3.axhline(y=active_users_stats["std"].iloc[0], label=f"Std Dev: {active_users_stats['std'].iloc[0]:,.0f}", 
                   color='green', linewidth=2, linestyle='--')
    ax3.plot(x, active_users_stats["active_user_count"], marker='o', markersize=3, linewidth=1, alpha=0.6, color='blue')
    ax3.set_xlabel("Time Window", fontsize=10)
    ax3.set_ylabel("Active User Count", fontsize=10)
    ax3.set_title("Active User Count Variance and Std Dev", fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Min and max
    ax4 = axes[1, 1]
    if "min" in active_users_stats.columns:
        ax4.axhline(y=active_users_stats["min"].iloc[0], label=f"Min: {active_users_stats['min'].iloc[0]:,.0f}", 
                   color='blue', linewidth=2, linestyle='-')
    if "max" in active_users_stats.columns:
        ax4.axhline(y=active_users_stats["max"].iloc[0], label=f"Max: {active_users_stats['max'].iloc[0]:,.0f}", 
                   color='red', linewidth=2, linestyle='-')
    ax4.plot(x, active_users_stats["active_user_count"], marker='o', markersize=3, linewidth=1, alpha=0.6, color='gray')
    ax4.set_xlabel("Time Window", fontsize=10)
    ax4.set_ylabel("Active User Count", fontsize=10)
    ax4.set_title("Active User Count Min and Max", fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_sequence_length_by_time_window_from_df(
    df: pd.DataFrame,
    window_type: str = "1d",
    show_progress: bool = False,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze user interaction sequence length distribution and active user count distribution
    from an already loaded DataFrame
    
    Returns:
        (user sequence lengths DataFrame, window statistics DataFrame, active users statistics DataFrame)
    """
    total_start = time.time()
    logger.info("="*80)
    logger.info("Starting analysis of user interaction sequence length and active user count distribution")
    logger.info(f"Time window type: {window_type}")
    logger.info("="*80)
    
    # Compute sequence lengths per user per window
    user_seq_lengths = compute_user_sequence_lengths_by_window(
        df, window_type, show_progress
    )
    
    # Compute statistics per window
    window_stats = compute_window_statistics(user_seq_lengths, percentiles, show_progress)
    
    # Compute active users
    active_users = compute_active_users_by_window(df, window_type, show_progress)
    active_users_stats = compute_active_users_statistics(active_users, show_progress)
    
    total_elapsed = time.time() - total_start
    logger.info("="*80)
    logger.info(f"Analysis completed! Total time: {total_elapsed:.2f} seconds")
    logger.info("="*80)
    
    return user_seq_lengths, window_stats, active_users_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Criteo dataset user interaction sequence length and active user count distribution across different time windows'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/comp/cswjyu/downloads/criteo-tb/days',
        help='Path to data directory or processed npz file (default: /home/comp/cswjyu/downloads/criteo-tb/days)'
    )
    parser.add_argument(
        '--window_type',
        type=str,
        choices=['30min', '1h', '12h', '1d'],
        default='1d',
        help='Time window type: 30min/1h/12h/1d (default: 1d)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: reports folder in script directory)'
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='Show progress information'
    )
    parser.add_argument(
        '--all_windows',
        action='store_true',
        help='Analyze all time window types'
    )
    parser.add_argument(
        '--no_mmap',
        action='store_true',
        help='Disable memory mapping for large files (memory mapping is enabled by default)'
    )
    
    args = parser.parse_args()
    
    # Expand path
    data_path = os.path.expanduser(args.data_path)
    
    # Set output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "reports"
    else:
        output_dir = Path(os.path.expanduser(args.output_dir))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Determine window types to analyze
    if args.all_windows:
        window_types = ['30min', '1h', '12h', '1d']
    else:
        window_types = [args.window_type]
    
    # Determine if memory mapping should be used (enabled by default)
    use_mmap = not args.no_mmap
    
    # Load data
    logger.info("="*80)
    logger.info("Loading Criteo dataset...")
    logger.info("="*80)
    df = load_criteo_data(data_path, use_mmap=use_mmap, show_progress=args.show_progress)
    
    if df.empty:
        logger.error("Failed to load data. Please check the data path and format.")
        return
    
    logger.info(f"Loaded {len(df):,} records")
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info("="*80)
    
    # Analyze each time window
    for window_type in window_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting analysis for time window: {window_type}")
        logger.info(f"{'='*80}\n")
        
        # Perform analysis
        user_seq_lengths, window_stats, active_users_stats = analyze_sequence_length_by_time_window_from_df(
            df,
            window_type=window_type,
            show_progress=args.show_progress,
        )
        
        # Save CSV files
        logger.info("Saving results...")
        stats_csv = output_dir / f"sequence_length_stats_{window_type}.csv"
        window_stats.to_csv(stats_csv, index=False)
        logger.info(f"Window statistics saved to: {stats_csv}")
        
        active_users_csv = output_dir / f"active_users_stats_{window_type}.csv"
        active_users_stats.to_csv(active_users_csv, index=False)
        logger.info(f"Active user statistics saved to: {active_users_csv}")
        
        # Save JSON
        stats_json = output_dir / f"sequence_length_stats_{window_type}.json"
        stats_dict = window_stats.copy()
        stats_dict["time_window"] = stats_dict["time_window"].astype(str)
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump(stats_dict.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        logger.info(f"Window statistics JSON saved to: {stats_json}")
        
        # Print summary statistics
        logger.info("\n" + "="*80)
        logger.info(f"Time window type: {window_type}")
        logger.info(f"Number of time windows: {len(window_stats)}")
        logger.info("="*80)
        logger.info("\nSequence length overall statistics:")
        logger.info(f"  Mean: {window_stats['mean'].mean():.2f}")
        logger.info(f"  Std Dev: {window_stats['std'].mean():.2f}")
        logger.info(f"  Median: {window_stats['median'].mean():.2f}")
        if "p90" in window_stats.columns:
            logger.info(f"  P90: {window_stats['p90'].mean():.2f}")
        if "p99" in window_stats.columns:
            logger.info(f"  P99: {window_stats['p99'].mean():.2f}")
        logger.info(f"  Mean Variance: {window_stats['variance'].mean():.2f}")
        
        logger.info("\nActive user count overall statistics:")
        logger.info(f"  Mean active users: {active_users_stats['active_user_count'].mean():.2f}")
        logger.info(f"  Std Dev: {active_users_stats['active_user_count'].std():.2f}")
        logger.info(f"  Median: {active_users_stats['active_user_count'].median():.2f}")
        logger.info(f"  Min: {active_users_stats['active_user_count'].min():.0f}")
        logger.info(f"  Max: {active_users_stats['active_user_count'].max():.0f}")
        if "p90" in active_users_stats.columns:
            logger.info(f"  P90: {active_users_stats['p90'].iloc[0]:.0f}")
        if "p99" in active_users_stats.columns:
            logger.info(f"  P99: {active_users_stats['p99'].iloc[0]:.0f}")
        
        # Generate plots
        if HAS_MATPLOTLIB:
            logger.info("\nGenerating plots...")
            
            # Sequence length statistics plots
            plot_main = output_dir / f"sequence_length_over_time_{window_type}.png"
            plot_sequence_length_over_time(window_stats, plot_main, window_type)
            logger.info(f"Main statistics plot saved to: {plot_main}")
            
            plot_detailed = output_dir / f"sequence_length_detailed_stats_{window_type}.png"
            plot_detailed_statistics(window_stats, plot_detailed, window_type)
            logger.info(f"Detailed statistics plot saved to: {plot_detailed}")
            
            # Active user count plots
            plot_active_users = output_dir / f"active_users_over_time_{window_type}.png"
            plot_active_users_over_time(active_users_stats, plot_active_users, window_type)
            logger.info(f"Active user statistics plot saved to: {plot_active_users}")
        else:
            logger.warning("matplotlib not installed, skipping plots")
        
        logger.info(f"\nTime window {window_type} analysis completed!\n")


if __name__ == '__main__':
    main()


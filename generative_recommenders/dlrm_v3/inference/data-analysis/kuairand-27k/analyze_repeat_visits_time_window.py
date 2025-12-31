#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze repeat visits within time windows for KuaiRand dataset
- Count users with repeat visits in each time window (5min, 10min, 30min, 1h)
- Count how many times each user repeats visits in each time window
- Visualize the statistics with English titles
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


def load_kuairand_data(
    data_dir: Path,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Load KuaiRand dataset from CSV files
    
    Args:
        data_dir: Directory containing KuaiRand log CSV files
        show_progress: Whether to show progress
    
    Returns:
        DataFrame with columns: user_id, video_id, datetime
    """
    logger.info(f"Loading KuaiRand data from: {data_dir}")
    start_time = time.time()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all log files
    log_paths = sorted(data_dir.glob("log_standard*.csv"))
    if not log_paths:
        raise FileNotFoundError(
            f"No KuaiRand log files found in {data_dir}. "
            "Expected files like 'log_standard_*.csv'."
        )
    
    logger.info(f"Found {len(log_paths)} log files")
    
    # Check if cached parquet file exists and is valid
    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "kuairand_processed.parquet"
    
    # Check if cache is valid by comparing modification times
    cache_valid = False
    if cache_file.exists():
        try:
            cache_mtime = cache_file.stat().st_mtime
            # Check if cache is newer than all source files
            source_files_newer = False
            for csv_path in log_paths:
                if csv_path.stat().st_mtime > cache_mtime:
                    source_files_newer = True
                    break
            
            if not source_files_newer:
                cache_valid = True
                logger.info(f"Found valid cached parquet file: {cache_file}")
                logger.info("Loading from cache...")
                df = pd.read_parquet(cache_file)
                elapsed = time.time() - start_time
                logger.info(f"Data loaded from cache: {len(df):,} rows, elapsed {elapsed:.2f} seconds")
                return df
            else:
                logger.info("Cache is outdated (source files have been modified), will regenerate cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will reload from source")
    
    # Load CSV files
    frames = []
    iterable = log_paths
    if HAS_TQDM and show_progress:
        iterable = tqdm(log_paths, desc="Reading CSV files", unit="file")
    
    for csv_path in iterable:
        try:
            frame = pd.read_csv(
                csv_path,
                dtype={
                    'user_id': 'int32',
                    'video_id': 'int32',
                    'time_ms': 'int64',
                },
                engine='c',
                low_memory=False,
            )
        except (ValueError, TypeError, KeyError):
            # If dtype specification fails, use default reading
            frame = pd.read_csv(csv_path, engine='c', low_memory=False)
        
        frame["__source_file"] = csv_path.name
        frames.append(frame)
    
    logger.info("Concatenating data frames...")
    df = pd.concat(frames, ignore_index=True, sort=False)
    
    # Check required columns
    required_cols = {"user_id", "video_id", "time_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # Convert time_ms to datetime
    logger.info("Converting time_ms to datetime...")
    df['datetime'] = pd.to_datetime(df['time_ms'], unit='ms', errors='coerce')
    
    # Remove invalid timestamps
    invalid_count = df['datetime'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid timestamp records, removing them")
        df = df.dropna(subset=['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save to cache
    try:
        logger.info(f"Saving to cache: {cache_file}")
        # Use fastparquet or pyarrow engine if available, fallback to default
        try:
            df.to_parquet(cache_file, compression='snappy', index=False, engine='pyarrow')
        except Exception:
            # Fallback to default engine
            df.to_parquet(cache_file, compression='snappy', index=False)
        logger.info(f"Cache saved successfully ({cache_file.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading complete: {len(df):,} rows")
    logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Unique users: {df['user_id'].nunique():,}")
    logger.info(f"Unique videos: {df['video_id'].nunique():,}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    
    return df


def analyze_repeat_visits_time_window(
    df: pd.DataFrame,
    window_size: str,
    show_progress: bool = False
) -> Dict:
    """
    Analyze repeat visits within a specific time window
    
    Args:
        df: DataFrame with columns: user_id, datetime, video_id
        window_size: Time window size (e.g., '5min', '10min', '30min', '1h')
        show_progress: Whether to show progress
    
    Returns:
        Dictionary with statistics
    """
    logger.info(f"\nAnalyzing repeat visits for time window: {window_size}")
    start_time = time.time()
    
    # Convert window size to timedelta
    window_map = {
        '5min': pd.Timedelta(minutes=5),
        '10min': pd.Timedelta(minutes=10),
        '30min': pd.Timedelta(minutes=30),
        '1h': pd.Timedelta(hours=1)
    }
    
    if window_size not in window_map:
        raise ValueError(f"Invalid window size: {window_size}. Must be one of {list(window_map.keys())}")
    
    window_td = window_map[window_size]
    
    # Create time window bins
    min_time = df['datetime'].min()
    max_time = df['datetime'].max()
    
    # Round down to the nearest window boundary
    min_time_rounded = min_time.floor(freq=window_size)
    
    # Create time window labels using vectorized operation
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['time_window'] = (df['datetime'] - min_time_rounded) // window_td
    
    num_windows = df['time_window'].nunique()
    if show_progress:
        logger.info(f"Created {num_windows:,} time windows")
    
    # Vectorized approach: group by time_window and user_id at once
    if show_progress:
        logger.info("Computing visit counts per window per user...")
    
    # Count visits per (time_window, user_id) pair - this is the key optimization
    window_user_visits = df.groupby(['time_window', 'user_id'], observed=True).size().reset_index(name='visit_count')
    
    # Calculate repeat visit counts (visit_count - 1, only for repeat users)
    window_user_visits['repeat_visit_count'] = (window_user_visits['visit_count'] - 1).clip(lower=0)
    
    # Filter to only repeat users (visit_count > 1)
    repeat_window_user = window_user_visits[window_user_visits['visit_count'] > 1].copy()
    
    # Aggregate statistics per time window
    if show_progress:
        logger.info("Aggregating statistics per time window...")
    
    # Window-level statistics
    window_stats = window_user_visits.groupby('time_window', observed=True).agg({
        'user_id': 'count',  # Total users per window
        'visit_count': 'sum'  # Total visits per window
    }).rename(columns={'user_id': 'num_total_users', 'visit_count': 'total_visits_in_window'})
    
    # Repeat user statistics per window
    if len(repeat_window_user) > 0:
        repeat_window_stats = repeat_window_user.groupby('time_window', observed=True).agg({
            'user_id': 'count',  # Number of repeat users
            'repeat_visit_count': ['sum', 'mean', 'max']  # Repeat visit statistics
        })
        repeat_window_stats.columns = ['num_repeat_users', 'total_repeat_visits', 'avg_repeat_visits_per_repeat_user', 'max_repeat_visits']
    else:
        # Create empty DataFrame with same index as window_stats
        repeat_window_stats = pd.DataFrame(index=window_stats.index)
        repeat_window_stats['num_repeat_users'] = 0
        repeat_window_stats['total_repeat_visits'] = 0
        repeat_window_stats['avg_repeat_visits_per_repeat_user'] = 0.0
        repeat_window_stats['max_repeat_visits'] = 0
    
    # Merge window statistics
    window_stats = window_stats.join(repeat_window_stats, how='left').fillna(0)
    window_stats['num_repeat_users'] = window_stats['num_repeat_users'].astype(int)
    window_stats['total_repeat_visits'] = window_stats['total_repeat_visits'].astype(int)
    window_stats['max_repeat_visits'] = window_stats['max_repeat_visits'].astype(int)
    window_stats['num_repeat_users_pct'] = (window_stats['num_repeat_users'] / window_stats['num_total_users'] * 100).fillna(0)
    
    # Create window_results with time_window_start
    window_stats_reset = window_stats.reset_index()
    window_stats_reset['time_window_start'] = min_time_rounded + window_stats_reset['time_window'] * window_td
    
    # Convert to list of dicts for window_results
    results_df = window_stats_reset.copy()
    results_df['time_window'] = results_df['time_window'].astype(int)
    
    # Overall statistics
    total_windows = len(results_df)
    total_repeat_users_all_windows = results_df['num_repeat_users'].sum()
    avg_repeat_users_per_window = results_df['num_repeat_users'].mean()
    total_repeat_visits_all_windows = results_df['total_repeat_visits'].sum()
    avg_repeat_visits_per_window = results_df['total_repeat_visits'].mean()
    avg_total_visits_per_window = results_df['total_visits_in_window'].mean()
    min_total_visits_per_window = results_df['total_visits_in_window'].min()
    max_total_visits_per_window = results_df['total_visits_in_window'].max()
    total_visits_all_windows = results_df['total_visits_in_window'].sum()
    
    # User-level statistics: count how many times each user had repeat visits (vectorized)
    if show_progress:
        logger.info("Computing user-level statistics...")
    
    if len(repeat_window_user) > 0:
        user_repeat_window_counts_series = repeat_window_user.groupby('user_id', observed=True)['time_window'].count()
        user_repeat_visit_counts_series = repeat_window_user.groupby('user_id', observed=True)['repeat_visit_count'].sum()
        
        user_repeat_window_counts = user_repeat_window_counts_series.to_dict()
        user_repeat_visit_counts = user_repeat_visit_counts_series.to_dict()
    else:
        user_repeat_window_counts = {}
        user_repeat_visit_counts = {}
    
    # Statistics about users who had repeat visits
    num_users_with_repeats = len(user_repeat_window_counts)
    if num_users_with_repeats > 0:
        avg_windows_per_user = np.mean(list(user_repeat_window_counts.values()))
        avg_repeat_visits_per_user = np.mean(list(user_repeat_visit_counts.values()))
        max_repeat_visits_per_user = max(user_repeat_visit_counts.values())
    else:
        avg_windows_per_user = 0.0
        avg_repeat_visits_per_user = 0.0
        max_repeat_visits_per_user = 0
    
    # Distribution of repeat visit counts per user per window (vectorized)
    # Include all users (including those with 0 repeat visits)
    if show_progress:
        logger.info("Computing repeat visit count distribution (including 0 repeats)...")
    
    # Include all users, not just repeat users
    repeat_visit_count_distribution_series = window_user_visits['repeat_visit_count'].value_counts().sort_index()
    repeat_visit_count_distribution = repeat_visit_count_distribution_series.to_dict()
    
    # Compute visit percentage by repeat count level across all windows
    # For each repeat count level, calculate total visits and percentage
    if show_progress:
        logger.info("Computing visit percentage by repeat count level...")
    
    # Merge repeat_visit_count back to calculate total visits per level
    repeat_visit_visit_pct = {}
    total_visits_all = window_user_visits['visit_count'].sum()
    
    for repeat_count in sorted(repeat_visit_count_distribution.keys()):
        # Filter users with this repeat count
        users_with_count = window_user_visits[window_user_visits['repeat_visit_count'] == repeat_count]
        # Total visits by these users (note: visit_count = repeat_count + 1 for repeat users, or 1 for single visits)
        total_visits_for_level = users_with_count['visit_count'].sum()
        pct = (total_visits_for_level / total_visits_all * 100) if total_visits_all > 0 else 0
        repeat_visit_visit_pct[repeat_count] = {
            'total_visits': int(total_visits_for_level),
            'percentage': float(pct),
            'user_occurrences': int(repeat_visit_count_distribution[repeat_count])
        }
    
    elapsed = time.time() - start_time
    
    # Print summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Repeat Visit Statistics for {window_size} Time Window")
    logger.info(f"{'='*60}")
    logger.info(f"Total time windows: {total_windows:,}")
    logger.info(f"Average total visits (requests) per window: {avg_total_visits_per_window:.2f} (Min: {min_total_visits_per_window:,}, Max: {max_total_visits_per_window:,})")
    logger.info(f"Total visits (requests) across all windows: {total_visits_all_windows:,}")
    logger.info(f"Average repeat users per window: {avg_repeat_users_per_window:.2f}")
    logger.info(f"Total repeat user occurrences across all windows (sum of all windows): {total_repeat_users_all_windows:,}")
    logger.info(f"Average repeat visits per window: {avg_repeat_visits_per_window:.2f}")
    logger.info(f"Total repeat visits across all windows: {total_repeat_visits_all_windows:,}")
    logger.info(f"")
    logger.info(f"Users with at least one repeat visit: {num_users_with_repeats:,}")
    logger.info(f"Average windows with repeats per user: {avg_windows_per_user:.2f}")
    logger.info(f"Average total repeat visits per user: {avg_repeat_visits_per_user:.2f}")
    logger.info(f"Max total repeat visits for a user: {max_repeat_visits_per_user:,}")
    logger.info(f"")
    logger.info(f"Repeat visit count distribution (per user per window, including 0 repeats):")
    logger.info(f"  Note: Larger windows may have fewer occurrences in lower repeat counts")
    logger.info(f"        because users move to higher repeat count categories.")
    logger.info(f"  Percentage: percentage of total visits across ALL windows (not per window)")
    for repeat_count in sorted(repeat_visit_count_distribution.keys())[:20]:
        count = repeat_visit_count_distribution[repeat_count]
        if repeat_count in repeat_visit_visit_pct:
            pct_info = repeat_visit_visit_pct[repeat_count]
            logger.info(f"  {repeat_count} repeat visit(s): {count:,} occurrences, "
                       f"{pct_info['total_visits']:,} visits ({pct_info['percentage']:.2f}% of total visits across all windows)")
        else:
            logger.info(f"  {repeat_count} repeat visit(s): {count:,} occurrences")
    logger.info(f"\nAnalysis complete, elapsed {elapsed:.2f} seconds")
    
    return {
        'window_size': window_size,
        'total_windows': int(total_windows),
        'window_results': results_df.to_dict('records'),
        'total_repeat_users_all_windows': int(total_repeat_users_all_windows),
        'avg_repeat_users_per_window': float(avg_repeat_users_per_window),
        'total_visits_all_windows': int(total_visits_all_windows),
        'avg_total_visits_per_window': float(avg_total_visits_per_window),
        'total_repeat_visits_all_windows': int(total_repeat_visits_all_windows),
        'avg_repeat_visits_per_window': float(avg_repeat_visits_per_window),
        'num_users_with_repeats': int(num_users_with_repeats),
        'avg_windows_per_user': float(avg_windows_per_user),
        'avg_repeat_visits_per_user': float(avg_repeat_visits_per_user),
        'max_repeat_visits_per_user': int(max_repeat_visits_per_user),
        'repeat_visit_count_distribution': {int(k): int(v) for k, v in repeat_visit_count_distribution.items()},
        'repeat_visit_visit_percentage': {int(k): v for k, v in repeat_visit_visit_pct.items()},
        'user_repeat_window_counts': dict(user_repeat_window_counts),
        'user_repeat_visit_counts': dict(user_repeat_visit_counts)
    }


def plot_repeat_visits_statistics(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """Plot statistics for all time windows"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Repeat Visits Statistics Across Time Windows', fontsize=16, fontweight='bold')
    
    # 1. Number of repeat users per window (average)
    ax1 = axes[0, 0]
    window_labels = []
    avg_repeat_users = []
    for ws in window_sizes:
        if ws in stats_dict:
            window_labels.append(ws)
            avg_repeat_users.append(stats_dict[ws]['avg_repeat_users_per_window'])
    
    bars1 = ax1.bar(window_labels, avg_repeat_users, alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax1.set_xlabel('Time Window Size', fontsize=12)
    ax1.set_ylabel('Average Number of Repeat Users per Window', fontsize=12)
    ax1.set_title('Average Repeat Users per Time Window', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Total repeat visits per window (average)
    ax2 = axes[0, 1]
    avg_repeat_visits = []
    for ws in window_labels:
        avg_repeat_visits.append(stats_dict[ws]['avg_repeat_visits_per_window'])
    
    bars2 = ax2.bar(window_labels, avg_repeat_visits, alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax2.set_xlabel('Time Window Size', fontsize=12)
    ax2.set_ylabel('Average Number of Repeat Visits per Window', fontsize=12)
    ax2.set_title('Average Repeat Visits per Time Window', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Number of users with repeat visits
    ax3 = axes[1, 0]
    num_users_with_repeats = []
    for ws in window_labels:
        num_users_with_repeats.append(stats_dict[ws]['num_users_with_repeats'])
    
    bars3 = ax3.bar(window_labels, num_users_with_repeats, alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax3.set_xlabel('Time Window Size', fontsize=12)
    ax3.set_ylabel('Number of Users with Repeat Visits', fontsize=12)
    ax3.set_title('Total Users with Repeat Visits', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Average repeat visits per user
    ax4 = axes[1, 1]
    avg_repeat_visits_per_user = []
    for ws in window_labels:
        avg_repeat_visits_per_user.append(stats_dict[ws]['avg_repeat_visits_per_user'])
    
    bars4 = ax4.bar(window_labels, avg_repeat_visits_per_user, alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax4.set_xlabel('Time Window Size', fontsize=12)
    ax4.set_ylabel('Average Repeat Visits per User', fontsize=12)
    ax4.set_title('Average Total Repeat Visits per User', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repeat_visits_time_window_summary.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved summary plot: {output_path}")
    
    # Plot distribution of repeat visit counts for each window (including 0 repeats)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Distribution of Repeat Visit Counts per User per Window (Including 0 Repeats)', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        ax = axes[idx // 2, idx % 2]
        dist = stats_dict[ws]['repeat_visit_count_distribution']
        
        if dist:
            repeat_counts = sorted(dist.keys())[:20]  # Show top 20
            counts = [dist[rc] for rc in repeat_counts]
            
            bars = ax.bar(repeat_counts, counts, alpha=0.7, color=colors[idx], edgecolor='black')
            ax.set_xlabel('Number of Repeat Visits (0 = No Repeats)', fontsize=10)
            ax.set_ylabel('Frequency (Occurrences)', fontsize=10)
            ax.set_title(f'Repeat Visit Count Distribution ({ws})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Repeat Visit Count Distribution ({ws})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "repeat_visits_count_distribution.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved distribution plot: {output_path}")


def plot_repeat_count_distribution_curve(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot repeat visit count distribution as curves for all time windows"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Repeat Visit Count Distribution Curves', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        dist = stats_dict[ws]['repeat_visit_count_distribution']
        
        if dist:
            # Sort by repeat count
            sorted_counts = sorted(dist.keys())
            # Limit to reasonable range for better visualization
            max_repeat_count = min(50, max(sorted_counts))
            sorted_counts = [c for c in sorted_counts if c <= max_repeat_count]
            frequencies = [dist[rc] for rc in sorted_counts]
            
            # Plot as line/curve
            ax.plot(sorted_counts, frequencies, 'o-', linewidth=2, markersize=4, 
                   label=ws, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Number of Repeat Visits (0 = No Repeats)', fontsize=12)
    ax.set_ylabel('Frequency (Occurrences)', fontsize=12)
    ax.set_title('Repeat Visit Count Distribution (Line Chart)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repeat_count_distribution_curve.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved repeat count distribution curve plot: {output_path}")
    
    # Also create a normalized version (percentage of occurrences)
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
    fig2.suptitle('Normalized Repeat Visit Count Distribution Curves', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        dist = stats_dict[ws]['repeat_visit_count_distribution']
        
        if dist:
            sorted_counts = sorted(dist.keys())
            max_repeat_count = min(50, max(sorted_counts))
            sorted_counts = [c for c in sorted_counts if c <= max_repeat_count]
            frequencies = [dist[rc] for rc in sorted_counts]
            
            # Normalize to percentages
            total_occurrences = sum(dist.values())
            percentages = [f / total_occurrences * 100 for f in frequencies]
            
            ax2.plot(sorted_counts, percentages, 'o-', linewidth=2, markersize=4,
                    label=ws, color=colors[idx], alpha=0.8)
    
    ax2.set_xlabel('Number of Repeat Visits (0 = No Repeats)', fontsize=12)
    ax2.set_ylabel('Percentage of Occurrences (%)', fontsize=12)
    ax2.set_title('Normalized Repeat Visit Count Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    output_path2 = output_dir / "repeat_count_distribution_curve_normalized.png"
    fig2.savefig(output_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved normalized repeat count distribution curve plot: {output_path2}")


def plot_repeat_users_over_time(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """Plot number of repeat users over time for each window size"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Number of Repeat Users Over Time', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        ax = axes[idx // 2, idx % 2]
        
        window_results = stats_dict[ws]['window_results']
        results_df = pd.DataFrame(window_results)
        
        if len(results_df) > 0:
            # Sample data if too many points
            max_points = 1000
            if len(results_df) > max_points:
                # Sample uniformly
                step = len(results_df) // max_points
                results_df = results_df.iloc[::step]
            
            time_starts = pd.to_datetime(results_df['time_window_start'])
            num_repeat_users = results_df['num_repeat_users']
            
            ax.plot(time_starts, num_repeat_users, linewidth=1.5, color=colors[idx], alpha=0.7)
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Number of Repeat Users', fontsize=10)
            ax.set_title(f'Repeat Users Over Time ({ws})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Repeat Users Over Time ({ws})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repeat_users_over_time.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved time series plot: {output_path}")


def plot_user_repeat_visits_per_window(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """Plot statistics about how many times a user repeats visits within a single time window"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('User Repeat Visits Statistics Within a Single Time Window', fontsize=16, fontweight='bold')
    
    # Collect data for each window size
    window_labels = []
    avg_repeat_visits_per_repeat_user_list = []
    median_repeat_visits_per_repeat_user_list = []
    max_repeat_visits_list = []
    
    for ws in window_sizes:
        if ws not in stats_dict:
            continue
        
        window_labels.append(ws)
        window_results = stats_dict[ws]['window_results']
        
        if window_results:
            results_df = pd.DataFrame(window_results)
            # Filter out windows with no repeat users
            windows_with_repeats = results_df[results_df['num_repeat_users'] > 0]
            
            if len(windows_with_repeats) > 0:
                avg_repeat_visits_per_repeat_user_list.append(
                    windows_with_repeats['avg_repeat_visits_per_repeat_user'].mean()
                )
                median_repeat_visits_per_repeat_user_list.append(
                    windows_with_repeats['avg_repeat_visits_per_repeat_user'].median()
                )
                max_repeat_visits_list.append(
                    windows_with_repeats['max_repeat_visits'].max()
                )
            else:
                avg_repeat_visits_per_repeat_user_list.append(0)
                median_repeat_visits_per_repeat_user_list.append(0)
                max_repeat_visits_list.append(0)
    
    # 1. Average repeat visits per repeat user per window
    ax1 = axes[0, 0]
    bars1 = ax1.bar(window_labels, avg_repeat_visits_per_repeat_user_list, 
                    alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax1.set_xlabel('Time Window Size', fontsize=12)
    ax1.set_ylabel('Average Repeat Visits per Repeat User', fontsize=12)
    ax1.set_title('Average Repeat Visits per Repeat User Within a Window', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Median repeat visits per repeat user per window
    ax2 = axes[0, 1]
    bars2 = ax2.bar(window_labels, median_repeat_visits_per_repeat_user_list, 
                    alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax2.set_xlabel('Time Window Size', fontsize=12)
    ax2.set_ylabel('Median Repeat Visits per Repeat User', fontsize=12)
    ax2.set_title('Median Repeat Visits per Repeat User Within a Window', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Maximum repeat visits in any window
    ax3 = axes[1, 0]
    bars3 = ax3.bar(window_labels, max_repeat_visits_list, 
                    alpha=0.7, color=colors[:len(window_labels)], edgecolor='black')
    ax3.set_xlabel('Time Window Size', fontsize=12)
    ax3.set_ylabel('Maximum Repeat Visits in Any Window', fontsize=12)
    ax3.set_title('Maximum Repeat Visits by a User in Any Window', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Distribution of avg_repeat_visits_per_repeat_user across windows
    ax4 = axes[1, 1]
    for idx, ws in enumerate(window_labels):
        window_results = stats_dict[ws]['window_results']
        if window_results:
            results_df = pd.DataFrame(window_results)
            windows_with_repeats = results_df[results_df['num_repeat_users'] > 0]
            if len(windows_with_repeats) > 0:
                # Sample if too many points for histogram
                data = windows_with_repeats['avg_repeat_visits_per_repeat_user'].values
                if len(data) > 10000:
                    data = np.random.choice(data, 10000, replace=False)
                ax4.hist(data, bins=50, alpha=0.5, label=ws, color=colors[idx], edgecolor='black')
    
    ax4.set_xlabel('Average Repeat Visits per Repeat User', fontsize=12)
    ax4.set_ylabel('Frequency (Number of Windows)', fontsize=12)
    ax4.set_title('Distribution of Avg Repeat Visits per Repeat User Across Windows', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "user_repeat_visits_per_window_stats.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved user repeat visits per window statistics plot: {output_path}")


def plot_repeat_visit_percentage_by_level(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """Plot percentage of total visits by repeat visit count level"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Percentage of Total Visits by Repeat Visit Count Level', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        ax = axes[idx // 2, idx % 2]
        
        if 'repeat_visit_visit_percentage' not in stats_dict[ws]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Visit Percentage by Repeat Level ({ws})', fontsize=12, fontweight='bold')
            continue
        
        visit_pct = stats_dict[ws]['repeat_visit_visit_percentage']
        
        if visit_pct:
            # Sort by repeat count
            sorted_levels = sorted(visit_pct.keys())
            # Limit to top 20 for readability
            sorted_levels = sorted_levels[:20]
            
            percentages = [visit_pct[level]['percentage'] for level in sorted_levels]
            total_visits = [visit_pct[level]['total_visits'] for level in sorted_levels]
            
            # Create bar chart
            bars = ax.bar(sorted_levels, percentages, alpha=0.7, color=colors[idx], edgecolor='black')
            ax.set_xlabel('Number of Repeat Visits (0 = No Repeats)', fontsize=10)
            ax.set_ylabel('Percentage of Total Visits (%)', fontsize=10)
            ax.set_title(f'Visit Percentage by Repeat Level ({ws})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars (only for significant percentages)
            for i, (bar, pct, visits) in enumerate(zip(bars, percentages, total_visits)):
                if pct > 0.1:  # Only show label if percentage > 0.1%
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{pct:.2f}%',
                           ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Visit Percentage by Repeat Level ({ws})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repeat_visit_percentage_by_level.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved repeat visit percentage by level plot: {output_path}")
    
    # Also create a stacked area/bar chart showing cumulative percentages
    fig2, axes2 = plt.subplots(2, 2, figsize=figsize)
    fig2.suptitle('Cumulative Percentage of Total Visits by Repeat Visit Count Level', fontsize=16, fontweight='bold')
    
    for idx, ws in enumerate(window_sizes):
        if ws not in stats_dict:
            continue
        
        ax = axes2[idx // 2, idx % 2]
        
        if 'repeat_visit_visit_percentage' not in stats_dict[ws]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cumulative Visit Percentage ({ws})', fontsize=12, fontweight='bold')
            continue
        
        visit_pct = stats_dict[ws]['repeat_visit_visit_percentage']
        
        if visit_pct:
            sorted_levels = sorted(visit_pct.keys())
            sorted_levels = sorted_levels[:20]  # Top 20
            
            percentages = [visit_pct[level]['percentage'] for level in sorted_levels]
            cumulative_pct = np.cumsum(percentages)
            
            # Create bar chart with cumulative line
            bars = ax.bar(sorted_levels, percentages, alpha=0.7, color=colors[idx], edgecolor='black', label='Percentage')
            line = ax.plot(sorted_levels, cumulative_pct, 'ro-', linewidth=2, markersize=4, label='Cumulative %', alpha=0.8)
            
            ax.set_xlabel('Number of Repeat Visits (0 = No Repeats)', fontsize=10)
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_title(f'Cumulative Visit Percentage ({ws})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='best', fontsize=9)
            ax.set_ylim([0, min(100, max(cumulative_pct) * 1.1)])
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cumulative Visit Percentage ({ws})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path2 = output_dir / "repeat_visit_cumulative_percentage.png"
    fig2.savefig(output_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved cumulative repeat visit percentage plot: {output_path2}")


def plot_repeat_visit_count_boxplot(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot boxplot of maximum repeat visits per window for each time window size
    Shows distribution of max_repeat_visits (max repeat visits by a user in each window) across all windows
    
    Args:
        stats_dict: Dictionary with statistics for each window size
        output_dir: Output directory for the plot
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    logger.info("Creating boxplot of maximum repeat visits per window by window size...")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Distribution of Maximum Repeat Visits per Window by Time Window Size', 
                 fontsize=16, fontweight='bold')
    
    window_sizes = ['5min', '10min', '30min', '1h']
    colors = ['steelblue', 'orange', 'green', 'red']
    
    boxplot_data = []
    boxplot_labels = []
    
    for window_size in window_sizes:
        if window_size not in stats_dict:
            continue
        
        window_results = stats_dict[window_size]['window_results']
        if not window_results:
            continue
        
        results_df = pd.DataFrame(window_results)
        # Filter out windows with no repeat users
        windows_with_repeats = results_df[results_df['num_repeat_users'] > 0]
        
        if len(windows_with_repeats) > 0:
            # Get maximum repeat visits for each window (max repeat visits by a user in that window)
            max_repeat_visits = windows_with_repeats['max_repeat_visits'].values
            
            boxplot_data.append(max_repeat_visits)
            boxplot_labels.append(window_size)
            
            # Log statistics
            min_val = float(max_repeat_visits.min())
            max_val = float(max_repeat_visits.max())
            mean_val = float(max_repeat_visits.mean())
            median_val = float(np.median(max_repeat_visits))
            logger.info(f"  {window_size}: Min={min_val:.0f}, Max={max_val:.0f}, Mean={mean_val:.2f}, Median={median_val:.2f}")
    
    if len(boxplot_data) > 0:
        # Create boxplot - show all data (no outliers, all are real data)
        bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True, 
                       showmeans=True, meanline=False, widths=0.6,
                       showfliers=False,  # Don't mark any points as outliers (all are real data)
                       whis=(0, 100),  # Use 0th and 100th percentiles to include all data
                       meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6))
        
        # Color the boxes
        for idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[idx % len(colors)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Make median line more visible
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        ax.set_xlabel('Time Window Size', fontsize=12)
        ax.set_ylabel('Maximum Repeat Visits per Window', fontsize=12)
        ax.set_title('Distribution of Maximum Repeat Visits per Window Across All Windows', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Distribution of Maximum Repeat Visits per Window Across All Windows', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repeat_visit_count_boxplot.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved boxplot: {output_path}")


def save_results_to_json(results: Dict, output_path: Path) -> None:
    """Save analysis results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    results_converted = convert_types(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repeat visits within time windows for KuaiRand dataset"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing KuaiRand log CSV files'
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
    data_dir = Path(args.data_dir).expanduser()
    df = load_kuairand_data(data_dir, args.show_progress)
    
    # Analyze repeat visits for each time window
    window_sizes = ['5min', '10min', '30min', '1h']
    all_stats = {}
    
    for window_size in window_sizes:
        stats = analyze_repeat_visits_time_window(df, window_size, args.show_progress)
        all_stats[window_size] = stats
        
        # Save individual window results to CSV
        window_results_df = pd.DataFrame(stats['window_results'])
        csv_path = output_dir / f"repeat_visits_{window_size}_window_results.csv"
        window_results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {window_size} window results to: {csv_path}")
    
    # Plot statistics
    plot_repeat_visits_statistics(all_stats, output_dir)
    plot_repeat_users_over_time(all_stats, output_dir)
    plot_user_repeat_visits_per_window(all_stats, output_dir)
    plot_repeat_visit_percentage_by_level(all_stats, output_dir)
    plot_repeat_count_distribution_curve(all_stats, output_dir)
    plot_repeat_visit_count_boxplot(all_stats, output_dir)
    
    # Save summary statistics to JSON
    summary_stats = {}
    for ws in window_sizes:
        if ws in all_stats:
            summary_stats[ws] = {
                k: v for k, v in all_stats[ws].items() 
                if k not in ['window_results', 'user_repeat_window_counts', 'user_repeat_visit_counts']
            }
    
    results_json = output_dir / "repeat_visits_time_window_results.json"
    save_results_to_json(summary_stats, results_json)
    
    logger.info("\n" + "="*60)
    logger.info("All analyses complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


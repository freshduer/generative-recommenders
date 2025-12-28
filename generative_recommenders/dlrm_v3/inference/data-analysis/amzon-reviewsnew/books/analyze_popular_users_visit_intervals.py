#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析热门用户的重复访问间隔时间
- 识别热门用户（按访问次数排序）
- 计算每个热门用户的访问间隔（相邻两次访问之间的时间差）
- 统计访问间隔的分布
- 可视化结果
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


def identify_popular_users(
    df: pd.DataFrame,
    top_n: Optional[int] = None,
    top_pct: Optional[float] = None,
    min_visits: int = 2
) -> pd.DataFrame:
    """
    识别热门用户
    
    Args:
        df: 数据DataFrame
        top_n: 取前N个用户（如果指定）
        top_pct: 取前X%的用户（如果指定，例如0.1表示前10%）
        min_visits: 最小访问次数要求
    
    Returns:
        热门用户的DataFrame，包含user_id和visit_count
    """
    logger.info("\n" + "="*60)
    logger.info("Identifying Popular Users")
    logger.info("="*60)
    
    # 统计每个用户的访问次数
    user_visits_all = df.groupby('user_id', observed=True).size().reset_index(name='visit_count')
    total_all_users = len(user_visits_all)
    logger.info(f"Total users in dataset: {total_all_users:,}")
    
    # 统计访问次数分布
    single_visit_users = (user_visits_all['visit_count'] == 1).sum()
    multi_visit_users = (user_visits_all['visit_count'] >= 2).sum()
    logger.info(f"  - Users with 1 visit: {single_visit_users:,} ({single_visit_users/total_all_users*100:.2f}%)")
    logger.info(f"  - Users with 2+ visits: {multi_visit_users:,} ({multi_visit_users/total_all_users*100:.2f}%)")
    
    # 筛选满足最小访问次数要求的用户
    user_visits = user_visits_all[user_visits_all['visit_count'] >= min_visits].copy()
    user_visits = user_visits.sort_values('visit_count', ascending=False).reset_index(drop=True)
    
    total_eligible_users = len(user_visits)
    logger.info(f"\nUsers with at least {min_visits} visits: {total_eligible_users:,}")
    logger.info(f"  (This is {total_eligible_users/total_all_users*100:.2f}% of all users)")
    
    # 根据参数选择热门用户
    if top_n is not None:
        popular_users = user_visits.head(top_n)
        logger.info(f"\nSelected top {top_n} users by visit count")
        logger.info(f"  (This is {top_n/total_all_users*100:.2f}% of all users)")
    elif top_pct is not None:
        n_users = max(1, int(total_eligible_users * top_pct))
        popular_users = user_visits.head(n_users)
        logger.info(f"\nSelected top {top_pct*100:.1f}% of eligible users ({n_users:,} users) by visit count")
        logger.info(f"  (This is {n_users/total_all_users*100:.2f}% of all users)")
        logger.info(f"  (Based on {total_eligible_users:,} users with {min_visits}+ visits)")
    else:
        # 默认取前10%
        n_users = max(1, int(total_eligible_users * 0.1))
        popular_users = user_visits.head(n_users)
        logger.info(f"\nDefault: Selected top 10% of eligible users ({n_users:,} users) by visit count")
        logger.info(f"  (This is {n_users/total_all_users*100:.2f}% of all users)")
        logger.info(f"  (Based on {total_eligible_users:,} users with {min_visits}+ visits)")
    
    logger.info(f"\nPopular users statistics:")
    logger.info(f"  Visit count range: {popular_users['visit_count'].min():,} - {popular_users['visit_count'].max():,}")
    logger.info(f"  Average visits per user: {popular_users['visit_count'].mean():.2f}")
    logger.info(f"  Median visits per user: {popular_users['visit_count'].median():.2f}")
    
    return popular_users


def compute_visit_intervals(
    df: pd.DataFrame,
    popular_user_ids: pd.Series,
    show_progress: bool = False,
    use_multiprocessing: bool = True,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    计算热门用户的访问间隔（使用向量化操作优化）
    
    Args:
        df: 完整数据DataFrame
        popular_user_ids: 热门用户的user_id Series
        show_progress: 是否显示进度条
        use_multiprocessing: 是否使用多进程
        n_workers: 工作进程数
    
    Returns:
        包含访问间隔信息的DataFrame
    """
    logger.info("\n" + "="*60)
    logger.info("Computing Visit Intervals for Popular Users")
    logger.info("="*60)
    
    start_time = time.time()
    
    # 筛选热门用户的数据
    popular_df = df[df['user_id'].isin(popular_user_ids)].copy()
    popular_df = popular_df.sort_values(['user_id', 'datetime']).reset_index(drop=True)
    
    logger.info(f"Processing {len(popular_user_ids):,} popular users")
    logger.info(f"Total visits from popular users: {len(popular_df):,}")
    
    # 使用向量化操作计算间隔（比循环快很多）
    if show_progress:
        logger.info("Computing intervals using vectorized operations...")
    
    # 按用户分组，计算每个用户内的访问间隔
    popular_df['prev_timestamp'] = popular_df.groupby('user_id', observed=True)['timestamp'].shift(1)
    popular_df['prev_datetime'] = popular_df.groupby('user_id', observed=True)['datetime'].shift(1)
    
    # 只保留有前一次访问的记录（即间隔记录）
    intervals_df = popular_df[popular_df['prev_timestamp'].notna()].copy()
    
    if len(intervals_df) == 0:
        logger.warning("No intervals computed! Users may have less than 2 visits.")
        return pd.DataFrame()
    
    # 计算间隔（秒）
    intervals_df['interval_seconds'] = intervals_df['timestamp'] - intervals_df['prev_timestamp']
    
    # 转换为其他时间单位
    intervals_df['interval_minutes'] = intervals_df['interval_seconds'] / 60.0
    intervals_df['interval_hours'] = intervals_df['interval_seconds'] / 3600.0
    intervals_df['interval_days'] = intervals_df['interval_seconds'] / (24.0 * 3600.0)
    
    # 计算每个用户内的间隔索引
    intervals_df['interval_index'] = intervals_df.groupby('user_id', observed=True).cumcount() + 1
    
    # 重命名列
    intervals_df = intervals_df.rename(columns={
        'timestamp': 'next_timestamp',
        'datetime': 'next_datetime'
    })
    
    # 选择需要的列
    result_df = intervals_df[[
        'user_id',
        'interval_index',
        'interval_seconds',
        'interval_minutes',
        'interval_hours',
        'interval_days',
        'prev_timestamp',
        'next_timestamp',
        'prev_datetime',
        'next_datetime'
    ]].copy()
    
    # 转换数据类型
    result_df['interval_seconds'] = result_df['interval_seconds'].astype(float)
    result_df['interval_minutes'] = result_df['interval_minutes'].astype(float)
    result_df['interval_hours'] = result_df['interval_hours'].astype(float)
    result_df['interval_days'] = result_df['interval_days'].astype(float)
    result_df['prev_timestamp'] = result_df['prev_timestamp'].astype(int)
    result_df['next_timestamp'] = result_df['next_timestamp'].astype(int)
    result_df['interval_index'] = result_df['interval_index'].astype(int)
    
    elapsed = time.time() - start_time
    logger.info(f"Computed {len(result_df):,} visit intervals, elapsed {elapsed:.2f} seconds")
    
    return result_df


def analyze_visit_intervals(
    intervals_df: pd.DataFrame,
    show_progress: bool = False
) -> Dict:
    """
    分析访问间隔的统计信息
    
    Returns:
        包含统计信息的字典
    """
    logger.info("\n" + "="*60)
    logger.info("Analyzing Visit Intervals")
    logger.info("="*60)
    
    if len(intervals_df) == 0:
        logger.warning("No intervals to analyze!")
        return {}
    
    # 基本统计
    stats = {
        'total_intervals': len(intervals_df),
        'unique_users': intervals_df['user_id'].nunique(),
        'mean_interval_days': float(intervals_df['interval_days'].mean()),
        'median_interval_days': float(intervals_df['interval_days'].median()),
        'std_interval_days': float(intervals_df['interval_days'].std()),
        'min_interval_days': float(intervals_df['interval_days'].min()),
        'max_interval_days': float(intervals_df['interval_days'].max()),
        'mean_interval_hours': float(intervals_df['interval_hours'].mean()),
        'median_interval_hours': float(intervals_df['interval_hours'].median()),
        'mean_interval_minutes': float(intervals_df['interval_minutes'].mean()),
        'median_interval_minutes': float(intervals_df['interval_minutes'].median()),
    }
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}_interval_days'] = float(intervals_df['interval_days'].quantile(p / 100.0))
        stats[f'p{p}_interval_hours'] = float(intervals_df['interval_hours'].quantile(p / 100.0))
    
    # 按时间范围分类统计（更细粒度，特别是1小时内）
    # 转换为分钟以便更精确分类
    intervals_df['interval_minutes_calc'] = intervals_df['interval_minutes']
    
    intervals_df['interval_category'] = pd.cut(
        intervals_df['interval_minutes_calc'],
        bins=[0, 5, 10, 30, 60, 60*24, 60*24*7, 60*24*30, 60*24*90, 60*24*365, float('inf')],
        labels=[
            '<5 min', 
            '5-10 min', 
            '10-30 min', 
            '30 min-1 hour', 
            '1 hour-1 day', 
            '1-7 days', 
            '7-30 days', 
            '30-90 days', 
            '90-365 days', 
            '>365 days'
        ]
    )
    
    category_counts = intervals_df['interval_category'].value_counts().to_dict()
    category_pcts = (intervals_df['interval_category'].value_counts(normalize=True) * 100).to_dict()
    
    stats['interval_category_counts'] = {str(k): int(v) for k, v in category_counts.items()}
    stats['interval_category_percentages'] = {str(k): float(v) for k, v in category_pcts.items()}
    
    # 打印统计信息
    logger.info(f"总访问间隔数: {stats['total_intervals']:,}")
    logger.info(f"涉及用户数: {stats['unique_users']:,}")
    logger.info(f"\n间隔时间统计（天）:")
    logger.info(f"  平均值: {stats['mean_interval_days']:.2f} 天")
    logger.info(f"  中位数: {stats['median_interval_days']:.2f} 天")
    logger.info(f"  标准差: {stats['std_interval_days']:.2f} 天")
    logger.info(f"  最小值: {stats['min_interval_days']:.4f} 天")
    logger.info(f"  最大值: {stats['max_interval_days']:.2f} 天")
    
    logger.info(f"\n间隔时间统计（小时）:")
    logger.info(f"  平均值: {stats['mean_interval_hours']:.2f} 小时")
    logger.info(f"  中位数: {stats['median_interval_hours']:.2f} 小时")
    
    logger.info(f"\n分位数（天）:")
    for p in percentiles:
        logger.info(f"  P{p}: {stats[f'p{p}_interval_days']:.2f} 天")
    
    logger.info(f"\n间隔时间分布:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = category_pcts[category]
        logger.info(f"  {category}: {count:,} ({pct:.2f}%)")
    
    # 添加1小时内的详细统计
    intervals_under_1h = intervals_df[intervals_df['interval_minutes'] < 60]
    if len(intervals_under_1h) > 0:
        logger.info(f"\n1小时内访问间隔详细统计:")
        logger.info(f"  总数量: {len(intervals_under_1h):,} ({len(intervals_under_1h)/len(intervals_df)*100:.2f}%)")
        logger.info(f"  <5分钟: {(intervals_df['interval_minutes'] < 5).sum():,} ({(intervals_df['interval_minutes'] < 5).sum()/len(intervals_df)*100:.2f}%)")
        logger.info(f"  5-10分钟: {((intervals_df['interval_minutes'] >= 5) & (intervals_df['interval_minutes'] < 10)).sum():,} ({((intervals_df['interval_minutes'] >= 5) & (intervals_df['interval_minutes'] < 10)).sum()/len(intervals_df)*100:.2f}%)")
        logger.info(f"  10-30分钟: {((intervals_df['interval_minutes'] >= 10) & (intervals_df['interval_minutes'] < 30)).sum():,} ({((intervals_df['interval_minutes'] >= 10) & (intervals_df['interval_minutes'] < 30)).sum()/len(intervals_df)*100:.2f}%)")
        logger.info(f"  30-60分钟: {((intervals_df['interval_minutes'] >= 30) & (intervals_df['interval_minutes'] < 60)).sum():,} ({((intervals_df['interval_minutes'] >= 30) & (intervals_df['interval_minutes'] < 60)).sum()/len(intervals_df)*100:.2f}%)")
        logger.info(f"  平均间隔: {intervals_under_1h['interval_minutes'].mean():.2f} 分钟")
        logger.info(f"  中位数间隔: {intervals_under_1h['interval_minutes'].median():.2f} 分钟")
    
    return stats


def plot_visit_intervals(
    intervals_df: pd.DataFrame,
    stats: Dict,
    output_path: Path,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    绘制访问间隔的可视化图表
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    if len(intervals_df) == 0:
        logger.warning("No intervals to plot!")
        return
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 间隔时间直方图（天）- 线性尺度
    ax1 = fig.add_subplot(gs[0, 0])
    intervals_days = intervals_df['interval_days'].values
    # 过滤极端值以便更好地可视化
    intervals_filtered = intervals_days[intervals_days <= np.percentile(intervals_days, 95)]
    ax1.hist(intervals_filtered, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(stats['median_interval_days'], color='red', linestyle='--', linewidth=2,
                label=f'Median: {stats["median_interval_days"]:.2f} days')
    ax1.axvline(stats['mean_interval_days'], color='green', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean_interval_days"]:.2f} days')
    ax1.set_xlabel('Visit Interval (days)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Visit Interval Distribution (Histogram, filtered top 5%)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 间隔时间直方图（天）- 对数尺度
    ax2 = fig.add_subplot(gs[0, 1])
    intervals_log = np.log10(intervals_days + 1)  # +1避免log(0)
    ax2.hist(intervals_log, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Visit Interval (log10(days+1))', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Visit Interval Distribution (Log Scale)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布函数（CDF）
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_intervals = np.sort(intervals_days)
    y = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
    ax3.plot(sorted_intervals, y * 100, linewidth=2, color='steelblue')
    ax3.axvline(stats['median_interval_days'], color='red', linestyle='--', linewidth=2,
                label=f'Median: {stats["median_interval_days"]:.2f} days')
    ax3.set_xlabel('Visit Interval (days)', fontsize=10)
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=10)
    ax3.set_title('Visit Interval Cumulative Distribution (CDF)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. 箱线图
    ax4 = fig.add_subplot(gs[1, 0])
    # 按间隔类别分组绘制箱线图
    category_data = []
    category_labels = []
    # 使用新的细粒度分类
    for category in ['<5 min', '5-10 min', '10-30 min', '30 min-1 hour', '1 hour-1 day', '1-7 days', '7-30 days', '30-90 days', '90-365 days', '>365 days']:
        cat_data = intervals_df[intervals_df['interval_category'] == category]['interval_days'].values
        if len(cat_data) > 0:
            category_data.append(cat_data)
            category_labels.append(category)
    
    if category_data:
        bp = ax4.boxplot(category_data, tick_labels=category_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax4.set_ylabel('Visit Interval (days)', fontsize=10)
        ax4.set_title('Visit Interval Boxplot (by Category)', fontsize=11, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_yscale('log')
    
    # 5. 间隔时间类别分布（饼图）
    ax5 = fig.add_subplot(gs[1, 1])
    category_counts = intervals_df['interval_category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    wedges, texts, autotexts = ax5.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    ax5.set_title('Visit Interval Category Distribution', fontsize=11, fontweight='bold')
    
    # 6. 每个用户的平均间隔时间分布
    ax6 = fig.add_subplot(gs[1, 2])
    user_avg_intervals = intervals_df.groupby('user_id', as_index=False)['interval_days'].mean()
    user_avg_intervals.columns = ['user_id', 'avg_interval_days']
    ax6.hist(user_avg_intervals['avg_interval_days'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(user_avg_intervals['avg_interval_days'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {user_avg_intervals["avg_interval_days"].median():.2f} days')
    ax6.set_xlabel('Average Visit Interval per User (days)', fontsize=10)
    ax6.set_ylabel('Number of Users', fontsize=10)
    ax6.set_title('Distribution of Average Visit Interval per User', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    
    # 7. 间隔时间vs用户访问次数（散点图）
    ax7 = fig.add_subplot(gs[2, 0])
    user_stats = intervals_df.groupby('user_id', as_index=False).agg({
        'interval_days': 'mean',
        'interval_index': 'count'
    })
    user_stats.columns = ['user_id', 'avg_interval_days', 'num_intervals']
    # 采样以避免过度拥挤
    if len(user_stats) > 5000:
        user_stats = user_stats.sample(n=5000, random_state=42)
    ax7.scatter(user_stats['num_intervals'], user_stats['avg_interval_days'],
                alpha=0.5, s=10, color='steelblue')
    ax7.set_xlabel('Number of Visit Intervals per User', fontsize=10)
    ax7.set_ylabel('Average Visit Interval (days)', fontsize=10)
    ax7.set_title('Number of Intervals vs Average Interval Time', fontsize=11, fontweight='bold')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    
    # 8. 时间序列：访问间隔随时间的变化
    ax8 = fig.add_subplot(gs[2, 1])
    # 按时间排序，计算移动平均
    intervals_sorted = intervals_df.sort_values('prev_datetime')
    window_size = max(100, len(intervals_sorted) // 100)
    intervals_sorted['ma_interval_days'] = intervals_sorted['interval_days'].rolling(
        window=window_size, min_periods=1
    ).mean()
    # 采样以加快绘图
    sample_size = min(10000, len(intervals_sorted))
    if sample_size < len(intervals_sorted):
        intervals_sampled = intervals_sorted.iloc[::len(intervals_sorted)//sample_size]
    else:
        intervals_sampled = intervals_sorted
    ax8.plot(intervals_sampled['prev_datetime'], intervals_sampled['ma_interval_days'],
             linewidth=1, alpha=0.7, color='steelblue')
    ax8.set_xlabel('Time', fontsize=10)
    ax8.set_ylabel('Moving Average Visit Interval (days)', fontsize=10)
    ax8.set_title(f'Visit Interval Over Time ({window_size}-point moving average)', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45)
    
    # 9. 统计摘要
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = "Visit Interval Statistics Summary\n\n"
    summary_text += f"Total Intervals: {stats['total_intervals']:,}\n"
    summary_text += f"Unique Users: {stats['unique_users']:,}\n\n"
    summary_text += f"Mean: {stats['mean_interval_days']:.2f} days\n"
    summary_text += f"Median: {stats['median_interval_days']:.2f} days\n"
    summary_text += f"Std Dev: {stats['std_interval_days']:.2f} days\n\n"
    summary_text += f"P25: {stats['p25_interval_days']:.2f} days\n"
    summary_text += f"P75: {stats['p75_interval_days']:.2f} days\n"
    summary_text += f"P90: {stats['p90_interval_days']:.2f} days\n"
    summary_text += f"P95: {stats['p95_interval_days']:.2f} days\n"
    ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Popular Users Visit Interval Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visit intervals plot: {output_path}")


def save_results_to_json(results: Dict, output_path: Path) -> None:
    """Save analysis results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析热门用户的重复访问间隔时间"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Amazon Reviews JSONL文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认: reports/ 在脚本目录下）'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=None,
        help='选择前N个热门用户（如果指定，将覆盖top_pct）'
    )
    parser.add_argument(
        '--top_pct',
        type=float,
        default=0.1,
        help='选择前X%%的热门用户（默认: 0.1，即前10%%）'
    )
    parser.add_argument(
        '--min_visits',
        type=int,
        default=2,
        help='用户最小访问次数要求（默认: 2）'
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='显示进度条'
    )
    parser.add_argument(
        '--no_multiprocessing',
        action='store_true',
        help='禁用多进程数据加载'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=32,
        help='工作进程数（默认: 32）'
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
    
    # Identify popular users
    popular_users = identify_popular_users(
        df,
        top_n=args.top_n,
        top_pct=args.top_pct if args.top_n is None else None,
        min_visits=args.min_visits
    )
    
    # Compute visit intervals
    intervals_df = compute_visit_intervals(
        df,
        popular_users['user_id'],
        show_progress=args.show_progress,
        use_multiprocessing=not args.no_multiprocessing,
        n_workers=args.n_workers
    )
    
    if len(intervals_df) == 0:
        logger.error("No intervals computed! Exiting.")
        return
    
    # Analyze intervals
    stats = analyze_visit_intervals(intervals_df, args.show_progress)
    
    # Plot results
    plot_path = output_dir / "popular_users_visit_intervals.png"
    plot_visit_intervals(intervals_df, stats, plot_path)
    
    # Save intervals data to CSV
    intervals_csv = output_dir / "popular_users_visit_intervals.csv"
    intervals_df.to_csv(intervals_csv, index=False)
    logger.info(f"Saved intervals data to: {intervals_csv}")
    
    # Save popular users info
    popular_users_csv = output_dir / "popular_users_info.csv"
    popular_users.to_csv(popular_users_csv, index=False)
    logger.info(f"Saved popular users info to: {popular_users_csv}")
    
    # Save statistics to JSON
    all_results = {
        'popular_users_count': len(popular_users),
        'popular_users_visit_count_range': {
            'min': int(popular_users['visit_count'].min()),
            'max': int(popular_users['visit_count'].max()),
            'mean': float(popular_users['visit_count'].mean()),
            'median': float(popular_users['visit_count'].median())
        },
        'visit_interval_statistics': stats,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    results_json = output_dir / "popular_users_visit_intervals_results.json"
    save_results_to_json(all_results, results_json)
    
    logger.info("\n" + "="*60)
    logger.info("分析完成！")
    logger.info(f"结果已保存到: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


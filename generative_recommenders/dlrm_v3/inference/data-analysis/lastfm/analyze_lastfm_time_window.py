#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 LastFM 行为数据集不同时间窗口下用户交互序列的长度分布和活跃用户数分布
支持时间窗口：半小时、1小时、半天、一天
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
    matplotlib.use('Agg')  # 非交互式后端
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 设置字体
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


def load_lastfm_data(data_path: str, show_progress: bool = False) -> pd.DataFrame:
    """加载 LastFM 行为数据集"""
    logger.info(f"正在加载数据: {data_path}")
    start_time = time.time()
    
    # LastFM 数据集格式：user_id \t artist_id \t artist_name \t plays
    # 使用chunksize分块读取大文件
    chunks = []
    chunk_size = 1000000  # 每次读取100万行
    
    try:
        # 分块读取
        if HAS_TQDM and show_progress:
            chunk_iterator = pd.read_csv(
                data_path,
                sep='\t',
                header=None,
                names=['user_id', 'artist_id', 'artist_name', 'plays'],
                chunksize=chunk_size,
                dtype={'user_id': 'str', 'artist_id': 'str', 'plays': 'int64'}
            )
            total_rows = sum(1 for _ in open(data_path)) - 1  # 估算总行数
            chunk_iterator = tqdm(chunk_iterator, desc="读取数据块", unit="块", total=max(1, total_rows // chunk_size))
        else:
            chunk_iterator = pd.read_csv(
                data_path,
                sep='\t',
                header=None,
                names=['user_id', 'artist_id', 'artist_name', 'plays'],
                chunksize=chunk_size,
                dtype={'user_id': 'str', 'artist_id': 'str', 'plays': 'int64'}
            )
        
        for chunk in chunk_iterator:
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
    except Exception as e:
        logger.error(f"读取数据时出错: {e}")
        # 如果分块读取失败，尝试直接读取
        logger.info("尝试直接读取整个文件...")
        df = pd.read_csv(
            data_path,
            sep='\t',
            header=None,
            names=['user_id', 'artist_id', 'artist_name', 'plays'],
            dtype={'user_id': 'str', 'artist_id': 'str', 'plays': 'int64'}
        )
    
    elapsed = time.time() - start_time
    logger.info(f"数据加载完成，共 {len(df):,} 行，耗时 {elapsed:.2f} 秒")
    
    # 为每个用户-艺术家的交互创建时间序列
    # 策略：按照用户分组，然后为每个用户的交互分配时间戳
    # 时间戳基于播放次数和交互顺序
    logger.info("正在生成时间序列...")
    
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    
    # 添加原始索引列，以便后续恢复顺序
    df['_original_index'] = df.index
    
    # 方法：为每个用户创建一个时间范围，然后按照播放次数分配时间戳
    # 假设数据收集时间范围为1年（2009年）
    start_date = datetime(2009, 1, 1)
    end_date = datetime(2009, 12, 31, 23, 59, 59)
    total_seconds = (end_date - start_date).total_seconds()
    
    def assign_timestamps(user_group):
        """为单个用户组分配时间戳"""
        user_df = user_group.copy()
        user_df_sorted = user_df.sort_values('plays', ascending=False).reset_index(drop=True)
        n_interactions = len(user_df_sorted)
        
        if n_interactions == 0:
            return pd.Series([start_date.timestamp()] * len(user_df), index=user_df.index)
        
        max_plays = user_df_sorted['plays'].max()
        timestamps = []
        
        # 为每个交互分配时间戳，按照播放次数排序
        # 播放次数多的分配更早的时间戳
        for idx in range(n_interactions):
            row = user_df_sorted.iloc[idx]
            play_weight = row['plays']
            
            if max_plays > 0:
                # 归一化播放次数权重 (0-1)
                normalized_weight = play_weight / max_plays
                # 反向权重（播放次数多的更早）
                time_position = 1.0 - normalized_weight * 0.7  # 保留一些随机性
            else:
                time_position = 1.0
            
            # 添加一些随机性，避免完全相同的时间戳
            time_position += np.random.uniform(-0.1, 0.1) * (idx / n_interactions)
            time_position = np.clip(time_position, 0.0, 1.0)
            
            # 计算时间戳
            timestamp_seconds = start_date.timestamp() + time_position * total_seconds
            timestamps.append(timestamp_seconds)
        
        # 创建映射：从排序后的索引到原始索引
        timestamp_series = pd.Series(timestamps, index=user_df_sorted.index)
        # 映射回原始索引
        result = timestamp_series.reindex(user_df.index)
        return result
    
    # 按用户分组并应用函数
    if HAS_TQDM and show_progress:
        logger.info("正在为每个用户生成时间戳...")
        unique_users = df['user_id'].nunique()
        user_groups = tqdm(df.groupby('user_id', sort=False), desc="生成时间戳", unit="用户", total=unique_users)
    else:
        user_groups = df.groupby('user_id', sort=False)
    
    timestamp_series = pd.concat([assign_timestamps(group) for _, group in user_groups])
    
    # 按原始索引排序，确保顺序一致
    timestamp_series = timestamp_series.sort_index()
    
    # 将时间戳添加到数据框
    df['timestamp'] = timestamp_series.values
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 删除临时索引列
    df = df.drop('_original_index', axis=1)
    
    # 按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    return df


def create_time_window(datetime_series: pd.Series, window_type: str) -> pd.Series:
    """
    根据窗口类型创建时间窗口标签
    
    Args:
        datetime_series: datetime类型的Series
        window_type: 时间窗口类型 ('30min', '1h', '12h', '1d')
    
    Returns:
        时间窗口标签的Series
    """
    if window_type == "30min":
        # 半小时窗口：向下取整到最近的30分钟
        return datetime_series.dt.floor('30min')
    elif window_type == "1h":
        # 1小时窗口
        return datetime_series.dt.floor('H')
    elif window_type == "12h":
        # 半天窗口：向下取整到最近的12小时（0点和12点）
        hours = datetime_series.dt.hour
        # 将时间调整到0点或12点
        adjusted = datetime_series.dt.normalize() + pd.to_timedelta((hours // 12) * 12, unit='h')
        return adjusted
    elif window_type == "1d":
        # 一天窗口
        return datetime_series.dt.date
    else:
        raise ValueError(f"不支持的时间窗口类型: {window_type}")


def compute_user_sequence_lengths_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算不同时间窗口下每个用户的累计序列长度
    
    Args:
        df: 包含datetime列的数据框
        window_type: 时间窗口类型
        show_progress: 是否显示进度
    
    Returns:
        包含时间窗口、用户ID和累计序列长度的数据框
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"开始按{window_type}分组计算用户累计序列长度...")
    
    # 创建时间窗口标签
    time_window = create_time_window(df["datetime"], window_type)
    
    # 创建临时数据框用于分组
    group_df = pd.DataFrame({
        'time_window': time_window,
        'user_id': df['user_id'].values
    })
    
    # 计算每个时间窗口内每个用户的交互次数
    window_interactions = group_df.groupby(["time_window", "user_id"], observed=True, sort=False).size()
    window_interactions = window_interactions.reset_index(name="window_count")
    
    # 按时间窗口和用户ID排序
    window_interactions = window_interactions.sort_values(['time_window', 'user_id'])
    
    if show_progress:
        logger.info("正在计算累计序列长度...")
    
    # 对每个用户，按时间窗口排序后计算累计和
    window_interactions['sequence_length'] = window_interactions.groupby('user_id', observed=True, sort=False)['window_count'].cumsum()
    
    result = window_interactions[['time_window', 'user_id', 'sequence_length']]
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"累计序列长度计算完成: {len(result):,} 条记录，耗时 {elapsed:.2f} 秒")
    
    return result


def compute_active_users_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算每个时间窗口的活跃用户数
    
    Args:
        df: 包含datetime列的数据框
        window_type: 时间窗口类型
        show_progress: 是否显示进度
    
    Returns:
        包含时间窗口和活跃用户数的数据框
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"开始计算{window_type}时间窗口的活跃用户数...")
    
    # 创建时间窗口标签
    time_window = create_time_window(df["datetime"], window_type)
    
    # 计算每个时间窗口的唯一用户数
    active_users = df.groupby(time_window)['user_id'].nunique().reset_index()
    active_users.columns = ['time_window', 'active_user_count']
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"活跃用户数计算完成: {len(active_users)} 个时间窗口，耗时 {elapsed:.2f} 秒")
    
    return active_users


def compute_window_statistics(
    df: pd.DataFrame,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算每个时间窗口的统计指标
    
    Returns:
        包含每个时间窗口统计指标的数据框
    """
    start_time = time.time()
    if show_progress:
        logger.info("开始计算窗口统计指标...")
        unique_windows = df["time_window"].nunique()
        logger.info(f"共有 {unique_windows} 个时间窗口需要计算")
    
    # 使用groupby一次性计算所有统计指标
    grouped = df.groupby("time_window", observed=True, sort=True)["sequence_length"]
    
    # 计算基本统计量
    count_df = grouped.count().reset_index(name='count')
    mean_df = grouped.mean().reset_index(name='mean')
    std_df = grouped.std().reset_index(name='std')
    min_df = grouped.min().reset_index(name='min')
    max_df = grouped.max().reset_index(name='max')
    median_df = grouped.median().reset_index(name='median')
    
    # 合并所有统计量
    stats = count_df.merge(mean_df, on='time_window')
    stats = stats.merge(std_df, on='time_window')
    stats = stats.merge(min_df, on='time_window')
    stats = stats.merge(max_df, on='time_window')
    stats = stats.merge(median_df, on='time_window')
    
    # 确保所有数值列都是数值类型
    numeric_cols = ['count', 'mean', 'std', 'min', 'max', 'median']
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors='coerce').astype(float)
    
    # 计算方差
    stats['variance'] = stats['std'] ** 2
    
    # 填充NaN值
    stats['std'] = stats['std'].fillna(0.0).astype(float)
    stats['variance'] = stats['variance'].fillna(0.0).astype(float)
    
    # 计算百分位数
    if show_progress:
        logger.info("计算百分位数...")
    
    all_percentiles = sorted(set(percentiles + [0.25, 0.5, 0.75]))
    
    # 设置p50为median
    if 0.5 in all_percentiles:
        stats['p50'] = stats['median']
    
    # 计算其他百分位数
    percentiles_to_compute = sorted(set([p for p in all_percentiles if p != 0.5]))
    
    if percentiles_to_compute:
        for p in percentiles_to_compute:
            col_name = f"p{int(p*100)}"
            if col_name not in stats.columns:
                quantile_series = grouped.quantile(p)
                stats[col_name] = stats['time_window'].map(quantile_series).astype(float)
    
    # 确保p25和p75存在
    if 'p25' not in stats.columns:
        stats['p25'] = stats['time_window'].map(grouped.quantile(0.25)).astype(float)
    if 'p75' not in stats.columns:
        stats['p75'] = stats['time_window'].map(grouped.quantile(0.75)).astype(float)
    
    # 计算IQR
    stats['p25'] = stats['p25'].astype(float)
    stats['p75'] = stats['p75'].astype(float)
    stats['iqr'] = stats['p75'] - stats['p25']
    
    # 计算偏度和峰度
    if show_progress:
        logger.info("计算偏度和峰度...")
    
    try:
        skew_vals = grouped.skew().fillna(0.0)
        kurt_vals = grouped.apply(lambda x: x.kurtosis() if len(x) > 2 else 0.0).fillna(0.0)
        stats['skewness'] = stats['time_window'].map(skew_vals).fillna(0.0)
        stats['kurtosis'] = stats['time_window'].map(kurt_vals).fillna(0.0)
    except Exception as e:
        logger.warning(f"计算偏度和峰度时出错: {e}，将设置为0")
        stats['skewness'] = 0.0
        stats['kurtosis'] = 0.0
    
    # 确保所有数值列都是float类型
    numeric_cols = stats.select_dtypes(include=[np.number]).columns
    stats[numeric_cols] = stats[numeric_cols].astype(float)
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"统计指标计算完成，耗时 {elapsed:.2f} 秒")
    
    return stats


def compute_active_users_statistics(
    active_users_df: pd.DataFrame,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算活跃用户数的统计指标
    
    Returns:
        包含每个时间窗口活跃用户数统计的数据框
    """
    start_time = time.time()
    if show_progress:
        logger.info("开始计算活跃用户数统计指标...")
    
    stats = active_users_df.copy()
    
    # 计算滚动统计（可选）
    window_size = min(10, len(stats) // 4)  # 使用较小的窗口
    if window_size > 1:
        stats['active_user_count_mean_rolling'] = stats['active_user_count'].rolling(window=window_size, center=True).mean()
        stats['active_user_count_std_rolling'] = stats['active_user_count'].rolling(window=window_size, center=True).std()
    
    # 计算活跃用户数的基本统计量
    active_user_counts = stats['active_user_count']
    stats['mean'] = active_user_counts.mean()
    stats['std'] = active_user_counts.std()
    stats['median'] = active_user_counts.median()
    stats['min'] = active_user_counts.min()
    stats['max'] = active_user_counts.max()
    stats['variance'] = active_user_counts.var()
    
    # 计算百分位数
    for p in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        col_name = f"p{int(p*100)}"
        stats[col_name] = active_user_counts.quantile(p)
    
    stats['iqr'] = stats['p75'] - stats['p25']
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"活跃用户数统计完成，耗时 {elapsed:.2f} 秒")
    
    return stats


def plot_sequence_length_over_time(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """绘制序列长度随时间窗口的变化图（波动图）"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib未安装，无法绘图")
    
    # 确保time_window是datetime类型以便绘图
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        if isinstance(window_stats["time_window"].iloc[0], (datetime, pd.Timestamp)):
            window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
        else:
            # 如果是date类型，转换为datetime
            window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'User Interaction Sequence Length Distribution Statistics - {window_type} Time Window', fontsize=16, fontweight='bold')
    
    x = window_stats["time_window"]
    
    # 1. 均值、中位数、p90、p99 时间序列图
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
    
    # 2. 方差和标准差时间序列图
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
    
    # 3. 最小值和最大值时间序列图
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
    
    # 4. IQR (四分位距) 时间序列图
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
    """绘制更详细的统计信息图"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib未安装，无法绘图")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    x = window_stats["time_window"]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'Detailed Statistical Analysis of User Interaction Sequence Length - {window_type} Time Window', fontsize=16, fontweight='bold')
    
    # 1. 所有百分位数对比
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
    
    # 2. 偏度和峰度
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
    
    # 3. 用户数量（每个窗口的活跃用户数）
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
    
    # 4. 均值 ± 标准差区域图
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
    
    # 5. 变异系数 (CV = std/mean)
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
    
    # 6. 箱线图风格的统计摘要
    ax6 = axes[2, 1]
    if "p25" in window_stats.columns:
        ax6.fill_between(x, window_stats["p25"], window_stats["p75"], alpha=0.3, color='lightblue', label='IQR Range')
    ax6.plot(x, window_stats["median"], label="Median", marker='o', markersize=4, linewidth=2, alpha=0.8, color='blue')
    ax6.plot(x, window_stats["min"], label="Min", marker='_', markersize=8, linewidth=1, alpha=0.6, color='green')
    ax6.plot(x, window_stats["max"], label="Max", marker='_', markersize=8, linewidth=1, alpha=0.6, color='red')
    ax6.set_xlabel("Time Window", fontsize=10)
    ax6.set_ylabel("Sequence Length", fontsize=10)
    ax6.set_title("Sequence Length Distribution Range (Median, IQR, Extremes)", fontsize=12)
    ax6.legend(loc='best', fontsize=8)
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
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """绘制活跃用户数随时间窗口的变化图"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib未安装，无法绘图")
    
    active_users_stats = active_users_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(active_users_stats["time_window"]):
        active_users_stats["time_window"] = pd.to_datetime(active_users_stats["time_window"])
    
    active_users_stats = active_users_stats.sort_values("time_window")
    x = active_users_stats["time_window"]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Active User Count Distribution Statistics - {window_type} Time Window', fontsize=16, fontweight='bold')
    
    # 1. 活跃用户数时间序列
    ax1 = axes[0]
    ax1.plot(x, active_users_stats["active_user_count"], label="Active User Count", marker='o', markersize=4, linewidth=2, alpha=0.8, color='steelblue')
    if "active_user_count_mean_rolling" in active_users_stats.columns:
        ax1.plot(x, active_users_stats["active_user_count_mean_rolling"], label="Rolling Mean", marker='s', markersize=3, linewidth=1.5, alpha=0.7, color='orange', linestyle='--')
    ax1.set_xlabel("Time Window", fontsize=12)
    ax1.set_ylabel("Active User Count", fontsize=12)
    ax1.set_title("Active User Count Over Time", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. 活跃用户数分布直方图
    ax2 = axes[1]
    ax2.hist(active_users_stats["active_user_count"], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    mean_val = active_users_stats["active_user_count"].mean()
    median_val = active_users_stats["active_user_count"].median()
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
    ax2.set_xlabel("Active User Count", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Active User Count Distribution Histogram", fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_sequence_length_by_time_window(
    data_path: str,
    window_type: str = "1d",
    show_progress: bool = False,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分析不同时间窗口下用户交互序列长度分布和活跃用户数分布
    
    Returns:
        (用户序列长度数据框, 窗口统计数据框, 活跃用户数数据框)
    """
    # 加载数据
    df = load_lastfm_data(data_path, show_progress)
    
    # 调用接受DataFrame的函数进行分析
    return analyze_sequence_length_by_time_window_from_df(
        df, window_type, show_progress, percentiles
    )


def analyze_sequence_length_by_time_window_from_df(
    df: pd.DataFrame,
    window_type: str = "1d",
    show_progress: bool = False,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从已加载的DataFrame分析不同时间窗口下用户交互序列长度分布和活跃用户数分布
    
    Args:
        df: 已加载的数据框（必须包含datetime列）
        window_type: 时间窗口类型
        show_progress: 是否显示进度
        percentiles: 要计算的百分位数列表
    
    Returns:
        (用户序列长度数据框, 窗口统计数据框, 活跃用户数数据框)
    """
    total_start = time.time()
    logger.info("="*80)
    logger.info("开始分析用户交互序列长度分布和活跃用户数分布")
    logger.info(f"时间窗口类型: {window_type}")
    logger.info("="*80)
    
    # 计算每个时间窗口内每个用户的序列长度
    user_seq_lengths = compute_user_sequence_lengths_by_window(
        df, window_type, show_progress
    )
    
    # 计算每个时间窗口的统计指标
    window_stats = compute_window_statistics(user_seq_lengths, percentiles, show_progress)
    
    # 计算活跃用户数
    active_users = compute_active_users_by_window(df, window_type, show_progress)
    active_users_stats = compute_active_users_statistics(active_users, show_progress)
    
    total_elapsed = time.time() - total_start
    logger.info("="*80)
    logger.info(f"分析完成！总耗时: {total_elapsed:.2f} 秒")
    logger.info("="*80)
    
    return user_seq_lengths, window_stats, active_users_stats


def main():
    parser = argparse.ArgumentParser(description='分析 LastFM 行为数据集不同时间窗口下用户交互序列长度分布和活跃用户数分布')
    parser.add_argument(
        '--data_path',
        type=str,
        default='~/data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
        help='数据文件路径（默认: ~/data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv）'
    )
    parser.add_argument(
        '--window_type',
        type=str,
        choices=['30min', '1h', '12h', '1d'],
        default='1d',
        help='时间窗口类型: 30min/1h/12h/1d (默认: 1d)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认: 脚本所在目录的reports文件夹）'
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='显示进度信息'
    )
    parser.add_argument(
        '--all_windows',
        action='store_true',
        help='对所有时间窗口类型进行分析'
    )
    
    args = parser.parse_args()
    
    # 展开路径
    data_path = os.path.expanduser(args.data_path)
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "reports"
    else:
        output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(data_path):
        logger.error(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 确定要分析的时间窗口
    if args.all_windows:
        window_types = ['30min', '1h', '12h', '1d']
    else:
        window_types = [args.window_type]
    
    # 只加载一次数据
    logger.info("="*80)
    logger.info("正在加载数据...")
    logger.info("="*80)
    df = load_lastfm_data(data_path, args.show_progress)
    logger.info(f"数据加载完成，共 {len(df):,} 条记录")
    logger.info(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    logger.info("="*80)
    
    # 对每个时间窗口进行分析
    for window_type in window_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"开始分析时间窗口: {window_type}")
        logger.info(f"{'='*80}\n")
        
        # 执行分析（使用已加载的数据）
        user_seq_lengths, window_stats, active_users_stats = analyze_sequence_length_by_time_window_from_df(
            df,
            window_type=window_type,
            show_progress=args.show_progress,
        )
        
        # 保存CSV
        logger.info("开始保存结果...")
        stats_csv = output_dir / f"sequence_length_stats_{window_type}.csv"
        window_stats.to_csv(stats_csv, index=False)
        logger.info(f"窗口统计数据已保存至: {stats_csv}")
        
        active_users_csv = output_dir / f"active_users_stats_{window_type}.csv"
        active_users_stats.to_csv(active_users_csv, index=False)
        logger.info(f"活跃用户数统计数据已保存至: {active_users_csv}")
        
        # 保存JSON
        stats_json = output_dir / f"sequence_length_stats_{window_type}.json"
        stats_dict = window_stats.copy()
        stats_dict["time_window"] = stats_dict["time_window"].astype(str)
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump(stats_dict.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        logger.info(f"窗口统计数据JSON已保存至: {stats_json}")
        
        # 打印统计摘要
        logger.info("\n" + "="*80)
        logger.info(f"时间窗口类型: {window_type}")
        logger.info(f"时间窗口数量: {len(window_stats)}")
        logger.info("="*80)
        logger.info("\n序列长度总体统计:")
        logger.info(f"  均值: {window_stats['mean'].mean():.2f}")
        logger.info(f"  标准差: {window_stats['std'].mean():.2f}")
        logger.info(f"  中位数: {window_stats['median'].mean():.2f}")
        if "p90" in window_stats.columns:
            logger.info(f"  P90: {window_stats['p90'].mean():.2f}")
        if "p99" in window_stats.columns:
            logger.info(f"  P99: {window_stats['p99'].mean():.2f}")
        logger.info(f"  平均方差: {window_stats['variance'].mean():.2f}")
        
        logger.info("\n活跃用户数总体统计:")
        logger.info(f"  平均活跃用户数: {active_users_stats['active_user_count'].mean():.2f}")
        logger.info(f"  活跃用户数标准差: {active_users_stats['active_user_count'].std():.2f}")
        logger.info(f"  活跃用户数中位数: {active_users_stats['active_user_count'].median():.2f}")
        logger.info(f"  最小活跃用户数: {active_users_stats['active_user_count'].min():.0f}")
        logger.info(f"  最大活跃用户数: {active_users_stats['active_user_count'].max():.0f}")
        if "p90" in active_users_stats.columns:
            logger.info(f"  活跃用户数P90: {active_users_stats['p90'].iloc[0]:.0f}")
        if "p99" in active_users_stats.columns:
            logger.info(f"  活跃用户数P99: {active_users_stats['p99'].iloc[0]:.0f}")
        
        # 绘制图表
        if HAS_MATPLOTLIB:
            logger.info("\n开始生成图表...")
            
            # 序列长度统计图
            plot_main = output_dir / f"sequence_length_over_time_{window_type}.png"
            plot_sequence_length_over_time(window_stats, plot_main, window_type)
            logger.info(f"主要统计图已保存至: {plot_main}")
            
            plot_detailed = output_dir / f"sequence_length_detailed_stats_{window_type}.png"
            plot_detailed_statistics(window_stats, plot_detailed, window_type)
            logger.info(f"详细统计图已保存至: {plot_detailed}")
            
            # 活跃用户数统计图
            plot_active_users = output_dir / f"active_users_over_time_{window_type}.png"
            plot_active_users_over_time(active_users_stats, plot_active_users, window_type)
            logger.info(f"活跃用户数统计图已保存至: {plot_active_users}")
        else:
            logger.warning("matplotlib未安装，跳过绘图")
        
        logger.info(f"\n时间窗口 {window_type} 分析完成！\n")


if __name__ == '__main__':
    main()


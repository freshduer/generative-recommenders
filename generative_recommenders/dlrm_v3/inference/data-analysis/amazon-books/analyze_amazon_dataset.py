#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Books 数据集分析脚本
分析数据集的基本统计信息、分布特征等
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 尝试导入 tqdm 用于进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度显示函数
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置网格样式（不依赖 seaborn）
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # matplotlib 3.6+
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')  # matplotlib 3.5-
    except OSError:
        # 如果都没有，手动设置网格
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.axisbelow'] = True


def load_data(data_path: str) -> pd.DataFrame:
    """加载 Amazon 数据集"""
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"数据加载完成，共 {len(df)} 行")
    return df


def parse_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """解析序列数据，展开为单条交互记录（优化版本，使用向量化操作）"""
    print("正在解析序列数据...")
    
    records = []
    total_rows = len(df)
    
    # 使用 itertuples 比 iterrows 快很多（快5-10倍）
    # 批量处理，减少函数调用开销
    # 添加进度条
    if HAS_TQDM:
        # 使用 tqdm 包装迭代器
        iterator = tqdm(df.itertuples(index=False), total=total_rows, desc="Parsing sequences", unit="rows")
    else:
        # 如果没有 tqdm，使用索引迭代并显示简单进度
        iterator = df.itertuples(index=False)
        print(f"总共需要处理 {total_rows} 行数据...")
        last_progress = -1
    
    for idx, row in enumerate(iterator):
        # 如果没有 tqdm，每处理 10% 显示一次进度
        if not HAS_TQDM:
            progress = int((idx + 1) / total_rows * 100)
            if progress != last_progress and progress % 10 == 0:
                print(f"进度: {progress}% ({idx + 1}/{total_rows})")
                last_progress = progress
        user_id = row.user_id
        
        # 直接使用字符串，避免重复转换
        item_ids_str = row.sequence_item_ids if isinstance(row.sequence_item_ids, str) else str(row.sequence_item_ids)
        ratings_str = row.sequence_ratings if isinstance(row.sequence_ratings, str) else str(row.sequence_ratings)
        timestamps_str = row.sequence_timestamps if isinstance(row.sequence_timestamps, str) else str(row.sequence_timestamps)
        
        # 一次性分割并转换类型
        item_ids = [int(x) for x in item_ids_str.split(',')]
        ratings = [float(x) for x in ratings_str.split(',')]
        timestamps = [int(x) for x in timestamps_str.split(',')]
        
        # 使用元组列表，比字典列表更快
        records.extend(zip([user_id] * len(item_ids), item_ids, ratings, timestamps))
    
    # 一次性创建 DataFrame（比逐行添加快很多）
    print("\n正在创建 DataFrame...")
    expanded_df = pd.DataFrame(records, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    print(f"展开后共 {len(expanded_df)} 条交互记录")
    return expanded_df


def basic_statistics(df: pd.DataFrame, expanded_df: pd.DataFrame):
    """基本统计信息"""
    print("\n" + "="*60)
    print("基本统计信息")
    print("="*60)
    
    num_users = df['user_id'].nunique()
    num_items = expanded_df['item_id'].nunique()
    num_interactions = len(expanded_df)
    avg_interactions_per_user = num_interactions / num_users
    avg_interactions_per_item = num_interactions / num_items
    sparsity = 1 - num_interactions / (num_users * num_items)
    
    print(f"用户数量: {num_users:,}")
    print(f"物品数量: {num_items:,}")
    print(f"交互总数: {num_interactions:,}")
    print(f"平均每个用户的交互数: {avg_interactions_per_user:.2f}")
    print(f"平均每个物品的交互数: {avg_interactions_per_item:.2f}")
    print(f"数据稀疏度: {sparsity:.6f}")
    
    return {
        'num_users': int(num_users),
        'num_items': int(num_items),
        'num_interactions': int(num_interactions),
        'avg_interactions_per_user': float(avg_interactions_per_user),
        'avg_interactions_per_item': float(avg_interactions_per_item),
        'sparsity': float(sparsity)
    }


def sequence_length_statistics(df: pd.DataFrame):
    """序列长度统计"""
    print("\n" + "="*60)
    print("序列长度统计")
    print("="*60)
    
    def get_seq_length(row):
        return len(str(row['sequence_item_ids']).split(','))
    
    seq_lengths = df.apply(get_seq_length, axis=1)
    
    mean_len = seq_lengths.mean()
    median_len = seq_lengths.median()
    min_len = seq_lengths.min()
    max_len = seq_lengths.max()
    std_len = seq_lengths.std()
    
    print(f"平均序列长度: {mean_len:.2f}")
    print(f"中位数序列长度: {median_len:.2f}")
    print(f"最小序列长度: {min_len}")
    print(f"最大序列长度: {max_len}")
    print(f"标准差: {std_len:.2f}")
    
    # 分位数统计
    quantiles = {}
    print("\n序列长度分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        quantile_val = seq_lengths.quantile(q)
        quantiles[f'p{int(q*100)}'] = float(quantile_val)
        print(f"  {q*100:4.0f}%: {quantile_val:.2f}")
    
    # 序列长度分布
    length_counts = Counter(seq_lengths)
    length_distribution = {}
    print("\n序列长度分布:")
    for length in sorted(length_counts.keys())[:20]:
        count = length_counts[length]
        pct = count / len(seq_lengths) * 100
        length_distribution[int(length)] = {'count': int(count), 'percentage': float(pct)}
        print(f"  长度 {length:3d}: {count:6,} 用户 ({pct:5.2f}%)")
    
    if len(length_counts) > 20:
        print(f"  ... (共 {len(length_counts)} 种不同的序列长度)")
    
    return {
        'mean': float(mean_len),
        'median': float(median_len),
        'min': int(min_len),
        'max': int(max_len),
        'std': float(std_len),
        'quantiles': quantiles,
        'distribution': length_distribution,
        'num_unique_lengths': len(length_counts)
    }


def rating_statistics(expanded_df: pd.DataFrame):
    """评分统计"""
    print("\n" + "="*60)
    print("评分统计")
    print("="*60)
    
    mean_rating = expanded_df['rating'].mean()
    median_rating = expanded_df['rating'].median()
    std_rating = expanded_df['rating'].std()
    min_rating = expanded_df['rating'].min()
    max_rating = expanded_df['rating'].max()
    
    print(f"平均评分: {mean_rating:.2f}")
    print(f"中位数评分: {median_rating:.2f}")
    print(f"评分标准差: {std_rating:.2f}")
    print(f"最小评分: {min_rating}")
    print(f"最大评分: {max_rating}")
    
    rating_distribution = {}
    print("\n评分分布:")
    rating_counts = expanded_df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        pct = count / len(expanded_df) * 100
        rating_distribution[float(rating)] = {'count': int(count), 'percentage': float(pct)}
        print(f"  评分 {rating:.1f}: {count:10,} 次 ({pct:5.2f}%)")
    
    return {
        'mean': float(mean_rating),
        'median': float(median_rating),
        'std': float(std_rating),
        'min': float(min_rating),
        'max': float(max_rating),
        'distribution': rating_distribution
    }


def timestamp_statistics(expanded_df: pd.DataFrame):
    """时间戳统计"""
    print("\n" + "="*60)
    print("时间戳统计")
    print("="*60)
    
    timestamps = expanded_df['timestamp']
    min_ts = timestamps.min()
    max_ts = timestamps.max()
    
    min_date = datetime.fromtimestamp(min_ts)
    max_date = datetime.fromtimestamp(max_ts)
    span_days = (max_ts - min_ts) / (24 * 3600)
    
    print(f"最早时间戳: {min_ts} ({min_date.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"最晚时间戳: {max_ts} ({max_date.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"时间跨度: {span_days:.0f} 天 ({span_days/365:.2f} 年)")
    
    # 按年份统计
    expanded_df['year'] = pd.to_datetime(expanded_df['timestamp'], unit='s').dt.year
    year_counts = expanded_df['year'].value_counts().sort_index()
    year_distribution = {}
    print("\n按年份统计交互数:")
    for year, count in year_counts.items():
        pct = count / len(expanded_df) * 100
        year_distribution[int(year)] = {'count': int(count), 'percentage': float(pct)}
        print(f"  {year}: {count:10,} 次 ({pct:5.2f}%)")
    
    return {
        'min_timestamp': int(min_ts),
        'max_timestamp': int(max_ts),
        'min_date': min_date.strftime('%Y-%m-%d %H:%M:%S'),
        'max_date': max_date.strftime('%Y-%m-%d %H:%M:%S'),
        'span_days': float(span_days),
        'span_years': float(span_days / 365),
        'year_distribution': year_distribution
    }


def user_activity_statistics(expanded_df: pd.DataFrame):
    """用户活跃度统计"""
    print("\n" + "="*60)
    print("用户活跃度统计")
    print("="*60)
    
    user_counts = expanded_df.groupby('user_id').size()
    
    mean_interactions = user_counts.mean()
    median_interactions = user_counts.median()
    min_interactions = user_counts.min()
    max_interactions = user_counts.max()
    
    print(f"平均每个用户的交互数: {mean_interactions:.2f}")
    print(f"中位数每个用户的交互数: {median_interactions:.2f}")
    print(f"最少交互数: {min_interactions}")
    print(f"最多交互数: {max_interactions}")
    
    quantiles = {}
    print("\n用户交互数分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        quantile_val = user_counts.quantile(q)
        quantiles[f'p{int(q*100)}'] = float(quantile_val)
        print(f"  {q*100:4.0f}%: {quantile_val:.2f}")
    
    return {
        'mean': float(mean_interactions),
        'median': float(median_interactions),
        'min': int(min_interactions),
        'max': int(max_interactions),
        'quantiles': quantiles
    }


def item_popularity_statistics(expanded_df: pd.DataFrame):
    """物品流行度统计"""
    print("\n" + "="*60)
    print("物品流行度统计")
    print("="*60)
    
    item_counts = expanded_df.groupby('item_id').size()
    
    mean_interactions = item_counts.mean()
    median_interactions = item_counts.median()
    min_interactions = item_counts.min()
    max_interactions = item_counts.max()
    
    print(f"平均每个物品的交互数: {mean_interactions:.2f}")
    print(f"中位数每个物品的交互数: {median_interactions:.2f}")
    print(f"最少交互数: {min_interactions}")
    print(f"最多交互数: {max_interactions}")
    
    quantiles = {}
    print("\n物品交互数分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        quantile_val = item_counts.quantile(q)
        quantiles[f'p{int(q*100)}'] = float(quantile_val)
        print(f"  {q*100:4.0f}%: {quantile_val:.2f}")
    
    # 长尾分布分析
    sorted_counts = item_counts.sort_values(ascending=False)
    top_10_pct_items = sorted_counts[:int(len(sorted_counts) * 0.1)]
    top_10_pct_interactions = top_10_pct_items.sum()
    top_10_pct_ratio = top_10_pct_interactions / len(expanded_df) * 100
    print(f"\n长尾分布分析:")
    print(f"  前 10% 的物品占总交互数的比例: {top_10_pct_ratio:.2f}%")
    
    return {
        'mean': float(mean_interactions),
        'median': float(median_interactions),
        'min': int(min_interactions),
        'max': int(max_interactions),
        'quantiles': quantiles,
        'top_10_percent_ratio': float(top_10_pct_ratio)
    }


def cold_start_analysis(expanded_df: pd.DataFrame):
    """冷启动问题分析"""
    print("\n" + "="*60)
    print("冷启动问题分析")
    print("="*60)
    
    item_counts = expanded_df.groupby('item_id').size()
    user_counts = expanded_df.groupby('user_id').size()
    
    # 定义冷启动阈值
    cold_item_threshold = 5
    cold_user_threshold = 5
    
    cold_items = (item_counts < cold_item_threshold).sum()
    cold_users = (user_counts < cold_user_threshold).sum()
    cold_items_pct = cold_items / len(item_counts) * 100
    cold_users_pct = cold_users / len(user_counts) * 100
    
    print(f"交互数 < {cold_item_threshold} 的物品数: {cold_items:,} ({cold_items_pct:.2f}%)")
    print(f"交互数 < {cold_user_threshold} 的用户数: {cold_users:,} ({cold_users_pct:.2f}%)")
    
    return {
        'cold_item_threshold': cold_item_threshold,
        'cold_user_threshold': cold_user_threshold,
        'cold_items': int(cold_items),
        'cold_items_percentage': float(cold_items_pct),
        'cold_users': int(cold_users),
        'cold_users_percentage': float(cold_users_pct)
    }


def analyze_user_repeat_visits(expanded_df: pd.DataFrame):
    """分析用户重复访问（同一用户访问同一物品多次）"""
    print("\n" + "="*60)
    print("用户重复访问分析")
    print("="*60)
    
    # 统计每个用户-物品对的访问次数
    user_item_counts = expanded_df.groupby(['user_id', 'item_id']).size()
    
    # 找出重复访问的用户-物品对（访问次数 > 1）
    repeat_visits = user_item_counts[user_item_counts > 1]
    
    num_repeat_pairs = len(repeat_visits)
    num_total_pairs = len(user_item_counts)
    num_users_with_repeat = repeat_visits.reset_index()['user_id'].nunique() if num_repeat_pairs > 0 else 0
    num_total_users = expanded_df['user_id'].nunique()
    repeat_pairs_pct = num_repeat_pairs / num_total_pairs * 100 if num_total_pairs > 0 else 0
    users_with_repeat_pct = num_users_with_repeat / num_total_users * 100 if num_total_users > 0 else 0
    avg_visits_per_pair = float(user_item_counts.mean())
    
    # 处理没有重复访问的情况
    if num_repeat_pairs > 0:
        avg_repeat_visits = float(repeat_visits.mean())
        max_repeat_visits = int(repeat_visits.max())
    else:
        avg_repeat_visits = 0.0
        max_repeat_visits = 0
    
    print(f"总用户-物品对数量: {num_total_pairs:,}")
    print(f"重复访问的用户-物品对数量: {num_repeat_pairs:,} ({repeat_pairs_pct:.2f}%)")
    print(f"有重复访问的用户数: {num_users_with_repeat:,} / {num_total_users:,} ({users_with_repeat_pct:.2f}%)")
    print(f"平均每个用户-物品对的访问次数: {avg_visits_per_pair:.2f}")
    if num_repeat_pairs > 0:
        print(f"重复访问的平均访问次数: {avg_repeat_visits:.2f}")
        print(f"最大重复访问次数: {max_repeat_visits}")
    else:
        print("重复访问的平均访问次数: 0 (无重复访问)")
        print("最大重复访问次数: 0 (无重复访问)")
    
    # 重复访问次数分布
    repeat_distribution = {}
    if num_repeat_pairs > 0:
        print("\n重复访问次数分布:")
        repeat_counts_dist = repeat_visits.value_counts().sort_index()
        for count, num_pairs in repeat_counts_dist.head(10).items():
            repeat_distribution[int(count)] = int(num_pairs)
            print(f"  访问 {count} 次: {num_pairs:,} 个用户-物品对")
    else:
        print("\n重复访问次数分布: 无重复访问")
    
    return {
        'total_pairs': int(num_total_pairs),
        'repeat_pairs': int(num_repeat_pairs),
        'repeat_pairs_percentage': float(repeat_pairs_pct),
        'users_with_repeat': int(num_users_with_repeat),
        'total_users': int(num_total_users),
        'users_with_repeat_percentage': float(users_with_repeat_pct),
        'avg_visits_per_pair': float(avg_visits_per_pair),
        'avg_repeat_visits': float(avg_repeat_visits),
        'max_repeat_visits': int(max_repeat_visits),
        'repeat_distribution': repeat_distribution
    }


def analyze_time_interval_visits(expanded_df: pd.DataFrame, output_dir: str):
    """按时间段统计访问用户量和重复访问用户数"""
    print("\n" + "="*60)
    print("时间段访问统计")
    print("="*60)
    
    # 将时间戳转换为日期
    expanded_df['date'] = pd.to_datetime(expanded_df['timestamp'], unit='s')
    expanded_df['date_str'] = expanded_df['date'].dt.date
    
    # 按日期统计
    daily_stats = []
    
    for date_str in sorted(expanded_df['date_str'].unique()):
        day_data = expanded_df[expanded_df['date_str'] == date_str]
        
        # 该日期的唯一用户数
        unique_users = day_data['user_id'].nunique()
        
        # 该日期访问的用户-物品对
        user_item_pairs = day_data.groupby(['user_id', 'item_id']).size()
        # 重复访问的用户-物品对（同一天多次访问同一物品）
        repeat_pairs = user_item_pairs[user_item_pairs > 1]
        num_repeat_pairs = len(repeat_pairs)
        users_with_repeat = repeat_pairs.reset_index()['user_id'].nunique() if len(repeat_pairs) > 0 else 0
        
        daily_stats.append({
            'date': date_str,
            'unique_users': unique_users,
            'users_with_repeat': users_with_repeat,
            'repeat_ratio': users_with_repeat / unique_users if unique_users > 0 else 0,
            'num_interactions': len(day_data),
            'num_repeat_pairs': num_repeat_pairs
        })
    
    daily_df = pd.DataFrame(daily_stats)
    
    total_days = len(daily_df)
    avg_daily_users = daily_df['unique_users'].mean()
    avg_daily_repeat_users = daily_df['users_with_repeat'].mean()
    avg_repeat_ratio = daily_df['repeat_ratio'].mean() * 100
    
    print(f"总天数: {total_days}")
    print(f"平均每日访问用户数: {avg_daily_users:.2f}")
    print(f"平均每日重复访问用户数: {avg_daily_repeat_users:.2f}")
    print(f"平均重复访问用户比例: {avg_repeat_ratio:.2f}%")
    
    # 绘制时间序列图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 每日用户数
    axes[0].plot(daily_df['date'], daily_df['unique_users'], label='Total Users', linewidth=1.5)
    axes[0].plot(daily_df['date'], daily_df['users_with_repeat'], label='Users with Repeat Visits', linewidth=1.5, color='orange')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Number of Users', fontsize=12)
    axes[0].set_title('Daily User Visit Statistics', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 重复访问比例
    axes[1].plot(daily_df['date'], daily_df['repeat_ratio'] * 100, label='Repeat Visit User Ratio', linewidth=1.5, color='green')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Ratio (%)', fontsize=12)
    axes[1].set_title('Daily Repeat Visit User Ratio', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_interval_visits.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n时间序列图已保存到: {os.path.join(output_dir, 'time_interval_visits.png')}")
    
    # 转换日期为字符串以便JSON序列化
    daily_stats_json = []
    for stat in daily_stats:
        daily_stats_json.append({
            'date': str(stat['date']),
            'unique_users': int(stat['unique_users']),
            'users_with_repeat': int(stat['users_with_repeat']),
            'repeat_ratio': float(stat['repeat_ratio']),
            'num_interactions': int(stat['num_interactions']),
            'num_repeat_pairs': int(stat['num_repeat_pairs'])
        })
    
    return {
        'total_days': int(total_days),
        'avg_daily_users': float(avg_daily_users),
        'avg_daily_repeat_users': float(avg_daily_repeat_users),
        'avg_repeat_ratio': float(avg_repeat_ratio),
        'daily_stats': daily_stats_json
    }


def plot_sequence_length_distribution(df: pd.DataFrame, output_dir: str):
    """绘制用户序列长度分布图"""
    print("\n正在绘制用户序列长度分布图...")
    
    def get_seq_length(row):
        return len(str(row['sequence_item_ids']).split(','))
    
    seq_lengths = df.apply(get_seq_length, axis=1)
    
    # 计算合适的bins数量（基于数据范围）
    min_len = seq_lengths.min()
    max_len = seq_lengths.max()
    # 使用更多的bins，或者基于数据范围动态计算
    num_bins = min(200, max(50, int((max_len - min_len) / 2)))  # 至少50个，最多200个bins
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 直方图 - 使用更多bins并限制显示范围到95%分位数
    p95 = seq_lengths.quantile(0.95)
    filtered_lengths = seq_lengths[seq_lengths <= p95]
    
    axes[0].hist(filtered_lengths, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Number of Users', fontsize=12)
    axes[0].set_title(f'User Sequence Length Distribution (Histogram, showing up to 95th percentile: {p95:.0f})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(seq_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {seq_lengths.mean():.2f}')
    axes[0].axvline(seq_lengths.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {seq_lengths.median():.2f}')
    axes[0].legend()
    
    # 对数尺度直方图（更好地显示长尾分布）
    axes[1].hist(seq_lengths, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('Number of Users (Log Scale)', fontsize=12)
    axes[1].set_title('User Sequence Length Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"序列长度分布图已保存到: {os.path.join(output_dir, 'sequence_length_distribution.png')}")


def plot_item_access_cdf(expanded_df: pd.DataFrame, output_dir: str):
    """绘制物品访问CDF图"""
    print("\n正在绘制物品访问CDF图...")
    
    item_counts = expanded_df.groupby('item_id').size().sort_values(ascending=False)
    
    # 计算CDF
    sorted_counts = item_counts.values
    n = len(sorted_counts)
    x = sorted_counts
    y = np.arange(1, n + 1) / n
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 线性尺度CDF
    axes[0].plot(x, y * 100, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Item Access Count', fontsize=12)
    axes[0].set_ylabel('Cumulative Distribution (%)', fontsize=12)
    axes[0].set_title('Item Access Count CDF (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(item_counts.median(), color='red', linestyle='--', linewidth=2, 
                    label=f'Median: {item_counts.median():.2f}')
    axes[0].legend()
    
    # 对数尺度CDF（更好地显示长尾分布）
    axes[1].plot(x, y * 100, linewidth=2, color='steelblue')
    axes[1].set_xlabel('Item Access Count (Log Scale)', fontsize=12)
    axes[1].set_ylabel('Cumulative Distribution (%)', fontsize=12)
    axes[1].set_title('Item Access Count CDF (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(item_counts.median(), color='red', linestyle='--', linewidth=2, 
                    label=f'Median: {item_counts.median():.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'item_access_cdf.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"物品访问CDF图已保存到: {os.path.join(output_dir, 'item_access_cdf.png')}")


def save_results_to_json(results: dict, output_dir: str):
    """将所有分析结果保存到JSON文件"""
    json_path = os.path.join(output_dir, 'analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存到: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='分析 Amazon Books 数据集')
    parser.add_argument(
        '--data_path',
        type=str,
        default='~/data/amzn_books/sasrec_format.csv',
        help='数据文件路径（默认: ~/data/amzn_books/sasrec_format.csv）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./amazon_analysis_output',
        help='输出目录（默认: ./amazon_analysis_output）'
    )
    parser.add_argument(
        '--skip_expand',
        action='store_true',
        help='跳过展开序列数据（如果只需要序列统计）'
    )
    
    args = parser.parse_args()
    
    # 展开路径
    data_path = os.path.expanduser(args.data_path)
    output_dir = os.path.expanduser(args.output_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 加载数据
    df = load_data(data_path)
    
    # 基本统计（基于序列数据）
    print("\n" + "="*60)
    print("数据集概览（基于序列数据）")
    print("="*60)
    print(f"用户序列数: {len(df):,}")
    
    # 收集所有分析结果
    all_results = {
        'dataset': os.path.basename(data_path),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_sequences': int(len(df))
    }
    
    # 序列长度统计
    seq_length_stats = sequence_length_statistics(df)
    all_results['sequence_length_statistics'] = seq_length_stats
    
    # 绘制序列长度分布图
    plot_sequence_length_distribution(df, output_dir)
    
    if not args.skip_expand:
        # 展开数据用于详细分析
        expanded_df = parse_sequences(df)
        
        # 各项统计
        basic_stats = basic_statistics(df, expanded_df)
        all_results['basic_statistics'] = basic_stats
        
        rating_stats = rating_statistics(expanded_df)
        all_results['rating_statistics'] = rating_stats
        
        timestamp_stats = timestamp_statistics(expanded_df)
        all_results['timestamp_statistics'] = timestamp_stats
        
        user_activity_stats = user_activity_statistics(expanded_df)
        all_results['user_activity_statistics'] = user_activity_stats
        
        item_popularity_stats = item_popularity_statistics(expanded_df)
        all_results['item_popularity_statistics'] = item_popularity_stats
        
        cold_start_stats = cold_start_analysis(expanded_df)
        all_results['cold_start_analysis'] = cold_start_stats
        
        # 用户重复访问分析
        repeat_visit_stats = analyze_user_repeat_visits(expanded_df)
        all_results['user_repeat_visits'] = repeat_visit_stats
        
        # 时间段访问统计
        time_interval_stats = analyze_time_interval_visits(expanded_df, output_dir)
        all_results['time_interval_statistics'] = time_interval_stats
        
        # 绘制物品访问CDF图
        plot_item_access_cdf(expanded_df, output_dir)
        
        # 保存每日统计数据到CSV（保留原有功能）
        daily_df = pd.DataFrame(time_interval_stats['daily_stats'])
        daily_df.to_csv(os.path.join(output_dir, 'daily_visit_statistics.csv'), index=False)
        print(f"\n每日统计结果已保存到: {os.path.join(output_dir, 'daily_visit_statistics.csv')}")
    
    # 保存所有结果到JSON
    save_results_to_json(all_results, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print(f"所有图表已保存到: {output_dir}")
    print(f"分析结果JSON已保存到: {os.path.join(output_dir, 'analysis_results.json')}")
    print("="*60)


if __name__ == '__main__':
    main()


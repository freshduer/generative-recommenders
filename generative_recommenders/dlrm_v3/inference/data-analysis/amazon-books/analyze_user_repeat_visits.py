#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Books 用户重复访问分析脚本
统计用户是否会重复访问相同的item，以及每个用户的重复访问频率
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
    """解析序列数据，展开为单条交互记录"""
    print("正在解析序列数据...")
    
    records = []
    total_rows = len(df)
    
    if HAS_TQDM:
        iterator = tqdm(df.itertuples(index=False), total=total_rows, desc="Parsing sequences", unit="rows")
    else:
        iterator = df.itertuples(index=False)
        print(f"总共需要处理 {total_rows} 行数据...")
        last_progress = -1
    
    for idx, row in enumerate(iterator):
        if not HAS_TQDM:
            progress = int((idx + 1) / total_rows * 100)
            if progress != last_progress and progress % 10 == 0:
                print(f"进度: {progress}% ({idx + 1}/{total_rows})")
                last_progress = progress
        
        user_id = row.user_id
        
        item_ids_str = row.sequence_item_ids if isinstance(row.sequence_item_ids, str) else str(row.sequence_item_ids)
        ratings_str = row.sequence_ratings if isinstance(row.sequence_ratings, str) else str(row.sequence_ratings)
        timestamps_str = row.sequence_timestamps if isinstance(row.sequence_timestamps, str) else str(row.sequence_timestamps)
        
        item_ids = [int(x) for x in item_ids_str.split(',')]
        ratings = [float(x) for x in ratings_str.split(',')]
        timestamps = [int(x) for x in timestamps_str.split(',')]
        
        records.extend(zip([user_id] * len(item_ids), item_ids, ratings, timestamps))
    
    print("\n正在创建 DataFrame...")
    expanded_df = pd.DataFrame(records, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    print(f"展开后共 {len(expanded_df)} 条交互记录")
    return expanded_df


def analyze_user_repeat_visits(expanded_df: pd.DataFrame):
    """分析每个用户的重复访问情况（优化版本，使用向量化操作）"""
    print("\n" + "="*60)
    print("用户重复访问分析")
    print("="*60)
    
    # 统计每个用户-物品对的访问次数
    print("正在统计每个用户-物品对的访问次数...")
    user_item_counts = expanded_df.groupby(['user_id', 'item_id']).size().reset_index(name='visit_count')
    
    # 使用向量化操作统计每个用户的访问情况（避免循环）
    print("正在统计每个用户的访问情况（向量化操作）...")
    
    # 按用户分组，一次性计算所有统计量
    user_stats = user_item_counts.groupby('user_id').agg({
        'visit_count': [
            ('total_visits', 'sum'),           # 总访问次数
            ('unique_items', 'count'),         # 唯一item数量
            ('max_repeat_visits', 'max'),      # 最大重复访问次数
        ]
    }).reset_index()
    
    # 展平列名
    user_stats.columns = ['user_id', 'total_visits', 'unique_items', 'max_repeat_visits']
    
    # 计算重复访问的item数量（访问次数>1的item数）
    repeat_items_df = user_item_counts[user_item_counts['visit_count'] > 1].groupby('user_id').agg({
        'visit_count': [
            ('repeat_items', 'count'),         # 重复访问的item数量
            ('repeat_visits', 'sum'),         # 重复访问的总次数
            ('avg_repeat_visits', 'mean'),     # 重复访问item的平均访问次数
        ]
    }).reset_index()
    
    # 展平列名
    repeat_items_df.columns = ['user_id', 'repeat_items', 'repeat_visits', 'avg_repeat_visits']
    
    # 合并结果
    user_stats_df = user_stats.merge(repeat_items_df, on='user_id', how='left')
    
    # 填充缺失值（没有重复访问的用户）
    user_stats_df['repeat_items'] = user_stats_df['repeat_items'].fillna(0).astype(int)
    user_stats_df['repeat_visits'] = user_stats_df['repeat_visits'].fillna(0).astype(int)
    user_stats_df['avg_repeat_visits'] = user_stats_df['avg_repeat_visits'].fillna(0.0)
    
    # 计算其他指标
    user_stats_df['repeat_frequency'] = (user_stats_df['repeat_visits'] / user_stats_df['total_visits']).fillna(0.0)
    user_stats_df['avg_visits_per_item'] = (user_stats_df['total_visits'] / user_stats_df['unique_items']).fillna(0.0)
    user_stats_df['has_repeat'] = user_stats_df['repeat_items'] > 0
    
    # 整体统计
    total_users = len(user_stats_df)
    users_with_repeat = (user_stats_df['has_repeat'] == True).sum()
    users_without_repeat = total_users - users_with_repeat
    
    print(f"\n总体统计:")
    print(f"  总用户数: {total_users:,}")
    print(f"  有重复访问的用户数: {users_with_repeat:,} ({users_with_repeat/total_users*100:.2f}%)")
    print(f"  无重复访问的用户数: {users_without_repeat:,} ({users_without_repeat/total_users*100:.2f}%)")
    
    # 有重复访问用户的统计
    users_with_repeat_df = user_stats_df[user_stats_df['has_repeat'] == True]
    if len(users_with_repeat_df) > 0:
        print(f"\n有重复访问用户的统计:")
        print(f"  平均总访问次数: {users_with_repeat_df['total_visits'].mean():.2f}")
        print(f"  平均唯一item数: {users_with_repeat_df['unique_items'].mean():.2f}")
        print(f"  平均重复访问item数: {users_with_repeat_df['repeat_items'].mean():.2f}")
        print(f"  平均重复访问次数: {users_with_repeat_df['repeat_visits'].mean():.2f}")
        print(f"  平均重复访问频率: {users_with_repeat_df['repeat_frequency'].mean()*100:.2f}%")
        print(f"  平均每个item访问次数: {users_with_repeat_df['avg_visits_per_item'].mean():.2f}")
        print(f"  平均重复item访问次数: {users_with_repeat_df['avg_repeat_visits'].mean():.2f}")
        print(f"  最大重复访问次数: {users_with_repeat_df['max_repeat_visits'].max()}")
    
    # 重复访问频率分布
    print(f"\n重复访问频率分布:")
    repeat_freq_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    repeat_freq_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                          '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    user_stats_df['repeat_freq_bin'] = pd.cut(user_stats_df['repeat_frequency'], 
                                               bins=repeat_freq_bins, 
                                               labels=repeat_freq_labels,
                                               include_lowest=True)
    freq_dist = user_stats_df['repeat_freq_bin'].value_counts().sort_index()
    for bin_label, count in freq_dist.items():
        pct = count / total_users * 100
        print(f"  {bin_label}: {count:,} 用户 ({pct:.2f}%)")
    
    # 重复访问item数量分布
    print(f"\n重复访问item数量分布:")
    repeat_items_dist = user_stats_df[user_stats_df['has_repeat'] == True]['repeat_items'].value_counts().sort_index()
    print(f"  有重复访问的用户中:")
    for num_items, count in repeat_items_dist.head(20).items():
        pct = count / users_with_repeat * 100 if users_with_repeat > 0 else 0
        print(f"    {num_items} 个重复item: {count:,} 用户 ({pct:.2f}%)")
    
    return user_stats_df


def plot_repeat_visit_statistics(user_stats_df: pd.DataFrame, output_dir: str):
    """绘制重复访问统计图表"""
    print("\n正在绘制重复访问统计图表...")
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 用户重复访问比例饼图
    ax1 = plt.subplot(3, 3, 1)
    users_with_repeat = (user_stats_df['has_repeat'] == True).sum()
    users_without_repeat = len(user_stats_df) - users_with_repeat
    ax1.pie([users_with_repeat, users_without_repeat], 
            labels=[f'With Repeat\n({users_with_repeat:,})', f'No Repeat\n({users_without_repeat:,})'],
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('User Repeat Visit Ratio', fontsize=12, fontweight='bold')
    
    # 2. 重复访问频率分布直方图
    ax2 = plt.subplot(3, 3, 2)
    users_with_repeat_df = user_stats_df[user_stats_df['has_repeat'] == True]
    if len(users_with_repeat_df) > 0:
        ax2.hist(users_with_repeat_df['repeat_frequency'] * 100, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Repeat Visit Frequency (%)', fontsize=10)
        ax2.set_ylabel('Number of Users', fontsize=10)
        ax2.set_title('Repeat Visit Frequency Distribution', fontsize=12, fontweight='bold')
        ax2.axvline(users_with_repeat_df['repeat_frequency'].mean() * 100, 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {users_with_repeat_df["repeat_frequency"].mean()*100:.2f}%')
        ax2.legend()
    
    # 3. 重复访问item数量分布
    ax3 = plt.subplot(3, 3, 3)
    if len(users_with_repeat_df) > 0:
        repeat_items_dist = users_with_repeat_df['repeat_items'].value_counts().sort_index()
        top_20 = repeat_items_dist.head(20)
        ax3.bar(range(len(top_20)), top_20.values, edgecolor='black', alpha=0.7)
        ax3.set_xticks(range(len(top_20)))
        ax3.set_xticklabels(top_20.index, rotation=45, ha='right')
        ax3.set_xlabel('Number of Repeat Items', fontsize=10)
        ax3.set_ylabel('Number of Users', fontsize=10)
        ax3.set_title('Repeat Items Distribution (Top 20)', fontsize=12, fontweight='bold')
    
    # 4. 总访问次数 vs 重复访问频率散点图
    ax4 = plt.subplot(3, 3, 4)
    if len(users_with_repeat_df) > 0:
        # 采样以避免点太多
        sample_size = min(10000, len(users_with_repeat_df))
        sample_df = users_with_repeat_df.sample(n=sample_size, random_state=42)
        ax4.scatter(sample_df['total_visits'], sample_df['repeat_frequency'] * 100, 
                   alpha=0.3, s=1)
        ax4.set_xlabel('Total Visits', fontsize=10)
        ax4.set_ylabel('Repeat Visit Frequency (%)', fontsize=10)
        ax4.set_title('Total Visits vs Repeat Frequency', fontsize=12, fontweight='bold')
        ax4.set_xscale('log')
    
    # 5. 唯一item数 vs 重复访问频率散点图
    ax5 = plt.subplot(3, 3, 5)
    if len(users_with_repeat_df) > 0:
        sample_df = users_with_repeat_df.sample(n=sample_size, random_state=42)
        ax5.scatter(sample_df['unique_items'], sample_df['repeat_frequency'] * 100, 
                   alpha=0.3, s=1)
        ax5.set_xlabel('Unique Items', fontsize=10)
        ax5.set_ylabel('Repeat Visit Frequency (%)', fontsize=10)
        ax5.set_title('Unique Items vs Repeat Frequency', fontsize=12, fontweight='bold')
        ax5.set_xscale('log')
    
    # 6. 平均每个item访问次数分布
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(user_stats_df['avg_visits_per_item'], bins=50, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Avg Visits per Item', fontsize=10)
    ax6.set_ylabel('Number of Users', fontsize=10)
    ax6.set_title('Avg Visits per Item Distribution', fontsize=12, fontweight='bold')
    ax6.axvline(user_stats_df['avg_visits_per_item'].mean(), 
               color='red', linestyle='--', linewidth=2, 
               label=f'均值: {user_stats_df["avg_visits_per_item"].mean():.2f}')
    ax6.legend()
    ax6.set_xscale('log')
    
    # 7. 重复访问次数分布
    ax7 = plt.subplot(3, 3, 7)
    if len(users_with_repeat_df) > 0:
        ax7.hist(users_with_repeat_df['repeat_visits'], bins=50, edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Repeat Visits', fontsize=10)
        ax7.set_ylabel('Number of Users', fontsize=10)
        ax7.set_title('Repeat Visits Distribution', fontsize=12, fontweight='bold')
        ax7.axvline(users_with_repeat_df['repeat_visits'].mean(), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {users_with_repeat_df["repeat_visits"].mean():.2f}')
        ax7.legend()
        ax7.set_xscale('log')
    
    # 8. 最大重复访问次数分布
    ax8 = plt.subplot(3, 3, 8)
    if len(users_with_repeat_df) > 0:
        ax8.hist(users_with_repeat_df['max_repeat_visits'], bins=50, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Max Repeat Visits', fontsize=10)
        ax8.set_ylabel('Number of Users', fontsize=10)
        ax8.set_title('Max Repeat Visits Distribution', fontsize=12, fontweight='bold')
        ax8.axvline(users_with_repeat_df['max_repeat_visits'].mean(), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {users_with_repeat_df["max_repeat_visits"].mean():.2f}')
        ax8.legend()
        ax8.set_xscale('log')
    
    # 9. 重复访问频率CDF
    ax9 = plt.subplot(3, 3, 9)
    if len(users_with_repeat_df) > 0:
        sorted_freq = np.sort(users_with_repeat_df['repeat_frequency'] * 100)
        y = np.arange(1, len(sorted_freq) + 1) / len(sorted_freq) * 100
        ax9.plot(sorted_freq, y, linewidth=2)
        ax9.set_xlabel('Repeat Visit Frequency (%)', fontsize=10)
        ax9.set_ylabel('Cumulative Distribution (%)', fontsize=10)
        ax9.set_title('Repeat Visit Frequency CDF', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.axvline(users_with_repeat_df['repeat_frequency'].mean() * 100, 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {users_with_repeat_df["repeat_frequency"].mean()*100:.2f}%')
        ax9.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_repeat_visit_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {os.path.join(output_dir, 'user_repeat_visit_statistics.png')}")


def save_statistics(user_stats_df: pd.DataFrame, output_dir: str):
    """保存统计结果到CSV和JSON"""
    # 保存详细数据到CSV
    csv_path = os.path.join(output_dir, 'user_repeat_visit_statistics.csv')
    user_stats_df.to_csv(csv_path, index=False)
    print(f"\n详细统计数据已保存到: {csv_path}")
    
    # 计算汇总统计并保存到JSON
    total_users = len(user_stats_df)
    users_with_repeat = (user_stats_df['has_repeat'] == True).sum()
    users_without_repeat = total_users - users_with_repeat
    users_with_repeat_df = user_stats_df[user_stats_df['has_repeat'] == True]
    
    summary_stats = {
        'total_users': int(total_users),
        'users_with_repeat': int(users_with_repeat),
        'users_without_repeat': int(users_without_repeat),
        'users_with_repeat_percentage': float(users_with_repeat / total_users * 100),
        'users_without_repeat_percentage': float(users_without_repeat / total_users * 100),
    }
    
    if len(users_with_repeat_df) > 0:
        summary_stats['users_with_repeat_stats'] = {
            'mean_total_visits': float(users_with_repeat_df['total_visits'].mean()),
            'mean_unique_items': float(users_with_repeat_df['unique_items'].mean()),
            'mean_repeat_items': float(users_with_repeat_df['repeat_items'].mean()),
            'mean_repeat_visits': float(users_with_repeat_df['repeat_visits'].mean()),
            'mean_repeat_frequency': float(users_with_repeat_df['repeat_frequency'].mean()),
            'mean_avg_visits_per_item': float(users_with_repeat_df['avg_visits_per_item'].mean()),
            'mean_avg_repeat_visits': float(users_with_repeat_df['avg_repeat_visits'].mean()),
            'max_repeat_visits': int(users_with_repeat_df['max_repeat_visits'].max()),
        }
    
    summary_stats['all_users_stats'] = {
        'mean_total_visits': float(user_stats_df['total_visits'].mean()),
        'mean_unique_items': float(user_stats_df['unique_items'].mean()),
        'mean_repeat_items': float(user_stats_df['repeat_items'].mean()),
        'mean_repeat_visits': float(user_stats_df['repeat_visits'].mean()),
        'mean_repeat_frequency': float(user_stats_df['repeat_frequency'].mean()),
        'mean_avg_visits_per_item': float(user_stats_df['avg_visits_per_item'].mean()),
    }
    
    json_path = os.path.join(output_dir, 'user_repeat_visit_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"汇总统计已保存到: {json_path}")


def analyze_top_users_distribution(user_stats_df: pd.DataFrame, output_dir: str):
    """分析热门用户占比，统计前10%用户占据的访问比例"""
    print("\n" + "="*60)
    print("热门用户占比分析")
    print("="*60)
    
    # 按总访问次数降序排序
    sorted_users = user_stats_df.sort_values('total_visits', ascending=False).reset_index(drop=True)
    
    # 计算累积访问次数和累积比例
    total_visits_all = sorted_users['total_visits'].sum()
    sorted_users['cumulative_visits'] = sorted_users['total_visits'].cumsum()
    sorted_users['cumulative_visits_ratio'] = sorted_users['cumulative_visits'] / total_visits_all
    
    # 计算用户排名比例
    sorted_users['user_rank_ratio'] = (sorted_users.index + 1) / len(sorted_users)
    
    # 统计不同百分位用户的访问占比
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    percentile_stats = {}
    
    print(f"\n总访问次数: {total_visits_all:,}")
    print(f"总用户数: {len(sorted_users):,}")
    print(f"\n不同百分位用户的访问占比:")
    
    for pct in percentiles:
        num_users = int(len(sorted_users) * pct / 100)
        if num_users > 0:
            top_users_visits = sorted_users.iloc[:num_users]['total_visits'].sum()
            visits_ratio = top_users_visits / total_visits_all * 100
            percentile_stats[f'top_{pct}_pct'] = {
                'num_users': num_users,
                'visits': int(top_users_visits),
                'visits_ratio': float(visits_ratio)
            }
            print(f"  前 {pct:2d}% 用户 ({num_users:7,} 用户): {visits_ratio:6.2f}% 的访问 ({top_users_visits:12,} 次)")
    
    # 特别关注前10%
    top_10_pct_users = int(len(sorted_users) * 0.1)
    top_10_pct_visits = sorted_users.iloc[:top_10_pct_users]['total_visits'].sum()
    top_10_pct_ratio = top_10_pct_visits / total_visits_all * 100
    
    print(f"\n重点统计:")
    print(f"  前 10% 用户 ({top_10_pct_users:,} 用户) 占据 {top_10_pct_ratio:.2f}% 的访问")
    print(f"  前 10% 用户平均访问次数: {sorted_users.iloc[:top_10_pct_users]['total_visits'].mean():.2f}")
    print(f"  后 90% 用户平均访问次数: {sorted_users.iloc[top_10_pct_users:]['total_visits'].mean():.2f}")
    
    # 绘制可视化图表
    print("\n正在绘制热门用户占比图表...")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. CDF曲线 - 用户排名 vs 累积访问占比
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(sorted_users['user_rank_ratio'] * 100, sorted_users['cumulative_visits_ratio'] * 100, 
             linewidth=2, color='steelblue')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50%')
    ax1.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='80%')
    ax1.axvline(x=10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='10% Users')
    ax1.axvline(x=20, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='20% Users')
    ax1.set_xlabel('User Rank Percentile (%)', fontsize=12)
    ax1.set_ylabel('Cumulative Visit Ratio (%)', fontsize=12)
    ax1.set_title('User Visit Distribution CDF', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    
    # 2. 对数尺度CDF曲线
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(sorted_users['user_rank_ratio'] * 100, sorted_users['cumulative_visits_ratio'] * 100, 
             linewidth=2, color='steelblue')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=10, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('User Rank Percentile (%)', fontsize=12)
    ax2.set_ylabel('Cumulative Visit Ratio (%)', fontsize=12)
    ax2.set_title('User Visit Distribution CDF (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.1, 100])
    ax2.set_ylim([0, 100])
    
    # 3. 百分位用户访问占比柱状图
    ax3 = plt.subplot(2, 2, 3)
    pct_labels = [f'{p}%' for p in percentiles]
    pct_ratios = [percentile_stats[f'top_{p}_pct']['visits_ratio'] for p in percentiles]
    bars = ax3.bar(range(len(percentiles)), pct_ratios, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xticks(range(len(percentiles)))
    ax3.set_xticklabels(pct_labels, rotation=45, ha='right')
    ax3.set_xlabel('User Percentile', fontsize=12)
    ax3.set_ylabel('Visit Ratio (%)', fontsize=12)
    ax3.set_title('Visit Ratio by User Percentile', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for i, (bar, ratio) in enumerate(zip(bars, pct_ratios)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. 帕累托图 - 前N%用户占据的访问比例
    ax4 = plt.subplot(2, 2, 4)
    user_pct_range = np.arange(1, 101, 1)  # 1% 到 100%
    visits_ratio_range = []
    for pct in user_pct_range:
        num_users = int(len(sorted_users) * pct / 100)
        if num_users > 0:
            top_visits = sorted_users.iloc[:num_users]['total_visits'].sum()
            visits_ratio = top_visits / total_visits_all * 100
            visits_ratio_range.append(visits_ratio)
        else:
            visits_ratio_range.append(0)
    
    ax4.plot(user_pct_range, visits_ratio_range, linewidth=2, color='steelblue', label='Cumulative Visit Ratio')
    ax4.plot([0, 100], [0, 100], 'r--', linewidth=1, alpha=0.5, label='Uniform Distribution')
    ax4.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='80% Visits')
    ax4.axvline(x=20, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='20% Users')
    ax4.set_xlabel('User Percentage (%)', fontsize=12)
    ax4.set_ylabel('Cumulative Visit Ratio (%)', fontsize=12)
    ax4.set_title('Pareto Chart (80-20 Rule)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_xlim([0, 100])
    ax4.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_users_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {os.path.join(output_dir, 'top_users_distribution.png')}")
    
    # 保存详细数据到CSV
    csv_path = os.path.join(output_dir, 'user_visit_distribution.csv')
    sorted_users[['user_id', 'total_visits', 'cumulative_visits', 'cumulative_visits_ratio', 
                  'user_rank_ratio']].to_csv(csv_path, index=False)
    print(f"详细分布数据已保存到: {csv_path}")
    
    # 保存百分位统计到JSON
    json_path = os.path.join(output_dir, 'top_users_percentile_stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_users': int(len(sorted_users)),
            'total_visits': int(total_visits_all),
            'top_10_pct': {
                'num_users': int(top_10_pct_users),
                'visits': int(top_10_pct_visits),
                'visits_ratio': float(top_10_pct_ratio),
                'avg_visits': float(sorted_users.iloc[:top_10_pct_users]['total_visits'].mean())
            },
            'percentile_stats': percentile_stats
        }, f, indent=2, ensure_ascii=False)
    print(f"百分位统计已保存到: {json_path}")
    
    return percentile_stats


def main():
    parser = argparse.ArgumentParser(description='分析 Amazon Books 用户重复访问情况')
    parser.add_argument(
        '--data_path',
        type=str,
        default='~/data/amzn_books/sasrec_format.csv',
        help='数据文件路径（默认: ~/data/amzn_books/sasrec_format.csv）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./reports',
        help='输出目录（默认: ./reports）'
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
    
    # 解析序列数据
    expanded_df = parse_sequences(df)
    
    # 分析用户重复访问
    user_stats_df = analyze_user_repeat_visits(expanded_df)
    
    # 绘制统计图表
    plot_repeat_visit_statistics(user_stats_df, output_dir)
    
    # 保存统计结果
    save_statistics(user_stats_df, output_dir)
    
    # 分析热门用户占比
    analyze_top_users_distribution(user_stats_df, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()


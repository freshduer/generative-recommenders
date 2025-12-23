#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热门用户占比分析脚本
从已有的用户统计数据CSV文件中分析热门用户占比，统计前10%用户占据的访问比例
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置网格样式
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.axisbelow'] = True


def analyze_top_users_distribution(csv_path: str, output_dir: str):
    """分析热门用户占比，统计前10%用户占据的访问比例"""
    print("\n" + "="*60)
    print("热门用户占比分析")
    print("="*60)
    
    # 读取CSV文件
    print(f"正在读取数据: {csv_path}")
    user_stats_df = pd.read_csv(csv_path)
    print(f"数据加载完成，共 {len(user_stats_df)} 个用户")
    
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
    csv_output_path = os.path.join(output_dir, 'user_visit_distribution.csv')
    sorted_users[['user_id', 'total_visits', 'cumulative_visits', 'cumulative_visits_ratio', 
                  'user_rank_ratio']].to_csv(csv_output_path, index=False)
    print(f"详细分布数据已保存到: {csv_output_path}")
    
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
    parser = argparse.ArgumentParser(description='分析热门用户占比，统计前10%用户占据的访问比例')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='./reports/user_repeat_visit_statistics.csv',
        help='用户统计数据CSV文件路径（默认: ./reports/user_repeat_visit_statistics.csv）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./reports',
        help='输出目录（默认: ./reports）'
    )
    
    args = parser.parse_args()
    
    # 展开路径
    csv_path = os.path.expanduser(args.csv_path)
    output_dir = os.path.expanduser(args.output_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    # 分析热门用户占比
    analyze_top_users_distribution(csv_path, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()


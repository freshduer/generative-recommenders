#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析用户重复访问频率和热门用户分布
- 统计用户访问次数分布
- 计算重复访问用户比例
- 绘制访问频率分布图
- 分析10%热门用户占据的访问比例（帕累托分析）
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd  # type: ignore[import]

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
        
        return {
            'user_id': user_id,
            'asin': record.get('asin', '')
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


def load_user_visits(
    data_path: str,
    show_progress: bool = False,
    use_multiprocessing: bool = True,
    chunk_size: int = 100000,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Load user visits from Amazon Reviews dataset
    优先从cache加载（如果存在）
    
    Returns:
        DataFrame with columns: user_id, asin
    """
    logger.info(f"Loading user visits from: {data_path}")
    start_time = time.time()
    
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # 首先尝试从cache加载
    cache_dir = data_path_obj.parent / ".cache"
    cache_file = cache_dir / f"{data_path_obj.stem}_processed.parquet"
    
    if cache_file.exists():
        if not HAS_PYARROW:
            logger.warning("pyarrow未安装，无法读取parquet cache文件，将从原始文件加载")
        else:
            logger.info(f"找到cache文件: {cache_file}")
            logger.info("从cache加载数据...")
            try:
                # 从parquet文件读取，只需要user_id和asin列
                df = pd.read_parquet(cache_file, columns=['user_id', 'asin'])
                elapsed = time.time() - start_time
                logger.info(f"从cache加载完成: {len(df):,} 条记录，耗时 {elapsed:.2f} 秒")
                logger.info(f"唯一用户数: {df['user_id'].nunique():,}")
                return df
            except Exception as e:
                logger.warning(f"从cache加载失败: {e}，将从原始文件加载")
    else:
        logger.info(f"未找到cache文件: {cache_file}，将从原始文件加载")
    
    # 如果没有cache，从原始文件加载
    # Count total lines for progress bar
    total_lines = 0
    if HAS_TQDM and show_progress:
        logger.info("统计总行数...")
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"总行数: {total_lines:,}")
    
    all_records = []
    
    if use_multiprocessing and n_workers != 1:
        # Use multiprocessing
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
            
            line_count = 0
            for line in f:
                line_count += 1
                record = parse_amazon_review_line(line)
                if record:
                    all_records.append(record)
                
                if line_count % chunk_size == 0 and show_progress:
                    logger.info(f"Processed {line_count:,} lines, {len(all_records):,} valid records...")
    
    if not all_records:
        raise ValueError("No valid records found in the data file")
    
    # Convert to DataFrame
    logger.info("转换为DataFrame...")
    df = pd.DataFrame(all_records)
    
    # 只保留需要的列
    df = df[['user_id', 'asin']]
    
    elapsed = time.time() - start_time
    logger.info(f"数据加载完成: {len(df):,} 条访问记录")
    logger.info(f"唯一用户数: {df['user_id'].nunique():,}")
    logger.info(f"耗时: {elapsed:.2f} 秒")
    
    return df


def compute_user_visit_stats(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    计算每个用户的访问次数统计
    
    Returns:
        DataFrame with columns: user_id, visit_count
    """
    start_time = time.time()
    if show_progress:
        logger.info("Computing user visit statistics...")
    
    # Count visits per user
    user_visits = df.groupby('user_id', sort=False).size().reset_index(name='visit_count')
    user_visits = user_visits.sort_values('visit_count', ascending=False).reset_index(drop=True)
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"Computed visit statistics for {len(user_visits):,} users, elapsed {elapsed:.2f} seconds")
    
    return user_visits


def analyze_repeat_visits(user_visits: pd.DataFrame) -> Dict:
    """
    分析重复访问统计
    """
    total_users = len(user_visits)
    total_visits = user_visits['visit_count'].sum()
    
    # 只访问1次的用户（非重复访问用户）
    single_visit_users = (user_visits['visit_count'] == 1).sum()
    repeat_visit_users = total_users - single_visit_users
    
    # 重复访问用户的访问次数
    repeat_user_visits = user_visits[user_visits['visit_count'] > 1]['visit_count'].sum()
    
    stats = {
        'total_users': total_users,
        'total_visits': total_visits,
        'single_visit_users': single_visit_users,
        'single_visit_users_pct': single_visit_users / total_users * 100,
        'repeat_visit_users': repeat_visit_users,
        'repeat_visit_users_pct': repeat_visit_users / total_users * 100,
        'repeat_user_visits': repeat_user_visits,
        'repeat_user_visits_pct': repeat_user_visits / total_visits * 100,
        'avg_visits_per_user': total_visits / total_users,
        'avg_visits_per_repeat_user': repeat_user_visits / repeat_visit_users if repeat_visit_users > 0 else 0,
    }
    
    return stats


def compute_percentile_contributions(user_visits: pd.DataFrame) -> pd.DataFrame:
    """
    计算不同百分位用户的访问贡献（帕累托分析）
    """
    # 按访问次数降序排列
    sorted_visits = user_visits.sort_values('visit_count', ascending=False).reset_index(drop=True)
    sorted_visits['cumulative_visits'] = sorted_visits['visit_count'].cumsum()
    sorted_visits['cumulative_user_pct'] = (sorted_visits.index + 1) / len(sorted_visits) * 100
    sorted_visits['cumulative_visit_pct'] = sorted_visits['cumulative_visits'] / sorted_visits['visit_count'].sum() * 100
    
    # 计算关键百分位点
    total_users = len(sorted_visits)
    total_visits = sorted_visits['visit_count'].sum()
    
    percentile_stats = []
    
    for pct in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        n_users = int(total_users * pct / 100)
        if n_users > 0:
            user_subset = sorted_visits.head(n_users)
            visits = user_subset['visit_count'].sum()
            visit_pct = visits / total_visits * 100
            avg_visits = visits / n_users
            
            percentile_stats.append({
                'user_percentile': pct,
                'n_users': n_users,
                'total_visits': visits,
                'visit_percentage': visit_pct,
                'avg_visits_per_user': avg_visits
            })
    
    return pd.DataFrame(percentile_stats)


def plot_visit_frequency_distribution(
    user_visits: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    绘制访问频率分布图
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('User Visit Frequency Distribution Analysis', fontsize=16, fontweight='bold')
    
    visit_counts = user_visits['visit_count'].values
    
    # 1. Visit count histogram (linear scale)
    ax1 = axes[0, 0]
    # Limit display range to avoid long tail
    max_display = min(visit_counts.max(), np.percentile(visit_counts, 99))
    counts_to_plot = visit_counts[visit_counts <= max_display]
    ax1.hist(counts_to_plot, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(visit_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {visit_counts.mean():.2f}')
    ax1.axvline(np.median(visit_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(visit_counts):.0f}')
    ax1.set_xlabel('Visit Count', fontsize=11)
    ax1.set_ylabel('Number of Users', fontsize=11)
    ax1.set_title(f'Visit Count Distribution (up to P99: {max_display:.0f})', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Visit count histogram (log scale)
    ax2 = axes[0, 1]
    ax2.hist(visit_counts, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Visit Count (log scale)', fontsize=11)
    ax2.set_ylabel('Number of Users (log scale)', fontsize=11)
    ax2.set_title('Visit Count Distribution (Log-Log Scale)', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Cumulative Distribution Function (CDF)
    ax3 = axes[1, 0]
    sorted_counts = np.sort(visit_counts)
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax3.plot(sorted_counts, cumulative_pct, linewidth=2, color='purple', alpha=0.8)
    ax3.axvline(visit_counts.mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {visit_counts.mean():.2f}')
    ax3.axhline(50, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Users')
    ax3.set_xlabel('Visit Count', fontsize=11)
    ax3.set_ylabel('Cumulative User Percentage (%)', fontsize=11)
    ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Visit frequency statistics (showing user distribution across visit count ranges)
    ax4 = axes[1, 1]
    # Define visit count bins
    # n bin edges produce n-1 intervals, so labels should have n-1 items
    # Using right=False, intervals are left-closed right-open [a, b)
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['1', '2', '3', '4', '5-9', '10-19', '20-49', '50-99', '100-199', '200-499', '500-999', '1000+']
    visit_bins = pd.cut(visit_counts, bins=bins, labels=bin_labels, right=False)
    bin_counts = visit_bins.value_counts().sort_index()
    
    bars = ax4.bar(range(len(bin_counts)), bin_counts.values, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.set_xticks(range(len(bin_counts)))
    ax4.set_xticklabels(bin_counts.index, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Number of Users', fontsize=11)
    ax4.set_xlabel('Visit Count Range', fontsize=11)
    ax4.set_title('User Distribution by Visit Count Range', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, bin_counts.values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(visit_counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visit frequency distribution plot: {output_path}")


def plot_pareto_analysis(
    percentile_stats: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    绘制帕累托分析图（用户百分比 vs 访问百分比）
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed, cannot plot")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('User Visit Concentration Analysis (Pareto Analysis)', fontsize=16, fontweight='bold')
    
    user_pct = percentile_stats['user_percentile'].values
    visit_pct = percentile_stats['visit_percentage'].values
    
    # 1. Pareto curve (user percentage vs visit percentage)
    ax1 = axes[0]
    ax1.plot(user_pct, visit_pct, marker='o', markersize=6, linewidth=2.5, color='steelblue', alpha=0.8, label='Actual Curve')
    # Add diagonal line (if perfectly uniform distribution)
    ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.6, label='Uniform Distribution (Reference)')
    # Mark 10% users
    idx_10 = np.where(user_pct == 10)[0]
    if len(idx_10) > 0:
        ax1.plot(user_pct[idx_10], visit_pct[idx_10], 'ro', markersize=10, label=f'10% Users: {visit_pct[idx_10][0]:.1f}% Visits')
    ax1.set_xlabel('User Percentage (%)', fontsize=11)
    ax1.set_ylabel('Visit Percentage (%)', fontsize=11)
    ax1.set_title('Pareto Curve: User Share vs Visit Share', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    
    # Add annotations for key points
    for i, row in percentile_stats.iterrows():
        if row['user_percentile'] in [10, 20, 50, 90]:
            ax1.annotate(f"{row['user_percentile']:.0f}% Users\n{row['visit_percentage']:.1f}% Visits",
                        xy=(row['user_percentile'], row['visit_percentage']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=8)
    
    # 2. Bar chart: visit contribution by user percentile
    ax2 = axes[1]
    bars = ax2.bar(range(len(percentile_stats)), visit_pct, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xticks(range(len(percentile_stats)))
    ax2.set_xticklabels([f"{p:.0f}%" for p in user_pct], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Visit Percentage (%)', fontsize=11)
    ax2.set_xlabel('User Percentage', fontsize=11)
    ax2.set_title('Visit Contribution by User Percentile', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, pct) in enumerate(zip(bars, visit_pct)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 标记10%用户
    if len(idx_10) > 0:
        bars[idx_10[0]].set_color('red')
        bars[idx_10[0]].set_edgecolor('darkred')
        bars[idx_10[0]].set_linewidth(2)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved Pareto analysis plot: {output_path}")


def save_statistics(
    repeat_stats: Dict,
    percentile_stats: pd.DataFrame,
    user_visits: pd.DataFrame,
    output_dir: Path
) -> None:
    """保存统计结果到CSV文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存重复访问统计
    repeat_df = pd.DataFrame([repeat_stats])
    repeat_file = output_dir / "repeat_visit_stats.csv"
    repeat_df.to_csv(repeat_file, index=False)
    logger.info(f"Saved repeat visit statistics to: {repeat_file}")
    
    # 保存百分位统计
    percentile_file = output_dir / "percentile_contribution_stats.csv"
    percentile_stats.to_csv(percentile_file, index=False)
    logger.info(f"Saved percentile statistics to: {percentile_file}")
    
    # 保存用户访问统计（前10000名，避免文件过大）
    top_users_file = output_dir / "top_users_visit_stats.csv"
    top_users = user_visits.head(10000)
    top_users.to_csv(top_users_file, index=False)
    logger.info(f"Saved top 10000 users statistics to: {top_users_file}")


def print_summary_statistics(repeat_stats: Dict, percentile_stats: pd.DataFrame) -> None:
    """打印汇总统计信息"""
    logger.info("\n" + "="*60)
    logger.info("用户访问频率分析汇总")
    logger.info("="*60)
    
    logger.info("\n【重复访问统计】")
    logger.info(f"总用户数: {repeat_stats['total_users']:,}")
    logger.info(f"总访问次数: {repeat_stats['total_visits']:,}")
    logger.info(f"平均每用户访问次数: {repeat_stats['avg_visits_per_user']:.2f}")
    logger.info(f"\n只访问1次的用户: {repeat_stats['single_visit_users']:,} ({repeat_stats['single_visit_users_pct']:.2f}%)")
    logger.info(f"重复访问用户: {repeat_stats['repeat_visit_users']:,} ({repeat_stats['repeat_visit_users_pct']:.2f}%)")
    logger.info(f"重复访问用户平均访问次数: {repeat_stats['avg_visits_per_repeat_user']:.2f}")
    logger.info(f"重复访问用户总访问次数: {repeat_stats['repeat_user_visits']:,} ({repeat_stats['repeat_user_visits_pct']:.2f}%)")
    
    logger.info("\n【热门用户访问集中度（帕累托分析）】")
    for _, row in percentile_stats.iterrows():
        logger.info(f"{row['user_percentile']:3.0f}% 用户 ({row['n_users']:,}人) 占据了 {row['visit_percentage']:5.2f}% 的访问 "
                   f"(平均每人 {row['avg_visits_per_user']:.2f} 次)")
    
    # 重点显示10%用户
    row_10 = percentile_stats[percentile_stats['user_percentile'] == 10]
    if len(row_10) > 0:
        logger.info("\n" + "="*60)
        logger.info(f"【重点】10% 最活跃用户占据了 {row_10.iloc[0]['visit_percentage']:.2f}% 的访问！")
        logger.info("="*60)
    
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="分析用户重复访问频率和热门用户分布"
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
        help='输出目录（默认：脚本目录下的reports/）'
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help='显示进度条'
    )
    parser.add_argument(
        '--no_multiprocessing',
        action='store_true',
        help='禁用多进程处理'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='工作进程数（默认：自动）'
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_user_visits(
        args.data_path,
        show_progress=args.show_progress,
        use_multiprocessing=not args.no_multiprocessing,
        n_workers=args.n_workers
    )
    
    # 计算用户访问统计
    user_visits = compute_user_visit_stats(df, args.show_progress)
    
    # 分析重复访问
    repeat_stats = analyze_repeat_visits(user_visits)
    
    # 计算百分位贡献
    percentile_stats = compute_percentile_contributions(user_visits)
    
    # 打印汇总统计
    print_summary_statistics(repeat_stats, percentile_stats)
    
    # 绘制图表
    if HAS_MATPLOTLIB:
        # 访问频率分布图
        freq_plot_path = output_dir / "visit_frequency_distribution.png"
        plot_visit_frequency_distribution(user_visits, freq_plot_path)
        
        # 帕累托分析图
        pareto_plot_path = output_dir / "pareto_analysis.png"
        plot_pareto_analysis(percentile_stats, pareto_plot_path)
    else:
        logger.warning("matplotlib未安装，跳过图表生成")
    
    # 保存统计结果
    save_statistics(repeat_stats, percentile_stats, user_visits, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("分析完成！")
    logger.info(f"结果已保存到: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


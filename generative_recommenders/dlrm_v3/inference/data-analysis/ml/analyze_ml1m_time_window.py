#!/usr/bin/env python3
"""分析 MovieLens 1M 数据集中不同时间窗口下用户交互序列的长度分布和活跃用户数

支持的时间窗口：
- 30min: 30分钟窗口
- 1h: 1小时窗口
- 12h: 半天（12小时）窗口
- 1d: 一天（24小时）窗口

该脚本期望的数据格式：
- 包含 user_id, item_id (或 movie_id), timestamp 列的CSV文件
- timestamp 可以是Unix时间戳（秒）或datetime格式
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
except Exception:  # pragma: no cover - plotting optional
    plt = None
    mdates = None
    FuncFormatter = None
    sns = None

try:
    from tqdm import tqdm
    tqdm.pandas()
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _find_csvs(data_dir: Path) -> List[Path]:
    """查找 MovieLens 数据文件（CSV 或 DAT 格式）"""
    candidates = []
    # 首先查找 ratings.dat（MovieLens标准格式，优先）
    ratings_dat = data_dir / "ratings.dat"
    if ratings_dat.exists():
        candidates.append(ratings_dat)
    
    # 常见的 MovieLens CSV 文件格式
    for pattern in [
        "ratings.csv",
        "ml-1m.csv",
        "interactions.csv",
        "*.csv",
    ]:
        found = sorted(data_dir.glob(pattern))
        # 排除sequence格式的文件（这些文件需要特殊处理）
        for p in found:
            if "sasrec_format" not in p.name or "sequence" not in p.name.lower():
                candidates.append(p)
    
    # 去重并保持顺序
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _load_frames(paths: List[Path], show_progress: bool = False) -> pd.DataFrame:
    """加载并合并所有数据文件（CSV 或 DAT 格式）"""
    start_time = time.time()
    frames = []
    iterable = paths
    if show_progress and tqdm is not None:
        iterable = tqdm(paths, desc="读取数据文件", unit="文件", ncols=100)
    elif show_progress and tqdm is None:
        logger.info("tqdm未安装，继续执行但无进度条显示")
        show_progress = False
    
    for p in iterable:
        # 处理 .dat 文件（MovieLens标准格式，使用 :: 分隔符）
        if p.suffix == ".dat" and "ratings" in p.name.lower():
            try:
                df = pd.read_csv(
                    p,
                    sep="::",
                    engine="python",
                    header=None,
                    names=["user_id", "item_id", "rating", "timestamp"],
                    dtype={"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"}
                )
                # .dat 文件已经正确读取，直接使用，不需要后续查找列名的步骤
                df = df[["user_id", "item_id", "timestamp"]].copy()
            except Exception as e:
                logger.warning(f"读取 {p.name} 失败: {e}，跳过")
                continue
        else:
            # 处理 CSV 文件
            try:
                df = pd.read_csv(p, engine="pyarrow", low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(p, engine="c", low_memory=False)
                except Exception:
                    df = pd.read_csv(p, low_memory=False)
            
            # 标准化列名（创建小写键到原始列的映射）
            lower_cols = {c.lower(): c for c in df.columns}
            
            # 查找用户ID列（支持多种命名格式）
            user_col = None
            for key in ["user", "user_id", "userid", "uid"]:
                if key in lower_cols:
                    user_col = lower_cols[key]
                    break
            
            # 查找物品ID列（支持多种命名格式）
            item_col = None
            for key in ["item", "item_id", "itemid", "movie", "movie_id", "movieid", "iid"]:
                if key in lower_cols:
                    item_col = lower_cols[key]
                    break
            
            # 查找时间戳列
            time_col = None
            for key in ["timestamp", "time", "time_ms", "ts", "datetime", "date"]:
                if key in lower_cols:
                    time_col = lower_cols[key]
                    break
            
            if user_col is None or item_col is None:
                logger.warning(f"文件 {p.name} 缺少必需的列（user_id, item_id），跳过")
                continue
            
            if time_col is None:
                logger.warning(f"文件 {p.name} 缺少时间戳列，将无法进行时间窗口分析，跳过")
                continue
            
            # 只保留需要的列
            df = df[[user_col, item_col, time_col]].rename(columns={
                user_col: "user_id",
                item_col: "item_id",
                time_col: "timestamp"
            })
        
        # 转换为整数类型（如果可能）
        for col in ["user_id", "item_id"]:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            except Exception:
                pass
        
        df["__source_file"] = p.name
        frames.append(df)
        if show_progress and tqdm is None:
            logger.debug(f"已读取: {p.name} ({len(df):,} 行)")
    
    if not frames:
        raise FileNotFoundError("未找到可用的数据文件（需要包含 user_id, item_id, timestamp 列）")
    
    logger.info("正在合并数据框...")
    concatenated = pd.concat(frames, ignore_index=True, sort=False)
    elapsed = time.time() - start_time
    logger.info(f"数据加载完成: {len(concatenated):,} 行，耗时 {elapsed:.2f} 秒")
    return concatenated


def _convert_timestamp_to_datetime(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """将时间戳转换为 datetime（优化版）"""
    if "timestamp" not in df.columns:
        raise ValueError("数据中缺少'timestamp'列")
    
    start_time = time.time()
    if show_progress:
        logger.info("正在转换时间戳...")
    
    # 尝试不同的时间戳格式
    try:
        # 尝试作为 Unix 时间戳（秒）
        if df["timestamp"].dtype in ['int64', 'int32', 'float64', 'float32']:
            # 检查是否是毫秒时间戳（大于 1e12）还是秒时间戳
            sample = df["timestamp"].dropna().head(1000)
            if len(sample) > 0 and sample.max() > 1e12:
                # 毫秒时间戳
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors='coerce')
            else:
                # 秒时间戳
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors='coerce')
        else:
            # 尝试直接解析为 datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], errors='coerce')
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}，尝试直接解析")
        df["datetime"] = pd.to_datetime(df["timestamp"], errors='coerce')
    
    # 检查转换是否成功
    null_count = df["datetime"].isna().sum()
    if null_count > 0:
        logger.warning(f"有 {null_count:,} 条记录的时间戳转换失败")
    
    # 删除时间戳转换失败的行
    df = df.dropna(subset=["datetime"])
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"时间戳转换完成，耗时 {elapsed:.2f} 秒")
    
    return df


def _compute_user_sequence_lengths_by_window(
    df: pd.DataFrame,
    window_type: str,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算不同时间窗口下每个用户的累计序列长度（到该时间窗口为止的总交互次数）
    
    Args:
        df: 包含datetime列的数据框
        window_type: 时间窗口类型 ('30min', '1h', '12h', '1d')
        show_progress: 是否显示进度
    
    Returns:
        包含时间窗口、用户ID和累计序列长度的数据框
    """
    start_time = time.time()
    if show_progress:
        logger.info(f"开始按{window_type}分组计算用户累计序列长度...")
    
    # 根据窗口类型创建时间窗口标签
    if window_type == "30min":
        time_window = df["datetime"].dt.floor("30min")
    elif window_type == "1h":
        time_window = df["datetime"].dt.floor("H")
    elif window_type == "12h":
        time_window = df["datetime"].dt.floor("12H")
    elif window_type == "1d":
        time_window = df["datetime"].dt.floor("D")
    else:
        raise ValueError(f"不支持的时间窗口类型: {window_type}")
    
    # 创建临时数据框用于分组（只包含需要的列）
    group_df = pd.DataFrame({
        'time_window': time_window.values,
        'user_id': df['user_id'].values
    })
    
    # 计算每个时间窗口内每个用户的交互次数（当前窗口的交互数）
    window_interactions = group_df.groupby(["time_window", "user_id"], observed=True, sort=False).size()
    window_interactions = window_interactions.reset_index(name="window_count")
    
    # 按时间窗口和用户ID排序
    window_interactions = window_interactions.sort_values(['time_window', 'user_id'])
    
    if show_progress:
        logger.info("正在计算累计序列长度...")
    
    # 对每个用户，按时间窗口排序后计算累计和（从最早的时间窗口累计到当前）
    window_interactions['sequence_length'] = window_interactions.groupby('user_id', observed=True, sort=False)['window_count'].cumsum()
    
    result = window_interactions[['time_window', 'user_id', 'sequence_length']]
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(f"累计序列长度计算完成: {len(result):,} 条记录，耗时 {elapsed:.2f} 秒")
    
    return result


def _compute_window_statistics(
    df: pd.DataFrame,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    show_progress: bool = False
) -> pd.DataFrame:
    """
    计算每个时间窗口的统计指标（优化版：使用向量化操作）
    
    Returns:
        包含每个时间窗口统计指标的数据框，包括活跃用户数
    """
    start_time = time.time()
    if show_progress:
        logger.info("开始计算窗口统计指标...")
        unique_windows = df["time_window"].nunique()
        logger.info(f"共有 {unique_windows} 个时间窗口需要计算")
    
    # 使用groupby一次性计算所有统计指标
    grouped = df.groupby("time_window", observed=True, sort=True)["sequence_length"]
    
    # 构建聚合函数字典（包含所有需要的统计量）
    all_percentiles = sorted(set(percentiles + [0.25, 0.5, 0.75]))
    
    # 对于SeriesGroupBy，分别计算各个统计量然后合并
    count_df = grouped.count().reset_index(name='count')  # 活跃用户数
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
    
    # 计算方差（std的平方）
    stats['variance'] = stats['std'] ** 2
    
    # 填充NaN值（当只有一个元素时std为NaN）
    stats['std'] = stats['std'].fillna(0.0).astype(float)
    stats['variance'] = stats['variance'].fillna(0.0).astype(float)
    
    # 一次性计算所有百分位数
    if show_progress:
        logger.info("计算百分位数...")
    
    # 设置p50为median（如果percentiles包含0.5）
    if 0.5 in all_percentiles:
        stats['p50'] = stats['median']
    
    # 计算所有需要的百分位数（排除0.5，因为已经用median了）
    percentiles_to_compute = sorted(set([p for p in all_percentiles if p != 0.5]))
    
    # 批量计算所有百分位数
    if percentiles_to_compute:
        for p in percentiles_to_compute:
            col_name = f"p{int(p*100)}"
            if col_name not in stats.columns:
                quantile_series = grouped.quantile(p)
                stats[col_name] = stats['time_window'].map(quantile_series).astype(float)
    
    # 确保p25和p75存在（用于IQR计算），如果不存在则计算
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


def _plot_sequence_length_over_time(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    dataset_name: str = "MovieLens",
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """绘制序列长度随时间窗口的变化图（波动图）"""
    if plt is None:
        raise ImportError("matplotlib未安装，无法绘图")
    
    # 确保time_window是datetime类型以便绘图
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    
    # 格式化数据集名称用于显示
    display_name = dataset_name.upper().replace("-", "-").replace("_", "-")
    if display_name.startswith("ML"):
        display_name = f"MovieLens-{display_name[2:]}" if len(display_name) > 2 else "MovieLens"
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{display_name}: User Interaction Sequence Length Distribution Statistics by {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    x = window_stats["time_window"]
    
    # 1. 均值、中位数、p90、p99 时间序列图
    ax1 = axes[0, 0]
    ax1.plot(x, window_stats["mean"], label="Mean", marker='o', markersize=4, linewidth=2, alpha=0.8)
    ax1.plot(x, window_stats["median"], label="Median", marker='s', markersize=4, linewidth=2, alpha=0.8)
    ax1.plot(x, window_stats["p90"], label="P90", marker='^', markersize=4, linewidth=2, alpha=0.8, linestyle='--')
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


def _plot_detailed_statistics(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    dataset_name: str = "MovieLens",
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """绘制更详细的统计信息图，包括活跃用户数"""
    if plt is None:
        raise ImportError("matplotlib未安装，无法绘图")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    x = window_stats["time_window"]
    
    # 格式化数据集名称用于显示
    display_name = dataset_name.upper().replace("-", "-").replace("_", "-")
    if display_name.startswith("ML"):
        display_name = f"MovieLens-{display_name[2:]}" if len(display_name) > 2 else "MovieLens"
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'{display_name}: Detailed Statistical Analysis of User Interaction Sequence Length by {window_type} Time Window', 
                 fontsize=16, fontweight='bold')
    
    # 1. 所有百分位数对比
    ax1 = axes[0, 0]
    percentiles = ["p50", "p75", "p90", "p95", "p99"]
    try:
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(percentiles)))
    except (AttributeError, ImportError):
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
    
    # 6. 箱线图风格的统计摘要（使用IQR和极值）
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


def _plot_active_user_count(
    window_stats: pd.DataFrame,
    output_path: Path,
    window_type: str,
    dataset_name: str = "MovieLens",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """专门绘制活跃用户数随时间窗口的变化图（优化显示）"""
    if plt is None:
        raise ImportError("matplotlib未安装，无法绘图")
    
    window_stats = window_stats.copy()
    if not pd.api.types.is_datetime64_any_dtype(window_stats["time_window"]):
        window_stats["time_window"] = pd.to_datetime(window_stats["time_window"])
    
    window_stats = window_stats.sort_values("time_window")
    x = window_stats["time_window"]
    
    # 格式化数据集名称用于显示
    display_name = dataset_name.upper().replace("-", "-").replace("_", "-")
    if display_name.startswith("ML"):
        display_name = f"MovieLens-{display_name[2:]}" if len(display_name) > 2 else "MovieLens"
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    count_values = window_stats["count"]
    ax.plot(x, count_values, label="Active User Count", marker='o', markersize=4, linewidth=2, alpha=0.8, color='green')
    
    # 不截断y轴，显示完整数据范围
    min_val = count_values.min()
    max_val = count_values.max()
    mean_val = count_values.mean()
    
    ax.set_xlabel("Time Window", fontsize=12)
    ax.set_ylabel("User Count", fontsize=12)
    title_text = f'{display_name}: Active User Count per Time Window ({window_type})\n(Full range: {min_val:.0f} to {max_val:.0f}, Mean: {mean_val:.0f})'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    if FuncFormatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    if mdates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_sequence_length_by_time_window(
    data_dir: Path,
    window_type: str = "1d",
    show_progress: bool = False,
    percentiles: List[float] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分析不同时间窗口下用户交互序列长度分布
    
    Returns:
        (用户序列长度数据框, 窗口统计数据框)
    """
    total_start = time.time()
    logger.info("="*80)
    # 根据数据目录名称推断数据集显示名称
    dataset_display_name = data_dir.name.upper().replace("-", "-").replace("_", "-")
    logger.info(f"开始分析 {dataset_display_name} 用户交互序列长度分布")
    logger.info(f"时间窗口类型: {window_type}")
    logger.info("="*80)
    
    # 加载数据
    data_files = _find_csvs(data_dir)
    if not data_files:
        raise FileNotFoundError("未找到数据文件（需要包含 user_id, item_id, timestamp 列）")
    df = _load_frames(data_files, show_progress)
    logger.info(f"从{len(data_files)}个文件加载了{len(df):,}条交互记录")
    
    # 检查必需列
    required_cols = {"user_id", "item_id", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需的列: {sorted(missing)}")
    
    # 转换时间
    df = _convert_timestamp_to_datetime(df, show_progress)
    logger.info(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    # 计算每个时间窗口内每个用户的序列长度
    user_seq_lengths = _compute_user_sequence_lengths_by_window(
        df, window_type, show_progress
    )
    
    # 计算每个时间窗口的统计指标
    window_stats = _compute_window_statistics(user_seq_lengths, percentiles, show_progress)
    
    total_elapsed = time.time() - total_start
    logger.info("="*80)
    logger.info(f"分析完成！总耗时: {total_elapsed:.2f} 秒")
    logger.info("="*80)
    
    return user_seq_lengths, window_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="包含MovieLens CSV文件的目录",
    )
    parser.add_argument(
        "--window-type",
        type=str,
        choices=["30min", "1h", "12h", "1d"],
        default="1d",
        help="时间窗口类型: 30min/1h/12h/1d (默认: 1d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认: 脚本所在目录的painting文件夹）",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="可选：保存窗口统计数据的CSV文件路径",
    )
    parser.add_argument(
        "--user-seq-lengths-csv",
        type=Path,
        default=None,
        help="可选：保存用户序列长度数据的CSV文件路径",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="可选：保存窗口统计数据的JSON文件路径",
    )
    parser.add_argument(
        "--plot-main",
        type=Path,
        default=None,
        help="可选：主要统计图PNG文件路径",
    )
    parser.add_argument(
        "--plot-detailed",
        type=Path,
        default=None,
        help="可选：详细统计图PNG文件路径",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示进度信息",
    )
    
    args = parser.parse_args()
    
    # 确定数据目录
    data_dir = args.data_dir.expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据目录 '{data_dir}' 不存在。"
            " 请使用 --data-dir 指定包含MovieLens CSV文件的文件夹。"
        )
    
    # 根据数据目录名称推断数据集名称（用于文件命名）
    dataset_name = data_dir.name.lower().replace("-", "").replace("_", "")
    # 如果无法从目录名推断，使用默认值
    if "ml" not in dataset_name:
        dataset_name = "ml"
    logger.info(f"数据集名称: {dataset_name}")
    
    # 确定输出目录（默认使用脚本所在目录的 painting 文件夹）
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "painting"
    else:
        output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行分析
    user_seq_lengths, window_stats = analyze_sequence_length_by_time_window(
        data_dir,
        window_type=args.window_type,
        show_progress=args.show_progress,
    )
    
    # 保存CSV
    logger.info("开始保存结果...")
    stats_csv = args.stats_csv or output_dir / f"{dataset_name}_sequence_length_stats_{args.window_type}.csv"
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.time()
    window_stats.to_csv(stats_csv, index=False)
    logger.info(f"窗口统计数据已保存至: {stats_csv} (耗时 {time.time() - save_start:.2f} 秒)")
    
    if args.user_seq_lengths_csv:
        args.user_seq_lengths_csv.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.time()
        user_seq_lengths.to_csv(args.user_seq_lengths_csv, index=False)
        logger.info(f"用户序列长度数据已保存至: {args.user_seq_lengths_csv} (耗时 {time.time() - save_start:.2f} 秒)")
    
    # 保存JSON
    if args.stats_json:
        stats_json_path = args.stats_json
    else:
        stats_json_path = output_dir / f"{dataset_name}_sequence_length_stats_{args.window_type}.json"
    
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)
    # 转换时间窗口为字符串以便JSON序列化
    stats_dict = window_stats.copy()
    stats_dict["time_window"] = stats_dict["time_window"].astype(str)
    save_start = time.time()
    stats_json_path.write_text(
        json.dumps(stats_dict.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str)
    )
    logger.info(f"窗口统计数据JSON已保存至: {stats_json_path} (耗时 {time.time() - save_start:.2f} 秒)")
    
    # 打印统计摘要
    logger.info("\n" + "="*80)
    logger.info(f"时间窗口类型: {args.window_type}")
    logger.info(f"时间窗口数量: {len(window_stats)}")
    logger.info("="*80)
    logger.info("\n总体统计:")
    logger.info(f"  序列长度均值: {window_stats['mean'].mean():.2f}")
    logger.info(f"  序列长度标准差: {window_stats['std'].mean():.2f}")
    logger.info(f"  序列长度中位数: {window_stats['median'].mean():.2f}")
    logger.info(f"  序列长度P90: {window_stats['p90'].mean():.2f}")
    logger.info(f"  序列长度P99: {window_stats['p99'].mean():.2f}")
    logger.info(f"  平均方差: {window_stats['variance'].mean():.2f}")
    logger.info(f"  平均活跃用户数: {window_stats['count'].mean():.2f}")
    logger.info("\n各时间窗口统计摘要:")
    # 只在窗口数量不太多时打印详细摘要
    if len(window_stats) <= 50:
        logger.info("\n" + window_stats[["time_window", "count", "mean", "std", "variance", "median", "p90", "p99"]].to_string(index=False))
    else:
        logger.info(f"(共{len(window_stats)}个窗口，前10个窗口的统计摘要)")
        logger.info("\n" + window_stats.head(10)[["time_window", "count", "mean", "std", "variance", "median", "p90", "p99"]].to_string(index=False))
    
    # 绘制图表
    if plt is not None:
        logger.info("\n开始生成图表...")
        plot_main = args.plot_main or output_dir / f"{dataset_name}_sequence_length_over_time_{args.window_type}.png"
        plot_start = time.time()
        _plot_sequence_length_over_time(window_stats, plot_main, args.window_type, dataset_name)
        logger.info(f"主要统计图已保存至: {plot_main} (耗时 {time.time() - plot_start:.2f} 秒)")
        
        plot_detailed = args.plot_detailed or output_dir / f"{dataset_name}_sequence_length_detailed_stats_{args.window_type}.png"
        plot_start = time.time()
        _plot_detailed_statistics(window_stats, plot_detailed, args.window_type, dataset_name)
        logger.info(f"详细统计图已保存至: {plot_detailed} (耗时 {time.time() - plot_start:.2f} 秒)")
        
        # 生成单独的活跃用户数图表
        plot_active_users = output_dir / f"{dataset_name}_active_user_count_{args.window_type}.png"
        plot_start = time.time()
        _plot_active_user_count(window_stats, plot_active_users, args.window_type, dataset_name)
        logger.info(f"活跃用户数图表已保存至: {plot_active_users} (耗时 {time.time() - plot_start:.2f} 秒)")
    else:
        logger.warning("matplotlib未安装，跳过绘图")


if __name__ == "__main__":
    main()


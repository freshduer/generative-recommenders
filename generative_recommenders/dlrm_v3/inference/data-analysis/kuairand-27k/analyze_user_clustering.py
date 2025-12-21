#!/usr/bin/env python3
"""分析用户之间的item重合度和聚类效应，用于GPU调度优化。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter
except Exception:
    plt = None
    sns = None
    FuncFormatter = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _load_log_frames(
    data_dir: Path, show_progress: bool
) -> Tuple[pd.DataFrame, List[Path]]:
    """加载并合并所有KuaiRand日志CSV文件（优化版本）。"""
    log_paths = sorted(data_dir.glob("log_standard*.csv"))
    if not log_paths:
        msg = (
            "未找到KuaiRand日志文件。期望文件格式："
            "'log_standard_4_08_to_4_21_1k.csv'。"
        )
        raise FileNotFoundError(msg)

    # 只读取需要的列（加速读取）
    usecols = ["user_id", "video_id"]  # 只读取聚类分析需要的列
    
    iterable = log_paths
    if show_progress and tqdm is not None:
        iterable = tqdm(log_paths, desc="读取日志", unit="文件")
    elif show_progress and tqdm is None:
        print("tqdm未安装；继续执行但不显示进度条。")
        show_progress = False

    frames = []
    for csv_path in iterable:
        # 优化读取参数
        try:
            # 使用优化的读取参数
            frame = pd.read_csv(
                csv_path,
                usecols=usecols,
                dtype={
                    "user_id": "int64",
                    "video_id": "int64",
                },
                engine="c",  # C引擎通常最快
                low_memory=False,  # 避免类型推断的开销
                na_filter=False,  # 跳过NA值检查（如果确定没有NA）
            )
        except KeyError:
            # 如果某些文件没有这些列，尝试读取所有列
            frame = pd.read_csv(
                csv_path,
                dtype={
                    "user_id": "int64",
                    "video_id": "int64",
                },
                engine="c",
                low_memory=False,
            )
            # 只保留需要的列
            if "user_id" in frame.columns and "video_id" in frame.columns:
                frame = frame[["user_id", "video_id"]]
            else:
                raise ValueError(f"文件 {csv_path} 缺少必需的列: user_id, video_id")
        except Exception as e:
            # 如果优化读取失败，回退到标准读取
            if show_progress:
                print(f"警告: 文件 {csv_path.name} 使用标准读取方式: {e}")
            frame = pd.read_csv(csv_path)
            if "user_id" in frame.columns and "video_id" in frame.columns:
                frame = frame[["user_id", "video_id"]]
            else:
                raise ValueError(f"文件 {csv_path} 缺少必需的列: user_id, video_id")
        
        frame["__source_file"] = csv_path.name
        frames.append(frame)

    if show_progress:
        print(f"合并 {len(frames)} 个数据框...")
    concatenated = pd.concat(frames, ignore_index=True)
    return concatenated, log_paths


def compute_user_item_matrix(
    df: pd.DataFrame,
    sample_users: Optional[int] = None,
    min_interactions: int = 5,
    show_progress: bool = False,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    构建用户-物品矩阵（稀疏矩阵）。
    
    Returns:
        user_item_matrix: 稀疏矩阵，shape=(n_users, n_items)
        user_ids: 用户ID数组
        item_ids: 物品ID数组
    """
    required_cols = {"user_id", "video_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需的列: {sorted(missing)}")

    # 过滤交互次数太少的用户
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df_filtered = df[df["user_id"].isin(valid_users)].copy()

    if show_progress:
        print(f"过滤后用户数: {len(valid_users):,} (原始: {df['user_id'].nunique():,})")

    # 采样用户（如果指定）
    if sample_users is not None and sample_users < len(valid_users):
        sampled_user_ids = np.random.choice(
            valid_users, size=sample_users, replace=False
        )
        df_filtered = df_filtered[df_filtered["user_id"].isin(sampled_user_ids)]
        valid_users = sampled_user_ids
        if show_progress:
            print(f"采样用户数: {len(valid_users):,}")

    # 获取每个用户访问的唯一物品集合
    user_items = df_filtered.groupby("user_id")["video_id"].apply(set).to_dict()

    # 构建用户和物品的映射
    unique_items = set()
    for items in user_items.values():
        unique_items.update(items)
    unique_items = sorted(unique_items)
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    user_to_idx = {user: idx for idx, user in enumerate(valid_users)}

    # 构建稀疏矩阵
    rows = []
    cols = []
    data = []

    if show_progress:
        print("构建用户-物品矩阵...")
        iter_users = tqdm(user_items.items()) if tqdm else user_items.items()
    else:
        iter_users = user_items.items()

    for user_id, items in iter_users:
        user_idx = user_to_idx[user_id]
        for item_id in items:
            item_idx = item_to_idx[item_id]
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1.0)

    user_item_matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(valid_users), len(unique_items))
    )

    user_ids = np.array(valid_users)
    item_ids = np.array(unique_items)

    return user_item_matrix, user_ids, item_ids


def compute_jaccard_similarity(
    user_item_matrix: csr_matrix, show_progress: bool = False
) -> np.ndarray:
    """
    计算用户之间的Jaccard相似度矩阵。
    
    Jaccard相似度 = |A ∩ B| / |A ∪ B|
    """
    n_users = user_item_matrix.shape[0]

    if show_progress:
        print(f"计算 {n_users} 个用户之间的Jaccard相似度...")

    # 计算交集：矩阵乘法得到交集大小
    intersection = user_item_matrix.dot(user_item_matrix.T).toarray()

    # 计算并集：|A ∪ B| = |A| + |B| - |A ∩ B|
    user_sizes = np.array(user_item_matrix.sum(axis=1)).flatten()
    union = user_sizes[:, None] + user_sizes[None, :] - intersection

    # 避免除以零
    union = np.maximum(union, 1e-10)
    jaccard = intersection / union

    # 将对角线设为1（自己与自己的相似度）
    np.fill_diagonal(jaccard, 1.0)

    return jaccard


def compute_cosine_similarity(
    user_item_matrix: csr_matrix, show_progress: bool = False
) -> np.ndarray:
    """计算用户之间的余弦相似度矩阵。"""
    n_users = user_item_matrix.shape[0]

    if show_progress:
        print(f"计算 {n_users} 个用户之间的余弦相似度...")

    # L2归一化
    norms = np.sqrt(np.array(user_item_matrix.power(2).sum(axis=1))).flatten()
    norms = np.maximum(norms, 1e-10)
    normalized_matrix = user_item_matrix.multiply(1.0 / norms[:, None])

    # 计算余弦相似度
    cosine = normalized_matrix.dot(normalized_matrix.T).toarray()

    return cosine


def perform_kmeans_clustering(
    user_item_matrix: csr_matrix,
    similarity_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    compute_silhouette: bool = True,
    silhouette_sample_size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    使用K-means对用户进行聚类。
    
    Args:
        compute_silhouette: 是否计算轮廓系数（大规模数据可能很慢）
        silhouette_sample_size: 计算轮廓系数时的采样大小（None表示不采样）
    
    Returns:
        labels: 聚类标签
        silhouette: 轮廓系数（如果compute_silhouette=False则为None）
    """
    n_users = user_item_matrix.shape[0]
    
    # 使用PCA降维以提高效率（对于大量用户）
    if n_users > 1000:
        # 对于大规模数据，使用PCA降维
        n_components = min(50, n_users - 1)
        if n_users > 10000:
            # 对于超大规模数据，进一步减少组件数
            n_components = min(30, n_users - 1)
        
        # 使用稀疏矩阵的SVD进行PCA（更高效）
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        features = svd.fit_transform(user_item_matrix)
    else:
        features = similarity_matrix

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    # 计算轮廓系数（可选，大规模数据可能很慢）
    silhouette = None
    if compute_silhouette:
        # 对于大规模数据，使用采样方法计算轮廓系数
        if silhouette_sample_size is not None and silhouette_sample_size < n_users:
            # 采样用户计算轮廓系数
            sample_indices = np.random.choice(
                n_users, size=silhouette_sample_size, replace=False
            )
            sample_labels = labels[sample_indices]
            sample_distance = 1 - similarity_matrix[np.ix_(sample_indices, sample_indices)]
            np.fill_diagonal(sample_distance, 0)
            silhouette = silhouette_score(
                sample_distance, sample_labels, metric="precomputed"
            )
        else:
            # 计算完整轮廓系数
            distance_matrix = 1 - similarity_matrix
            np.fill_diagonal(distance_matrix, 0)
            silhouette = silhouette_score(distance_matrix, labels, metric="precomputed")

    return labels, silhouette


def perform_hierarchical_clustering(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    method: str = "complete",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用层次聚类对用户进行聚类。
    
    Args:
        similarity_matrix: 相似度矩阵
        n_clusters: 聚类数量
        method: 链接方法 ('complete', 'average', 'ward')
        注意: 'ward' 方法需要欧氏距离，会自动使用平方欧氏距离
    """
    # 将相似度转换为距离
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    # ward方法需要欧氏距离，使用平方欧氏距离
    if method == "ward":
        # 将距离矩阵转换为平方欧氏距离形式
        # 使用 condensed distance matrix
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=method)
    else:
        # 对于其他方法，使用condensed distance matrix
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=method)

    # 获取聚类标签
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1

    return labels, linkage_matrix


def analyze_cluster_statistics(
    user_ids: np.ndarray,
    user_item_matrix: csr_matrix,
    labels: np.ndarray,
    similarity_matrix: np.ndarray,
) -> Dict:
    """分析聚类统计信息。"""
    n_clusters = len(np.unique(labels))
    cluster_stats = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_users = user_ids[mask]
        cluster_size = len(cluster_users)

        # 计算簇内平均相似度
        cluster_sim = similarity_matrix[mask][:, mask]
        intra_cluster_sim = cluster_sim[np.triu_indices(cluster_size, k=1)].mean()

        # 计算簇间平均相似度
        inter_cluster_sims = []
        for other_cluster_id in range(n_clusters):
            if other_cluster_id != cluster_id:
                other_mask = labels == other_cluster_id
                inter_sim = similarity_matrix[mask][:, other_mask].mean()
                inter_cluster_sims.append(inter_sim)
        inter_cluster_sim = np.mean(inter_cluster_sims) if inter_cluster_sims else 0.0

        # 计算簇内用户访问的物品集合
        cluster_items = user_item_matrix[mask].sum(axis=0).A1 > 0
        cluster_unique_items = cluster_items.sum()

        cluster_stats.append({
            "cluster_id": int(cluster_id),
            "size": int(cluster_size),
            "intra_cluster_similarity": float(intra_cluster_sim),
            "inter_cluster_similarity": float(inter_cluster_sim),
            "unique_items": int(cluster_unique_items),
            "user_ids": cluster_users.tolist(),
        })

    # 计算整体统计
    overall_intra_sim = np.mean([
        stat["intra_cluster_similarity"] for stat in cluster_stats
    ])
    overall_inter_sim = np.mean([
        stat["inter_cluster_similarity"] for stat in cluster_stats
    ])

    return {
        "n_clusters": n_clusters,
        "total_users": len(user_ids),
        "overall_intra_cluster_similarity": float(overall_intra_sim),
        "overall_inter_cluster_similarity": float(overall_inter_sim),
        "clustering_quality": float(overall_intra_sim - overall_inter_sim),
        "cluster_statistics": cluster_stats,
    }


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    max_users: int = 500,
) -> None:
    """绘制相似度矩阵热力图。"""
    if plt is None or sns is None:
        print("matplotlib/seaborn未安装，跳过绘图。")
        return

    n_users = similarity_matrix.shape[0]

    # 如果用户太多，采样显示
    if n_users > max_users:
        indices = np.random.choice(n_users, size=max_users, replace=False)
        sim_subset = similarity_matrix[np.ix_(indices, indices)]
        labels_subset = labels[indices] if labels is not None else None
    else:
        sim_subset = similarity_matrix
        labels_subset = labels
        indices = np.arange(n_users)

    # 如果有标签，按聚类排序
    if labels_subset is not None:
        sort_order = np.argsort(labels_subset)
        sim_subset = sim_subset[np.ix_(sort_order, sort_order)]
        labels_subset = labels_subset[sort_order]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sim_subset,
        cmap="YlOrRd",
        square=True,
        cbar_kws={"label": "相似度"},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"用户相似度矩阵热力图 (采样 {len(sim_subset)} 个用户)", fontsize=14)
    ax.set_xlabel("用户索引")
    ax.set_ylabel("用户索引")

    if labels_subset is not None:
        # 添加聚类边界
        unique_labels = np.unique(labels_subset)
        for label in unique_labels:
            mask = labels_subset == label
            boundary = np.where(mask)[0]
            if len(boundary) > 0:
                start = boundary[0]
                end = boundary[-1] + 1
                ax.axhline(start, color="blue", linewidth=2, alpha=0.7)
                ax.axhline(end, color="blue", linewidth=2, alpha=0.7)
                ax.axvline(start, color="blue", linewidth=2, alpha=0.7)
                ax.axvline(end, color="blue", linewidth=2, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_distribution(
    labels: np.ndarray,
    similarity_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """绘制聚类分布和相似度分布。"""
    if plt is None:
        print("matplotlib未安装，跳过绘图。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 聚类大小分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[0].bar(unique_labels, counts, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("聚类ID")
    axes[0].set_ylabel("用户数量")
    axes[0].set_title("各聚类的用户数量分布")
    axes[0].grid(True, alpha=0.3)

    # 簇内和簇间相似度分布
    n_clusters = len(unique_labels)
    intra_sims = []
    inter_sims = []

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_sim = similarity_matrix[mask][:, mask]
        intra_sims.extend(
            cluster_sim[np.triu_indices(np.sum(mask), k=1)].tolist()
        )

        for other_cluster_id in unique_labels:
            if other_cluster_id != cluster_id:
                other_mask = labels == other_cluster_id
                inter_sim = similarity_matrix[mask][:, other_mask]
                inter_sims.extend(inter_sim.flatten().tolist())

    axes[1].hist(
        intra_sims,
        bins=50,
        alpha=0.6,
        label=f"簇内相似度 (均值={np.mean(intra_sims):.3f})",
        color="green",
    )
    axes[1].hist(
        inter_sims,
        bins=50,
        alpha=0.6,
        label=f"簇间相似度 (均值={np.mean(inter_sims):.3f})",
        color="red",
    )
    axes[1].set_xlabel("相似度")
    axes[1].set_ylabel("频数")
    axes[1].set_title("簇内 vs 簇间相似度分布")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    max_display: int = 100,
) -> None:
    """绘制层次聚类树状图。"""
    if plt is None:
        print("matplotlib未安装，跳过绘图。")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=max_display,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title("用户层次聚类树状图", fontsize=14)
    ax.set_xlabel("用户索引")
    ax.set_ylabel("距离")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析用户聚类效应")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="包含KuaiRand日志CSV文件的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="输出目录（默认: reports）。",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help="采样用户数量（用于加速计算，默认不采样）。",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="用户最少交互次数（默认: 5）。",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="聚类数量（默认: 5）。",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=["kmeans", "hierarchical"],
        default="kmeans",
        help="聚类方法（默认: kmeans）。",
    )
    parser.add_argument(
        "--similarity-metric",
        type=str,
        choices=["jaccard", "cosine"],
        default="jaccard",
        help="相似度度量方法（默认: jaccard）。",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示进度信息。",
    )
    parser.add_argument(
        "--max-heatmap-users",
        type=int,
        default=500,
        help="热力图最大显示用户数（默认: 500）。",
    )
    parser.add_argument(
        "--skip-silhouette",
        action="store_true",
        help="跳过轮廓系数计算（大规模数据时推荐使用，可显著加速）。",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=None,
        help="计算轮廓系数时的采样大小（默认不采样，使用全部数据）。",
    )

    args = parser.parse_args()

    # 确定数据目录
    default_data_dir = (
        Path(__file__).resolve().parents[1] / "data" / "KuaiRand-1K" / "data"
    )
    data_dir = (args.data_dir or default_data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据目录 '{data_dir}' 不存在。"
            "请使用 --data-dir 指定包含KuaiRand CSV文件的目录。"
        )

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.show_progress:
        print("加载数据...")
    df, log_paths = _load_log_frames(data_dir, args.show_progress)
    if args.show_progress:
        print(f"加载了 {len(df):,} 条交互记录，来自 {len(log_paths)} 个日志文件。")

    # 构建用户-物品矩阵
    user_item_matrix, user_ids, item_ids = compute_user_item_matrix(
        df,
        sample_users=args.sample_users,
        min_interactions=args.min_interactions,
        show_progress=args.show_progress,
    )

    n_users = user_item_matrix.shape[0]
    n_items = user_item_matrix.shape[1]
    
    if args.show_progress:
        print(
            f"用户-物品矩阵: {n_users:,} 用户 × "
            f"{n_items:,} 物品"
        )
        
        # 内存使用估算和警告
        estimated_memory_gb = (n_users * n_users * 8) / (1024**3)  # float64
        print(
            f"预计相似度矩阵内存使用: ~{estimated_memory_gb:.2f} GB"
        )
        if estimated_memory_gb > 5:
            print(
                f"⚠️  警告: 相似度矩阵较大，可能需要较多内存和时间。"
            )
            if not args.skip_silhouette:
                print(
                    f"   建议使用 --skip-silhouette 跳过轮廓系数计算以加速。"
                )

    # 计算相似度矩阵
    if args.similarity_metric == "jaccard":
        similarity_matrix = compute_jaccard_similarity(
            user_item_matrix, show_progress=args.show_progress
        )
    else:
        similarity_matrix = compute_cosine_similarity(
            user_item_matrix, show_progress=args.show_progress
        )

    if args.show_progress:
        # 计算用户平均访问的物品数量
        user_item_counts = np.array(user_item_matrix.sum(axis=1)).flatten()
        avg_items_per_user = user_item_counts.mean()
        
        # 排除对角线后的相似度统计（对角线都是1，会影响统计）
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        off_diagonal_sim = similarity_matrix[mask]
        
        # 计算分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(off_diagonal_sim, percentiles)
        
        # 统计高相似度用户对的数量
        high_sim_thresholds = [0.1, 0.2, 0.3, 0.5]
        high_sim_counts = {
            threshold: (off_diagonal_sim >= threshold).sum()
            for threshold in high_sim_thresholds
        }
        
        print(
            f"\n相似度矩阵统计:"
        )
        print(
            f"  用户平均访问物品数: {avg_items_per_user:.1f} "
            f"(范围: {user_item_counts.min():.0f} - {user_item_counts.max():.0f})"
        )
        print(
            f"  总物品数: {user_item_matrix.shape[1]:,}"
        )
        print(
            f"  完整矩阵统计 (包含对角线): "
            f"均值={similarity_matrix.mean():.4f}, "
            f"中位数={np.median(similarity_matrix):.4f}"
        )
        print(
            f"  非对角线相似度统计 (排除自己): "
            f"均值={off_diagonal_sim.mean():.4f}, "
            f"中位数={np.median(off_diagonal_sim):.4f}, "
            f"最大值={off_diagonal_sim.max():.4f}"
        )
        print(
            f"  相似度分位数: "
            + ", ".join([
                f"P{p}={v:.4f}" 
                for p, v in zip(percentiles, percentile_values)
            ])
        )
        print(
            f"  高相似度用户对数量: "
            + ", ".join([
                f"≥{t:.1f}: {cnt:,} ({cnt*100/len(off_diagonal_sim):.2f}%)"
                for t, cnt in high_sim_counts.items()
            ])
        )
        
        # 判断是否存在聚类效应
        if off_diagonal_sim.mean() < 0.01:
            print(
                f"\n⚠️  警告: 用户间平均相似度很低 ({off_diagonal_sim.mean():.4f})，"
                f"可能存在以下情况："
            )
            print(
                f"  1. 物品空间很大，用户访问的物品集合重叠很小"
            )
            print(
                f"  2. 用户偏好差异很大，不存在明显的聚类效应"
            )
            print(
                f"  3. 可能需要增加采样用户数或调整min-interactions参数"
            )
        elif off_diagonal_sim.mean() > 0.1:
            print(
                f"\n✓ 用户间平均相似度较高 ({off_diagonal_sim.mean():.4f})，"
                f"可能存在聚类效应"
            )

    # 验证cluster数量的合理性
    n_users = user_item_matrix.shape[0]
    if args.n_clusters > n_users:
        raise ValueError(
            f"错误: cluster数量 ({args.n_clusters}) 不能大于用户数量 ({n_users})"
        )
    if args.n_clusters > n_users / 2:
        print(
            f"\n⚠️  警告: cluster数量 ({args.n_clusters}) 相对于用户数量 ({n_users}) 较多，"
            f"平均每个cluster只有 {n_users/args.n_clusters:.1f} 个用户。"
        )
        print(
            f"   这可能导致过度分割，建议cluster数量不超过用户数量的1/10 ({n_users//10})。"
        )
    elif args.n_clusters < n_users / 100:
        print(
            f"\n💡 提示: cluster数量 ({args.n_clusters}) 相对较少，"
            f"平均每个cluster有 {n_users/args.n_clusters:.1f} 个用户。"
        )
        print(
            f"   如果用户偏好差异较大，可能需要更多cluster来区分不同的用户群体。"
        )

    # 执行聚类
    if args.clustering_method == "kmeans":
        labels, silhouette = perform_kmeans_clustering(
            user_item_matrix,
            similarity_matrix,
            args.n_clusters,
            compute_silhouette=not args.skip_silhouette,
            silhouette_sample_size=args.silhouette_sample_size,
        )
        linkage_matrix = None
        if args.show_progress:
            if silhouette is not None:
                print(f"K-means聚类完成，轮廓系数: {silhouette:.4f}")
            else:
                print(f"K-means聚类完成（已跳过轮廓系数计算）")
    else:
        labels, linkage_matrix = perform_hierarchical_clustering(
            similarity_matrix,
            args.n_clusters,
        )
        # 计算轮廓系数（可选）
        silhouette = None
        if not args.skip_silhouette:
            if args.silhouette_sample_size is not None and args.silhouette_sample_size < n_users:
                # 采样计算
                sample_indices = np.random.choice(
                    n_users, size=args.silhouette_sample_size, replace=False
                )
                sample_labels = labels[sample_indices]
                sample_distance = 1 - similarity_matrix[np.ix_(sample_indices, sample_indices)]
                np.fill_diagonal(sample_distance, 0)
                silhouette = silhouette_score(
                    sample_distance, sample_labels, metric="precomputed"
                )
            else:
                # 完整计算
                distance_matrix = 1 - similarity_matrix
                np.fill_diagonal(distance_matrix, 0)
                silhouette = silhouette_score(
                    distance_matrix, labels, metric="precomputed"
                )
        if args.show_progress:
            if silhouette is not None:
                print(f"层次聚类完成，轮廓系数: {silhouette:.4f}")
            else:
                print(f"层次聚类完成（已跳过轮廓系数计算）")

    # 分析聚类统计
    cluster_stats = analyze_cluster_statistics(
        user_ids, user_item_matrix, labels, similarity_matrix
    )

    if args.show_progress:
        # 计算cluster大小统计
        cluster_sizes = [stat["size"] for stat in cluster_stats["cluster_statistics"]]
        cluster_sizes_array = np.array(cluster_sizes)
        
        print(f"\n聚类分析结果:")
        print(f"  聚类数量: {cluster_stats['n_clusters']}")
        print(f"  总用户数: {cluster_stats['total_users']}")
        print(f"  平均每个cluster用户数: {np.mean(cluster_sizes_array):.1f}")
        print(f"  Cluster大小统计: "
              f"最小={cluster_sizes_array.min()}, "
              f"最大={cluster_sizes_array.max()}, "
              f"中位数={np.median(cluster_sizes_array):.1f}, "
              f"标准差={cluster_sizes_array.std():.1f}")
        
        # 统计小cluster（少于5个用户）的数量
        small_clusters = (cluster_sizes_array < 5).sum()
        if small_clusters > 0:
            print(f"  ⚠️  小cluster数量（<5用户）: {small_clusters} ({small_clusters*100/cluster_stats['n_clusters']:.1f}%)")
        
        print(f"  平均簇内相似度: {cluster_stats['overall_intra_cluster_similarity']:.4f}")
        print(f"  平均簇间相似度: {cluster_stats['overall_inter_cluster_similarity']:.4f}")
        print(f"  聚类质量 (簇内-簇间): {cluster_stats['clustering_quality']:.4f}")
        if silhouette is not None:
            print(f"  轮廓系数: {silhouette:.4f}")
        else:
            print(f"  轮廓系数: 未计算（使用 --skip-silhouette 跳过）")

        # 只显示前20个cluster的详细信息（如果cluster太多）
        max_display = 20
        if len(cluster_stats["cluster_statistics"]) > max_display:
            print(f"\n各聚类统计（显示前{max_display}个，按用户数排序）:")
            sorted_stats = sorted(
                cluster_stats["cluster_statistics"],
                key=lambda x: x["size"],
                reverse=True
            )
            for stat in sorted_stats[:max_display]:
                print(
                    f"  聚类 {stat['cluster_id']}: "
                    f"用户数={stat['size']}, "
                    f"簇内相似度={stat['intra_cluster_similarity']:.4f}, "
                    f"簇间相似度={stat['inter_cluster_similarity']:.4f}, "
                    f"唯一物品数={stat['unique_items']}"
                )
            print(f"  ... (还有 {len(cluster_stats['cluster_statistics']) - max_display} 个cluster未显示)")
        else:
            print(f"\n各聚类统计:")
            for stat in cluster_stats["cluster_statistics"]:
                print(
                    f"  聚类 {stat['cluster_id']}: "
                    f"用户数={stat['size']}, "
                    f"簇内相似度={stat['intra_cluster_similarity']:.4f}, "
                    f"簇间相似度={stat['inter_cluster_similarity']:.4f}, "
                    f"唯一物品数={stat['unique_items']}"
                )

    # 保存结果
    results = {
        "similarity_metric": args.similarity_metric,
        "clustering_method": args.clustering_method,
        "n_clusters": args.n_clusters,
        "silhouette_score": float(silhouette) if silhouette is not None else None,
        "cluster_statistics": cluster_stats,
        "user_cluster_mapping": {
            str(uid): int(label) for uid, label in zip(user_ids, labels)
        },
    }

    results_json_path = args.output_dir / "user_clustering_results.json"
    results_json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    if args.show_progress:
        print(f"\n结果已保存到: {results_json_path}")

    # 绘制可视化
    if plt is not None:
        # 相似度热力图
        heatmap_path = args.output_dir / "user_similarity_heatmap.png"
        plot_similarity_heatmap(
            similarity_matrix,
            heatmap_path,
            labels=labels,
            max_users=args.max_heatmap_users,
        )
        if args.show_progress:
            print(f"相似度热力图已保存到: {heatmap_path}")

        # 聚类分布
        cluster_dist_path = args.output_dir / "cluster_distribution.png"
        plot_cluster_distribution(labels, similarity_matrix, cluster_dist_path)
        if args.show_progress:
            print(f"聚类分布图已保存到: {cluster_dist_path}")

        # 层次聚类树状图（如果使用层次聚类）
        if linkage_matrix is not None:
            dendrogram_path = args.output_dir / "dendrogram.png"
            plot_dendrogram(linkage_matrix, dendrogram_path)
            if args.show_progress:
                print(f"树状图已保存到: {dendrogram_path}")

    # 保存用户聚类映射CSV
    user_cluster_df = pd.DataFrame({
        "user_id": user_ids,
        "cluster_id": labels,
    })
    user_cluster_csv_path = args.output_dir / "user_cluster_mapping.csv"
    user_cluster_df.to_csv(user_cluster_csv_path, index=False)
    if args.show_progress:
        print(f"用户聚类映射已保存到: {user_cluster_csv_path}")


if __name__ == "__main__":
    main()


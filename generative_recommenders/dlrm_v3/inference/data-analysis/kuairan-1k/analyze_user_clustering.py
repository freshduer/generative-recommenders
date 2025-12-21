#!/usr/bin/env python3
"""分析用户之间的item重合度和聚类效应，用于GPU调度优化。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
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


def _load_single_csv(csv_path: Path) -> pd.DataFrame:
    """加载单个CSV文件（用于多进程）"""
    usecols = ["user_id", "video_id"]
    try:
        # 尝试使用pyarrow引擎（更快，如果可用）
        try:
            frame = pd.read_csv(
                csv_path,
                usecols=usecols,
                dtype={
                    "user_id": "int32",  # 使用int32节省内存和加速
                    "video_id": "int32",
                },
                engine="pyarrow",
                na_filter=False,
            )
        except (ImportError, ValueError, TypeError):
            # 回退到C引擎
            frame = pd.read_csv(
                csv_path,
                usecols=usecols,
                dtype={
                    "user_id": "int32",
                    "video_id": "int32",
                },
                engine="c",
                low_memory=False,
                na_filter=False,
            )
    except KeyError:
        # 如果某些文件没有这些列，尝试读取所有列
        frame = pd.read_csv(
            csv_path,
            dtype={
                "user_id": "int32",
                "video_id": "int32",
            },
            engine="c",
            low_memory=False,
        )
        if "user_id" in frame.columns and "video_id" in frame.columns:
            frame = frame[["user_id", "video_id"]]
        else:
            raise ValueError(f"文件 {csv_path} 缺少必需的列: user_id, video_id")
    
    return frame


def _load_log_frames(
    data_dir: Path, show_progress: bool, n_workers: Optional[int] = None
) -> Tuple[pd.DataFrame, List[Path]]:
    """加载并合并所有KuaiRand日志CSV文件（多进程优化版本）。"""
    log_paths = sorted(data_dir.glob("log_standard*.csv"))
    if not log_paths:
        msg = (
            "未找到KuaiRand日志文件。期望文件格式："
            "'log_standard_4_08_to_4_21_1k.csv'。"
        )
        raise FileNotFoundError(msg)

    # 确定工作进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(log_paths), 4)  # 减少到最多4个进程，避免内存问题
    
    frames = []
    if n_workers > 1 and len(log_paths) > 1:
        # 多进程并行读取（带错误处理和重试）
        if show_progress:
            print(f"尝试使用 {n_workers} 个进程并行读取 {len(log_paths)} 个文件...")
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_path = {
                    executor.submit(_load_single_csv, path): path 
                    for path in log_paths
                }
                
                iter_futures = future_to_path.items()
                if show_progress and tqdm is not None:
                    iter_futures = tqdm(future_to_path.items(), desc="读取日志", unit="文件", total=len(log_paths))
                
                failed_paths = []
                for future_item in iter_futures:
                    if isinstance(future_item, tuple):
                        future, path = future_item
                    else:
                        future = future_item
                        path = future_to_path[future]
                    
                    try:
                        frame = future.result(timeout=600)  # 10分钟超时
                        frames.append(frame)
                    except Exception as e:
                        if show_progress:
                            print(f"警告: 读取文件 {path.name} 失败: {e}")
                        failed_paths.append(path)
            
            # 如果多进程读取失败太多，回退到单进程
            if len(frames) == 0:
                if show_progress:
                    print(f"所有文件多进程读取失败，回退到单进程读取...")
                n_workers = 1
            elif len(failed_paths) > 0:
                if show_progress:
                    print(f"{len(failed_paths)} 个文件读取失败，尝试单进程重新读取...")
                # 单进程重试失败的文件
                for path in failed_paths:
                    try:
                        frame = _load_single_csv(path)
                        frames.append(frame)
                        if show_progress:
                            print(f"成功重新读取文件: {path.name}")
                    except Exception as e:
                        if show_progress:
                            print(f"错误: 重新读取文件 {path.name} 仍然失败: {e}")
                        raise RuntimeError(f"无法读取文件 {path.name}: {e}")
        except Exception as e:
            if show_progress:
                print(f"多进程读取出现严重错误: {e}")
                print(f"回退到单进程读取...")
            n_workers = 1
            frames = []
    # 单进程读取（如果多进程失败或未启用）
    if n_workers == 1 or len(frames) == 0:
        if show_progress:
            print(f"使用单进程读取 {len(log_paths)} 个文件...")
        iter_paths = log_paths
        if show_progress and tqdm is not None:
            iter_paths = tqdm(log_paths, desc="读取日志", unit="文件")
        elif show_progress and tqdm is None:
            print("tqdm未安装；继续执行但不显示进度条。")
        
        frames = []
        for csv_path in iter_paths:
            try:
                frame = _load_single_csv(csv_path)
                frames.append(frame)
            except Exception as e:
                if show_progress:
                    print(f"错误: 读取文件 {csv_path.name} 失败: {e}")
                raise RuntimeError(f"无法读取文件 {csv_path.name}: {e}")

    # 检查是否有数据
    if len(frames) == 0:
        raise RuntimeError("没有成功读取任何数据文件！请检查数据目录和文件格式。")

    if show_progress:
        print(f"成功读取 {len(frames)} 个文件，开始合并数据框...")
    # 使用sort=False加速concat
    concatenated = pd.concat(frames, ignore_index=True, sort=False)
    if show_progress:
        print(f"数据合并完成，共 {len(concatenated):,} 条记录")
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

    if show_progress:
        print("构建用户-物品矩阵...")
    
    # 优化：使用更高效的方法获取唯一物品
    unique_items = np.sort(df_filtered["video_id"].unique())
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    user_to_idx = {user: idx for idx, user in enumerate(valid_users)}
    
    # 优化：使用groupby获取每个用户的唯一物品（使用numpy unique更快）
    user_items_dict = df_filtered.groupby("user_id", sort=False)["video_id"].apply(
        lambda x: np.unique(x.values)
    ).to_dict()

    # 构建稀疏矩阵的索引
    rows = []
    cols = []
    
    # 使用列表推导式可能更快，但这里保持循环以便调试
    for user_id, items in user_items_dict.items():
        user_idx = user_to_idx[user_id]
        for item_id in items:
            item_idx = item_to_idx[item_id]
            rows.append(user_idx)
            cols.append(item_idx)

    # 使用float32节省内存
    data = np.ones(len(rows), dtype=np.float32)
    user_item_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(valid_users), len(unique_items)),
        dtype=np.float32
    )

    user_ids = np.array(valid_users, dtype=np.int32)
    item_ids = np.array(unique_items, dtype=np.int32)

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

    # 优化：使用float32减少内存使用
    # 计算交集：矩阵乘法得到交集大小
    intersection = user_item_matrix.dot(user_item_matrix.T)
    intersection = intersection.toarray().astype(np.float32)

    # 计算并集：|A ∪ B| = |A| + |B| - |A ∩ B|
    user_sizes = np.array(user_item_matrix.sum(axis=1), dtype=np.float32).flatten()
    union = user_sizes[:, None] + user_sizes[None, :] - intersection

    # 避免除以零
    union = np.maximum(union, 1e-10)
    jaccard = (intersection / union).astype(np.float32)

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

    # 优化：使用float32
    # L2归一化
    norms = np.sqrt(np.array(user_item_matrix.power(2).sum(axis=1), dtype=np.float32)).flatten()
    norms = np.maximum(norms, 1e-10)
    normalized_matrix = user_item_matrix.multiply(1.0 / norms[:, None])

    # 计算余弦相似度
    cosine = normalized_matrix.dot(normalized_matrix.T).toarray().astype(np.float32)

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
    
    # 优化：对于大规模数据，直接使用TruncatedSVD降维（比PCA快）
    if n_users > 1000:
        n_components = min(50, n_users - 1, user_item_matrix.shape[1])
        if n_users > 10000:
            n_components = min(30, n_users - 1, user_item_matrix.shape[1])
        
        # 使用TruncatedSVD（适合稀疏矩阵，比PCA快）
        svd = TruncatedSVD(
            n_components=n_components, 
            random_state=random_state,
            n_iter=5,  # 减少迭代次数以加速
            algorithm='arpack'  # 对于稀疏矩阵更快
        )
        features = svd.fit_transform(user_item_matrix).astype(np.float32)
    else:
        features = similarity_matrix.astype(np.float32)

    # 优化：减少n_init次数（大规模数据时3-5次通常足够）
    n_init = 3 if n_users > 5000 else 5
    
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=n_init,
        max_iter=300,  # 减少最大迭代次数
        algorithm='lloyd',  # 明确指定算法
    )
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="并行读取CSV文件的进程数（默认：自动选择，最多8个）。",
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始加载数据...")
    load_start = time.time()
    n_workers = getattr(args, 'n_workers', None)
    df, log_paths = _load_log_frames(data_dir, args.show_progress, n_workers=n_workers)
    load_time = time.time() - load_start
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 数据加载完成: {len(df):,} 条交互记录，来自 {len(log_paths)} 个日志文件 (耗时: {load_time:.1f}秒)")

    # 构建用户-物品矩阵
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始构建用户-物品矩阵...")
    matrix_start = time.time()
    user_item_matrix, user_ids, item_ids = compute_user_item_matrix(
        df,
        sample_users=args.sample_users,
        min_interactions=args.min_interactions,
        show_progress=args.show_progress,
    )
    matrix_time = time.time() - matrix_start

    n_users = user_item_matrix.shape[0]
    n_items = user_item_matrix.shape[1]
    
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 用户-物品矩阵构建完成: {n_users:,} 用户 × {n_items:,} 物品 (耗时: {matrix_time:.1f}秒)")
        
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
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始计算相似度矩阵 (方法: {args.similarity_metric})...")
    similarity_start = time.time()
    if args.similarity_metric == "jaccard":
        similarity_matrix = compute_jaccard_similarity(
            user_item_matrix, show_progress=args.show_progress
        )
    else:
        similarity_matrix = compute_cosine_similarity(
            user_item_matrix, show_progress=args.show_progress
        )
    similarity_time = time.time() - similarity_start
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 相似度矩阵计算完成 (耗时: {similarity_time:.1f}秒, {similarity_time/60:.1f}分钟)")

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
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行聚类 (方法: {args.clustering_method}, n_clusters: {args.n_clusters})...")
    clustering_start = time.time()
    if args.clustering_method == "kmeans":
        labels, silhouette = perform_kmeans_clustering(
            user_item_matrix,
            similarity_matrix,
            args.n_clusters,
            compute_silhouette=not args.skip_silhouette,
            silhouette_sample_size=args.silhouette_sample_size,
        )
        linkage_matrix = None
        clustering_time = time.time() - clustering_start
        if args.show_progress:
            if silhouette is not None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] K-means聚类完成，轮廓系数: {silhouette:.4f} (耗时: {clustering_time:.1f}秒, {clustering_time/60:.1f}分钟)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] K-means聚类完成（已跳过轮廓系数计算）(耗时: {clustering_time:.1f}秒, {clustering_time/60:.1f}分钟)")
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
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始分析聚类统计...")
    stats_start = time.time()
    cluster_stats = analyze_cluster_statistics(
        user_ids, user_item_matrix, labels, similarity_matrix
    )
    stats_time = time.time() - stats_start
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 聚类统计分析完成 (耗时: {stats_time:.1f}秒)")

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
    # 保存用户聚类映射
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始保存结果...")
    save_start = time.time()
    
    user_cluster_df = pd.DataFrame({
        "user_id": user_ids,
        "cluster_id": labels,
    })
    user_cluster_csv_path = args.output_dir / "user_cluster_mapping.csv"
    user_cluster_df.to_csv(user_cluster_csv_path, index=False)
    save_time = time.time() - save_start
    total_time = time.time() - load_start
    
    if args.show_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 用户聚类映射已保存到: {user_cluster_csv_path}")
        print("")
        print("=" * 80)
        print("实验完成!")
        print("=" * 80)
        print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟, {total_time/3600:.2f}小时)")
        if hasattr(args, 'show_progress') and args.show_progress:
            print(f"  数据加载: {load_time:.1f}秒 ({load_time/total_time*100:.1f}%)")
            print(f"  矩阵构建: {matrix_time:.1f}秒 ({matrix_time/total_time*100:.1f}%)")
            print(f"  相似度计算: {similarity_time:.1f}秒 ({similarity_time/total_time*100:.1f}%)")
            print(f"  聚类执行: {clustering_time:.1f}秒 ({clustering_time/total_time*100:.1f}%)")
            print(f"  统计分析: {stats_time:.1f}秒 ({stats_time/total_time*100:.1f}%)")
            print(f"  结果保存: {save_time:.1f}秒 ({save_time/total_time*100:.1f}%)")
        print("=" * 80)


if __name__ == "__main__":
    main()


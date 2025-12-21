#!/usr/bin/env python3
"""并行运行多个聚类实验，从200到2000的n-clusters"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
from datetime import datetime

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 创建一个占位符类
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else kwargs.get('iterable', [])
            self.total = kwargs.get('total', len(self.iterable))
            self.n = 0
        def __iter__(self):
            return iter(self.iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def set_description(self, desc):
            print(f"\n{desc}")
        def set_postfix(self, **kwargs):
            pass

# 设置日志，同时输出到控制台和文件
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def run_single_clustering_experiment(
    n_clusters: int,
    data_dir: Path,
    output_dir: Path,
    log_dir: Path,
    sample_users: int = 27000,
    min_interactions: int = 5,
    clustering_method: str = "kmeans",
    similarity_metric: str = "jaccard",
    skip_silhouette: bool = True,
    n_workers: Optional[int] = None,
) -> tuple[int, bool, str]:
    """
    运行单个聚类实验
    
    Returns:
        (n_clusters, success, message)
    """
    script_path = Path(__file__).parent / "analyze_user_clustering.py"
    log_file = log_dir / f"clustering_n{n_clusters}.log"
    err_file = log_dir / f"clustering_n{n_clusters}.err"
    
    # 为每个实验创建独立的输出目录
    exp_output_dir = output_dir / f"n_clusters_{n_clusters}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir", str(data_dir),
        "--output-dir", str(exp_output_dir),
        "--sample-users", str(sample_users),
        "--min-interactions", str(min_interactions),
        "--n-clusters", str(n_clusters),
        "--clustering-method", clustering_method,
        "--similarity-metric", similarity_metric,
        "--skip-silhouette" if skip_silhouette else "",
        "--show-progress",
    ]
    if n_workers is not None:
        cmd.extend(["--n-workers", str(n_workers)])
    # 移除空字符串
    cmd = [c for c in cmd if c]
    
    # 写入开始信息到日志文件
    with open(log_file, "w") as log_f:
        log_f.write(f"开始实验: n_clusters={n_clusters}\n")
        log_f.write(f"开始时间: {start_time_str}\n")
        log_f.write(f"命令: {' '.join(cmd)}\n")
        log_f.write("=" * 80 + "\n\n")
    
    start_time = time.time()
    try:
        with open(log_file, "a") as log_f, open(err_file, "w") as err_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=err_f,
                cwd=script_path.parent,
                timeout=3600 * 2,  # 2小时超时
            )
        
        elapsed = time.time() - start_time
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 追加结束信息到日志文件
        with open(log_file, "a") as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"结束时间: {end_time_str}\n")
            log_f.write(f"总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)\n")
            log_f.write(f"返回码: {result.returncode}\n")
        
        if result.returncode == 0:
            msg = f"✓ n_clusters={n_clusters} 完成 (耗时: {elapsed:.1f}秒, {elapsed/60:.1f}分钟)"
            logger.info(msg)
            return (n_clusters, True, msg)
        else:
            msg = f"✗ n_clusters={n_clusters} 失败 (返回码: {result.returncode}, 耗时: {elapsed:.1f}秒, {elapsed/60:.1f}分钟)"
            logger.error(msg)
            return (n_clusters, False, msg)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"结束时间: {end_time_str}\n")
            log_f.write(f"状态: 超时\n")
            log_f.write(f"耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)\n")
        msg = f"✗ n_clusters={n_clusters} 超时 (耗时: {elapsed:.1f}秒, {elapsed/60:.1f}分钟)"
        logger.error(msg)
        return (n_clusters, False, msg)
    except Exception as e:
        elapsed = time.time() - start_time
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"结束时间: {end_time_str}\n")
            log_f.write(f"状态: 异常\n")
            log_f.write(f"错误: {str(e)}\n")
            log_f.write(f"耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)\n")
        msg = f"✗ n_clusters={n_clusters} 异常: {str(e)} (耗时: {elapsed:.1f}秒, {elapsed/60:.1f}分钟)"
        logger.error(msg)
        return (n_clusters, False, msg)


def main():
    parser = argparse.ArgumentParser(
        description="并行运行多个聚类实验"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/comp/cswjyu/data/KuaiRand-27K/data"),
        help="数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/clustering_experiments"),
        help="输出目录",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/clustering_experiments"),
        help="日志目录",
    )
    parser.add_argument(
        "--n-clusters-start",
        type=int,
        default=200,
        help="起始聚类数量",
    )
    parser.add_argument(
        "--n-clusters-end",
        type=int,
        default=2000,
        help="结束聚类数量",
    )
    parser.add_argument(
        "--n-clusters-step",
        type=int,
        default=100,
        help="聚类数量步长",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=27000,
        help="采样用户数量",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="最少交互次数",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "hierarchical"],
        help="聚类方法",
    )
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "cosine"],
        help="相似度度量",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="最大并行进程数（默认：CPU核心数）",
    )
    parser.add_argument(
        "--skip-silhouette",
        action="store_true",
        default=True,
        help="跳过轮廓系数计算",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="每个实验的并行读取进程数（默认：自动选择，最多8个）。",
    )
    
    args = parser.parse_args()
    
    # 创建输出和日志目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成所有n_clusters值
    n_clusters_list = list(range(
        args.n_clusters_start,
        args.n_clusters_end + 1,
        args.n_clusters_step
    ))
    
    logger.info("=" * 80)
    logger.info("聚类实验配置")
    logger.info("=" * 80)
    logger.info(f"实验总数: {len(n_clusters_list)} 个")
    logger.info(f"n_clusters范围: {args.n_clusters_start} 到 {args.n_clusters_end} (步长: {args.n_clusters_step})")
    logger.info(f"n_clusters列表: {n_clusters_list}")
    logger.info(f"最大并行进程数: {args.max_workers or 'CPU核心数'}")
    logger.info(f"每个实验的读取进程数: {args.n_workers or '自动选择(最多8个)'}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"日志目录: {args.log_dir}")
    logger.info(f"采样用户数: {args.sample_users}")
    logger.info(f"最少交互次数: {args.min_interactions}")
    logger.info(f"聚类方法: {args.clustering_method}")
    logger.info(f"相似度度量: {args.similarity_metric}")
    logger.info(f"跳过轮廓系数: {args.skip_silhouette}")
    logger.info("=" * 80)
    logger.info("")
    
    # 准备参数
    max_workers = args.max_workers
    
    # 运行实验
    start_time = time.time()
    results = []
    completed_count = 0
    running_experiments = set()
    
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] 开始提交所有实验任务...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_n_clusters = {
            executor.submit(
                run_single_clustering_experiment,
                n_clusters=n_clusters,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                log_dir=args.log_dir,
                sample_users=args.sample_users,
                min_interactions=args.min_interactions,
                clustering_method=args.clustering_method,
                similarity_metric=args.similarity_metric,
                skip_silhouette=args.skip_silhouette,
                n_workers=args.n_workers,
            ): n_clusters
            for n_clusters in n_clusters_list
        }
        
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] 已提交 {len(future_to_n_clusters)} 个任务，开始执行...")
        logger.info("")
        
        # 使用tqdm显示进度条
        with tqdm(total=len(n_clusters_list), desc="实验进度", unit="实验", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            # 收集结果
            for future in as_completed(future_to_n_clusters):
                n_clusters = future_to_n_clusters[future]
                running_experiments.discard(n_clusters)
                
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    n_clusters_val, success, msg = result
                    elapsed_total = time.time() - start_time
                    avg_time = elapsed_total / completed_count if completed_count > 0 else 0
                    remaining = len(n_clusters_list) - completed_count
                    est_remaining_time = avg_time * remaining if remaining > 0 else 0
                    
                    # 更新进度条
                    status_icon = "✓" if success else "✗"
                    pbar.set_description(f"实验进度 [{status_icon} n_clusters={n_clusters_val}]")
                    pbar.set_postfix({
                        '完成': f"{completed_count}/{len(n_clusters_list)}",
                        '成功': sum(1 for _, s, _ in results if s),
                        '失败': sum(1 for _, s, _ in results if not s),
                        '预计剩余': f"{est_remaining_time/60:.1f}分钟" if est_remaining_time > 0 else "计算中"
                    })
                    pbar.update(1)
                    
                    # 详细日志
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    logger.info(f"  进度: {completed_count}/{len(n_clusters_list)} 完成 | "
                              f"成功: {sum(1 for _, s, _ in results if s)} | "
                              f"失败: {sum(1 for _, s, _ in results if not s)} | "
                              f"总耗时: {elapsed_total/60:.1f}分钟 | "
                              f"平均: {avg_time/60:.1f}分钟/实验 | "
                              f"预计剩余: {est_remaining_time/60:.1f}分钟")
                    logger.info("")
                    
                except Exception as e:
                    completed_count += 1
                    error_msg = f"执行异常: {str(e)}"
                    logger.error(f"n_clusters={n_clusters} {error_msg}")
                    results.append((n_clusters, False, error_msg))
                    pbar.update(1)
    
    # 汇总结果
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    avg_time_per_exp = total_time / len(results) if results else 0
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("实验汇总")
    logger.info("=" * 80)
    logger.info(f"总实验数: {len(results)}")
    logger.info(f"成功: {successful} ({successful*100/len(results):.1f}%)")
    logger.info(f"失败: {failed} ({failed*100/len(results):.1f}%)")
    logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟, {total_time/3600:.2f}小时)")
    logger.info(f"平均每个实验耗时: {avg_time_per_exp:.1f}秒 ({avg_time_per_exp/60:.1f}分钟)")
    logger.info("")
    
    # 显示成功的实验
    successful_experiments = [n for n, s, _ in results if s]
    if successful_experiments:
        logger.info(f"成功的实验 (n_clusters): {sorted(successful_experiments)}")
        logger.info("")
    
    # 显示失败的实验
    if failed > 0:
        failed_experiments = [n for n, s, _ in results if not s]
        logger.warning(f"失败的实验 (n_clusters): {sorted(failed_experiments)}")
        logger.warning("请查看对应的日志文件了解详情:")
        for n_clusters in sorted(failed_experiments):
            log_file = args.log_dir / f"clustering_n{n_clusters}.log"
            err_file = args.log_dir / f"clustering_n{n_clusters}.err"
            logger.warning(f"  - n_clusters={n_clusters}:")
            logger.warning(f"    日志: {log_file}")
            logger.warning(f"    错误: {err_file}")
        logger.info("")
    
    logger.info("=" * 80)
    
    # 保存汇总结果
    summary_file = args.output_dir / "experiment_summary.txt"
    with open(summary_file, "w") as f:
        f.write("聚类实验汇总\n")
        f.write("=" * 80 + "\n")
        f.write(f"总实验数: {len(results)}\n")
        f.write(f"成功: {successful}\n")
        f.write(f"失败: {failed}\n")
        f.write(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)\n")
        f.write("\n详细结果:\n")
        for n_clusters, success, msg in sorted(results):
            status = "✓" if success else "✗"
            f.write(f"{status} n_clusters={n_clusters}: {msg}\n")
    
    logger.info(f"汇总结果已保存到: {summary_file}")
    
    # 如果有失败的实验，列出它们
    if failed > 0:
        failed_experiments = [n for n, success, _ in results if not success]
        logger.warning(f"失败的实验 (n_clusters): {failed_experiments}")
        logger.warning("请查看对应的日志文件了解详情")


if __name__ == "__main__":
    main()


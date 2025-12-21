#!/bin/bash
#
# SLURM 脚本：并行运行聚类实验（n_clusters从200到2000）
#

# 设置作业的名称
#SBATCH --job-name=ClusteringExp

# 指定要使用的分区
#SBATCH --partition=long

# 请求的节点数
#SBATCH --nodes=1

# 请求的 CPU 核心数（用于多进程并行）
#SBATCH --cpus-per-task=64

# 请求的内存
#SBATCH --mem=450G

# 设置最长运行时间（根据实验数量调整，每个实验可能需要较长时间）
#SBATCH --time=12:00:00

# 标准输出和错误输出文件路径
#SBATCH --output=logs/clustering_experiments_%j.out
#SBATCH --error=logs/clustering_experiments_%j.err

# --------------------------------------------------------------------------------
# >>> conda initialize >>>
__conda_setup="$('/home/comp/cswjyu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/comp/cswjyu/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/comp/cswjyu/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/comp/cswjyu/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# 激活 Conda 环境
echo "激活 Conda 环境: hstu_py310"
conda activate hstu_py310

# 导航到脚本执行目录
EXEC_DIR="/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference/data-analysis/kuairan-1k"
echo "导航到执行目录: $EXEC_DIR"
cd "$EXEC_DIR" || { echo "无法进入目录 $EXEC_DIR. 退出."; exit 1; }

# 检查 Python 及其环境
echo "当前 Python 路径: $(which python)"
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
echo "可用CPU核心数: $SLURM_CPUS_PER_TASK"

# 创建必要的目录
mkdir -p logs/clustering_experiments
mkdir -p reports/clustering_experiments

# 运行并行聚类实验
# 参数说明：
# --n-clusters-start 200: 起始聚类数量
# --n-clusters-end 2000: 结束聚类数量
# --n-clusters-step 100: 步长（200, 300, 400, ..., 2000）
# --max-workers: 使用所有可用的CPU核心
# --sample-users 27000: 使用27K用户数据集
python run_clustering_experiments.py \
  --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
  --output-dir reports/clustering_experiments \
  --log-dir logs/clustering_experiments \
  --n-clusters-start 200 \
  --n-clusters-end 2000 \
  --n-clusters-step 100 \
  --sample-users 27000 \
  --min-interactions 5 \
  --clustering-method kmeans \
  --similarity-metric jaccard \
  --max-workers $SLURM_CPUS_PER_TASK \
  --n-workers 8 \
  --skip-silhouette

echo "脚本执行完成。"


#!/bin/bash
#
# SLURM 脚本示例
#
# 注意：请根据您的集群的具体要求调整以下参数。

# 设置作业的名称
#SBATCH --job-name=InferenceTest

# 指定要使用的分区
#SBATCH --partition=short

# 请求的节点数
#SBATCH --nodes=1

# 请求的 GPU 数量
#SBATCH --gres=gpu:2

# 指定节点列表（node25, 26、node27 或 node28）
# SBATCH --nodelist=gpu25

# 请求的 CPU 核心数
#SBATCH --cpus-per-task=32

# 请求的内存
#SBATCH --mem=256G

# 设置最长运行时间
#SBATCH --time=01:00:00

# 标准输出和错误输出文件路径
#SBATCH --output=logs-shard/%x_%j.out
#SBATCH --error=logs-shard/%x_%j.err

# --------------------------------------------------------------------------------
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
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

# 1. 激活 Conda 环境
# 假设 conda 初始化脚本已在您的 .bashrc 或 .profile 中配置
# 如果没有，您可能需要先 source 您 conda 安装路径下的 etc/profile.d/conda.sh
echo "激活 Conda 环境: hstu_py310"
conda env list
conda init
conda activate hstu_py310

# 2. 导航到脚本执行目录
EXEC_DIR="/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference"
echo "导航到执行目录: $EXEC_DIR"
cd "$EXEC_DIR" || { echo "无法进入目录 $EXEC_DIR. 退出."; exit 1; }

# 3. 检查 Python 及其环境
echo "当前 Python 路径: $(which python)"
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"

# 4. 执行 Python 脚本
echo "开始执行脚本"
# python tests/inference_test.py
# WORLD_SIZE=1 python main.py --dataset kuairand-1k
# WORLD_SIZE=1 python main.py --dataset movielens-20m
# WORLD_SIZE=1 python main.py --dataset debug
torchrun --nproc_per_node=2 main.py

# 5. 完成并清理
# Conda 环境在脚本结束时会自动去激活
echo "脚本执行完成。"
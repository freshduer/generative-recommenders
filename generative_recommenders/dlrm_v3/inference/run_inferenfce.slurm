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
#SBATCH --gres=gpu:1

# 指定节点列表（node25, 26、node27 或 node28）
#SBATCH -w gpu21

# 请求的 CPU 核心数
#SBATCH --cpus-per-task=64

# 请求的内存
#SBATCH --mem=450G

# 设置最长运行时间
#SBATCH --time=01:00:00

# 标准输出和错误输出文件路径
#SBATCH --output=logs-simulate/%x_%j.out
#SBATCH --error=logs-simulate/%x_%j.err

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
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 动态选择一个可用端口，避免与已有作业冲突
if [ -z "${MASTER_PORT}" ]; then
    MASTER_PORT=$(shuf -i 20000-29999 -n 1)
    export MASTER_PORT
fi
echo "MASTER_PORT set to ${MASTER_PORT}"

## hstu
# torchrun --nproc_per_node=1 main.py

## shard
# torchrun --nproc_per_node=2 test_custom_sharding.py
# torchrun --nproc_per_node=2 benchmark/b_emb1.py
# torchrun --nproc_per_node=1 benchmark/b_emb2.py

## test fbgem
# python benchmark/check_fbgemm_async_cumsum.py --lengths 3 5 2 4

## profile STU
# python benchmark/profile_stustack_kvcache.py --phase both --logdir ./stu_traces


## STU kv cache
# python benchmark/benchmark_stustack_kvcache.py \
#   --device cuda \
#   --kernel triton \
#   --dtype float16 \
#   --batch-size 1 \
#   --num-layers 8 \
#   --num-heads 4 \
#   --embedding-dim 256 \
#   --attention-dim 64 \
#   --hidden-dim 128 \
#   --max-uih-len 15000 \
#   --delta-size 64 \
#   --warmup 2 \
#   --iters 20

## STU kv cache with recompute
# KV swap（prefill 与 embedding 相同，decode 匹配短序列体量）
# python benchmark/benchmark_stustack_kvcache_limit3.py \
#   --num-users 200 \
#   --prefill-len 15000 \
#   --batch-size 16 \
#   --delta-size 1 \
#   --warmup-requests 400 \
#   --measure-requests 400 \
#   --stats-interval 30 \
#   --limit 2GB --limit 5GB --limit 8GB --limit 10GB --limit 12GB --limit 15GB \
#   --limit 20GB --limit 25GB --limit 30GB --limit 35GB --limit 40GB \
#   --seed 2025

# 模拟热门用户占 10%，每个会话 3~6 个请求，150 个请求后热点整体平移 40。
# python benchmark/benchmark_stustack_kvcache_recompute.py \
#   --trace-mode mixture \
#   --hot-user-ratio 0.01 \
#   --hot-request-ratio 0.85 \
#   --session-min-len 3 \
#   --session-max-len 6 \
#   --drift-interval 150 \
#   --drift-shift 40 \
#   --num-users 512 \
#   --warmup-requests 128 \
#   --measure-requests 512 \
#   --seed 2025

# embedding with cpu swap
# torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 benchmark/benchmark_embedding_cache_limit.py \
#   --batch-size 16 \
#   --warmup-steps 5 \
#   --measure-steps 100 \
#   --num-embeddings 60000000 \
#   --embedding-dim 256 \
#   --limit 2GB --limit 5GB --limit 8GB --limit 10GB --limit 12GB --limit 15GB \
#   --limit 20GB --limit 25GB --limit 30GB --limit 35GB --limit 40GB \
#   --sharding row_wise \
#   --prebuilt-pin-memory \
#   --include-cpu-baseline \
#   --resample-per-step

## simulate
python3 simulator/simulate_latency_budget.py \
    --workload simulator/workload.json \
    --baseline simulator/baseline.json \
    --num-gpus 1 \
    --gpu-mem 40 \
    --nvlink-bandwidth 150.0 \
    --sweep "5:35,10:30,20:20" \
    --out simulator/sim_singlegpu.csv \
    --plot simulator/sim_singlegpu.png

 # NVLink
 python3 simulator/simulate_latency_budget.py \
  --workload simulator/workload.json \
  --baseline simulator/baseline.json \
  --num-gpus 4 \
  --gpu-mem 40 \
  --interconnect nvlink \
  --nvlink-bandwidth 150.0 \
  --sweep "5:35,10:30,20:20" \
  --out simulator/sim_calibrated.csv \
  --plot simulator/sim_calibrated.png

# PCIe Only (e.g. 4090 集群)
python3 simulator/simulate_latency_budget.py \
  --num-gpus 4 \
  --gpu-mem 40 \
  --interconnect pcie \
  --pcie-bandwidth 16.0 \
  --interconnect-latency-ms 0.6 \
  --workload simulator/workload.json \
  --baseline simulator/baseline.json \
  --sweep "5:35,10:30,20:20,25:15,30:10,35:5" \
  --out simulator/sim_pcie.csv \
  --plot simulator/sim_pcie.png

# 5. 完成并清理
# Conda 环境在脚本结束时会自动去激活
echo "脚本执行完成。"
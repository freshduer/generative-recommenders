#!/bin/bash
# 分析热门用户的重复访问间隔时间

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/Books.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews Books - 热门用户访问间隔分析"
echo "=========================================="
echo "数据路径: ${DATA_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 运行热门用户访问间隔分析脚本
# 默认分析前10%的热门用户
python3 "${SCRIPT_DIR}/analyze_popular_users_visit_intervals.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_pct 0.2 \
    --min_visits 20 \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "热门用户访问间隔分析完成！结果已保存到: ${OUTPUT_DIR}"
echo "=========================================="


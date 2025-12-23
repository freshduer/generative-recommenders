#!/bin/bash
# 分析用户访问频率和重复访问模式

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/Electronics.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "用户访问频率和重复访问分析"
echo "=========================================="
echo "数据路径: ${DATA_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 运行分析脚本
python3 "${SCRIPT_DIR}/analyze_user_visit_frequency.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "分析完成！结果已保存到: ${OUTPUT_DIR}"
echo "=========================================="


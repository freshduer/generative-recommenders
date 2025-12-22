#!/bin/bash
# 分析 LastFM 行为数据集不同时间窗口下的用户交互序列长度分布和活跃用户数分布

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "LastFM 行为数据集时间窗口分析"
echo "=========================================="
echo "数据路径: ${DATA_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 运行分析脚本，分析所有时间窗口
python3 "${SCRIPT_DIR}/analyze_lastfm_time_window.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --all_windows \
    --show_progress

echo "=========================================="
echo "分析完成！结果保存在: ${OUTPUT_DIR}"
echo "=========================================="


#!/bin/bash
# Analyze user repeat visits in Amazon Reviews Video_Games dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/Video_Games.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews Video_Games - Repeat Visits Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run repeat visit analysis script
python3 "${SCRIPT_DIR}/analyze_user_repeat_visits.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_percent 0.1 \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "Repeat visit analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


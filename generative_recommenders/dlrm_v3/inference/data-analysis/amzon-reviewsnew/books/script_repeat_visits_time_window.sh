#!/bin/bash
# Analyze repeat visits within time windows (5min, 10min, 30min, 1h) for Amazon Reviews Books dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/Books.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews Books - Repeat Visits Time Window Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Time windows: 5min, 10min, 30min, 1h"
echo "=========================================="

# Run repeat visits time window analysis script
python3 "${SCRIPT_DIR}/analyze_repeat_visits_time_window.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "Repeat visits time window analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


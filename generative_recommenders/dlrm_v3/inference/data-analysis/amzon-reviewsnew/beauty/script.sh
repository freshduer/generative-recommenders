#!/bin/bash
# Analyze Amazon Reviews All_Beauty dataset across different time windows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/All_Beauty.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews All_Beauty Dataset Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run analysis script for all time windows
python3 "${SCRIPT_DIR}/analyze_amazon_reviews_time_window.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --all_windows \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "Analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


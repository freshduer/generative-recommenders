#!/bin/bash
# Analyze Amazon Reviews Electronics dataset across different time windows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/Electronics.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews Electronics Dataset Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run analysis script for all time windows
python3 "${SCRIPT_DIR}/analyze_amazon_reviews_time_window.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --all_windows \
    --show_progress

echo "=========================================="
echo "Analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


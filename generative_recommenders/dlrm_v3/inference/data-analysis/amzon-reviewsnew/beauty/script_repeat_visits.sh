#!/bin/bash
# Analyze user repeat visit behavior and access concentration for Amazon Reviews All_Beauty dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/data/amazon-reviews/All_Beauty.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Amazon Reviews All_Beauty - Repeat Visit Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run repeat visit analysis script
python3 "${SCRIPT_DIR}/analyze_repeat_visits.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --show_progress \
    --n_workers 32

echo "=========================================="
echo "Repeat visit analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


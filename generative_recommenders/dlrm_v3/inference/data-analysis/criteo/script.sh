#!/bin/bash
# Analyze Criteo dataset user interaction sequence length distribution and active user count distribution
# across different time windows (30min, 1h, 12h, 1d)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/comp/cswjyu/downloads/criteo-tb/days"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "Criteo Dataset Time Window Analysis"
echo "=========================================="
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run analysis script for all time windows
python3 "${SCRIPT_DIR}/analyze_criteo_time_window.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --all_windows \
    --show_progress \

echo "=========================================="
echo "Analysis completed! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


#!/bin/bash
# Analyze repeat visits within time windows (5min, 10min, 30min, 1h) for KuaiRand dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/comp/cswjyu/data/KuaiRand-27K/data"
OUTPUT_DIR="${SCRIPT_DIR}/reports"

echo "=========================================="
echo "KuaiRand - Repeat Visits Time Window Analysis"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Time windows: 5min, 10min, 30min, 1h"
echo "=========================================="

# Run repeat visits time window analysis script
python3 "${SCRIPT_DIR}/analyze_repeat_visits_time_window.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --show_progress

echo "=========================================="
echo "Repeat visits time window analysis complete! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="


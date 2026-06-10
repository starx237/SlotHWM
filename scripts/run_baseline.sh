#!/usr/bin/env bash
# Baseline 训练脚本（原论文实现对比实验）
# Usage: bash scripts/run_baseline.sh <dataset> [extra_args]
# Example: bash scripts/run_baseline.sh baseline_clevrer
# Log: log_baseline.txt (自动追加时间戳)

set -euo pipefail

DATASET="${1:-baseline_clevrer}"
CONFIG="config/${DATASET}.yaml"
WORKDIR="experiments/${DATASET}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG="log_baseline.txt"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file ${CONFIG} not found!"
    exit 1
fi

mkdir -p "$WORKDIR"

{
    echo ""
    echo "========================================"
    echo "Baseline Training - $(date)"
    echo "Dataset: ${DATASET}"
    echo "Config:  ${CONFIG}"
    echo "Workdir: ${WORKDIR}"
    echo "========================================"
} >> "$LOG"

python baseline/train_baseline.py --config "$CONFIG" --workdir "$WORKDIR" "${@:2}" 2>&1 | tee -a "$LOG"

echo "[$(date)] Baseline training finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

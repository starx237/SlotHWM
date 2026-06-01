#!/usr/bin/env bash
# SlotPi 端到端训练脚本（Ubuntu/Linux）
# Usage: bash scripts/run_train.sh <dataset> [extra_args]
# Example: bash scripts/run_train.sh obj3d
# Log: log.txt (自动追加时间戳)

set -euo pipefail

DATASET="${1:-obj3d}"
CONFIG="config/${DATASET}.yaml"
WORKDIR="experiments/${DATASET}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG="log.txt"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file ${CONFIG} not found!"
    exit 1
fi

mkdir -p "$WORKDIR"

{
    echo ""
    echo "========================================"
    echo "SlotPi Training - $(date)"
    echo "Dataset: ${DATASET}"
    echo "Config:  ${CONFIG}"
    echo "Workdir: ${WORKDIR}"
    echo "========================================"
} >> "$LOG"

python scripts/train.py --config "$CONFIG" --workdir "$WORKDIR" "${@:2}" 2>&1 | tee -a "$LOG"

echo "[$(date)] Training finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

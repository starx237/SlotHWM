#!/usr/bin/env bash
# SlotPi E2E + JEPA 训练脚本
# Usage: bash scripts/run_e2etrain.sh <dataset> [extra_args]
# Example: bash scripts/run_e2etrain.sh clevrer

set -euo pipefail

DATASET="${1:-clevrer}"
CONFIG="config/e2e_${DATASET}.yaml"
WORKDIR="experiments/e2e_${DATASET}"
LOG="log.txt"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file ${CONFIG} not found!"
    exit 1
fi

mkdir -p "$WORKDIR"

{
    echo ""
    echo "========================================"
    echo "SlotPi E2E+JEPA - $(date)"
    echo "Dataset: ${DATASET}"
    echo "Config:  ${CONFIG}"
    echo "Workdir: ${WORKDIR}"
    echo "========================================"
} >> "$LOG"

python scripts/train.py --config "$CONFIG" --workdir "$WORKDIR" "${@:2}" 2>&1 | tee -a "$LOG"

echo "[$(date)] E2E+JEPA finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

#!/usr/bin/env bash
# SlotPi 预训练脚本（仅 STATM-SAVi，无 rollout）
# Usage: bash scripts/run_pretrain.sh <dataset> [extra_args]
# Example: bash scripts/run_pretrain.sh clevrer

set -euo pipefail

DATASET="${1:-clevrer}"
CONFIG="config/pretrain_${DATASET}.yaml"
WORKDIR="pretrained/${DATASET}"
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
    echo "SlotPi Pretrain - $(date)"
    echo "Dataset: ${DATASET}"
    echo "Config:  ${CONFIG}"
    echo "Workdir: ${WORKDIR}"
    echo "========================================"
} >> "$LOG"

python scripts/train.py --config "$CONFIG" --workdir "$WORKDIR" "${@:2}" 2>&1 | tee -a "$LOG"

echo "[$(date)] Pretrain finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

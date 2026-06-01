#!/usr/bin/env bash
# =============================================
# SlotPi Stage 1 Training Script (Ubuntu/Linux)
# Usage: bash run_stage1_train.sh <dataset> [extra_args]
# Example: bash run_stage1_train.sh clevrer
# =============================================

exec > >(tee -a log.txt) 2>&1

set -euo pipefail

DATASET="${1:-obj3d}"
CONFIG="config/stage1/${DATASET}.yaml"
WORKDIR="experiments/stage1/${DATASET}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file ${CONFIG} not found!"
    exit 1
fi

mkdir -p "$WORKDIR"

python scripts/train_stage1.py --config "$CONFIG" --workdir "$WORKDIR" "${@:2}"

echo "Stage 1 training complete. Output saved to ${WORKDIR}"

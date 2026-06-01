#!/usr/bin/env bash
# =============================================
# SlotPi Evaluation Script (Ubuntu/Linux)
# Usage: bash run_evaluate.sh <stage> <dataset> <checkpoint>
# Example: bash run_evaluate.sh 2 obj3d experiments/stage2/obj3d/checkpoints/best.pth
# =============================================

exec > >(tee -a log.txt) 2>&1

set -euo pipefail

STAGE="${1:-2}"
DATASET="${2:-obj3d}"
CHECKPOINT="${3:-experiments/stage${STAGE}/${DATASET}/checkpoints/best.pth}"
CONFIG="config/stage${STAGE}/${DATASET}.yaml"
WORKDIR="experiments/eval/stage${STAGE}/${DATASET}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file ${CONFIG} not found!"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint ${CHECKPOINT} not found!"
    exit 1
fi

mkdir -p "$WORKDIR"

python scripts/evaluate.py --config "$CONFIG" --stage "$STAGE" --checkpoint "$CHECKPOINT" --workdir "$WORKDIR" "${@:4}"

echo "Evaluation complete. Results saved to ${WORKDIR}"

#!/usr/bin/env bash
# C 交换可解释性测试运行脚本
# Usage: bash scripts/run_interpret_swap_c.sh <checkpoint_path>
# Example: bash scripts/run_interpret_swap_c.sh experiments/obj3d/checkpoints/best.pt

set -euo pipefail

CKPT="${1:-experiments/obj3d/checkpoints/best.pt}"
CONFIG="config/interpret_obj3d.yaml"
WORKDIR="experiments/interpret_obj3d"
NUM_SAMPLES="${2:-10}"

if [ ! -f "$CKPT" ]; then
    echo "Error: Checkpoint not found: ${CKPT}"
    exit 1
fi

mkdir -p "$WORKDIR"

echo "========================================"
echo "C-Swap Interpretability Test"
echo "Config:  ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Samples: ${NUM_SAMPLES}"
echo "Workdir: ${WORKDIR}"
echo "========================================"

python scripts/interpret_swap_c.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --workdir "$WORKDIR" \
    --num_samples "$NUM_SAMPLES"

echo "[$(date)] Done."

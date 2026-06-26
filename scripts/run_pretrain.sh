#!/usr/bin/env bash
# SlotPi 预训练脚本（5 阶段）
# Usage: bash scripts/run_pretrain.sh <dataset> <phase> [extra_args]
#   phase: 1=单帧ISA, 2=ISA+DepthSpread, 3=GRU2全量
# Example: bash scripts/run_pretrain.sh obj3d 2

set -euo pipefail

DATASET="${1:-obj3d}"
PHASE="${2:-1}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG="log.txt"

case "$PHASE" in
    1)
        CONFIG="config/pretrain_phase1.yaml"
        WORKDIR="experiments/phase1_isa_single"
        ;;
    2)
        CONFIG="config/pretrain_phase2.yaml"
        WORKDIR="experiments/phase2_depth_spread"
        ;;
    3)
        CONFIG="config/pretrain_phase3.yaml"
        WORKDIR="experiments/phase3_gru2_full"
        ;;
    *)
        echo "Error: Unknown phase '${PHASE}'. Use 1, 2, or 3."
        echo "  1 = 单帧 ISA 预训练"
        echo "  2 = ISA + Depth→Spread Predictor"
        echo "  3 = GRU2 全量预测 (app,pos,depth)"
        exit 1
        ;;
esac

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
    echo "Phase:   ${PHASE}"
    echo "Config:  ${CONFIG}"
    echo "Workdir: ${WORKDIR}"
    echo "========================================"
} >> "$LOG"

python scripts/train.py --config "$CONFIG" --workdir "$WORKDIR" "${@:3}" 2>&1 | tee -a "$LOG"

echo "[$(date)] Pretrain phase ${PHASE} finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

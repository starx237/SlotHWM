#!/usr/bin/env bash
# SlotPi 数据预处理脚本
# Usage:
#   bash scripts/run_preprocess.sh obj3d        # OBJ3D 全部
#   bash scripts/run_preprocess.sh obj3d 500    # OBJ3D 500 个
#   bash scripts/run_preprocess.sh clevrer      # CLEVRER 从解压目录
#   bash scripts/run_preprocess.sh clevrer 1000 # CLEVRER 1000 个

set -euo pipefail

DATASET="${1:-obj3d}"
NUM_VIDEOS="${2:-}"
LOG="log.txt"

{
    echo ""
    echo "========================================"
    echo "SlotPi Preprocess - $(date)"
    echo "Dataset: ${DATASET}"
    echo "Num videos: ${NUM_VIDEOS:-all}"
    echo "========================================"
} >> "$LOG"

if [ "$DATASET" = "clevrer" ]; then
    if [ -n "$NUM_VIDEOS" ]; then
        python scripts/preprocess_clevrer.py --num_videos "$NUM_VIDEOS" 2>&1 | tee -a "$LOG"
    else
        python scripts/preprocess_clevrer.py 2>&1 | tee -a "$LOG"
    fi
else
    if [ -n "$NUM_VIDEOS" ]; then
        python scripts/preprocess_obj3d.py --num_videos "$NUM_VIDEOS" 2>&1 | tee -a "$LOG"
    else
        python scripts/preprocess_obj3d.py 2>&1 | tee -a "$LOG"
    fi
fi

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Preprocess finished. Exit code: ${EXIT_CODE}" >> "$LOG"
exit ${EXIT_CODE}

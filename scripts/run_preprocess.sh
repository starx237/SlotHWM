#!/usr/bin/env bash
# SlotPi 数据下载 + 预处理脚本
# 支持断点续传（自动跳过已处理的视频）
# Usage:
#   bash scripts/run_preprocess.sh                  # 处理全部
#   bash scripts/run_preprocess.sh 500              # 处理 500 个视频
#   bash scripts/run_preprocess.sh 500 no-download  # 处理本地已有 TFRecord

set -euo pipefail

NUM_VIDEOS="${1:-}"
DOWNLOAD="${2:-download}"
LOG="log.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

{
    echo ""
    echo "========================================"
    echo "SlotPi Preprocess - $(date)"
    echo "Num videos: ${NUM_VIDEOS:-all}"
    echo "Mode: ${DOWNLOAD}"
    echo "========================================"
} >> "$LOG"

if [ "$DOWNLOAD" = "download" ]; then
    echo "[$(date)] Downloading from GCS and preprocessing..." >> "$LOG"
    if [ -n "$NUM_VIDEOS" ]; then
        gsutil cat gs://multi-object-datasets/objects_room/objects_room_train.tfrecords | \
            python scripts/preprocess_obj3d.py --stdin --num_videos "$NUM_VIDEOS" 2>&1 | tee -a "$LOG"
    else
        gsutil cat gs://multi-object-datasets/objects_room/objects_room_train.tfrecords | \
            python scripts/preprocess_obj3d.py --stdin 2>&1 | tee -a "$LOG"
    fi
else
    echo "[$(date)] Processing local TFRecord..." >> "$LOG"
    if [ -n "$NUM_VIDEOS" ]; then
        python scripts/preprocess_obj3d.py --num_videos "$NUM_VIDEOS" 2>&1 | tee -a "$LOG"
    else
        python scripts/preprocess_obj3d.py 2>&1 | tee -a "$LOG"
    fi
fi

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Preprocess finished. Exit code: ${EXIT_CODE}" >> "$LOG"
exit ${EXIT_CODE}

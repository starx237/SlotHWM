#!/usr/bin/env bash
set -euo pipefail

LOG="/autodl-fs/data/SlotHWM/log.txt"
cd /autodl-fs/data/SlotHWM

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "SlotPi Streaming Preprocess - $(date)" >> "$LOG"
echo "Target: 500 videos" >> "$LOG"
echo "========================================" >> "$LOG"

gsutil cat gs://multi-object-datasets/objects_room/objects_room_train.tfrecords 2>/dev/null | \
python scripts/preprocess_obj3d.py --stdin --num_videos 500 >> "$LOG" 2>&1

echo "[$(date)] Preprocess finished." >> "$LOG"

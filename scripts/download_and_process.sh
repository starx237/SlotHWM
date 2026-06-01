#!/usr/bin/env bash
set -euo pipefail

LOG="/autodl-fs/data/SlotHWM/log.txt"
URL="https://storage.googleapis.com/multi-object-datasets/objects_room/objects_room_train.tfrecords"
OUTDIR="/autodl-fs/data/SlotHWM/downloads/obj3d"
OUTFILE="$OUTDIR/objects_room_train_full.tfrecords"

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "Step 1: Download (aria2c, 4 connections) - $(date)" >> "$LOG"
echo "========================================" >> "$LOG"

mkdir -p "$OUTDIR"
aria2c -x 4 -s 4 -k 1M "$URL" -d "$OUTDIR" -o objects_room_train_full.tfrecords \
  --console-log-level=error >> "$LOG" 2>&1

echo "[$(date)] Download complete. Size: $(stat --printf=%s "$OUTFILE" | numfmt --to=iec)" >> "$LOG"

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "Step 2: Preprocess - $(date)" >> "$LOG"
echo "========================================" >> "$LOG"

cd /autodl-fs/data/SlotHWM
rm -f data/obj3d/obj3d_data.pt
zcat "$OUTFILE" | python scripts/preprocess_obj3d.py --stdin --num_videos 500 >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "Step 3: Cleanup - $(date)" >> "$LOG"
echo "========================================" >> "$LOG"
rm -f "$OUTFILE"
echo "Downloaded file removed." >> "$LOG"
echo "ALL DONE. obj3d_data.pt ready for GPU training." >> "$LOG"

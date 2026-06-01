#!/usr/bin/env bash
# SlotPi Smoke Test 脚本（Ubuntu/Linux）
# 验证完整的前向+反向+参数更新流程
# Log: log.txt（自动追加时间戳）

set -euo pipefail

LOG="log.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

{
    echo ""
    echo "========================================"
    echo "SlotPi Smoke Test - $(date)"
    echo "========================================"
} >> "$LOG"

python scripts/smoke_test.py 2>&1 | tee -a "$LOG"

echo "[$(date)] Smoke test finished. Exit code: ${PIPESTATUS[0]}" >> "$LOG"

#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watch_s1q35.log"
prev=""
while true; do
  info="$(ssh -o ConnectTimeout=20 -o BatchMode=yes decision-gpu "tail -1 /root/s1q35.log 2>/dev/null; echo ---; grep -c STAGE1Q35_DONE /root/s1q35.log 2>/dev/null" 2>/dev/null)"
  last="$(echo "$info" | sed -n '1p')"; done_count="$(echo "$info" | sed -n '3p')"
  if [ "${done_count:-0}" -gt 0 ] 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] stage1 qwen3.5 training done: $last" | tee -a "$LOG"
    rsync -az decision-gpu:/root/decision-model/runs/stage1_qwen35/ "$ROOT/runs_remote/stage1_qwen35/" >>"$LOG" 2>&1
    osascript -e 'display notification "Stage1 Qwen3.5 训练完成，结果已回传" with title "decision-model" sound name "Glass"' 2>/dev/null
    break
  fi
  if [ "$last" != "$prev" ]; then echo "[$(date '+%H:%M:%S')] $last" | tee -a "$LOG"; prev="$last"; fi
  sleep 300
done

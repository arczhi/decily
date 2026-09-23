#!/bin/bash
# Watch Stage 2c (normk + bigk) and build the 5-way comparison on completion.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-decision-gpu}"
REMOTE_DIR="${REMOTE_DIR:-/root/decision-model}"
REMOTE_LOG="${REMOTE_LOG:-/root/s2c.log}"
DONE_MARK="${DONE_MARK:-S2C_DONE}"
RUNS="cx_qwen_s2c_normk cx_qwen_s2c_bigk"
INTERVAL="${INTERVAL:-300}"

LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/runs_remote"
LOG="$LOG_DIR/watch_s2c.log"
PIDFILE="$LOG_DIR/watch_s2c.pid"
mkdir -p "$LOG_DIR" "$OUT_DIR"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() {
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null \
    || log "notify: $1 - $2"
}

fetch_and_report() {
  for r in $RUNS; do
    log "fetching $r"
    rsync -az "$HOST:$REMOTE_DIR/runs/$r/" "$OUT_DIR/$r/" >>"$LOG" 2>&1
  done
  SUMMARY="$OUT_DIR/s2c_comparison.txt"
  {
    echo "=== Stage 2c comparison ($(date '+%F %T')) ==="
    echo
    /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
      "$OUT_DIR/baseline_a0/eval_report.json" \
      "$OUT_DIR/cx_qwen_lora/eval_report.json" \
      "$OUT_DIR/cx_qwen_s2b/eval_report_fixed.json" \
      "$OUT_DIR/cx_qwen_s2c_normk/eval_report.json" \
      "$OUT_DIR/cx_qwen_s2c_bigk/eval_report.json" \
      --labels "A(4t)" "B(4t)" "B(24t)" "B_normk" "B_bigk" 2>&1
    echo
    echo "--- per-task (held-out) ---"
    /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
      "$OUT_DIR/cx_qwen_s2b/eval_report_fixed.json" \
      "$OUT_DIR/cx_qwen_s2c_normk/eval_report.json" \
      "$OUT_DIR/cx_qwen_s2c_bigk/eval_report.json" \
      --labels "B(24t)" "B_normk" "B_bigk" --per-task 2>&1
  } > "$SUMMARY" 2>&1
  log "summary: $SUMMARY"
  tail -25 "$SUMMARY" | tee -a "$LOG"
  notify "decision-model: Stage 2c done" "五方对比: runs_remote/s2c_comparison.txt"
}

log "watcher started (pid $$) watching $REMOTE_LOG"
prev=""
stale=0
while true; do
  info="$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" \
    "tail -1 '$REMOTE_LOG' 2>/dev/null; echo '---'; grep -c '$DONE_MARK' '$REMOTE_LOG' 2>/dev/null" 2>/dev/null)"
  last="$(echo "$info" | sed -n '1p')"
  done_count="$(echo "$info" | sed -n '3p')"
  done_count="${done_count:-0}"

  if [ -z "$info" ]; then
    log "WARN ssh check failed (will retry)"
  elif [ "$done_count" -gt 0 ] 2>/dev/null; then
    log "finished: $last"
    fetch_and_report
    log "watcher exiting"
    rm -f "$PIDFILE"
    exit 0
  else
    log "progress: $last"
    if [ "$last" = "$prev" ]; then stale=$((stale + 1)); else stale=0; fi
    prev="$last"
    if [ "$stale" -ge 3 ]; then
      log "WARN no movement for $((stale * INTERVAL / 60)) min"
      notify "decision-model: training可能卡住" "日志 $((stale * INTERVAL / 60)) 分钟无变化"
      stale=0
    fi
  fi
  sleep "$INTERVAL"
done

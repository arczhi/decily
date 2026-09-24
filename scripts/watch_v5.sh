#!/bin/bash
# Watch Stage 2c (normk + bigk) and build the 5-way comparison on completion.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-decision-gpu}"
REMOTE_DIR="${REMOTE_DIR:-/root/decision-model}"
REMOTE_LOG="${REMOTE_LOG:-/root/v5.log}"
DONE_MARK="${DONE_MARK:-RLCD5_DONE}"
RUNS="rlcd_v5_distill"
INTERVAL="${INTERVAL:-300}"

LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/runs_remote"
LOG="$LOG_DIR/watch_v5.log"
PIDFILE="$LOG_DIR/watch_v5.pid"
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
  SUMMARY="$OUT_DIR/v5_comparison.txt"
  {
    echo "=== Stage 2c comparison ($(date '+%F %T')) ==="
    echo
    /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
      "$OUT_DIR/cx_qwen17b_bigk/eval_report.json" \
      "$OUT_DIR/rlcd_17b/eval_report.json" \
      "$OUT_DIR/rlcd_v2_17b/eval_report.json" \
      "$OUT_DIR/rlcd_v5_distill/eval_report.json" \
      --labels "SFT" "v1" "v2" "v5_distill" 2>&1
    echo
    echo "--- per-task (held-out) ---"
    /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
      "$OUT_DIR/cx_qwen17b_bigk/eval_report.json" \
      "$OUT_DIR/rlcd_17b/eval_report.json" \
      --labels "SFT_1.7B" "RLCD_1.7B" --per-task 2>&1
    echo
    echo "--- belief + per-task T (RLCD v4) ---"
    /usr/bin/python3 -c "
import json
r = json.load(open('$OUT_DIR/rlcd_v4_distill/eval_report.json'))
print('belief_eval:', json.dumps(r.get('belief_eval', {}), indent=1))
print('per_task_T:', json.dumps(r.get('per_task_temperature', {}), indent=1))
" 2>&1
  } > "$SUMMARY" 2>&1
  log "summary: $SUMMARY"
  tail -25 "$SUMMARY" | tee -a "$LOG"
  notify "decision-model: Stage 2c done" "五方对比: runs_remote/v5_comparison.txt"
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

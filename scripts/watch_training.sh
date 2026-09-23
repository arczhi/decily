#!/bin/bash
# Watch the remote Stage 2b training every 5 minutes.
# On completion: fetch artifacts, build the comparison summary, notify (macOS).
#
#   start:  nohup scripts/watch_training.sh >/dev/null 2>&1 &
#   stop:   kill "$(cat logs/watch_training.pid)"
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-decision-gpu}"
REMOTE_DIR="${REMOTE_DIR:-/root/decision-model}"
RUN_NAME="${RUN_NAME:-cx_qwen_s2b}"
REMOTE_LOG="${REMOTE_LOG:-/root/s2b.log}"
DONE_MARK="${DONE_MARK:-S2B_DONE}"
INTERVAL="${INTERVAL:-300}"

LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/runs_remote"
LOG="$LOG_DIR/watch_training.log"
PIDFILE="$LOG_DIR/watch_training.pid"
STATE="$LOG_DIR/watch_training.state"
mkdir -p "$LOG_DIR" "$OUT_DIR"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

notify() {
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null \
    || log "notify: $1 - $2"
}

fetch_and_report() {
  log "fetching artifacts -> $OUT_DIR/$RUN_NAME"
  rsync -az "$HOST:$REMOTE_DIR/runs/$RUN_NAME/" "$OUT_DIR/$RUN_NAME/" >>"$LOG" 2>&1
  for peer in baseline_a0 cx_qwen_lora; do
    [ -f "$OUT_DIR/$peer/eval_report.json" ] || \
      rsync -az "$HOST:$REMOTE_DIR/runs/$peer/eval_report.json" "$OUT_DIR/$peer/" >>"$LOG" 2>&1
  done

  SUMMARY="$OUT_DIR/${RUN_NAME}_comparison.txt"
  {
    echo "=== Stage 2b comparison ($(date '+%F %T')) ==="
    echo
    if [ -f "$OUT_DIR/$RUN_NAME/eval_report.json" ]; then
      /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
        "$OUT_DIR/baseline_a0/eval_report.json" \
        "$OUT_DIR/cx_qwen_lora/eval_report.json" \
        "$OUT_DIR/$RUN_NAME/eval_report.json" \
        --labels "A_qwen(4tasks)" "B_qwen(4tasks)" "B_qwen(24tasks)" 2>&1
    else
      echo "eval_report.json not found - eval may still be running"
    fi
  } > "$SUMMARY" 2>&1
  log "summary written: $SUMMARY"
  tail -20 "$SUMMARY" | tee -a "$LOG"
  notify "decision-model: Stage 2b done" "评测已下载，对比摘要: runs_remote/${RUN_NAME}_comparison.txt"
}

log "watcher started (pid $$, every ${INTERVAL}s, watching $REMOTE_LOG)"
notify "decision-model: watcher started" "每 5 分钟检查 Stage 2b 训练进度"

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
    log "training finished: $last"
    fetch_and_report
    log "watcher exiting (done)"
    rm -f "$PIDFILE"
    exit 0
  else
    log "progress: $last"
    if [ "$last" = "$prev" ]; then
      stale=$((stale + 1))
    else
      stale=0
    fi
    prev="$last"
    if [ "$stale" -ge 3 ]; then
      log "WARN no log movement for $((stale * INTERVAL / 60)) min"
      notify "decision-model: training可能卡住" "日志 $((stale * INTERVAL / 60)) 分钟无变化，请检查"
      stale=0
    fi
  fi
  sleep "$INTERVAL"
done

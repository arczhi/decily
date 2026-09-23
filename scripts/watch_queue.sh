#!/bin/bash
# Queue watcher: polls the ablation queue, fetches artifacts periodically
# (server shuts down at 03:16), and builds the final comparison on QUEUE_DONE.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-decision-gpu}"
REMOTE_DIR="${REMOTE_DIR:-/root/decision-model}"
REMOTE_LOG="${REMOTE_LOG:-/root/queue.log}"
DONE_MARK="${DONE_MARK:-QUEUE_DONE}"
RUNS="rlcd_v3b_noaction rlcd_v3c_abs35 rlcd_v3d_logonly"
ALL_RUNS="rlcd_v3_17b rlcd_v3b_noaction rlcd_v3c_abs35 rlcd_v3d_logonly"
INTERVAL="${INTERVAL:-300}"

LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/runs_remote"
LOG="$LOG_DIR/watch_queue.log"
PIDFILE="$LOG_DIR/watch_queue.pid"
mkdir -p "$LOG_DIR" "$OUT_DIR"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() {
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null \
    || log "notify: $1 - $2"
}

fetch_small() {
  for r in $ALL_RUNS; do
    mkdir -p "$OUT_DIR/$r"
    rsync -az --include 'eval_report.json' --include 'train_log.jsonl' \
      --include 'abstention_calibration.json' --exclude '*' \
      "$HOST:$REMOTE_DIR/runs/$r/" "$OUT_DIR/$r/" >>"$LOG" 2>&1
  done
}

fetch_full() {
  for r in $ALL_RUNS; do
    log "fetching full $r"
    rsync -az "$HOST:$REMOTE_DIR/runs/$r/" "$OUT_DIR/$r/" >>"$LOG" 2>&1
  done
}

final_report() {
  SUMMARY="$OUT_DIR/queue_comparison.txt"
  {
    echo "=== RLCD v3 ablation queue ($(date '+%F %T')) ==="
    echo
    /usr/bin/python3 "$ROOT/scripts/compare_reports.py" \
      "$OUT_DIR/cx_qwen17b_bigk/eval_report.json" \
      "$OUT_DIR/rlcd_v2_17b/eval_report.json" \
      "$OUT_DIR/rlcd_v3_17b/eval_report.json" \
      "$OUT_DIR/rlcd_v3b_noaction/eval_report.json" \
      "$OUT_DIR/rlcd_v3c_abs35/eval_report.json" \
      "$OUT_DIR/rlcd_v3d_logonly/eval_report.json" \
      --labels "SFT" "v2" "v3" "v3b_noaction" "v3c_abs35" "v3d_logonly" 2>&1
    echo
    echo "--- abstention (5% false budget) ---"
    /usr/bin/python3 -c "
import json, os
for r in ['rlcd_v2_17b','rlcd_v3_17b','rlcd_v3b_noaction','rlcd_v3c_abs35','rlcd_v3d_logonly']:
    p = os.path.join('$OUT_DIR', r, 'eval_report.json')
    if not os.path.exists(p): continue
    d = json.load(open(p)).get('abstention', {})
    ch = d.get('chosen_5pct_budget', {})
    print('%-18s bias=%s correct=%s false=%s' % (r, ch.get('bias'), ch.get('correct_rate'), ch.get('false_rate')))
" 2>&1
  } > "$SUMMARY" 2>&1
  log "summary: $SUMMARY"
  tail -30 "$SUMMARY" | tee -a "$LOG"
  notify "decision-model: RLCD v3 queue done" "对比: runs_remote/queue_comparison.txt"
}

log "queue watcher started (pid $$) watching $REMOTE_LOG"
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
    log "queue finished: $last"
    fetch_full
    final_report
    log "watcher exiting"
    rm -f "$PIDFILE"
    exit 0
  else
    log "progress: $last"
    fetch_small
    if [ "$last" = "$prev" ]; then stale=$((stale + 1)); else stale=0; fi
    prev="$last"
    if [ "$stale" -ge 4 ]; then
      log "WARN no movement for $((stale * INTERVAL / 60)) min"
      stale=0
    fi
  fi
  sleep "$INTERVAL"
done

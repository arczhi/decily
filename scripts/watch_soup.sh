#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-decision-gpu}"
OUT_DIR="$ROOT/runs_remote/soup"
LOG="$ROOT/logs/watch_soup.log"
PIDFILE="$ROOT/logs/watch_soup.pid"
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"
echo $$ > "$PIDFILE"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() { osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null || log "notify: $1 - $2"; }
log "soup watcher started (pid $$)"
prev=""
while true; do
  info="$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "tail -1 /root/soup.log 2>/dev/null; echo ---; grep -c SOUP_DONE /root/soup.log 2>/dev/null" 2>/dev/null)"
  last="$(echo "$info" | sed -n '1p')"
  done_count="$(echo "$info" | sed -n '3p')"
  if [ "${done_count:-0}" -gt 0 ] 2>/dev/null; then
    log "soup finished: $last"
    rsync -az "$HOST:/root/decision-model/runs/soup/" "$OUT_DIR/" >>"$LOG" 2>&1
    log "fetched -> $OUT_DIR"
    tail -30 "$OUT_DIR/report.json" 2>/dev/null >>"$LOG"
    notify "decision-model: model soup done" "report: runs_remote/soup/report.json"
    break
  fi
  if [ "$last" != "$prev" ]; then log "progress: $last"; prev="$last"; fi
  sleep 300
done
rm -f "$PIDFILE"

#!/bin/bash
# Sync local CODE to the GPU box. Only known code dirs are pushed; remote-only
# artifacts (data, runs, exports, models) are never touched.
set -euo pipefail
HOST="${HOST:-decision-gpu}"
DEST="${DEST:-/root/decision-model}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "$SRC/src/" "$HOST:$DEST/src/"
rsync -az --delete \
  --exclude '__pycache__' \
  "$SRC/scripts/" "$HOST:$DEST/scripts/"
rsync -az "$SRC/configs/" "$HOST:$DEST/configs/"
for f in pyproject.toml AGENTS.md DESIGN.md README.md; do
  [ -f "$SRC/$f" ] && rsync -az "$SRC/$f" "$HOST:$DEST/"
done
echo "[sync] code -> $HOST:$DEST (data/runs/models untouched)"

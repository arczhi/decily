#!/bin/bash
# Sync local code to the GPU box (data/runs stay remote-only).
set -euo pipefail
HOST="${HOST:-decision-gpu}"
DEST="${DEST:-/root/decision-model}"
rsync -az --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'runs' \
  --exclude 'data/raw' \
  --exclude 'data/distill*' \
  --exclude 'data/merged*' \
  --exclude 'data/eval' \
  --exclude '*.pt' \
  --exclude '.git' \
  "$(cd "$(dirname "$0")/.." && pwd)/" "$HOST:$DEST/"
echo "[sync] -> $HOST:$DEST"

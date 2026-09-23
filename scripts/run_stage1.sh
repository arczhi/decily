#!/bin/bash
# Stage 1: train Route B and Route A baseline sequentially, then evaluate.
set -euo pipefail
cd /root/decision-model
export PYTHONPATH=/root/decision-model/src
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
PY=/root/venvs/decision/bin/python

echo "=== Route B training (cross-encoder) ==="
$PY -m decision_model.train.stage1 --config configs/cx_b0.yaml

echo "=== Route A training ==="
$PY -m decision_model.train.stage1 --config configs/baseline_a0.yaml

echo "=== eval Route B ==="
$PY scripts/eval_ckpt.py --config configs/cx_b0.yaml --ckpt runs/cx_b0/ckpt_last.pt \
    --eval data/eval/stage1_intask.jsonl data/eval/stage1_heldout.jsonl \
    --fit-temperature --per-task --out runs/cx_b0/eval_report.json

echo "=== eval Route A ==="
$PY scripts/eval_ckpt.py --config configs/baseline_a0.yaml --ckpt runs/baseline_a0/ckpt_last.pt \
    --eval data/eval/stage1_intask.jsonl data/eval/stage1_heldout.jsonl \
    --fit-temperature --per-task --out runs/baseline_a0/eval_report.json

echo "=== STAGE1 DONE ==="

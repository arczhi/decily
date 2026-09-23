#!/bin/bash
# Stage 1 data: 4 training tasks + 4 in-task eval sets + 2 held-out tasks.
set -euo pipefail
cd /root/decision-model
export PYTHONPATH=/root/decision-model/src
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
PY=/root/venvs/decision/bin/python

echo "=== train shards ==="
$PY scripts/convert_data.py --dataset sst2 --split train --limit 6000 --out data/raw/sst2_train.jsonl
$PY scripts/convert_data.py --dataset ag_news --split train --limit 8000 --out data/raw/ag_news_train.jsonl
$PY scripts/convert_data.py --dataset mnli --split train --limit 12000 --out data/raw/mnli_train.jsonl
$PY scripts/convert_data.py --dataset arc_easy --split train --limit 2500 --out data/raw/arc_easy_train.jsonl

echo "=== in-task eval ==="
$PY scripts/convert_data.py --dataset sst2 --split validation --limit 300 --out data/raw/sst2_val.jsonl
$PY scripts/convert_data.py --dataset ag_news --split test --limit 300 --out data/raw/ag_news_val.jsonl
$PY scripts/convert_data.py --dataset mnli --split validation --limit 300 --out data/raw/mnli_val.jsonl
$PY scripts/convert_data.py --dataset arc_easy --split validation --limit 300 --out data/raw/arc_easy_val.jsonl

echo "=== held-out tasks (never trained) ==="
$PY scripts/convert_data.py --dataset massive --split validation --limit 400 --out data/raw/massive_val.jsonl
$PY scripts/convert_data.py --dataset banking77 --split test --limit 400 --out data/raw/banking77_val.jsonl
$PY scripts/convert_data.py --dataset emotion --split validation --limit 400 --out data/raw/emotion_val.jsonl

echo "=== mixed eval sets ==="
$PY scripts/build_eval.py --spec sst2=300,ag_news=300,mnli=300,arc_easy=300 --split val --out data/eval/stage1_intask.jsonl
$PY scripts/build_eval.py --spec massive=400,banking77=400,emotion=400 --split val --out data/eval/stage1_heldout.jsonl
echo "=== DATA OK ==="

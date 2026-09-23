#!/bin/bash
# Stage 2 data: 24 training tasks + expanded in-task eval; held-out stays untouched.
# NOTE: massive_scenario is deliberately NOT trained (same utterances as the
#       held-out massive-intent task -> would leak).
set -euo pipefail
cd /root/decision-model
export PYTHONPATH=/root/decision-model/src
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
PY=/root/venvs/decision/bin/python

echo "=== new train shards ==="
$PY scripts/convert_data.py --dataset amazon_polarity --split train --limit 4000 --out data/raw/amazon_polarity_train.jsonl
$PY scripts/convert_data.py --dataset imdb --split train --limit 4000 --out data/raw/imdb_train.jsonl
$PY scripts/convert_data.py --dataset tweet_sentiment --split train --limit 4000 --out data/raw/tweet_sentiment_train.jsonl
$PY scripts/convert_data.py --dataset toxic_conversations --split train --limit 4000 --out data/raw/toxic_conversations_train.jsonl
$PY scripts/convert_data.py --dataset counterfactual --split train --limit 3000 --out data/raw/counterfactual_train.jsonl
$PY scripts/convert_data.py --dataset tweet_topic --split train --limit 3000 --out data/raw/tweet_topic_train.jsonl
$PY scripts/convert_data.py --dataset poem_sentiment --split train --limit 1000 --out data/raw/poem_sentiment_train.jsonl
$PY scripts/convert_data.py --dataset dbpedia --split train --limit 4000 --out data/raw/dbpedia_train.jsonl
$PY scripts/convert_data.py --dataset rte --split train --limit 2500 --out data/raw/rte_train.jsonl
$PY scripts/convert_data.py --dataset qqp --split train --limit 4000 --out data/raw/qqp_train.jsonl
$PY scripts/convert_data.py --dataset mrpc --split train --limit 3000 --out data/raw/mrpc_train.jsonl
$PY scripts/convert_data.py --dataset arc_challenge --split train --limit 1200 --out data/raw/arc_challenge_train.jsonl

for s in high_school_world_history high_school_biology high_school_physics \
         high_school_government_and_politics college_computer_science \
         moral_scenarios professional_law world_religions; do
  $PY scripts/convert_data.py --dataset mmlu_$s --split test --limit 400 --out data/raw/mmlu_${s}_train.jsonl
done

echo "=== in-task eval shards ==="
$PY scripts/convert_data.py --dataset amazon_polarity --split test --limit 250 --out data/raw/amazon_polarity_val.jsonl
$PY scripts/convert_data.py --dataset imdb --split test --limit 250 --out data/raw/imdb_val.jsonl
$PY scripts/convert_data.py --dataset tweet_sentiment --split test --limit 250 --out data/raw/tweet_sentiment_val.jsonl
$PY scripts/convert_data.py --dataset toxic_conversations --split test --limit 250 --out data/raw/toxic_conversations_val.jsonl
$PY scripts/convert_data.py --dataset counterfactual --split test --limit 250 --out data/raw/counterfactual_val.jsonl
$PY scripts/convert_data.py --dataset tweet_topic --split test --limit 250 --out data/raw/tweet_topic_val.jsonl
$PY scripts/convert_data.py --dataset poem_sentiment --split test --limit 100 --out data/raw/poem_sentiment_val.jsonl
$PY scripts/convert_data.py --dataset dbpedia --split test --limit 250 --out data/raw/dbpedia_val.jsonl
$PY scripts/convert_data.py --dataset rte --split validation --limit 250 --out data/raw/rte_val.jsonl
$PY scripts/convert_data.py --dataset qqp --split validation --limit 250 --out data/raw/qqp_val.jsonl
$PY scripts/convert_data.py --dataset mrpc --split validation --limit 250 --out data/raw/mrpc_val.jsonl
$PY scripts/convert_data.py --dataset arc_challenge --split validation --limit 250 --out data/raw/arc_challenge_val.jsonl
for s in high_school_world_history high_school_biology high_school_physics \
         high_school_government_and_politics college_computer_science \
         moral_scenarios professional_law world_religions; do
  $PY scripts/convert_data.py --dataset mmlu_$s --split test --limit 100 --out data/raw/mmlu_${s}_val.jsonl
done

echo "=== mixed eval sets ==="
SPEC="sst2=250,ag_news=250,mnli=250,arc_easy=250,amazon_polarity=250,imdb=250,tweet_sentiment=250,toxic_conversations=250,counterfactual=250,tweet_topic=250,poem_sentiment=100,dbpedia=250,rte=250,qqp=250,mrpc=250,arc_challenge=250"
for s in high_school_world_history high_school_biology high_school_physics \
         high_school_government_and_politics college_computer_science \
         moral_scenarios professional_law world_religions; do
  SPEC="$SPEC,mmlu_$s=100"
done
$PY scripts/build_eval.py --spec "$SPEC" --split val --out data/eval/stage2_intask.jsonl
echo "=== STAGE2 DATA OK ==="

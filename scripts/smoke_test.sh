#!/bin/bash
# End-to-end smoke test on the GPU box.
set -euo pipefail
cd /root/decision-model
export PYTHONPATH=/root/decision-model/src
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
PY=/root/venvs/decision/bin/python

echo "=== 0. env ==="
$PY -c "import torch, transformers, datasets; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), '| transformers', transformers.__version__, '| datasets', datasets.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 1. unit tests ==="
$PY -m pytest tests -q

echo "=== 2. convert small slices ==="
$PY scripts/convert_data.py --dataset sst2 --split train --limit 400 --out data/raw/sst2_train.jsonl
$PY scripts/convert_data.py --dataset ag_news --split train --limit 400 --out data/raw/ag_news_train.jsonl
$PY scripts/convert_data.py --dataset sst2 --split validation --limit 100 --out data/raw/sst2_val.jsonl
$PY scripts/convert_data.py --dataset ag_news --split test --limit 100 --out data/raw/ag_news_val.jsonl
$PY scripts/build_eval.py --spec sst2=100,ag_news=100 --split val --out data/eval/smoke_eval.jsonl

echo "=== 3. tiny Route B training ==="
$PY -m decision_model.train.stage1 --config configs/smoke.yaml

echo "=== 4. temperature fit on smoke eval ==="
$PY - << 'EOF'
import yaml
from transformers import AutoTokenizer
from decision_model.data.schema import read_jsonl
from decision_model.data.collate import RouteBCollator
from decision_model.models.decision_model import RouteBDecisionModel, RouteBConfig
from decision_model.eval.harness import evaluate, fit_temperature

cfg = yaml.safe_load(open("configs/smoke.yaml"))
model_cfg = {k: v for k, v in cfg["model"].items() if k != "route"}
m = RouteBDecisionModel(RouteBConfig(**model_cfg))
ckpt = __import__("torch").load("runs/smoke_b/ckpt_last.pt", map_location="cpu")
missing = m.load_state_dict(ckpt["state"], strict=False)
print("missing:", [k for k in missing.missing_keys if "encoder" not in k])
m.to("cuda").eval()
tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
coll = RouteBCollator(tok, **cfg["data"]["collator"])
ex = list(read_jsonl("data/eval/smoke_eval.jsonl"))
print("T=1.0 :", evaluate(m, ex, coll, temperature=1.0, batch_size=32, device="cuda"))
t, nll = fit_temperature(m, ex, coll, batch_size=32, device="cuda")
print(f"best T={t:.2f} nll={nll:.4f}")
EOF

echo "=== SMOKE OK ==="

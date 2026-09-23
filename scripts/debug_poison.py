"""Scan the sst2 pool for poisoned examples: per-example loss with a fresh model."""

from __future__ import annotations

import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, "/root/decision-model/src")

import yaml
from transformers import AutoTokenizer

from decision_model.data.collate import RouteBCollator
from decision_model.data.schema import read_jsonl
from decision_model.models.decision_model import (
    RouteBConfig,
    RouteBDecisionModel,
)

CFG = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
TORCH_SEED = 0
torch.manual_seed(TORCH_SEED)
model_cfg = {k: v for k, v in CFG["model"].items() if k != "route"}
model = RouteBDecisionModel(RouteBConfig(**model_cfg)).cuda().eval()
tok = AutoTokenizer.from_pretrained(CFG["model"]["backbone"])
coll = RouteBCollator(tok, **CFG["data"]["collator"])

examples = list(read_jsonl("/root/decision-model/data/raw/sst2_train.jsonl"))
print(f"pool size: {len(examples)}")

worst = []
with torch.no_grad():
    for i in range(0, len(examples), 64):
        chunk = examples[i : i + 64]
        batch = coll(chunk)
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if isinstance(v, torch.Tensor):
                setattr(batch, f, v.cuda())
        out = model(batch)
        logits = out["logits"][:, 0].float()
        gold = batch.answer_index[:, 0]
        losses = F.cross_entropy(logits, gold, reduction="none").cpu()
        for j, l in enumerate(losses.tolist()):
            worst.append((l, i + j, chunk[j]))

worst.sort(key=lambda x: -x[0])
print("=== top 10 worst ===")
for l, idx, ex in worst[:10]:
    print(f"loss={l:.4f} idx={idx} answer={ex.questions[0].answer_ids} "
          f"len={len(ex.state)} :: {ex.state[:120]!r}")

vals = torch.tensor([w[0] for w in worst])
print(
    "loss stats: min=%.4f p50=%.4f p99=%.4f max=%.4f mean=%.4f"
    % (
        vals.min(),
        vals.median(),
        vals.quantile(0.99),
        vals.max(),
        vals.mean(),
    )
)

# min-side: suspiciously easy examples (label leakage?)
print("=== top 3 lowest loss (label leakage check) ===")
for l, idx, ex in worst[-3:]:
    print(f"loss={l:.4f} idx={idx} answer={ex.questions[0].answer_ids} :: {ex.state[:120]!r}")

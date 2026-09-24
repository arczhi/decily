"""Controlled experiment: why does the Trainer not learn while a bare loop does?

Runs the same fixed sst2 batch through several loop variants with fresh models.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/decision-model/src")

import yaml
from transformers import AutoTokenizer

from decision_model.data.collate import RouteBCollator
from decision_model.data.schema import read_jsonl
from decision_model.models.decision_model import (
    RouteBConfig,
    RouteBDecisionModel,
    decision_loss,
)


def make_model(cfg: dict, device: str):
    model_cfg = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    m = RouteBDecisionModel(RouteBConfig(**model_cfg)).to(device)
    return m


def train_variant(
    name: str,
    batch,
    device: str,
    trainer_semantics: bool,
    steps: int = 80,
    lr: float = 1e-3,
):
    torch.manual_seed(0)
    m = make_model(CFG, device)
    batch = to_device(batch, device)
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    if not trainer_semantics:
        m.train()
    losses = []
    for i in range(1, steps + 1):
        if trainer_semantics:
            m.train()
        out = m(batch)
        loss = decision_loss(out["logits"], batch, 0.05)["total"]
        loss.backward()
        if trainer_semantics:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            opt.zero_grad()
        losses.append(float(loss))
        if i % 20 == 0:
            with torch.no_grad():
                logits = out["logits"][0, 0]
                print(
                    f"[{name}] step {i}: loss={losses[-1]:.4f} "
                    f"logits={[round(float(x), 3) for x in logits]}",
                    flush=True,
                )
    return losses


def to_device(batch, device):
    for field in batch.__dataclass_fields__:
        v = getattr(batch, field)
        if isinstance(v, torch.Tensor):
            setattr(batch, field, v.to(device))
    return batch


CFG = yaml.safe_load(open("/root/decision-model/configs/mini_b0.yaml"))
TOK = AutoTokenizer.from_pretrained(CFG["model"]["backbone"])
COLL = RouteBCollator(TOK, **CFG["data"]["collator"])
EXS = list(read_jsonl("/root/decision-model/data/raw/sst2_train.jsonl"))[:16]

print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
fixed_batch = COLL(EXS)

for name, device, trainer_sem in [
    ("A cuda custom", "cuda", False),
    ("B cuda trainer", "cuda", True),
    ("C cpu custom", "cpu", False),
    ("D cpu trainer", "cpu", True),
]:
    train_variant(name, fixed_batch, device, trainer_sem)

print("EXPERIMENT DONE")

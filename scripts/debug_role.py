"""Test: role embedding drift as the collapse cause.

Runs the real Trainer (sampled batches) with:
  E1: role embedding frozen at init
  E2: no role embeddings at all
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/decision-model/src")

import yaml

from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build


def run(name: str, cfg: dict, mutate_model=None, steps: int = 150):
    print(f"########## {name} ##########", flush=True)
    model, collator, sampler, eval_examples, probe_examples = build(cfg)
    if mutate_model:
        mutate_model(model)
    train_cfg = TrainConfig(**cfg["train"])
    train_cfg.steps = steps
    train_cfg.eval_every = 0
    train_cfg.log_every = 10
    train_cfg.out_dir = f"/tmp/role_{name}"
    Trainer(
        model, collator, sampler, eval_examples,
        eval_collator=collator, cfg=train_cfg, probe_examples=probe_examples,
    ).run()


CFG = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))


def freeze_role(model):
    model.encoder.role_emb.weight.requires_grad_(False)
    print("role embedding frozen; trainable:",
          sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")


CFG2 = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
CFG2["model"]["add_role_embeddings"] = False

run("frozen_role", CFG, freeze_role)
run("no_role", CFG2)

print("ROLE EXPERIMENT DONE")

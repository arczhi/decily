"""Compare scorer fixes against the uniform-collapse failure.

Each variant: real Trainer, sampled sst2 batches, 200 steps.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/decision-model/src")

import yaml

from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build


def run(name: str, cfg: dict, mutate_model=None, steps: int = 200):
    model, collator, sampler, eval_examples, probe_examples = build(cfg)
    if mutate_model:
        mutate_model(model)
    train_cfg = TrainConfig(**cfg["train"])
    train_cfg.steps = steps
    train_cfg.eval_every = 0
    train_cfg.log_every = 50
    train_cfg.out_dir = f"/tmp/fix_{name}"
    Trainer(
        model, collator, sampler, eval_examples,
        eval_collator=collator, cfg=train_cfg, probe_examples=probe_examples,
    ).run()
    import json

    lines = [json.loads(l) for l in open(f"/tmp/fix_{name}/train_log.jsonl")]
    last = lines[-1]
    print(
        f"RESULT {name}: step={last['step']} loss={last['total']:.4f} "
        f"probe={last.get('probe_loss', float('nan')):.4f} grad={last['grad_norm']:.3f}",
        flush=True,
    )


BASE = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
BASE["train"]["log_every"] = 50


def cfg_with(**model_overrides):
    c = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
    c["train"]["log_every"] = 50
    c["model"].update(model_overrides)
    return c


run("baseline", cfg_with())
run("l2", cfg_with(l2_normalize=True))
run("center", cfg_with(center_pooled=True))
run("l2_center", cfg_with(l2_normalize=True, center_pooled=True))


def freeze_blocks(model):
    for layer in model.scorer.layers:
        for p in layer.parameters():
            p.requires_grad_(False)


run("frozen_blocks", cfg_with(), freeze_blocks)
print("FIX EXPERIMENT DONE")

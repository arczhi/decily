"""Bisect: real Trainer code path, fixed batch vs sampled batch."""

from __future__ import annotations

import sys
import random

import torch

sys.path.insert(0, "/root/decision-model/src")

import yaml

from decision_model.data.schema import read_jsonl
from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build


class FixedSampler:
    """Mimics MixtureSampler.sample but always yields the same examples."""

    def __init__(self, examples):
        self.examples = examples

    def sample(self, n, start_seed=None):
        for i in range(n):
            yield self.examples[i % len(self.examples)]


def run(name: str, cfg: dict, sampler, steps: int = 150):
    print(f"########## {name} ##########", flush=True)
    model, collator, _, eval_examples, probe_examples = build(cfg)
    train_cfg = TrainConfig(**cfg["train"])
    train_cfg.steps = steps
    train_cfg.eval_every = 0
    train_cfg.out_dir = f"/tmp/bisect_{name}"
    trainer = Trainer(
        model,
        collator,
        sampler,
        eval_examples,
        eval_collator=collator,
        cfg=train_cfg,
        probe_examples=probe_examples,
    )
    trainer.run()


CFG = yaml.safe_load(open("/root/decision-model/configs/diag_b1.yaml"))
CFG["train"]["log_every"] = 10
EXS = list(read_jsonl("/root/decision-model/data/raw/sst2_train.jsonl"))[:16]

with open("/root/decision-model/configs/diag_b1.yaml") as f:
    _ = yaml.safe_load(f)

model, collator, real_sampler, eval_examples, probe_examples = build(CFG)

run("fixed", CFG, FixedSampler(EXS))

# fresh sampler instance for the second run (pools are read fresh inside build)
model, collator, real_sampler, eval_examples, probe_examples = build(CFG)
run("sampled", CFG, real_sampler)

print("BISECT DONE")

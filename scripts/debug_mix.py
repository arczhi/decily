"""Diagnose 4-task mixture failure: is it task interference or mixed-K batches?"""

from __future__ import annotations

import copy
import json
import sys

sys.path.insert(0, "/root/decision-model/src")

import yaml

from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build

BASE = yaml.safe_load(open("/root/decision-model/configs/diag_b4.yaml"))
BASE["model"]["center_pooled"] = True
BASE["train"]["log_every"] = 50
BASE["train"]["eval_every"] = 0


def manifest(*tasks):
    paths = {
        "sst2": "data/raw/sst2_train.jsonl",
        "ag_news": "data/raw/ag_news_train.jsonl",
        "mnli": "data/raw/mnli_train.jsonl",
        "arc_easy": "data/raw/arc_easy_train.jsonl",
    }
    return [{"task": t, "path": paths[t], "weight": 1.0} for t in tasks]


def run(name: str, tasks: tuple[str, ...], steps: int = 300, train_over: dict | None = None):
    cfg = copy.deepcopy(BASE)
    cfg["data"]["manifest"] = manifest(*tasks)
    cfg["train"].update(train_over or {})
    model, collator, sampler, eval_examples, probe_examples = build(cfg)
    tc = TrainConfig(**cfg["train"])
    tc.steps = steps
    tc.out_dir = f"/tmp/mix_{name}"
    Trainer(
        model, collator, sampler, eval_examples,
        eval_collator=collator, cfg=tc, probe_examples=probe_examples,
    ).run()
    lines = [json.loads(l) for l in open(f"/tmp/mix_{name}/train_log.jsonl")]
    probes = [(l["step"], round(l.get("probe_loss", float("nan")), 4)) for l in lines if "probe_loss" in l]
    last = lines[-1]
    print(f"RESULT {name}: loss={last['total']:.4f} probes={probes[-3:]}", flush=True)


run("sst2", ("sst2",))
run("sst2+agnews", ("sst2", "ag_news"))
run("sst2+mnli", ("sst2", "mnli"))
run("sst2+arc", ("sst2", "arc_easy"))
run("all4", ("sst2", "ag_news", "mnli", "arc_easy"))
run("all4_lr3e4", ("sst2", "ag_news", "mnli", "arc_easy"), train_over={"lr": 3e-4})
print("MIX EXPERIMENT DONE")

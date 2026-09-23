"""Decisive test: is the bottleneck task conditioning?

Train late-interaction scorer on the 4-task mixture but hand it an explicit
per-task id through task_type_emb (needs n_task_types >= 4). If this learns,
conditioning on the (weak) question text is the bottleneck.
"""

from __future__ import annotations

import copy
import json
import sys

sys.path.insert(0, "/root/decision-model/src")

import torch
import yaml

from decision_model.data.collate import RouteBCollator, RouteBBatch
from decision_model.data.mixture import MixtureItem, MixtureSampler
from decision_model.data.schema import read_jsonl
from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build

BASE = yaml.safe_load(open("/root/decision-model/configs/diag_b4.yaml"))
BASE["model"]["center_pooled"] = True
BASE["model"]["n_task_types"] = 8
BASE["train"]["log_every"] = 50
BASE["train"]["eval_every"] = 0

TASK_ID = {"sst2": 0, "ag_news": 1, "mnli": 2, "arc_easy": 3}


class TaggedCollator(RouteBCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        batch.task_type = torch.tensor([TASK_ID[ex.task] for ex in examples], dtype=torch.long)
        return batch


def run(name, steps=300, tagged=True):
    cfg = copy.deepcopy(BASE)
    model, _, sampler, eval_examples, probe_examples = build(cfg)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    coll = TaggedCollator(tok, **cfg["data"]["collator"]) if tagged else RouteBCollator(tok, **cfg["data"]["collator"])
    tc = TrainConfig(**cfg["train"])
    tc.steps = steps
    tc.out_dir = f"/tmp/tag_{name}"
    Trainer(
        model, coll, sampler, eval_examples,
        eval_collator=coll, cfg=tc, probe_examples=probe_examples,
    ).run()
    lines = [json.loads(l) for l in open(f"/tmp/tag_{name}/train_log.jsonl")]
    probes = [(l["step"], round(l.get("probe_loss", float("nan")), 4)) for l in lines if "probe_loss" in l]
    print(f"RESULT {name}: loss={lines[-1]['total']:.4f} probes={probes[-3:]}", flush=True)


run("tagged_4task", tagged=True)
run("untagged_4task", tagged=False)
print("TAG DONE")

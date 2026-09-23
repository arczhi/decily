"""Isolate mixed-batch padding vs true task interference.

Variants:
  A: per-task batches (round robin), sst2+ag_news
  B: per-task batches (round robin), all 4 tasks
  C: mixed batches, sst2 only but every example padded to K=4 (dummy slots)
"""

from __future__ import annotations

import copy
import json
import random
import sys

sys.path.insert(0, "/root/decision-model/src")

import yaml

from decision_model.data.schema import DecisionExample, read_jsonl
from decision_model.train.common import TrainConfig, Trainer
from decision_model.train.stage1 import build

BASE = yaml.safe_load(open("/root/decision-model/configs/diag_b4.yaml"))
BASE["model"]["center_pooled"] = True
BASE["train"]["log_every"] = 50
BASE["train"]["eval_every"] = 0

PATHS = {
    "sst2": "data/raw/sst2_train.jsonl",
    "ag_news": "data/raw/ag_news_train.jsonl",
    "mnli": "data/raw/mnli_train.jsonl",
    "arc_easy": "data/raw/arc_easy_train.jsonl",
}


class PerTaskSampler:
    """Round-robin: each batch comes from a single task."""

    def __init__(self, tasks, batch_size, seed=0):
        self.pools = {t: list(read_jsonl(PATHS[t])) for t in tasks}
        self.tasks = list(tasks)
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.cursor = 0

    def sample(self, n, start_seed=None):
        for _ in range(n):
            task = self.tasks[self.cursor % len(self.tasks)]
            if self.rng.random() < 1.0 / self.batch_size:
                self.cursor += 1
            pool = self.pools[task]
            yield pool[self.rng.randrange(len(pool))]


def strip_extras(ex: DecisionExample, extra_options: int) -> DecisionExample:
    return ex


def make_padded_sst2():
    from decision_model.data.schema import Option, Question

    out = []
    for ex in list(read_jsonl(PATHS["sst2"])):
        q = ex.questions[0]
        extra = [
            Option(id="max_positive", text="very positive"),
            Option(id="very_negative", text="somewhat negative"),
        ]
        q2 = Question(text=q.text, options=q.options + extra, answer_ids=q.answer_ids, type=q.type)
        out.append(DecisionExample(task="sst2", state=ex.state, questions=[q2], meta=ex.meta))
    return out


class ListSampler:
    def __init__(self, examples, seed=0):
        self.examples = examples
        self.rng = random.Random(seed)

    def sample(self, n, start_seed=None):
        for _ in range(n):
            yield self.examples[self.rng.randrange(len(self.examples))]


def run(name, cfg, sampler, steps=300):
    model, collator, _, eval_examples, probe_examples = build(cfg)
    tc = TrainConfig(**cfg["train"])
    tc.steps = steps
    tc.out_dir = f"/tmp/bmix_{name}"
    Trainer(
        model, collator, sampler, eval_examples,
        eval_collator=collator, cfg=tc, probe_examples=probe_examples,
    ).run()
    lines = [json.loads(l) for l in open(f"/tmp/bmix_{name}/train_log.jsonl")]
    probes = [(l["step"], round(l.get("probe_loss", float("nan")), 4)) for l in lines if "probe_loss" in l]
    print(f"RESULT {name}: loss={lines[-1]['total']:.4f} probes={probes[-3:]}", flush=True)


def cfg_for(tasks):
    c = copy.deepcopy(BASE)
    c["data"]["manifest"] = [{"task": t, "path": PATHS[t], "weight": 1.0} for t in tasks]
    return c


run("A_pertask_2task", cfg_for(("sst2", "ag_news")), PerTaskSampler(("sst2", "ag_news"), 16))
run("B_pertask_4task", cfg_for(("sst2", "ag_news", "mnli", "arc_easy")),
    PerTaskSampler(("sst2", "ag_news", "mnli", "arc_easy"), 16))
run("C_sst2_padded_k4", cfg_for(("sst2",)), ListSampler(make_padded_sst2()))
print("BMIX DONE")

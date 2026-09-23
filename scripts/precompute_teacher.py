"""Precompute ensemble teacher distributions for distillation (RLCD v4).

For every example in the training shards, run the K teacher checkpoints,
average their softmax distributions (over each example's options) and write a
new shard carrying Question.probs = teacher distribution. The student then
trains with the proper-score (log+spherical) term against these targets -
i.e. ensemble distillation inside the RLCD objective, matching the Laya/Jev
recipe of training on teacher-generated reference probabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.schema import DecisionExample, read_jsonl, write_jsonl  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def load_model(cfg: dict, ckpt: str):
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "route"}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
    state = torch.load(ckpt, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    model.load_state_dict(state, strict=False)
    return model.to("cuda").eval()


@torch.no_grad()
def ensemble_probs(models, collator, examples, batch_size=16, device="cuda"):
    """Average teacher distributions per example (aligned with its options)."""
    out = [None] * len(examples)
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collator(chunk)
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if isinstance(v, torch.Tensor):
                setattr(batch, f, v.to(device))
        valid = batch.option_valid[:, 0].cpu()
        sums = [None] * len(chunk)
        for model in models:
            logits = model(batch)["logits"][:, 0].float().cpu()
            for i in range(len(chunk)):
                p = torch.softmax(logits[i][valid[i]], dim=-1).numpy()
                sums[i] = p if sums[i] is None else sums[i] + p
        for i in range(len(chunk)):
            out[start + i] = sums[i] / len(models)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/distill")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **cfg["data"]["collator"])

    models = [load_model(cfg, c) for c in args.ckpts]
    print(f"[teacher] {len(models)} teachers loaded", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for item in cfg["data"]["manifest"]:
        task, path = item["task"], item["path"]
        examples = list(read_jsonl(path))
        probs = ensemble_probs(models, coll, examples, args.batch_size)
        labeled = []
        for ex, p in zip(examples, probs):
            q = ex.questions[0]
            assert len(p) == len(q.options), (task, len(p), len(q.options))
            labeled.append(
                DecisionExample(
                    task=ex.task,
                    state=ex.state,
                    questions=[
                        type(q)(
                            text=q.text,
                            options=q.options,
                            answer_ids=q.answer_ids,
                            type=q.type,
                            probs=[float(x) for x in p],
                        )
                    ],
                    meta=ex.meta,
                )
            )
        out_path = os.path.join(args.out_dir, f"{task}_train.jsonl")
        n = write_jsonl(out_path, iter(labeled))
        print(f"[teacher] {task}: {n} -> {out_path}", flush=True)

    print("[teacher] done", flush=True)


if __name__ == "__main__":
    main()

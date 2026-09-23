"""Benchmark two-stage inference vs single-pass on a 60/77-way task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from decision_model.data.schema import Option, read_jsonl  # noqa: E402
from decision_model.infer.two_stage import _to_device, score_candidates  # noqa: E402


def build(cfg_path: str, ckpt_path: str):
    import yaml
    from transformers import AutoTokenizer

    from decision_model.data.collate import RouteBCollator
    from decision_model.models.cross_encoder import (
        CrossEncoderConfig,
        CrossEncoderDecisionModel,
    )

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "route"}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state"] if "state" in ckpt else ckpt
    missing = model.load_state_dict(state, strict=False)
    hard = [k for k in missing.missing_keys if "encoder.model" not in k]
    if hard:
        print("WARNING missing:", hard[:5])
    model.to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **cfg["data"]["collator"])
    return model, coll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", default="data/eval/stage1_heldout.jsonl")
    ap.add_argument("--tasks", default="massive,banking77")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    model, coll = build(args.config, args.ckpt)
    tasks = set(args.tasks.split(","))
    examples = [e for e in read_jsonl(args.eval) if e.task in tasks][: args.n]
    print(f"examples: {len(examples)}")

    results = {
        "full_softmax": {"correct": [], "conf": [], "time": 0.0},
        f"top{args.top_k}_two_stage": {"correct": [], "conf": [], "time": 0.0},
    }

    for ex in examples:
        q = ex.questions[0]
        gold = q.answer_ids[0]
        options: list[Option] = q.options

        t0 = time.time()
        with torch.no_grad():
            batch = _to_device(coll([ex]), "cuda")
            logits = model(batch)["logits"][0, 0].float().cpu()
        probs_full = torch.softmax(logits, dim=-1)
        results["full_softmax"]["time"] += time.time() - t0
        pred_idx = int(probs_full.argmax())
        results["full_softmax"]["correct"].append(int(options[pred_idx].id == gold))
        results["full_softmax"]["conf"].append(float(probs_full.max()))

        t0 = time.time()
        lg = score_candidates(
            model, coll, ex.state, q.text, options, args.chunk, "cuda"
        )
        results[f"top{args.top_k}_two_stage"]["time"] += time.time() - t0
        k = min(args.top_k, len(options))
        top = torch.topk(lg, k)
        probs_top = torch.softmax(top.values, dim=-1)
        gold_idx = next(i for i, o in enumerate(options) if o.id == gold)
        hit = (top.indices == gold_idx).nonzero()
        if len(hit) > 0:
            pos = int(hit[0][0])
            correct = int(float(probs_top[pos]) == float(probs_top.max()))
            results.setdefault("gold_in_topk", []).append(1)
        else:
            correct = 0
            results.setdefault("gold_in_topk", []).append(0)
        results[f"top{args.top_k}_two_stage"]["correct"].append(correct)
        results[f"top{args.top_k}_two_stage"]["conf"].append(float(probs_top.max()))

    out = {}
    for name, r in results.items():
        if name == "gold_in_topk":
            out[name] = {"rate": float(np.mean(r))}
            continue
        acc = float(np.mean(r["correct"]))
        out[name] = {
            "acc": acc,
            "mean_conf": float(np.mean(r["conf"])),
            "total_sec": round(r["time"], 2),
            "per_example_ms": round(1000 * r["time"] / max(len(examples), 1), 1),
        }
    print(json.dumps(out, indent=2))
    print(f"candidates per question: {len(examples[0].questions[0].options)}")


if __name__ == "__main__":
    main()

"""Calibrate the none-option bias for a trained checkpoint and report the curve.

Usage:
    python scripts/calibrate_abstention.py --config configs/rlcd_v2_17b.yaml \
        --ckpt runs/rlcd_v2_17b/ckpt_last.pt --out runs/rlcd_v2_17b/abstention_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.data.transforms import add_abstention  # noqa: E402
from decision_model.infer.abstention import abstention_curve, collect_none_logits  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", default="data/eval/stage1_heldout.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--target-false", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "route"}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state"] if "state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to("cuda").eval()

    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **cfg["data"]["collator"])

    examples = list(read_jsonl(args.heldout))[: args.n]
    rng = random.Random(0)
    present = [add_abstention(ex, rng, 1.0, mode="keep_gold", shuffle=False) for ex in examples]
    absent = [add_abstention(ex, rng, 1.0, mode="drop_gold", shuffle=False) for ex in examples]

    present_logits = collect_none_logits(model, coll, present)
    absent_logits = collect_none_logits(model, coll, absent)
    curve = abstention_curve(present_logits, absent_logits)
    chosen = curve.pick(target_false=args.target_false)

    report = {
        "n": len(examples),
        "target_false_rate": args.target_false,
        "chosen": chosen,
        "curve": {
            "bias": curve.biases,
            "correct": curve.correct_rates,
            "false": curve.false_rates,
        },
    }
    print(json.dumps({k: report[k] for k in ("n", "target_false_rate", "chosen")}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

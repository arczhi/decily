"""Evaluate a trained checkpoint on one or more eval JSONL files.

Usage:
    python scripts/eval_ckpt.py --config configs/mini_b0.yaml \
        --ckpt runs/mini_b0/ckpt_last.pt \
        --eval data/eval/stage1_intask.jsonl data/eval/stage1_heldout.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteACollator, RouteBCollator  # noqa: E402
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.eval.harness import (  # noqa: E402
    collect_logits,
    fit_temperature_rows,
    metrics_by_task,
    metrics_from_rows,
)


def build_model(cfg: dict, ckpt_path: str | None):
    from decision_model.models.decision_model import RouteBConfig, RouteBDecisionModel
    from decision_model.models.route_a import RouteAConfig, RouteADecisionModel

    model_cfg = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    route = cfg["model"].get("route", "b")
    if route == "b":
        model = RouteBDecisionModel(RouteBConfig(**model_cfg))
        collator_cls = RouteBCollator
    elif route == "cx":
        from decision_model.models.cross_encoder import (
            CrossEncoderConfig,
            CrossEncoderDecisionModel,
        )

        model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
        collator_cls = RouteBCollator
    else:
        model = RouteADecisionModel(RouteAConfig(**model_cfg))
        collator_cls = RouteACollator

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
        incompatible = model.load_state_dict(state, strict=False)
        hard_missing = [
            k
            for k in incompatible.missing_keys
            if not k.startswith(("encoder.model.", "model."))
        ]
        if hard_missing:
            print(f"[eval] WARNING missing keys: {hard_missing[:5]}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = collator_cls(tok, **cfg["data"].get("collator", {}))
    return model, coll, route


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fit-temperature", action="store_true")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--per-task", action="store_true")
    ap.add_argument("--belief", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, coll, route = build_model(cfg, args.ckpt)
    model.to(args.device).eval()

    report = {}
    for path in args.eval:
        examples = list(read_jsonl(path))
        rows = collect_logits(
            model, examples, coll, batch_size=args.batch_size, device=args.device
        )
        t_fit, _ = fit_temperature_rows(rows)
        row = {
            "n": float(len(rows)),
            "fitted_T": t_fit,
            "T=1.0": metrics_from_rows(rows, 1.0),
            "fitted": metrics_from_rows(rows, t_fit),
        }
        if args.per_task:
            row["per_task"] = metrics_by_task(rows, temperature=1.0)
        if args.belief and cfg["data"].get("belief_eval_path"):
            from decision_model.eval.belief import evaluate_belief

            row["belief_eval"] = evaluate_belief(
                model, coll, list(read_jsonl(cfg["data"]["belief_eval_path"])),
                batch_size=args.batch_size, device=args.device,
            )
        report[path] = row
        print(f"\n=== {path} (route {route}) ===")
        print(json.dumps(row, indent=2, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()

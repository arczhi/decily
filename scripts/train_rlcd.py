"""RLCD driver: load an SFT checkpoint, run the RLCD trainer, final evals.

Usage:
    python scripts/train_rlcd.py --config configs/rlcd_17b.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.mixture import MixtureItem, MixtureSampler  # noqa: E402
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.eval.belief import evaluate_belief  # noqa: E402
from decision_model.eval.harness import collect_logits, fit_temperature_rows, metrics_from_rows  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)
from decision_model.train.rlcd import RLCDConfig, RLCDTrainer  # noqa: E402


def build_model(model_cfg: dict, ckpt_path: str):
    cfg = {k: v for k, v in model_cfg.items() if k != "route"}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**cfg))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    hard = [k for k in incompatible.missing_keys if "encoder.model" not in k]
    if hard:
        raise RuntimeError(f"checkpoint mismatch, missing: {hard[:5]}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rlcd_kwargs = dict(cfg["rlcd"])
    sft_ckpt = rlcd_kwargs.pop("sft_ckpt")
    rcfg = RLCDConfig(**rlcd_kwargs)

    model = build_model(cfg["model"], sft_ckpt)
    reference = build_model(cfg["model"], sft_ckpt)
    print("[rlcd] loaded SFT checkpoint:", sft_ckpt, flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    collator = RouteBCollator(tok, **cfg["data"].get("collator", {}))

    items = [MixtureItem(**it) for it in cfg["data"]["manifest"]]
    sampler = MixtureSampler(
        items,
        seed=cfg.get("seed", 0),
        min_options=cfg["data"].get("min_options", 2),
        max_options=cfg["data"].get("max_options", 16),
    )
    belief_examples = list(read_jsonl(cfg["data"]["belief_path"]))
    belief_eval_examples = list(read_jsonl(cfg["data"]["belief_eval_path"]))
    eval_examples = list(read_jsonl(cfg["data"]["eval_path"]))
    print(
        f"[rlcd] belief train={len(belief_examples)} belief eval={len(belief_eval_examples)} "
        f"decision eval={len(eval_examples)}",
        flush=True,
    )

    trainer = RLCDTrainer(
        model, reference, collator, sampler,
        belief_examples, eval_examples, belief_eval_examples, rcfg,
    )
    trainer.run()

    # final: standard eval with temperature fit + belief eval
    report = {}
    for path in [cfg["data"]["eval_path"], cfg["data"]["heldout_path"]]:
        examples = list(read_jsonl(path))
        rows = collect_logits(
            model, examples, collator, batch_size=rcfg.eval_batch_size,
            device=rcfg.device, max_batches=rcfg.eval_max_batches,
        )
        t_fit, _ = fit_temperature_rows(rows)
        report[path] = {
            "n": float(len(rows)),
            "fitted_T": t_fit,
            "T=1.0": metrics_from_rows(rows, 1.0),
            "fitted": metrics_from_rows(rows, t_fit),
        }
    report["belief_eval"] = evaluate_belief(
        model, collator, belief_eval_examples,
        batch_size=rcfg.eval_batch_size, device=rcfg.device,
    )
    out = os.path.join(rcfg.out_dir, "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[rlcd] final report:", json.dumps(report, indent=2)[:2000], flush=True)
    print("[rlcd] wrote", out, flush=True)


if __name__ == "__main__":
    main()

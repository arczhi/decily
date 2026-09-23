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
from decision_model.data.transforms import add_abstention  # noqa: E402
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

    # practical open-set checks: abstention when gold is absent / false abstention
    import random as _random

    heldout_examples = list(read_jsonl(cfg["data"]["heldout_path"]))
    rng = _random.Random(0)
    absent = [add_abstention(ex, rng, 1.0, mode="drop_gold", shuffle=False) for ex in heldout_examples]
    present = [add_abstention(ex, rng, 1.0, mode="keep_gold", shuffle=False) for ex in heldout_examples]
    absent_rows = collect_logits(
        model, absent, collator, batch_size=rcfg.eval_batch_size,
        device=rcfg.device, max_batches=rcfg.eval_max_batches,
    )
    present_rows = collect_logits(
        model, present, collator, batch_size=rcfg.eval_batch_size,
        device=rcfg.device, max_batches=rcfg.eval_max_batches,
    )
    import numpy as _np

    def _abstain_rate(rows) -> float:
        hits = 0
        for lg in rows.logits:
            if int(_np.argmax(lg)) == len(lg) - 1:
                hits += 1
        return hits / max(len(rows), 1)

    report["abstention"] = {
        "gold_absent_correct_rate": float(metrics_from_rows(absent_rows, 1.0).get("acc_all", 0.0)),
        "gold_present_false_abstain_rate": _abstain_rate(present_rows),
        "gold_present_acc_with_none": float(metrics_from_rows(present_rows, 1.0).get("acc_all", 0.0)),
    }

    # per-task-family temperatures (practical calibration)
    from decision_model.eval.harness import fit_temperature_rows as _fit

    per_task_T = {}
    for task in sorted(set(rows.tasks)):
        idx = [i for i, t in enumerate(rows.tasks) if t == task]
        if len(idx) < 20:
            continue
        sub = type(rows)(
            [rows.logits[i] for i in idx], [rows.valid[i] for i in idx],
            [rows.targets[i] for i in idx], [rows.types[i] for i in idx],
            [rows.tasks[i] for i in idx],
        )
        t_task, _ = _fit(sub)
        per_task_T[task] = t_task
    report["per_task_temperature"] = per_task_T
    out = os.path.join(rcfg.out_dir, "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[rlcd] final report:", json.dumps(report, indent=2)[:2000], flush=True)
    print("[rlcd] wrote", out, flush=True)


if __name__ == "__main__":
    main()

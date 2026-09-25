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



def _collect_with_reject(model, collator, examples, batch_size=16, device="cuda", max_batches=None):
    """Candidate logits + reject logit appended as the last class (v3)."""
    import numpy as np

    rows = []
    chunk = []
    n_batches = 0
    with torch.no_grad():
        for ex in examples:
            chunk.append(ex)
            if len(chunk) >= batch_size:
                rows.extend(_flush_reject(model, collator, chunk, device))
                chunk = []
                n_batches += 1
                if max_batches is not None and n_batches >= max_batches:
                    break
        if chunk and (max_batches is None or n_batches < max_batches):
            rows.extend(_flush_reject(model, collator, chunk, device))
    return rows


def _flush_reject(model, collator, chunk, device):
    import numpy as np

    batch = collator(chunk)
    for f in batch.__dataclass_fields__:
        v = getattr(batch, f)
        if isinstance(v, torch.Tensor):
            setattr(batch, f, v.to(device))
    out = model(batch)
    logits = out["logits"][:, 0].float().cpu()
    reject = out["reject_logit"][:, 0].float().cpu()
    valid = batch.option_valid[:, 0].cpu()
    rows = []
    for i in range(len(chunk)):
        cand = logits[i][valid[i]].numpy()
        rows.append(np.concatenate([cand, reject[i].reshape(1).numpy()]))
    return rows


def build_model(model_cfg: dict, ckpt_path: str):
    cfg = {k: v for k, v in model_cfg.items() if k not in ("route", "init_ckpt")}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**cfg))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    allow_prefixes = ("encoder.model", "reject_head", "reject_pool")
    hard = [
        k for k in incompatible.missing_keys
        if not k.startswith(allow_prefixes)
    ]
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

    if sft_ckpt:
        model = build_model(cfg["model"], sft_ckpt)
    else:
        mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
        from decision_model.models.cross_encoder import CrossEncoderConfig as _CEC, CrossEncoderDecisionModel as _CEM
        model = _CEM(_CEC(**mc))
        print("[rlcd] training from base (no sft_ckpt)", flush=True)
    if rcfg.kl_weight > 0:
        reference = build_model(cfg["model"], sft_ckpt)
    else:
        reference = model
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
    distill_sampler = None
    if cfg["data"].get("distill_manifest") and rcfg.distill_ratio > 0:
        distill_items = [MixtureItem(**it) for it in cfg["data"]["distill_manifest"]]
        distill_sampler = MixtureSampler(
            distill_items,
            seed=cfg.get("seed", 0) + 1,
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
        distill_sampler=distill_sampler,
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

    # practical open-set checks via the reject head (v3 models only)
    import numpy as _np
    import random as _random

    from decision_model.data.transforms import drop_gold
    from decision_model.infer.abstention import abstention_curve

    if getattr(model, "reject_head", None) is None:
        report["abstention"] = {"note": "model has no rejector head"}
    else:
        heldout_examples = list(read_jsonl(cfg["data"]["heldout_path"]))
        absent = [drop_gold(ex) for ex in heldout_examples if not ex.questions[0].defer]
        present_rows = _collect_with_reject(
            model, collator, heldout_examples, batch_size=rcfg.eval_batch_size,
            device=rcfg.device, max_batches=rcfg.eval_max_batches,
        )
        absent_rows = _collect_with_reject(
            model, collator, absent, batch_size=rcfg.eval_batch_size,
            device=rcfg.device, max_batches=rcfg.eval_max_batches,
        )
        curve = abstention_curve(present_rows, absent_rows)
        chosen = curve.pick(target_false=0.05)
        report["abstention"] = {
            "chosen_5pct_budget": chosen,
            "false_abstain_rate_at_0": float(
                _np.mean([int(_np.argmax(lg)) == len(lg) - 1 for lg in present_rows])
            ),
            "correct_abstain_rate_at_0": float(
                _np.mean([int(_np.argmax(lg)) == len(lg) - 1 for lg in absent_rows])
            ),
            "curve": {
                "bias": curve.biases,
                "correct": curve.correct_rates,
                "false": curve.false_rates,
            },
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

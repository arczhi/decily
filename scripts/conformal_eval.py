"""Conformal selective-prediction evaluation for decision models.

Stores full probability distributions per example so temperature scaling and
probability averaging (ensembles) are exact:
    p(T) = softmax(log(p) / T)      (log p = logits - logsumexp, shift-invariant)

Reports coverage / selective error at several alpha targets on a held-out test
split, calibrated on a disjoint calibration split, plus risk-coverage curves.

Usage:
    python scripts/conformal_eval.py --config configs/rlcd_v3_17b.yaml \
        --ckpts runs/cx_qwen17b_bigk/ckpt_last.pt runs/rlcd_v3_17b/ckpt_last.pt \
        --labels SFT v3 --ensemble --out runs/conformal_eval.json
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
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.infer.conformal import risk_coverage_curve, threshold_report  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def load_model(cfg: dict, ckpt: str):
    model_cfg = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**model_cfg))
    state = torch.load(ckpt, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    model.load_state_dict(state, strict=False)
    return model.to("cuda").eval()


@torch.no_grad()
def collect_probs(model, collator, examples, batch_size=16, device="cuda"):
    """Returns (probs [n, K_i] at T=1, gold index per example)."""
    probs, golds = [], []
    chunk = []
    for ex in examples:
        chunk.append(ex)
        if len(chunk) >= batch_size:
            _flush(model, collator, chunk, device, probs, golds)
            chunk = []
    if chunk:
        _flush(model, collator, chunk, device, probs, golds)
    return probs, np.array(golds)


def _flush(model, collator, chunk, device, probs, golds):
    batch = collator(chunk)
    for f in batch.__dataclass_fields__:
        v = getattr(batch, f)
        if isinstance(v, torch.Tensor):
            setattr(batch, f, v.to(device))
    logits = model(batch)["logits"][:, 0].float().cpu()
    valid = batch.option_valid[:, 0].cpu()
    targets = batch.answer_mask[:, 0].cpu()
    for i in range(len(chunk)):
        z = logits[i][valid[i]]
        p = torch.softmax(z, dim=-1).numpy()
        gold = int(np.argmax(targets[i][valid[i]].numpy()))
        probs.append(p)
        golds.append(gold)


def probs_at(probs, T: float) -> list[np.ndarray]:
    if abs(T - 1.0) < 1e-9:
        return probs
    out = []
    for p in probs:
        z = np.log(np.clip(p, 1e-12, 1.0)) / T
        z -= z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


def stats(probs, golds):
    p_max = np.array([float(p.max()) for p in probs])
    p_gold = np.array([float(p[g]) for p, g in zip(probs, golds)])
    correct = np.array([float(int(p.argmax()) == g) for p, g in zip(probs, golds)])
    return p_max, p_gold, correct


def fit_T(probs, golds, grid=np.arange(0.2, 3.01, 0.05)) -> float:
    best_t, best = 1.0, float("inf")
    for t in grid:
        p_gold = np.array(
            [float(probs_at([p], t)[0][g]) for p, g in zip(probs, golds)]
        )
        nll = float(-np.log(np.clip(p_gold, 1e-12, 1.0)).mean())
        if nll < best:
            best_t, best = float(t), nll
    return best_t


def evaluate_probs(calib_probs, calib_golds, test_probs, test_golds) -> dict:
    t_fit = fit_T(calib_probs, calib_golds)
    cp, cg = probs_at(calib_probs, t_fit), calib_golds
    tp, tg = probs_at(test_probs, t_fit), test_golds
    pm_c, _, corr_c = stats(cp, cg)
    pm_t, _, corr_t = stats(tp, tg)
    return {
        "fitted_T": t_fit,
        "coverage_at_risk": threshold_report(pm_c, corr_c, pm_t, corr_t),
        "risk_coverage": risk_coverage_curve(pm_t, corr_t),
    }


def _avg_dists(dist_lists: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Average probability distributions across models (ragged K -> pad+renorm)."""
    n = len(dist_lists[0])
    out = []
    for i in range(n):
        width = max(len(d[i]) for d in dist_lists)
        acc = np.zeros(width)
        for d in dist_lists:
            p = d[i]
            acc[: len(p)] += p
        acc = acc / acc.sum()
        out.append(acc[: len(dist_lists[0][i])] * 1.0)
        out[-1] = out[-1] / out[-1].sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--eval", default="data/eval/stage1_heldout.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **cfg["data"]["collator"])

    examples = list(read_jsonl(args.eval))[: args.n]
    half = len(examples) // 2
    calib, test = examples[:half], examples[half:]
    labels = args.labels or [f"ckpt{i}" for i in range(len(args.ckpts))]

    report: dict = {"n_calib": len(calib), "n_test": len(test)}
    calib_probs_all, test_probs_all = [], []
    calib_golds = test_golds = None
    for ckpt, label in zip(args.ckpts, labels):
        model = load_model(cfg, ckpt)
        cp, calib_golds = collect_probs(model, coll, calib, batch_size=args.batch_size)
        tp, test_golds = collect_probs(model, coll, test, batch_size=args.batch_size)
        entry = evaluate_probs(cp, calib_golds, tp, test_golds)
        report[label] = entry
        calib_probs_all.append(cp)
        test_probs_all.append(tp)
        print(f"[{label}] T={entry['fitted_T']:.2f} " + json.dumps(entry["coverage_at_risk"]))
        del model
        torch.cuda.empty_cache()

    if args.ensemble and len(test_probs_all) > 1:
        ens_c = _avg_dists(calib_probs_all)
        ens_t = _avg_dists(test_probs_all)
        report["ensemble"] = evaluate_probs(ens_c, calib_golds, ens_t, test_golds)
        print("[ensemble] T=%.2f %s" % (report["ensemble"]["fitted_T"], json.dumps(report["ensemble"]["coverage_at_risk"])))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

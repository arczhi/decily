"""Distribution-space ensemble of decision models + singles comparison.

Loads N checkpoints in their native architectures (no weight merging), gets the
full probability distribution per example, averages them, and reports accuracy /
NLL / ECE at T=1 and at a fitted temperature, plus per-task breakdowns.

Usage:
    python scripts/ensemble_eval.py \
        --labels SFT v1 v2 fullFT \
        --configs configs/cx_qwen17b_bigk.yaml configs/cx_qwen17b_bigk.yaml \
                  configs/cx_qwen17b_bigk.yaml configs/fullft_from_sft.yaml \
        --ckpts runs/cx_qwen17b_bigk/ckpt_last.pt runs/rlcd_17b/ckpt_last.pt \
                runs/rlcd_v2_17b/ckpt_last.pt runs/fullft_sft17b/ckpt_step1500.pt \
        --evals data/eval/stage1_heldout.jsonl data/eval/stage2_intask.jsonl \
        --max-per-eval 1200 --out runs/ensemble_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from conformal_eval import collect_probs, fit_T, load_model, probs_at, _avg_dists  # noqa: E402
from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.eval.metrics import MetricsAccumulator  # noqa: E402


def metrics_from_probs(probs: list[np.ndarray], golds: np.ndarray, T: float) -> dict:
    acc = MetricsAccumulator()
    for p, g in zip(probs, golds):
        pp = probs_at([p], T)[0]
        y = np.zeros_like(pp)
        y[int(g)] = 1.0
        acc.add(pp[None, :], y[None, :], "choice", np.ones_like(y, dtype=bool)[None, :])
    return acc.compute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--max-per-eval", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.configs[0]) as f:
        ref_cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(ref_cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **ref_cfg["data"]["collator"])

    report: dict = {}
    for eval_path in args.evals:
        examples = list(read_jsonl(eval_path))[: args.max_per_eval]
        per_model: dict[str, list[np.ndarray]] = {}
        golds = None
        for label, cfgp, ckpt in zip(args.labels, args.configs, args.ckpts):
            model = load_model(yaml.safe_load(open(cfgp)), ckpt)
            probs, golds = collect_probs(
                model, coll, examples, batch_size=args.batch_size
            )
            per_model[label] = probs
            del model
            torch.cuda.empty_cache()

        entry: dict = {"n": len(examples), "singles": {}}
        for label in args.labels:
            p = per_model[label]
            t = fit_T(p, golds)
            entry["singles"][label] = {
                "fitted_T": t,
                "T=1.0": metrics_from_probs(p, golds, 1.0),
                "fitted": metrics_from_probs(p, golds, t),
            }
            m = entry["singles"][label]["fitted"]
            print(
                f"[{eval_path.split('/')[-1]}] {label:8s} acc={m['acc_all']:.4f} "
                f"nll={m['choice/nll']:.3f} ece={m['choice/ece']:.4f} T={t:.2f}",
                flush=True,
            )

        ens = _avg_dists([per_model[l] for l in args.labels])
        t_ens = fit_T(ens, golds)
        entry["ensemble"] = {
            "fitted_T": t_ens,
            "T=1.0": metrics_from_probs(ens, golds, 1.0),
            "fitted": metrics_from_probs(ens, golds, t_ens),
        }
        m = entry["ensemble"]["fitted"]
        print(
            f"[{eval_path.split('/')[-1]}] ENSEMBLE acc={m['acc_all']:.4f} "
            f"nll={m['choice/nll']:.3f} ece={m['choice/ece']:.4f} T={t_ens:.2f}",
            flush=True,
        )
        report[eval_path] = entry

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)
    print("ENSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()

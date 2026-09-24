"""Evaluate the released decider-2b on OUR evaluation sets (their prompt/API).

Their training list overlaps our held-out suite (banking77 / massive_intent /
emotion are trained by decider-2b), so those numbers are their in-task
performance; tasks outside their 95-task list are zero-shot for them.

Usage (on the GPU box):
    PYTHONPATH=/root/models/decider-2b python scripts/eval_decider.py \
        --model /root/models/decider-2b \
        --eval data/eval/stage1_heldout.jsonl data/eval/stage2_intask.jsonl \
        --out runs/eval_decider2b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from decision_model.data.schema import read_jsonl  # noqa: E402
from decision_model.eval.metrics import MetricsAccumulator  # noqa: E402


def fit_T(probs: list[np.ndarray], golds: list[int], grid=np.arange(0.2, 3.01, 0.05)) -> float:
    best_t, best = 1.0, float("inf")
    for t in grid:
        total = 0.0
        for p, g in zip(probs, golds):
            z = np.log(np.clip(p, 1e-12, 1.0)) / t
            z -= z.max()
            e = np.exp(z)
            q = e / e.sum()
            total += -float(np.log(max(q[g], 1e-12)))
        nll = total / max(len(probs), 1)
        if nll < best:
            best_t, best = float(t), nll
    return best_t


def probs_at(probs: list[np.ndarray], T: float) -> list[np.ndarray]:
    if abs(T - 1.0) < 1e-9:
        return probs
    out = []
    for p in probs:
        z = np.log(np.clip(p, 1e-12, 1.0)) / T
        z -= z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


def metrics(probs: list[np.ndarray], golds: list[int], T: float) -> dict:
    acc = MetricsAccumulator()
    for p, g in zip(probs_at(probs, T), golds):
        y = np.zeros_like(p)
        y[g] = 1.0
        acc.add(p[None, :], y[None, :], "choice", np.ones_like(y, dtype=bool)[None, :])
    return acc.compute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per-eval", type=int, default=1200)
    args = ap.parse_args()

    from decider.infer import Decider

    d = Decider(args.model, use_graphs=False)  # eager: avoids compile+capture per shape
    report: dict = {}
    for path in args.eval:
        examples = list(read_jsonl(path))[: args.max_per_eval]
        per_task: dict[str, list] = {}
        probs_all, golds_all = [], []
        t0 = time.time()
        for i, ex in enumerate(examples):
            q = ex.questions[0]
            options = [o.render() for o in q.options]
            try:
                out = d.decide(
                    ex.state,
                    [{"question": q.text or "Which option fits best?", "options": options}],
                )[0]
                p = np.array(out["probs_list"], dtype=np.float64)
                p = p / p.sum()
                gold = q.index_of(q.answer_ids[0])
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {ex.task} skipped: {e}", flush=True)
                continue
            probs_all.append(p)
            golds_all.append(gold)
            per_task.setdefault(ex.task, []).append((p, gold))
            if (i + 1) % 200 == 0:
                print(f"  {path.split('/')[-1]}: {i+1}/{len(examples)} ({time.time()-t0:.0f}s)", flush=True)

        t_fit = fit_T(probs_all, golds_all)
        entry = {
            "n": len(probs_all),
            "fitted_T": t_fit,
            "T=1.0": metrics(probs_all, golds_all, 1.0),
            "T=1.3 (theirs)": metrics(probs_all, golds_all, 1.3),
            "fitted": metrics(probs_all, golds_all, t_fit),
            "per_task": {
                task: {
                    "n": len(rows),
                    "acc": float(np.mean([np.argmax(p) == g for p, g in rows])),
                }
                for task, rows in sorted(per_task.items())
            },
        }
        report[path] = entry
        print(f"[{path}] n={entry['n']} fitted_T={t_fit:.2f} "
              f"acc={entry['fitted']['acc_all']:.4f} "
              f"nll={entry['fitted']['choice/nll']:.3f} "
              f"ece={entry['fitted']['choice/ece']:.4f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)
    print("EVAL_DECIDER_DONE", flush=True)


if __name__ == "__main__":
    main()

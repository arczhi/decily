"""Salience evaluation: ranking agreement between a student and the teacher.

Metrics on held-out articles:
  - top1/top3 hit: does the top-ranked sentence match the teacher's top pick
  - Spearman rank correlation of per-candidate scores (student vs teacher)
  - per-sentence sigmoid agreement (MAE)

Usage (remote):
    python scripts/eval_salience.py --student-config configs/student_salience.yaml \
        --student-ckpt runs/student_salience/ckpt_last.pt \
        --teacher-config configs/teacher_sal.yaml --teacher-ckpt runs/stage1_qwen35/ckpt_last.pt \
        --eval data/raw/salience_eval.jsonl --n 150
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
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)


def load(cfg_path, ckpt, device="cuda"):
    cfg = yaml.safe_load(open(cfg_path))
    mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    m = CrossEncoderDecisionModel(CrossEncoderConfig(**mc))
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["state"] if "state" in st else st, strict=False)
    return m.to(device).eval(), cfg


@torch.no_grad()
def scores(model, coll, examples, device="cuda", batch=8):
    out = []
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        b = coll(chunk)
        for f in b.__dataclass_fields__:
            v = getattr(b, f)
            if isinstance(v, torch.Tensor):
                setattr(b, f, v.to(device))
        lg = model(b)["logits"][:, 0].float().cpu()
        for j in range(len(chunk)):
            n = int(b.option_valid[j, 0].sum())
            out.append(lg[j, :n].numpy())
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-config", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--teacher-config", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    examples = list(read_jsonl(args.eval))[: args.n]
    sm, scfg = load(args.student_config, args.student_ckpt)
    tm, tcfg = load(args.teacher_config, args.teacher_ckpt)
    tok = AutoTokenizer.from_pretrained(scfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    scoll = RouteBCollator(tok, **scfg["data"]["collator"])
    tcoll = RouteBCollator(tok, **tcfg["data"]["collator"])

    s_list = scores(sm, scoll, examples)
    t_list = scores(tm, tcoll, examples)

    top1 = top3 = 0
    rhos, maes = [], []
    for s, t in zip(s_list, t_list):
        n = min(len(s), len(t))
        s, t = s[:n], t[:n]
        ps = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
        pt = np.exp(t - t.max()) / np.exp(t - t.max()).sum()
        top1 += int(np.argmax(ps) == np.argmax(pt))
        top3 += int(np.argmax(pt) in np.argsort(-ps)[:3])
        rhos.append(spearman(s, t))
        maes.append(float(np.abs(ps - pt).mean()))
    n = len(examples)
    print(json.dumps({
        "n": n,
        "top1_match": round(top1 / n, 3),
        "top3_contains_teacher_top1": round(top3 / n, 3),
        "mean_spearman": round(float(np.mean(rhos)), 3),
        "mean_abs_prob_diff": round(float(np.mean(maes)), 4),
    }, indent=1))
    print("SALIENCE_EVAL_DONE")


if __name__ == "__main__":
    main()

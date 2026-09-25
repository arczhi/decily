#!/usr/bin/env python3
"""Baseline eval: the deployed cross-encoder ONNX model on the salience eval set.

Runs `OnnxDecisionModel.decide` (softmax over candidate sentences) exactly like
the reader does, and compares against the stored teacher probabilities.

Usage:
    python scripts/eval_onnx_salience.py \
        --model /Users/alex/coding/reader/models_onnx/student_salience/model.int8.onnx \
        --eval data/salience/salience_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/alex/coding/reader")


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def eval_doc_model(args):
    """Evaluate a whole-document salience ONNX graph against stored teacher probs."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from salience_pack import pack_segments, split_builder, tokenize_sentences

    tok = AutoTokenizer.from_pretrained(Path(args.model).parent)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]

    top1 = n = 0
    sps, kls = [], []
    for row in rows:
        q = row["questions"][0]
        probs = q.get("probs")
        if not probs:
            continue
        sents = split_builder(row["state"])
        if len(sents) < 2:
            continue
        text_to_idx: dict[str, list[int]] = {}
        for i, s in enumerate(sents):
            text_to_idx.setdefault(s, []).append(i)
        cand, t = [], []
        used: dict[str, int] = {}
        for opt, p in zip(q["options"], probs):
            txt = opt["text"].strip()
            ids = text_to_idx.get(txt)
            if not ids:
                continue
            k = used.get(txt, 0)
            if k >= len(ids):
                continue
            used[txt] = k + 1
            cand.append(ids[k])
            t.append(float(p))
        if len(cand) < 2:
            continue
        tot = sum(t)
        t = [p / tot for p in t]

        segments, spans = pack_segments(tok, tokenize_sentences(tok, sents), args.max_seg_tokens)
        T = max(len(s) for s in segments)
        Smax = max(len(s) for s in spans)
        B = len(segments)
        ids = np.zeros((B, T), dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.int64)
        smask = np.zeros((B, Smax, T), dtype=np.float32)
        loc: dict[int, tuple[int, int]] = {}
        for b, (seg, sp) in enumerate(zip(segments, spans)):
            ids[b, : len(seg)] = seg
            mask[b, : len(seg)] = 1
            for si, (gi, s0, e0) in enumerate(sp):
                smask[b, si, s0:e0] = 1.0
                loc[gi] = (b, si)
        out = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask, "span_mask": smask})[0]
        lg = np.array([out[loc[gi][0], loc[gi][1]] for gi in cand if gi in loc], dtype=np.float64)
        if len(lg) != len(t):
            continue
        lg = lg - lg.max()
        p = np.exp(lg) / np.exp(lg).sum()
        if int(np.argmax(p)) == int(np.argmax(t)):
            top1 += 1
        kls.append(float((np.array(t) * (np.log(np.array(t) + 1e-9) - np.log(p + 1e-9))).sum()))
        sps.append(spearman([float(x) for x in p], t))
        n += 1
    print(json.dumps({
        "model": args.model,
        "n": n,
        "top1": round(top1 / n, 3) if n else None,
        "spearman": round(sum(s for s in sps if not math.isnan(s)) / max(sum(0 if math.isnan(s) else 1 for s in sps), 1), 3),
        "kl": round(sum(kls) / len(kls), 3) if kls else None,
    }, indent=1))
    print("BASELINE_EVAL_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--doc-model", action="store_true", help="whole-document salience graph (input_ids/attention_mask/span_mask)")
    ap.add_argument("--max-seg-tokens", type=int, default=512)
    args = ap.parse_args()

    if args.doc_model:
        return eval_doc_model(args)

    from reader.onnx_model import OnnxDecisionModel

    model = OnnxDecisionModel(args.model)
    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]

    top1 = n = 0
    sps, kls = [], []
    for row in rows:
        q = row["questions"][0]
        opts = [o["text"] for o in q["options"]]
        probs = q.get("probs")
        if not probs or len(probs) != len(opts):
            continue
        tot = sum(probs)
        if tot <= 0:
            continue
        t = [p / tot for p in probs]
        ranked = model.decide(row["state"], q["text"], opts, temperature=1.0, max_state_tokens=1024)
        p_map = dict(ranked)
        p = [p_map[o] for o in opts]
        if int(max(range(len(p)), key=lambda i: p[i])) == int(max(range(len(t)), key=lambda i: t[i])):
            top1 += 1
        kls.append(sum(ti * math.log((ti + 1e-9) / (pi + 1e-9)) for ti, pi in zip(t, p)))
        sps.append(spearman(p, t))
        n += 1
    print(json.dumps({
        "n": n,
        "top1": round(top1 / n, 3) if n else None,
        "spearman": round(sum(s for s in sps if not math.isnan(s)) / max(sum(0 if math.isnan(s) else 1 for s in sps), 1), 3),
        "kl": round(sum(kls) / len(kls), 3) if kls else None,
    }, indent=1))
    print("BASELINE_EVAL_DONE")


if __name__ == "__main__":
    main()

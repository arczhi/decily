#!/usr/bin/env python3
"""Build whole-document salience JSONL from CNN/DailyMail 3.0.0 (abisee).

Needs revision="refs/convert/parquet" and config "default" to bypass the
removed loading-script path in datasets 5.x.
"""

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset


def split_sent_spans(text):
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for m in re.finditer(r"[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$", text):
        s = m.group().strip()
        if len(s) >= 20:
            out.append((m.start(), m.end(), s))
    return out


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _rouge1_f(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    ov = len(ta & tb)
    p, r = ov / len(ta), ov / len(tb)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def process(article, highlights):
    arts = split_sent_spans(article)
    if len(arts) < 6:
        return None
    sums = [re.sub(r"\s+", " ", h).strip() for h in highlights.split("\n") if h.strip()]
    if not sums:
        return None
    labels = []
    for _, _, s_text in arts:
        score = max(_rouge1_f(s_text, h) for h in sums)
        labels.append(1 if score >= 0.5 else 0)
    pos = sum(labels)
    if pos < 1 or pos > 8:
        return None
    return json.dumps({
        "article": article,
        "sents": [s for _, _, s in arts],
        "spans": [(a, b) for a, b, _ in arts],
        "labels": labels,
    }, ensure_ascii=False)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ds = load_dataset("abisee/cnn_dailymail", "default", split="train", revision="refs/convert/parquet")
    out_path = Path(__file__).resolve().parent.parent / "data" / "doc_salience" / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if limit and n >= limit:
                break
            rec = process(row["article"], row["highlights"])
            if rec is None:
                continue
            f.write(rec + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n} ...", flush=True)
    print(f"wrote {n} articles to {out_path}")
    # split: last 300 for val
    val_path = out_path.parent / "val.jsonl"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 300:
        train = lines[:-300]
        val = lines[-300:]
        out_path.write_text("\n".join(train) + "\n", encoding="utf-8")
        val_path.write_text("\n".join(val) + "\n", encoding="utf-8")
        print(f"split: train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()

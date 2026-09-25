#!/usr/bin/env python3
"""CPU benchmark for the exported salience ONNX model (reader-style flow).

Packages a document into segments the same way training did, runs the ONNX
graph in segment batches, and reports end-to-end latency.

Usage:
    python scripts/bench_salience_onnx.py \
        --model /Users/alex/coding/reader/models_onnx/salience_minilm/model.int8.onnx \
        --ints8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from salience_pack import pack_segments  # noqa: E402


def make_english(n_chars: int) -> str:
    rng = np.random.default_rng(0)
    words = (
        "market analysts expect the central bank to hold rates steady while inflation "
        "continues its slow decline and employment remains resilient across most sectors"
    ).split()
    sents = []
    while sum(len(s) + 1 for s in sents) < n_chars:
        k = int(rng.integers(12, 28))
        s = " ".join(rng.choice(words, size=k))
        sents.append(s[0].upper() + s[1:] + ".")
    return " ".join(sents)


def make_chinese(n_chars: int) -> str:
    rng = np.random.default_rng(0)
    chars = "市场分析师预计央行将维持利率不变通胀继续缓慢回落就业保持韧性经济增长稳定政策空间充足"
    sents = []
    while sum(len(s) + 1 for s in sents) < n_chars:
        k = int(rng.integers(20, 60))
        sents.append("".join(rng.choice(list(chars), size=k)) + "。")
    return "".join(sents)


def split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        for part in re.split(r"(?<=[。！？；!?])\s*|(?<=[.!?])\s+(?=[A-Z\"'(])", para):
            part = part.strip()
            if len(part) >= 8:
                out.append(part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--segments-per-batch", type=int, default=8)
    ap.add_argument("--max-seg-tokens", type=int, default=512)
    ap.add_argument("--chars", type=int, nargs="+", default=[2000, 4000, 8000])
    args = ap.parse_args()

    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or os.path.dirname(args.model))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"model: {args.model}")
    print(f"providers: {sess.get_providers()}")

    for lang, maker in [("en", make_english), ("zh", make_chinese)]:
        for n_chars in args.chars:
            text = maker(n_chars)
            sents = split_sentences(text)
            sents_tok = [tok(s, add_special_tokens=False)["input_ids"] for s in sents]
            segments, spans = pack_segments(tok, sents_tok, args.max_seg_tokens)
            total_tokens = sum(len(s) for s in segments)

            def run_once():
                results: list[tuple[int, float]] = []
                for i in range(0, len(segments), args.segments_per_batch):
                    seg_batch = segments[i : i + args.segments_per_batch]
                    span_batch = spans[i : i + args.segments_per_batch]
                    T = max(len(s) for s in seg_batch)
                    S = max(len(s) for s in span_batch)
                    ids = np.zeros((len(seg_batch), T), dtype=np.int64)
                    mask = np.zeros((len(seg_batch), T), dtype=np.int64)
                    smask = np.zeros((len(seg_batch), S, T), dtype=np.float32)
                    for b, (seg, sp) in enumerate(zip(seg_batch, span_batch)):
                        ids[b, : len(seg)] = seg
                        mask[b, : len(seg)] = 1
                        for si, (gi, s, e) in enumerate(sp):
                            smask[b, si, s:e] = 1.0
                    out = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask, "span_mask": smask})[0]
                    for b, sp in enumerate(span_batch):
                        for si, (gi, _, _) in enumerate(sp):
                            results.append((gi, float(out[b, si])))
                return results

            run_once()
            t0 = time.perf_counter()
            run_once()
            dt = time.perf_counter() - t0
            print(
                f"{lang} {n_chars:6d} chars | {len(sents):4d} sents | {len(segments):3d} segs "
                f"| {total_tokens:6d} tok | {dt*1000:7.0f} ms | {total_tokens/dt:8.0f} tok/s",
                flush=True,
            )
    print("BENCH_DONE")


if __name__ == "__main__":
    main()

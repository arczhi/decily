"""Sentence packing utilities for whole-document salience models (torch-free).

Shared by training, ONNX export/bench and the salience eval scripts.
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_builder(text: str) -> list[str]:
    """Sentence split used when the teacher data was built (build_salience_source)."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SPLIT.split(text) if len(s.strip()) >= 30]


def tokenize_sentences(tok, sents: list[str]) -> list[list[int]]:
    out = []
    for s in sents:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        out.append(ids if ids else [tok.unk_token_id or 0])
    return out


def pack_segments(tok, sents_tokens: list[list[int]], max_seg: int):
    """Greedy sentence packing into segments.

    Returns (segments, spans) where segments[b] is a token id list and
    spans[b] is a list of (sent_global_idx, tok_start, tok_end) within it.
    """
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    body = max_seg - (1 if cls_id is not None else 0) - (1 if sep_id is not None else 0)
    segments: list[list[int]] = []
    spans: list[list[tuple[int, int, int]]] = []
    cur_ids: list[int] = []
    cur_spans: list[tuple[int, int, int]] = []

    def flush():
        nonlocal cur_ids, cur_spans
        if cur_spans:
            ids = ([cls_id] if cls_id is not None else []) + cur_ids + (
                [sep_id] if sep_id is not None else []
            )
            off = 1 if cls_id is not None else 0
            segments.append(ids)
            spans.append([(gi, s + off, e + off) for gi, s, e in cur_spans])
        cur_ids, cur_spans = [], []

    for gi, st in enumerate(sents_tokens):
        if len(st) > body:
            st = st[:body]
        if cur_ids and len(cur_ids) + len(st) > body:
            flush()
        s0 = len(cur_ids)
        cur_ids.extend(st)
        cur_spans.append((gi, s0, s0 + len(st)))
    flush()
    return segments, spans

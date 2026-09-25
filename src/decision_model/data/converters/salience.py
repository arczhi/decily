"""Salience task: which article sentences are key points? (extractive labels)

CNN/DailyMail: article sentences are labeled gold when they overlap
(ROUGE-1 F>0.5) with any reference summary sentence. Task type is `noul`
(multi-label, independent sigmoid per candidate) so the reader gets absolute
per-sentence scores usable across chunks.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from ..schema import DecisionExample, Option, Question
from .hf_tasks import _limit, register

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = _SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 25]


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _rouge1_f(pred: str, ref: str) -> float:
    a, b = _tokens(pred), _tokens(ref)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    precision = overlap / len(a)
    recall = overlap / len(b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _label(article_sents: list[str], summary_sents: list[str], threshold: float = 0.5) -> list[int]:
    gold = []
    for i, s in enumerate(article_sents):
        if any(_rouge1_f(s, h) >= threshold for h in summary_sents):
            gold.append(i)
    return gold


def build(
    split: str,
    limit: int | None,
    max_candidates: int = 12,
    max_article_sents: int = 60,
    seed: int = 0,
) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("ccdv/cnn_dailymail", "3.0.0", split=split)
    rng = random.Random(seed)
    n = 0
    for row in ds:
        if limit is not None and n >= limit:
            break
        article_sents = split_sentences(row["article"])[:max_article_sents]
        summary_sents = split_sentences(row["highlights"])
        if len(article_sents) < 8 or not summary_sents:
            continue
        gold_idx = _label(article_sents, summary_sents)
        if not gold_idx:
            continue
        # candidates: all gold + sampled negatives, shuffled, capped
        neg_idx = [i for i in range(len(article_sents)) if i not in set(gold_idx)]
        rng.shuffle(neg_idx)
        n_neg = max(min(len(neg_idx), max_candidates - len(gold_idx)), 1)
        chosen = sorted(gold_idx + neg_idx[:n_neg])
        if len(chosen) > max_candidates:
            chosen = chosen[:max_candidates]
        if len(chosen) < 3:
            continue
        rng.shuffle(chosen)
        options = [Option(id=f"s{i}", text=article_sents[i]) for i in chosen]
        answers = [f"s{i}" for i in chosen if i in set(gold_idx)]
        if not answers:
            continue
        n += 1
        yield DecisionExample(
            task="salience_cnn",
            state=row["article"],
            questions=[
                Question(
                    text="Which sentences are essential to the key points of this article?",
                    options=options,
                    answer_ids=answers,
                    type="noul",
                )
            ],
            meta={"n_gold": len(gold_idx), "n_candidates": len(chosen)},
        )


@register("salience_cnn")
def salience_cnn(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    yield from build(split, limit, seed=0)


@register("salience_cnn_val")
def salience_cnn_val(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    s = "validation" if split in ("train", "val") else split
    yield from build(s, limit, seed=7)

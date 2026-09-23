"""Real-ambiguity belief data: GoEmotions rater-agreement distributions.

The `raw` config of GoEmotions has one row per (text, rater). We aggregate the
three raters' votes into a distribution over the top emotion candidates, so the
model is trained to report *human disagreement* instead of forcing 0/1.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator

from ..schema import DecisionExample, Option, Question
from .hf_tasks import register

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

_QUESTION = "Which emotion does this text express most strongly?"


def _build(split: str, max_texts: int, top_k: int = 5, skip_texts: int = 0) -> list[DecisionExample]:
    """Note: the `raw` config only has a train split; held-out slices come from
    a later, disjoint range of text ids (skip_texts)."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "raw", split=split)
    groups: dict[str, dict] = {}
    for row in ds:
        if row["example_very_unclear"]:
            continue
        g = groups.get(row["id"])
        if g is None:
            if len(groups) >= max_texts + skip_texts:
                continue
            g = groups[row["id"]] = {"text": row["text"], "votes": Counter()}
        for emo in EMOTIONS:
            if row[emo]:
                g["votes"][emo] += 1

    out: list[DecisionExample] = []
    for g in list(groups.values())[skip_texts:]:
        votes: Counter = g["votes"]
        if not votes:
            continue
        ranked = votes.most_common(top_k)
        total = sum(v for _, v in ranked)
        labels = [e for e, _ in ranked]
        probs = [v / total for _, v in ranked]
        options = [Option(id=e, text=e.replace("_", " ")) for e in labels]
        answer = labels[0]
        out.append(
            DecisionExample(
                task="goemotions",
                state=g["text"],
                questions=[
                    Question(
                        text=_QUESTION,
                        options=options,
                        answer_ids=[answer],
                        type="choice",
                        probs=probs,
                    )
                ],
                meta={"raters": int(sum(votes.values())), "votes": dict(votes)},
            )
        )
    return out


@register("goemotions_belief")
def goemotions_belief(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    rows = _build("train", max_texts=limit if limit is not None else 8000)
    yield from rows


@register("goemotions_belief_eval")
def goemotions_belief_eval(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    n = limit if limit is not None else 1000
    rows = _build("train", max_texts=n, skip_texts=8000)
    yield from rows

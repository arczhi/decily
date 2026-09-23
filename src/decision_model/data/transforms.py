"""Practical data transforms: abstention ("none of the above") construction."""

from __future__ import annotations

import random

from .schema import DecisionExample, Option, Question

NONE_ID = "none_of_the_above"
NONE_TEXT = "none of the above"


def add_abstention(
    example: DecisionExample,
    rng: random.Random,
    p: float = 0.2,
    mode: str = "drop_gold",
    shuffle: bool = True,
) -> DecisionExample:
    """Add a "none of the above" option.

    mode="drop_gold": gold option is removed and none becomes correct (teaches
    abstention when the answer is absent from the candidate set).
    mode="keep_gold": none is added as a distractor (for measuring false
    abstention). With shuffle=False the none option is always last.
    """
    if rng.random() >= p:
        return example
    q = example.questions[0]
    gold = q.answer_set()
    if len(q.options) < 2 or len(gold) != 1:
        return example
    if any(o.id == NONE_ID for o in q.options):
        return example
    if mode == "drop_gold":
        gold_id = next(iter(gold))
        kept = [o for o in q.options if o.id != gold_id]
        answer_ids = [NONE_ID]
    elif mode == "keep_gold":
        kept = list(q.options)
        answer_ids = list(q.answer_ids)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    kept.append(Option(id=NONE_ID, text=NONE_TEXT))
    if shuffle:
        rng.shuffle(kept)
    new_q = Question(
        text=q.text,
        options=kept,
        answer_ids=answer_ids,
        type="choice",
        probs=None,
    )
    return DecisionExample(task=example.task, state=example.state, questions=[new_q], meta=example.meta)


def drop_gold(example: DecisionExample) -> DecisionExample:
    """Remove the gold option and mark the question as defer (answer absent)."""
    q = example.questions[0]
    if q.defer or len(q.answer_ids) != 1:
        return example
    gold_id = q.answer_ids[0]
    kept = [o for o in q.options if o.id != gold_id]
    if len(kept) < 2:
        return example
    new_q = Question(
        text=q.text,
        options=kept,
        answer_ids=[],
        type=q.type,
        probs=None,
        defer=True,
    )
    return DecisionExample(task=example.task, state=example.state, questions=[new_q], meta=example.meta)

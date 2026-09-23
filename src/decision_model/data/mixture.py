"""Mixture sampling and candidate sub-sampling for training.

Training-time transformations (matching the decider recipe):
- large option sets are sub-sampled to at most `max_options` candidates,
  always keeping the gold answer(s) and shuffling the rest;
- a random single question is kept per state (Q=1 by default).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from .schema import DecisionExample, Option, Question


@dataclass
class MixtureItem:
    task: str
    path: str
    weight: float


def subsample_question(
    question: Question,
    rng: random.Random,
    min_options: int = 2,
    max_options: int = 10,
) -> Question:
    """Sub-sample distractors for large label sets; keep every gold option."""
    opts = list(question.options)
    gold = question.answer_set()
    if len(opts) <= max_options:
        return question

    keep_gold = [o for o in opts if o.id in gold]
    distractors = [o for o in opts if o.id not in gold]
    rng.shuffle(distractors)
    n_slots = max(max_options, len(keep_gold) + 1)
    need = max(min_options - len(keep_gold), 0)
    chosen = keep_gold + distractors[: max(n_slots - len(keep_gold), need)]
    rng.shuffle(chosen)
    return Question(options=chosen, answer_ids=list(question.answer_ids),
                    text=question.text, type=question.type)


def pick_question(example: DecisionExample, rng: random.Random) -> DecisionExample:
    if len(example.questions) == 1:
        return example
    q = rng.choice(example.questions)
    return DecisionExample(task=example.task, state=example.state,
                           questions=[q], meta=example.meta)


def prepare_example(
    example: DecisionExample,
    rng: random.Random,
    min_options: int = 2,
    max_options: int = 10,
) -> DecisionExample:
    example = pick_question(example, rng)
    q = subsample_question(example.questions[0], rng, min_options, max_options)
    return DecisionExample(task=example.task, state=example.state,
                           questions=[q], meta=example.meta)


class MixtureSampler:
    """Weighted sampler over pre-converted JSONL shards."""

    def __init__(
        self,
        items: list[MixtureItem],
        seed: int = 0,
        min_options: int = 2,
        max_options: int = 10,
    ) -> None:
        if not items:
            raise ValueError("empty mixture")
        self.items = items
        self.seed = seed
        self.min_options = min_options
        self.max_options = max_options
        self._pools: list[list[DecisionExample]] = []
        for item in items:
            from .schema import read_jsonl

            self._pools.append(list(read_jsonl(item.path)))

    @property
    def weights(self) -> list[float]:
        w = [max(i.weight, 0.0) for i in self.items]
        s = sum(w)
        return [x / s for x in w]

    def sample(self, n: int, start_seed: int | None = None) -> Iterator[DecisionExample]:
        rng = random.Random(self.seed if start_seed is None else start_seed)
        w = self.weights
        for _ in range(n):
            pool_idx = rng.choices(range(len(self._pools)), weights=w, k=1)[0]
            pool = self._pools[pool_idx]
            ex = pool[rng.randrange(len(pool))]
            yield prepare_example(ex, rng, self.min_options, self.max_options)

    def task_counts(self, n: int = 2000) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ex in self.sample(n, start_seed=self.seed + 1):
            counts[ex.task] = counts.get(ex.task, 0) + 1
        return counts

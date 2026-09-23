"""Synthetic belief tasks: stochastic processes with known probability laws.

These carry soft probability targets (`Question.probs`), so the model can be
trained/scored with proper scoring rules (log score) against the true law,
which is the core of the RLCD belief objective.

Registered:
  belief       - training stream (many templates, seeded)
  belief_eval  - held-out stream (different seed/templates for generalization)
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

from ..schema import DecisionExample, Option, Question
from .hf_tasks import register


def _q(text: str, labels: list[str], probs: list[float], texts: list[str] | None = None) -> Question:
    options = [
        Option(id=l, text=(texts[i] if texts else l)) for i, l in enumerate(labels)
    ]
    answer = labels[max(range(len(probs)), key=lambda i: probs[i])]
    return Question(text=text, options=options, answer_ids=[answer], type="choice", probs=probs)


def _t_coin(rng: random.Random) -> DecisionExample:
    p = rng.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
    pct = int(round(p * 100))
    style = rng.randrange(3)
    if style == 0:
        state = f"A coin is biased: it lands heads with probability {pct}%."
    elif style == 1:
        state = f"You flip a weighted coin. It shows heads {pct}% of the time."
    else:
        state = f"A trick coin has a {pct} in 100 chance of landing heads."
    return DecisionExample(
        task="belief_coin",
        state=state,
        questions=[
            _q("Will the coin land heads?", ["heads", "tails"], [p, 1 - p])
        ],
    )


def _t_fair_coin(rng: random.Random) -> DecisionExample:
    n = rng.choice([2, 3])
    p = 0.5**n
    return DecisionExample(
        task="belief_coins",
        state=f"{n} fair coins are flipped independently.",
        questions=[
            _q(
                f"Will all {n} coins land heads?",
                ["yes", "no"],
                [p, 1 - p],
                texts=[f"all {n} heads", "otherwise"],
            )
        ],
    )


def _t_die(rng: random.Random) -> DecisionExample:
    style = rng.randrange(3)
    if style == 0:
        n = rng.randint(1, 6)
        return DecisionExample(
            task="belief_die",
            state="A fair six-sided die is rolled once.",
            questions=[_q(f"Will the die show a {n}?", ["yes", "no"], [1 / 6, 5 / 6])],
        )
    if style == 1:
        k = rng.randint(1, 5)
        p = (6 - k) / 6
        return DecisionExample(
            task="belief_die",
            state="A fair six-sided die is rolled once.",
            questions=[
                _q(
                    f"Will the die show a number greater than {k}?",
                    ["yes", "no"],
                    [p, 1 - p],
                )
            ],
        )
    return DecisionExample(
        task="belief_die",
        state="A fair six-sided die is rolled once.",
        questions=[
            _q("Will the die show an even number?", ["yes", "no"], [0.5, 0.5])
        ],
    )


def _t_cards(rng: random.Random) -> DecisionExample:
    style = rng.randrange(3)
    state = "A card is drawn uniformly at random from a standard 52-card deck."
    if style == 0:
        return DecisionExample(
            task="belief_cards",
            state=state,
            questions=[_q("Is the card a heart?", ["yes", "no"], [13 / 52, 39 / 52])],
        )
    if style == 1:
        return DecisionExample(
            task="belief_cards",
            state=state,
            questions=[
                _q("Is the card a face card (J, Q, K)?", ["yes", "no"], [12 / 52, 40 / 52])
            ],
        )
    return DecisionExample(
        task="belief_cards",
        state=state,
        questions=[
            _q(
                "Is the card either red or an ace?",
                ["yes", "no"],
                [(26 + 4 - 2) / 52, 1 - (26 + 4 - 2) / 52],
            )
        ],
    )


def _t_urn(rng: random.Random) -> DecisionExample:
    r = rng.randint(1, 12)
    b = rng.randint(1, 12)
    p = r / (r + b)
    color = rng.choice(["red", "blue", "green"])
    other = rng.choice([c for c in ["red", "blue", "green", "yellow"] if c != color])
    return DecisionExample(
        task="belief_urn",
        state=(
            f"An urn contains {r} {color} balls and {b} {other} balls. "
            "One ball is drawn uniformly at random."
        ),
        questions=[
            _q(f"Will the drawn ball be {color}?", ["yes", "no"], [p, 1 - p])
        ],
    )


def _t_dice_sum(rng: random.Random) -> DecisionExample:
    n = rng.choice([2, 3])
    k = rng.choice([7, 8, 9, 10, 11, 12, 14, 15, 16])
    # enumerate exact law for n dice
    from itertools import product

    outcomes = list(product(range(1, 7), repeat=n))
    hits = sum(1 for o in outcomes if sum(o) >= k)
    p = hits / len(outcomes)
    return DecisionExample(
        task="belief_dice_sum",
        state=f"{n} fair six-sided dice are rolled.",
        questions=[
            _q(
                f"Will the sum of the dice be at least {k}?",
                ["yes", "no"],
                [p, 1 - p],
            )
        ],
    )


TEMPLATES: list[Callable[[random.Random], DecisionExample]] = [
    _t_coin,
    _t_fair_coin,
    _t_die,
    _t_cards,
    _t_urn,
    _t_dice_sum,
]


def _stream(seed: int, limit: int | None) -> Iterator[DecisionExample]:
    rng = random.Random(seed)
    i = 0
    while limit is None or i < limit:
        tpl = TEMPLATES[i % len(TEMPLATES)]
        yield tpl(rng)
        i += 1


@register("belief")
def belief(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    yield from _stream(seed=1234, limit=limit if limit is not None else 20000)


@register("belief_eval")
def belief_eval(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    # different seed -> different parameter draws; same laws
    yield from _stream(seed=98765, limit=limit if limit is not None else 2000)

"""Dataset converters: HF datasets -> unified decision tuples.

Each converter yields DecisionExample records. Add a converter with the
@register decorator; `scripts/convert_data.py` exposes them on the CLI.
"""

from __future__ import annotations

from typing import Callable, Iterator

from ..schema import DecisionExample, Option, Question

Converter = Callable[..., Iterator[DecisionExample]]

REGISTRY: dict[str, Converter] = {}


def register(name: str) -> Callable[[Converter], Converter]:
    def deco(fn: Converter) -> Converter:
        REGISTRY[name] = fn
        return fn

    return deco


def simple_choice(
    task: str,
    state: str,
    question_text: str,
    labels: list[str],
    answer: str,
    label_texts: dict[str, str] | None = None,
    meta: dict | None = None,
) -> DecisionExample:
    texts = label_texts or {}
    options = [Option(id=l, text=texts.get(l, l)) for l in labels]
    return DecisionExample(
        task=task,
        state=state,
        questions=[
            Question(text=question_text, options=options, answer_ids=[answer], type="choice")
        ],
        meta=meta or {},
    )


def _limit(split_ds, limit: int | None, offset: int = 0):
    if offset:
        split_ds = split_ds.select(range(offset, len(split_ds)))
    if limit is not None:
        split_ds = split_ds.select(range(min(limit, len(split_ds))))
    return split_ds


@register("ag_news")
def ag_news(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    labels = ["world", "sports", "business", "sci_tech"]
    texts = {
        "world": "world news",
        "sports": "sports",
        "business": "business",
        "sci_tech": "science and technology",
    }
    ds = load_dataset("fancyzhx/ag_news", split=split)
    for row in _limit(ds, limit):
        yield simple_choice(
            task="ag_news",
            state=row["text"],
            question_text="Which topic does this news article belong to?",
            labels=labels,
            answer=labels[row["label"]],
            label_texts=texts,
        )


@register("sst2")
def sst2(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    labels = ["negative", "positive"]
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    for row in _limit(ds, limit):
        yield simple_choice(
            task="sst2",
            state=row["sentence"],
            question_text="What is the sentiment of this sentence?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("mnli")
def mnli(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    labels = ["entailment", "neutral", "contradiction"]
    texts = {
        "entailment": "the hypothesis follows from the premise",
        "neutral": "the hypothesis may or may not be true given the premise",
        "contradiction": "the hypothesis contradicts the premise",
    }
    if split == "validation":
        split = "validation_matched"
    ds = load_dataset("nyu-mll/glue", "mnli", split=split)
    for row in _limit(ds, limit):
        state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
        yield simple_choice(
            task="mnli",
            state=state,
            question_text="What is the relationship between the hypothesis and the premise?",
            labels=labels,
            answer=labels[row["label"]],
            label_texts=texts,
        )


@register("arc_easy")
def arc_easy(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    for row in _limit(ds, limit):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer = row["answerKey"]
        if answer not in labels:
            continue
        state = row["question"]
        if row.get("context"):
            state = f"{row['context']}\n\n{state}"
        yield simple_choice(
            task="arc_easy",
            state=state,
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=answer,
            label_texts=dict(zip(labels, texts)),
        )


@register("emotion")
def emotion(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("mteb/emotion", split=split)
    labels = sorted(set(load_dataset("mteb/emotion", split="train")["label_text"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="emotion",
            state=row["text"],
            question_text="What emotion does this text express?",
            labels=labels,
            answer=row["label_text"],
        )


@register("massive")
def massive(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("mteb/amazon_massive_intent", "en", split=split)
    labels = sorted(set(load_dataset("mteb/amazon_massive_intent", "en", split="train")["label"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="massive",
            state=row["text"],
            question_text="Which intent does this user request belong to?",
            labels=labels,
            answer=row["label"],
        )


@register("banking77")
def banking77(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    if split == "val":
        split = "test"
    ds = load_dataset("mteb/banking77", split=split)
    labels = sorted(set(load_dataset("mteb/banking77", split="train")["label_text"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="banking77",
            state=row["text"],
            question_text="Which banking intent does this customer message belong to?",
            labels=labels,
            answer=row["label_text"],
        )


@register("clinc")
def clinc(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    """Legacy: PolyAI/clinc_oos was removed from the Hub (2026)."""
    from datasets import load_dataset

    ds = load_dataset("PolyAI/clinc_oos", "plus", split=split)
    intents = ds.features["intent"].names
    for row in _limit(ds, limit):
        intent = intents[row["intent"]]
        if intent == "oos":
            continue
        yield simple_choice(
            task="clinc",
            state=row["text"],
            question_text="Which intent does this user request belong to?",
            labels=intents,
            answer=intent,
        )

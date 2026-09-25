"""Fair-suite converters: tasks held out by BOTH our model and decider-2b.

All eight tasks are marked heldout=true in decider-2b's published eval and are
outside our 24-task training mixture:
  paws, sciq, pubmedqa, bbc_news, quality, fin_phrasebank, dolly_category,
  xstory_cloze
"""

from __future__ import annotations

import random
from typing import Iterator

from ..schema import DecisionExample
from .hf_tasks import _limit, register, simple_choice


@register("paws")
def paws(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    labels = ["not paraphrase", "paraphrase"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="paws",
            state=f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
            question_text="Are these two sentences paraphrases of each other?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("sciq")
def sciq(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split=split)
    rng = random.Random(0)
    for row in _limit(ds, limit):
        choices = [
            row["correct_answer"],
            row["distractor1"],
            row["distractor2"],
            row["distractor3"],
        ]
        rng.shuffle(choices)
        labels = [f"o{i}" for i in range(len(choices))]
        state = row["question"]
        if row.get("support"):
            state = f"{row['support']}\n\n{state}"
        yield simple_choice(
            task="sciq",
            state=state,
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=labels[choices.index(row["correct_answer"])],
            label_texts=dict(zip(labels, choices)),
        )


@register("pubmedqa")
def pubmedqa(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    labels = ["yes", "no", "maybe"]
    for row in _limit(ds, limit):
        ctx = row["context"]
        context = " ".join(ctx["contexts"]) if isinstance(ctx, dict) else str(ctx)
        yield simple_choice(
            task="pubmedqa",
            state=f"{context}\n\nQuestion: {row['question']}",
            question_text="Does the evidence support the claim?",
            labels=labels,
            answer=row["final_decision"],
        )


@register("bbc_news")
def bbc_news(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("SetFit/bbc-news", split=split)
    labels = sorted(set(ds["label_text"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="bbc_news",
            state=row["text"],
            question_text="Which topic is this news article about?",
            labels=labels,
            answer=row["label_text"],
        )


@register("quality")
def quality(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("emozilla/quality", split=split)
    for row in _limit(ds, limit):
        options = list(row["options"])
        labels = [f"o{i}" for i in range(len(options))]
        yield simple_choice(
            task="quality",
            state=f"{row['article']}\n\nQuestion: {row['question']}",
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=labels[int(row["answer"])],
            label_texts=dict(zip(labels, options)),
        )


@register("fin_phrasebank")
def fin_phrasebank(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("mteb/financial_phrasebank", split=split)
    # note: this dataset's ClassLabel names are literally [0, 1, 2] -> use semantics
    labels = ["negative", "neutral", "positive"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="fin_phrasebank",
            state=row["text"],
            question_text="What is the sentiment of this financial sentence?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("dolly_category")
def dolly_category(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    labels = sorted(set(ds["category"]))
    for row in _limit(ds, limit):
        state = row["instruction"]
        if row.get("context"):
            state = f"{state}\n\nContext: {row['context']}"
        yield simple_choice(
            task="dolly_category",
            state=state,
            question_text="Which category does this instruction belong to?",
            labels=labels,
            answer=row["category"],
        )


@register("xstory_cloze")
def xstory_cloze(split: str = "eval", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("juletxara/xstory_cloze", "en", split=split)
    labels = ["ending1", "ending2"]
    texts = {"ending1": None, "ending2": None}
    for row in _limit(ds, limit):
        story = "\n".join(row[f"input_sentence_{i}"] for i in range(1, 5))
        endings = [row["sentence_quiz1"], row["sentence_quiz2"]]
        answer = labels[int(row["answer_right_ending"]) - 1]
        yield simple_choice(
            task="xstory_cloze",
            state=story,
            question_text="Which ending makes the most sense?",
            labels=labels,
            answer=answer,
            label_texts={"ending1": endings[0], "ending2": endings[1]},
        )

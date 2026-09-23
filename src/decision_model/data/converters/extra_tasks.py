"""Stage 2 extra task converters (parquet-only datasets, no loading scripts)."""

from __future__ import annotations

from typing import Iterator

from ..schema import DecisionExample
from .hf_tasks import _limit, register, simple_choice

_LABEL_CACHE: dict[tuple, list[str]] = {}


def _labels_from_train(repo: str, config: str | None = None, column: str = "label_text") -> list[str]:
    key = (repo, config, column)
    if key not in _LABEL_CACHE:
        from datasets import load_dataset

        ds = (
            load_dataset(repo, config, split="train")
            if config
            else load_dataset(repo, split="train")
        )
        _LABEL_CACHE[key] = sorted(set(ds[column]))
    return _LABEL_CACHE[key]


def _mteb_task(
    repo: str,
    question: str,
    task: str,
    split: str = "train",
    config: str | None = None,
    limit: int | None = None,
    text_col: str = "text",
    label_col: str = "label_text",
) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    if split == "val":
        split = "test"
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    labels = _labels_from_train(repo, config, label_col)
    for row in _limit(ds, limit):
        yield simple_choice(
            task=task,
            state=row[text_col],
            question_text=question,
            labels=labels,
            answer=row[label_col],
        )


@register("amazon_polarity")
def amazon_polarity(split="train", limit=None):
    yield from _mteb_task(
        "mteb/amazon_polarity",
        "What is the sentiment of this product review?",
        "amazon_polarity",
        split,
        limit=limit,
    )


@register("imdb")
def imdb(split="train", limit=None):
    yield from _mteb_task(
        "mteb/imdb",
        "Is this movie review positive or negative?",
        "imdb",
        split,
        limit=limit,
    )


@register("tweet_sentiment")
def tweet_sentiment(split="train", limit=None):
    yield from _mteb_task(
        "mteb/tweet_sentiment_extraction",
        "What is the sentiment of this tweet?",
        "tweet_sentiment",
        split,
        limit=limit,
    )


@register("toxic_conversations")
def toxic_conversations(split="train", limit=None):
    yield from _mteb_task(
        "mteb/toxic_conversations",
        "Is this message toxic?",
        "toxic_conversations",
        split,
        limit=limit,
    )


@register("counterfactual")
def counterfactual(split="train", limit=None):
    yield from _mteb_task(
        "mteb/amazon_counterfactual",
        "Does flipping one word change this review's sentiment label?",
        "counterfactual",
        split,
        config="en",
        limit=limit,
    )


@register("massive_scenario")
def massive_scenario(split="train", limit=None):
    yield from _mteb_task(
        "mteb/amazon_massive_scenario",
        "Which assistant scenario does this user request belong to?",
        "massive_scenario",
        split,
        config="en",
        limit=limit,
    )


@register("tweet_topic")
def tweet_topic(split="train", limit=None):
    yield from _mteb_task(
        "mteb/tweet_topic_single",
        "What topic is this tweet about?",
        "tweet_topic",
        split,
        limit=limit,
        label_col="label_name",
    )


@register("poem_sentiment")
def poem_sentiment(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "test"
    ds = load_dataset("mteb/poem_sentiment", split=split)
    labels = list(ds.features["label"].names)
    for row in _limit(ds, limit):
        yield simple_choice(
            task="poem_sentiment",
            state=row["text"],
            question_text="What is the sentiment of this poem verse?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("dbpedia")
def dbpedia(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "test"
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    labels = list(ds.features["label"].names)
    for row in _limit(ds, limit):
        yield simple_choice(
            task="dbpedia",
            state=f"{row['title']}\n{row['content']}",
            question_text="Which category does this Wikipedia article belong to?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("rte")
def rte(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("nyu-mll/glue", "rte", split=split)
    labels = list(ds.features["label"].names)
    for row in _limit(ds, limit):
        yield simple_choice(
            task="rte",
            state=f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}",
            question_text="Does the hypothesis follow from the premise?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("qqp")
def qqp(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("nyu-mll/glue", "qqp", split=split)
    labels = ["not duplicate", "duplicate"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="qqp",
            state=f"Question 1: {row['question1']}\nQuestion 2: {row['question2']}",
            question_text="Are these two questions duplicates of each other?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("mrpc")
def mrpc(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("nyu-mll/glue", "mrpc", split=split)
    labels = ["not paraphrase", "paraphrase"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="mrpc",
            state=f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
            question_text="Are these two sentences paraphrases of each other?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("arc_challenge")
def arc_challenge(split="train", limit=None):
    from datasets import load_dataset

    if split == "val":
        split = "validation"
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    for row in _limit(ds, limit):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer = row["answerKey"]
        if answer not in labels:
            continue
        yield simple_choice(
            task="arc_challenge",
            state=row["question"],
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=answer,
            label_texts=dict(zip(labels, texts)),
        )


MMLU_SUBJECTS = [
    "high_school_world_history",
    "high_school_biology",
    "high_school_physics",
    "high_school_government_and_politics",
    "college_computer_science",
    "moral_scenarios",
    "professional_law",
    "world_religions",
]


def mmlu_converter(subject: str):
    def conv(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", subject, split=split)
        for row in _limit(ds, limit):
            choices = list(row["choices"])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            yield simple_choice(
                task=f"mmlu_{subject}",
                state=row["question"],
                question_text="Which option answers the question correctly?",
                labels=labels,
                answer=labels[int(row["answer"])],
                label_texts=dict(zip(labels, choices)),
            )

    return conv


for _subject in MMLU_SUBJECTS:
    register(f"mmlu_{_subject}")(mmlu_converter(_subject))

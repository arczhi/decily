"""Stage-1 expansion tasks: 24 -> ~45 tasks for the new base model.

All datasets verified loadable on the hf-mirror (parquet, no scripts) and
disjoint from the fair-suite benchmark (paws/sciq/pubmedqa/bbc_news/quality/
fin_phrasebank/dolly_category/xstory_cloze stay held out).
"""

from __future__ import annotations

import json
import random
import re
from typing import Iterator

from ..schema import DecisionExample, Option, Question
from .hf_tasks import _limit, register, simple_choice


# ---------------- commonsense / reasoning ----------------

@register("hellaswag")
def hellaswag(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("allenai/hellaswag", split=split)
    for row in _limit(ds, limit):
        endings = list(row["endings"])
        labels = [f"e{i}" for i in range(len(endings))]
        yield simple_choice(
            task="hellaswag",
            state=row["ctx"],
            question_text="Which ending continues the situation most plausibly?",
            labels=labels,
            answer=labels[int(row["label"])],
            label_texts=dict(zip(labels, endings)),
        )


def _arc_style(task: str, repo: str, config: str | None, split: str, question: str,
               limit: int | None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    for row in _limit(ds, limit):
        choices = row["choices"]
        labels = list(choices["label"])
        texts = list(choices["text"])
        answer = row["answerKey"]
        if answer not in labels:
            continue
        yield simple_choice(
            task=task,
            state=row.get("question_stem") or row["question"],
            question_text=question,
            labels=labels,
            answer=answer,
            label_texts=dict(zip(labels, texts)),
        )


@register("openbookqa")
def openbookqa(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    yield from _arc_style(
        "openbookqa", "allenai/openbookqa", "main", split,
        "Which option answers the question correctly?", limit,
    )


@register("commonsense_qa")
def commonsense_qa(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    yield from _arc_style(
        "commonsense_qa", "tau/commonsense_qa", None, split,
        "Which option answers the question correctly?", limit,
    )


@register("qasc")
def qasc(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    yield from _arc_style(
        "qasc", "allenai/qasc", None, split,
        "Which option answers the question correctly?", limit,
    )


@register("winogrande")
def winogrande(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    labels = ["option1", "option2"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="winogrande",
            state=row["sentence"],
            question_text="Which option fills the blank correctly?",
            labels=labels,
            answer=labels[int(row["answer"]) - 1],
            label_texts={"option1": row["option1"], "option2": row["option2"]},
        )


# ---------------- GLUE family ----------------

@register("boolq")
def boolq(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split=split)
    labels = ["no", "yes"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="boolq",
            state=f"{row['passage']}\n\nQuestion: {row['question']}",
            question_text="Does the passage answer the question with yes?",
            labels=labels,
            answer=labels[int(bool(row["answer"]))],
        )


@register("cola")
def cola(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "cola", split=split)
    labels = ["unacceptable", "acceptable"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="cola",
            state=row["sentence"],
            question_text="Is this sentence grammatically acceptable?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("qnli")
def qnli(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "qnli", split=split)
    labels = ["entailment", "not entailment"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="qnli",
            state=f"Sentence: {row['sentence']}\nQuestion: {row['question']}",
            question_text="Does the sentence contain the answer to the question?",
            labels=labels,
            answer=labels[row["label"]],
        )


# ---------------- knowledge QA ----------------

@register("medmcqa")
def medmcqa(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    labels = ["A", "B", "C", "D"]
    for row in _limit(ds, limit):
        texts = [row["opa"], row["opb"], row["opc"], row["opd"]]
        yield simple_choice(
            task="medmcqa",
            state=row["question"],
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=labels[int(row["cop"])],
            label_texts=dict(zip(labels, texts)),
        )


@register("medqa")
def medqa(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    for row in _limit(ds, limit):
        opts = row["options"]
        labels = sorted(opts)
        yield simple_choice(
            task="medqa",
            state=row["question"],
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=row["answer_idx"],
            label_texts={k: opts[k] for k in labels},
        )


@register("race")
def race(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("ehovy/race", "high", split=split)
    labels = ["A", "B", "C", "D"]
    for row in _limit(ds, limit):
        options = list(row["options"])
        yield simple_choice(
            task="race",
            state=f"{row['article']}\n\nQuestion: {row['question']}",
            question_text="Which option answers the question correctly?",
            labels=labels,
            answer=row["answer"],
            label_texts=dict(zip(labels, options)),
        )


@register("wiki_qa")
def wiki_qa(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("microsoft/wiki_qa", split=split)
    labels = ["not relevant", "relevant"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="wiki_qa",
            state=f"Question: {row['question']}\nAnswer: {row['answer']}",
            question_text="Is this answer relevant to the question?",
            labels=labels,
            answer=labels[int(row["label"])],
        )


# ---------------- sentiment / topic / toxicity ----------------

@register("yelp_polarity")
def yelp_polarity(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/yelp_polarity", split=split)
    labels = ["negative", "positive"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="yelp_polarity",
            state=row["text"],
            question_text="What is the sentiment of this restaurant review?",
            labels=labels,
            answer=labels[row["label"]],
        )


@register("sst5")
def sst5(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split=split)
    labels = sorted(set(ds["label_text"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="sst5",
            state=row["text"],
            question_text="What is the sentiment of this movie review?",
            labels=labels,
            answer=row["label_text"],
        )


@register("newsgroups20")
def newsgroups20(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("SetFit/20_newsgroups", split=split)
    labels = sorted(set(ds["label_text"]))
    for row in _limit(ds, limit):
        yield simple_choice(
            task="newsgroups20",
            state=row["text"],
            question_text="Which newsgroup topic does this post belong to?",
            labels=labels,
            answer=row["label_text"],
        )


@register("civil_comments")
def civil_comments(split: str = "validation", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split=split)
    labels = ["not toxic", "toxic"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="civil_comments",
            state=row["text"],
            question_text="Is this comment toxic?",
            labels=labels,
            answer=labels[int(float(row["toxicity"]) >= 0.5)],
        )


@register("sms_spam")
def sms_spam(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("ucirvine/sms_spam", split=split)
    labels = ["ham", "spam"]
    for row in _limit(ds, limit):
        yield simple_choice(
            task="sms_spam",
            state=row["sms"],
            question_text="Is this SMS message spam?",
            labels=labels,
            answer=labels[row["label"]],
        )


BIAS_IN_BIOS_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior designer", "journalist",
    "model", "nurse", "painter", "photographer", "physician", "poet", "professor",
    "psychologist", "rapper", "software engineer", "surgeon", "teacher",
    "yoga teacher", "personal trainer", "paralegal", "pastor",
]


@register("bias_in_bios")
def bias_in_bios(split: str = "test", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("LabHC/bias_in_bios", split=split)
    labels = BIAS_IN_BIOS_PROFESSIONS
    for row in _limit(ds, limit):
        yield simple_choice(
            task="bias_in_bios",
            state=row["hard_text"],
            question_text="Which profession does this biography describe?",
            labels=labels,
            answer=labels[int(row["profession"])],
        )


# ---------------- tool selection ----------------

_FUNC_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _tool_names(text: str) -> list[str]:
    return sorted(set(_FUNC_RE.findall(text or "")))


@register("glaive_tools")
def glaive_tools(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split=split)
    out = 0
    for row in ds:
        if limit is not None and out >= limit:
            break
        tools = _tool_names(row["system"])
        chat = row["chat"]
        m = re.search(r"<functioncall>\s*(\{.*?\})\s*<\|endoftext\|>", chat, re.S)
        nm = _FUNC_RE.search(m.group(1)) if m else None
        if not tools or nm is None:
            continue
        name = nm.group(1)
        if name not in tools or len(tools) < 2:
            continue
        user = re.search(r"USER:\s*(.*?)(?:\n\n|$)", chat, re.S)
        if not user:
            continue
        out += 1
        yield simple_choice(
            task="glaive_tools",
            state=user.group(1).strip(),
            question_text="Which function should be called for this request?",
            labels=tools,
            answer=name,
        )


@register("hermes_tools")
def hermes_tools(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("NousResearch/hermes-function-calling-v1", split=split)
    out = 0
    for row in ds:
        if limit is not None and out >= limit:
            break
        try:
            tools = [t["function"]["name"] for t in json.loads(row["tools"])]
        except Exception:  # noqa: BLE001
            continue
        convo = row["conversations"]
        user_msg = next((c["value"] for c in convo if c.get("from") == "human"), None)
        call = None
        for c in convo:
            if c.get("from") == "gpt" and ("<tool_call>" in c.get("value", "") or "tool_call" in c.get("value", "")):
                m = re.search(r"\{.*\}", c["value"], re.S)
                if m:
                    try:
                        call = json.loads(m.group(0))["name"]
                    except Exception:  # noqa: BLE001
                        pass
                break
        if not (user_msg and call and call in tools and len(tools) >= 2):
            continue
        out += 1
        yield simple_choice(
            task="hermes_tools",
            state=user_msg,
            question_text="Which function should be called for this request?",
            labels=sorted(set(tools)),
            answer=call,
        )


@register("toolace")
def toolace(split: str = "train", limit: int | None = None) -> Iterator[DecisionExample]:
    from datasets import load_dataset

    ds = load_dataset("Team-ACE/ToolACE", split=split)
    out = 0
    for row in ds:
        if limit is not None and out >= limit:
            break
        tools = _tool_names(row["system"])
        convo = row["conversations"]
        user_msg = next((c["value"] for c in convo if c.get("from") == "user"), None)
        call = None
        for c in convo:
            if c.get("from") == "assistant":
                m = re.match(r"\[?([^\[(]+?)\(", c.get("value", ""))
                if m:
                    call = m.group(1).strip()
                    break
        if not (user_msg and call and call in tools and len(tools) >= 2):
            continue
        out += 1
        yield simple_choice(
            task="toolace",
            state=user_msg,
            question_text="Which function should be called for this request?",
            labels=tools,
            answer=call,
        )

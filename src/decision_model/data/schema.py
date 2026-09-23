"""Unified decision-data schema.

One example = one state + N questions. A question carries runtime-defined
options and the answer(s). Task types:

- choice: single answer, probabilities from a masked softmax over options
- score:  ordinal levels, every level judged independently (sigmoid)
- noul:   subset/multi-label, every option judged independently (sigmoid)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal

TaskType = Literal["choice", "score", "noul"]
TASK_TYPES: tuple[str, ...] = ("choice", "score", "noul")


@dataclass
class Option:
    id: str
    text: str
    description: str | None = None

    def render(self) -> str:
        if self.description:
            return f"{self.text}: {self.description}"
        return self.text


@dataclass
class Question:
    options: list[Option]
    answer_ids: list[str]
    text: str = ""
    type: TaskType = "choice"
    probs: list[float] | None = None  # soft targets (belief tasks), aligned with options

    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]

    def answer_set(self) -> set[str]:
        return set(self.answer_ids)

    def index_of(self, option_id: str) -> int:
        for i, o in enumerate(self.options):
            if o.id == option_id:
                return i
        raise KeyError(f"option {option_id!r} not in {self.option_ids()}")


@dataclass
class DecisionExample:
    task: str
    state: str
    questions: list[Question]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionExample":
        questions = []
        for q in d["questions"]:
            opts = [Option(**o) for o in q["options"]]
            questions.append(
                Question(
                    options=opts,
                    answer_ids=list(q["answer_ids"]),
                    text=q.get("text", ""),
                    type=q.get("type", "choice"),
                    probs=q.get("probs"),
                )
            )
        return cls(
            task=d["task"],
            state=d["state"],
            questions=questions,
            meta=d.get("meta", {}),
        )

    @classmethod
    def from_json(cls, line: str) -> "DecisionExample":
        return cls.from_dict(json.loads(line))


def validate(example: DecisionExample) -> list[str]:
    errors: list[str] = []
    if not example.task:
        errors.append("empty task")
    if not example.state:
        errors.append("empty state")
    if not example.questions:
        errors.append("no questions")
    for i, q in enumerate(example.questions):
        tag = f"question[{i}]"
        if q.type not in TASK_TYPES:
            errors.append(f"{tag}: bad type {q.type!r}")
        if not q.options:
            errors.append(f"{tag}: no options")
        ids = q.option_ids()
        if len(ids) != len(set(ids)):
            errors.append(f"{tag}: duplicate option ids")
        if not q.answer_ids:
            errors.append(f"{tag}: no answer")
        for a in q.answer_ids:
            if a not in ids:
                errors.append(f"{tag}: answer {a!r} not in options")
        if q.type in ("choice", "score") and len(q.answer_ids) != 1:
            errors.append(f"{tag}: {q.type} expects exactly one answer")
        if q.type == "noul" and len(q.options) < 2:
            errors.append(f"{tag}: noul needs at least 2 options")
        if q.probs is not None:
            if len(q.probs) != len(q.options):
                errors.append(f"{tag}: probs length != options")
            elif abs(sum(q.probs) - 1.0) > 1e-3:
                errors.append(f"{tag}: probs must sum to 1 (got {sum(q.probs):.3f})")
            elif any(p < 0 for p in q.probs):
                errors.append(f"{tag}: negative prob")
    return errors


def read_jsonl(path: str) -> Iterator[DecisionExample]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield DecisionExample.from_json(line)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no}: {e}") from e


def write_jsonl(path: str, examples: Iterator[DecisionExample]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")
            n += 1
    return n

"""Collators for Route B (explicit scorer) and Route A (LM-head readout)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .schema import DecisionExample

TASK_TYPE_IDS = {"choice": 0, "score": 1, "noul": 2}

LETTERS = [chr(ord("A") + i) for i in range(26)]
for _a in range(26):
    for _b in range(26):
        LETTERS.append(chr(ord("A") + _a) + chr(ord("A") + _b))
LETTERS = LETTERS[:255]


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    width = max((len(s) for s in seqs), default=0)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


@dataclass
class RouteBBatch:
    state_input_ids: Tensor
    state_mask: Tensor
    question_input_ids: Tensor
    question_mask: Tensor
    option_input_ids: Tensor
    option_mask: Tensor
    option_valid: Tensor
    answer_mask: Tensor
    answer_index: Tensor
    task_type: Tensor
    tasks: list[str]
    task_types: list[str]
    soft_targets: Tensor | None = None
    has_soft: Tensor | None = None
    defer: Tensor | None = None


class RouteBCollator:
    """State / question / candidates tokenized independently (Siamese roles)."""

    def __init__(
        self,
        tokenizer,
        max_state_tokens: int = 8192,
        max_question_tokens: int = 128,
        max_option_tokens: int = 64,
        question_per_example: int = 1,
    ) -> None:
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_question_tokens = max_question_tokens
        self.max_option_tokens = max_option_tokens
        self.qpe = question_per_example

    def _enc(self, text: str, max_len: int) -> list[int]:
        ids = self.tok(text, truncation=True, max_length=max_len)["input_ids"]
        return ids

    def __call__(self, examples: list[DecisionExample]) -> RouteBBatch:
        pad_id = self.tok.pad_token_id
        b = len(examples)
        questions = [ex.questions[: self.qpe] for ex in examples]
        q = max(len(qs) for qs in questions)
        k = max(len(q.options) for qs in questions for q in qs)

        state_ids = [self._enc(ex.state, self.max_state_tokens) for ex in examples]
        question_ids: list[list[list[int]]] = []
        option_ids: list[list[list[list[int]]]] = []
        option_valid = torch.zeros((b, q, k), dtype=torch.bool)
        answer_mask = torch.zeros((b, q, k), dtype=torch.float)
        answer_index = torch.full((b, q), -1, dtype=torch.long)
        task_type = torch.zeros(b, dtype=torch.long)
        task_types: list[str] = []
        soft_targets = torch.zeros((b, q, k), dtype=torch.float)
        has_soft = torch.zeros(b, dtype=torch.bool)
        defer = torch.zeros(b, dtype=torch.bool)

        for i, qs in enumerate(questions):
            task_type[i] = TASK_TYPE_IDS[qs[0].type]
            task_types.append(qs[0].type)
            row_q, row_o = [], []
            for j in range(q):
                if j < len(qs):
                    question = qs[j]
                    row_q.append(self._enc(question.text, self.max_question_tokens))
                    row_o.append(
                        [self._enc(o.render(), self.max_option_tokens) for o in question.options]
                    )
                    gold = question.answer_set()
                    for m, o in enumerate(question.options):
                        option_valid[i, j, m] = True
                        if o.id in gold:
                            answer_mask[i, j, m] = 1.0
                            if question.type in ("choice", "score") and len(gold) == 1:
                                answer_index[i, j] = m
                    if question.defer:
                        defer[i] = True
                    if question.probs is not None:
                        has_soft[i] = True
                        for m, pv in enumerate(question.probs):
                            soft_targets[i, j, m] = float(pv)
                else:
                    row_q.append(self._enc("", self.max_question_tokens))
                    row_o.append([])
            question_ids.append(row_q)
            option_ids.append(row_o)

        flat_q = [question_ids[i][j] for i in range(b) for j in range(q)]
        q_ids, q_mask = _pad(flat_q, pad_id)
        question_input_ids = q_ids.view(b, q, -1)
        question_mask = q_mask.view(b, q, -1)

        flat_o: list[list[int]] = []
        for i in range(b):
            for j in range(q):
                row = option_ids[i][j]
                for m in range(k):
                    flat_o.append(row[m] if m < len(row) else [])
        o_ids, o_mask = _pad(flat_o, pad_id)
        option_input_ids = o_ids.view(b, q, k, -1)
        option_mask = o_mask.view(b, q, k, -1)

        s_ids, s_mask = _pad(state_ids, pad_id)
        return RouteBBatch(
            state_input_ids=s_ids,
            state_mask=s_mask,
            question_input_ids=question_input_ids,
            question_mask=question_mask,
            option_input_ids=option_input_ids,
            option_mask=option_mask,
            option_valid=option_valid,
            answer_mask=answer_mask,
            answer_index=answer_index,
            task_type=task_type,
            tasks=[ex.task for ex in examples],
            task_types=task_types,
            soft_targets=soft_targets,
            has_soft=has_soft,
            defer=defer,
        )


@dataclass
class RouteABatch:
    input_ids: Tensor
    attention_mask: Tensor
    slot_positions: Tensor
    option_token_ids: Tensor
    option_valid: Tensor
    answer_mask: Tensor
    answer_index: Tensor
    task_type: Tensor
    tasks: list[str]
    task_types: list[str]


def label_token_ids(tokenizer) -> list[int]:
    ids = []
    for label in LETTERS:
        enc = tokenizer.encode(label, add_special_tokens=False)
        ids.append(enc[0])
    return ids


class RouteACollator:
    """Prompt with lettered options and an answer slot per question."""

    def __init__(
        self,
        tokenizer,
        max_state_tokens: int = 4096,
        max_question_tokens: int = 128,
        max_option_tokens: int = 48,
        max_total_tokens: int = 8192,
    ) -> None:
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_question_tokens = max_question_tokens
        self.max_option_tokens = max_option_tokens
        self.max_total_tokens = max_total_tokens
        self.option_token_ids = label_token_ids(tokenizer)

    def build_prompt(self, ex: DecisionExample, qps: int = 1) -> str:
        state = self.tok.decode(
            self.tok.encode(ex.state, truncation=True, max_length=self.max_state_tokens)
        )
        parts = [f"Context:\n{state}\n"]
        for j, q in enumerate(ex.questions[:qps]):
            q_text = f"Question:\n{q.text}\n" if q.text else ""
            opts = "\n".join(
                f"({LETTERS[m]}) {o.render()}" for m, o in enumerate(q.options)
            )
            parts.append(f"{q_text}Options:\n{opts}\nAnswer: (")
        return "\n".join(parts)

    def __call__(self, examples: list[DecisionExample]) -> RouteABatch:
        pad_id = self.tok.pad_token_id
        b = len(examples)
        q = max(len(ex.questions[:1]) for ex in examples)
        k = max(len(ex.questions[0].options) for ex in examples)

        seqs: list[list[int]] = []
        slots: list[int] = []
        option_valid = torch.zeros((b, q, k), dtype=torch.bool)
        answer_mask = torch.zeros((b, q, k), dtype=torch.float)
        answer_index = torch.full((b, q), -1, dtype=torch.long)
        option_token_ids = torch.zeros((b, q, k), dtype=torch.long)
        task_type = torch.zeros(b, dtype=torch.long)
        task_types: list[str] = []

        for i, ex in enumerate(examples):
            prompt = self.build_prompt(ex, qps=q)
            ids = self.tok(prompt, truncation=True, max_length=self.max_total_tokens)[
                "input_ids"
            ]
            seqs.append(ids)
            slots.append(len(ids) - 1)
            question = ex.questions[0]
            task_type[i] = TASK_TYPE_IDS[question.type]
            task_types.append(question.type)
            gold = question.answer_set()
            for m, o in enumerate(question.options):
                option_valid[i, 0, m] = True
                option_token_ids[i, 0, m] = self.option_token_ids[m]
                if o.id in gold:
                    answer_mask[i, 0, m] = 1.0
                    if question.type in ("choice", "score") and len(gold) == 1:
                        answer_index[i, 0] = m

        input_ids, attention_mask = _pad(seqs, pad_id)
        return RouteABatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            slot_positions=torch.tensor(slots, dtype=torch.long),
            option_token_ids=option_token_ids,
            option_valid=option_valid,
            answer_mask=answer_mask,
            answer_index=answer_index,
            task_type=task_type,
            tasks=[ex.task for ex in examples],
            task_types=task_types,
        )

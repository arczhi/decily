"""Two-stage inference for large candidate sets (up to 255 options).

Stage 1: score every candidate independently in chunks (logits are
set-independent for the cross-encoder, so chunking changes nothing).
Stage 2: keep the top-K candidates and normalize with a masked softmax.

This matches the Jev/TypeSafe recipe: independent scoring first, then an
explicit choice among a shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..data.schema import DecisionExample, Option, Question


@dataclass
class Decision:
    options: list[str]
    probs: list[float]
    logits: list[float]
    independent: list[float]
    top_k: int
    all_scored: int


def _to_device(batch, device: str):
    for field in batch.__dataclass_fields__:
        v = getattr(batch, field)
        if isinstance(v, torch.Tensor):
            setattr(batch, field, v.to(device))
    return batch


@torch.no_grad()
def score_candidates(
    model,
    collator,
    state: str,
    question_text: str,
    options: list[Option],
    chunk_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    if not options:
        raise ValueError("no options")
    rows = []
    for i in range(0, len(options), chunk_size):
        chunk = options[i : i + chunk_size]
        ex = DecisionExample(
            task="_inference",
            state=state,
            questions=[
                Question(
                    text=question_text,
                    options=chunk,
                    answer_ids=[chunk[0].id],
                    type="choice",
                )
            ],
        )
        batch = _to_device(collator([ex]), device)
        logits = model(batch)["logits"][0, 0][: len(chunk)].float().cpu()
        rows.append(logits)
    return torch.cat(rows)


def decide(
    model,
    collator,
    state: str,
    question_text: str,
    options: list[Option],
    top_k: int = 8,
    temperature: float = 1.0,
    chunk_size: int = 32,
    device: str = "cuda",
) -> Decision:
    logits = score_candidates(
        model, collator, state, question_text, options, chunk_size, device
    )
    k = min(max(top_k, 1), len(options))
    top = torch.topk(logits, k)
    probs = F.softmax(top.values / temperature, dim=-1)
    return Decision(
        options=[options[i].id for i in top.indices.tolist()],
        probs=probs.tolist(),
        logits=top.values.tolist(),
        independent=torch.sigmoid(logits / temperature).tolist(),
        top_k=k,
        all_scored=len(options),
    )

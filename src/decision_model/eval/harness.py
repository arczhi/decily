"""Evaluation harness: logits -> probabilities -> metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from ..data.schema import DecisionExample
from .metrics import MetricsAccumulator


def probs_from_logits(
    logits: Tensor, valid: Tensor, task_types: list[str], temperature: float = 1.0
) -> np.ndarray:
    """[N, K] probabilities; masked softmax for choice/score, sigmoid for noul."""
    p = torch.zeros_like(logits, dtype=torch.float64)
    for i, t in enumerate(task_types):
        row = logits[i][valid[i]] / temperature
        if t in ("choice", "score"):
            p[i][valid[i]] = torch.softmax(row, dim=-1).double()
        else:
            p[i][valid[i]] = torch.sigmoid(row).double()
    return p.numpy()


def targets_from_batch(answer_mask: Tensor, valid: Tensor) -> np.ndarray:
    y = answer_mask.clone().float()
    y[~valid] = 0
    return y.numpy()


@torch.no_grad()
def evaluate(
    model,
    examples: Iterable[DecisionExample],
    collator,
    temperature: float = 1.0,
    batch_size: int = 32,
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    acc = MetricsAccumulator()
    batch_examples: list[DecisionExample] = []
    n_batches = 0

    def flush(batch_examples: list[DecisionExample]) -> None:
        nonlocal n_batches
        if not batch_examples:
            return
        batch = collator(batch_examples)
        batch = _to_device(batch, device)
        out = model(batch)
        logits = out["logits"][:, 0]  # Q=1
        valid = batch.option_valid[:, 0]
        task_types = batch.task_types
        probs = probs_from_logits(logits.float().cpu(), valid.cpu(), task_types, temperature)
        targets = targets_from_batch(batch.answer_mask[:, 0].cpu(), valid.cpu())
        valid_np = valid.cpu().numpy()
        for i, t in enumerate(task_types):
            acc.add(probs[i : i + 1], targets[i : i + 1], t, valid_np[i : i + 1])
        n_batches += 1

    for ex in examples:
        batch_examples.append(ex)
        if len(batch_examples) >= batch_size:
            flush(batch_examples)
            batch_examples = []
            if max_batches is not None and n_batches >= max_batches:
                break
    if max_batches is None or n_batches < max_batches:
        flush(batch_examples)
    return acc.compute()


def _to_device(batch, device: str):
    for field in batch.__dataclass_fields__:
        v = getattr(batch, field)
        if isinstance(v, Tensor):
            setattr(batch, field, v.to(device))
    return batch


def fit_temperature(
    model,
    examples: Iterable[DecisionExample],
    collator,
    batch_size: int = 32,
    device: str = "cuda",
    grid: tuple[float, ...] = tuple(0.5 + 0.03 * i for i in range(61)),
) -> tuple[float, float]:
    """Grid-search temperature by NLL on the given examples (single-answer rows)."""
    examples = list(examples)
    logit_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            batch = _to_device(collator(chunk), device)
            out = model(batch)
            logits = out["logits"][:, 0].float().cpu()
            valid = batch.option_valid[:, 0].cpu()
            y = targets_from_batch(batch.answer_mask[:, 0].cpu(), valid)
            for j, t in enumerate(batch.task_types):
                if t == "noul":
                    continue
                logit_rows.append(logits[j][valid[j]].numpy())
                target_rows.append(y[j][valid[j]])
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        total, n = 0.0, 0
        for logits, y in zip(logit_rows, target_rows):
            z = torch.tensor(logits, dtype=torch.float64) / t
            p = torch.softmax(z, dim=-1)
            gold = int(np.argmax(y))
            total += -float(np.log(max(float(p[gold]), 1e-12)))
            n += 1
        nll = total / max(n, 1)
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t, best_nll

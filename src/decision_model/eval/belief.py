"""Belief calibration metrics against known probability laws (RLCD belief goal).

Headline metric: log score minus the entropy of the true law ("nats above the
law"), i.e. how much worse the model is than the irreducible uncertainty.
Also: total-variation error and a binned reliability ECE over option probs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..data.schema import DecisionExample

EPS = 1e-12


def belief_scores(pred: np.ndarray, true: np.ndarray, n_bins: int = 15) -> dict:
    """pred / true: [N, K] probability distributions over options."""
    pred = np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0)
    true = np.asarray(true, dtype=np.float64)
    log_score = float(-(true * np.log(pred)).sum(-1).mean())
    law_entropy = float(-(true * np.log(np.clip(true, EPS, 1.0))).sum(-1).mean())
    tv = 0.5 * np.abs(pred - true).sum(-1)
    # reliability over all (example, option) probability pairs
    p = pred.ravel()
    t = true.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        n = int(m.sum())
        if n:
            ece += (n / total) * abs(p[m].mean() - t[m].mean())
    return {
        "log_score": log_score,
        "law_entropy": law_entropy,
        "excess_nats": log_score - law_entropy,
        "tv_mae": float(tv.mean()),
        "belief_ece": float(ece),
        "n": float(len(pred)),
    }


@torch.no_grad()
def evaluate_belief(
    model,
    collator,
    examples: Iterable[DecisionExample],
    batch_size: int = 16,
    device: str = "cuda",
    temperature: float = 1.0,
) -> dict:
    preds, trues = [], []
    chunk: list[DecisionExample] = []

    def flush(rows: list[DecisionExample]) -> None:
        if not rows:
            return
        batch = collator(rows)
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if isinstance(v, torch.Tensor):
                setattr(batch, f, v.to(device))
        logits = model(batch)["logits"][:, 0].float().cpu()
        valid = batch.option_valid[:, 0].cpu()
        soft = batch.soft_targets[:, 0].cpu()
        for i in range(len(rows)):
            v = valid[i]
            z = logits[i][v] / max(temperature, 1e-6)
            preds.append(torch.softmax(z, dim=-1).numpy())
            trues.append(soft[i][v].numpy())

    for ex in examples:
        chunk.append(ex)
        if len(chunk) >= batch_size:
            flush(chunk)
            chunk = []
    flush(chunk)
    return belief_scores(np.stack(preds), np.stack(trues))

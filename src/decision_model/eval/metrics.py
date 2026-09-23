"""Decision-model metrics: acc / NLL / Brier / ECE / selective accuracy / AURC.

Pure numpy so it can be unit-tested without torch. Rows may have different
numbers of options; everything works on padded arrays plus a validity mask.

Conventions
-----------
probs:   [N, K] float, probabilities over the K padded options
targets: [N, K] float, one-hot (choice/score) or multi-hot (noul); padding = 0
valid:   [N, K] bool, True for real options (padding = False)
types:   list of task types ("choice" | "score" | "noul"), one per row
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-12


def _masked_argmax(probs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, probs, -np.inf)
    return masked.argmax(axis=1)


def _default_valid(targets: np.ndarray) -> np.ndarray:
    return np.ones_like(targets, dtype=bool)


def accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    types: list[str],
    valid: np.ndarray | None = None,
) -> float:
    if valid is None:
        valid = _default_valid(targets)
    hits, total = 0.0, 0
    pred_idx = _masked_argmax(probs, valid)
    gold_idx = _masked_argmax(targets, valid)
    for i, t in enumerate(types):
        if t == "noul":
            pred = (probs[i] >= 0.5) & valid[i]
            gold = (targets[i] > 0) & valid[i]
            hits += float(np.array_equal(pred, gold))
        else:
            hits += float(pred_idx[i] == gold_idx[i])
        total += 1
    return hits / max(total, 1)


def nll(
    probs: np.ndarray,
    targets: np.ndarray,
    task_type: str,
    valid: np.ndarray | None = None,
) -> float:
    if valid is None:
        valid = _default_valid(targets)
    p = np.clip(probs, EPS, 1.0)
    if task_type == "noul":
        bce = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
        return float(bce[valid].mean())
    gold = _masked_argmax(targets, valid)
    return float(-np.log(p[np.arange(p.shape[0]), gold]).mean())


def brier(
    probs: np.ndarray, targets: np.ndarray, valid: np.ndarray | None = None
) -> float:
    if valid is None:
        valid = _default_valid(targets)
    sq = (probs - targets) ** 2
    return float(sq[valid].mean())


def ece(
    probs: np.ndarray,
    targets: np.ndarray,
    task_type: str,
    valid: np.ndarray | None = None,
    n_bins: int = 15,
) -> float:
    if valid is None:
        valid = _default_valid(targets)
    if task_type == "noul":
        mask = valid.ravel()
        p = probs.ravel()[mask]
        y = targets.ravel()[mask] > 0
        conf = np.where(p >= 0.5, p, 1.0 - p)
        correct = (p >= 0.5) == y
        conf = conf.astype(np.float64)
        correct = correct.astype(np.float64)
    else:
        conf = np.where(valid, probs, -np.inf).max(axis=1)
        correct = (
            _masked_argmax(probs, valid) == _masked_argmax(targets, valid)
        ).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    if total == 0:
        return float("nan")
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        err += (n / total) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(err)


def selective_acc(conf: np.ndarray, correct: np.ndarray, coverage: float = 0.8) -> float:
    n = len(conf)
    k = max(int(np.ceil(coverage * n)), 1)
    order = np.argsort(-conf, kind="stable")[:k]
    return float(correct[order].mean())


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf, kind="stable")
    risk = 1.0 - correct[order]
    n = len(risk)
    cum = np.cumsum(risk) / (np.arange(n) + 1)
    if n == 1:
        return float(cum[0] / 2.0)
    area = 0.0
    for i in range(1, n):
        area += (cum[i - 1] + cum[i]) / 2.0 * (1.0 / n)
    return float(area)


@dataclass
class MetricsAccumulator:
    n_bins: int = 15
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        task_type: str,
        valid: np.ndarray | None = None,
    ) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if valid is None:
            valid = np.ones_like(targets, dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        assert probs.shape == targets.shape == valid.shape, (
            probs.shape,
            targets.shape,
            valid.shape,
        )
        bucket = self.rows.setdefault(task_type, [])
        for i in range(probs.shape[0]):
            bucket.append({"p": probs[i], "y": targets[i], "v": valid[i]})

    def _stack(self, task_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.rows[task_type]
        k = max(len(r["p"]) for r in rows)
        n = len(rows)
        p = np.zeros((n, k), dtype=np.float64)
        y = np.zeros((n, k), dtype=np.float64)
        v = np.zeros((n, k), dtype=bool)
        for i, r in enumerate(rows):
            w = len(r["p"])
            p[i, :w] = r["p"]
            y[i, :w] = r["y"]
            v[i, :w] = r["v"]
        return p, y, v

    def compute(self) -> dict[str, float]:
        if not self.rows:
            return {}
        out: dict[str, float] = {}
        all_p, all_y, all_v, all_types = [], [], [], []
        for task_type, rows in self.rows.items():
            p, y, v = self._stack(task_type)
            out.update(self._per_type(p, y, v, task_type))
            all_p.append(p)
            all_y.append(y)
            all_v.append(v)
            all_types.extend([task_type] * p.shape[0])

        k = max(a.shape[1] for a in all_p)
        def pad(a, dtype):
            out_a = np.zeros((sum(x.shape[0] for x in a), k), dtype=dtype)
            i = 0
            for x in a:
                out_a[i : i + x.shape[0], : x.shape[1]] = x
                i += x.shape[0]
            return out_a

        P = pad(all_p, np.float64)
        Y = pad(all_y, np.float64)
        V = pad(all_v, bool)
        out["n"] = float(P.shape[0])
        out["acc_all"] = float(accuracy(P, Y, all_types, V))
        return out

    def _per_type(
        self, p: np.ndarray, y: np.ndarray, v: np.ndarray, task_type: str
    ) -> dict[str, float]:
        prefix = task_type
        types = [task_type] * p.shape[0]
        out = {
            f"{prefix}/acc": float(accuracy(p, y, types, v)),
            f"{prefix}/nll": float(nll(p, y, task_type, v)),
            f"{prefix}/brier": float(brier(p, y, v)),
            f"{prefix}/ece": float(ece(p, y, task_type, v, self.n_bins)),
        }
        if task_type in ("choice", "score"):
            conf = np.where(v, p, -np.inf).max(axis=1)
            correct = (
                _masked_argmax(p, v) == _masked_argmax(y, v)
            ).astype(np.float64)
            out[f"{prefix}/acc@80"] = float(selective_acc(conf, correct, 0.8))
            out[f"{prefix}/aurc"] = float(aurc(conf, correct))
            out[f"{prefix}/aurc_excess"] = float(
                aurc(conf, correct) - (1.0 - correct.mean())
            )
        return out

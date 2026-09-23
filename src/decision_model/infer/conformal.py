"""Selective risk control: threshold calibration for accepted decisions.

Note on theory: split-conformal classification bounds P(Y not in C(X)) (set
coverage), which does NOT directly bound the error rate among accepted
(singleton) predictions. The practically deployable target is a high-probability
bound on the *selective risk*:

    choose the most lenient threshold t on p_max such that
        risk_hat(t) + sqrt(log(1/delta) / (2 n_acc(t)))  <=  alpha
    (Hoeffding upper confidence bound on P(error | accepted))

which is the standard PAC-style selective classification guarantee
(Geifman & El-Yaniv 2017; Angelopoulos & Bates 2023). Calibrated once on a
held-out split; reused at inference.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_lambda(p_gold_calib: np.ndarray, alpha: float = 0.05) -> float:
    """Threshold on 1 - p_true from a calibration set of true-label probabilities."""
    scores = np.sort(1.0 - np.asarray(p_gold_calib, dtype=np.float64))
    n = len(scores)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def selective_metrics(
    p_max: np.ndarray,
    correct: np.ndarray,
    lam: float,
) -> dict:
    """Coverage and selective risk for the acceptance rule p_max >= 1 - lam."""
    accept = np.asarray(p_max) >= 1.0 - lam
    n = len(accept)
    cov = float(accept.mean()) if n else 0.0
    if accept.sum() == 0:
        return {"coverage": 0.0, "selective_error": float("nan"), "n_accepted": 0}
    err = float(1.0 - np.asarray(correct)[accept].mean())
    return {"coverage": cov, "selective_error": err, "n_accepted": int(accept.sum())}


def risk_coverage_curve(
    p_max: np.ndarray, correct: np.ndarray, n_points: int = 40
) -> dict:
    """Standard risk-coverage curve over confidence thresholds."""
    p_max = np.asarray(p_max)
    correct = np.asarray(correct)
    order = np.argsort(-p_max, kind="stable")
    p_sorted, c_sorted = p_max[order], correct[order]
    n = len(p_sorted)
    idx = np.linspace(max(int(n * 0.05), 1), n, n_points).astype(int)
    coverages = idx / n
    risks = 1.0 - np.cumsum(c_sorted)[idx - 1] / idx
    return {"coverage": coverages.tolist(), "risk": risks.tolist()}


def selective_threshold(
    p_max: np.ndarray,
    correct: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.05,
    n_grid: int = 200,
) -> float:
    """Most lenient threshold t on p_max with Hoeffding-bounded selective risk <= alpha."""
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    grid = np.unique(np.quantile(p_max, np.linspace(0.0, 0.995, n_grid)))
    best_t = 1.01
    for t in np.sort(grid):
        acc = p_max >= t
        n_acc = int(acc.sum())
        if n_acc < 10:
            continue
        risk = 1.0 - correct[acc].mean()
        bound = risk + math.sqrt(math.log(1.0 / max(delta, 1e-6)) / (2.0 * n_acc))
        if bound <= alpha:
            best_t = float(t)
            break
    return best_t


def threshold_report(
    p_max_calib: np.ndarray,
    correct_calib: np.ndarray,
    p_max_test: np.ndarray,
    correct_test: np.ndarray,
    alphas: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
) -> dict:
    out = {}
    for alpha in alphas:
        t = selective_threshold(p_max_calib, correct_calib, alpha)
        out[f"alpha={alpha}"] = {
            "threshold": t,
            **selective_metrics(p_max_test, correct_test, 1.0 - t),
        }
    return out

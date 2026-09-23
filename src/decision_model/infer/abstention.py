"""Post-hoc threshold calibration for the "none of the above" option.

Sweeps a bias added to the none logit and picks the operating point that
maximizes correct abstention (gold absent) subject to a false-abstention
budget (gold present). No retraining required.

Requires the none option to be the LAST valid option (build eval sets with
`add_abstention(..., shuffle=False)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AbstentionCurve:
    biases: list[float]
    correct_rates: list[float]
    false_rates: list[float]

    def pick(self, target_false: float = 0.05) -> dict:
        best = None
        for b, corr, false in zip(self.biases, self.correct_rates, self.false_rates):
            if false <= target_false and (best is None or corr > best["correct_rate"]):
                best = {
                    "bias": b,
                    "correct_rate": corr,
                    "false_rate": false,
                }
        if best is None:
            # budget unreachable: take the lowest false rate
            i = int(np.argmin(self.false_rates))
            best = {
                "bias": self.biases[i],
                "correct_rate": self.correct_rates[i],
                "false_rate": self.false_rates[i],
            }
        return best


def _argmax_with_bias(logits: np.ndarray, bias: float) -> int:
    lg = logits.copy()
    lg[-1] += bias
    return int(np.argmax(lg))


def abstention_curve(
    present_logits: list[np.ndarray],
    absent_logits: list[np.ndarray],
    biases: np.ndarray | None = None,
) -> AbstentionCurve:
    """present: gold present + none distractor; absent: gold dropped, none correct."""
    if biases is None:
        biases = np.linspace(-6.0, 6.0, 121)
    correct, false = [], []
    for b in biases:
        c = np.mean([_argmax_with_bias(lg, b) == len(lg) - 1 for lg in absent_logits])
        f = np.mean([_argmax_with_bias(lg, b) == len(lg) - 1 for lg in present_logits])
        correct.append(float(c))
        false.append(float(f))
    return AbstentionCurve([float(b) for b in biases], correct, false)


@torch.no_grad()
def collect_none_logits(model, collator, examples, batch_size: int = 16, device: str = "cuda"):
    """Logits rows where the none option is last (built with shuffle=False)."""
    from ..eval.harness import collect_logits

    rows = collect_logits(
        model, examples, collator, batch_size=batch_size, device=device
    )
    return rows.logits

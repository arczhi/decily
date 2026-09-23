import numpy as np

from decision_model.eval.metrics import (
    MetricsAccumulator,
    accuracy,
    aurc,
    brier,
    ece,
    nll,
    selective_acc,
)


def test_perfect_predictions():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert accuracy(p, y, ["choice", "choice"]) == 1.0
    assert nll(p, y, "choice") == 0.0
    assert brier(p, y) == 0.0
    assert ece(p, y, "choice", n_bins=15) == 0.0


def test_uniform_choice_values():
    p = np.array([[0.5, 0.5]])
    y = np.array([[1.0, 0.0]])
    assert accuracy(p, y, ["choice"]) == 1.0
    assert abs(nll(p, y, "choice") - np.log(2)) < 1e-9
    assert abs(brier(p, y) - 0.25) < 1e-9
    assert abs(ece(p, y, "choice", n_bins=15) - 0.5) < 1e-9


def test_wrong_confident():
    p = np.array([[0.9, 0.1]])
    y = np.array([[0.0, 1.0]])
    assert accuracy(p, y, ["choice"]) == 0.0
    assert abs(nll(p, y, "choice") + np.log(0.1)) < 1e-9
    assert abs(ece(p, y, "choice", n_bins=15) - 0.9) < 1e-9


def test_aurc_hand_computed():
    conf = np.array([0.9, 0.5])
    correct = np.array([1.0, 0.0])
    assert abs(aurc(conf, correct) - 0.125) < 1e-9


def test_selective_acc():
    conf = np.array([0.9, 0.8, 0.2])
    correct = np.array([1.0, 1.0, 0.0])
    assert selective_acc(conf, correct, coverage=2 / 3) == 1.0


def test_noul_bce():
    p = np.array([[0.9, 0.2, 0.6]])
    y = np.array([[1.0, 0.0, 1.0]])
    expected = -(
        np.log(0.9) + np.log(1 - 0.2) + np.log(0.6)
    ) / 3
    assert abs(nll(p, y, "noul") - expected) < 1e-9
    assert accuracy(p, y, ["noul"]) == 1.0  # thresholds: [1, 0, 1]
    y2 = np.array([[1.0, 1.0, 1.0]])
    assert accuracy(p, y2, ["noul"]) == 0.0


def test_per_type_acc_uses_all_rows():
    acc = MetricsAccumulator()
    acc.add(np.array([[0.9, 0.1]]), np.array([[1.0, 0.0]]), "choice")
    acc.add(np.array([[0.2, 0.8]]), np.array([[1.0, 0.0]]), "choice")
    out = acc.compute()
    assert out["choice/acc"] == 0.5
    assert out["acc_all"] == 0.5


def test_mixed_k_rows():
    acc = MetricsAccumulator()
    acc.add(np.array([[0.6, 0.4]]), np.array([[1.0, 0.0]]), "choice")
    acc.add(np.array([[0.2, 0.3, 0.5]]), np.array([[0.0, 0.0, 1.0]]), "choice")
    out = acc.compute()
    assert out["n"] == 2
    assert out["acc_all"] == 1.0
    assert out["choice/acc"] == 1.0


def test_valid_mask_ignores_padding():
    acc = MetricsAccumulator()
    probs = np.array([[0.7, 0.3, 0.0]])
    y = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[True, True, False]])
    acc.add(probs, y, "choice", v)
    out = acc.compute()
    assert out["choice/acc"] == 1.0
    expected_brier = (0.3**2 + 0.3**2) / 2
    assert abs(out["choice/brier"] - expected_brier) < 1e-9


def test_accumulator_shapes():
    acc = MetricsAccumulator()
    acc.add(np.array([[0.7, 0.3]]), np.array([[1.0, 0.0]]), "choice")
    acc.add(np.array([[0.2, 0.8]]), np.array([[0.0, 1.0]]), "choice")
    out = acc.compute()
    assert out["n"] == 2
    assert out["acc_all"] == 1.0
    assert "choice/nll" in out
    assert "choice/aurc" in out

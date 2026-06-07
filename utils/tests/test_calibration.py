"""Tests for calibration metrics: ECE and reliability diagram."""
from __future__ import annotations

import numpy as np
import pytest

from typo_utils.eval.calibration import expected_calibration_error, reliability_diagram


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated():
    """A perfectly calibrated model has ECE < 1e-6."""
    # Construct samples where, within every bin, the empirical accuracy equals
    # the mean confidence exactly.  We do this by using only confidence values
    # 0.0 and 1.0 (all in the two extreme bins).
    # Bin [0, 1/n_bins): 10 items with conf=0.0, all wrong  → acc=conf=0.0
    # Bin [last]:        10 items with conf=1.0, all correct → acc=conf=1.0
    n_bins = 10
    confidences = [0.0] * 10 + [1.0] * 10
    correct = [0] * 10 + [1] * 10
    ece = expected_calibration_error(confidences, correct, n_bins=n_bins)
    assert ece < 1e-6, f"Expected ECE < 1e-6 for perfectly calibrated, got {ece}"


def test_ece_hand_computed_two_bins():
    """Hand-computed 2-bin ECE example."""
    # Bin 0: [0, 0.5)  → 4 items, conf=0.3, acc=0.5  → |0.5 - 0.3| = 0.2
    # Bin 1: [0.5, 1]  → 6 items, conf=0.8, acc=2/6   → |1/3 - 0.8| = 0.4667
    # ECE = (4/10)*0.2 + (6/10)*0.4667 = 0.08 + 0.28 = 0.36
    confidences = [0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    correct     = [1,   0,   1,   0,   1,   0,   0,   0,   0,   1  ]
    # bin 0: 4 items, 2 correct, acc=0.5, mean_conf=0.3  -> diff=0.2
    # bin 1: 6 items, 2 correct, acc=1/3, mean_conf=0.8  -> diff=0.8-1/3
    expected = (4 / 10) * abs(0.5 - 0.3) + (6 / 10) * abs(2 / 6 - 0.8)
    ece = expected_calibration_error(confidences, correct, n_bins=2)
    assert ece == pytest.approx(expected, abs=1e-9)


def test_ece_all_correct_conf_one():
    """All predictions correct with confidence 1.0 -> ECE 0.0."""
    confidences = [1.0] * 20
    correct = [1] * 20
    ece = expected_calibration_error(confidences, correct)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_all_wrong_conf_one():
    """All predictions wrong but confidence 1.0 -> ECE 1.0."""
    confidences = [1.0] * 20
    correct = [0] * 20
    ece = expected_calibration_error(confidences, correct)
    assert ece == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


def test_reliability_diagram_lengths():
    """Returned arrays have length n_bins."""
    rng = np.random.default_rng(42)
    n = 50
    confidences = rng.random(n).tolist()
    correct = rng.integers(0, 2, n).tolist()
    n_bins = 15
    bin_acc, bin_conf, bin_count = reliability_diagram(confidences, correct, n_bins=n_bins)
    assert len(bin_acc) == n_bins
    assert len(bin_conf) == n_bins
    assert len(bin_count) == n_bins


def test_reliability_diagram_counts_sum_to_n():
    """bin_count must sum to N (total samples)."""
    rng = np.random.default_rng(7)
    n = 80
    confidences = rng.random(n).tolist()
    correct = rng.integers(0, 2, n).tolist()
    _, _, bin_count = reliability_diagram(confidences, correct, n_bins=15)
    assert int(np.sum(bin_count)) == n


def test_reliability_diagram_empty_bins_zero():
    """Empty bins must have bin_acc=0 and bin_conf=0."""
    # All confidences in [0.9, 1.0], so only last bin is populated.
    confidences = [0.95] * 10
    correct = [1] * 10
    n_bins = 10
    bin_acc, bin_conf, bin_count = reliability_diagram(confidences, correct, n_bins=n_bins)
    # bins 0..8 are empty -> acc and conf should be 0
    for i in range(n_bins - 1):
        assert bin_count[i] == 0
        assert bin_acc[i] == 0.0
        assert bin_conf[i] == 0.0
    # last bin has 10 items
    assert bin_count[n_bins - 1] == 10

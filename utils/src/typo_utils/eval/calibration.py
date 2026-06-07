"""Calibration metrics: Expected Calibration Error + reliability diagram."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _bin_stats(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin accuracy, mean confidence, and count.

    Equal-width bins over [0, 1].  Items with confidence == 1.0 are placed in
    the last bin.  Empty bins receive 0 for accuracy and confidence.

    Returns
    -------
    bin_acc : np.ndarray, shape (n_bins,)
    bin_conf : np.ndarray, shape (n_bins,)
    bin_count : np.ndarray, shape (n_bins,), dtype int64
    """
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    # Map confidence to bin index: floor(conf * n_bins), clipped to [0, n_bins-1]
    indices = np.floor(confidences * n_bins).astype(int)
    indices = np.clip(indices, 0, n_bins - 1)

    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        bin_count[b] = count
        if count > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()

    return bin_acc, bin_conf, bin_count


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 15
) -> float:
    """ECE from per-item confidence and 0/1 correctness.

    Equal-width bins over [0, 1] by confidence.
    ECE = sum_b (count_b / N) * |acc_b - conf_b|.

    Parameters
    ----------
    confidences : sequence of float in [0, 1]
    correct : sequence of int (0 or 1)
    n_bins : number of equal-width bins (default 15)

    Returns
    -------
    float
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0

    bin_acc, bin_conf, bin_count = _bin_stats(conf, corr, n_bins)
    ece: float = float(np.sum((bin_count / n) * np.abs(bin_acc - bin_conf)))
    return ece


def reliability_diagram(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(bin_acc, bin_conf, bin_count)`` for plotting.

    Equal-width bins over [0, 1].  Empty bins have ``bin_acc=0`` and
    ``bin_conf=0``.  ``bin_count`` sums to N (total samples).

    Parameters
    ----------
    confidences : sequence of float in [0, 1]
    correct : sequence of int (0 or 1)
    n_bins : number of equal-width bins (default 15)

    Returns
    -------
    bin_acc : np.ndarray, shape (n_bins,)
    bin_conf : np.ndarray, shape (n_bins,)
    bin_count : np.ndarray, shape (n_bins,), dtype int64
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    return _bin_stats(conf, corr, n_bins)

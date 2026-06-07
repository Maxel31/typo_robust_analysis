"""Calibration metrics: Expected Calibration Error + reliability diagram.

STATUS: stub. Implemented in feature/quant_typo_neuron/m2-ece-calibration.
"""
from __future__ import annotations

from typing import Any, Sequence


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 15
) -> float:
    """ECE from per-item confidence and 0/1 correctness. STATUS: stub."""
    raise NotImplementedError("implemented in feature/quant_typo_neuron/m2-ece-calibration")


def reliability_diagram(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 15
) -> Any:
    """Return ``(bin_acc, bin_conf, bin_count)`` for plotting. STATUS: stub."""
    raise NotImplementedError("implemented in feature/quant_typo_neuron/m2-ece-calibration")

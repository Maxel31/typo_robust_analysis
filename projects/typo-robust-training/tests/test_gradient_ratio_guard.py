"""The state-gradient hard stop protects calibration startup, not convergence."""

from __future__ import annotations

import math

from typo_robust_training.training.runtime import _finite_ppl_ratio, next_gradient_ratio_violations


def test_gradient_ratio_guard_counts_only_consecutive_startup_violations() -> None:
    violations = 0
    violations = next_gradient_ratio_violations(
        violations, ratio=0.8, optimizer_steps=10, guard_steps=50
    )
    violations = next_gradient_ratio_violations(
        violations, ratio=0.9, optimizer_steps=11, guard_steps=50
    )
    assert violations == 2

    assert (
        next_gradient_ratio_violations(violations, ratio=0.2, optimizer_steps=12, guard_steps=50)
        == 0
    )


def test_gradient_ratio_becomes_diagnostic_after_the_first_checkpoint() -> None:
    assert next_gradient_ratio_violations(2, ratio=4.0, optimizer_steps=50, guard_steps=50) == 0


def test_collapsed_model_ppl_ratio_stays_finite_for_the_safety_gate() -> None:
    ratio = _finite_ppl_ratio(1_000.0)
    assert math.isfinite(ratio)
    assert ratio > 1e300

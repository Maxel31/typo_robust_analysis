from __future__ import annotations

import pytest

from typo_robust_training.probe.scoring import (
    ProbeSeedTrajectory,
    select_probe_transition,
)


def _trajectory(
    seed: int,
    *,
    clean: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    noisy: tuple[float, ...],
) -> ProbeSeedTrajectory:
    return ProbeSeedTrajectory(
        seed=seed,
        clean_cross_entropy=clean,
        noisy_cross_entropy=noisy,
    )


def test_probe_transition_selects_largest_mean_noise_penalty_drop() -> None:
    result = select_probe_transition(
        (
            _trajectory(42, noisy=(2.0, 1.9, 1.3, 1.2)),
            _trajectory(43, noisy=(2.1, 2.0, 1.2, 1.1)),
        )
    )

    assert result.selected_layer == 2
    assert result.seed_selected_layers == ((42, 2), (43, 2))


def test_probe_transition_breaks_exact_ties_toward_shallower_layer() -> None:
    result = select_probe_transition(
        (
            _trajectory(42, noisy=(2.0, 1.5, 1.0, 1.0)),
            _trajectory(43, noisy=(2.0, 1.5, 1.0, 1.0)),
        )
    )

    assert result.selected_layer == 1


@pytest.mark.parametrize(
    "trajectories, message",
    [
        ((), "exactly two"),
        ((_trajectory(42, noisy=(2.0, 1.0, 1.0, 1.0)),), "exactly two"),
        (
            (
                _trajectory(42, noisy=(2.0, 1.0, 1.0, 1.0)),
                _trajectory(42, noisy=(2.0, 1.0, 1.0, 1.0)),
            ),
            "distinct",
        ),
    ],
)
def test_probe_transition_rejects_invalid_seed_inventory(
    trajectories: tuple[ProbeSeedTrajectory, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_probe_transition(trajectories)


def test_probe_transition_rejects_layer_inventory_mismatch() -> None:
    with pytest.raises(ValueError, match="same decoder-layer inventory"):
        select_probe_transition(
            (
                _trajectory(42, noisy=(2.0, 1.0, 1.0, 1.0)),
                _trajectory(
                    43,
                    clean=(1.0, 1.0, 1.0),
                    noisy=(2.0, 1.0, 1.0),
                ),
            )
        )


def test_probe_transition_uses_noise_penalty_not_raw_noisy_loss() -> None:
    result = select_probe_transition(
        (
            _trajectory(
                42,
                clean=(0.5, 0.1, 0.1, 0.1),
                noisy=(1.5, 1.0, 0.8, 0.7),
            ),
            _trajectory(
                43,
                clean=(0.5, 0.1, 0.1, 0.1),
                noisy=(1.5, 1.0, 0.8, 0.7),
            ),
        )
    )

    # Raw noisy loss drops most at layer 1, but the clean-adjusted penalty drops
    # most at layer 2. This test prevents silently selecting generic probe gain.
    assert result.selected_layer == 2


@pytest.mark.parametrize("invalid", [True, "1.0"])
def test_probe_trajectory_rejects_non_numeric_loss_types(invalid: object) -> None:
    with pytest.raises(ValueError, match="only JSON numbers"):
        ProbeSeedTrajectory(
            seed=42,
            clean_cross_entropy=(1.0, invalid),  # type: ignore[arg-type]
            noisy_cross_entropy=(2.0, 1.0),
        )

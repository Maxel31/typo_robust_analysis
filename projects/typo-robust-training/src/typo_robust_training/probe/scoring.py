"""Pure scoring rule for selecting a typo-denoising probe transition.

The selector deliberately consumes only held-out word-identity probe losses.
It does not inspect task accuracy, activation-patching outcomes, or downstream
adapter behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_losses(values: tuple[float, ...], *, field: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < 2:
        raise ValueError(f"{field} must contain at least two decoder-layer losses")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{field} must contain only JSON numbers")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{field} must contain finite non-negative losses")
    return result


@dataclass(frozen=True, slots=True)
class ProbeSeedTrajectory:
    """Clean/noisy word-identity cross-entropy by decoder layer for one seed."""

    seed: int
    clean_cross_entropy: tuple[float, ...]
    noisy_cross_entropy: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("probe seed must be a non-negative integer")
        clean = _finite_losses(self.clean_cross_entropy, field="clean cross-entropy")
        noisy = _finite_losses(self.noisy_cross_entropy, field="noisy cross-entropy")
        if len(clean) != len(noisy):
            raise ValueError("clean and noisy probe trajectories must have equal length")
        object.__setattr__(self, "clean_cross_entropy", clean)
        object.__setattr__(self, "noisy_cross_entropy", noisy)

    @property
    def decoder_layers(self) -> int:
        return len(self.clean_cross_entropy)

    @property
    def noise_penalty(self) -> tuple[float, ...]:
        return tuple(
            noisy - clean
            for clean, noisy in zip(
                self.clean_cross_entropy,
                self.noisy_cross_entropy,
                strict=True,
            )
        )

    @property
    def transition_drop(self) -> tuple[float, ...]:
        penalty = self.noise_penalty
        return tuple(penalty[layer - 1] - penalty[layer] for layer in range(1, len(penalty)))


@dataclass(frozen=True, slots=True)
class ProbeTransitionSelection:
    """Fully determined selection plus seed-level audit values."""

    selected_layer: int
    mean_noise_penalty: tuple[float, ...]
    mean_transition_drop: tuple[float, ...]
    seed_selected_layers: tuple[tuple[int, int], ...]


def _earliest_argmax(values: tuple[float, ...]) -> int:
    maximum = max(values)
    return next(index for index, value in enumerate(values) if value == maximum)


def select_probe_transition(
    trajectories: tuple[ProbeSeedTrajectory, ...],
) -> ProbeTransitionSelection:
    """Select the earliest largest reduction in the mean clean/noisy penalty.

    For decoder layer ``l >= 1`` the selection statistic is

    ``mean_seed[(CE_noisy[l-1] - CE_clean[l-1])
                - (CE_noisy[l] - CE_clean[l])]``.

    The offset returned by ``argmax`` is shifted by one because layer zero has
    no preceding decoder layer. Exact ties are resolved toward the shallower
    layer. Exactly two independently initialized probes are required by the
    frozen method contract.
    """

    if not isinstance(trajectories, tuple) or len(trajectories) != 2:
        raise ValueError("transition selection requires exactly two probe seeds")
    seeds = tuple(trajectory.seed for trajectory in trajectories)
    if len(set(seeds)) != len(seeds):
        raise ValueError("probe seeds must be distinct")
    decoder_layers = trajectories[0].decoder_layers
    if any(trajectory.decoder_layers != decoder_layers for trajectory in trajectories):
        raise ValueError("probe trajectories must use the same decoder-layer inventory")

    penalties_by_seed = tuple(trajectory.noise_penalty for trajectory in trajectories)
    mean_penalty = tuple(
        sum(penalties[layer] for penalties in penalties_by_seed) / len(penalties_by_seed)
        for layer in range(decoder_layers)
    )
    mean_drop = tuple(
        mean_penalty[layer - 1] - mean_penalty[layer] for layer in range(1, decoder_layers)
    )
    selected_layer = _earliest_argmax(mean_drop) + 1
    per_seed = tuple(
        (trajectory.seed, _earliest_argmax(trajectory.transition_drop) + 1)
        for trajectory in sorted(trajectories, key=lambda item: item.seed)
    )
    return ProbeTransitionSelection(
        selected_layer=selected_layer,
        mean_noise_penalty=mean_penalty,
        mean_transition_drop=mean_drop,
        seed_selected_layers=per_seed,
    )


__all__ = [
    "ProbeSeedTrajectory",
    "ProbeTransitionSelection",
    "select_probe_transition",
]

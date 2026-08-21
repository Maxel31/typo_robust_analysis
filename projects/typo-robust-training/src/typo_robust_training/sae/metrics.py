"""Streaming SAE statistics, energy score, and preregistered gates."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SaeStatistics:
    tokens: int
    fvu: float
    median_l0: float
    firing_probabilities: tuple[float, ...]
    dead_feature_rate: float
    reconstruction_error_median: float


class SaeStatisticsAccumulator:
    """Accumulate exact token statistics without retaining residual vectors."""

    def __init__(self, *, d_sae: int, dead_probability_below: float = 1e-5) -> None:
        import torch

        if isinstance(d_sae, bool) or not isinstance(d_sae, int) or d_sae <= 0:
            raise ValueError("SAE statistics d_sae must be positive")
        if not 0.0 < dead_probability_below < 1.0:
            raise ValueError("SAE dead-feature threshold must be in (0, 1)")
        self.d_sae = d_sae
        self.dead_probability_below = float(dead_probability_below)
        self._torch = torch
        self._tokens = 0
        self._input_mean = None
        self._input_m2 = 0.0
        self._error_sum = 0.0
        self._feature_counts = torch.zeros(d_sae, dtype=torch.int64)
        self._l0_counts = torch.zeros(d_sae + 1, dtype=torch.int64)
        self._errors = array("f")

    def update(self, inputs: Any, reconstruction: Any, features: Any) -> None:
        torch = self._torch
        if (
            not isinstance(inputs, torch.Tensor)
            or not isinstance(reconstruction, torch.Tensor)
            or not isinstance(features, torch.Tensor)
            or inputs.ndim != 2
            or reconstruction.shape != inputs.shape
            or features.ndim != 2
            or int(features.shape[0]) != int(inputs.shape[0])
            or int(features.shape[1]) != self.d_sae
        ):
            raise ValueError("SAE statistics tensor shapes differ")
        values = inputs.detach().to(dtype=torch.float64, device="cpu")
        recon = reconstruction.detach().to(dtype=torch.float64, device="cpu")
        active = features.detach().cpu() > 0
        count = int(values.shape[0])
        if count == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_m2 = float(((values - batch_mean) ** 2).sum().item())
        if self._input_mean is None:
            self._input_mean = batch_mean
            self._input_m2 = batch_m2
        else:
            previous_tokens = self._tokens
            total_tokens = previous_tokens + count
            delta = batch_mean - self._input_mean
            self._input_m2 += batch_m2 + float((delta**2).sum().item()) * (
                previous_tokens * count / total_tokens
            )
            self._input_mean = self._input_mean + delta * (count / total_tokens)
        self._tokens += count
        errors = ((values - recon) ** 2).sum(dim=-1)
        self._error_sum += float(errors.sum().item())
        self._errors.extend(float(value) for value in errors.tolist())
        self._feature_counts += active.sum(dim=0, dtype=torch.int64)
        l0 = active.sum(dim=-1, dtype=torch.int64)
        self._l0_counts += torch.bincount(l0, minlength=self.d_sae + 1)

    @staticmethod
    def _histogram_median(counts: Any) -> float:
        total = int(counts.sum().item())
        if total == 0:
            raise ValueError("cannot compute a median from no SAE tokens")
        lower_rank = (total - 1) // 2
        upper_rank = total // 2
        cumulative = 0
        lower = upper = None
        for value, count in enumerate(counts.tolist()):
            previous = cumulative
            cumulative += int(count)
            if lower is None and previous <= lower_rank < cumulative:
                lower = value
            if previous <= upper_rank < cumulative:
                upper = value
                break
        if lower is None or upper is None:
            raise RuntimeError("SAE L0 histogram is incomplete")
        return (lower + upper) / 2.0

    def finalize(self) -> SaeStatistics:
        if self._tokens == 0 or self._input_mean is None:
            raise ValueError("SAE statistics contain no tokens")
        total_variance = self._input_m2
        if total_variance <= 0.0:
            raise ValueError("SAE validation activations have zero variance")
        probabilities = tuple(
            float(value) / self._tokens for value in self._feature_counts.tolist()
        )
        dead = sum(value < self.dead_probability_below for value in probabilities) / self.d_sae
        ordered_errors = sorted(self._errors)
        middle = self._tokens // 2
        if self._tokens % 2:
            error_median = float(ordered_errors[middle])
        else:
            error_median = float((ordered_errors[middle - 1] + ordered_errors[middle]) / 2.0)
        return SaeStatistics(
            tokens=self._tokens,
            fvu=self._error_sum / total_variance,
            median_l0=self._histogram_median(self._l0_counts),
            firing_probabilities=probabilities,
            dead_feature_rate=dead,
            reconstruction_error_median=error_median,
        )


def energy_score(
    *,
    reconstruction_error: Any,
    features: Any,
    firing_probabilities: Any,
    reconstruction_error_scale: float,
) -> Any:
    import torch

    if (
        not isinstance(reconstruction_error, torch.Tensor)
        or reconstruction_error.ndim != 1
        or not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or int(features.shape[0]) != int(reconstruction_error.shape[0])
        or not isinstance(firing_probabilities, torch.Tensor)
        or firing_probabilities.ndim != 1
        or int(features.shape[1]) != int(firing_probabilities.shape[0])
    ):
        raise ValueError("SAE energy tensor shapes differ")
    if (
        isinstance(reconstruction_error_scale, bool)
        or not isinstance(reconstruction_error_scale, (int, float))
        or not math.isfinite(float(reconstruction_error_scale))
        or float(reconstruction_error_scale) <= 0.0
    ):
        raise ValueError("SAE energy reconstruction scale must be positive")
    probabilities = firing_probabilities.float().clamp(1e-8, 1.0 - 1e-8)
    rarity = torch.log((1.0 - probabilities) / probabilities)
    return reconstruction_error.float() / float(reconstruction_error_scale) + (
        features.float() * rarity
    ).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class SaeGateResult:
    passed: bool
    conditions: Mapping[str, bool]


def evaluate_sae_gates(
    *,
    fvu: float,
    median_l0: float,
    dead_feature_rate: float,
    splice_kl_median: float,
    statistics_saved: bool,
    fvu_max: float,
    median_l0_range: tuple[int, int],
    dead_feature_rate_max: float,
    splice_kl_max: float,
) -> SaeGateResult:
    values = (
        fvu,
        median_l0,
        dead_feature_rate,
        splice_kl_median,
        fvu_max,
        dead_feature_rate_max,
        splice_kl_max,
    )
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        raise ValueError("SAE gate metrics must be finite numbers")
    low, high = median_l0_range
    conditions = {
        "fvu": float(fvu) <= float(fvu_max),
        "median_l0": low <= float(median_l0) <= high,
        "dead_feature_rate": float(dead_feature_rate) <= float(dead_feature_rate_max),
        "splice_kl": float(splice_kl_median) <= float(splice_kl_max),
        "statistics_saved": type(statistics_saved) is bool and statistics_saved,
    }
    return SaeGateResult(
        passed=all(conditions.values()),
        conditions=MappingProxyType(conditions),
    )


__all__ = [
    "SaeGateResult",
    "SaeStatistics",
    "SaeStatisticsAccumulator",
    "energy_score",
    "evaluate_sae_gates",
]

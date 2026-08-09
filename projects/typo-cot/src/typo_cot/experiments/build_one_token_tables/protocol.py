"""Frozen analysis protocol for the one-token paper artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


OUTPUT_NAMES: Final = (
    "one_token_tables.json",
    "table10_one_token.csv",
    "table11_position_controls.csv",
    "one_token_tables.md",
    "one_token_tables.tex",
    "figure5_validation.json",
)


@dataclass(frozen=True, slots=True)
class InferenceProtocol:
    """Submitted-analysis constants; reduced draws are allowed only by APIs/tests."""

    seed: int = 42
    bootstrap_draws: int = 50_000
    sign_flip_draws: int = 200_000

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("inference seed must be an integer")
        for field in ("bootstrap_draws", "sign_flip_draws"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "cluster_key": ["benchmark", "sample_id"],
            "distant_interval": "intercept-only-CR1-cluster-robust-Student-t-G-minus-1/v1",
            "adjacent_interval": "numpy-percentile-cluster-bootstrap/v1",
            "bootstrap_seed_derivation": "sha256(seed|analysis-label)-first-64-bits/v1",
            "test": "monte-carlo-cluster-sign-flip-plus-one/v1",
        }


PUBLIC_INFERENCE: Final = InferenceProtocol()
ANALYSIS_PROTOCOL: Final = {
    "schema_version": "build-one-token-tables-analysis/v1",
    "population": {
        "table10": "events.table10 common paired eligibility",
        "distant_pdf_literal": "events.distant_factorial",
        "distant_submitted_producer": "events.distant_factorial_submitted_producer",
        "adjacent": "events.adjacent in three prespecified settings",
        "extension_pool_excludes_primary": True,
    },
    "inference": PUBLIC_INFERENCE.to_dict(),
    "historical_values_are_acceptance_targets": False,
}


__all__ = ["ANALYSIS_PROTOCOL", "InferenceProtocol", "OUTPUT_NAMES", "PUBLIC_INFERENCE"]

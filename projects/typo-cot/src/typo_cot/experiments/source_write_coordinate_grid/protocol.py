"""Strict result-independent protocol for the source/write coordinate grid."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "source-write-coordinate-grid-config/v1"
_ARMS = ("E->E", "E->O", "O->E", "O->O")
_COHORTS = {
    "primary": ("google/gemma-3-4b-it", "gsm8k"),
    "replication": ("mistralai/Mistral-7B-Instruct-v0.3", "mmlu"),
}
_VALIDITY = "correct-and-offset-valid/v1"
_CONTRASTS = (("E->E", "E->O"), ("E->E", "O->E"))
_MULTIPLICITY = "holm-4-cohort-contrast-tests/v1"


@dataclass(frozen=True, slots=True)
class SourceWriteCoordinateGridProtocol:
    """Complete prospective protocol loaded from the public config."""

    schema_version: str
    window: tuple[int, int]
    arms: tuple[str, ...]
    cohorts: dict[str, tuple[str, str]]
    common_validity: str
    contrasts: tuple[tuple[str, str], ...]
    pair_bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    multiplicity: str
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "window": {"start": self.window[0], "stop": self.window[1]},
            "arms": list(self.arms),
            "cohorts": {
                name: {"model": setting[0], "task": setting[1]}
                for name, setting in self.cohorts.items()
            },
            "common_validity": self.common_validity,
            "contrasts": [list(contrast) for contrast in self.contrasts],
            "statistics": {
                "pair_bootstrap_replicates": self.pair_bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "multiplicity": self.multiplicity,
            },
        }


def _integer(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def load_source_write_coordinate_grid_protocol(
    path: Path,
) -> SourceWriteCoordinateGridProtocol:
    """Load JSON-compatible YAML and reject every frozen-field drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"source/write grid config is not a file: {resolved}")
    raw = resolved.read_bytes()
    payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "window",
        "arms",
        "cohorts",
        "common_validity",
        "contrasts",
        "statistics",
    }:
        raise ValueError("source/write grid config fields differ from the frozen contract")
    if payload.get("schema_version") != _SCHEMA:
        raise ValueError("source/write grid config schema differs")
    window = payload.get("window")
    if not isinstance(window, dict) or set(window) != {"start", "stop"}:
        raise ValueError("source/write grid window must contain start and stop")
    start = _integer(window.get("start"), field="window.start", minimum=0)
    stop = _integer(window.get("stop"), field="window.stop")
    if (start, stop) != (0, 6):
        raise ValueError("source/write grid requires the frozen [0,6) window")
    if not isinstance(payload.get("arms"), list) or tuple(payload["arms"]) != _ARMS:
        raise ValueError("source/write grid arm order differs")
    raw_cohorts = payload.get("cohorts")
    if not isinstance(raw_cohorts, dict) or tuple(raw_cohorts) != tuple(_COHORTS):
        raise ValueError("source/write grid cohorts differ")
    cohorts: dict[str, tuple[str, str]] = {}
    for name, expected in _COHORTS.items():
        cohort = raw_cohorts.get(name)
        if not isinstance(cohort, dict) or set(cohort) != {"model", "task"}:
            raise ValueError(f"source/write grid cohort {name} fields differ")
        actual = cohort.get("model"), cohort.get("task")
        if actual != expected:
            raise ValueError(f"source/write grid cohort {name} setting differs")
        cohorts[name] = expected
    if payload.get("common_validity") != _VALIDITY:
        raise ValueError("source/write grid common-validity rule differs")
    contrasts = payload.get("contrasts")
    if (
        not isinstance(contrasts, list)
        or any(not isinstance(item, list) for item in contrasts)
        or tuple(tuple(item) for item in contrasts) != _CONTRASTS
    ):
        raise ValueError("source/write grid contrasts differ")
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict) or set(statistics) != {
        "pair_bootstrap_replicates",
        "bootstrap_seed",
        "confidence_level",
        "multiplicity",
    }:
        raise ValueError("source/write grid statistics fields differ")
    confidence = statistics.get("confidence_level")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("statistics.confidence_level must be between zero and one")
    if statistics.get("multiplicity") != _MULTIPLICITY:
        raise ValueError("source/write grid multiplicity family differs")
    return SourceWriteCoordinateGridProtocol(
        schema_version=_SCHEMA,
        window=(start, stop),
        arms=_ARMS,
        cohorts=cohorts,
        common_validity=_VALIDITY,
        contrasts=_CONTRASTS,
        pair_bootstrap_replicates=_integer(
            statistics.get("pair_bootstrap_replicates"),
            field="statistics.pair_bootstrap_replicates",
        ),
        bootstrap_seed=_integer(
            statistics.get("bootstrap_seed"),
            field="statistics.bootstrap_seed",
            minimum=0,
        ),
        confidence_level=float(confidence),
        multiplicity=_MULTIPLICITY,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "SourceWriteCoordinateGridProtocol",
    "load_source_write_coordinate_grid_protocol",
]

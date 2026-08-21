"""Strict, result-independent configuration for six-setting controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "six-setting-patch-controls-config/v1"
_CONTROLS = ("correct", "offset-2", "cross-item")
_MULTIPLICITY = "holm-12-setting-level-tests/v1"
_PAIR_BOOTSTRAP_REPLICATES = 10_000
_NESTED_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 42
_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class SixSettingPatchControlsProtocol:
    """Complete confirmatory protocol loaded from the public config file."""

    schema_version: str
    window: tuple[int, int]
    controls: tuple[str, ...]
    primary_denominator: str
    pair_bootstrap_replicates: int
    nested_bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    multiplicity: str
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "window": {"start": self.window[0], "stop": self.window[1]},
            "controls": list(self.controls),
            "primary_denominator": self.primary_denominator,
            "statistics": {
                "pair_bootstrap_replicates": self.pair_bootstrap_replicates,
                "nested_bootstrap_replicates": self.nested_bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "multiplicity": self.multiplicity,
            },
        }


def _integer(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def load_six_setting_patch_controls_protocol(
    path: Path,
) -> SixSettingPatchControlsProtocol:
    """Load the JSON-compatible YAML config and reject protocol drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"six-setting controls config is not a file: {resolved}")
    raw = resolved.read_bytes()
    payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    if not isinstance(payload, dict):
        raise ValueError("six-setting controls config must be a JSON object")
    if set(payload) != {
        "schema_version",
        "window",
        "controls",
        "primary_denominator",
        "statistics",
    }:
        raise ValueError("six-setting controls config fields differ from the frozen contract")
    if payload.get("schema_version") != _SCHEMA:
        raise ValueError("six-setting controls config schema differs")
    window = payload.get("window")
    if not isinstance(window, dict) or set(window) != {"start", "stop"}:
        raise ValueError("six-setting controls window must contain start and stop")
    start = _integer(window.get("start"), field="window.start", minimum=0)
    stop = _integer(window.get("stop"), field="window.stop")
    if (start, stop) != (0, 6):
        raise ValueError("six-setting controls require the frozen [0,6) window")
    controls = payload.get("controls")
    if not isinstance(controls, list) or tuple(controls) != _CONTROLS:
        raise ValueError("six-setting controls must be correct, offset-2, cross-item")
    if payload.get("primary_denominator") != "common-valid":
        raise ValueError("six-setting controls require the common-valid denominator")
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict) or set(statistics) != {
        "pair_bootstrap_replicates",
        "nested_bootstrap_replicates",
        "bootstrap_seed",
        "confidence_level",
        "multiplicity",
    }:
        raise ValueError("six-setting controls statistics fields differ")
    confidence = statistics.get("confidence_level")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("statistics.confidence_level must be numeric")
    confidence = float(confidence)
    if confidence != _CONFIDENCE_LEVEL:
        raise ValueError("statistics.confidence_level must equal the frozen 0.95")
    multiplicity = statistics.get("multiplicity")
    if multiplicity != _MULTIPLICITY:
        raise ValueError("six-setting controls require the frozen Holm family")
    pair_replicates = _integer(
        statistics.get("pair_bootstrap_replicates"),
        field="statistics.pair_bootstrap_replicates",
    )
    if pair_replicates != _PAIR_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "statistics.pair_bootstrap_replicates must equal the frozen 10,000"
        )
    nested_replicates = _integer(
        statistics.get("nested_bootstrap_replicates"),
        field="statistics.nested_bootstrap_replicates",
    )
    if nested_replicates != _NESTED_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "statistics.nested_bootstrap_replicates must equal the frozen 10,000"
        )
    bootstrap_seed = _integer(
        statistics.get("bootstrap_seed"),
        field="statistics.bootstrap_seed",
        minimum=0,
    )
    if bootstrap_seed != _BOOTSTRAP_SEED:
        raise ValueError("statistics.bootstrap_seed must equal the frozen 42")
    return SixSettingPatchControlsProtocol(
        schema_version=_SCHEMA,
        window=(start, stop),
        controls=_CONTROLS,
        primary_denominator="common-valid",
        pair_bootstrap_replicates=_PAIR_BOOTSTRAP_REPLICATES,
        nested_bootstrap_replicates=_NESTED_BOOTSTRAP_REPLICATES,
        bootstrap_seed=_BOOTSTRAP_SEED,
        confidence_level=_CONFIDENCE_LEVEL,
        multiplicity=_MULTIPLICITY,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "SixSettingPatchControlsProtocol",
    "load_six_setting_patch_controls_protocol",
]

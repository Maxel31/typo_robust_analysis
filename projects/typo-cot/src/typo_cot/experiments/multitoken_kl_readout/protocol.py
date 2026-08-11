"""Strict, result-independent protocol for multi-token KL readout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_SCHEMA = "multitoken-kl-readout-config/v1"
_WINDOW = (0, 6)
_TEACHER_FORCED_TOKENS = 16
_TARGET_SOURCE = "clean-continuation-token-ids/v1"
_PREFIX_VALIDATION = "exact-token-id-prefix/v1"
_MODEL_INPUTS = "prompt-plus-first-15-target-token-ids/v1"
_DIVERGENCE = "kl-clean-to-condition/v1"
_PRIMARY_RANGE = (2, 16)
_SECONDARY_RANGES = ((2, 4), (2, 8))
_DENOMINATOR_EPSILON = 1e-9
_LOGITS_DTYPE = "float32"
_KL_DTYPE = "float64"
_NEGATIVE_RESTORATION = "retain"
_ROUND_OFF_TOLERANCE = 1e-12
_SETTING_ESTIMATOR = "median-pair-score/v1"
_BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True, slots=True)
class MultiTokenKLReadoutProtocol:
    """Complete prospective protocol loaded from the public config."""

    schema_version: str
    window: tuple[int, int]
    teacher_forced_tokens: int
    target_source: str
    prompt_prefix_validation: str
    model_inputs: str
    divergence: str
    primary_token_range: tuple[int, int]
    secondary_token_ranges: tuple[tuple[int, int], ...]
    denominator_epsilon: float
    logits_materialized_dtype: str
    kl_dtype: str
    negative_restoration: str
    negative_kl_roundoff_tolerance: float
    setting_estimator: str
    pair_bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    config_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "window": {"start": self.window[0], "stop": self.window[1]},
            "teacher_forcing": {
                "tokens": self.teacher_forced_tokens,
                "target_source": self.target_source,
                "prompt_prefix_validation": self.prompt_prefix_validation,
                "model_inputs": self.model_inputs,
            },
            "readout": {
                "divergence": self.divergence,
                "primary_token_range": {
                    "start": self.primary_token_range[0],
                    "stop": self.primary_token_range[1],
                },
                "secondary_token_ranges": [
                    {"start": start, "stop": stop}
                    for start, stop in self.secondary_token_ranges
                ],
                "denominator_epsilon": self.denominator_epsilon,
                "logits_materialized_dtype": self.logits_materialized_dtype,
                "kl_dtype": self.kl_dtype,
                "negative_restoration": self.negative_restoration,
                "negative_kl_roundoff_tolerance": self.negative_kl_roundoff_tolerance,
            },
            "statistics": {
                "setting_estimator": self.setting_estimator,
                "pair_bootstrap_replicates": self.pair_bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
            },
        }


def _mapping(value: object, *, field: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} fields differ from the frozen contract")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if (positive and parsed <= 0.0) or (not positive and parsed < 0.0):
        operator = "> 0" if positive else ">= 0"
        raise ValueError(f"{field} must be {operator}")
    return parsed


def _range(value: object, *, field: str) -> tuple[int, int]:
    payload = _mapping(value, field=field, keys={"start", "stop"})
    start = _integer(payload.get("start"), field=f"{field}.start", minimum=1)
    stop = _integer(payload.get("stop"), field=f"{field}.stop", minimum=1)
    if start > stop:
        raise ValueError(f"{field} start must not exceed stop")
    return start, stop


def load_multitoken_kl_readout_protocol(path: Path) -> MultiTokenKLReadoutProtocol:
    """Load JSON-compatible YAML and reject every frozen-field drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"multi-token KL config is not a file: {resolved}")
    raw = resolved.read_bytes()
    payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    root = _mapping(
        payload,
        field="multi-token KL config",
        keys={"schema_version", "window", "teacher_forcing", "readout", "statistics"},
    )
    if root.get("schema_version") != _SCHEMA:
        raise ValueError("multi-token KL config schema differs")

    window_payload = _mapping(root.get("window"), field="window", keys={"start", "stop"})
    window = (
        _integer(window_payload.get("start"), field="window.start"),
        _integer(window_payload.get("stop"), field="window.stop", minimum=1),
    )
    if window != _WINDOW:
        raise ValueError("multi-token KL readout requires the frozen [0,6) window")

    forcing = _mapping(
        root.get("teacher_forcing"),
        field="teacher_forcing",
        keys={"tokens", "target_source", "prompt_prefix_validation", "model_inputs"},
    )
    if forcing != {
        "tokens": _TEACHER_FORCED_TOKENS,
        "target_source": _TARGET_SOURCE,
        "prompt_prefix_validation": _PREFIX_VALIDATION,
        "model_inputs": _MODEL_INPUTS,
    }:
        raise ValueError("multi-token KL teacher-forcing contract differs")

    readout = _mapping(
        root.get("readout"),
        field="readout",
        keys={
            "divergence",
            "primary_token_range",
            "secondary_token_ranges",
            "denominator_epsilon",
            "logits_materialized_dtype",
            "kl_dtype",
            "negative_restoration",
            "negative_kl_roundoff_tolerance",
        },
    )
    primary_range = _range(readout.get("primary_token_range"), field="primary token range")
    if primary_range != _PRIMARY_RANGE:
        raise ValueError("multi-token KL primary token range must remain 2:16")
    raw_secondary = readout.get("secondary_token_ranges")
    if not isinstance(raw_secondary, list):
        raise ValueError("secondary token ranges must be a list")
    secondary_ranges = tuple(
        _range(value, field=f"secondary token ranges[{index}]")
        for index, value in enumerate(raw_secondary)
    )
    if secondary_ranges != _SECONDARY_RANGES:
        raise ValueError("multi-token KL secondary token ranges differ")
    if (
        readout.get("divergence") != _DIVERGENCE
        or readout.get("denominator_epsilon") != _DENOMINATOR_EPSILON
        or readout.get("logits_materialized_dtype") != _LOGITS_DTYPE
        or readout.get("kl_dtype") != _KL_DTYPE
        or readout.get("negative_restoration") != _NEGATIVE_RESTORATION
        or readout.get("negative_kl_roundoff_tolerance") != _ROUND_OFF_TOLERANCE
    ):
        raise ValueError("multi-token KL readout contract differs")

    statistics = _mapping(
        root.get("statistics"),
        field="statistics",
        keys={
            "setting_estimator",
            "pair_bootstrap_replicates",
            "bootstrap_seed",
            "confidence_level",
        },
    )
    replicates = _integer(
        statistics.get("pair_bootstrap_replicates"),
        field="statistics.pair_bootstrap_replicates",
        minimum=1,
    )
    if replicates != _BOOTSTRAP_REPLICATES:
        raise ValueError("statistics.pair_bootstrap_replicates must equal the frozen 10,000")
    confidence = _number(
        statistics.get("confidence_level"), field="statistics.confidence_level"
    )
    if confidence >= 1.0:
        raise ValueError("statistics.confidence_level must be below one")
    if statistics.get("setting_estimator") != _SETTING_ESTIMATOR:
        raise ValueError("multi-token KL setting estimator differs")

    return MultiTokenKLReadoutProtocol(
        schema_version=_SCHEMA,
        window=_WINDOW,
        teacher_forced_tokens=_TEACHER_FORCED_TOKENS,
        target_source=_TARGET_SOURCE,
        prompt_prefix_validation=_PREFIX_VALIDATION,
        model_inputs=_MODEL_INPUTS,
        divergence=_DIVERGENCE,
        primary_token_range=_PRIMARY_RANGE,
        secondary_token_ranges=_SECONDARY_RANGES,
        denominator_epsilon=_DENOMINATOR_EPSILON,
        logits_materialized_dtype=_LOGITS_DTYPE,
        kl_dtype=_KL_DTYPE,
        negative_restoration=_NEGATIVE_RESTORATION,
        negative_kl_roundoff_tolerance=_ROUND_OFF_TOLERANCE,
        setting_estimator=_SETTING_ESTIMATOR,
        pair_bootstrap_replicates=_BOOTSTRAP_REPLICATES,
        bootstrap_seed=_integer(
            statistics.get("bootstrap_seed"), field="statistics.bootstrap_seed"
        ),
        confidence_level=confidence,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["MultiTokenKLReadoutProtocol", "load_multitoken_kl_readout_protocol"]

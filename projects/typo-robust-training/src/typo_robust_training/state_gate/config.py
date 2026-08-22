"""Closed-world preregistration for the transition-layer causal gate."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_TOP = {"schema_version", "model", "inputs", "cohort", "intervention", "gate"}
_MODEL = {"id", "revision", "code_revision", "decoder_layers", "dtype"}
_INPUTS = {
    "parent_probe_artifact_sha256",
    "cohort_manifest_sha256",
    "protected_registry_sha256",
    "donor_plan_sha256",
    "runtime_manifest_sha256",
}
_COHORT = {"records", "minimum_valid_records", "stratum_counts", "minimum_valid_per_stratum"}
_INTERVENTION = {
    "hook_site",
    "coordinate",
    "direction",
    "layer_source",
    "offset_control_tokens",
    "cross_item_rule",
    "self_copy_control",
    "teacher_forced_tokens",
    "readout_offsets",
    "denominator_min_exclusive",
}
_GATE = {
    "estimator",
    "bootstrap_resamples",
    "bootstrap_seed",
    "confidence",
    "bootstrap_unit",
    "minimum_correct_ci_lower",
    "minimum_correct_minus_offset_ci_lower",
    "minimum_correct_minus_cross_ci_lower",
    "maximum_absolute_self_copy_restoration",
}


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{field} fields differ")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _counts(value: object, *, field: str, minimum: int) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    parsed = {
        _string(key, field=f"{field} key"): _integer(item, field=f"{field}.{key}", minimum=minimum)
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(parsed.items())))


@dataclass(frozen=True, slots=True)
class SingleLayerGateProtocol:
    model: str
    model_revision: str
    code_revision: str
    decoder_layers: int
    input_sha256: Mapping[str, str]
    records: int
    minimum_valid_records: int
    stratum_counts: Mapping[str, int]
    minimum_valid_per_stratum: Mapping[str, int]
    offset_control_tokens: int
    teacher_forced_tokens: int
    readout_offsets: tuple[int, int]
    denominator_min_exclusive: float
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence: float
    minimum_correct_ci_lower: float
    minimum_correct_minus_offset_ci_lower: float
    minimum_correct_minus_cross_ci_lower: float
    maximum_absolute_self_copy_restoration: float
    config_sha256: str


def load_single_layer_gate_config(path: Path) -> SingleLayerGateProtocol:
    """Load the only accepted gate protocol and reject additional knobs."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("single-layer gate config must not be a symlink")
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ValueError(f"single-layer gate config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("single-layer gate config must be UTF-8") from exc
    root = _mapping(payload, field="single-layer gate config", fields=_TOP)
    if root["schema_version"] != "probe-transition-single-layer-gate-config/v1":
        raise ValueError("single-layer gate config schema differs")
    model = _mapping(root["model"], field="model", fields=_MODEL)
    if model["dtype"] != "bfloat16":
        raise ValueError("single-layer gate dtype must be bfloat16")
    revision = _string(model["revision"], field="model.revision")
    code_revision = _string(model["code_revision"], field="model.code_revision")
    if _REVISION.fullmatch(revision) is None or _REVISION.fullmatch(code_revision) is None:
        raise ValueError("single-layer gate revisions must be pinned commit SHAs")
    inputs = _mapping(root["inputs"], field="inputs", fields=_INPUTS)
    input_hashes = MappingProxyType(
        {
            field.removesuffix("_sha256"): _sha(value, field=f"inputs.{field}")
            for field, value in inputs.items()
        }
    )
    cohort = _mapping(root["cohort"], field="cohort", fields=_COHORT)
    records = _integer(cohort["records"], field="cohort.records", minimum=2)
    if records != 200:
        raise ValueError("single-layer gate requires exactly 200 independent generic pairs")
    minimum_valid = _integer(
        cohort["minimum_valid_records"], field="cohort.minimum_valid_records", minimum=2
    )
    strata = _counts(cohort["stratum_counts"], field="cohort.stratum_counts", minimum=1)
    valid_strata = _counts(
        cohort["minimum_valid_per_stratum"],
        field="cohort.minimum_valid_per_stratum",
        minimum=1,
    )
    if sum(strata.values()) != records or set(strata) != set(valid_strata):
        raise ValueError("single-layer gate cohort strata differ from record inventory")
    if minimum_valid > records or any(valid_strata[key] > strata[key] for key in strata):
        raise ValueError("single-layer gate valid-count requirements exceed the cohort")
    intervention = _mapping(
        root["intervention"], field="intervention", fields=_INTERVENTION
    )
    expected_intervention = {
        "hook_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "direction": "clean-to-typo",
        "layer_source": "parent-probe-selected-transition-layer/v1",
        "cross_item_rule": "first-valid-cyclic-source-group-derangement/v1",
        "self_copy_control": "typo-to-identical-typo-coordinate/v1",
    }
    for field, expected in expected_intervention.items():
        if intervention[field] != expected:
            raise ValueError(f"single-layer gate intervention.{field} differs")
    offset = _integer(
        intervention["offset_control_tokens"],
        field="intervention.offset_control_tokens",
        minimum=1,
    )
    if offset != 2:
        raise ValueError("single-layer gate offset control must equal +2 tokens")
    forced = _integer(
        intervention["teacher_forced_tokens"],
        field="intervention.teacher_forced_tokens",
        minimum=2,
    )
    raw_offsets = intervention["readout_offsets"]
    if raw_offsets != [2, 16] or forced != 16:
        raise ValueError("single-layer gate readout must be R_2:16")
    denominator = _number(
        intervention["denominator_min_exclusive"],
        field="intervention.denominator_min_exclusive",
        minimum=0.0,
    )
    if denominator != 1e-9:
        raise ValueError("single-layer gate denominator threshold differs")
    gate = _mapping(root["gate"], field="gate", fields=_GATE)
    if (
        gate["estimator"] != "source-group-equal-mean-pairwise-restoration/v1"
        or gate["bootstrap_unit"] != "source-group"
    ):
        raise ValueError("single-layer gate estimator differs")
    resamples = _integer(
        gate["bootstrap_resamples"], field="gate.bootstrap_resamples", minimum=1
    )
    confidence = _number(gate["confidence"], field="gate.confidence", minimum=0.0)
    if resamples != 10_000 or confidence != 0.95:
        raise ValueError("single-layer gate bootstrap protocol differs")
    correct_threshold = _number(
        gate["minimum_correct_ci_lower"],
        field="gate.minimum_correct_ci_lower",
    )
    offset_threshold = _number(
        gate["minimum_correct_minus_offset_ci_lower"],
        field="gate.minimum_correct_minus_offset_ci_lower",
    )
    cross_threshold = _number(
        gate["minimum_correct_minus_cross_ci_lower"],
        field="gate.minimum_correct_minus_cross_ci_lower",
    )
    if (correct_threshold, offset_threshold, cross_threshold) != (0.0, 0.0, 0.0):
        raise ValueError("single-layer gate causal CI thresholds must all equal zero")
    self_tolerance = _number(
        gate["maximum_absolute_self_copy_restoration"],
        field="gate.maximum_absolute_self_copy_restoration",
        minimum=0.0,
    )
    if self_tolerance > 1e-3:
        raise ValueError("single-layer gate self-copy tolerance exceeds the safe bound")
    return SingleLayerGateProtocol(
        model=_string(model["id"], field="model.id"),
        model_revision=revision,
        code_revision=code_revision,
        decoder_layers=_integer(model["decoder_layers"], field="model.decoder_layers", minimum=2),
        input_sha256=input_hashes,
        records=records,
        minimum_valid_records=minimum_valid,
        stratum_counts=strata,
        minimum_valid_per_stratum=valid_strata,
        offset_control_tokens=offset,
        teacher_forced_tokens=forced,
        readout_offsets=(2, 16),
        denominator_min_exclusive=denominator,
        bootstrap_resamples=resamples,
        bootstrap_seed=_integer(gate["bootstrap_seed"], field="gate.bootstrap_seed"),
        confidence=confidence,
        minimum_correct_ci_lower=correct_threshold,
        minimum_correct_minus_offset_ci_lower=offset_threshold,
        minimum_correct_minus_cross_ci_lower=cross_threshold,
        maximum_absolute_self_copy_restoration=self_tolerance,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["SingleLayerGateProtocol", "load_single_layer_gate_config"]

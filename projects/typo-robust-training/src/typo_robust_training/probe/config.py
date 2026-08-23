"""Strict preregistration for producing linear-probe transition evidence."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLES = ("fit", "selection", "validation")
_PAIRED_ROLES = ("selection", "validation")
_TOP = {"schema_version", "model", "inputs", "cohorts", "probe", "selection"}
_MODEL = {
    "id",
    "revision",
    "code_revision",
    "decoder_layers",
    "hidden_size",
    "dtype",
}
_INPUTS = {
    "class_inventory_sha256",
    "fit_manifest_sha256",
    "selection_manifest_sha256",
    "validation_manifest_sha256",
    "protected_registry_sha256",
}
_COHORTS = {
    "records_per_class",
    "min_source_groups_per_class",
    "stratum_counts",
}
_PROBE_V2 = {
    "seeds",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "epsilon",
    "epochs",
    "batch_size",
    "hook_site",
    "coordinate",
}
_PROBE_V3 = {
    "seeds",
    "optimizer",
    "standardization",
    "l2_penalty",
    "max_iterations",
    "max_evaluations",
    "history_size",
    "gradient_tolerance",
    "change_tolerance",
    "solver_objective_relative_tolerance",
    "solver_parameter_relative_tolerance",
    "folded_logit_tolerance",
    "serialized_logit_tolerance",
    "hook_site",
    "coordinate",
}
_SELECTION = {
    "metric",
    "rule",
    "tie_break",
    "stability_rule",
    "validation_rule",
    "bootstrap",
}
_BOOTSTRAP = {"resamples", "seed", "confidence", "unit"}

SELECTION_METRIC = "largest-group-mean-paired-noise-penalty-drop/v2"
SELECTION_RULE = "min-argmax-over-layers-one-through-last/v1"
TIE_BREAK = "smallest-layer/v1"
STABILITY_RULE = "selection-exact-and-validation-within-one-layer-for-both-seeds/v1"
VALIDATION_RULE = "group-bootstrap-95pct-lower-positive-for-both-seeds/v1"
HOOK_SITE = "complete-decoder-block-residual-output"
COORDINATE = "edited-word-final-token/v1"


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be finite and >= {minimum}{suffix}")
    return result


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _strict_role_counts(value: object, *, field: str) -> Mapping[str, int]:
    if not isinstance(value, dict) or set(value) != set(_ROLES):
        raise ValueError(f"{field} must define exactly fit, selection, and validation")
    return MappingProxyType(
        {role: _integer(value[role], field=f"{field}.{role}", minimum=2) for role in _ROLES}
    )


def _stratum_counts(value: object) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, dict) or set(value) != set(_PAIRED_ROLES):
        raise ValueError("stratum_counts must define exactly selection and validation")
    parsed: dict[str, Mapping[str, int]] = {}
    for role in _PAIRED_ROLES:
        rows = value[role]
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"stratum_counts.{role} must be one non-empty mapping")
        role_counts: dict[str, int] = {}
        for raw_key, raw_count in rows.items():
            if not isinstance(raw_key, str) or len(raw_key.split("|")) != 3:
                raise ValueError(
                    "stratum keys must be canonical edit_type|edit_count|token_inflation_bucket"
                )
            edit_type, edit_count, inflation = raw_key.split("|")
            if not edit_type or not inflation or not edit_count.isdigit() or int(edit_count) < 1:
                raise ValueError(f"stratum_counts.{role} contains an invalid key")
            role_counts[raw_key] = _integer(
                raw_count, field=f"stratum_counts.{role}.{raw_key}", minimum=1
            )
        parsed[role] = MappingProxyType(role_counts)
    return MappingProxyType(parsed)


@dataclass(frozen=True, slots=True)
class ProbeProducerProtocol:
    """One fully pinned, behavior-independent probe fitting protocol."""

    model: str
    model_revision: str
    code_revision: str
    decoder_layers: int
    hidden_size: int
    input_sha256: Mapping[str, str]
    records_per_class: Mapping[str, int]
    min_source_groups_per_class: Mapping[str, int]
    stratum_counts: Mapping[str, Mapping[str, int]]
    probe_seeds: tuple[int, int]
    learning_rate: float | None
    weight_decay: float | None
    beta1: float | None
    beta2: float | None
    epsilon: float | None
    epochs: int | None
    batch_size: int | None
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_confidence: float
    config_sha256: str
    schema_version: str = "typo-linear-probe-producer-config/v2"
    dtype: str = "bfloat16"
    optimizer: str = "adamw"
    standardization: str | None = None
    l2_penalty: str | None = None
    max_iterations: int | None = None
    max_evaluations: int | None = None
    history_size: int | None = None
    gradient_tolerance: float | None = None
    change_tolerance: float | None = None
    solver_objective_relative_tolerance: float | None = None
    solver_parameter_relative_tolerance: float | None = None
    folded_logit_tolerance: float | None = None
    serialized_logit_tolerance: float | None = None
    hook_site: str = HOOK_SITE
    coordinate: str = COORDINATE
    selection_metric: str = SELECTION_METRIC
    selection_rule: str = SELECTION_RULE
    tie_break: str = TIE_BREAK
    stability_rule: str = STABILITY_RULE
    validation_rule: str = VALIDATION_RULE
    bootstrap_unit: str = "source-group"


def load_probe_producer_config(path: Path) -> ProbeProducerProtocol:
    """Load a closed-world producer preregistration before model initialization."""

    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        resolved = supplied.resolve()
        raise ValueError(f"probe producer config is not one regular file: {resolved}")
    resolved = supplied.resolve()
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("probe producer config must be UTF-8") from exc
    if not isinstance(payload, dict) or set(payload) != _TOP:
        raise ValueError("probe producer config fields differ")
    schema_version = payload["schema_version"]
    if schema_version not in {
        "typo-linear-probe-producer-config/v2",
        "typo-linear-probe-producer-config/v3",
    }:
        raise ValueError("probe producer config schema differs")
    model = payload["model"]
    inputs = payload["inputs"]
    cohorts = payload["cohorts"]
    probe = payload["probe"]
    selection = payload["selection"]
    if not isinstance(model, dict) or set(model) != _MODEL:
        raise ValueError("probe producer model fields differ")
    if not isinstance(inputs, dict) or set(inputs) != _INPUTS:
        raise ValueError("probe producer input fields differ")
    if not isinstance(cohorts, dict) or set(cohorts) != _COHORTS:
        raise ValueError("probe producer cohort fields differ")
    expected_probe_fields = _PROBE_V3 if schema_version.endswith("/v3") else _PROBE_V2
    if not isinstance(probe, dict) or set(probe) != expected_probe_fields:
        raise ValueError("probe producer probe fields differ")
    if not isinstance(selection, dict) or set(selection) != _SELECTION:
        raise ValueError("probe producer selection fields differ")
    bootstrap = selection["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != _BOOTSTRAP:
        raise ValueError("probe producer bootstrap fields differ")

    model_id = model["id"]
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("probe producer model id must be non-empty")
    revision = model["revision"]
    code_revision = model["code_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("probe producer model revision must be a pinned commit SHA")
    if not isinstance(code_revision, str) or _REVISION.fullmatch(code_revision) is None:
        raise ValueError("probe producer code revision must be a pinned commit SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("probe producer model dtype differs")

    seeds = probe["seeds"]
    if not isinstance(seeds, list) or seeds != [42, 43]:
        raise ValueError("probe producer seeds must be exactly [42, 43]")
    expected = {
        "optimizer": (
            "full-batch-lbfgs-strong-wolfe-float64/v1"
            if schema_version.endswith("/v3")
            else "adamw"
        ),
        "hook_site": HOOK_SITE,
        "coordinate": COORDINATE,
    }
    for field, literal in expected.items():
        if probe[field] != literal:
            raise ValueError(f"probe producer {field} differs")
    expected_selection = {
        "metric": SELECTION_METRIC,
        "rule": SELECTION_RULE,
        "tie_break": TIE_BREAK,
        "stability_rule": STABILITY_RULE,
        "validation_rule": VALIDATION_RULE,
    }
    for field, literal in expected_selection.items():
        if selection[field] != literal:
            raise ValueError(f"probe producer selection {field} differs")
    if bootstrap != {
        "resamples": 10_000,
        "seed": 1729,
        "confidence": 0.95,
        "unit": "source-group",
    }:
        raise ValueError("probe producer bootstrap protocol differs")

    input_hashes = MappingProxyType(
        {
            "class_inventory": _sha(inputs["class_inventory_sha256"], field="class inventory hash"),
            "fit_manifest": _sha(inputs["fit_manifest_sha256"], field="fit manifest hash"),
            "selection_manifest": _sha(
                inputs["selection_manifest_sha256"], field="selection manifest hash"
            ),
            "validation_manifest": _sha(
                inputs["validation_manifest_sha256"], field="validation manifest hash"
            ),
            "protected_split_registry": _sha(
                inputs["protected_registry_sha256"], field="protected registry hash"
            ),
        }
    )
    records_per_class = _strict_role_counts(cohorts["records_per_class"], field="records_per_class")
    minimum_groups = _strict_role_counts(
        cohorts["min_source_groups_per_class"],
        field="min_source_groups_per_class",
    )
    if any(minimum_groups[role] > records_per_class[role] for role in _ROLES):
        raise ValueError("minimum source groups cannot exceed records per class")
    strata = _stratum_counts(cohorts["stratum_counts"])

    if schema_version.endswith("/v3"):
        if probe["standardization"] != "fit-only-per-layer-scalar-rms-folded/v1":
            raise ValueError("probe producer standardization differs")
        if probe["l2_penalty"] != "unit-prior-sum-loss/v1":
            raise ValueError("probe producer L2 penalty differs")
        fixed_numbers = {
            "max_iterations": 1000,
            "max_evaluations": 1250,
            "history_size": 100,
            "gradient_tolerance": 1e-7,
            "change_tolerance": 1e-12,
            "solver_objective_relative_tolerance": 1e-9,
            "solver_parameter_relative_tolerance": 1e-5,
            "folded_logit_tolerance": 1e-8,
            "serialized_logit_tolerance": 1e-5,
        }
        for field, expected_value in fixed_numbers.items():
            if probe[field] != expected_value:
                raise ValueError(f"probe producer {field} differs")
        beta1 = beta2 = None
    else:
        beta1 = _number(probe["beta1"], field="probe beta1", maximum=1.0)
        beta2 = _number(probe["beta2"], field="probe beta2", maximum=1.0)
        if beta1 >= 1.0 or beta2 >= 1.0:
            raise ValueError("probe optimizer beta values must be below one")
    return ProbeProducerProtocol(
        model=model_id,
        model_revision=revision,
        code_revision=code_revision,
        decoder_layers=_integer(model["decoder_layers"], field="decoder layers", minimum=2),
        hidden_size=_integer(model["hidden_size"], field="hidden size", minimum=1),
        input_sha256=input_hashes,
        records_per_class=records_per_class,
        min_source_groups_per_class=minimum_groups,
        stratum_counts=strata,
        probe_seeds=(42, 43),
        learning_rate=(
            None
            if schema_version.endswith("/v3")
            else _number(probe["learning_rate"], field="probe learning rate", minimum=1e-12)
        ),
        weight_decay=(
            None
            if schema_version.endswith("/v3")
            else _number(probe["weight_decay"], field="probe weight decay")
        ),
        beta1=beta1,
        beta2=beta2,
        epsilon=(
            None
            if schema_version.endswith("/v3")
            else _number(probe["epsilon"], field="probe epsilon", minimum=1e-12)
        ),
        epochs=(
            None
            if schema_version.endswith("/v3")
            else _integer(probe["epochs"], field="probe epochs", minimum=1)
        ),
        batch_size=(
            None
            if schema_version.endswith("/v3")
            else _integer(probe["batch_size"], field="probe batch size", minimum=1)
        ),
        bootstrap_resamples=10_000,
        bootstrap_seed=1729,
        bootstrap_confidence=0.95,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        optimizer=str(probe["optimizer"]),
        standardization=(str(probe["standardization"]) if schema_version.endswith("/v3") else None),
        l2_penalty=(str(probe["l2_penalty"]) if schema_version.endswith("/v3") else None),
        max_iterations=(int(probe["max_iterations"]) if schema_version.endswith("/v3") else None),
        max_evaluations=(int(probe["max_evaluations"]) if schema_version.endswith("/v3") else None),
        history_size=(int(probe["history_size"]) if schema_version.endswith("/v3") else None),
        gradient_tolerance=(
            float(probe["gradient_tolerance"]) if schema_version.endswith("/v3") else None
        ),
        change_tolerance=(
            float(probe["change_tolerance"]) if schema_version.endswith("/v3") else None
        ),
        solver_objective_relative_tolerance=(
            float(probe["solver_objective_relative_tolerance"])
            if schema_version.endswith("/v3")
            else None
        ),
        solver_parameter_relative_tolerance=(
            float(probe["solver_parameter_relative_tolerance"])
            if schema_version.endswith("/v3")
            else None
        ),
        folded_logit_tolerance=(
            float(probe["folded_logit_tolerance"]) if schema_version.endswith("/v3") else None
        ),
        serialized_logit_tolerance=(
            float(probe["serialized_logit_tolerance"]) if schema_version.endswith("/v3") else None
        ),
    )


__all__ = [
    "COORDINATE",
    "HOOK_SITE",
    "ProbeProducerProtocol",
    "SELECTION_METRIC",
    "SELECTION_RULE",
    "STABILITY_RULE",
    "TIE_BREAK",
    "VALIDATION_RULE",
    "load_probe_producer_config",
]

"""Strict configuration for layer-constrained neuron/head localization."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from typo_robust_training.data.config import strict_loads


_REVISION = re.compile(r"[0-9a-f]{40}")
_TASKS = ("gsm8k", "mmlu", "arc")
_KINDS = ("mlp-neuron", "attention-head")
_TOP = {"schema_version", "model", "architecture", "partition", "screening", "causal_validation"}
_MODEL = {"id", "revision", "dtype"}
_ARCHITECTURE = {
    "decoder_layers",
    "hidden_size",
    "mlp_intermediate_size",
    "attention_heads",
    "attention_head_dim",
    "mlp_site",
    "attention_site",
}
_PARTITION = {
    "tasks",
    "algorithm",
    "seed",
    "screening_role",
    "causal_role",
    "require_disjoint",
}
_SCREENING = {
    "coordinate",
    "objective",
    "activation_difference",
    "gradient_attribution",
    "normalization",
    "activation_weight",
    "attribution_weight",
    "minimum_positive_attribution_tasks",
    "mlp_shortlist_per_layer",
    "attention_shortlist_per_layer",
    "causal_candidate_limits",
    "ranking",
}
_CAUSAL = {
    "direction",
    "readouts",
    "teacher_forced_targets",
    "untreated_mean_kl_min_exclusive",
    "minimum_kl_eligible_per_task",
    "minimum_kl_eligible_fraction_per_task",
    "minimum_answer_cohort_per_task",
    "score_formula",
    "beta",
    "gamma",
    "minimum_beneficial_tasks",
    "maximum_harm_rate_per_task",
    "component_weighting",
    "minimum_selected_components",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence_level",
}


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if set(value) != fields:
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


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True, slots=True)
class ComponentLocalizationProtocol:
    """Resolved immutable component-localization protocol."""

    schema_version: str
    model: str
    model_revision: str
    dtype: str
    decoder_layers: int
    hidden_size: int
    mlp_intermediate_size: int
    attention_heads: int
    attention_head_dim: int
    mlp_site: str
    attention_site: str
    tasks: tuple[str, ...]
    partition_algorithm: str
    partition_seed: int
    screening_role: str
    causal_role: str
    coordinate: str
    objective: str
    activation_difference: str
    gradient_attribution: str
    normalization: str
    activation_weight: float
    attribution_weight: float
    minimum_positive_attribution_tasks: int
    mlp_shortlist_per_layer: int
    attention_shortlist_per_layer: int
    causal_candidate_limits: Mapping[str, int]
    screening_ranking: str
    direction: str
    readouts: tuple[str, ...]
    teacher_forced_targets: str
    untreated_mean_kl_min_exclusive: float
    minimum_kl_eligible_per_task: int
    minimum_kl_eligible_fraction_per_task: float
    minimum_answer_cohort_per_task: int
    score_formula: str
    beta: float
    gamma: float
    minimum_beneficial_tasks: int
    maximum_harm_rate_per_task: float
    component_weighting: str
    minimum_selected_components: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    config_sha256: str


def load_component_localization_config(path: Path) -> ComponentLocalizationProtocol:
    """Load JSON-in-YAML without accepting silent protocol drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"component localization config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError(f"component localization config is not UTF-8: {resolved}") from exc
    root = _mapping(payload, field="config", fields=_TOP)
    if root["schema_version"] != "robustness-component-localization-config/v1":
        raise ValueError("component localization schema_version differs")
    model = _mapping(root["model"], field="model", fields=_MODEL)
    revision = _string(model["revision"], field="model.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be a pinned 40-character SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")

    architecture = _mapping(root["architecture"], field="architecture", fields=_ARCHITECTURE)
    expected_sites = {
        "mlp_site": "input-to-down-proj-post-swiglu/v1",
        "attention_site": "input-to-o-proj-concatenated-query-head-output/v1",
    }
    for field, expected in expected_sites.items():
        if architecture[field] != expected:
            raise ValueError(f"architecture.{field} differs from {expected}")

    partition = _mapping(root["partition"], field="partition", fields=_PARTITION)
    if tuple(partition["tasks"]) != _TASKS:  # type: ignore[arg-type]
        raise ValueError("partition.tasks must be gsm8k, mmlu, arc in frozen order")
    expected_partition = {
        "algorithm": "sha256-order-half-per-task/v1",
        "screening_role": "component-screening",
        "causal_role": "component-causal-validation",
        "require_disjoint": True,
    }
    for field, expected in expected_partition.items():
        if partition[field] != expected:
            raise ValueError(f"partition.{field} differs from {expected}")

    screening = _mapping(root["screening"], field="screening", fields=_SCREENING)
    expected_screening = {
        "coordinate": "edited-word-final-token/v1",
        "objective": "mean-kl-clean-to-typo-tokens-2-through-16/v1",
        "activation_difference": "mean-absolute-neuron-or-head-l2/v1",
        "gradient_attribution": "negative-gradient-dot-clean-minus-typo/v1",
        "normalization": "within-task-layer-type-midrank-percentile/v1",
        "ranking": "equal-task-macro-descending-then-component-id/v1",
    }
    for field, expected in expected_screening.items():
        if screening[field] != expected:
            raise ValueError(f"screening.{field} differs from {expected}")
    limits = _mapping(
        screening["causal_candidate_limits"],
        field="screening.causal_candidate_limits",
        fields=set(_KINDS),
    )
    causal_limits = {
        kind: _integer(limits[kind], field=f"screening.causal_candidate_limits.{kind}", minimum=1)
        for kind in _KINDS
    }
    activation_weight = _number(screening["activation_weight"], field="screening.activation_weight")
    attribution_weight = _number(
        screening["attribution_weight"], field="screening.attribution_weight"
    )
    if not math.isclose(activation_weight + attribution_weight, 1.0, abs_tol=1e-12):
        raise ValueError("screening weights must sum to one")

    causal = _mapping(root["causal_validation"], field="causal_validation", fields=_CAUSAL)
    expected_causal = {
        "direction": "clean-to-typo",
        "readouts": ["answer", "multitoken-kl"],
        "teacher_forced_targets": "reuse-layer-scan-clean-targets/v1",
        "score_formula": "R_KL_2:16+beta*R_answer-gamma*H/v1",
        "component_weighting": "positive-macro-causal-score-normalized/v1",
    }
    for field, expected in expected_causal.items():
        if causal[field] != expected:
            raise ValueError(f"causal_validation.{field} differs from {expected}")
    fraction = _number(
        causal["minimum_kl_eligible_fraction_per_task"],
        field="causal_validation.minimum_kl_eligible_fraction_per_task",
    )
    harm_limit = _number(
        causal["maximum_harm_rate_per_task"],
        field="causal_validation.maximum_harm_rate_per_task",
    )
    confidence = _number(causal["confidence_level"], field="causal_validation.confidence_level")
    if fraction > 1.0 or harm_limit > 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("component causal probabilities are outside their valid range")

    return ComponentLocalizationProtocol(
        schema_version="robustness-component-localization-config/v1",
        model=_string(model["id"], field="model.id"),
        model_revision=revision,
        dtype="bfloat16",
        decoder_layers=_integer(
            architecture["decoder_layers"], field="architecture.decoder_layers", minimum=1
        ),
        hidden_size=_integer(
            architecture["hidden_size"], field="architecture.hidden_size", minimum=1
        ),
        mlp_intermediate_size=_integer(
            architecture["mlp_intermediate_size"],
            field="architecture.mlp_intermediate_size",
            minimum=1,
        ),
        attention_heads=_integer(
            architecture["attention_heads"], field="architecture.attention_heads", minimum=1
        ),
        attention_head_dim=_integer(
            architecture["attention_head_dim"],
            field="architecture.attention_head_dim",
            minimum=1,
        ),
        mlp_site=str(architecture["mlp_site"]),
        attention_site=str(architecture["attention_site"]),
        tasks=_TASKS,
        partition_algorithm=str(partition["algorithm"]),
        partition_seed=_integer(partition["seed"], field="partition.seed"),
        screening_role=str(partition["screening_role"]),
        causal_role=str(partition["causal_role"]),
        coordinate=str(screening["coordinate"]),
        objective=str(screening["objective"]),
        activation_difference=str(screening["activation_difference"]),
        gradient_attribution=str(screening["gradient_attribution"]),
        normalization=str(screening["normalization"]),
        activation_weight=activation_weight,
        attribution_weight=attribution_weight,
        minimum_positive_attribution_tasks=_integer(
            screening["minimum_positive_attribution_tasks"],
            field="screening.minimum_positive_attribution_tasks",
            minimum=1,
        ),
        mlp_shortlist_per_layer=_integer(
            screening["mlp_shortlist_per_layer"],
            field="screening.mlp_shortlist_per_layer",
            minimum=1,
        ),
        attention_shortlist_per_layer=_integer(
            screening["attention_shortlist_per_layer"],
            field="screening.attention_shortlist_per_layer",
            minimum=1,
        ),
        causal_candidate_limits=MappingProxyType(causal_limits),
        screening_ranking=str(screening["ranking"]),
        direction=str(causal["direction"]),
        readouts=tuple(causal["readouts"]),  # type: ignore[arg-type]
        teacher_forced_targets=str(causal["teacher_forced_targets"]),
        untreated_mean_kl_min_exclusive=_number(
            causal["untreated_mean_kl_min_exclusive"],
            field="causal_validation.untreated_mean_kl_min_exclusive",
        ),
        minimum_kl_eligible_per_task=_integer(
            causal["minimum_kl_eligible_per_task"],
            field="causal_validation.minimum_kl_eligible_per_task",
            minimum=1,
        ),
        minimum_kl_eligible_fraction_per_task=fraction,
        minimum_answer_cohort_per_task=_integer(
            causal["minimum_answer_cohort_per_task"],
            field="causal_validation.minimum_answer_cohort_per_task",
            minimum=1,
        ),
        score_formula=str(causal["score_formula"]),
        beta=_number(causal["beta"], field="causal_validation.beta"),
        gamma=_number(causal["gamma"], field="causal_validation.gamma"),
        minimum_beneficial_tasks=_integer(
            causal["minimum_beneficial_tasks"],
            field="causal_validation.minimum_beneficial_tasks",
            minimum=1,
        ),
        maximum_harm_rate_per_task=harm_limit,
        component_weighting=str(causal["component_weighting"]),
        minimum_selected_components=_integer(
            causal["minimum_selected_components"],
            field="causal_validation.minimum_selected_components",
            minimum=1,
        ),
        bootstrap_replicates=_integer(
            causal["bootstrap_replicates"],
            field="causal_validation.bootstrap_replicates",
            minimum=1,
        ),
        bootstrap_seed=_integer(causal["bootstrap_seed"], field="causal_validation.bootstrap_seed"),
        confidence_level=confidence,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["ComponentLocalizationProtocol", "load_component_localization_config"]

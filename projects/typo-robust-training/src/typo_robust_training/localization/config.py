"""Strict configuration for diagnostic layer selection."""

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
_MODEL_FIELDS = {"id", "revision", "dtype"}
_DIAGNOSTIC_FIELDS = {
    "tasks",
    "prompt_template",
    "patch_direction",
    "patch_site",
    "coordinate",
    "clean_continuation_source",
    "teacher_forced_tokens",
    "readout_token_start",
    "readout_token_stop",
    "untreated_mean_kl_min_exclusive",
    "minimum_kl_eligible_per_task",
    "minimum_kl_eligible_fraction_per_task",
    "minimum_answer_cohort_per_task",
    "generation",
}
_GENERATION = {
    "max_new_tokens": 512,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
    "termination_protocol": "effective-eos-vs-length-cap/v1",
    "answer_extraction": "primary-then-empty-only-positional/v1",
}
_SELECTION_FIELDS = {
    "score_formula",
    "beta",
    "gamma",
    "window_width",
    "layer_to_window_aggregation",
    "cross_task_aggregation",
    "ranking",
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
class LayerSelectionProtocol:
    """Resolved, immutable layer-selection protocol."""

    schema_version: str
    model: str
    model_revision: str
    dtype: str
    tasks: tuple[str, ...]
    prompt_template: str
    patch_direction: str
    patch_site: str
    coordinate: str
    clean_continuation_source: str
    teacher_forced_tokens: int
    readout_token_start: int
    readout_token_stop: int
    untreated_mean_kl_min_exclusive: float
    minimum_kl_eligible_per_task: int
    minimum_kl_eligible_fraction_per_task: float
    minimum_answer_cohort_per_task: int
    generation: Mapping[str, object]
    score_formula: str
    beta: float
    gamma: float
    window_width: int
    layer_to_window_aggregation: str
    cross_task_aggregation: str
    ranking: str
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    config_sha256: str

    @property
    def readout_token_range(self) -> tuple[int, int]:
        return self.readout_token_start, self.readout_token_stop

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": {
                "id": self.model,
                "revision": self.model_revision,
                "dtype": self.dtype,
            },
            "diagnostic": {
                "tasks": list(self.tasks),
                "prompt_template": self.prompt_template,
                "patch_direction": self.patch_direction,
                "patch_site": self.patch_site,
                "coordinate": self.coordinate,
                "clean_continuation_source": self.clean_continuation_source,
                "teacher_forced_tokens": self.teacher_forced_tokens,
                "readout_token_start": self.readout_token_start,
                "readout_token_stop": self.readout_token_stop,
                "untreated_mean_kl_min_exclusive": self.untreated_mean_kl_min_exclusive,
                "minimum_kl_eligible_per_task": self.minimum_kl_eligible_per_task,
                "minimum_kl_eligible_fraction_per_task": (
                    self.minimum_kl_eligible_fraction_per_task
                ),
                "minimum_answer_cohort_per_task": self.minimum_answer_cohort_per_task,
                "generation": dict(self.generation),
            },
            "selection": {
                "score_formula": self.score_formula,
                "beta": self.beta,
                "gamma": self.gamma,
                "window_width": self.window_width,
                "layer_to_window_aggregation": self.layer_to_window_aggregation,
                "cross_task_aggregation": self.cross_task_aggregation,
                "ranking": self.ranking,
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
            },
        }


def load_layer_selection_config(path: Path) -> LayerSelectionProtocol:
    """Load a strict JSON-in-YAML layer-selection configuration."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"layer selection config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError(f"layer selection config is not UTF-8: {resolved}") from exc
    root = _mapping(
        payload,
        field="config",
        fields={"schema_version", "model", "diagnostic", "selection"},
    )
    if root["schema_version"] != "robustness-layer-selection-config/v1":
        raise ValueError("layer selection schema_version differs")

    model = _mapping(root["model"], field="model", fields=_MODEL_FIELDS)
    revision = _string(model["revision"], field="model.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be a pinned 40-character SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")

    diagnostic = _mapping(root["diagnostic"], field="diagnostic", fields=_DIAGNOSTIC_FIELDS)
    raw_tasks = diagnostic["tasks"]
    if not isinstance(raw_tasks, list) or tuple(raw_tasks) != _TASKS:
        raise ValueError("diagnostic.tasks must be gsm8k, mmlu, arc in frozen order")
    expected_strings = {
        "prompt_template": "paper-few-shot-cot/v1",
        "patch_direction": "clean-to-typo",
        "patch_site": "complete-decoder-block-residual-output",
        "coordinate": "edited-word-final-token/v1",
        "clean_continuation_source": "greedy-clean-generation/v1",
    }
    for field, expected in expected_strings.items():
        if diagnostic[field] != expected:
            raise ValueError(f"diagnostic.{field} differs from {expected}")
    teacher_forced_tokens = _integer(
        diagnostic["teacher_forced_tokens"], field="diagnostic.teacher_forced_tokens", minimum=1
    )
    readout_start = _integer(
        diagnostic["readout_token_start"], field="diagnostic.readout_token_start", minimum=1
    )
    readout_stop = _integer(
        diagnostic["readout_token_stop"], field="diagnostic.readout_token_stop", minimum=1
    )
    if (teacher_forced_tokens, readout_start, readout_stop) != (16, 2, 16):
        raise ValueError("diagnostic teacher-forced readout must be tokens 2--16 of 16")
    threshold = _number(
        diagnostic["untreated_mean_kl_min_exclusive"],
        field="diagnostic.untreated_mean_kl_min_exclusive",
    )
    if threshold <= 0.0:
        raise ValueError("diagnostic KL threshold must be positive")
    minimum_fraction = _number(
        diagnostic["minimum_kl_eligible_fraction_per_task"],
        field="diagnostic.minimum_kl_eligible_fraction_per_task",
    )
    if minimum_fraction > 1.0:
        raise ValueError("diagnostic minimum KL eligible fraction must be <= 1")
    generation = _mapping(
        diagnostic["generation"],
        field="diagnostic.generation",
        fields=set(_GENERATION),
    )
    if dict(generation) != _GENERATION:
        raise ValueError("diagnostic.generation differs from the paper protocol")

    selection = _mapping(root["selection"], field="selection", fields=_SELECTION_FIELDS)
    expected_selection_strings = {
        "score_formula": "R_KL_2:16+beta*R_answer-gamma*H/v1",
        "layer_to_window_aggregation": "arithmetic-mean/v1",
        "cross_task_aggregation": "equal-task-macro-mean/v1",
        "ranking": "score-descending-then-start-ascending/v1",
    }
    for field, expected in expected_selection_strings.items():
        if selection[field] != expected:
            raise ValueError(f"selection.{field} differs from {expected}")
    confidence_level = _number(selection["confidence_level"], field="selection.confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("selection.confidence_level must be strictly between zero and one")

    return LayerSelectionProtocol(
        schema_version="robustness-layer-selection-config/v1",
        model=_string(model["id"], field="model.id"),
        model_revision=revision,
        dtype="bfloat16",
        tasks=_TASKS,
        prompt_template=str(diagnostic["prompt_template"]),
        patch_direction=str(diagnostic["patch_direction"]),
        patch_site=str(diagnostic["patch_site"]),
        coordinate=str(diagnostic["coordinate"]),
        clean_continuation_source=str(diagnostic["clean_continuation_source"]),
        teacher_forced_tokens=teacher_forced_tokens,
        readout_token_start=readout_start,
        readout_token_stop=readout_stop,
        untreated_mean_kl_min_exclusive=threshold,
        minimum_kl_eligible_per_task=_integer(
            diagnostic["minimum_kl_eligible_per_task"],
            field="diagnostic.minimum_kl_eligible_per_task",
            minimum=1,
        ),
        minimum_kl_eligible_fraction_per_task=minimum_fraction,
        minimum_answer_cohort_per_task=_integer(
            diagnostic["minimum_answer_cohort_per_task"],
            field="diagnostic.minimum_answer_cohort_per_task",
            minimum=1,
        ),
        generation=MappingProxyType(dict(generation)),
        score_formula=str(selection["score_formula"]),
        beta=_number(selection["beta"], field="selection.beta"),
        gamma=_number(selection["gamma"], field="selection.gamma"),
        window_width=_integer(selection["window_width"], field="selection.window_width", minimum=1),
        layer_to_window_aggregation=str(selection["layer_to_window_aggregation"]),
        cross_task_aggregation=str(selection["cross_task_aggregation"]),
        ranking=str(selection["ranking"]),
        bootstrap_replicates=_integer(
            selection["bootstrap_replicates"],
            field="selection.bootstrap_replicates",
            minimum=1,
        ),
        bootstrap_seed=_integer(selection["bootstrap_seed"], field="selection.bootstrap_seed"),
        confidence_level=confidence_level,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["LayerSelectionProtocol", "load_layer_selection_config"]

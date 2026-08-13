"""Strict protocol for clean/typo adapter evaluation."""

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
_TOP = {
    "schema_version",
    "model",
    "sequence",
    "prompting",
    "generation",
    "metrics",
    "paired_patch",
    "gate",
}
_MODEL = {"id", "revision", "dtype"}
_SEQUENCE = {
    "max_input_tokens",
    "max_new_tokens",
    "teacher_forced_tokens",
    "readout_token_range",
}
_PROMPTING = {"protocol", "answer_extraction"}
_GENERATION = {
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "temperature",
    "top_p",
    "top_k",
    "use_cache",
    "termination_protocol",
}
_METRICS = {"bootstrap_replicates", "bootstrap_seed", "confidence_level", "seed_inventory"}
_PATCH = {"position", "window_source"}
_GATE = {
    "minimum_typo_accuracy_gain_points",
    "maximum_clean_accuracy_drop_points",
    "require_wrong_to_right_above_right_to_wrong",
    "require_positive_unseen_task_gain",
    "minimum_directional_seeds",
    "minimum_patch_gain_reduction_fraction",
}


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"evaluation {field} fields differ")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation {field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"evaluation {field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"evaluation {field} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True, slots=True)
class RobustnessEvaluationProtocol:
    schema_version: str
    model: str
    model_revision: str
    dtype: str
    max_input_tokens: int
    max_new_tokens: int
    teacher_forced_tokens: int
    readout_token_range: tuple[int, int]
    prompt_protocol: str
    answer_extraction: str
    generation: Mapping[str, object]
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    seed_inventory: tuple[int, ...]
    patch_position: str
    patch_window_source: str
    gate: Mapping[str, object]
    config_sha256: str


def load_robustness_evaluation_config(path: Path) -> RobustnessEvaluationProtocol:
    """Load JSON-in-YAML while rejecting any scientific protocol drift."""

    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("evaluation config must be UTF-8") from exc
    top = _mapping(payload, field="config", fields=_TOP)
    if top["schema_version"] != "robustness-evaluation-config/v1":
        raise ValueError("evaluation config schema differs")
    model = _mapping(top["model"], field="model", fields=_MODEL)
    model_id = _text(model["id"], field="model.id")
    revision = _text(model["revision"], field="model.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("evaluation model revision must be a pinned 40-character SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("evaluation model dtype must be bfloat16")

    sequence = _mapping(top["sequence"], field="sequence", fields=_SEQUENCE)
    max_input = _integer(sequence["max_input_tokens"], field="max_input_tokens", minimum=1)
    max_new = _integer(sequence["max_new_tokens"], field="max_new_tokens", minimum=1)
    forced = _integer(sequence["teacher_forced_tokens"], field="teacher_forced_tokens", minimum=2)
    if forced != 16:
        raise ValueError("evaluation requires exactly sixteen teacher-forced tokens")
    readout = sequence["readout_token_range"]
    if (
        not isinstance(readout, list)
        or len(readout) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in readout)
        or not 1 <= readout[0] <= readout[1] <= forced
    ):
        raise ValueError("evaluation readout_token_range is invalid")
    if readout != [2, 16]:
        raise ValueError("evaluation readout must cover tokens two through sixteen")

    prompting = _mapping(top["prompting"], field="prompting", fields=_PROMPTING)
    if prompting != {
        "protocol": "paper-cot-templates/v1",
        "answer_extraction": "paper-task-extractors/v1",
    }:
        raise ValueError("evaluation prompting protocol differs")
    generation = dict(_mapping(top["generation"], field="generation", fields=_GENERATION))
    expected_generation = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "termination_protocol": "effective-eos-vs-length-cap/v1",
    }
    if generation != expected_generation:
        raise ValueError("evaluation generation protocol differs")

    metrics = _mapping(top["metrics"], field="metrics", fields=_METRICS)
    replicates = _integer(metrics["bootstrap_replicates"], field="bootstrap_replicates", minimum=1)
    bootstrap_seed = _integer(metrics["bootstrap_seed"], field="bootstrap_seed")
    confidence = _number(metrics["confidence_level"], field="confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("evaluation confidence_level must be between zero and one")
    seeds = metrics["seed_inventory"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or tuple(sorted(set(seeds))) != tuple(seeds)
    ):
        raise ValueError("evaluation seed_inventory must be unique and sorted")

    patch = _mapping(top["paired_patch"], field="paired_patch", fields=_PATCH)
    if patch != {
        "position": "edited-word-final-token",
        "window_source": "frozen-layer-selection",
    }:
        raise ValueError("evaluation paired-patch protocol differs")
    gate = dict(_mapping(top["gate"], field="gate", fields=_GATE))
    for field in (
        "require_wrong_to_right_above_right_to_wrong",
        "require_positive_unseen_task_gain",
    ):
        if type(gate[field]) is not bool:
            raise ValueError(f"evaluation gate.{field} must be boolean")
    for field in (
        "minimum_typo_accuracy_gain_points",
        "maximum_clean_accuracy_drop_points",
        "minimum_patch_gain_reduction_fraction",
    ):
        gate[field] = _number(gate[field], field=f"gate.{field}")
    gate["minimum_directional_seeds"] = _integer(
        gate["minimum_directional_seeds"], field="gate.minimum_directional_seeds", minimum=1
    )
    if gate["minimum_directional_seeds"] > len(seeds):
        raise ValueError("evaluation directional seed gate exceeds seed inventory")

    return RobustnessEvaluationProtocol(
        schema_version=str(top["schema_version"]),
        model=model_id,
        model_revision=revision,
        dtype="bfloat16",
        max_input_tokens=max_input,
        max_new_tokens=max_new,
        teacher_forced_tokens=forced,
        readout_token_range=(readout[0], readout[1]),
        prompt_protocol=str(prompting["protocol"]),
        answer_extraction=str(prompting["answer_extraction"]),
        generation=MappingProxyType(generation),
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence,
        seed_inventory=tuple(seeds),
        patch_position=str(patch["position"]),
        patch_window_source=str(patch["window_source"]),
        gate=MappingProxyType(gate),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["RobustnessEvaluationProtocol", "load_robustness_evaluation_config"]

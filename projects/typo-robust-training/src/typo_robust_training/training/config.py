"""Strict configuration for separate typo-robustness training conditions."""

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
_CONDITIONS = (
    "noisy-language-model",
    "output-matching",
    "global-state-alignment",
    "localized-state-distillation",
)
_LOSS_NAMES = ("noisy_language_model", "answer", "output", "state", "clean")
_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_TOP = {
    "schema_version",
    "condition",
    "model",
    "sequence",
    "adapter",
    "optimization",
    "objective",
}
_MODEL = {"id", "revision", "dtype"}
_SEQUENCE = {
    "max_length",
    "on_the_fly_typo",
    "natural_pairs",
    "answer_format",
}
_ADAPTER = {
    "method",
    "rank",
    "alpha",
    "dropout",
    "target_modules",
    "layer_scope",
    "bias",
    "task_type",
}
_OPTIMIZATION = {
    "optimizer",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "scheduler",
    "gradient_checkpointing",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "max_optimizer_steps",
    "max_grad_norm",
    "checkpoint_every_optimizer_steps",
    "log_every_micro_steps",
    "seed_inventory",
    "resume_contract",
}
_OBJECTIVE = {
    "weights",
    "state_scope",
    "state_distance",
    "output_scope",
    "noisy_language_model_scope",
    "temperature",
    "epsilon",
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
class AdapterTrainingProtocol:
    """One immutable scientific training condition."""

    schema_version: str
    condition: str
    model: str
    model_revision: str
    dtype: str
    max_sequence_length: int
    on_the_fly_typo: str
    natural_pairs: str
    answer_format: str
    adapter_method: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    layer_scope: str
    adapter_bias: str
    adapter_task_type: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    scheduler: str
    gradient_checkpointing: bool
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    max_grad_norm: float
    checkpoint_every_optimizer_steps: int
    log_every_micro_steps: int
    seed_inventory: tuple[int, ...]
    resume_contract: str
    loss_weights: Mapping[str, float]
    state_scope: str
    state_distance: str
    output_scope: str
    noisy_language_model_scope: str
    temperature: float
    epsilon: float
    config_sha256: str


def _validate_condition(
    condition: str,
    *,
    layer_scope: str,
    state_scope: str,
    weights: Mapping[str, float],
) -> None:
    valid = False
    if condition == "noisy-language-model":
        valid = (
            layer_scope == "all-decoder-layers"
            and state_scope == "none"
            and weights
            == {
                "noisy_language_model": 1.0,
                "answer": 0.0,
                "output": 0.0,
                "state": 0.0,
                "clean": 0.0,
            }
        )
    elif condition == "output-matching":
        valid = (
            layer_scope == "all-decoder-layers"
            and state_scope == "none"
            and weights["noisy_language_model"] == 0.0
            and weights["answer"] == 1.0
            and weights["output"] == 1.0
            and weights["state"] == 0.0
            and weights["clean"] in {0.25, 0.5, 1.0}
        )
    elif condition == "global-state-alignment":
        valid = (
            layer_scope == "all-decoder-layers"
            and state_scope == "all-layers-all-aligned-tokens"
            and weights["noisy_language_model"] == 0.0
            and weights["answer"] == 1.0
            and weights["output"] == 1.0
            and weights["state"] in {0.1, 0.5, 1.0}
            and weights["clean"] in {0.25, 0.5, 1.0}
        )
    elif condition == "localized-state-distillation":
        valid = (
            layer_scope == "selected-component-containing-layers"
            and state_scope == "selected-components-edited-word-final-tokens"
            and weights["noisy_language_model"] == 0.0
            and weights["answer"] == 1.0
            and weights["output"] == 1.0
            and weights["state"] in {0.1, 0.5, 1.0}
            and weights["clean"] in {0.25, 0.5, 1.0}
        )
    if not valid:
        raise ValueError("training condition and objective disagree")


def load_adapter_training_config(path: Path) -> AdapterTrainingProtocol:
    """Load JSON-in-YAML and reject any silent training-protocol drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter training config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        root = _mapping(
            strict_loads(raw.decode("utf-8"), context=str(resolved)),
            field="config",
            fields=_TOP,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"adapter training config is not UTF-8: {resolved}") from exc
    if root["schema_version"] != "robustness-adapter-training-config/v1":
        raise ValueError("adapter training schema_version differs")
    condition = _string(root["condition"], field="condition")
    if condition not in _CONDITIONS:
        raise ValueError("training condition is unsupported")

    model = _mapping(root["model"], field="model", fields=_MODEL)
    revision = _string(model["revision"], field="model.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be a pinned 40-character SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")

    sequence = _mapping(root["sequence"], field="sequence", fields=_SEQUENCE)
    expected_sequence = {
        "on_the_fly_typo": "record-epoch-counter-based/v1",
        "natural_pairs": "fixed-clean-typo-source-pairs/v1",
        "answer_format": "short-answer-suffix/v1",
    }
    for field, expected in expected_sequence.items():
        if sequence[field] != expected:
            raise ValueError(f"sequence.{field} differs from {expected}")

    adapter = _mapping(root["adapter"], field="adapter", fields=_ADAPTER)
    if adapter["method"] != "lora" or adapter["bias"] != "none":
        raise ValueError("adapter method or bias differs")
    if adapter["task_type"] != "CAUSAL_LM":
        raise ValueError("adapter.task_type must be CAUSAL_LM")
    target_modules = tuple(adapter["target_modules"])  # type: ignore[arg-type]
    if target_modules != _TARGET_MODULES:
        raise ValueError("adapter.target_modules differ from the frozen module order")
    dropout = _number(adapter["dropout"], field="adapter.dropout")
    if dropout > 1.0:
        raise ValueError("adapter.dropout must be at most one")

    optimization = _mapping(root["optimization"], field="optimization", fields=_OPTIMIZATION)
    expected_optimization = {
        "optimizer": "adamw",
        "scheduler": "cosine",
        "resume_contract": "exact-next-sample-and-rng/v1",
    }
    for field, expected in expected_optimization.items():
        if optimization[field] != expected:
            raise ValueError(f"optimization.{field} differs from {expected}")
    gradient_checkpointing = optimization["gradient_checkpointing"]
    if type(gradient_checkpointing) is not bool:
        raise ValueError("optimization.gradient_checkpointing must be boolean")
    seeds_raw = optimization["seed_inventory"]
    if not isinstance(seeds_raw, list):
        raise ValueError("optimization.seed_inventory must be a list")
    seeds = tuple(
        _integer(value, field="optimization.seed_inventory", minimum=0) for value in seeds_raw
    )
    if seeds != (42, 43, 44):
        raise ValueError("optimization.seed_inventory must be 42, 43, 44")
    warmup = _number(optimization["warmup_ratio"], field="optimization.warmup_ratio")
    if warmup > 1.0:
        raise ValueError("optimization.warmup_ratio must be at most one")

    objective = _mapping(root["objective"], field="objective", fields=_OBJECTIVE)
    weights_raw = _mapping(objective["weights"], field="objective.weights", fields=set(_LOSS_NAMES))
    weights = MappingProxyType(
        {
            name: _number(weights_raw[name], field=f"objective.weights.{name}")
            for name in _LOSS_NAMES
        }
    )
    state_scope = _string(objective["state_scope"], field="objective.state_scope")
    layer_scope = _string(adapter["layer_scope"], field="adapter.layer_scope")
    _validate_condition(
        condition,
        layer_scope=layer_scope,
        state_scope=state_scope,
        weights=weights,
    )
    if objective["state_distance"] != "normalized-squared-error/v1":
        raise ValueError("objective.state_distance differs")
    if objective["output_scope"] != "aligned-non-edited-next-token/v1":
        raise ValueError("objective.output_scope differs")
    if objective["noisy_language_model_scope"] != "all-nonpadding-next-tokens/v1":
        raise ValueError("objective.noisy_language_model_scope differs")

    return AdapterTrainingProtocol(
        schema_version="robustness-adapter-training-config/v1",
        condition=condition,
        model=_string(model["id"], field="model.id"),
        model_revision=revision,
        dtype="bfloat16",
        max_sequence_length=_integer(
            sequence["max_length"], field="sequence.max_length", minimum=2
        ),
        on_the_fly_typo=str(sequence["on_the_fly_typo"]),
        natural_pairs=str(sequence["natural_pairs"]),
        answer_format=str(sequence["answer_format"]),
        adapter_method="lora",
        lora_rank=_integer(adapter["rank"], field="adapter.rank", minimum=1),
        lora_alpha=_number(adapter["alpha"], field="adapter.alpha"),
        lora_dropout=dropout,
        lora_target_modules=_TARGET_MODULES,
        layer_scope=layer_scope,
        adapter_bias="none",
        adapter_task_type="CAUSAL_LM",
        optimizer="adamw",
        learning_rate=_number(optimization["learning_rate"], field="optimization.learning_rate"),
        weight_decay=_number(optimization["weight_decay"], field="optimization.weight_decay"),
        warmup_ratio=warmup,
        scheduler="cosine",
        gradient_checkpointing=gradient_checkpointing,
        micro_batch_size=_integer(
            optimization["micro_batch_size"],
            field="optimization.micro_batch_size",
            minimum=1,
        ),
        gradient_accumulation_steps=_integer(
            optimization["gradient_accumulation_steps"],
            field="optimization.gradient_accumulation_steps",
            minimum=1,
        ),
        max_optimizer_steps=_integer(
            optimization["max_optimizer_steps"],
            field="optimization.max_optimizer_steps",
            minimum=1,
        ),
        max_grad_norm=_number(optimization["max_grad_norm"], field="optimization.max_grad_norm"),
        checkpoint_every_optimizer_steps=_integer(
            optimization["checkpoint_every_optimizer_steps"],
            field="optimization.checkpoint_every_optimizer_steps",
            minimum=1,
        ),
        log_every_micro_steps=_integer(
            optimization["log_every_micro_steps"],
            field="optimization.log_every_micro_steps",
            minimum=1,
        ),
        seed_inventory=seeds,
        resume_contract=str(optimization["resume_contract"]),
        loss_weights=weights,
        state_scope=state_scope,
        state_distance=str(objective["state_distance"]),
        output_scope=str(objective["output_scope"]),
        noisy_language_model_scope=str(objective["noisy_language_model_scope"]),
        temperature=_number(objective["temperature"], field="objective.temperature", minimum=1e-12),
        epsilon=_number(objective["epsilon"], field="objective.epsilon", minimum=1e-12),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["AdapterTrainingProtocol", "load_adapter_training_config"]

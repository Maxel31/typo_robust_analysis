"""Strict configuration for separate typo-robustness training conditions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from typo_robust_training.data.config import strict_loads


_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONDITIONS = (
    "noisy-language-model",
    "output-matching",
    "random-window-state-distillation",
    "global-state-alignment",
    "localized-state-distillation",
    "probe-transition-output-matching",
    "probe-transition-single-layer-state-distillation",
    "probe-semantic-subspace-distillation",
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
_TOP_V4 = _TOP | {"method_evidence"}
_METHOD_EVIDENCE_V4 = {"schema_version", "artifact_sha256"}
_MODEL = {"id", "revision", "dtype"}
_MODEL_V2 = _MODEL | {"decoder_layers"}
_SEQUENCE = {
    "max_length",
    "on_the_fly_typo",
    "natural_pairs",
    "answer_format",
}
_SEQUENCE_V2 = _SEQUENCE | {"pairing_policy"}
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
_ADAPTER_V4 = _ADAPTER | {"layer_policy"}
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
_OPTIMIZATION_V3 = _OPTIMIZATION | {"max_student_tokens"}
_OBJECTIVE = {
    "weights",
    "state_scope",
    "state_distance",
    "output_scope",
    "noisy_language_model_scope",
    "temperature",
    "epsilon",
}
_OBJECTIVE_V2 = _OBJECTIVE | {
    "state_gradient_ratio",
    "calibration_micro_batches",
    "state_window_policy",
}
_GRADIENT_RATIO_GUARD_OPTIMIZER_STEPS = 50
_V6_FROZEN_SECTIONS: Mapping[str, object] = {
    "model": {
        "id": "google/gemma-3-4b-it",
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "dtype": "bfloat16",
        "decoder_layers": 34,
    },
    "sequence": {
        "max_length": 512,
        "on_the_fly_typo": "record-epoch-counter-based/v1",
        "natural_pairs": "fixed-clean-typo-source-pairs/v1",
        "answer_format": "no-hard-answer-target/v1",
        "pairing_policy": "exact-alternating-clean-noisy/v1",
    },
    "adapter": {
        "method": "lora",
        "rank": 16,
        "alpha": 8,
        "dropout": 0,
        "target_modules": list(_TARGET_MODULES),
        "layer_scope": "probe-transition-suffix",
        "layer_policy": "validated-probe-semantic-subspace-suffix/v1",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "optimization": {
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "scheduler": "constant-with-warmup",
        "gradient_checkpointing": True,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "max_optimizer_steps": 10000,
        "max_student_tokens": 10000000,
        "max_grad_norm": 1.0,
        "checkpoint_every_optimizer_steps": 50,
        "log_every_micro_steps": 1,
        "seed_inventory": [42, 43, 44],
        "resume_contract": "exact-next-sample-and-rng/v1",
    },
    "objective": {
        "weights": {
            "noisy_language_model": 0,
            "answer": 0,
            "output": 1,
            "state": 1,
            "clean": 0,
        },
        "state_scope": "probe-semantic-subspace-edited-word-final-token",
        "state_distance": "frozen-probe-classifier-forward-kl/v1",
        "output_scope": "aligned-non-edited-next-token/v1",
        "noisy_language_model_scope": "all-nonpadding-next-tokens/v1",
        "temperature": 1.0,
        "epsilon": 1e-8,
        "state_gradient_ratio": 0.05,
        "calibration_micro_batches": 8,
        "state_window_policy": "single-probe-transition-layer/v1",
    },
}
_V5_FROZEN_SECTIONS: Mapping[str, object] = {
    "model": _V6_FROZEN_SECTIONS["model"],
    "sequence": _V6_FROZEN_SECTIONS["sequence"],
    "adapter": {
        "method": "lora",
        "rank": 16,
        "alpha": 8,
        "dropout": 0,
        "target_modules": list(_TARGET_MODULES),
        "layer_scope": "probe-transition-suffix",
        "layer_policy": "validated-linear-probe-transition-suffix/v1",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "optimization": _V6_FROZEN_SECTIONS["optimization"],
    "objective": {
        "weights": {
            "noisy_language_model": 0,
            "answer": 0,
            "output": 1,
            "state": 1,
            "clean": 0,
        },
        "state_scope": "probe-transition-single-layer-edited-word-final-token/v1",
        "state_distance": "cosine-residual/v1",
        "output_scope": "aligned-non-edited-next-token/v1",
        "noisy_language_model_scope": "all-nonpadding-next-tokens/v1",
        "temperature": 1.0,
        "epsilon": 1e-8,
        "state_gradient_ratio": 0.05,
        "calibration_micro_batches": 8,
        "state_window_policy": "validated-probe-transition-single-layer/v1",
    },
}
_V4_FROZEN_SECTIONS: Mapping[str, object] = {
    "model": _V5_FROZEN_SECTIONS["model"],
    "sequence": _V5_FROZEN_SECTIONS["sequence"],
    "adapter": _V5_FROZEN_SECTIONS["adapter"],
    "optimization": _V5_FROZEN_SECTIONS["optimization"],
    "objective": {
        "weights": {
            "noisy_language_model": 0,
            "answer": 0,
            "output": 1,
            "state": 0,
            "clean": 0,
        },
        "state_scope": "none",
        "state_distance": "none",
        "output_scope": "aligned-non-edited-next-token/v1",
        "noisy_language_model_scope": "all-nonpadding-next-tokens/v1",
        "temperature": 1.0,
        "epsilon": 1e-8,
        "state_gradient_ratio": None,
        "calibration_micro_batches": 0,
        "state_window_policy": "none",
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    decoder_layers: int | None
    dtype: str
    max_sequence_length: int
    on_the_fly_typo: str
    natural_pairs: str
    answer_format: str
    pairing_policy: str
    adapter_method: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    layer_scope: str
    layer_policy: str
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
    max_student_tokens: int | None
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
    state_gradient_ratio: float | None
    gradient_ratio_guard_optimizer_steps: int
    calibration_micro_batches: int
    state_window_policy: str
    expected_method_evidence_sha256: str | None
    config_sha256: str


def _validate_condition(
    condition: str,
    *,
    layer_scope: str,
    state_scope: str,
    weights: Mapping[str, float],
    schema_version: str,
    state_window_policy: str,
    layer_policy: str,
) -> None:
    if schema_version == "robustness-adapter-training-config/v5":
        if (
            condition != "probe-transition-single-layer-state-distillation"
            or layer_scope != "probe-transition-suffix"
            or layer_policy != "validated-linear-probe-transition-suffix/v1"
            or state_scope
            != "probe-transition-single-layer-edited-word-final-token/v1"
            or state_window_policy
            != "validated-probe-transition-single-layer/v1"
            or dict(weights)
            != {
                "noisy_language_model": 0.0,
                "answer": 0.0,
                "output": 1.0,
                "state": 1.0,
                "clean": 0.0,
            }
        ):
            raise ValueError("probe-transition state training objective differs")
        return
    if schema_version == "robustness-adapter-training-config/v6":
        if (
            condition != "probe-semantic-subspace-distillation"
            or layer_scope != "probe-transition-suffix"
            or layer_policy != "validated-probe-semantic-subspace-suffix/v1"
            or state_scope != "probe-semantic-subspace-edited-word-final-token"
            or state_window_policy != "single-probe-transition-layer/v1"
            or dict(weights)
            != {
                "noisy_language_model": 0.0,
                "answer": 0.0,
                "output": 1.0,
                "state": 1.0,
                "clean": 0.0,
            }
        ):
            raise ValueError("probe semantic training condition and objective disagree")
        return
    if schema_version == "robustness-adapter-training-config/v4":
        if (
            condition != "probe-transition-output-matching"
            or layer_scope != "probe-transition-suffix"
            or layer_policy != "validated-linear-probe-transition-suffix/v1"
            or state_scope != "none"
            or state_window_policy != "none"
            or dict(weights)
            != {
                "noisy_language_model": 0.0,
                "answer": 0.0,
                "output": 1.0,
                "state": 0.0,
                "clean": 0.0,
            }
        ):
            raise ValueError("probe-transition training condition and objective disagree")
        return
    if schema_version in {
        "robustness-adapter-training-config/v2",
        "robustness-adapter-training-config/v3",
    }:
        if layer_policy != "legacy/v1":
            raise ValueError("cycle-2 adapter layer policy differs")
        expected_weights = {
            "noisy_language_model": 0.0,
            "answer": 0.0,
            "output": 1.0,
            "state": 0.0 if condition == "output-matching" else 1.0,
            "clean": 0.0,
        }
        expected_state = {
            "output-matching": ("none", "none"),
            "localized-state-distillation": (
                "causal-window-edited-word-final-tokens",
                "frozen-causal-window/v1",
            ),
            "random-window-state-distillation": (
                "random-window-edited-word-final-tokens",
                "sha256-seed42-middle-late-nonoverlap-same-width/v1",
            ),
            "global-state-alignment": (
                "all-layers-edited-word-final-tokens",
                "all-decoder-layers/v1",
            ),
        }
        if (
            condition not in expected_state
            or layer_scope != "all-decoder-layers"
            or dict(weights) != expected_weights
            or (state_scope, state_window_policy) != expected_state[condition]
        ):
            raise ValueError("cycle-2 training condition and objective disagree")
        return
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
    if not valid or state_window_policy != "legacy/v1":
        raise ValueError("training condition and objective disagree")


def load_adapter_training_config(path: Path) -> AdapterTrainingProtocol:
    """Load JSON-in-YAML and reject any silent training-protocol drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter training config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        decoded = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError(f"adapter training config is not UTF-8: {resolved}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("config must be an object")
    root = _mapping(
        decoded,
        field="config",
        fields=(
            _TOP_V4
            if decoded.get("schema_version")
            in {
                "robustness-adapter-training-config/v4",
                "robustness-adapter-training-config/v5",
                "robustness-adapter-training-config/v6",
            }
            else _TOP
        ),
    )
    schema_version = root["schema_version"]
    if schema_version not in {
        "robustness-adapter-training-config/v1",
        "robustness-adapter-training-config/v2",
        "robustness-adapter-training-config/v3",
        "robustness-adapter-training-config/v4",
        "robustness-adapter-training-config/v5",
        "robustness-adapter-training-config/v6",
    }:
        raise ValueError("adapter training schema_version differs")
    condition = _string(root["condition"], field="condition")
    if condition not in _CONDITIONS:
        raise ValueError("training condition is unsupported")
    if schema_version.endswith("/v6"):
        if condition != "probe-semantic-subspace-distillation":
            raise ValueError("semantic training condition differs from the frozen template")
        for section, expected in _V6_FROZEN_SECTIONS.items():
            if _canonical_json(root[section]) != _canonical_json(expected):
                raise ValueError(f"semantic training {section} differs from the frozen template")

    expected_method_evidence_sha256: str | None = None
    if schema_version.endswith(("/v4", "/v5", "/v6")):
        method_evidence = _mapping(
            root["method_evidence"],
            field="method_evidence",
            fields=_METHOD_EVIDENCE_V4,
        )
        expected_evidence_schema = {
            "robustness-adapter-training-config/v4": "probe-transition-evidence-binding/v1",
            "robustness-adapter-training-config/v5": (
                "probe-transition-state-gate-binding/v1"
            ),
            "robustness-adapter-training-config/v6": (
                "probe-semantic-subspace-evidence-binding/v1"
            ),
        }[str(schema_version)]
        if method_evidence["schema_version"] != expected_evidence_schema:
            raise ValueError("method_evidence.schema_version differs")
        digest = method_evidence["artifact_sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("method_evidence.artifact_sha256 must be a lowercase SHA-256")
        expected_method_evidence_sha256 = digest

    model = _mapping(
        root["model"],
        field="model",
        fields=_MODEL if schema_version.endswith("/v1") else _MODEL_V2,
    )
    revision = _string(model["revision"], field="model.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("model.revision must be a pinned 40-character SHA")
    if model["dtype"] != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")

    sequence = _mapping(
        root["sequence"],
        field="sequence",
        fields=_SEQUENCE if schema_version.endswith("/v1") else _SEQUENCE_V2,
    )
    expected_sequence: dict[str, object] = {
        "on_the_fly_typo": "record-epoch-counter-based/v1",
        "natural_pairs": "fixed-clean-typo-source-pairs/v1",
        "answer_format": (
            "short-answer-suffix/v1"
            if schema_version.endswith("/v1")
            else "no-hard-answer-target/v1"
        ),
    }
    if not schema_version.endswith("/v1"):
        expected_sequence["pairing_policy"] = "exact-alternating-clean-noisy/v1"
    for field, expected in expected_sequence.items():
        if sequence[field] != expected:
            raise ValueError(f"sequence.{field} differs from {expected}")

    adapter = _mapping(
        root["adapter"],
        field="adapter",
        fields=(
            _ADAPTER_V4
            if schema_version.endswith(("/v4", "/v5", "/v6"))
            else _ADAPTER
        ),
    )
    if adapter["method"] != "lora" or adapter["bias"] != "none":
        raise ValueError("adapter method or bias differs")
    if adapter["task_type"] != "CAUSAL_LM":
        raise ValueError("adapter.task_type must be CAUSAL_LM")
    target_modules = tuple(adapter["target_modules"])  # type: ignore[arg-type]
    if target_modules != _TARGET_MODULES:
        raise ValueError("adapter.target_modules differ from the frozen module order")
    lora_rank = _integer(adapter["rank"], field="adapter.rank", minimum=1)
    lora_alpha = _number(adapter["alpha"], field="adapter.alpha")
    dropout = _number(adapter["dropout"], field="adapter.dropout")
    if dropout > 1.0:
        raise ValueError("adapter.dropout must be at most one")
    if schema_version.endswith("/v4") and (
        lora_rank != 16 or lora_alpha != 8.0 or dropout != 0.0
    ):
        raise ValueError("probe-transition LoRA must use rank 16, alpha 8, and dropout 0")
    if schema_version.endswith("/v6") and (
        lora_rank != 16
        or lora_alpha != 8.0
        or dropout != 0.0
    ):
        raise ValueError("semantic training LoRA must be frozen at r16 alpha8 dropout0")

    optimization = _mapping(
        root["optimization"],
        field="optimization",
        fields=(
            _OPTIMIZATION_V3
            if schema_version.endswith(("/v3", "/v4", "/v5", "/v6"))
            else _OPTIMIZATION
        ),
    )
    expected_optimization = {
        "optimizer": "adamw",
        "scheduler": "cosine" if schema_version.endswith("/v1") else "constant-with-warmup",
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
    micro_batch_size = _integer(
        optimization["micro_batch_size"],
        field="optimization.micro_batch_size",
        minimum=1,
    )
    if micro_batch_size != 1:
        raise ValueError("optimization.micro_batch_size must equal 1")
    gradient_accumulation_steps = _integer(
        optimization["gradient_accumulation_steps"],
        field="optimization.gradient_accumulation_steps",
        minimum=1,
    )

    objective = _mapping(
        root["objective"],
        field="objective",
        fields=_OBJECTIVE if schema_version.endswith("/v1") else _OBJECTIVE_V2,
    )
    weights_raw = _mapping(objective["weights"], field="objective.weights", fields=set(_LOSS_NAMES))
    weights = MappingProxyType(
        {
            name: _number(weights_raw[name], field=f"objective.weights.{name}")
            for name in _LOSS_NAMES
        }
    )
    state_scope = _string(objective["state_scope"], field="objective.state_scope")
    layer_scope = _string(adapter["layer_scope"], field="adapter.layer_scope")
    layer_policy = (
        _string(adapter["layer_policy"], field="adapter.layer_policy")
        if schema_version.endswith(("/v4", "/v5", "/v6"))
        else "legacy/v1"
    )
    state_window_policy = (
        "legacy/v1"
        if schema_version.endswith("/v1")
        else _string(objective["state_window_policy"], field="objective.state_window_policy")
    )
    _validate_condition(
        condition,
        layer_scope=layer_scope,
        state_scope=state_scope,
        weights=weights,
        schema_version=str(schema_version),
        state_window_policy=state_window_policy,
        layer_policy=layer_policy,
    )
    expected_distance = (
        "normalized-squared-error/v1"
        if schema_version.endswith("/v1")
        else "frozen-probe-classifier-forward-kl/v1"
        if schema_version.endswith("/v6")
        else "none"
        if schema_version.endswith("/v4")
        else "cosine-residual/v1"
    )
    if objective["state_distance"] != expected_distance:
        raise ValueError("objective.state_distance differs")
    if objective["output_scope"] != "aligned-non-edited-next-token/v1":
        raise ValueError("objective.output_scope differs")
    if objective["noisy_language_model_scope"] != "all-nonpadding-next-tokens/v1":
        raise ValueError("objective.noisy_language_model_scope differs")
    temperature = _number(
        objective["temperature"], field="objective.temperature", minimum=1e-12
    )
    epsilon = _number(objective["epsilon"], field="objective.epsilon", minimum=1e-12)
    if schema_version.endswith("/v6") and (temperature != 1.0 or epsilon != 1e-8):
        raise ValueError("semantic distillation temperature and epsilon differ")

    if not schema_version.endswith("/v1"):
        ratio_raw = objective["state_gradient_ratio"]
        calibration = _integer(
            objective["calibration_micro_batches"],
            field="objective.calibration_micro_batches",
        )
        state_active = weights["state"] > 0.0 and state_scope != "none"
        if not state_active:
            if ratio_raw is not None or calibration != 0:
                raise ValueError("output-only training cannot calibrate a state loss")
            gradient_ratio = None
        else:
            gradient_ratio = _number(
                ratio_raw,
                field="objective.state_gradient_ratio",
                minimum=1e-12,
            )
            if gradient_ratio > 0.5 or calibration < 1:
                raise ValueError("state gradient calibration differs from the safe range")
            if schema_version.endswith("/v6") and (
                gradient_ratio != 0.05 or calibration != 8
            ):
                raise ValueError("semantic state calibration must be rho=0.05 over 8 batches")
    else:
        gradient_ratio = None
        calibration = 0

    if schema_version.endswith("/v5") and (
        gradient_ratio != 0.05
        or calibration != 8
        or _number(objective["temperature"], field="objective.temperature") != 1.0
        or _number(objective["epsilon"], field="objective.epsilon") != 1e-8
        or _integer(adapter["rank"], field="adapter.rank", minimum=1) != 16
        or _number(adapter["alpha"], field="adapter.alpha") != 8.0
        or dropout != 0.0
    ):
        raise ValueError("probe-transition state dosage or LoRA recipe differs")
    if schema_version.endswith("/v6") and (
        gradient_ratio != 0.05
        or calibration != 8
        or _number(objective["temperature"], field="objective.temperature") != 1.0
        or _number(objective["epsilon"], field="objective.epsilon") != 1e-8
        or _integer(adapter["rank"], field="adapter.rank", minimum=1) != 16
        or _number(adapter["alpha"], field="adapter.alpha") != 8.0
        or dropout != 0.0
    ):
        raise ValueError("semantic state dosage or LoRA recipe differs")

    pairing_policy = str(sequence.get("pairing_policy", "generator-probability/v1"))
    state_active = weights["state"] > 0.0 and state_scope != "none"
    if (
        state_active
        and pairing_policy == "exact-alternating-clean-noisy/v1"
        and (gradient_accumulation_steps < 2 or gradient_accumulation_steps % 2 != 0)
    ):
        raise ValueError(
            "state training with exact alternating pairs requires an even "
            "gradient_accumulation_steps >= 2"
        )

    frozen_recipe = (
        _V4_FROZEN_SECTIONS
        if schema_version.endswith("/v4")
        else _V5_FROZEN_SECTIONS
        if schema_version.endswith("/v5")
        else None
    )
    if frozen_recipe is not None:
        for section, expected in frozen_recipe.items():
            if _canonical_json(root[section]) != _canonical_json(expected):
                raise ValueError(
                    f"probe-transition {section} differs from the frozen 10M recipe"
                )

    return AdapterTrainingProtocol(
        schema_version=str(schema_version),
        condition=condition,
        model=_string(model["id"], field="model.id"),
        model_revision=revision,
        decoder_layers=(
            None
            if schema_version.endswith("/v1")
            else _integer(model["decoder_layers"], field="model.decoder_layers", minimum=2)
        ),
        dtype="bfloat16",
        max_sequence_length=_integer(
            sequence["max_length"], field="sequence.max_length", minimum=2
        ),
        on_the_fly_typo=str(sequence["on_the_fly_typo"]),
        natural_pairs=str(sequence["natural_pairs"]),
        answer_format=str(sequence["answer_format"]),
        pairing_policy=pairing_policy,
        adapter_method="lora",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        lora_target_modules=_TARGET_MODULES,
        layer_scope=layer_scope,
        layer_policy=layer_policy,
        adapter_bias="none",
        adapter_task_type="CAUSAL_LM",
        optimizer="adamw",
        learning_rate=_number(optimization["learning_rate"], field="optimization.learning_rate"),
        weight_decay=_number(optimization["weight_decay"], field="optimization.weight_decay"),
        warmup_ratio=warmup,
        scheduler=str(optimization["scheduler"]),
        gradient_checkpointing=gradient_checkpointing,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_optimizer_steps=_integer(
            optimization["max_optimizer_steps"],
            field="optimization.max_optimizer_steps",
            minimum=1,
        ),
        max_student_tokens=(
            _integer(
                optimization["max_student_tokens"],
                field="optimization.max_student_tokens",
                minimum=1,
            )
            if schema_version.endswith(("/v3", "/v4", "/v5", "/v6"))
            else None
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
        temperature=temperature,
        epsilon=epsilon,
        state_gradient_ratio=gradient_ratio,
        gradient_ratio_guard_optimizer_steps=(
            0 if schema_version.endswith("/v1") else _GRADIENT_RATIO_GUARD_OPTIMIZER_STEPS
        ),
        calibration_micro_batches=calibration,
        state_window_policy=state_window_policy,
        expected_method_evidence_sha256=expected_method_evidence_sha256,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["AdapterTrainingProtocol", "load_adapter_training_config"]

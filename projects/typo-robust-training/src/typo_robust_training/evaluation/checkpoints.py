"""Explicit adapter and frozen patch-window provenance for evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol
from typo_robust_training.integrity import sha256_file as _sha256_file
from typo_robust_training.integrity import sha256_tree
from typo_robust_training.training.provenance import (
    METHOD_EVIDENCE_CONDITIONS,
    SUPPORTED_ADAPTER_CONDITIONS,
    validate_condition_evidence,
)


_SHA64 = re.compile(r"[0-9a-f]{64}")
_REVISION40 = re.compile(r"[0-9a-f]{40}")
_CONDITIONS = (
    "noisy-language-model",
    "output-matching",
    "global-state-alignment",
    "localized-state-distillation",
    "random-window-state-distillation",
    "probe-transition-output-matching",
    "probe-transition-state-distillation",
    "causal-probe-subspace-distillation",
)
if frozenset(_CONDITIONS) != SUPPORTED_ADAPTER_CONDITIONS:  # pragma: no cover - import invariant
    raise RuntimeError("evaluation adapter condition order is incomplete")


def _object(path: Path, *, artifact: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"evaluation {artifact} is not a file: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"evaluation {artifact} must contain an object")
    return payload


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:
    path: Path
    condition: str
    seed: int
    config_sha256: str
    training_data_sha256: str
    data_identity_sha256: str
    localization_sha256: str | None
    adapter_sha256: str
    method_evidence_sha256: str | None = None

    @property
    def condition_id(self) -> str:
        return f"{self.condition}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class PatchWindow:
    start: int
    stop: int
    artifact_sha256: str
    localization_sha256: str | None = None

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


def _digest(value: object, *, field: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _SHA64.fullmatch(value) is None:
        raise ValueError(f"evaluation adapter {field} must be a SHA-256 digest")
    return value


def load_adapter_descriptors(
    paths: Sequence[Path],
    *,
    protocol: RobustnessEvaluationProtocol,
) -> tuple[AdapterDescriptor, ...]:
    """Validate every explicitly requested completed PEFT adapter."""

    if not isinstance(protocol, RobustnessEvaluationProtocol):
        raise TypeError("adapter evaluation requires a validated protocol")
    if not paths:
        raise ValueError("evaluation requires at least one adapter checkpoint")
    descriptors: list[AdapterDescriptor] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise ValueError(f"evaluation adapter is not a directory: {path}")
        run = _object(path.parent / "run.json", artifact="adapter training run")
        if (
            run.get("schema_version") != "robustness-adapter-training-run/v1"
            or run.get("status") != "completed"
        ):
            raise ValueError("evaluation adapter training run is not completed")
        condition = run.get("condition")
        seed = run.get("seed")
        if condition not in _CONDITIONS:
            raise ValueError("evaluation adapter condition is unsupported")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in protocol.seed_inventory
        ):
            raise ValueError("evaluation adapter seed is outside the frozen inventory")
        runtime = _object(path / "training_runtime.json", artifact="adapter runtime provenance")
        expected_runtime = {
            "model": protocol.model,
            "requested_revision": protocol.model_revision,
            "condition": condition,
            "seed": seed,
            "teacher_frozen": True,
            "student_base_frozen": True,
        }
        runtime_version = runtime.get("runtime")
        if runtime_version not in {
            "HuggingFaceAdapterTrainingRuntime/v1",
            "HuggingFaceAdapterTrainingRuntime/v2",
        } or any(runtime.get(field) != value for field, value in expected_runtime.items()):
            raise ValueError("evaluation adapter runtime identity differs")
        if condition in METHOD_EVIDENCE_CONDITIONS and (
            runtime_version != "HuggingFaceAdapterTrainingRuntime/v2"
        ):
            raise ValueError("evaluation v4 adapter requires runtime provenance v2")
        if runtime_version == "HuggingFaceAdapterTrainingRuntime/v2":
            actual_revision_fields = (
                "teacher_revision",
                "student_revision",
                "tokenizer_revision",
            )
            if any(
                runtime.get(field) != protocol.model_revision
                for field in actual_revision_fields
            ):
                raise ValueError("evaluation adapter actual runtime revision differs")
            code_revision = runtime.get("code_revision")
            if not isinstance(code_revision, str) or _REVISION40.fullmatch(code_revision) is None:
                raise ValueError("evaluation adapter code revision is not attested")
        adapter_config = _object(path / "adapter_config.json", artifact="PEFT adapter config")
        if (
            adapter_config.get("base_model_name_or_path") != protocol.model
            or adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("task_type") != "CAUSAL_LM"
        ):
            raise ValueError("evaluation PEFT adapter config identity differs")
        weights = tuple(
            candidate
            for candidate in (path / "adapter_model.safetensors", path / "adapter_model.bin")
            if candidate.is_file()
        )
        if len(weights) != 1:
            raise ValueError("evaluation adapter must contain exactly one PEFT weight file")
        try:
            localization, method_evidence = validate_condition_evidence(
                condition=condition,
                localization_sha256=run.get("localization_sha256"),
                method_evidence_sha256=run.get("method_evidence_sha256"),
            )
        except ValueError as exc:
            raise ValueError(f"evaluation {exc}") from exc
        runtime_method_evidence = runtime.get("method_evidence_sha256")
        if condition in METHOD_EVIDENCE_CONDITIONS:
            if (
                _digest(
                    runtime_method_evidence,
                    field="training_runtime.method_evidence_sha256",
                )
                != method_evidence
            ):
                raise ValueError("evaluation adapter runtime method evidence differs from run")
        elif runtime_method_evidence is not None:
            raise ValueError("evaluation legacy adapter runtime contains method evidence")
        outputs = run.get("outputs")
        adapter_output = outputs.get("adapter") if isinstance(outputs, Mapping) else None
        adapter_sha256 = sha256_tree(path)
        if (
            not isinstance(adapter_output, Mapping)
            or adapter_output.get("sha256") != adapter_sha256
        ):
            raise ValueError("evaluation adapter content hash differs from completed training")
        descriptors.append(
            AdapterDescriptor(
                path=path,
                condition=str(condition),
                seed=seed,
                config_sha256=str(_digest(run.get("config_sha256"), field="config_sha256")),
                training_data_sha256=str(
                    _digest(run.get("training_data_sha256"), field="training_data_sha256")
                ),
                data_identity_sha256=str(
                    _digest(run.get("data_identity_sha256"), field="data_identity_sha256")
                ),
                localization_sha256=localization,
                adapter_sha256=adapter_sha256,
                method_evidence_sha256=method_evidence,
            )
        )
    identities = [descriptor.condition_id for descriptor in descriptors]
    if len(set(identities)) != len(identities):
        raise ValueError("evaluation adapter identity is duplicated")
    for condition in _CONDITIONS:
        config_hashes = {
            descriptor.config_sha256
            for descriptor in descriptors
            if descriptor.condition == condition
        }
        if len(config_hashes) > 1:
            raise ValueError(f"evaluation {condition} training configuration differs across seeds")
        if condition in METHOD_EVIDENCE_CONDITIONS:
            evidence_hashes = {
                descriptor.method_evidence_sha256
                for descriptor in descriptors
                if descriptor.condition == condition
            }
            if len(evidence_hashes) > 1:
                raise ValueError(f"evaluation {condition} method evidence differs across seeds")
    condition_order = {condition: index for index, condition in enumerate(_CONDITIONS)}
    return tuple(sorted(descriptors, key=lambda item: (condition_order[item.condition], item.seed)))


def load_patch_window(
    path: Path,
    *,
    protocol: RobustnessEvaluationProtocol,
    validation_path: Path | None = None,
) -> PatchWindow:
    """Load one legacy window or independently validated generic-text window."""

    resolved = Path(path).resolve()
    payload = _object(resolved, artifact="layer selection")
    schema = payload.get("schema_version")
    expected = (
        {
            "operation": "select-distillation-layers",
            "model": protocol.model,
            "model_revision": protocol.model_revision,
        }
        if schema == "robustness-layer-selection/v1"
        else {
            "operation": "select-generic-joint-patch-window",
            "model": protocol.model,
            "model_revision": protocol.model_revision,
        }
        if schema == "robustness-joint-window-selection/v1"
        else {}
    )
    if not expected:
        raise ValueError("evaluation patch-window identity differs")
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError("evaluation patch-window identity differs")
    selected = payload.get("selected_window")
    expected_selected_fields = (
        {"start", "stop"}
        if schema == "robustness-layer-selection/v1"
        else {"start", "stop", "median_pairwise_restoration", "confidence_interval"}
    )
    if not isinstance(selected, Mapping) or set(selected) != expected_selected_fields:
        raise ValueError("evaluation patch-window coordinates differ")
    start, stop = selected.get("start"), selected.get("stop")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop
    ):
        raise ValueError("evaluation patch-window coordinates are invalid")
    selection_sha256 = _sha256_file(resolved)
    if schema == "robustness-layer-selection/v1":
        if validation_path is not None:
            raise ValueError("legacy evaluation patch-window cannot bind generic validation")
        artifact_sha256 = selection_sha256
        localization_sha256 = None
    else:
        if validation_path is None:
            raise ValueError("generic evaluation patch-window requires independent validation")
        width = payload.get("window_width")
        decoder_layers = payload.get("decoder_layers")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width != stop - start
            or isinstance(decoder_layers, bool)
            or not isinstance(decoder_layers, int)
            or stop > decoder_layers
        ):
            raise ValueError("generic evaluation patch-window dimensions differ")
        validation_resolved = Path(validation_path).resolve()
        validation = _object(validation_resolved, artifact="window validation")
        validation_expected = {
            "schema_version": "robustness-joint-window-validation/v1",
            "operation": "validate-generic-joint-patch-window",
            "model": protocol.model,
            "model_revision": protocol.model_revision,
            "config_sha256": payload.get("config_sha256"),
            "window_selection_sha256": selection_sha256,
            "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
        }
        if any(validation.get(field) != value for field, value in validation_expected.items()):
            raise ValueError("generic evaluation window validation identity differs")
        validation_window = validation.get("selected_window")
        if (
            not isinstance(validation_window, Mapping)
            or set(validation_window) != {"start", "stop"}
            or validation_window.get("start") != start
            or validation_window.get("stop") != stop
        ):
            raise ValueError("generic evaluation window validation coordinates differ")
        interval = validation.get("confidence_interval")
        if (
            validation.get("passed") is not True
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in interval
            )
            or float(interval[0]) <= 0.0
            or float(interval[0]) > float(interval[1])
        ):
            raise ValueError("generic evaluation window did not pass independent validation")
        validation_sha256 = _sha256_file(validation_resolved)
        artifact_sha256 = hashlib.sha256(
            (
                "validated-evaluation-patch-window/v1\0"
                f"{selection_sha256}\0{validation_sha256}\0{start}:{stop}"
            ).encode()
        ).hexdigest()
        policy = "frozen-causal-window/v1"
        localization_sha256 = hashlib.sha256(
            (
                f"residual-state-evidence/v1\0{selection_sha256}\0"
                f"{validation_sha256}\0{policy}\0" + ",".join(map(str, range(start, stop)))
            ).encode()
        ).hexdigest()
    return PatchWindow(
        start=start,
        stop=stop,
        artifact_sha256=artifact_sha256,
        localization_sha256=localization_sha256,
    )


__all__ = [
    "AdapterDescriptor",
    "PatchWindow",
    "load_adapter_descriptors",
    "load_patch_window",
]

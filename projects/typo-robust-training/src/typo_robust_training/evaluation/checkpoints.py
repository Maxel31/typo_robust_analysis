"""Explicit adapter and frozen patch-window provenance for evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol


_SHA64 = re.compile(r"[0-9a-f]{64}")
_CONDITIONS = (
    "noisy-language-model",
    "output-matching",
    "global-state-alignment",
    "localized-state-distillation",
)


def _object(path: Path, *, artifact: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"evaluation {artifact} is not a file: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"evaluation {artifact} must contain an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("evaluation adapter must contain regular files only")
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:
    path: Path
    condition: str
    seed: int
    config_sha256: str
    training_data_sha256: str
    localization_sha256: str | None
    adapter_sha256: str

    @property
    def condition_id(self) -> str:
        return f"{self.condition}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class PatchWindow:
    start: int
    stop: int
    artifact_sha256: str

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
            "runtime": "HuggingFaceAdapterTrainingRuntime/v1",
            "model": protocol.model,
            "requested_revision": protocol.model_revision,
            "condition": condition,
            "seed": seed,
            "teacher_frozen": True,
            "student_base_frozen": True,
        }
        if any(runtime.get(field) != value for field, value in expected_runtime.items()):
            raise ValueError("evaluation adapter runtime identity differs")
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
        localization = _digest(
            run.get("localization_sha256"),
            field="localization_sha256",
            optional=True,
        )
        if (condition == "localized-state-distillation") != (localization is not None):
            raise ValueError("evaluation adapter localization provenance differs")
        descriptors.append(
            AdapterDescriptor(
                path=path,
                condition=str(condition),
                seed=seed,
                config_sha256=str(_digest(run.get("config_sha256"), field="config_sha256")),
                training_data_sha256=str(
                    _digest(run.get("training_data_sha256"), field="training_data_sha256")
                ),
                localization_sha256=localization,
                adapter_sha256=_tree_sha256(path),
            )
        )
    identities = [descriptor.condition_id for descriptor in descriptors]
    if len(set(identities)) != len(identities):
        raise ValueError("evaluation adapter identity is duplicated")
    condition_order = {condition: index for index, condition in enumerate(_CONDITIONS)}
    return tuple(sorted(descriptors, key=lambda item: (condition_order[item.condition], item.seed)))


def load_patch_window(path: Path, *, protocol: RobustnessEvaluationProtocol) -> PatchWindow:
    """Load only a layer-selection artifact from the same pinned model."""

    resolved = Path(path).resolve()
    payload = _object(resolved, artifact="layer selection")
    expected = {
        "schema_version": "robustness-layer-selection/v1",
        "operation": "select-distillation-layers",
        "model": protocol.model,
        "model_revision": protocol.model_revision,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError("evaluation patch-window identity differs")
    selected = payload.get("selected_window")
    if not isinstance(selected, Mapping) or set(selected) != {"start", "stop"}:
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
    return PatchWindow(start=start, stop=stop, artifact_sha256=_sha256_file(resolved))


__all__ = [
    "AdapterDescriptor",
    "PatchWindow",
    "load_adapter_descriptors",
    "load_patch_window",
]

"""Adapter and patch-window inputs are explicit, complete, and hash-bound."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.evaluation.checkpoints import (
    load_adapter_descriptors,
    load_patch_window,
)
from typo_robust_training.evaluation.config import load_robustness_evaluation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_robustness_evaluation_config(PROJECT_ROOT / "configs/gemma4b-evaluation.yaml")


def _adapter(root: Path, *, condition: str, seed: int) -> Path:
    output = root / condition / f"seed-{seed}"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": PROTOCOL.model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "training_runtime.json").write_text(
        json.dumps(
            {
                "runtime": "HuggingFaceAdapterTrainingRuntime/v1",
                "model": PROTOCOL.model,
                "requested_revision": PROTOCOL.model_revision,
                "condition": condition,
                "seed": seed,
                "teacher_frozen": True,
                "student_base_frozen": True,
            }
        ),
        encoding="utf-8",
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "robustness-adapter-training-run/v1",
                "status": "completed",
                "condition": condition,
                "seed": seed,
                "config_sha256": "a" * 64,
                "training_data_sha256": "b" * 64,
                "localization_sha256": "c" * 64
                if condition == "localized-state-distillation"
                else None,
            }
        ),
        encoding="utf-8",
    )
    return adapter


def _layers(path: Path) -> Path:
    payload = {
        "schema_version": "robustness-layer-selection/v1",
        "operation": "select-distillation-layers",
        "model": PROTOCOL.model,
        "model_revision": PROTOCOL.model_revision,
        "selected_window": {"start": 0, "stop": 6},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_explicit_adapters_and_layer_window_are_validated_and_content_hashed(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _adapter(
            tmp_path,
            condition="localized-state-distillation",
            seed=seed,
        )
        for seed in (42, 43, 44)
    )
    descriptors = load_adapter_descriptors(paths, protocol=PROTOCOL)
    assert [descriptor.condition_id for descriptor in descriptors] == [
        f"localized-state-distillation:seed-{seed}" for seed in (42, 43, 44)
    ]
    assert all(len(descriptor.adapter_sha256) == 64 for descriptor in descriptors)
    assert all(descriptor.training_data_sha256 == "b" * 64 for descriptor in descriptors)
    assert all(descriptor.localization_sha256 == "c" * 64 for descriptor in descriptors)

    window = load_patch_window(_layers(tmp_path / "layers.json"), protocol=PROTOCOL)
    assert window.layers == tuple(range(6))
    assert len(window.artifact_sha256) == 64


def test_adapter_loader_rejects_duplicate_identity_incomplete_run_or_runtime_drift(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        tmp_path,
        condition="localized-state-distillation",
        seed=42,
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_adapter_descriptors((adapter, adapter), protocol=PROTOCOL)

    run_path = adapter.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["status"] = "failed"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        load_adapter_descriptors((adapter,), protocol=PROTOCOL)

    run["status"] = "completed"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    runtime_path = adapter / "training_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["requested_revision"] = "d" * 40
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime identity"):
        load_adapter_descriptors((adapter,), protocol=PROTOCOL)


def test_patch_window_rejects_model_or_revision_drift(tmp_path: Path) -> None:
    path = _layers(tmp_path / "layers.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "different/model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_patch_window(path, protocol=PROTOCOL)

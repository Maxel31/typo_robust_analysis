"""Adapter and patch-window inputs are explicit, complete, and hash-bound."""

from __future__ import annotations

import hashlib
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


def _tree_sha256(root: Path) -> str:
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _adapter(
    root: Path,
    *,
    condition: str,
    seed: int,
    runtime_version: str = "v2",
    method_evidence_sha256: str | None = None,
) -> Path:
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
                "runtime": f"HuggingFaceAdapterTrainingRuntime/{runtime_version}",
                "model": PROTOCOL.model,
                "requested_revision": PROTOCOL.model_revision,
                **(
                    {
                        "teacher_revision": PROTOCOL.model_revision,
                        "student_revision": PROTOCOL.model_revision,
                        "tokenizer_revision": PROTOCOL.model_revision,
                        "code_revision": "f" * 40,
                    }
                    if runtime_version == "v2"
                    else {}
                ),
                "condition": condition,
                "seed": seed,
                "teacher_frozen": True,
                "student_base_frozen": True,
                **(
                    {"method_evidence_sha256": method_evidence_sha256}
                    if method_evidence_sha256 is not None
                    else {}
                ),
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
                "data_identity_sha256": "e" * 64,
                "localization_sha256": "c" * 64
                if condition == "localized-state-distillation"
                else None,
                **(
                    {"method_evidence_sha256": method_evidence_sha256}
                    if method_evidence_sha256 is not None
                    else {}
                ),
                "outputs": {"adapter": {"sha256": _tree_sha256(adapter)}},
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


def _generic_layers(path: Path) -> Path:
    payload = {
        "schema_version": "robustness-joint-window-selection/v1",
        "operation": "select-generic-joint-patch-window",
        "model": PROTOCOL.model,
        "model_revision": PROTOCOL.model_revision,
        "config_sha256": "d" * 64,
        "decoder_layers": 34,
        "window_width": 6,
        "selected_window": {
            "start": 0,
            "stop": 6,
            "median_pairwise_restoration": 0.75,
            "confidence_interval": [0.6, 0.8],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _generic_validation(path: Path, selection: Path) -> Path:
    payload = {
        "schema_version": "robustness-joint-window-validation/v1",
        "operation": "validate-generic-joint-patch-window",
        "model": PROTOCOL.model,
        "model_revision": PROTOCOL.model_revision,
        "config_sha256": "d" * 64,
        "window_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "selected_window": {"start": 0, "stop": 6},
        "confidence_interval": [0.5, 0.7],
        "validation_rule": "bootstrap-95ci-lower-strictly-positive/v1",
        "passed": True,
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
    assert all(descriptor.data_identity_sha256 == "e" * 64 for descriptor in descriptors)
    assert all(descriptor.localization_sha256 == "c" * 64 for descriptor in descriptors)

    window = load_patch_window(_layers(tmp_path / "layers.json"), protocol=PROTOCOL)
    assert window.layers == tuple(range(6))
    assert len(window.artifact_sha256) == 64
    assert window.localization_sha256 is None


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


def test_adapter_loader_rejects_post_completion_mutation_or_mixed_seed_configs(
    tmp_path: Path,
) -> None:
    seed_42 = _adapter(
        tmp_path,
        condition="localized-state-distillation",
        seed=42,
    )
    (seed_42 / "adapter_model.safetensors").write_bytes(b"substituted-weights")
    with pytest.raises(ValueError, match="content hash"):
        load_adapter_descriptors((seed_42,), protocol=PROTOCOL)

    seed_43 = _adapter(
        tmp_path,
        condition="output-matching",
        seed=43,
    )
    seed_44 = _adapter(
        tmp_path,
        condition="output-matching",
        seed=44,
    )
    run_path = seed_44.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["config_sha256"] = "f" * 64
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs across seeds"):
        load_adapter_descriptors((seed_43, seed_44), protocol=PROTOCOL)


@pytest.mark.parametrize(
    "condition",
    (
        "probe-transition-output-matching",
        "probe-transition-state-distillation",
        "causal-probe-subspace-distillation",
    ),
)
def test_v4_adapter_requires_generic_method_evidence(
    tmp_path: Path,
    condition: str,
) -> None:
    missing = _adapter(tmp_path / "missing", condition=condition, seed=42)
    with pytest.raises(ValueError, match="method evidence provenance"):
        load_adapter_descriptors((missing,), protocol=PROTOCOL)

    adapter = _adapter(
        tmp_path / "valid",
        condition=condition,
        seed=42,
        method_evidence_sha256="9" * 64,
    )
    descriptor = load_adapter_descriptors((adapter,), protocol=PROTOCOL)[0]
    assert descriptor.method_evidence_sha256 == "9" * 64
    assert descriptor.localization_sha256 is None


@pytest.mark.parametrize(
    "field",
    ("teacher_revision", "student_revision", "tokenizer_revision"),
)
def test_v2_adapter_rejects_missing_or_mismatched_actual_revision(
    tmp_path: Path,
    field: str,
) -> None:
    for label, replacement in (("missing", None), ("mismatch", "0" * 40)):
        adapter = _adapter(
            tmp_path / label,
            condition="output-matching",
            seed=42,
        )
        runtime_path = adapter / "training_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if replacement is None:
            del runtime[field]
        else:
            runtime[field] = replacement
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

        with pytest.raises(ValueError, match="actual runtime revision"):
            load_adapter_descriptors((adapter,), protocol=PROTOCOL)


def test_v2_adapter_requires_attested_code_revision(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, condition="output-matching", seed=42)
    runtime_path = adapter / "training_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["code_revision"] = "not-a-commit"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="code revision is not attested"):
        load_adapter_descriptors((adapter,), protocol=PROTOCOL)


def test_v4_adapter_rejects_legacy_runtime_or_runtime_evidence_drift(tmp_path: Path) -> None:
    legacy = _adapter(
        tmp_path / "legacy-runtime",
        condition="probe-transition-output-matching",
        seed=42,
        runtime_version="v1",
        method_evidence_sha256="9" * 64,
    )
    with pytest.raises(ValueError, match="requires runtime provenance v2"):
        load_adapter_descriptors((legacy,), protocol=PROTOCOL)

    for label, replacement in (("missing", None), ("mismatch", "8" * 64)):
        adapter = _adapter(
            tmp_path / label,
            condition="probe-transition-output-matching",
            seed=42,
            method_evidence_sha256="9" * 64,
        )
        runtime_path = adapter / "training_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if replacement is None:
            del runtime["method_evidence_sha256"]
        else:
            runtime["method_evidence_sha256"] = replacement
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="training_runtime.method_evidence_sha256|runtime method evidence differs",
        ):
            load_adapter_descriptors((adapter,), protocol=PROTOCOL)


def test_v4_adapter_rejects_mixed_seed_evidence_and_unsupported_condition(
    tmp_path: Path,
) -> None:
    seed_42 = _adapter(
        tmp_path,
        condition="probe-transition-output-matching",
        seed=42,
        method_evidence_sha256="8" * 64,
    )
    seed_43 = _adapter(
        tmp_path,
        condition="probe-transition-output-matching",
        seed=43,
        method_evidence_sha256="9" * 64,
    )
    with pytest.raises(ValueError, match="method evidence differs across seeds"):
        load_adapter_descriptors((seed_42, seed_43), protocol=PROTOCOL)

    unsupported = _adapter(tmp_path, condition="invented-method", seed=44)
    with pytest.raises(ValueError, match="condition is unsupported"):
        load_adapter_descriptors((unsupported,), protocol=PROTOCOL)


def test_legacy_adapter_manifest_does_not_require_method_evidence(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        condition="output-matching",
        seed=42,
        runtime_version="v1",
    )
    run = json.loads((adapter.parent / "run.json").read_text(encoding="utf-8"))
    run.pop("method_evidence_sha256", None)
    (adapter.parent / "run.json").write_text(json.dumps(run), encoding="utf-8")

    descriptor = load_adapter_descriptors((adapter,), protocol=PROTOCOL)[0]
    assert descriptor.localization_sha256 is None
    assert descriptor.method_evidence_sha256 is None


def test_patch_window_rejects_model_or_revision_drift(tmp_path: Path) -> None:
    path = _layers(tmp_path / "layers.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "different/model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_patch_window(path, protocol=PROTOCOL)


def test_current_training_runtime_and_independently_validated_window_are_accepted(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        tmp_path,
        condition="localized-state-distillation",
        seed=42,
        runtime_version="v2",
    )
    descriptors = load_adapter_descriptors((adapter,), protocol=PROTOCOL)
    assert descriptors[0].condition_id == "localized-state-distillation:seed-42"

    selection = _generic_layers(tmp_path / "window_selection.json")
    validation = _generic_validation(tmp_path / "window_validation.json", selection)
    window = load_patch_window(
        selection,
        validation_path=validation,
        protocol=PROTOCOL,
    )

    assert window.layers == tuple(range(6))
    assert len(window.artifact_sha256) == 64
    expected_localization = hashlib.sha256(
        (
            "residual-state-evidence/v1\0"
            f"{hashlib.sha256(selection.read_bytes()).hexdigest()}\0"
            f"{hashlib.sha256(validation.read_bytes()).hexdigest()}\0"
            "frozen-causal-window/v1\0" + ",".join(map(str, range(6)))
        ).encode()
    ).hexdigest()
    assert window.localization_sha256 == expected_localization


def test_generic_patch_window_requires_matching_passing_validation(tmp_path: Path) -> None:
    selection = _generic_layers(tmp_path / "window_selection.json")
    with pytest.raises(ValueError, match="independent validation"):
        load_patch_window(selection, protocol=PROTOCOL)

    validation = _generic_validation(tmp_path / "window_validation.json", selection)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["passed"] = False
    validation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        load_patch_window(selection, validation_path=validation, protocol=PROTOCOL)

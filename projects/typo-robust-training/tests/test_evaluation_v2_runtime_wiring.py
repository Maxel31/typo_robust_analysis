"""Adversarial checks for evaluation-v2 phase gates at real runner boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.evaluation.calibration_v2 import load_evaluation_v2_protocol
from typo_robust_training.evaluation.runner import (
    RobustnessEvaluationRunConfig,
    run_robustness_evaluation,
)
from typo_robust_training.evaluation.runtime_registry_v2 import (
    load_evaluation_v2_runtime_registry_bundle,
    validate_confirmatory_evaluation_opening,
    validate_confirmatory_training_runtime,
)
from typo_robust_training.integrity import sha256_file, sha256_tree
from typo_robust_training.training.runner import (
    AdapterTrainingRunConfig,
    run_adapter_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_PROTOCOL = PROJECT_ROOT / "configs/robustness-evaluation-v2.yaml"
FAITHFUL_CONFIG = (
    PROJECT_ROOT / "configs/baselines/mistral7b-v01-kojima-faithful-output-matching-64m.yaml"
)
FACTORIAL_TEMPLATE = (
    PROJECT_ROOT / "configs/proposals/gemma4b-probe-output-factorial-10m.template.yaml"
)
FACTORIAL = (
    "factorial-all-layers-all-tokens",
    "factorial-all-layers-downstream-horizon",
    "factorial-probe-suffix-all-tokens",
    "factorial-probe-suffix-downstream-horizon",
    "factorial-random-layers-downstream-horizon",
)
MODELS = (
    ("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767"),
    ("mistralai/Mistral-7B-v0.1", "7231864981174d9bee8c7687c24c8344414eae6b"),
)
FAITHFUL = "kojima-faithful-output-matching"
ZERO = "0" * 64


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _factorial_config(tmp_path: Path, probe_sha: str) -> Path:
    value = json.loads(FACTORIAL_TEMPLATE.read_text(encoding="utf-8"))
    value["schema_version"] = "robustness-adapter-training-config/v7"
    value["condition"] = FACTORIAL[0]
    value["method_evidence"]["artifact_sha256"] = probe_sha
    value["adapter"]["layer_scope"] = "all-decoder-layers"
    value["adapter"]["layer_policy"] = "all-decoder-layers/v1"
    value["objective"]["output_scope"] = "aligned-non-edited-next-token/v1"
    return _write_json(tmp_path / "factorial-config.json", value)


def _registries(
    tmp_path: Path,
    *,
    config_path: Path,
    data_dir: Path,
    probe_path: Path,
) -> dict[str, Path]:
    config_sha = sha256_file(config_path)
    data_sha = sha256_tree(data_dir)
    probe_sha = sha256_file(probe_path)
    config_entries = [
        {
            "model_id": model,
            "model_revision": revision,
            "condition": condition,
            "config_sha256": (
                config_sha if (model, condition) == (MODELS[0][0], FACTORIAL[0]) else ZERO
            ),
        }
        for model, revision in MODELS
        for condition in FACTORIAL
    ] + [
        {
            "model_id": MODELS[1][0],
            "model_revision": MODELS[1][1],
            "condition": FAITHFUL,
            "config_sha256": ZERO,
        }
    ]
    data_entries = [
        {
            "model_id": model,
            "model_revision": revision,
            "condition": condition,
            "seed": seed,
            "training_data_tree_sha256": (
                data_sha if (model, condition, seed) == (MODELS[0][0], FACTORIAL[0], 42) else ZERO
            ),
        }
        for model, revision in MODELS
        for condition in FACTORIAL
        for seed in (42, 43, 44)
    ] + [
        {
            "model_id": MODELS[1][0],
            "model_revision": MODELS[1][1],
            "condition": FAITHFUL,
            "seed": seed,
            "training_data_tree_sha256": ZERO,
        }
        for seed in (1, 42, 43, 44)
    ]
    probe_entries = [
        {
            "model_id": model,
            "model_revision": revision,
            "artifact_sha256": probe_sha if model == MODELS[0][0] else ZERO,
        }
        for model, revision in MODELS
    ]
    arm_entries = [
        {
            "model_id": model,
            "model_revision": revision,
            "condition": condition,
            "config_sha256": (
                config_sha if (model, condition) == (MODELS[0][0], FACTORIAL[0]) else ZERO
            ),
            "probe_artifact_sha256": probe_sha if model == MODELS[0][0] else ZERO,
        }
        for model, revision in MODELS
        for condition in FACTORIAL
    ]
    return {
        "training_config_registry_path": _write_json(
            tmp_path / "training-config-registry.json",
            {
                "schema_version": "robustness-evaluation-v2-training-config-registry/v1",
                "entries": config_entries,
            },
        ),
        "training_data_registry_path": _write_json(
            tmp_path / "training-data-registry.json",
            {
                "schema_version": "robustness-evaluation-v2-training-data-registry/v1",
                "entries": data_entries,
            },
        ),
        "probe_artifact_registry_path": _write_json(
            tmp_path / "probe-artifact-registry.json",
            {
                "schema_version": "robustness-evaluation-v2-probe-artifact-registry/v1",
                "entries": probe_entries,
            },
        ),
        "factorial_arm_registry_path": _write_json(
            tmp_path / "factorial-arm-registry.json",
            {
                "schema_version": "robustness-evaluation-v2-factorial-arm-registry/v1",
                "entries": arm_entries,
            },
        ),
    }


def _bundle(
    tmp_path: Path,
    *,
    phase: str,
    registries: dict[str, Path],
    arm_checkpoint_registry: Path | None = None,
) -> Path:
    common = {
        "registry_path": "phase-registry.json",
        "protocol_path": str(V2_PROTOCOL),
        "repository_path": "repository",
        "calibration_observations_path": "base-observations.jsonl",
        "calibration_item_manifest_path": "calibration-items.jsonl",
        "calibration_typo_manifest_path": "calibration-typos.jsonl",
        "calibration_result_path": "calibration-result.json",
        "confirmatory_item_manifest_path": "evaluation/confirmatory-items.jsonl",
        "confirmatory_typo_manifest_path": "evaluation/confirmatory-typos.jsonl",
        "tier_role_manifest_path": "tier-role-manifest.jsonl",
        "factorial_arm_registry_path": str(registries["factorial_arm_registry_path"]),
        "probe_artifact_registry_path": str(registries["probe_artifact_registry_path"]),
        "training_config_registry_path": str(registries["training_config_registry_path"]),
        "training_data_registry_path": str(registries["training_data_registry_path"]),
        "legacy_random_2_registry_path": "legacy-random-2-registry.json",
    }
    if phase == "training-preregistered":
        post = {
            "training_preregistered_registry_path": None,
            "mistral_matched_seed_registry_path": None,
            "mistral_public_seed_1_checkpoint_path": None,
            "arm_checkpoint_registry_path": None,
            "opening_log_path": None,
        }
    else:
        post = {
            "training_preregistered_registry_path": "training-preregistered.json",
            "mistral_matched_seed_registry_path": "mistral-matched.json",
            "mistral_public_seed_1_checkpoint_path": "mistral-seed1.json",
            "arm_checkpoint_registry_path": str(arm_checkpoint_registry),
            "opening_log_path": "opening-log.json",
        }
    _write_json(tmp_path / "phase-registry.json", {"phase": phase})
    return _write_json(
        tmp_path / f"{phase}-bundle.json",
        {
            "schema_version": "robustness-evaluation-v2-runtime-registry-bundle/v1",
            "phase": phase,
            "artifacts": {**common, **post},
        },
    )


def test_training_runtime_binds_exact_config_data_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "probe.json"
    probe.write_text("frozen probe\n", encoding="utf-8")
    config = _factorial_config(tmp_path, sha256_file(probe))
    data = tmp_path / "data"
    data.mkdir()
    (data / "records.jsonl").write_text("frozen data\n", encoding="utf-8")
    registries = _registries(tmp_path, config_path=config, data_dir=data, probe_path=probe)
    bundle = _bundle(tmp_path, phase="training-preregistered", registries=registries)
    protocol = load_evaluation_v2_protocol(V2_PROTOCOL)
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2._load_phase",
        lambda _bundle_value: (protocol, {"state": "training-preregistered"}),
    )

    binding = validate_confirmatory_training_runtime(
        bundle_path=bundle,
        condition=FACTORIAL[0],
        seed=42,
        config_path=config,
        training_data_dir=data,
        probe_selection_path=probe,
    )
    assert binding.config_sha256 == sha256_file(config)
    assert binding.training_data_tree_sha256 == sha256_tree(data)
    assert binding.probe_artifact_sha256 == sha256_file(probe)

    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(config)
    with pytest.raises(ValueError, match="training config cannot be a symbolic link"):
        validate_confirmatory_training_runtime(
            bundle_path=bundle,
            condition=FACTORIAL[0],
            seed=42,
            config_path=config_link,
            training_data_dir=data,
            probe_selection_path=probe,
        )
    data_link = tmp_path / "data-link"
    data_link.symlink_to(data, target_is_directory=True)
    with pytest.raises(ValueError, match="training data directory cannot be a symbolic link"):
        validate_confirmatory_training_runtime(
            bundle_path=bundle,
            condition=FACTORIAL[0],
            seed=42,
            config_path=config,
            training_data_dir=data_link,
            probe_selection_path=probe,
        )
    probe_link = tmp_path / "probe-link.json"
    probe_link.symlink_to(probe)
    with pytest.raises(ValueError, match="probe artifact cannot be a symbolic link"):
        validate_confirmatory_training_runtime(
            bundle_path=bundle,
            condition=FACTORIAL[0],
            seed=42,
            config_path=config,
            training_data_dir=data,
            probe_selection_path=probe_link,
        )

    (data / "tamper.txt").write_text("late substitution\n", encoding="utf-8")
    with pytest.raises(ValueError, match="concrete training data hash differs"):
        validate_confirmatory_training_runtime(
            bundle_path=bundle,
            condition=FACTORIAL[0],
            seed=42,
            config_path=config,
            training_data_dir=data,
            probe_selection_path=probe,
        )
    (data / "tamper.txt").unlink()

    probe.write_text("different probe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="factorial config/probe binding differs"):
        validate_confirmatory_training_runtime(
            bundle_path=bundle,
            condition=FACTORIAL[0],
            seed=42,
            config_path=config,
            training_data_dir=data,
            probe_selection_path=probe,
        )


def test_runtime_bundle_rejects_wrong_phase_and_missing_registry(tmp_path: Path) -> None:
    registries = {
        name: _write_json(tmp_path / f"{name}.json", {"placeholder": name})
        for name in (
            "factorial_arm_registry_path",
            "probe_artifact_registry_path",
            "training_config_registry_path",
            "training_data_registry_path",
        )
    }
    wrong = _bundle(tmp_path, phase="evaluation-opening-sealed", registries=registries)
    with pytest.raises(ValueError, match="not training-preregistered"):
        load_evaluation_v2_runtime_registry_bundle(wrong, required_phase="training-preregistered")
    with pytest.raises(ValueError, match="requires --evaluation-v2-registry-bundle"):
        validate_confirmatory_training_runtime(
            bundle_path=None,
            condition=FAITHFUL,
            seed=42,
            config_path=FAITHFUL_CONFIG,
            training_data_dir=tmp_path,
            probe_selection_path=None,
        )
    bundle_link = tmp_path / "bundle-link.json"
    bundle_link.symlink_to(wrong)
    with pytest.raises(ValueError, match="bundle cannot be a symbolic link"):
        load_evaluation_v2_runtime_registry_bundle(
            bundle_link, required_phase="evaluation-opening-sealed"
        )
    linked_registry = tmp_path / "linked-phase-registry.json"
    linked_registry.symlink_to(tmp_path / "phase-registry.json")
    payload = json.loads(wrong.read_text(encoding="utf-8"))
    payload["artifacts"]["registry_path"] = str(linked_registry)
    registry_link_bundle = _write_json(tmp_path / "registry-link-bundle.json", payload)
    with pytest.raises(ValueError, match="registry_path cannot be a symbolic link"):
        load_evaluation_v2_runtime_registry_bundle(
            registry_link_bundle, required_phase="evaluation-opening-sealed"
        )


def test_training_phase_failure_precedes_data_or_gpu_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"data": False}

    def reject_phase(**_kwargs):
        raise ValueError("wrong frozen phase")

    def forbidden_data(*_args, **_kwargs):
        called["data"] = True
        raise AssertionError("data construction occurred before phase validation")

    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2.validate_confirmatory_training_runtime",
        reject_phase,
    )
    monkeypatch.setattr(
        "typo_robust_training.training.runner._load_protocol_training_bundle", forbidden_data
    )
    with pytest.raises(ValueError, match="wrong frozen phase"):
        run_adapter_training(
            AdapterTrainingRunConfig(
                condition=FAITHFUL,
                config_path=FAITHFUL_CONFIG,
                training_data_dir=tmp_path / "unread-data",
                layer_selection_path=None,
                component_selection_path=None,
                seed=42,
                gpu_id="0",
                wandb_project=None,
                wandb_entity=None,
                output_dir=tmp_path / "output",
                evaluation_v2_registry_bundle_path=tmp_path / "wrong-phase.json",
            )
        )
    assert called == {"data": False}


def _checkpoint(root: Path, *, model: tuple[str, str], condition: str, seed: int) -> Path:
    run_root = root / f"{condition}-seed-{seed}"
    adapter = run_root / "adapter"
    adapter.mkdir(parents=True)
    _write_json(
        adapter / "training_runtime.json",
        {
            "model": model[0],
            "requested_revision": model[1],
            "condition": condition,
            "seed": seed,
        },
    )
    (adapter / "weights.bin").write_bytes(f"{condition}:{seed}".encode())
    _write_json(run_root / "run.json", {"condition": condition, "seed": seed})
    return adapter


def _checkpoint_registry(tmp_path: Path) -> tuple[Path, list[Path]]:
    gemma_checkpoints = [
        _checkpoint(tmp_path / "checkpoints", model=MODELS[0], condition=condition, seed=seed)
        for condition in FACTORIAL
        for seed in (42, 43, 44)
    ]
    hashes = {
        (condition, seed): sha256_tree(checkpoint)
        for condition in FACTORIAL
        for seed, checkpoint in (
            (seed, gemma_checkpoints[FACTORIAL.index(condition) * 3 + index])
            for index, seed in enumerate((42, 43, 44))
        )
    }
    entries = [
        {
            "model_id": model,
            "model_revision": revision,
            "condition": condition,
            "seed": seed,
            "adapter_tree_sha256": (hashes[(condition, seed)] if model == MODELS[0][0] else ZERO),
        }
        for model, revision in MODELS
        for condition in FACTORIAL
        for seed in (42, 43, 44)
    ] + [
        {
            "model_id": MODELS[1][0],
            "model_revision": MODELS[1][1],
            "condition": FAITHFUL,
            "seed": seed,
            "adapter_tree_sha256": ZERO,
        }
        for seed in (42, 43, 44)
    ]
    path = _write_json(
        tmp_path / "arm-checkpoint-registry.json",
        {
            "schema_version": "robustness-evaluation-v2-arm-checkpoint-registry/v1",
            "entries": entries,
        },
    )
    return path, gemma_checkpoints


def test_opening_rejects_cherry_picked_checkpoint_subset_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("placeholder\n", encoding="utf-8")
    registries = {
        name: placeholder
        for name in (
            "factorial_arm_registry_path",
            "probe_artifact_registry_path",
            "training_config_registry_path",
            "training_data_registry_path",
        )
    }
    checkpoint_registry, checkpoints = _checkpoint_registry(tmp_path)
    bundle = _bundle(
        tmp_path,
        phase="evaluation-opening-sealed",
        registries=registries,
        arm_checkpoint_registry=checkpoint_registry,
    )
    protocol = load_evaluation_v2_protocol(V2_PROTOCOL)
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2._load_phase",
        lambda _bundle_value: (protocol, {"state": "evaluation-opening-sealed"}),
    )
    evaluation_data = tmp_path / "evaluation"
    evaluation_data.mkdir()
    evaluation_data_link = tmp_path / "evaluation-link"
    evaluation_data_link.symlink_to(evaluation_data, target_is_directory=True)
    with pytest.raises(ValueError, match="evaluation data directory cannot be a symbolic link"):
        validate_confirmatory_evaluation_opening(
            bundle_path=bundle,
            checkpoint_paths=checkpoints,
            evaluation_data_dir=evaluation_data_link,
        )
    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoints[0], target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint directory cannot be a symbolic link"):
        validate_confirmatory_evaluation_opening(
            bundle_path=bundle,
            checkpoint_paths=(checkpoint_link,),
            evaluation_data_dir=evaluation_data,
        )
    with pytest.raises(ValueError, match="checkpoint batch is incomplete"):
        validate_confirmatory_evaluation_opening(
            bundle_path=bundle,
            checkpoint_paths=checkpoints[:1],
            evaluation_data_dir=evaluation_data,
        )
    binding = validate_confirmatory_evaluation_opening(
        bundle_path=bundle,
        checkpoint_paths=checkpoints,
        evaluation_data_dir=evaluation_data,
    )
    assert len(binding.checkpoint_tree_sha256) == 15
    runtime_path = checkpoints[0] / "training_runtime.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    _write_json(runtime_path, {**runtime_payload, "seed": 43})
    with pytest.raises(ValueError, match="runtime/run identity differs"):
        validate_confirmatory_evaluation_opening(
            bundle_path=bundle,
            checkpoint_paths=checkpoints,
            evaluation_data_dir=evaluation_data,
        )
    _write_json(runtime_path, runtime_payload)
    (checkpoints[0] / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="concrete checkpoint hash differs"):
        validate_confirmatory_evaluation_opening(
            bundle_path=bundle,
            checkpoint_paths=checkpoints,
            evaluation_data_dir=evaluation_data,
        )


def test_evaluation_phase_failure_precedes_protocol_or_gpu_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"protocol": False}

    def reject_opening(**_kwargs):
        raise ValueError("opening is not sealed")

    def forbidden_protocol(*_args, **_kwargs):
        called["protocol"] = True
        raise AssertionError("protocol/model construction occurred before opening validation")

    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2.confirmatory_evaluation_is_required",
        lambda _checkpoint_paths: True,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2.validate_confirmatory_evaluation_opening",
        reject_opening,
    )
    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_robustness_evaluation_config",
        forbidden_protocol,
    )
    with pytest.raises(ValueError, match="opening is not sealed"):
        run_robustness_evaluation(
            RobustnessEvaluationRunConfig(
                config_path=tmp_path / "config.json",
                study_protocol_path=tmp_path / "study.json",
                training_data_dir=tmp_path / "training",
                evaluation_data_dir=tmp_path / "evaluation",
                evaluation_role="final-test",
                layer_selection_path=tmp_path / "window.json",
                window_validation_path=None,
                checkpoint_paths=(tmp_path / "checkpoint",),
                splits=("same-task",),
                gpu_id="0",
                output_dir=tmp_path / "output",
                confirm_sealed_role=True,
                resume=False,
                evaluation_v2_registry_bundle_path=tmp_path / "opening-bundle.json",
            )
        )
    assert called == {"protocol": False}


def test_confirmatory_evaluation_cannot_omit_opening_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"protocol": False}

    monkeypatch.setattr(
        "typo_robust_training.evaluation.runtime_registry_v2.confirmatory_evaluation_is_required",
        lambda _checkpoint_paths: True,
    )

    def forbidden_protocol(*_args, **_kwargs):
        called["protocol"] = True
        raise AssertionError("protocol/model construction occurred before missing-phase failure")

    monkeypatch.setattr(
        "typo_robust_training.evaluation.runner.load_robustness_evaluation_config",
        forbidden_protocol,
    )
    with pytest.raises(ValueError, match="requires --evaluation-v2-registry-bundle"):
        run_robustness_evaluation(
            RobustnessEvaluationRunConfig(
                config_path=tmp_path / "config.json",
                study_protocol_path=tmp_path / "study.json",
                training_data_dir=tmp_path / "training",
                evaluation_data_dir=tmp_path / "evaluation",
                evaluation_role="final-test",
                layer_selection_path=tmp_path / "window.json",
                window_validation_path=None,
                checkpoint_paths=(tmp_path / "checkpoint",),
                splits=("same-task",),
                gpu_id="0",
                output_dir=tmp_path / "output",
                confirm_sealed_role=True,
                resume=False,
            )
        )
    assert called == {"protocol": False}


def test_cli_requires_v2_bundle_only_for_confirmatory_training() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    register_commands(commands)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train-kojima-faithful-output-matching",
                "--config",
                str(FAITHFUL_CONFIG),
                "--training-data",
                "data",
                "--seed",
                "42",
                "--gpu-id",
                "0",
                "--wandb-project",
                "test",
                "--output-dir",
                "output",
            ]
        )
    legacy = parser.parse_args(
        [
            "train-output-matching",
            "--config",
            "legacy.json",
            "--training-data",
            "data",
            "--seed",
            "42",
            "--gpu-id",
            "0",
            "--wandb-project",
            "test",
            "--output-dir",
            "output",
        ]
    )
    assert not hasattr(legacy, "evaluation_v2_registry_bundle")

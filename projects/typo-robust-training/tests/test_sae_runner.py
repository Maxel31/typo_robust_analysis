"""CPU-only integration checks for SAE training artifacts and exact resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from typo_robust_training.sae.data import PreparedSaeSources
from typo_robust_training.sae.model import SparseAutoencoder
from typo_robust_training.sae.runner import (
    SaeCalibrationRunConfig,
    SaeTrainingRunConfig,
    _load_training_checkpoint,
    _load_final_saes,
    _load_wp2_attempts,
    _record_wp2_attempt,
    _save_training_checkpoint,
    run_calibrate_sae_l1,
    run_train_saes,
)
from typo_robust_training.sae.runtime import ActivationBuffer


class _Tracker:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.status: str | None = None

    def log_optimizer_step(self, _metrics, *, optimizer_step: int) -> None:
        self.steps.append(optimizer_step)

    def finish(self, *, status: str, summary) -> None:
        self.status = status

    def provenance(self):
        return {"provider": "fake-wandb/v1"}


class _Runtime:
    def __init__(self, *, protocol, gpu_id: str) -> None:
        self.protocol = protocol
        self.gpu_id = gpu_id
        self.device = torch.device("cpu")
        self.calls = 0

    def iter_activation_buffers(
        self,
        _sources,
        *,
        layer_indices,
        target_tokens: int,
        start_source_index: int = 0,
        start_source_offset: int = 0,
        start_buffer_index: int = 0,
    ):
        self.calls += 1
        values = {
            layer: torch.arange(target_tokens * 2, dtype=torch.bfloat16).reshape(target_tokens, 2)
            + layer
            for layer in layer_indices
        }
        yield ActivationBuffer(
            activations_by_layer=values,
            source_start=start_source_index,
            source_stop=start_source_index + 1,
            next_source_index=start_source_index + 1,
            next_source_offset=start_source_offset,
            tokens=target_tokens,
            buffer_index=start_buffer_index,
        )

    def provenance(self):
        return {"runtime": "fake-sae-runtime/v1"}


class _MultiBufferCalibrationRuntime:
    def __init__(self, *, protocol, gpu_id: str) -> None:
        self.protocol = protocol
        self.gpu_id = gpu_id
        self.device = torch.device("cpu")
        self.calls = 0
        self.streams: list[list[tuple[int, ...]]] = []

    def iter_activation_buffers(
        self,
        _sources,
        *,
        layer_indices,
        target_tokens: int,
        **_kwargs,
    ):
        assert target_tokens == 4
        self.calls += 1
        stream: list[tuple[int, ...]] = []
        self.streams.append(stream)
        for buffer_index, offset in enumerate((0, 2)):
            token_ids = tuple(range(offset, offset + 2))
            stream.append(token_ids)
            values = {
                layer: torch.tensor(
                    [[float(token), float(token * token + layer + 1)] for token in token_ids],
                    dtype=torch.bfloat16,
                )
                for layer in layer_indices
            }
            yield ActivationBuffer(
                activations_by_layer=values,
                source_start=offset,
                source_stop=offset + 1,
                next_source_index=offset + 1,
                next_source_offset=0,
                tokens=2,
                buffer_index=buffer_index,
            )

    def provenance(self):
        return {"runtime": "fake-multi-buffer-calibration/v1"}


def test_sae_calibration_rejects_a_gpu_that_disagrees_with_the_amendment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace()
    preregistration = SimpleNamespace(sae_gpu_id=0)
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )

    with pytest.raises(ValueError, match="must use preregistered GPU 0"):
        run_calibrate_sae_l1(
            SaeCalibrationRunConfig(
                config_path=tmp_path / "config.json",
                registry_path=tmp_path / "registry.json",
                training_data_paths=(tmp_path / "source.jsonl",),
                gpu_id="1",
                wandb_project="test-sae",
                wandb_entity=None,
                output_dir=tmp_path / "calibration",
            )
        )

    assert not (tmp_path / "calibration").exists()


def test_sae_calibration_closes_tracker_when_runtime_initialization_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        model="model",
        model_revision="b" * 40,
        expansion_factor=2,
        l1_calibration_tokens=4,
    )
    preregistration = SimpleNamespace(sha256="c" * 64, sae_gpu_id=1)
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(),),
        reserved=(SimpleNamespace(),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_tokens=10,
        protected_eligible_records=1,
        protected_eligible_source_tokens=10,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=1,
    )

    def fake_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text("{}\n", encoding="utf-8")
        return prepared

    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr("typo_robust_training.sae.runner._load_inputs", fake_inputs)
    tracker = _Tracker()

    def failing_runtime(**_kwargs):
        raise RuntimeError("SAE runtime requires exactly one requested CUDA GPU")

    with pytest.raises(RuntimeError, match="exactly one requested CUDA GPU"):
        run_calibrate_sae_l1(
            SaeCalibrationRunConfig(
                config_path=tmp_path / "config.json",
                registry_path=tmp_path / "registry.json",
                training_data_paths=(tmp_path / "source.jsonl",),
                gpu_id="1",
                wandb_project="test-sae",
                wandb_entity=None,
                output_dir=tmp_path / "calibration",
            ),
            runtime_factory=failing_runtime,
            tracker_factory=lambda **_kwargs: tracker,
        )

    assert tracker.status == "failed"


def test_sae_calibration_streams_and_replays_multiple_frozen_buffers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        model="model",
        model_revision="b" * 40,
        expansion_factor=2,
        d_model=2,
        d_sae=4,
        probe_layers=(5, 20),
        l1_coefficients=(0.01,),
        l1_calibration_tokens=4,
        learning_rate=1e-3,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        activation_batch_size=2,
        dead_feature_probability_below=1e-5,
        median_l0_range=(0, 4),
        l1_selection_rule="in-range-median-l0-then-lowest-fvu/v1",
    )
    preregistration = SimpleNamespace(sha256="c" * 64, sae_gpu_id=0)
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(),),
        reserved=(SimpleNamespace(),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_tokens=4,
        protected_eligible_records=1,
        protected_eligible_source_tokens=4,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=1,
    )

    def fake_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text("{}\n", encoding="utf-8")
        return prepared

    runtime = _MultiBufferCalibrationRuntime(protocol=protocol, gpu_id="0")
    tracker = _Tracker()
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr("typo_robust_training.sae.runner._load_inputs", fake_inputs)

    result = run_calibrate_sae_l1(
        SaeCalibrationRunConfig(
            config_path=tmp_path / "config.json",
            registry_path=tmp_path / "registry.json",
            training_data_paths=(tmp_path / "source.jsonl",),
            gpu_id="0",
            wandb_project="test-sae",
            wandb_entity=None,
            output_dir=tmp_path / "calibration",
        ),
        runtime_factory=lambda **_kwargs: runtime,
        tracker_factory=lambda **_kwargs: tracker,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert runtime.calls == 2
    assert runtime.streams == [[(0, 1), (2, 3)], [(0, 1), (2, 3)]]
    assert report["training_activation_buffers"] == 2
    assert report["evaluation_activation_buffers"] == 2
    assert report["evaluation_activation_tokens"] == 4
    assert report["selection_status"] == "selected"
    assert report["selection_error"] is None
    assert report["optimizer_steps"] == 2
    assert tracker.steps == [1, 2]
    assert tracker.status == "completed"


def test_sae_calibration_preserves_metrics_when_no_candidate_is_selectable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        model="model",
        model_revision="b" * 40,
        expansion_factor=2,
        d_model=2,
        d_sae=4,
        probe_layers=(5, 20),
        l1_coefficients=(0.01,),
        l1_calibration_tokens=4,
        learning_rate=1e-3,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        activation_batch_size=2,
        dead_feature_probability_below=1e-5,
        median_l0_range=(5, 6),
        l1_selection_rule="in-range-median-l0-then-lowest-fvu/v1",
    )
    preregistration = SimpleNamespace(sha256="c" * 64, sae_gpu_id=0)
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(),),
        reserved=(SimpleNamespace(),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_tokens=4,
        protected_eligible_records=1,
        protected_eligible_source_tokens=4,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=1,
    )

    def fake_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text("{}\n", encoding="utf-8")
        return prepared

    runtime = _MultiBufferCalibrationRuntime(protocol=protocol, gpu_id="0")
    tracker = _Tracker()
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr("typo_robust_training.sae.runner._load_inputs", fake_inputs)
    output = tmp_path / "calibration"

    with pytest.raises(RuntimeError, match="has no L1 candidate"):
        run_calibrate_sae_l1(
            SaeCalibrationRunConfig(
                config_path=tmp_path / "config.json",
                registry_path=tmp_path / "registry.json",
                training_data_paths=(tmp_path / "source.jsonl",),
                gpu_id="0",
                wandb_project="test-sae",
                wandb_entity=None,
                output_dir=output,
            ),
            runtime_factory=lambda **_kwargs: runtime,
            tracker_factory=lambda **_kwargs: tracker,
        )

    report = json.loads((output / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["selection_status"] == "failed"
    assert report["selection_error"].startswith("SAE layer 5 has no L1 candidate")
    assert report["candidate_metrics"]["5"]["0.01"]
    assert report["candidate_metrics"]["20"]["0.01"]
    assert report["selected_by_layer"] == {}
    assert not (output / "l1_selection.json").exists()
    assert runtime.calls == 2
    assert tracker.status == "failed"


def test_sae_training_writes_hash_bound_models_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        model="model",
        model_revision="b" * 40,
        expansion_factor=2,
        d_model=2,
        d_sae=4,
        learning_rate=1e-3,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        probe_layers=(5, 20),
        activation_subsample_layers=(5, 11, 20, 26),
        seeds_by_layer={5: (42, 43), 20: (42,)},
        minimum_training_tokens=4,
        statistics_tokens=1,
        activation_subsample_tokens=2,
        activation_batch_size=2,
    )
    preregistration = SimpleNamespace(sha256="c" * 64, sae_gpu_id=1)
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(),),
        reserved=(SimpleNamespace(),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_tokens=10,
        protected_eligible_records=1,
        protected_eligible_source_tokens=10,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=1,
    )

    def fake_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text("{}\n", encoding="utf-8")
        return prepared

    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr("typo_robust_training.sae.runner._load_inputs", fake_inputs)
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._load_l1_selection",
        lambda *_args, **_kwargs: {5: 1e-4, 20: 3e-4},
    )
    selection = tmp_path / "l1.json"
    selection.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "training"
    trackers: list[_Tracker] = []

    def tracker_factory(**_kwargs):
        tracker = _Tracker()
        trackers.append(tracker)
        return tracker

    base = dict(
        config_path=tmp_path / "config.json",
        registry_path=tmp_path / "registry.json",
        training_data_paths=(tmp_path / "source.jsonl",),
        l1_selection_path=selection,
        gpu_id="1",
        wandb_project="test-sae",
        wandb_entity=None,
        output_dir=output,
    )
    result = run_train_saes(
        SaeTrainingRunConfig(**base),
        runtime_factory=_Runtime,
        tracker_factory=tracker_factory,
    )
    assert result.trained_tokens == 4
    assert result.optimizer_steps == 2
    assert trackers[-1].steps == [1, 2]
    assert trackers[-1].status == "completed"
    assert (output / "checkpoint.json").is_file()
    assert len(tuple(output.glob("sae-layer-*.pt"))) == 3

    (output / "run.json").unlink()
    resumed_from_checkpoint = run_train_saes(
        SaeTrainingRunConfig(**base, resume=True),
        runtime_factory=_Runtime,
        tracker_factory=tracker_factory,
    )
    assert resumed_from_checkpoint.trained_tokens == 4
    assert resumed_from_checkpoint.optimizer_steps == 2
    assert trackers[-1].steps == []
    assert trackers[-1].status == "completed"

    tracker_count = len(trackers)
    resumed = run_train_saes(
        SaeTrainingRunConfig(**base, resume=True),
        runtime_factory=_Runtime,
        tracker_factory=tracker_factory,
    )
    assert resumed.trained_tokens == 4
    assert resumed.optimizer_steps == 2
    assert len(trackers) == tracker_count


def test_sae_checkpoint_metadata_failure_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "training"
    output.mkdir()
    bindings = {"config_sha256": "a" * 64}
    model = SparseAutoencoder(d_model=2, d_sae=4, seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    models = {"layer-5-seed-42": model}
    optimizers = {"layer-5-seed-42": optimizer}
    first_cursor = {
        "trained_tokens": 4,
        "optimizer_steps": 2,
        "next_source_index": 1,
        "next_source_offset": 0,
        "next_buffer_index": 1,
    }
    _save_training_checkpoint(
        output=output,
        bindings=bindings,
        models=models,
        optimizers=optimizers,
        cursor=first_cursor,
    )
    first_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    second_cursor = dict(first_cursor, trained_tokens=8, optimizer_steps=4, next_buffer_index=2)

    def fail_metadata_commit(*_args, **_kwargs):
        raise OSError("simulated metadata commit failure")

    monkeypatch.setattr(
        "typo_robust_training.sae.runner.write_json_atomic",
        fail_metadata_commit,
    )
    with pytest.raises(OSError, match="simulated metadata commit failure"):
        _save_training_checkpoint(
            output=output,
            bindings=bindings,
            models=models,
            optimizers=optimizers,
            cursor=second_cursor,
        )

    resumed_model = SparseAutoencoder(d_model=2, d_sae=4, seed=7)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    loaded_cursor = _load_training_checkpoint(
        output=output,
        bindings=bindings,
        models={"layer-5-seed-42": resumed_model},
        optimizers={"layer-5-seed-42": resumed_optimizer},
    )
    assert loaded_cursor == first_cursor
    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, first_parameters[name])


def test_wp2_attempt_ledger_enforces_one_preregistered_retrain(tmp_path: Path) -> None:
    protocol = SimpleNamespace(config_sha256="a" * 64, maximum_gate_retrains=1)
    preregistration = SimpleNamespace(sha256="b" * 64)

    def checkpoint(name: str, value: int) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "run.json").write_text(json.dumps({"value": value}) + "\n", encoding="utf-8")
        return root

    first = checkpoint("training-first", 1)
    ledger, attempts, digest = _load_wp2_attempts(
        checkpoint_dir=first,
        protocol=protocol,
        preregistration=preregistration,
    )
    _record_wp2_attempt(
        ledger_path=ledger,
        prior_attempts=attempts,
        checkpoint_run_sha256=digest,
        output_dir=tmp_path / "validation-first",
        passed=False,
        acceptance_sha256="c" * 64,
        protocol=protocol,
        preregistration=preregistration,
    )

    second = checkpoint("training-retrain", 2)
    ledger, attempts, digest = _load_wp2_attempts(
        checkpoint_dir=second,
        protocol=protocol,
        preregistration=preregistration,
    )
    _record_wp2_attempt(
        ledger_path=ledger,
        prior_attempts=attempts,
        checkpoint_run_sha256=digest,
        output_dir=tmp_path / "validation-retrain",
        passed=False,
        acceptance_sha256="d" * 64,
        protocol=protocol,
        preregistration=preregistration,
    )

    third = checkpoint("training-prohibited", 3)
    with pytest.raises(ValueError, match="retrain limit is exhausted"):
        _load_wp2_attempts(
            checkpoint_dir=third,
            protocol=protocol,
            preregistration=preregistration,
        )


def test_final_sae_loader_rejects_a_different_preregistration(tmp_path: Path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    source_registry = root / "source_registry.json"
    source_registry.write_text("{}\n", encoding="utf-8")
    bindings = {
        "config_sha256": "a" * 64,
        "preregistration_sha256": "old-" + "b" * 60,
    }
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "robustness-sae-training-run/v1",
                "operation": "train-sparse-autoencoders",
                "bindings": bindings,
                "cursor": {},
                "l1_by_model": {},
                "models": {},
                "source_registry_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                "runtime": {},
                "wandb": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        seeds_by_layer={5: (42, 43), 20: (42,)},
    )
    with pytest.raises(ValueError, match="training bindings differ"):
        _load_final_saes(
            checkpoint_dir=root,
            protocol=protocol,
            preregistration_sha256="new-" + "c" * 60,
            device=torch.device("cpu"),
        )

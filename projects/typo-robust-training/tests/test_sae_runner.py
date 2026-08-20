"""CPU-only integration checks for SAE training artifacts and exact resume."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from typo_robust_training.sae import runner as runner_module
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
from typo_robust_training.sae.retry import reserve_initial_wp2_validation
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


class _CrashAfterFirstStepTracker(_Tracker):
    def log_optimizer_step(self, metrics, *, optimizer_step: int) -> None:
        super().log_optimizer_step(metrics, optimizer_step=optimizer_step)
        raise RuntimeError("simulated crash before the first checkpoint")


def _install_minimal_training_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retry: bool,
    minimum_training_tokens: int = 4,
) -> dict[str, object]:
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
        probe_layers=(5,),
        activation_subsample_layers=(5,),
        seeds_by_layer={5: (42,)},
        minimum_training_tokens=minimum_training_tokens,
        protected_student_token_budget=10_000_000,
        statistics_tokens=1,
        activation_subsample_tokens=2,
        activation_batch_size=2,
    )
    preregistration = SimpleNamespace(
        sha256="c" * 64,
        sae_gpu_id=0,
        retry_inputs=SimpleNamespace() if retry else None,
    )
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(record_id="record", token_count=minimum_training_tokens + 1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
        source_tokens=minimum_training_tokens + 1,
        protected_eligible_records=1,
        protected_eligible_source_tokens=minimum_training_tokens + 1,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=0,
    )

    def write_sources(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_bytes(
            runner_module._source_registry_bytes(
                protocol=protocol,
                preregistration=preregistration,
                prepared=prepared,
            )
        )
        return prepared

    monkeypatch.setattr(runner_module, "load_sae_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        runner_module,
        "load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    if retry:
        monkeypatch.setattr(runner_module, "_prepare_inputs", lambda **_kwargs: prepared)
        monkeypatch.setattr(runner_module, "_write_source_registry", write_sources)
        monkeypatch.setattr(
            runner_module,
            "load_wp2_retry_lineage",
            lambda *_args, **_kwargs: SimpleNamespace(),
        )
        monkeypatch.setattr(
            runner_module,
            "hold_wp2_retry_training_lease",
            lambda _lineage: nullcontext(),
        )
        monkeypatch.setattr(
            runner_module,
            "_require_unchanged_source_files",
            lambda _prepared: None,
        )
        monkeypatch.setattr(
            runner_module,
            "record_wp2_retry_training_completion",
            lambda *_args, **_kwargs: None,
        )
    else:
        monkeypatch.setattr(runner_module, "_load_inputs", write_sources)
    monkeypatch.setattr(
        runner_module,
        "_load_l1_selection",
        lambda *_args, **_kwargs: {5: 0.1},
    )
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    return {
        "config_path": tmp_path / "config.json",
        "registry_path": tmp_path / "registry.json",
        "training_data_paths": (tmp_path / "source.jsonl",),
        "l1_selection_path": selection,
        "gpu_id": "0",
        "wandb_project": "test-sae",
        "wandb_entity": None,
    }


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
    preregistration = SimpleNamespace(
        sha256="c" * 64,
        sae_gpu_id=1,
        retry_inputs=None,
    )
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
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
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
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
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
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
    preregistration = SimpleNamespace(
        sha256="c" * 64,
        sae_gpu_id=1,
        retry_inputs=None,
    )
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
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


def test_wp2_retry_claim_precedes_runtime_and_model_initialization(
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
        probe_layers=(5,),
        activation_subsample_layers=(5,),
        seeds_by_layer={5: (42,)},
        minimum_training_tokens=2,
        protected_student_token_budget=10_000_000,
        statistics_tokens=1,
        activation_subsample_tokens=0,
        activation_batch_size=2,
    )
    preregistration = SimpleNamespace(
        sha256="c" * 64,
        sae_gpu_id=0,
        retry_inputs=SimpleNamespace(),
    )
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
        source_tokens=3,
        protected_eligible_records=1,
        protected_eligible_source_tokens=3,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=0,
    )

    def fake_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text("{}\n", encoding="utf-8")
        return prepared

    claim_created = False
    claimed_bindings: dict[str, object] = {}

    def fake_claim(*_args, training_bindings, **_kwargs):
        nonlocal claim_created
        claim_created = True
        claimed_bindings.update(training_bindings)
        return SimpleNamespace(sha256="f" * 64)

    def runtime_factory(**kwargs):
        assert claim_created, "runtime initialized before the project-global retry claim"
        return _Runtime(**kwargs)

    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._prepare_inputs",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._write_source_registry",
        lambda *, output_dir, **_kwargs: (output_dir / "source_registry.json").write_text(
            "{}\n", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._load_l1_selection",
        lambda *_args, **_kwargs: {5: 0.1},
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_wp2_retry_lineage",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.hold_wp2_retry_training_lease",
        lambda _lineage: nullcontext(),
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._require_unchanged_source_files",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.claim_wp2_retry_training",
        fake_claim,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.record_wp2_retry_training_completion",
        lambda *_args, **_kwargs: None,
    )
    selection = tmp_path / "l1.json"
    selection.write_text("{}\n", encoding="utf-8")

    run_train_saes(
        SaeTrainingRunConfig(
            config_path=tmp_path / "config.json",
            registry_path=tmp_path / "registry.json",
            training_data_paths=(tmp_path / "source.jsonl",),
            l1_selection_path=selection,
            gpu_id="0",
            wandb_project="test-sae",
            wandb_entity=None,
            output_dir=tmp_path / "retry-training",
        ),
        runtime_factory=runtime_factory,
        tracker_factory=lambda **_kwargs: _Tracker(),
    )

    assert claim_created is True
    assert claimed_bindings["source_stream_sha256"] == prepared.source_stream_sha256
    assert claimed_bindings["selected_l1_sha256"] == runner_module._selected_l1_sha256(
        {5: 0.1}
    )
    assert claimed_bindings["source_inputs"] == [
        {"path": str(prepared.input_paths[0].resolve()), "sha256": "d" * 64}
    ]
    assert claimed_bindings["wandb_project"] == "test-sae"
    assert claimed_bindings["wandb_entity"] is None
    assert isinstance(claimed_bindings["implementation_sha256"], str)


def test_retry_resume_does_not_mutate_output_before_claim_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = SimpleNamespace(
        config_sha256="a" * 64,
        model="model",
        model_revision="b" * 40,
        minimum_training_tokens=2,
        protected_student_token_budget=10_000_000,
        statistics_tokens=1,
    )
    preregistration = SimpleNamespace(
        sha256="c" * 64,
        sae_gpu_id=0,
        retry_inputs=SimpleNamespace(),
    )
    output = tmp_path / "retry-training"
    output.mkdir()
    source_registry = output / "source_registry.json"
    source_registry.write_text('{"original": true}\n', encoding="utf-8")
    before = source_registry.read_bytes()
    prepared = PreparedSaeSources(
        sources=(SimpleNamespace(record_id="record", token_count=1),),
        reserved=(SimpleNamespace(record_id="reserved", token_count=1),),
        input_paths=(tmp_path / "source.jsonl",),
        input_sha256=("d" * 64,),
        input_record_ids=frozenset({"record"}),
        input_source_ids=frozenset({"source"}),
        input_group_ids=frozenset({"group"}),
        record_id_sha256="e" * 64,
        source_stream_sha256="f" * 64,
        source_tokens=3,
        protected_eligible_records=1,
        protected_eligible_source_tokens=3,
        protected_eligible_record_ids_sha256="e" * 64,
        protected_normalized_duplicates_removed=0,
    )

    def mutating_inputs(*, output_dir: Path, **_kwargs):
        (output_dir / "source_registry.json").write_text('{"mutated": true}\n', encoding="utf-8")
        return prepared

    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._prepare_inputs",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._write_source_registry",
        mutating_inputs,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._load_l1_selection",
        lambda *_args, **_kwargs: {5: 0.1},
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.load_wp2_retry_lineage",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.hold_wp2_retry_training_lease",
        lambda _lineage: nullcontext(),
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner._require_unchanged_source_files",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.runner.claim_wp2_retry_training",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("resume differs")),
    )
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume differs"):
        run_train_saes(
            SaeTrainingRunConfig(
                config_path=tmp_path / "config.json",
                registry_path=tmp_path / "registry.json",
                training_data_paths=(tmp_path / "source.jsonl",),
                l1_selection_path=selection,
                gpu_id="0",
                wandb_project="test",
                wandb_entity=None,
                output_dir=output,
                resume=True,
            )
        )

    assert source_registry.read_bytes() == before


@pytest.mark.parametrize("resume", (False, True))
@pytest.mark.parametrize("relative", (False, True))
@pytest.mark.parametrize("target_exists", (False, True))
def test_sae_output_contract_rejects_a_final_symlink(
    tmp_path: Path,
    resume: bool,
    relative: bool,
    target_exists: bool,
) -> None:
    target = tmp_path / "real-output"
    if target_exists:
        target.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(target.name if relative else target, target_is_directory=True)

    with pytest.raises(ValueError, match="output path must not be a symlink"):
        runner_module._check_output(alias, resume=resume)
    with pytest.raises(ValueError, match="output path must not be a symlink"):
        runner_module._retry_output_before_claim(alias, resume=resume)
    with pytest.raises(ValueError, match="recovery output must not be a symlink"):
        runner_module._check_retry_claim_only_output(
            output=alias,
            expected_source_registry=b"{}\n",
        )


def test_retry_resume_replays_from_zero_after_claim_before_output_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    output = tmp_path / "retry-training"
    claim_calls: list[bool] = []
    claimed_output: Path | None = None
    claimed_bindings: dict[str, object] | None = None

    def exact_claim(*_args, output_dir: Path, training_bindings, resume: bool, **_kwargs):
        nonlocal claimed_output, claimed_bindings
        claim_calls.append(resume)
        if not resume:
            claimed_output = output_dir
            claimed_bindings = dict(training_bindings)
        else:
            assert output_dir == claimed_output
            assert dict(training_bindings) == claimed_bindings
        return SimpleNamespace(sha256="f" * 64)

    original_prepare_output = runner_module._prepare_output
    crash_pending = True

    def crash_after_claim(path: Path, *, resume: bool) -> Path:
        nonlocal crash_pending
        if crash_pending:
            crash_pending = False
            raise RuntimeError("simulated crash after claim before output creation")
        return original_prepare_output(path, resume=resume)

    tracker_kwargs: list[dict[str, object]] = []

    def tracker_factory(**kwargs):
        checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["cursor"]["trained_tokens"] == 0
        tracker_kwargs.append(dict(kwargs))
        return _Tracker()

    monkeypatch.setattr(runner_module, "claim_wp2_retry_training", exact_claim)
    monkeypatch.setattr(runner_module, "_prepare_output", crash_after_claim)

    with pytest.raises(RuntimeError, match="before output creation"):
        run_train_saes(SaeTrainingRunConfig(**base, output_dir=output))
    assert not output.exists()

    result = run_train_saes(
        SaeTrainingRunConfig(**base, output_dir=output, resume=True),
        runtime_factory=_Runtime,
        tracker_factory=tracker_factory,
    )

    assert claim_calls == [False, True]
    assert result.trained_tokens == 4
    assert tracker_kwargs[-1]["resume"] is False


def test_retry_resume_replays_from_durable_zero_checkpoint_in_the_same_wandb_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    output = tmp_path / "retry-training"
    claimed_output: Path | None = None
    claimed_bindings: dict[str, object] | None = None

    def exact_claim(*_args, output_dir: Path, training_bindings, resume: bool, **_kwargs):
        nonlocal claimed_output, claimed_bindings
        if not resume:
            claimed_output = output_dir
            claimed_bindings = dict(training_bindings)
        else:
            assert output_dir == claimed_output
            assert dict(training_bindings) == claimed_bindings
        return SimpleNamespace(sha256="f" * 64)

    tracker_kwargs: list[dict[str, object]] = []
    trackers: list[_Tracker] = []

    def crashing_tracker_factory(**kwargs):
        tracker_kwargs.append(dict(kwargs))
        output.mkdir(parents=True, exist_ok=True)
        (output / "wandb_run.json").write_text(
            '{"schema_version": "test-wandb-run/v1"}\n', encoding="utf-8"
        )
        tracker = _CrashAfterFirstStepTracker()
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(runner_module, "claim_wp2_retry_training", exact_claim)
    with pytest.raises(RuntimeError, match="before the first checkpoint"):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=output),
            runtime_factory=_Runtime,
            tracker_factory=crashing_tracker_factory,
        )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "robustness-sae-initial-training-checkpoint/v1"
    assert checkpoint["cursor"]["trained_tokens"] == 0
    registry_before = (output / "source_registry.json").read_bytes()
    activation_before = next((output / "activation_subsample").rglob("*.pt")).read_bytes()

    def resumed_tracker_factory(**kwargs):
        tracker_kwargs.append(dict(kwargs))
        tracker = _Tracker()
        trackers.append(tracker)
        return tracker

    result = run_train_saes(
        SaeTrainingRunConfig(**base, output_dir=output, resume=True),
        runtime_factory=_Runtime,
        tracker_factory=resumed_tracker_factory,
    )

    assert result.trained_tokens == 4
    assert tracker_kwargs[-1]["resume"] is True
    assert tracker_kwargs[-1]["resume_optimizer_step"] == 0
    assert trackers[-1].steps == [1, 2]
    assert (output / "source_registry.json").read_bytes() == registry_before
    assert next((output / "activation_subsample").rglob("*.pt")).read_bytes() == (
        activation_before
    )


def test_retry_claim_only_accepts_only_a_canonical_source_registry_temporary_prefix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retry-training"
    output.mkdir()
    expected = b'{\n  "schema_version": "robustness-sae-source-registry/v1"\n}\n'
    temporary = output / ".source_registry.json.12345.tmp"
    temporary.write_bytes(expected[:23])

    runner_module._check_retry_claim_only_output(
        output=output,
        expected_source_registry=expected,
    )

    temporary.write_bytes(b'{"attacker-controlled": true}\n')
    with pytest.raises(ValueError, match="source registry temporary differs"):
        runner_module._check_retry_claim_only_output(
            output=output,
            expected_source_registry=expected,
        )


def test_retry_claim_only_rejects_a_source_registry_temporary_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retry-training"
    output.mkdir()
    expected = b'{"expected": true}\n'
    outside = tmp_path / "outside.json"
    outside.write_bytes(expected)
    (output / ".source_registry.json.12345.tmp").symlink_to(outside)

    with pytest.raises(ValueError, match="source registry temporary is invalid"):
        runner_module._check_retry_claim_only_output(
            output=output,
            expected_source_registry=expected,
        )


@pytest.mark.parametrize("identifier", ("012345", "0", "١٢٣٤٥"))
def test_retry_claim_only_rejects_noncanonical_source_registry_temporary_names(
    tmp_path: Path,
    identifier: str,
) -> None:
    output = tmp_path / "retry-training"
    output.mkdir()
    expected = b'{"expected": true}\n'
    (output / f".source_registry.json.{identifier}.tmp").write_bytes(expected)

    with pytest.raises(ValueError, match="source registry temporary is invalid"):
        runner_module._check_retry_claim_only_output(
            output=output,
            expected_source_registry=expected,
        )


def test_retry_claim_only_rejects_a_source_registry_temporary_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    output = tmp_path / "retry-training"
    output.mkdir()
    os.mkfifo(output / ".source_registry.json.12345.tmp")

    with pytest.raises(ValueError, match="source registry temporary is invalid"):
        runner_module._check_retry_claim_only_output(
            output=output,
            expected_source_registry=b'{"expected": true}\n',
        )


def test_retry_claim_only_rejects_a_hard_linked_source_registry_temporary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retry-training"
    output.mkdir()
    expected = b'{"expected": true}\n'
    outside = tmp_path / "outside.json"
    outside.write_bytes(expected)
    os.link(outside, output / ".source_registry.json.12345.tmp")

    with pytest.raises(ValueError, match="source registry temporary is invalid"):
        runner_module._check_retry_claim_only_output(
            output=output,
            expected_source_registry=expected,
        )


@pytest.mark.parametrize(
    ("artifact_name", "is_directory"),
    (
        ("checkpoint.0.pt", False),
        ("wandb_run.json", False),
        (".wandb", True),
        ("activation_subsample", True),
        ("sae-layer-5-seed-42.pt", False),
        ("run.json", False),
        ("unknown.bin", False),
    ),
)
def test_retry_claim_only_resume_rejects_orphan_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    is_directory: bool,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    output = tmp_path / "retry-training"
    output.mkdir()
    artifact = output / artifact_name
    if is_directory:
        artifact.mkdir()
    else:
        artifact.write_bytes(b"orphan")
    monkeypatch.setattr(
        runner_module,
        "claim_wp2_retry_training",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="f" * 64),
    )

    with pytest.raises(
        ValueError,
        match="unexpected pre-checkpoint artifact|invalid strict JSON",
    ):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=output, resume=True),
            runtime_factory=_Runtime,
            tracker_factory=lambda **_kwargs: _Tracker(),
        )


def test_non_retry_resume_still_requires_a_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=False)
    output = tmp_path / "ordinary-training"
    output.mkdir()

    with pytest.raises(ValueError, match="requires checkpoint.json"):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=output, resume=True),
            runtime_factory=_Runtime,
            tracker_factory=lambda **_kwargs: _Tracker(),
        )


def test_retry_resume_with_missing_output_checks_claim_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    output = tmp_path / "missing-retry-training"
    monkeypatch.setattr(
        runner_module,
        "claim_wp2_retry_training",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("resume differs")),
    )

    with pytest.raises(ValueError, match="resume differs"):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=output, resume=True),
            runtime_factory=_Runtime,
            tracker_factory=lambda **_kwargs: _Tracker(),
        )

    assert not output.exists()


def test_retry_rejects_a_manifest_changed_after_it_was_loaded(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"first"}\n', encoding="utf-8")
    prepared = SimpleNamespace(
        input_paths=(source,),
        input_sha256=(hashlib.sha256(source.read_bytes()).hexdigest(),),
    )
    source.write_text('{"text":"second"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest changed while it was being loaded"):
        runner_module._require_unchanged_source_files(prepared)


def test_retry_rejects_an_l1_file_changed_while_its_values_are_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    selection = Path(base["l1_selection_path"])
    claim_reached = False

    def mutating_load(*_args, **_kwargs):
        selection.write_text('{"changed":true}\n', encoding="utf-8")
        return {5: 0.1}

    def claim(*_args, **_kwargs):
        nonlocal claim_reached
        claim_reached = True
        return SimpleNamespace(sha256="f" * 64)

    monkeypatch.setattr(runner_module, "_load_l1_selection", mutating_load)
    monkeypatch.setattr(runner_module, "claim_wp2_retry_training", claim)

    with pytest.raises(ValueError, match="L1 selection changed while it was being loaded"):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=tmp_path / "retry-training"),
            runtime_factory=_Runtime,
            tracker_factory=lambda **_kwargs: _Tracker(),
        )
    assert claim_reached is False


def test_retry_rejects_implementation_drift_immediately_after_the_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _install_minimal_training_fixture(tmp_path, monkeypatch, retry=True)
    digests = iter(("1" * 64, "2" * 64))
    runtime_reached = False

    monkeypatch.setattr(
        runner_module,
        "retry_implementation_sha256",
        lambda: next(digests),
    )
    monkeypatch.setattr(
        runner_module,
        "claim_wp2_retry_training",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="f" * 64),
    )

    def runtime_factory(**_kwargs):
        nonlocal runtime_reached
        runtime_reached = True
        return _Runtime(**_kwargs)

    with pytest.raises(ValueError, match="implementation changed after the global claim"):
        run_train_saes(
            SaeTrainingRunConfig(**base, output_dir=tmp_path / "retry-training"),
            runtime_factory=runtime_factory,
            tracker_factory=lambda **_kwargs: _Tracker(),
        )
    assert runtime_reached is False


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
    project_root = tmp_path / "fixed-project-root"
    project_root.mkdir()
    preregistration = SimpleNamespace(
        sha256="b" * 64,
        wp2_project_root=project_root.resolve(),
        wp2_project_root_device=project_root.stat().st_dev,
        wp2_project_root_inode=project_root.stat().st_ino,
    )

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
    reservation = reserve_initial_wp2_validation(
        project_root=ledger.parent,
        project_root_device=preregistration.wp2_project_root_device,
        project_root_inode=preregistration.wp2_project_root_inode,
        config_sha256=protocol.config_sha256,
        preregistration_sha256=preregistration.sha256,
        checkpoint_dir=first,
        checkpoint_run_sha256=digest,
        output_dir=tmp_path / "validation-first",
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
        reservation=reservation,
        checkpoint_dir=first,
    )

    second = checkpoint("training-retrain", 2)
    ledger, attempts, digest = _load_wp2_attempts(
        checkpoint_dir=second,
        protocol=protocol,
        preregistration=preregistration,
    )
    with pytest.raises(ValueError, match="initial validation slot is already reserved"):
        reserve_initial_wp2_validation(
            project_root=ledger.parent,
            project_root_device=preregistration.wp2_project_root_device,
            project_root_inode=preregistration.wp2_project_root_inode,
            config_sha256=protocol.config_sha256,
            preregistration_sha256=preregistration.sha256,
            checkpoint_dir=second,
            checkpoint_run_sha256=digest,
            output_dir=tmp_path / "validation-retrain",
        )


def test_relocated_checkpoint_parent_cannot_reset_initial_wp2_budget(tmp_path: Path) -> None:
    """Falsify checkpoint-parent scoped ledgers by moving every full bundle."""

    protocol = SimpleNamespace(config_sha256="a" * 64, maximum_gate_retrains=1)
    project_root = tmp_path / "fixed-project-root"
    project_root.mkdir()
    preregistration = SimpleNamespace(
        sha256="b" * 64,
        wp2_project_root=project_root.resolve(),
        wp2_project_root_device=project_root.stat().st_dev,
        wp2_project_root_inode=project_root.stat().st_ino,
    )
    accepted = 0
    for index in range(4):
        checkpoint = tmp_path / f"run-{index}" / "training"
        checkpoint.mkdir(parents=True)
        (checkpoint / "run.json").write_text(
            json.dumps({"value": index}) + "\n",
            encoding="utf-8",
        )
        output = tmp_path / f"external-results-{index}" / "validation"
        try:
            ledger, attempts, digest = _load_wp2_attempts(
                checkpoint_dir=checkpoint,
                protocol=protocol,
                preregistration=preregistration,
            )
            reservation = reserve_initial_wp2_validation(
                project_root=ledger.parent,
                project_root_device=preregistration.wp2_project_root_device,
                project_root_inode=preregistration.wp2_project_root_inode,
                config_sha256=protocol.config_sha256,
                preregistration_sha256=preregistration.sha256,
                checkpoint_dir=checkpoint,
                checkpoint_run_sha256=digest,
                output_dir=output,
            )
            _record_wp2_attempt(
                ledger_path=ledger,
                prior_attempts=attempts,
                checkpoint_run_sha256=digest,
                output_dir=output,
                passed=False,
                acceptance_sha256="c" * 64,
                protocol=protocol,
                preregistration=preregistration,
                reservation=reservation,
                checkpoint_dir=checkpoint,
            )
        except ValueError:
            break
        accepted += 1

    assert accepted == 1


def test_legacy_unrooted_v1_cannot_start_a_new_initial_validation(tmp_path: Path) -> None:
    """Old immutable artifacts stay readable, but cannot mint a fresh budget."""

    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    (checkpoint / "run.json").write_text("{}\n", encoding="utf-8")
    protocol = SimpleNamespace(config_sha256="a" * 64, maximum_gate_retrains=1)
    preregistration = SimpleNamespace(
        sha256="b" * 64,
        wp2_project_root=None,
    )

    with pytest.raises(ValueError, match="explicit absolute wp2_project_root"):
        _load_wp2_attempts(
            checkpoint_dir=checkpoint,
            protocol=protocol,
            preregistration=preregistration,
        )


def test_initial_validation_rejects_retry_claimed_checkpoint(tmp_path: Path) -> None:
    protocol = SimpleNamespace(config_sha256="a" * 64, maximum_gate_retrains=1)
    (tmp_path / "fixed-project-root").mkdir()
    preregistration = SimpleNamespace(
        sha256="b" * 64,
        wp2_project_root=(tmp_path / "fixed-project-root").resolve(),
    )
    checkpoint = tmp_path / "training"
    checkpoint.mkdir()
    (checkpoint / "run.json").write_text(
        json.dumps(
            {
                "bindings": {
                    "config_sha256": protocol.config_sha256,
                    "preregistration_sha256": preregistration.sha256,
                    "wp2_retry_claim_sha256": "c" * 64,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="retry checkpoint requires retry preregistration"):
        _load_wp2_attempts(
            checkpoint_dir=checkpoint,
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

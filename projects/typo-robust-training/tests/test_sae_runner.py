"""CPU-only integration checks for SAE training artifacts and exact resume."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from typo_robust_training.sae.data import PreparedSaeSources
from typo_robust_training.sae.runner import SaeTrainingRunConfig, run_train_saes
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
        record_id_sha256="e" * 64,
        source_tokens=10,
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

    tracker_count = len(trackers)
    resumed = run_train_saes(
        SaeTrainingRunConfig(**base, resume=True),
        runtime_factory=_Runtime,
        tracker_factory=tracker_factory,
    )
    assert resumed.trained_tokens == 4
    assert resumed.optimizer_steps == 2
    assert len(trackers) == tracker_count

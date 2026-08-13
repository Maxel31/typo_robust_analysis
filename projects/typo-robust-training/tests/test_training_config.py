"""Every training condition has one strict public command and frozen objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.training.config import load_adapter_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "noisy-language-model": PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml",
    "output-matching": PROJECT_ROOT / "configs/baselines/output-matching.yaml",
    "global-state-alignment": PROJECT_ROOT / "configs/baselines/global-state-alignment.yaml",
    "localized-state-distillation": (
        PROJECT_ROOT / "configs/ablations/gemma4b-component-state-cycle1.yaml"
    ),
}


def test_training_configs_freeze_model_optimizer_adapter_and_objective_scopes() -> None:
    protocols = {name: load_adapter_training_config(path) for name, path in CONFIGS.items()}
    for name, protocol in protocols.items():
        assert protocol.condition == name
        assert protocol.model == "google/gemma-3-4b-it"
        assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
        assert protocol.dtype == "bfloat16"
        assert protocol.max_sequence_length == 512
        assert protocol.optimizer == "adamw"
        assert protocol.learning_rate == pytest.approx(2e-4)
        assert protocol.weight_decay == pytest.approx(0.01)
        assert protocol.warmup_ratio == pytest.approx(0.03)
        assert protocol.scheduler == "cosine"
        assert protocol.gradient_checkpointing is True
        assert protocol.micro_batch_size == 1
        assert protocol.gradient_accumulation_steps == 32
        assert protocol.max_optimizer_steps == 100
        assert protocol.seed_inventory == (42, 43, 44)
        assert protocol.lora_target_modules == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    noisy = protocols["noisy-language-model"]
    assert noisy.layer_scope == "all-decoder-layers"
    assert noisy.state_scope == "none"
    assert noisy.loss_weights == {
        "noisy_language_model": 1.0,
        "answer": 0.0,
        "output": 0.0,
        "state": 0.0,
        "clean": 0.0,
    }

    output = protocols["output-matching"]
    assert output.layer_scope == "all-decoder-layers"
    assert output.state_scope == "none"
    assert output.loss_weights["output"] == 1.0
    assert output.loss_weights["answer"] == 1.0
    assert output.loss_weights["clean"] == 0.5

    global_state = protocols["global-state-alignment"]
    assert global_state.layer_scope == "all-decoder-layers"
    assert global_state.state_scope == "all-layers-all-aligned-tokens"

    component_ablation = protocols["localized-state-distillation"]
    assert component_ablation.layer_scope == "selected-component-containing-layers"
    assert component_ablation.state_scope == "selected-components-edited-word-final-tokens"
    assert component_ablation.lora_rank == 16
    assert component_ablation.lora_alpha == pytest.approx(32.0)
    assert component_ablation.loss_weights == {
        "noisy_language_model": 0.0,
        "answer": 1.0,
        "output": 1.0,
        "state": 0.5,
        "clean": 0.5,
    }


def test_training_config_rejects_condition_or_field_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIGS["localized-state-distillation"].read_text(encoding="utf-8"))
    payload["condition"] = "output-matching"
    mismatch = tmp_path / "mismatch.yaml"
    mismatch.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="condition and objective"):
        load_adapter_training_config(mismatch)

    payload = json.loads(CONFIGS["localized-state-distillation"].read_text(encoding="utf-8"))
    payload["unexpected"] = True
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        load_adapter_training_config(unknown)


def test_training_commands_expose_condition_specific_evidence_and_shared_resume() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    register_commands(commands)

    for command in (
        "train-noisy-language-model",
        "train-output-matching",
        "train-global-state-alignment",
    ):
        args = parser.parse_args(
            [
                command,
                "--config",
                "condition.yaml",
                "--training-data",
                "data",
                "--seed",
                "42",
                "--gpu-id",
                "3",
                "--output-dir",
                "output",
                "--resume",
            ]
        )
        assert args.resume is True
        assert not hasattr(args, "layer_selection")
        assert not hasattr(args, "component_selection")

    component_ablation = parser.parse_args(
        [
            "train-localized-state-distillation",
            "--config",
            "component-ablation.yaml",
            "--training-data",
            "data",
            "--layer-selection",
            "layers.json",
            "--component-selection",
            "components.json",
            "--seed",
            "44",
            "--gpu-id",
            "3",
            "--output-dir",
            "output",
            "--resume",
        ]
    )
    assert component_ablation.resume is True
    assert component_ablation.layer_selection == Path("layers.json")
    assert component_ablation.component_selection == Path("components.json")

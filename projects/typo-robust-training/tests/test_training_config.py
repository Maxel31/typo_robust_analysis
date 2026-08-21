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
    "localized-state-distillation": PROJECT_ROOT / "configs/gemma4b-targeted-lora.yaml",
}
CYCLE2_CONFIGS = {
    "output-matching": PROJECT_ROOT / "configs/cycle2/gemma4b-output-matching-100step.yaml",
    "localized-state-distillation": PROJECT_ROOT
    / "configs/cycle2/gemma4b-causal-window-100step.yaml",
    "random-window-state-distillation": PROJECT_ROOT
    / "configs/cycle2/gemma4b-random-window-100step.yaml",
    "global-state-alignment": PROJECT_ROOT / "configs/cycle2/gemma4b-all-layer-state-100step.yaml",
}
CYCLE2_STABLE_GRADIENT_CONFIGS = {
    "localized-state-distillation": PROJECT_ROOT
    / "configs/cycle2/gemma4b-causal-window-rho005-100step.yaml",
    "random-window-state-distillation": PROJECT_ROOT
    / "configs/cycle2/gemma4b-random-window-rho005-100step.yaml",
    "global-state-alignment": PROJECT_ROOT
    / "configs/cycle2/gemma4b-all-layer-state-rho005-100step.yaml",
}
CYCLE3_CONFIGS = {
    ("output-matching", 10_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-output-matching-10m.yaml",
    ("localized-state-distillation", 10_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-causal-window-10m.yaml",
    ("random-window-state-distillation", 10_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-random-window-10m.yaml",
    ("global-state-alignment", 10_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-all-layer-state-10m.yaml",
    ("output-matching", 64_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-output-matching-64m.yaml",
    ("localized-state-distillation", 64_000_000): PROJECT_ROOT
    / "configs/cycle3/gemma4b-causal-window-64m.yaml",
}


def test_training_configs_freeze_model_optimizer_adapter_and_objective_scopes() -> None:
    protocols = {name: load_adapter_training_config(path) for name, path in CONFIGS.items()}
    for name, protocol in protocols.items():
        assert protocol.condition == name
        assert protocol.model == "google/gemma-3-4b-it"
        assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
        assert protocol.dtype == "bfloat16"
        assert protocol.decoder_layers is None
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

    proposed = protocols["localized-state-distillation"]
    assert proposed.layer_scope == "selected-component-containing-layers"
    assert proposed.state_scope == "selected-components-edited-word-final-tokens"
    assert proposed.lora_rank == 16
    assert proposed.lora_alpha == pytest.approx(32.0)
    assert proposed.loss_weights == {
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


def test_training_config_rejects_unsupported_micro_batch_size(tmp_path: Path) -> None:
    payload = json.loads(CYCLE2_CONFIGS["output-matching"].read_text(encoding="utf-8"))
    payload["optimization"]["micro_batch_size"] = 2
    path = tmp_path / "micro-batch-size-two.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="micro_batch_size must equal 1"):
        load_adapter_training_config(path)


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 3])
def test_state_training_rejects_accumulation_that_cannot_pair_clean_and_noisy(
    tmp_path: Path,
    gradient_accumulation_steps: int,
) -> None:
    payload = json.loads(CYCLE2_CONFIGS["localized-state-distillation"].read_text(encoding="utf-8"))
    payload["optimization"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    path = tmp_path / f"state-accumulation-{gradient_accumulation_steps}.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="state training with exact alternating pairs requires an even "
        "gradient_accumulation_steps >= 2",
    ):
        load_adapter_training_config(path)


def test_cycle2_configs_share_capacity_data_schedule_and_output_objective() -> None:
    protocols = {name: load_adapter_training_config(path) for name, path in CYCLE2_CONFIGS.items()}
    for name, protocol in protocols.items():
        assert protocol.schema_version == "robustness-adapter-training-config/v2"
        assert protocol.condition == name
        assert protocol.pairing_policy == "exact-alternating-clean-noisy/v1"
        assert protocol.layer_scope == "all-decoder-layers"
        assert protocol.lora_rank == 16
        assert protocol.lora_alpha == pytest.approx(8.0)
        assert protocol.lora_dropout == pytest.approx(0.0)
        assert protocol.learning_rate == pytest.approx(1e-4)
        assert protocol.scheduler == "constant-with-warmup"
        assert protocol.loss_weights["answer"] == 0.0
        assert protocol.loss_weights["clean"] == 0.0
        assert protocol.loss_weights["output"] == 1.0
        assert protocol.state_distance == "cosine-residual/v1"
        assert protocol.decoder_layers == 34

    output = protocols["output-matching"]
    assert output.state_scope == "none"
    assert output.state_gradient_ratio is None
    assert output.calibration_micro_batches == 0

    for name in (
        "localized-state-distillation",
        "random-window-state-distillation",
        "global-state-alignment",
    ):
        assert protocols[name].loss_weights["state"] == 1.0
        assert protocols[name].state_gradient_ratio == pytest.approx(0.1)
        assert protocols[name].calibration_micro_batches == 8


def test_cycle2_stable_gradient_configs_preserve_arms_and_halve_initial_ratio() -> None:
    original = {
        name: load_adapter_training_config(CYCLE2_CONFIGS[name])
        for name in CYCLE2_STABLE_GRADIENT_CONFIGS
    }
    stable = {
        name: load_adapter_training_config(path)
        for name, path in CYCLE2_STABLE_GRADIENT_CONFIGS.items()
    }

    for name in stable:
        assert stable[name].condition == original[name].condition
        assert stable[name].model == original[name].model
        assert stable[name].model_revision == original[name].model_revision
        assert stable[name].loss_weights == original[name].loss_weights
        assert stable[name].state_scope == original[name].state_scope
        assert stable[name].state_window_policy == original[name].state_window_policy
        assert stable[name].state_gradient_ratio == pytest.approx(0.05)
        assert stable[name].calibration_micro_batches == 8


def test_cycle3_configs_match_every_axis_except_budget_and_localized_state_signal() -> None:
    protocols = {
        identity: load_adapter_training_config(path) for identity, path in CYCLE3_CONFIGS.items()
    }
    for (condition, token_budget), protocol in protocols.items():
        assert protocol.schema_version == "robustness-adapter-training-config/v3"
        assert protocol.condition == condition
        assert protocol.max_student_tokens == token_budget
        assert protocol.max_optimizer_steps == 10_000
        assert protocol.warmup_ratio == 0.0
        assert protocol.learning_rate == pytest.approx(1e-4)
        assert protocol.lora_rank == 16
        assert protocol.lora_alpha == pytest.approx(8.0)
        assert protocol.pairing_policy == "exact-alternating-clean-noisy/v1"
        assert protocol.loss_weights["output"] == 1.0
        assert protocol.loss_weights["answer"] == 0.0
        assert protocol.loss_weights["clean"] == 0.0
        if condition == "output-matching":
            assert protocol.loss_weights["state"] == 0.0
            assert protocol.state_gradient_ratio is None
        else:
            assert protocol.loss_weights["state"] == 1.0
            assert protocol.state_gradient_ratio == pytest.approx(0.05)
        assert protocol.gradient_ratio_guard_optimizer_steps == 50


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
                "--wandb-project",
                "typo-robustness-training",
                "--output-dir",
                "output",
                "--resume",
            ]
        )
        assert args.resume is True
        assert args.wandb_project == "typo-robustness-training"
        assert not hasattr(args, "layer_selection")
        assert not hasattr(args, "component_selection")

    proposed = parser.parse_args(
        [
            "train-localized-state-distillation",
            "--config",
            "proposed.yaml",
            "--training-data",
            "data",
            "--layer-selection",
            "layers.json",
            "--window-validation",
            "validation.json",
            "--component-selection",
            "components.json",
            "--seed",
            "44",
            "--gpu-id",
            "3",
            "--wandb-project",
            "typo-robustness-training",
            "--output-dir",
            "output",
            "--resume",
        ]
    )
    assert proposed.resume is True
    assert proposed.wandb_project == "typo-robustness-training"
    assert proposed.layer_selection == Path("layers.json")
    assert proposed.window_validation == Path("validation.json")
    assert proposed.component_selection == Path("components.json")

    random_control = parser.parse_args(
        [
            "train-random-window-state-distillation",
            "--config",
            "random.yaml",
            "--training-data",
            "data",
            "--layer-selection",
            "layers.json",
            "--window-validation",
            "validation.json",
            "--seed",
            "42",
            "--gpu-id",
            "1",
            "--wandb-project",
            "typo-robustness-training",
            "--output-dir",
            "output",
        ]
    )
    assert random_control.layer_selection == Path("layers.json")
    assert random_control.window_validation == Path("validation.json")
    assert not hasattr(random_control, "component_selection")

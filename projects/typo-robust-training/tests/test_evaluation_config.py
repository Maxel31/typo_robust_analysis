"""Held-out evaluation has one strict protocol and explicit public command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.evaluation.config import load_robustness_evaluation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gemma4b-evaluation.yaml"


def test_default_evaluation_protocol_freezes_model_generation_metrics_and_gate() -> None:
    protocol = load_robustness_evaluation_config(DEFAULT_CONFIG)

    assert protocol.schema_version == "robustness-evaluation-config/v1"
    assert protocol.model == "google/gemma-3-4b-it"
    assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert protocol.dtype == "bfloat16"
    assert protocol.max_input_tokens == 4096
    assert protocol.max_new_tokens == 512
    assert protocol.teacher_forced_tokens == 16
    assert protocol.readout_token_range == (2, 16)
    assert protocol.prompt_protocol == "paper-cot-templates/v1"
    assert protocol.answer_extraction == "paper-task-extractors/v1"
    assert protocol.generation == {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "termination_protocol": "effective-eos-vs-length-cap/v1",
    }
    assert protocol.bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == pytest.approx(0.95)
    assert protocol.patch_position == "edited-word-final-token"
    assert protocol.patch_window_source == "frozen-layer-selection"
    assert protocol.seed_inventory == (42, 43, 44)
    assert protocol.gate == {
        "minimum_typo_accuracy_gain_points": 3.0,
        "maximum_clean_accuracy_drop_points": 1.0,
        "require_wrong_to_right_above_right_to_wrong": True,
        "require_positive_unseen_task_gain": True,
        "minimum_directional_seeds": 2,
        "minimum_patch_gain_reduction_fraction": 0.30,
    }


def test_evaluation_config_rejects_unknown_fields_or_moving_revision(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        load_robustness_evaluation_config(unknown)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["revision"] = "main"
    moving = tmp_path / "moving.yaml"
    moving.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_robustness_evaluation_config(moving)


def test_evaluation_command_requires_explicit_role_checkpoints_and_resume() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    register_commands(commands)
    args = parser.parse_args(
        [
            "evaluate-typo-robustness",
            "--config",
            "evaluation.yaml",
            "--training-data",
            "data",
            "--evaluation-role",
            "pre-pr-gate",
            "--layer-selection",
            "layers.json",
            "--window-validation",
            "window-validation.json",
            "--checkpoint",
            "seed-42/adapter",
            "--checkpoint",
            "seed-43/adapter",
            "--splits",
            "same-task",
            "unseen-task",
            "unseen-content",
            "unseen-typo",
            "--gpu-id",
            "3",
            "--output-dir",
            "evaluation",
            "--confirm-sealed-role",
            "--resume",
        ]
    )

    assert args.command == "evaluate-typo-robustness"
    assert args.config == Path("evaluation.yaml")
    assert args.training_data == Path("data")
    assert args.evaluation_role == "pre-pr-gate"
    assert args.layer_selection == Path("layers.json")
    assert args.window_validation == Path("window-validation.json")
    assert args.checkpoints == [Path("seed-42/adapter"), Path("seed-43/adapter")]
    assert args.splits == ["same-task", "unseen-task", "unseen-content", "unseen-typo"]
    assert args.gpu_id == "3"
    assert args.output_dir == Path("evaluation")
    assert args.confirm_sealed_role is True
    assert args.resume is True

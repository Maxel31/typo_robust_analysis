"""Frozen configuration and CLI contract for diagnostic layer selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.localization.config import load_layer_selection_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-layer-selection.yaml"


def test_default_layer_selection_protocol_is_fully_frozen() -> None:
    protocol = load_layer_selection_config(DEFAULT_CONFIG)

    assert protocol.schema_version == "robustness-layer-selection-config/v1"
    assert protocol.model == "google/gemma-3-4b-it"
    assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert protocol.dtype == "bfloat16"
    assert protocol.tasks == ("gsm8k", "mmlu", "arc")
    assert protocol.teacher_forced_tokens == 16
    assert protocol.readout_token_range == (2, 16)
    assert protocol.untreated_mean_kl_min_exclusive == 1e-6
    assert protocol.minimum_kl_eligible_per_task == 50
    assert protocol.minimum_kl_eligible_fraction_per_task == 0.8
    assert protocol.minimum_answer_cohort_per_task == 10
    assert protocol.beta == 0.5
    assert protocol.gamma == 1.0
    assert protocol.window_width == 6
    assert protocol.bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == 0.95
    assert protocol.generation == {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
        "termination_protocol": "effective-eos-vs-length-cap/v1",
        "answer_extraction": "primary-then-empty-only-positional/v1",
    }


def test_layer_selection_config_rejects_duplicate_keys_and_moving_revision(
    tmp_path: Path,
) -> None:
    source = DEFAULT_CONFIG.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text('{"schema_version":"duplicate",' + source.lstrip()[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_layer_selection_config(duplicate)

    payload = json.loads(source)
    payload["model"]["revision"] = "main"
    moving = tmp_path / "moving.yaml"
    moving.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_layer_selection_config(moving)


def test_cli_freezes_every_public_layer_selection_argument() -> None:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command")
    register_commands(commands)
    args = root.parse_args(
        [
            "select-distillation-layers",
            "--config",
            "selection.yaml",
            "--diagnostic-manifest",
            "diagnostic.jsonl",
            "--tasks",
            "gsm8k",
            "mmlu",
            "arc",
            "--gpu-id",
            "3",
            "--output-dir",
            "layers",
            "--resume",
        ]
    )
    assert args.command == "select-distillation-layers"
    assert args.config == Path("selection.yaml")
    assert args.diagnostic_manifest == Path("diagnostic.jsonl")
    assert args.tasks == ["gsm8k", "mmlu", "arc"]
    assert args.gpu_id == "3"
    assert args.output_dir == Path("layers")
    assert args.resume is True

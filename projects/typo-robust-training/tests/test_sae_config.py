"""Frozen configuration contract for the SAE diagnostic track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.sae.config import load_sae_protocol
from typo_robust_training.sae.registry import load_sae_preregistration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sae" / "gemma4b-sae-v1.yaml"
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "sae" / "registry-v1.yaml"


def test_sae_protocol_freezes_wp1_and_wp2_without_touching_confirmatory_training() -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "robustness-sae-training-config/v1"
    assert protocol.model == "google/gemma-3-4b-it"
    assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert protocol.d_model == 2560
    assert protocol.decoder_layers == 34
    assert protocol.probe_layers == (5, 20)
    assert protocol.activation_subsample_layers == (5, 11, 20, 26)
    assert protocol.seeds_by_layer == {5: (42, 43), 20: (42,)}
    assert protocol.expansion_factor == 16
    assert protocol.d_sae == 40960
    assert protocol.activation == "relu"
    assert protocol.regularizer == "l1"
    assert protocol.decoder_normalization == "unit-column-every-step/v1"
    assert protocol.adam_betas == (0.9, 0.999)
    assert protocol.adam_epsilon == 1e-8
    assert protocol.activation_batch_size == 2048
    assert protocol.minimum_training_tokens == 100_000_000
    assert protocol.preferred_training_tokens == 200_000_000
    assert protocol.statistics_tokens == 10_000_000
    assert protocol.shuffle_buffer_activations == 1_000_000
    assert protocol.document_character_limit == 8192
    assert protocol.near_duplicate_shingle_size == 5
    assert protocol.near_duplicate_jaccard_threshold == 0.99
    assert protocol.supplement_stream_order == "pinned-unshuffled-stream/v1"
    assert protocol.reserved_prefix_records == 30_000
    assert protocol.reserved_order_seed == 42
    assert protocol.l1_coefficients == (0.0001, 0.0003, 0.001)
    assert protocol.fvu_max == 0.35
    assert protocol.median_l0_range == (30, 150)
    assert protocol.dead_feature_rate_max == 0.20
    assert protocol.splice_kl_max == 0.15
    assert protocol.maximum_gate_retrains == 1
    assert protocol.wp5_feature_sufficiency_ratio == 0.50
    assert protocol.wp5_suppression_ratio == 0.25
    preregistration = load_sae_preregistration(DEFAULT_REGISTRY, protocol=protocol)
    assert preregistration.sae_gpu_id == 1
    assert preregistration.source_manifest_sha256 == (
        "ed99e962f02564369ac9878ef7db1d3d9e7b7c4e4876f8e39938cbe4fbe73967"
    )


def test_sae_config_is_strict_and_rejects_top_k(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["sae"]["activation"] = "top-k"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="activation must be relu"):
        load_sae_protocol(path)


def test_sae_config_rejects_calibration_larger_than_shuffle_buffer(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["sae"]["l1_calibration_tokens"] = 1_000_001
    path = tmp_path / "bad-buffer.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fit in one frozen shuffle buffer"):
        load_sae_protocol(path)


def test_cli_keeps_sae_calibration_training_and_validation_separate() -> None:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command")
    register_commands(commands)

    corpus = root.parse_args(
        [
            "build-sae-clean-corpus",
            "--config",
            "sae.json",
            "--registry",
            "registry.json",
            "--existing-data",
            "clean-a.jsonl",
            "--exclude-data",
            "evaluation",
            "--exclude-data",
            "localization",
            "--training-budget",
            "minimum",
            "--output-dir",
            "supplement",
        ]
    )
    assert corpus.existing_data == [Path("clean-a.jsonl")]
    assert corpus.exclude_data == [Path("evaluation"), Path("localization")]
    assert corpus.training_budget == "minimum"

    calibration = root.parse_args(
        [
            "calibrate-sparse-autoencoder-l1",
            "--config",
            "sae.json",
            "--registry",
            "registry.json",
            "--training-data",
            "clean-a.jsonl",
            "--gpu-id",
            "1",
            "--wandb-project",
            "typo-sae",
            "--output-dir",
            "calibration",
        ]
    )
    assert calibration.training_data == [Path("clean-a.jsonl")]
    assert calibration.gpu_id == "1"

    training = root.parse_args(
        [
            "train-sparse-autoencoders",
            "--config",
            "sae.json",
            "--registry",
            "registry.json",
            "--training-data",
            "clean-a.jsonl",
            "--l1-selection",
            "l1-selection.json",
            "--gpu-id",
            "1",
            "--wandb-project",
            "typo-sae",
            "--output-dir",
            "sae",
            "--resume",
        ]
    )
    assert training.resume is True
    assert training.l1_selection == Path("l1-selection.json")

    validation = root.parse_args(
        [
            "validate-sparse-autoencoders",
            "--config",
            "sae.json",
            "--registry",
            "registry.json",
            "--validation-data",
            "held-in.jsonl",
            "--checkpoint-dir",
            "sae",
            "--gpu-id",
            "1",
            "--output-dir",
            "validation",
        ]
    )
    assert validation.validation_data == [Path("held-in.jsonl")]

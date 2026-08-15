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
    assert protocol.l1_coefficients == (0.01, 0.1, 1.0)
    assert protocol.l1_calibration_tokens == 10_000_000
    assert protocol.fvu_max == 0.35
    assert protocol.median_l0_range == (30, 150)
    assert protocol.dead_feature_rate_max == 0.20
    assert protocol.splice_kl_max == 0.15
    assert protocol.maximum_gate_retrains == 1
    assert protocol.wp5_feature_sufficiency_ratio == 0.50
    assert protocol.wp5_suppression_ratio == 0.25
    preregistration = load_sae_preregistration(DEFAULT_REGISTRY, protocol=protocol)
    assert preregistration.sae_gpu_id == 0
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


def test_sae_config_allows_calibration_to_span_frozen_shuffle_buffers(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert payload["sae"]["l1_calibration_tokens"] > payload["data"]["shuffle_buffer_activations"]
    path = tmp_path / "multi-buffer.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert load_sae_protocol(path).l1_calibration_tokens == 10_000_000


def test_sae_config_rejects_unregistered_l1_selection_rule(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["sae"]["l1_selection_rule"] = "choose-after-looking/v1"
    path = tmp_path / "bad-selection-rule.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="L1 selection rule differs"):
        load_sae_protocol(path)


def test_sae_registry_rejects_data_role_drift(tmp_path: Path) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    payload["data_contract"]["forbidden_roles"][1] = "pre-pr-gate"
    path = tmp_path / "bad-registry-role.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data preregistration differs"):
        load_sae_preregistration(path, protocol=protocol)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protected_runs", []),
        ("prohibited_changes", []),
        ("protected_runs", ["nothing is protected"]),
        ("prohibited_changes", ["anything goes now"]),
    ],
)
def test_sae_registry_rejects_non_interference_list_drift(
    tmp_path: Path,
    field: str,
    value: list[str],
) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    payload["non_interference"][field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-interference protection lists differ"):
        load_sae_preregistration(path, protocol=protocol)


def test_sae_registry_rejects_unrecorded_calibration_grid(tmp_path: Path) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    payload["l1_calibration_amendment"]["adjusted_coefficients"] = [0.001, 0.01, 0.1]
    path = tmp_path / "bad-calibration-amendment.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="L1 calibration amendment differs"):
        load_sae_preregistration(path, protocol=protocol)


def test_sae_registry_binds_the_adjusted_calibration_token_budget(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["sae"]["l1_calibration_tokens"] = 8_192
    config_path = tmp_path / "tampered-token-budget.json"
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    protocol = load_sae_protocol(config_path)

    with pytest.raises(ValueError, match="L1 calibration amendment differs"):
        load_sae_preregistration(DEFAULT_REGISTRY, protocol=protocol)


def test_sae_registry_binds_the_calibration_shuffle_buffer_partition(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["data"]["shuffle_buffer_activations"] = 100_000
    config_path = tmp_path / "tampered-shuffle-buffer.json"
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    protocol = load_sae_protocol(config_path)

    with pytest.raises(ValueError, match="L1 calibration amendment differs"):
        load_sae_preregistration(DEFAULT_REGISTRY, protocol=protocol)


def test_sae_registry_rejects_tampered_calibration_buffer_count(tmp_path: Path) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    payload["l1_calibration_amendment"]["calibration_activation_buffers"] = 100
    registry_path = tmp_path / "tampered-buffer-count.json"
    registry_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="L1 calibration amendment differs"):
        load_sae_preregistration(registry_path, protocol=protocol)


def test_sae_registry_binds_calibration_optimizer_step_size(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["sae"]["activation_batch_size"] = 20_480
    config_path = tmp_path / "tampered-activation-batch-size.json"
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    protocol = load_sae_protocol(config_path)

    with pytest.raises(ValueError, match="L1 calibration amendment differs"):
        load_sae_preregistration(DEFAULT_REGISTRY, protocol=protocol)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("non_interference.gpu_reassignment_amendment", "from_gpu_id", True),
        ("non_interference.gpu_reassignment_amendment", "to_gpu_id", False),
        ("l1_calibration_amendment", "remaining_adjustments", False),
    ],
)
def test_sae_registry_rejects_boolean_amendment_integers(
    tmp_path: Path,
    section: str,
    field: str,
    value: bool,
) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    target = payload
    for key in section.split("."):
        target = target[key]
    target[field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_sae_preregistration(path, protocol=protocol)


def test_sae_registry_rejects_gpu_different_from_recorded_reassignment(tmp_path: Path) -> None:
    protocol = load_sae_protocol(DEFAULT_CONFIG)
    payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    payload["non_interference"]["sae_gpu_id"] = 1
    path = tmp_path / "bad-gpu-amendment.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="GPU reassignment amendment differs"):
        load_sae_preregistration(path, protocol=protocol)


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
            "0",
            "--wandb-project",
            "typo-sae",
            "--output-dir",
            "calibration",
        ]
    )
    assert calibration.training_data == [Path("clean-a.jsonl")]
    assert calibration.gpu_id == "0"

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
            "0",
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
            "0",
            "--output-dir",
            "validation",
        ]
    )
    assert validation.validation_data == [Path("held-in.jsonl")]

"""Frozen public contract for layer-constrained component localization."""

from __future__ import annotations

import argparse
from pathlib import Path

from typo_robust_training.cli import register_commands
from typo_robust_training.localization.component_config import (
    load_component_localization_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-component-localization.yaml"


def test_default_component_protocol_freezes_architecture_screen_and_causal_gate() -> None:
    protocol = load_component_localization_config(DEFAULT_CONFIG)

    assert protocol.model == "google/gemma-3-4b-it"
    assert protocol.model_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert protocol.decoder_layers == 34
    assert protocol.hidden_size == 2560
    assert protocol.mlp_intermediate_size == 10240
    assert protocol.attention_heads == 8
    assert protocol.attention_head_dim == 256
    assert protocol.tasks == ("gsm8k", "mmlu", "arc")
    assert protocol.partition_seed == 42
    assert protocol.activation_weight == 0.5
    assert protocol.attribution_weight == 0.5
    assert protocol.minimum_positive_attribution_tasks == 2
    assert protocol.mlp_shortlist_per_layer == 32
    assert protocol.attention_shortlist_per_layer == 8
    assert protocol.causal_candidate_limits == {"mlp-neuron": 12, "attention-head": 6}
    assert protocol.minimum_kl_eligible_per_task == 40
    assert protocol.minimum_answer_cohort_per_task == 5
    assert protocol.minimum_beneficial_tasks == 2
    assert protocol.maximum_harm_rate_per_task == 0.05
    assert protocol.minimum_selected_components == 1


def test_component_cli_requires_layer_selection_and_explicit_readouts() -> None:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command")
    register_commands(commands)
    args = root.parse_args(
        [
            "localize-robustness-components",
            "--config",
            "components.yaml",
            "--diagnostic-manifest",
            "diagnostic.jsonl",
            "--layer-selection",
            "layer_selection.json",
            "--components",
            "mlp-neuron",
            "attention-head",
            "--causal-readouts",
            "answer",
            "multitoken-kl",
            "--gpu-id",
            "3",
            "--output-dir",
            "components",
            "--resume",
        ]
    )
    assert args.layer_selection == Path("layer_selection.json")
    assert args.components == ["mlp-neuron", "attention-head"]
    assert args.causal_readouts == ["answer", "multitoken-kl"]
    assert args.gpu_id == "3"
    assert args.resume is True

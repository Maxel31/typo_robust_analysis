"""Fail-closed runtime and protocol checks for the faithful Kojima comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

from typo_robust_training.cli import register_commands
from typo_robust_training.training.adapters import (
    attach_lora_adapters,
    trainable_parameter_report,
)
from typo_robust_training.training.config import (
    is_kojima_faithful_protocol,
    is_probe_factorial_protocol,
    load_adapter_training_config,
)
from typo_robust_training.training.losses import (
    aligned_output_kl,
    aligned_soft_cross_entropy,
)
from typo_robust_training.training.runner import (
    TrainingMicroStepResult,
    _load_protocol_training_bundle,
    validate_micro_step_student_tokens,
)
from typo_robust_training.training.runtime import (
    HuggingFaceAdapterTrainingRuntime,
    resolve_attention_head_dim,
)
from typo_robust_training.training.tracking import build_wandb_run_presentation
from typo_robust_training.training.step import _output_matching_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/baselines/mistral7b-v01-kojima-faithful-output-matching-64m.yaml"


def _tiny_mistral() -> MistralForCausalLM:
    return MistralForCausalLM(
        MistralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _small_protocol():
    return replace(
        load_adapter_training_config(CONFIG),
        lora_rank=2,
        lora_alpha=4.0,
        gradient_checkpointing=False,
    )


def _mutated_config(tmp_path: Path, section: str, field: str, value: object) -> Path:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / f"bad-{section}-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_faithful_protocol_freezes_model_data_capacity_and_budget() -> None:
    protocol = load_adapter_training_config(CONFIG)

    assert protocol.condition == "kojima-faithful-output-matching"
    assert protocol.method_identity == "kojima-faithful-output-matching/v1"
    assert protocol.model == "mistralai/Mistral-7B-v0.1"
    assert protocol.model_revision == "7231864981174d9bee8c7687c24c8344414eae6b"
    assert protocol.decoder_layers == 32
    assert protocol.training_corpus == "HuggingFaceFW/fineweb"
    assert protocol.training_corpus_revision == "9bb295ddab0e05d785b879661af7260fed5140fc"
    assert protocol.training_corpus_data_file == "sample/10BT/000_00000.parquet"
    assert protocol.packing_policy == "kojima-bos-overfill500-canonicalize-truncate8192/v2"
    assert protocol.data_runtime_policy == "hash-attested-8800-attempt-skip-replace-stream/v2"
    assert protocol.max_sequence_length == 8192
    assert protocol.micro_batch_size == 1
    assert protocol.gradient_accumulation_steps == 8
    assert protocol.max_optimizer_steps == 1000
    assert protocol.max_student_tokens == 65_536_000
    assert protocol.seed_inventory == (1, 42, 43, 44)
    assert protocol.public_anchor_seed == 1
    assert protocol.matched_replication_seeds == (42, 43, 44)
    assert protocol.lora_target_modules == (
        "embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    )
    assert (
        protocol.max_sequence_length
        * protocol.micro_batch_size
        * protocol.gradient_accumulation_steps
        * protocol.max_optimizer_steps
        == protocol.max_student_tokens
    )


def test_factorial_v7_keeps_generic_bundle_and_kl_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factorial_v7 = SimpleNamespace(
        schema_version="robustness-adapter-training-config/v7",
        condition="factorial-probe-suffix-downstream-horizon",
        method_identity="factorial-probe-suffix-downstream-horizon/v1",
    )
    faithful = SimpleNamespace(
        schema_version="robustness-adapter-training-config/v7",
        condition="kojima-faithful-output-matching",
        method_identity="kojima-faithful-output-matching/v1",
    )
    generic_bundle = object()
    calls: list[str] = []

    def generic_loader(root: Path, *, protocol: object, seed: int) -> object:
        assert root == tmp_path
        assert protocol is factorial_v7
        assert seed == 42
        calls.append("generic")
        return generic_bundle

    def faithful_loader(root: Path, *, seed: int) -> object:
        del root, seed
        raise AssertionError("factorial v7 was misrouted into Kojima packing")

    monkeypatch.setattr(
        "typo_robust_training.training.runner.load_training_data_bundle",
        generic_loader,
    )
    monkeypatch.setattr(
        "typo_robust_training.training.runner.load_kojima_faithful_data_bundle",
        faithful_loader,
    )
    assert not is_kojima_faithful_protocol(factorial_v7)  # type: ignore[arg-type]
    assert (
        _load_protocol_training_bundle(  # type: ignore[arg-type]
            factorial_v7,
            root=tmp_path,
            seed=42,
        )
        is generic_bundle
    )
    assert calls == ["generic"]
    assert _output_matching_loss(factorial_v7) is aligned_output_kl  # type: ignore[arg-type]
    assert _output_matching_loss(faithful) is aligned_soft_cross_entropy  # type: ignore[arg-type]


def test_shared_v7_encoding_routes_by_condition_not_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    pair = SimpleNamespace(is_noop=False)
    factorial = SimpleNamespace(
        schema_version="robustness-adapter-training-config/v7",
        condition="factorial-probe-suffix-downstream-horizon",
        max_sequence_length=512,
        loss_weights={"answer": 0.0},
    )
    faithful = SimpleNamespace(
        schema_version="robustness-adapter-training-config/v7",
        condition="kojima-faithful-output-matching",
    )
    generic_calls: list[dict[str, object]] = []
    faithful_calls: list[object] = []

    def fake_generic(candidate: object, **kwargs: object) -> object:
        assert candidate is pair
        generic_calls.append(kwargs)
        return sentinel

    def fake_faithful(candidate: object, *, tokenizer: object) -> object:
        assert candidate is pair
        faithful_calls.append(tokenizer)
        return sentinel

    monkeypatch.setattr(
        "typo_robust_training.training.runtime.encode_training_pair",
        fake_generic,
    )
    monkeypatch.setattr(
        "typo_robust_training.training.runtime.encode_kojima_faithful_pair",
        fake_faithful,
    )
    tokenizer = object()
    factorial_runtime = SimpleNamespace(protocol=factorial, tokenizer=tokenizer)
    assert is_probe_factorial_protocol(factorial)  # type: ignore[arg-type]
    assert (
        HuggingFaceAdapterTrainingRuntime._encode_pair(  # type: ignore[arg-type]
            factorial_runtime,
            pair,
        )
        is sentinel
    )
    assert generic_calls == [
        {
            "tokenizer": tokenizer,
            "max_length": 512,
            "require_answer_targets": False,
            "require_all_edits_visible": True,
            "require_downstream_targets": True,
        }
    ]
    assert faithful_calls == []

    faithful_runtime = SimpleNamespace(protocol=faithful, tokenizer=tokenizer)
    assert not is_probe_factorial_protocol(faithful)  # type: ignore[arg-type]
    assert (
        HuggingFaceAdapterTrainingRuntime._encode_pair(  # type: ignore[arg-type]
            faithful_runtime,
            pair,
        )
        is sentinel
    )
    assert faithful_calls == [tokenizer]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sequence", "max_length", 4096),
        ("optimization", "max_student_tokens", 64_000_000),
        ("optimization", "max_optimizer_steps", 999),
        (
            "adapter",
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        ),
        (
            "adapter",
            "target_modules",
            [
                "embed_tokens",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        (
            "adapter",
            "target_modules",
            [
                "embed_tokens",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
                "vision_tower",
            ],
        ),
    ],
)
def test_faithful_protocol_rejects_counterfactual_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Kojima-faithful"):
        load_adapter_training_config(_mutated_config(tmp_path, section, field, value))


def test_missing_mistral_head_dim_is_derived_with_strict_validation() -> None:
    assert (
        resolve_attention_head_dim(
            SimpleNamespace(
                model_type="mistral",
                hidden_size=4096,
                num_attention_heads=32,
            )
        )
        == 128
    )
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_attention_head_dim(
            SimpleNamespace(
                model_type="mistral",
                hidden_size=4097,
                num_attention_heads=32,
            )
        )
    with pytest.raises(ValueError, match="explicit head_dim disagrees"):
        resolve_attention_head_dim(
            SimpleNamespace(
                model_type="mistral",
                hidden_size=4096,
                num_attention_heads=32,
                head_dim=127,
            )
        )


def test_gemma_explicit_noncanonical_head_dim_remains_supported() -> None:
    assert (
        resolve_attention_head_dim(
            SimpleNamespace(
                model_type="gemma3_text",
                hidden_size=2560,
                num_attention_heads=8,
                head_dim=256,
            )
        )
        == 256
    )


def test_faithful_all_linear_lora_attests_embedding_decoder_and_head() -> None:
    protocol = _small_protocol()
    adapted = attach_lora_adapters(
        _tiny_mistral(),
        protocol=protocol,
        decoder_layers=(0, 1),
    )
    report = trainable_parameter_report(
        adapted,
        expected_layers=(0, 1),
        expected_modules=protocol.lora_target_modules,
    )

    assert report.decoder_layers == (0, 1)
    assert set(report.modules) == set(protocol.lora_target_modules)
    assert any(".embed_tokens.lora_embedding_" in name for name in report.parameter_names)
    assert any(".lm_head.lora_" in name for name in report.parameter_names)
    assert all("vision" not in name for name in report.parameter_names)
    assert all(
        not parameter.requires_grad
        for name, parameter in adapted.named_parameters()
        if "lora_" not in name
    )


@pytest.mark.parametrize("missing", ["embed_tokens", "lm_head"])
def test_faithful_lora_rejects_missing_global_targets(missing: str) -> None:
    model = _tiny_mistral()
    if missing == "embed_tokens":
        del model.model.embed_tokens
    else:
        del model.lm_head

    with pytest.raises(ValueError, match="global LoRA target modules are missing"):
        attach_lora_adapters(
            model,
            protocol=_small_protocol(),
            decoder_layers=(0, 1),
        )


def test_faithful_lora_rejects_a_vision_target_collision() -> None:
    model = _tiny_mistral()
    model.vision_tower = torch.nn.Module()
    model.vision_tower.embed_tokens = torch.nn.Embedding(8, 16)

    with pytest.raises(ValueError, match="vision module collides"):
        attach_lora_adapters(
            model,
            protocol=_small_protocol(),
            decoder_layers=(0, 1),
        )


def test_faithful_packed_runtime_rejects_short_micro_steps() -> None:
    protocol = load_adapter_training_config(CONFIG)
    with pytest.raises(ValueError, match="fill the 8192-token context"):
        validate_micro_step_student_tokens(
            protocol,
            TrainingMicroStepResult(losses={"output": 1.0}, total_loss=1.0, student_tokens=8191),
        )
    validate_micro_step_student_tokens(
        protocol,
        TrainingMicroStepResult(losses={"output": 1.0}, total_loss=1.0, student_tokens=8192),
    )


def test_faithful_command_and_presentation_are_not_named_kojima_inspired() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    register_commands(commands)
    args = parser.parse_args(
        [
            "train-kojima-faithful-output-matching",
            "--config",
            str(CONFIG),
            "--training-data",
            "packed-fineweb",
            "--seed",
            "1",
            "--gpu-id",
            "0",
            "--wandb-project",
            "typo-robustness-training",
            "--output-dir",
            "output",
            "--evaluation-v2-registry-bundle",
            "training-preregistered-bundle.json",
        ]
    )
    assert args._training_condition == "kojima-faithful-output-matching"
    assert args.evaluation_v2_registry_bundle == Path("training-preregistered-bundle.json")

    faithful = build_wandb_run_presentation(
        condition="kojima-faithful-output-matching",
        schema_version="robustness-adapter-training-config/v7",
        model="mistralai/Mistral-7B-v0.1",
        seed=1,
        max_optimizer_steps=1000,
        max_student_tokens=65_536_000,
        state_gradient_ratio=None,
        state_layers=(),
    )
    inspired = build_wandb_run_presentation(
        condition="output-matching",
        schema_version="robustness-adapter-training-config/v3",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=10_000,
        max_student_tokens=64_000_000,
        state_gradient_ratio=None,
        state_layers=(),
    )
    assert faithful.name.startswith("Kojima-faithful baseline")
    assert inspired.name.startswith("Kojima-inspired baseline")
    assert set(faithful.tags).isdisjoint(tag for tag in inspired.tags if tag.startswith("arm:"))
    assert "seed-role:public-anchor" in faithful.tags
    assert "reproduction:hash-attested-faithful" in faithful.tags
    assert "bit-exact:false" in faithful.tags
    replication = build_wandb_run_presentation(
        condition="kojima-faithful-output-matching",
        schema_version="robustness-adapter-training-config/v7",
        model="mistralai/Mistral-7B-v0.1",
        seed=42,
        max_optimizer_steps=1000,
        max_student_tokens=65_536_000,
        state_gradient_ratio=None,
        state_layers=(),
    )
    assert "seed-role:matched-replication" in replication.tags
    assert "matched-inference:true" in replication.tags

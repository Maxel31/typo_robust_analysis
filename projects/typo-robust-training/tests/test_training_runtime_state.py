"""Runtime checkpoints restore CUDA RNG from CPU byte tensors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

from typo_robust_training.training.pairs import UnusableTrainingPairError
from typo_robust_training.training.adapters import attach_lora_adapters
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.runtime import (
    HuggingFaceAdapterTrainingRuntime,
    _adapter_scope_contract,
    _cpu_cuda_rng_states,
    _validate_adapter_scope_before_resume,
    _validated_resume_state_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs/baselines/noisy-language-model.yaml"


def _tiny_multimodal_model() -> Gemma3ForConditionalGeneration:
    text = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=32,
        layer_types=["full_attention"] * 3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
    )
    return Gemma3ForConditionalGeneration(
        Gemma3Config(
            text_config=text,
            vision_config=vision,
            mm_tokens_per_image=4,
            boi_token_index=60,
            eoi_token_index=61,
            image_token_index=62,
        )
    )


def _protocol():
    return replace(
        load_adapter_training_config(BASELINE_CONFIG),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        gradient_checkpointing=False,
    )


def _legacy_adapter():
    protocol = _protocol()
    model = _tiny_multimodal_model()
    model.requires_grad_(False)
    return get_peft_model(
        model,
        LoraConfig(
            r=protocol.lora_rank,
            lora_alpha=protocol.lora_alpha,
            lora_dropout=protocol.lora_dropout,
            bias=protocol.adapter_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(protocol.lora_target_modules),
            layers_to_transform=[0, 1, 2],
            layers_pattern="layers",
        ),
    )


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )


def _optimizer_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def test_cuda_rng_states_are_normalized_to_cpu_byte_tensors() -> None:
    states = _cpu_cuda_rng_states([torch.tensor([1, 2, 3], dtype=torch.uint8)])

    assert isinstance(states, tuple)
    assert len(states) == 1
    assert states[0].device.type == "cpu"
    assert states[0].dtype == torch.uint8
    assert states[0].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "states",
    [[], [torch.tensor([1], dtype=torch.int64)], ["not-a-tensor"]],
)
def test_cuda_rng_state_normalization_rejects_invalid_payloads(states: object) -> None:
    with pytest.raises(ValueError, match="CUDA RNG"):
        _cpu_cuda_rng_states(states)


def _semantic_calibration() -> dict[str, object]:
    return {
        "schema_version": "state-gradient-calibration/v1",
        "micro_batches": 8,
        "record_ids": [f"calibration-{index}" for index in range(8)],
        "output_gradient_norms": [2.0] * 8,
        "state_gradient_norms": [0.5] * 8,
        "mean_output_gradient_norm": 2.0,
        "mean_state_gradient_norm": 0.5,
        "target_gradient_ratio": 0.05,
        "state_weight": 0.2,
        "achieved_initial_ratio": 0.05,
    }


def test_semantic_resume_rederives_state_weight_and_rejects_empty_calibration() -> None:
    """A checkpoint cannot self-report an arbitrary lambda or erase its derivation."""

    protocol = SimpleNamespace(
        state_gradient_ratio=0.05,
        calibration_micro_batches=8,
        loss_weights={"state": 1.0},
    )
    weight, calibration = _validated_resume_state_calibration(
        protocol=protocol,
        state_weight=0.2,
        calibration=_semantic_calibration(),
        expected_calibration=_semantic_calibration(),
    )
    assert weight == 0.2
    assert calibration == _semantic_calibration()

    for malicious_weight, malicious_calibration in ((999.0, _semantic_calibration()), (0.2, {})):
        with pytest.raises(ValueError, match="checkpoint calibration"):
            _validated_resume_state_calibration(
                protocol=protocol,
                state_weight=malicious_weight,
                calibration=malicious_calibration,
                expected_calibration=_semantic_calibration(),
            )

    forged = _semantic_calibration()
    forged.update(
        {
            "record_ids": [f"forged-{index}" for index in range(8)],
            "output_gradient_norms": [9990.0] * 8,
            "mean_output_gradient_norm": 9990.0,
            "state_weight": 999.0,
        }
    )
    with pytest.raises(ValueError, match="checkpoint calibration"):
        _validated_resume_state_calibration(
            protocol=protocol,
            state_weight=999.0,
            calibration=forged,
            expected_calibration=_semantic_calibration(),
        )


def test_pair_usability_treats_resampleable_encoding_failures_as_unusable() -> None:
    class Runtime:
        @staticmethod
        def _encode_pair(_pair: object) -> None:
            raise UnusableTrainingPairError("pair cannot supply frozen targets")

    assert HuggingFaceAdapterTrainingRuntime.pair_is_usable(Runtime(), object()) is False  # type: ignore[arg-type]


def test_pair_usability_does_not_hide_tokenizer_contract_failures() -> None:
    class Runtime:
        @staticmethod
        def _encode_pair(_pair: object) -> None:
            raise ValueError("training tokenizer returned inconsistent sequence fields")

    with pytest.raises(ValueError, match="tokenizer returned inconsistent"):
        HuggingFaceAdapterTrainingRuntime.pair_is_usable(Runtime(), object())  # type: ignore[arg-type]


def test_scope_contract_accepts_the_same_decoder_only_adapter_and_optimizer() -> None:
    current = attach_lora_adapters(
        _tiny_multimodal_model(),
        protocol=_protocol(),
        decoder_layers=(0, 1, 2),
    )
    optimizer = _optimizer(current)
    adapter_state = get_peft_model_state_dict(current)
    optimizer_state = optimizer.state_dict()
    names = _optimizer_parameter_names(current)
    scope = _adapter_scope_contract(
        adapter_state=adapter_state,
        optimizer_state=optimizer_state,
        optimizer_parameter_names=names,
    )

    _validate_adapter_scope_before_resume(
        checkpoint_adapter=adapter_state,
        checkpoint_optimizer=optimizer_state,
        checkpoint_scope=scope,
        current_adapter=adapter_state,
        current_optimizer=optimizer_state,
        current_optimizer_parameter_names=names,
    )


def test_runtime_rejects_legacy_vision_lora_checkpoint_before_mutating_student(
    tmp_path: Path,
) -> None:
    legacy = _legacy_adapter()
    legacy_optimizer = _optimizer(legacy)
    current = attach_lora_adapters(
        _tiny_multimodal_model(),
        protocol=_protocol(),
        decoder_layers=(0, 1, 2),
    )
    current_optimizer = _optimizer(current)
    before = {
        name: tensor.detach().clone()
        for name, tensor in get_peft_model_state_dict(current).items()
    }
    state_path = tmp_path / "legacy-state.pt"
    torch.save(
        {
            "schema_version": "robustness-adapter-runtime-state/v2",
            "condition": "output-matching-only",
            "config_sha256": "a" * 64,
            "seed": 42,
            "optimizer_steps": 1,
            "adapter": get_peft_model_state_dict(legacy),
            "optimizer": legacy_optimizer.state_dict(),
            "scheduler": {},
            "state_weight": 0.0,
            "state_calibration": None,
            "gradient_ratio_violations": 0,
            "python_rng": None,
            "numpy_rng": None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (),
        },
        state_path,
    )
    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime._torch = torch
    runtime.student = current
    runtime.optimizer = current_optimizer
    runtime._optimizer_parameter_names = _optimizer_parameter_names(current)
    runtime.protocol = SimpleNamespace(
        condition="output-matching-only",
        config_sha256="a" * 64,
        state_gradient_ratio=None,
    )
    runtime.seed = 42

    with pytest.raises(ValueError, match="LoRA/optimizer scope differs"):
        runtime.load_state(state_path)

    after = get_peft_model_state_dict(current)
    assert tuple(before) == tuple(after)
    assert all(torch.equal(before[name], after[name]) for name in before)

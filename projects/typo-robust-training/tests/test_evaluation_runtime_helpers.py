"""Runtime helpers preserve token coordinates, KL direction, and whole-window patching."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import peft
import pytest
import torch
from typo_cot.evaluation import generation as generation_module
from typo_cot.experiments.layerwise_kl_patching import patching as patching_module
from typo_cot.models import wrapper as wrapper_module

from typo_robust_training.evaluation.checkpoints import AdapterDescriptor, PatchWindow
from typo_robust_training.evaluation.config import load_robustness_evaluation_config
from typo_robust_training.evaluation.prompting import EvaluationPrompts
from typo_robust_training.evaluation.runtime import (
    HuggingFaceRobustnessEvaluationRuntimeFactory,
    aligned_forward_kl_sum,
    causal_nll_and_forward_kl,
    causal_nll_sum,
    evaluation_teacher_targets,
    prompt_tokenization_profile,
    teacher_forced_kl_readout,
    window_patched_forward,
)
from typo_robust_training.localization.prompting import PromptSide


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/gemma4b-evaluation.yaml"


class _PieceTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        del kwargs
        pieces: list[tuple[str, tuple[int, int]]] = []
        for match in re.finditer(r"[A-Za-z]+|[^\w\s]", text):
            word = match.group()
            start, stop = match.span()
            if word == "airport":
                pieces.extend([("air", (start, start + 3)), ("port", (start + 3, stop))])
            elif word == "arport":
                pieces.extend(
                    [
                        ("ar", (start, start + 2)),
                        ("po", (start + 2, start + 4)),
                        ("rt", (start + 4, stop)),
                    ]
                )
            else:
                pieces.append((word, (start, stop)))
        return {
            "input_ids": [1, *range(2, len(pieces) + 2)],
            "attention_mask": [1] * (len(pieces) + 1),
            "offset_mapping": [(0, 0), *(span for _piece, span in pieces)],
        }


def test_prompt_profile_counts_subtokens_and_resolves_final_coordinates() -> None:
    prompts = EvaluationPrompts(
        record_id="a" * 64,
        answer=None,
        task_for_prompt=None,
        task_for_extractor=None,
        clean=PromptSide("The airport works.", ((4, 11),)),
        typo=PromptSide("The arport works.", ((4, 10),)),
    )

    profile = prompt_tokenization_profile(_PieceTokenizer(), prompts=prompts, max_tokens=32)

    assert profile.clean_positions == (3,)
    assert profile.typo_positions == (4,)
    assert profile.clean_subtoken_counts == (2,)
    assert profile.typo_subtoken_counts == (3,)
    assert profile.tokenization_stratum == "fragmentation-increased"
    assert len(profile.clean_input_ids) == 6
    assert len(profile.typo_input_ids) == 7


def test_teacher_forced_kl_is_clean_to_candidate_and_excludes_first_token() -> None:
    clean = torch.zeros(16, 3)
    candidate = clean.clone()
    candidate[0, 0] = 10.0
    candidate[1:, 1] = 2.0

    readout = teacher_forced_kl_readout(clean, candidate, teacher_forced_tokens=16)

    assert len(readout) == 15
    assert all(value > 0.0 for value in readout)
    assert teacher_forced_kl_readout(clean, clean, teacher_forced_tokens=16) == pytest.approx(
        (0.0,) * 15,
        abs=1e-7,
    )


def test_monitor_reductions_use_causal_and_aligned_forward_kl() -> None:
    base = torch.zeros(1, 4, 3)
    candidate = base.clone()
    candidate[0, 0, 1] = 2.0
    ids = torch.tensor([[0, 1, 2, 0]])

    nll, tokens, kl_sum, kl_tokens = causal_nll_and_forward_kl(
        candidate,
        ids,
        base_logits=base,
    )
    assert nll > 0.0
    assert tokens == kl_tokens == 3
    assert kl_sum > 0.0

    aligned_sum, aligned_tokens = aligned_forward_kl_sum(
        base,
        candidate,
        token_pairs=((0, 0), (1, 1), (2, 2)),
    )
    assert aligned_tokens == 2
    assert aligned_sum > 0.0


def test_monitor_forward_kl_clamps_floating_point_roundoff_per_position() -> None:
    """A mathematically non-negative KL must not invalidate a corpus record."""

    torch.manual_seed(2)
    base = torch.randn(1, 2, 3)
    candidate = base + torch.randn(1, 2, 3) * 1e-6
    ids = torch.tensor([[0, 1]])

    base_log_probs = base[:, :1].float().log_softmax(dim=-1)
    candidate_log_probs = candidate[:, :1].float().log_softmax(dim=-1)
    raw_kl = (base_log_probs.exp() * (base_log_probs - candidate_log_probs)).sum()
    assert float(raw_kl) < 0.0  # Counterexample without the numerical guard.

    _nll, _tokens, causal_kl, causal_tokens = causal_nll_and_forward_kl(
        candidate,
        ids,
        base_logits=base,
    )
    aligned_kl, aligned_tokens = aligned_forward_kl_sum(
        base,
        candidate,
        token_pairs=((1, 1),),
    )

    assert causal_tokens == aligned_tokens == 1
    assert causal_kl == 0.0
    assert aligned_kl == 0.0


def test_monitor_reductions_chunk_long_sequences_without_changing_values() -> None:
    torch.manual_seed(42)
    base = torch.randn(1, 11, 17, dtype=torch.bfloat16)
    candidate = torch.randn(1, 11, 17, dtype=torch.bfloat16)
    ids = torch.randint(0, 17, (1, 11))

    nll, count, kl_sum, kl_count = causal_nll_and_forward_kl(
        candidate,
        ids,
        base_logits=base,
    )
    candidate_log_probs = candidate[:, :-1].float().log_softmax(dim=-1)
    base_log_probs = base[:, :-1].float().log_softmax(dim=-1)
    expected_nll = -candidate_log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).sum()
    expected_kl = (
        (base_log_probs.exp() * (base_log_probs - candidate_log_probs))
        .sum(dim=-1)
        .clamp_min(0.0)
        .sum()
    )

    assert count == kl_count == 10
    assert nll == pytest.approx(float(expected_nll), rel=1e-6)
    assert kl_sum == pytest.approx(float(expected_kl), rel=1e-6)
    teacher_nll, teacher_count = causal_nll_sum(base, ids)
    expected_teacher = -base_log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).sum()
    assert teacher_count == 10
    assert teacher_nll == pytest.approx(float(expected_teacher), rel=1e-6)


def test_evaluation_teacher_targets_drop_partial_prefix_when_readout_is_invalid() -> None:
    targets, reason = evaluation_teacher_targets(
        (11, 12, 99),
        termination="eos",
        effective_eos_token_ids=(99,),
        count=16,
    )

    assert targets == ()
    assert reason == "clean-continuation-lt-16-before-eos"

    targets, reason = evaluation_teacher_targets(
        tuple(range(16)) + (99,),
        termination="eos",
        effective_eos_token_ids=(99,),
        count=16,
    )

    assert targets == tuple(range(16))
    assert reason is None


class _Add(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value


def test_window_patch_overwrites_every_selected_layer_only_at_edited_position() -> None:
    layers = [_Add(1.0), _Add(2.0), _Add(4.0)]
    hidden = torch.zeros(1, 3, 1)
    donors = (
        torch.tensor([[100.0]]),
        torch.tensor([[250.0]]),
    )

    output = window_patched_forward(
        layers,
        layer_indices=(0, 1),
        positions=(1,),
        donor_values=donors,
        forward=lambda: layers[2](layers[1](layers[0](hidden))),
    )

    assert output[0, 1, 0].item() == pytest.approx(254.0)
    assert output[0, 0, 0].item() == pytest.approx(7.0)
    assert output[0, 2, 0].item() == pytest.approx(7.0)


class _BaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.layers = torch.nn.ModuleList([_Add(1.0) for _ in range(34)])
        self.config = SimpleNamespace(_commit_hash=None)
        self.generation_config = SimpleNamespace()


class _PeftModel(torch.nn.Module):
    def __init__(self, base: _BaseModel) -> None:
        super().__init__()
        self.base = base
        self.loaded: list[tuple[str, str]] = []
        self.active: list[tuple[str, bool]] = []

    @classmethod
    def from_pretrained(
        cls,
        base: _BaseModel,
        path: Path,
        *,
        adapter_name: str,
        is_trainable: bool,
    ) -> _PeftModel:
        assert is_trainable is False
        model = cls(base)
        model.loaded.append((adapter_name, str(path)))
        return model

    def load_adapter(
        self,
        path: Path,
        *,
        adapter_name: str,
        is_trainable: bool,
    ) -> None:
        assert is_trainable is False
        self.loaded.append((adapter_name, str(path)))

    def set_adapter(self, adapter_name: str, *, inference_mode: bool) -> None:
        self.active.append((adapter_name, inference_mode))


def _descriptor(root: Path, *, condition: str, seed: int) -> AdapterDescriptor:
    return AdapterDescriptor(
        path=root / condition / f"seed-{seed}" / "adapter",
        condition=condition,
        seed=seed,
        config_sha256="a" * 64,
        training_data_sha256="b" * 64,
        data_identity_sha256="c" * 64,
        localization_sha256=("d" * 64 if condition == "localized-state-distillation" else None),
        adapter_sha256=f"{seed + 1:064x}",
    )


def test_runtime_factory_loads_base_once_and_switches_only_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _BaseModel()
    wrapper_calls: list[dict[str, object]] = []
    empty_cache_calls: list[None] = []

    def fake_wrapper(**kwargs: object) -> SimpleNamespace:
        wrapper_calls.append(kwargs)
        return SimpleNamespace(
            model=base,
            tokenizer=SimpleNamespace(is_fast=True, padding_side=None),
        )

    monkeypatch.setattr(wrapper_module, "create_model_wrapper", fake_wrapper)
    monkeypatch.setattr(patching_module, "find_decoder_layers", lambda model: model.layers)
    monkeypatch.setattr(
        generation_module,
        "resolve_effective_eos_token_ids",
        lambda **_kwargs: ((1,), "test"),
    )
    monkeypatch.setattr(peft, "PeftModel", _PeftModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _index: None)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(None),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    factory = HuggingFaceRobustnessEvaluationRuntimeFactory(
        protocol=load_robustness_evaluation_config(CONFIG),
        gpu_id="5",
        patch_window=PatchWindow(0, 6, "e" * 64, "d" * 64),
    )
    base_runtime = factory(None)
    assert base_runtime.model is base
    base_runtime.close()

    output = _descriptor(tmp_path, condition="output-matching", seed=42)
    output_runtime = factory(output)
    shared_model = output_runtime.model
    output_runtime.close()

    localized = _descriptor(
        tmp_path,
        condition="localized-state-distillation",
        seed=42,
    )
    localized_runtime = factory(localized)
    assert localized_runtime.model is shared_model
    localized_runtime.close()
    factory.close()

    assert len(wrapper_calls) == 1
    assert isinstance(shared_model, _PeftModel)
    assert shared_model.loaded == [
        ("evaluation_output_matching_seed_42", str(output.path)),
        ("evaluation_localized_state_distillation_seed_42", str(localized.path)),
    ]
    assert shared_model.active == [
        ("evaluation_output_matching_seed_42", True),
        ("evaluation_localized_state_distillation_seed_42", True),
    ]
    assert len(empty_cache_calls) == 1

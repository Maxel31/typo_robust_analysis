from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.evaluation.checkpoints import _validate_factorial_runtime
from typo_robust_training.training.adapters import attach_lora_adapters
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.encoding import (
    EDIT_DOWNSTREAM_OUTPUT_SCOPE,
    encode_training_pair,
    output_logit_pairs_for_scope,
)
from typo_robust_training.training.methods import (
    PROBE_FACTORIAL_CONDITIONS,
    ProbeTransitionTrainingEvidence,
    count_matched_random_layers,
    materialize_probe_output_factorial_configs,
)
from typo_robust_training.training.pairs import TrainingPair, UnusableTrainingPairError
from typo_robust_training.training.runtime import HuggingFaceAdapterTrainingRuntime
from typo_robust_training.training.runner import (
    _optimizer_step_telemetry,
    factorial_group_balanced_accumulation_scales,
)
from typo_robust_training.training.tracking import build_wandb_run_presentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "configs/proposals/gemma4b-probe-output-factorial-10m.template.yaml"
LEGACY_BOUND = PROJECT_ROOT / "tests/fixtures/gemma4b-probe-transition-output-10m.bound.json"


class _WordTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"<bos>": 1}

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        pieces = [(match.group(), match.span()) for match in re.finditer(r"[A-Za-z]+", text)]
        ids = [1]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for piece, span in pieces:
            ids.append(self.vocabulary.setdefault(piece, len(self.vocabulary) + 1))
            offsets.append(span)
        maximum = int(kwargs.get("max_length", len(ids)))
        return {
            "input_ids": ids[:maximum],
            "attention_mask": [1] * len(ids[:maximum]),
            "offset_mapping": offsets[:maximum],
        }


def _noisy_pair(*, trailing_words: int = 20) -> TrainingPair:
    suffix = " ".join(f"word{chr(97 + index % 26)}" for index in range(trailing_words))
    return TrainingPair(
        record_id="d" * 64,
        clean_text=f"airport {suffix}",
        typo_text=f"arport {suffix}",
        task=None,
        answer=None,
        metadata={},
        edits=(TypoEdit("deletion", "airport", "arport", (0, 7), (0, 6)),),
        is_noop=False,
        epoch=0,
    )


def _clean_pair() -> TrainingPair:
    text = "clean text with enough unchanged words for the complete alignment"
    return TrainingPair(
        record_id="c" * 64,
        clean_text=text,
        typo_text=text,
        task=None,
        answer=None,
        metadata={},
        edits=(),
        is_noop=True,
        epoch=0,
    )


def test_noisy_horizon_uses_target_offsets_two_through_sixteen_without_leakage() -> None:
    encoding = encode_training_pair(
        _noisy_pair(),
        tokenizer=_WordTokenizer(),
        max_length=64,
        require_downstream_targets=True,
    )
    edit = encoding.typo_edit_positions[0]
    selected = output_logit_pairs_for_scope(
        encoding,
        output_scope=EDIT_DOWNSTREAM_OUTPUT_SCOPE,
    )
    target_positions = {typo_logit + 1 for _clean_logit, typo_logit in selected}

    assert target_positions == set(range(edit + 2, edit + 17))
    assert edit not in target_positions  # the edited token never becomes a target
    assert edit + 1 not in target_positions  # R_2:16 starts at +2, not +1
    assert edit + 16 in target_positions
    assert edit + 17 not in target_positions


def test_horizon_scope_never_horizonizes_clean_rows() -> None:
    encoding = encode_training_pair(_clean_pair(), tokenizer=_WordTokenizer(), max_length=64)

    assert encoding.downstream_output_logit_pairs == ()
    assert (
        output_logit_pairs_for_scope(
            encoding,
            output_scope=EDIT_DOWNSTREAM_OUTPUT_SCOPE,
        )
        == encoding.output_logit_pairs
    )


def test_multiple_edit_horizons_are_deduplicated_as_one_union() -> None:
    suffix = " ".join(f"tail{chr(97 + index % 26)}" for index in range(20))
    pair = TrainingPair(
        record_id="e" * 64,
        clean_text=f"airport alpha terminal {suffix}",
        typo_text=f"arport alpha termnal {suffix}",
        task=None,
        answer=None,
        metadata={},
        edits=(
            TypoEdit("deletion", "airport", "arport", (0, 7), (0, 6)),
            TypoEdit("deletion", "terminal", "termnal", (14, 22), (13, 20)),
        ),
        is_noop=False,
        epoch=0,
    )
    encoding = encode_training_pair(
        pair,
        tokenizer=_WordTokenizer(),
        max_length=64,
        require_downstream_targets=True,
    )
    selected = output_logit_pairs_for_scope(
        encoding,
        output_scope=EDIT_DOWNSTREAM_OUTPUT_SCOPE,
    )
    all_aligned_targets = {
        typo_logit + 1 for _clean_logit, typo_logit in encoding.output_logit_pairs
    }
    expected = {
        edit + offset for edit in encoding.typo_edit_positions for offset in range(2, 17)
    } & all_aligned_targets

    assert {typo_logit + 1 for _clean_logit, typo_logit in selected} == expected
    assert len(selected) == len(set(selected))


def test_noisy_empty_horizon_is_rejected_instead_of_becoming_a_zero_loss_row() -> None:
    with pytest.raises(UnusableTrainingPairError, match=r"no aligned downstream \+2\.\.\+16"):
        encode_training_pair(
            _noisy_pair(trailing_words=1),
            tokenizer=_WordTokenizer(),
            max_length=4,
            require_downstream_targets=True,
        )


def test_factorial_accumulation_balances_clean_and_noisy_groups_not_target_counts() -> None:
    scales = factorial_group_balanced_accumulation_scales(
        output_token_counts=(511, 15) * 16,
        is_noop_rows=(True, False) * 16,
    )

    assert sum(scale.output for scale in scales[0::2]) == pytest.approx(0.5)
    assert sum(scale.output for scale in scales[1::2]) == pytest.approx(0.5)
    assert sum(scale.output for scale in scales) == pytest.approx(1.0)
    assert all(scale.state == 0.0 for scale in scales)


@pytest.mark.parametrize(
    ("counts", "groups", "message"),
    [
        ((511, 511), (True, True), "alternate clean and noisy"),
        ((15, 15), (False, False), "alternate clean and noisy"),
        ((511,), (True,), "exact 1:1"),
        ((15,), (False,), "alternate clean and noisy"),
        ((511, 15), (False, True), "alternate clean and noisy"),
    ],
)
def test_factorial_accumulation_rejects_missing_or_non_alternating_groups(
    counts: tuple[int, ...],
    groups: tuple[bool, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factorial_group_balanced_accumulation_scales(
            output_token_counts=counts,
            is_noop_rows=groups,
        )


def test_group_balancing_does_not_change_non_factorial_output_matching() -> None:
    protocol = replace(
        load_adapter_training_config(LEGACY_BOUND),
        gradient_accumulation_steps=2,
    )
    runtime = object.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime.protocol = protocol
    runtime.tokenizer = _WordTokenizer()
    runtime._prepared_encodings = deque()
    pairs = (_clean_pair(), _noisy_pair())
    counts = tuple(
        len(
            output_logit_pairs_for_scope(
                runtime._encode_pair(pair),
                output_scope=protocol.output_scope,
            )
        )
        for pair in pairs
    )

    scales = runtime.prepare_accumulation(pairs)

    assert counts[0] != counts[1]
    assert [scale.output for scale in scales] == pytest.approx(
        [count / sum(counts) for count in counts]
    )


def test_count_matched_random_freeze_is_reproducible_distinct_and_same_size() -> None:
    first = count_matched_random_layers(decoder_layers=34, selected_transition_layer=7)
    second = count_matched_random_layers(decoder_layers=34, selected_transition_layer=7)

    assert first == second
    assert first == (
        1,
        2,
        3,
        5,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
    )
    assert len(first) == len(tuple(range(7, 34)))
    assert first != tuple(range(7, 34))


def _evidence() -> ProbeTransitionTrainingEvidence:
    return ProbeTransitionTrainingEvidence(
        model="google/gemma-3-4b-it",
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        decoder_layers=34,
        selected_transition_layer=7,
        evidence_sha256="a" * 64,
    )


def test_factorial_materializer_changes_only_the_two_axes_and_random_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "typo_robust_training.training.methods.load_probe_transition_training_evidence",
        lambda *_args, **_kwargs: _evidence(),
    )
    evidence = tmp_path / "probe.json"
    evidence.write_text("fixture", encoding="utf-8")
    protocols = materialize_probe_output_factorial_configs(
        TEMPLATE,
        evidence_path=evidence,
        output_dir=tmp_path / "arms",
    )

    assert tuple(protocols) == PROBE_FACTORIAL_CONDITIONS
    assert all(protocol.loss_weights["state"] == 0.0 for protocol in protocols.values())
    assert all(protocol.loss_weights["answer"] == 0.0 for protocol in protocols.values())
    # Eligibility is intentionally shared by all five arms.  Otherwise an
    # end-of-sequence typo would be skipped only by horizon arms and silently
    # desynchronize subsequent source order/typo realizations.
    for protocol in protocols.values():
        runtime = object.__new__(HuggingFaceAdapterTrainingRuntime)
        runtime.protocol = protocol
        runtime.tokenizer = _WordTokenizer()
        runtime._prepared_encodings = deque()
        with pytest.raises(UnusableTrainingPairError, match="no aligned downstream"):
            runtime._encode_pair(_noisy_pair(trailing_words=1))
        pairs = tuple(
            pair
            for _ in range(protocol.gradient_accumulation_steps // 2)
            for pair in (_clean_pair(), _noisy_pair())
        )
        scales = runtime.prepare_accumulation(pairs)
        assert sum(scale.output for scale in scales[0::2]) == pytest.approx(0.5)
        assert sum(scale.output for scale in scales[1::2]) == pytest.approx(0.5)
        runtime._prepared_encodings.clear()
        with pytest.raises(ValueError, match="prepared accumulation size differs"):
            runtime.prepare_accumulation(pairs[:-1])
    payloads = {
        condition: json.loads((tmp_path / "arms" / f"{condition}.json").read_text())
        for condition in PROBE_FACTORIAL_CONDITIONS
    }
    reference = payloads[PROBE_FACTORIAL_CONDITIONS[0]]
    for payload in payloads.values():
        assert payload["model"] == reference["model"]
        assert payload["sequence"] == reference["sequence"]
        assert payload["optimization"] == reference["optimization"]
        assert {
            key: value
            for key, value in payload["adapter"].items()
            if key not in {"layer_scope", "layer_policy"}
        } == {
            key: value
            for key, value in reference["adapter"].items()
            if key not in {"layer_scope", "layer_policy"}
        }
        assert {
            key: value for key, value in payload["objective"].items() if key != "output_scope"
        } == {key: value for key, value in reference["objective"].items() if key != "output_scope"}


def _tiny_model() -> Gemma3ForCausalLM:
    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
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
    )


def _layer_parameter_map(model: object, *, layer: int) -> dict[str, torch.Tensor]:
    return {
        name.split(f".layers.{layer}.", 1)[1]: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
        if parameter.requires_grad and f".layers.{layer}." in name
    }


def test_shared_lora_coordinates_have_identical_initial_values_across_arm_scopes() -> None:
    protocol = replace(
        load_adapter_training_config(LEGACY_BOUND),
        lora_rank=2,
        lora_alpha=4.0,
        gradient_checkpointing=False,
        adapter_initialization_policy="sha256-layer-keyed-kaiming-a-zero-b/v1",
    )
    suffix = attach_lora_adapters(
        _tiny_model(), protocol=protocol, decoder_layers=(1, 2), initialization_seed=42
    )
    all_layers = attach_lora_adapters(
        _tiny_model(), protocol=protocol, decoder_layers=(0, 1, 2), initialization_seed=42
    )

    for layer in (1, 2):
        suffix_values = _layer_parameter_map(suffix, layer=layer)
        all_values = _layer_parameter_map(all_layers, layer=layer)
        assert suffix_values.keys() == all_values.keys()
        assert all(torch.equal(suffix_values[name], all_values[name]) for name in suffix_values)
    assert any(
        torch.count_nonzero(value)
        for name, value in _layer_parameter_map(suffix, layer=1).items()
        if ".lora_A." in name
    )
    different_seed = attach_lora_adapters(
        _tiny_model(), protocol=protocol, decoder_layers=(1, 2), initialization_seed=43
    )
    seed42 = _layer_parameter_map(suffix, layer=1)
    seed43 = _layer_parameter_map(different_seed, layer=1)
    assert any(not torch.equal(seed42[name], seed43[name]) for name in seed42 if ".lora_A." in name)


def test_layer_keyed_adapter_insertion_preserves_the_training_rng_stream() -> None:
    protocol = replace(
        load_adapter_training_config(LEGACY_BOUND),
        lora_rank=2,
        lora_alpha=4.0,
        gradient_checkpointing=False,
        adapter_initialization_policy="sha256-layer-keyed-kaiming-a-zero-b/v1",
    )

    def attach_and_read_rng(layers: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(777)
        model = _tiny_model()
        torch.manual_seed(999)
        before = torch.get_rng_state().clone()
        attach_lora_adapters(
            model,
            protocol=protocol,
            decoder_layers=layers,
            initialization_seed=42,
        )
        return before, torch.get_rng_state().clone()

    suffix_before, suffix_after = attach_and_read_rng((1, 2))
    all_before, all_after = attach_and_read_rng((0, 1, 2))

    assert torch.equal(suffix_before, suffix_after)
    assert torch.equal(all_before, all_after)
    assert torch.equal(suffix_after, all_after)


def test_factorial_telemetry_separates_clean_and_noisy_weighted_output_means() -> None:
    rows: list[dict[str, object]] = []
    for is_noop, output in ((True, 2.0), (False, 10.0), (True, 4.0), (False, 20.0)):
        rows.append(
            {
                "student_tokens": 512 if is_noop else 16,
                "edit_count": 0 if is_noop else 1,
                "is_noop": is_noop,
                "total_loss": output,
                "losses": {
                    "output": output,
                    "state": 0.0,
                    "weighted_state": 0.0,
                    "output_accumulation_scale": 0.25,
                    "state_accumulation_scale": 0.0,
                    "backward_contribution": output * 0.25,
                },
            }
        )

    metrics = _optimizer_step_telemetry(
        rows,
        optimizer_step=1,
        micro_steps=4,
        cumulative_student_tokens=1056,
        gradient_norm=0.5,
        learning_rate=1e-4,
        elapsed_seconds=1.0,
        runtime=SimpleNamespace(telemetry=lambda: {}),
    )

    assert metrics["train/objective/output_clean_mean"] == pytest.approx(3.0)
    assert metrics["train/objective/output_noisy_mean"] == pytest.approx(15.0)
    assert metrics["train/objective/output"] == pytest.approx(9.0)


def test_factorial_wandb_presentation_calls_trainable_coordinates_adapter_layers() -> None:
    presentation = build_wandb_run_presentation(
        condition="factorial-probe-suffix-downstream-horizon",
        schema_version="robustness-adapter-training-config/v7",
        model="google/gemma-3-4b-it",
        seed=42,
        max_optimizer_steps=10_000,
        max_student_tokens=10_000_000,
        state_gradient_ratio=None,
        state_layers=tuple(range(7, 34)),
    )

    assert "Adapter layers: L7–33." in presentation.notes
    assert "State layers:" not in presentation.notes
    assert "role:proposed" in presentation.tags


def _factorial_runtime_payload(
    *, adapter_layers: list[int], output_scope: str
) -> dict[str, object]:
    return {
        "decoder_layers": 34,
        "adapter_layers": adapter_layers,
        "output_scope": output_scope,
        "adapter_initialization_policy": "sha256-layer-keyed-kaiming-a-zero-b/v1",
        "state_layers": [],
        "state_weight": 0.0,
        "state_calibration": None,
    }


def test_evaluation_binds_factorial_label_to_actual_runtime_axes() -> None:
    condition = "factorial-probe-suffix-downstream-horizon"
    valid = _factorial_runtime_payload(
        adapter_layers=list(range(7, 34)),
        output_scope=EDIT_DOWNSTREAM_OUTPUT_SCOPE,
    )
    _validate_factorial_runtime(condition, valid)

    wrong_horizon = dict(valid, output_scope="aligned-non-edited-next-token/v1")
    with pytest.raises(ValueError, match="objective or scope"):
        _validate_factorial_runtime(condition, wrong_horizon)

    wrong_layers = dict(valid, adapter_layers=[7, 9, *range(10, 34)])
    with pytest.raises(ValueError, match="adapter layers differ"):
        _validate_factorial_runtime(condition, wrong_layers)

    wrong_initialization = dict(valid, adapter_initialization_policy="peft-default/v1")
    with pytest.raises(ValueError, match="objective or scope"):
        _validate_factorial_runtime(condition, wrong_initialization)

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ucc_inj_reproduction.exp6 import (
    ADAPTATION_SCOPE,
    COMPARISON_POSITION,
    Exp6Config,
    ExtractedStates,
    HiddenStateExtractor,
    InputTooLongError,
    cosine_for_extracted_pair,
    cosine_per_layer,
    last_non_padding_index,
    run_exp6,
    stable_example_seed,
    summarise_layer_cosines,
    write_exp6_results,
)


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, key: Any) -> FakeTensor:
        return FakeTensor(self.values[key])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values)

    def to(self, _device: str) -> FakeTensor:
        return self


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    init_kwargs = {"_commit_hash": "tokenizer-commit"}
    chat_template = "<user>{{ content }}</user>"
    vocab_size = 100
    padding_side = "right"
    truncation_side = "right"
    model_max_length = 100

    def __init__(self, token_ids: dict[str, list[int]]) -> None:
        self.token_ids = token_ids
        self.tokenize_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any:
        del add_generation_prompt
        text = messages[0]["content"]
        if not tokenize:
            return f"<user>{text}</user>"
        self.tokenize_kwargs.append(kwargs)
        values = self.token_ids[text]
        return {
            "input_ids": FakeTensor([values]),
            "attention_mask": FakeTensor([[1] * len(values)]),
        }


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, input_ids: FakeTensor, **_kwargs: Any) -> Any:
        self.calls += 1
        length = input_ids.values.shape[-1]
        base = np.arange(1, length * 2 + 1, dtype=float).reshape(1, length, 2)
        return SimpleNamespace(hidden_states=(FakeTensor(base), FakeTensor(base + 1.0)))


class FakeTorch:
    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


def make_extracted(
    values: np.ndarray,
    *,
    input_ids: tuple[int, ...] = (1, 9),
    logical_position: str = COMPARISON_POSITION,
    suffix: str = "same-suffix",
) -> ExtractedStates:
    return ExtractedStates(
        hidden_states=np.asarray(values, dtype=float),
        input_ids=input_ids,
        input_token_count=len(input_ids),
        token_index=len(input_ids) - 1,
        terminal_token_id=input_ids[-1],
        input_ids_sha256="input-hash",
        template_suffix_sha256=suffix,
        logical_position=logical_position,
    )


class FakeExtractor:
    def states(self, text: str) -> ExtractedStates:
        if any(ord(character) >= 0xFE00 for character in text):
            return make_extracted(
                np.asarray([[1.0, 3.0], [2.0, 5.0]]),
                input_ids=(1, 2, 9),
            )
        return make_extracted(np.asarray([[1.0, 2.0], [3.0, 4.0]]))


class MismatchedPositionExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def states(self, _text: str) -> ExtractedStates:
        self.calls += 1
        position = COMPARISON_POSITION if self.calls == 1 else "different_position"
        return make_extracted(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            logical_position=position,
        )


def test_config_allows_level_zero_and_rejects_false_faithful_scope() -> None:
    Exp6Config(model="unit-test", noise_levels=(0, 1), device="cpu").validate()
    with pytest.raises(ValueError, match="non-negative"):
        Exp6Config(model="unit-test", noise_levels=(0, -1), device="cpu").validate()
    with pytest.raises(ValueError, match="level-0"):
        Exp6Config(model="unit-test", noise_levels=(1,), device="cpu").validate()
    with pytest.raises(ValueError, match="faithful"):
        Exp6Config(
            model="unit-test",
            protocol_scope="faithful",
            noise_levels=(0,),
            device="cpu",
        ).validate()


def test_last_non_padding_index_handles_left_and_right_padding() -> None:
    assert last_non_padding_index([0, 0, 1, 1]) == 3
    assert last_non_padding_index([1, 1, 0, 0]) == 1
    with pytest.raises(ValueError, match="no non-padding"):
        last_non_padding_index([0, 0])


def test_extractor_refuses_overlong_input_before_forward() -> None:
    tokenizer = FakeTokenizer({"too long": [1, 2, 3]})
    model = FakeModel()
    extractor = HiddenStateExtractor(
        tokenizer=tokenizer,
        model=model,
        config=Exp6Config(
            model="unit-test",
            noise_levels=(0,),
            max_length=2,
            device="cpu",
        ),
        torch_module=FakeTorch(),
    )

    with pytest.raises(InputTooLongError, match="refusing to truncate"):
        extractor.states("too long")

    assert model.calls == 0
    assert tokenizer.tokenize_kwargs[0]["truncation"] is False
    assert "max_length" not in tokenizer.tokenize_kwargs[0]


def test_extractor_uses_each_complete_inputs_own_terminal_token() -> None:
    tokenizer = FakeTokenizer({"clean": [1, 9], "noisy": [2, 3, 9]})
    model = FakeModel()
    extractor = HiddenStateExtractor(
        tokenizer=tokenizer,
        model=model,
        config=Exp6Config(model="unit-test", noise_levels=(0,), device="cpu"),
        torch_module=FakeTorch(),
    )

    clean = extractor.states("clean")
    noisy = extractor.states("noisy")

    assert clean.token_index == 1
    assert noisy.token_index == 2
    assert clean.terminal_token_id == noisy.terminal_token_id == 9
    assert clean.logical_position == noisy.logical_position == COMPARISON_POSITION
    assert model.calls == 2


def test_cosine_per_layer_rejects_undefined_or_nonfinite_values() -> None:
    result = cosine_per_layer(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.0, 1.0]]),
    )
    assert result[0] == 0.0
    with pytest.raises(ValueError, match="zero"):
        cosine_per_layer(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[1.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="non-finite"):
        cosine_per_layer(
            np.asarray([[float("nan"), 1.0]]),
            np.asarray([[1.0, 1.0]]),
        )


def test_extracted_pair_rejects_mismatched_logical_position() -> None:
    clean = make_extracted(np.asarray([[1.0, 2.0]]))
    noisy = make_extracted(
        np.asarray([[1.0, 2.0]]),
        logical_position="different_position",
    )
    with pytest.raises(ValueError, match="same logical position"):
        cosine_for_extracted_pair(clean, noisy)


def test_level_zero_runs_independently_as_identity_control() -> None:
    config = Exp6Config(
        model="unit-test",
        noise_levels=(0,),
        device="cpu",
        limit=None,
    )
    records, summary, provenance = run_exp6(
        config,
        questions=["one"],
        extractor=FakeExtractor(),
    )

    assert len(records) == 1
    record = records[0]
    assert record["protocol_scope"] == ADAPTATION_SCOPE
    assert record["clean_text_sha256"] == record["noisy_text_sha256"]
    assert record["clean_input_ids_sha256"] == record["noisy_input_ids_sha256"]
    assert record["cosine_similarity"] == pytest.approx([1.0, 1.0])
    assert summary[0]["hidden_state_index"] == 0
    assert provenance["protocol"]["scope"] == ADAPTATION_SCOPE


def test_exp6_records_every_question_and_noise_level() -> None:
    config = Exp6Config(
        model="unit-test",
        noise_levels=(0, 1, 3),
        device="cpu",
        limit=None,
    )
    records, summary, _ = run_exp6(
        config,
        questions=["one", "two"],
        extractor=FakeExtractor(),
    )
    assert [(record["example_index"], record["noise_level"]) for record in records] == [
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 3),
    ]
    assert {
        (row["noise_level"], row["hidden_state_index"], row["n"])
        for row in summary
    } == {
        (0, 0, 2),
        (0, 1, 2),
        (1, 0, 2),
        (1, 1, 2),
        (3, 0, 2),
        (3, 1, 2),
    }


def test_run_rejects_mismatched_logical_positions() -> None:
    config = Exp6Config(model="unit-test", noise_levels=(0,), device="cpu")
    with pytest.raises(ValueError, match="same logical position"):
        run_exp6(
            config,
            questions=["one"],
            extractor=MismatchedPositionExtractor(),
        )


def test_stable_seed_ignores_python_process_hash_randomisation() -> None:
    assert stable_example_seed(
        root_seed=42,
        example_index=1,
        noise_level=2,
    ) == stable_example_seed(root_seed=42, example_index=1, noise_level=2)
    assert stable_example_seed(
        root_seed=42,
        example_index=1,
        noise_level=2,
    ) != stable_example_seed(root_seed=42, example_index=1, noise_level=3)


def test_summary_rejects_nonfinite_cosines() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        summarise_layer_cosines(
            [{"noise_level": 1, "cosine_similarity": [float("nan")]}]
        )


def test_writer_serialises_provenance_and_refuses_nan_before_mkdir(
    tmp_path: Path,
) -> None:
    config = Exp6Config(model="unit-test", noise_levels=(0,), device="cpu")
    valid_output = tmp_path / "valid"
    provenance = {
        "schema_version": "ucc-inj-exp6-provenance/v2",
        "protocol": {"scope": ADAPTATION_SCOPE},
        "resolved": {"dataset": {"fingerprint": "dataset-fingerprint"}},
    }
    write_exp6_results(
        output_dir=valid_output,
        config=config,
        records=[{"noise_level": 0, "cosine_similarity": [1.0]}],
        summary=[],
        provenance=provenance,
    )
    assert json.loads((valid_output / "provenance.json").read_text()) == provenance

    invalid_output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        write_exp6_results(
            output_dir=invalid_output,
            config=config,
            records=[{"noise_level": 0, "cosine_similarity": [float("nan")]}],
            summary=[],
            provenance=provenance,
        )
    assert not invalid_output.exists()

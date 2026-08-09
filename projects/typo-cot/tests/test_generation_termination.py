"""Shared generation-stop contracts for source and fresh evaluations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from typo_cot.evaluation.generation import (
    classify_generated_token_ids,
    resolve_effective_eos_token_ids,
)


def test_effective_eos_ids_prefer_the_model_generation_config() -> None:
    generation_config = SimpleNamespace(
        stop_strings=None,
        forced_eos_token_id=None,
        eos_token_id=[106, 1, 106],
    )
    tokenizer = SimpleNamespace(eos_token_id=99)

    assert resolve_effective_eos_token_ids(
        generation_config=generation_config,
        tokenizer=tokenizer,
        operation="fixture generation",
    ) == ((1, 106), "model-generation-config")


def test_effective_eos_ids_fall_back_to_the_tokenizer() -> None:
    generation_config = SimpleNamespace(
        stop_strings=None,
        forced_eos_token_id=None,
        eos_token_id=None,
    )

    assert resolve_effective_eos_token_ids(
        generation_config=generation_config,
        tokenizer=SimpleNamespace(eos_token_id=(99, 100)),
        operation="fixture generation",
    ) == ((99, 100), "tokenizer-fallback")


@pytest.mark.parametrize(
    ("field", "value"),
    (("stop_strings", ["STOP"]), ("forced_eos_token_id", 99)),
)
def test_effective_eos_ids_reject_implicit_stop_mechanisms(
    field: str,
    value: object,
) -> None:
    generation_config = SimpleNamespace(
        stop_strings=None,
        forced_eos_token_id=None,
        eos_token_id=99,
    )
    setattr(generation_config, field, value)

    with pytest.raises(ValueError, match=field):
        resolve_effective_eos_token_ids(
            generation_config=generation_config,
            tokenizer=SimpleNamespace(eos_token_id=99),
            operation="fixture generation",
        )


def test_generation_classifier_distinguishes_eos_at_cap_from_length_cap() -> None:
    eos_at_cap = (*([41] * 511), 106)
    no_eos_at_cap = tuple([41] * 512)

    assert classify_generated_token_ids(
        eos_at_cap,
        effective_eos_token_ids=(1, 106),
        max_new_tokens=512,
        field="fixture",
    ) == (eos_at_cap, "eos")
    assert classify_generated_token_ids(
        no_eos_at_cap,
        effective_eos_token_ids=(1, 106),
        max_new_tokens=512,
        field="fixture",
    ) == (no_eos_at_cap, "length-cap")


def test_generation_classifier_truncates_batch_padding_after_the_first_eos() -> None:
    assert classify_generated_token_ids(
        (41, 106, 0, 0),
        effective_eos_token_ids=(1, 106),
        max_new_tokens=512,
        field="fixture",
    ) == ((41, 106), "eos")


def test_generation_classifier_rejects_an_unexplained_short_stop() -> None:
    with pytest.raises(ValueError, match="without EOS before the token cap"):
        classify_generated_token_ids(
            (41,),
            effective_eos_token_ids=(1, 106),
            max_new_tokens=512,
            field="fixture",
        )

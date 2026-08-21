"""Pure runtime contracts for target reuse, readout slicing, and answer work."""

from __future__ import annotations

import pytest

from typo_robust_training.localization.runtime import (
    clean_teacher_targets,
    readout_kl_2_16,
    should_generate_patched_answers,
)


def test_clean_targets_strip_eos_and_are_reused_as_exact_sixteen_token_prefix() -> None:
    targets, reason = clean_teacher_targets(
        tuple(range(20)) + (99,),
        termination="eos",
        effective_eos_token_ids=(99, 100),
        count=16,
    )
    assert targets == tuple(range(16))
    assert reason is None

    short, reason = clean_teacher_targets(
        (1, 2, 99),
        termination="eos",
        effective_eos_token_ids=(99,),
        count=16,
    )
    assert short == (1, 2)
    assert reason == "clean-continuation-lt-16-before-eos"


def test_target_contract_rejects_eos_inside_content_or_wrong_termination() -> None:
    with pytest.raises(ValueError, match="first EOS"):
        clean_teacher_targets(
            (1, 99, 2, 99),
            termination="eos",
            effective_eos_token_ids=(99,),
            count=2,
        )
    with pytest.raises(ValueError, match="termination"):
        clean_teacher_targets(
            tuple(range(16)),
            termination="other",
            effective_eos_token_ids=(99,),
            count=16,
        )


def test_readout_always_excludes_first_token_and_keeps_tokens_two_through_sixteen() -> None:
    values = tuple(float(index) for index in range(16))
    assert readout_kl_2_16(values, teacher_forced_tokens=16) == tuple(
        float(index) for index in range(1, 16)
    )
    with pytest.raises(ValueError, match="sixteen"):
        readout_kl_2_16(values[:15], teacher_forced_tokens=16)


def test_answer_patches_are_generated_only_for_clean_correct_records() -> None:
    assert should_generate_patched_answers(clean_correct=True) is True
    assert should_generate_patched_answers(clean_correct=False) is False
    with pytest.raises(TypeError, match="boolean"):
        should_generate_patched_answers(clean_correct=1)  # type: ignore[arg-type]

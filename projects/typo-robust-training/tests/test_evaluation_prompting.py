"""Paper prompts transport every edited-word coordinate without heuristic shifts."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.evaluation.data import EvaluationPair
from typo_robust_training.evaluation.prompting import (
    build_evaluation_prompts,
    classify_tokenization_counts,
)


def _pair(
    *,
    task: str | None,
    answer: str | None,
    metadata: dict[str, object] | None = None,
) -> EvaluationPair:
    return EvaluationPair(
        record_id="a" * 64,
        kind="synthetic",
        source="mmlu_pro" if task == "mmlu_pro" else "fineweb_edu",
        source_revision="b" * 40,
        source_split="test",
        source_id="source-1",
        group_id="group-1",
        role="pre_pr_gate",
        clean_text="The airport is open.",
        typo_text="The arport is open.",
        task=task,
        answer=answer,
        operation="deletion",
        edits=(
            TypoEdit(
                operation="deletion",
                clean_word="airport",
                typo_word="arport",
                clean_char_span=(4, 11),
                typo_char_span=(4, 10),
            ),
        ),
        mechanistic_audit=False,
        metadata=MappingProxyType(metadata or {"category": "business"}),
        strata=("unseen-task",) if task else ("unseen-content",),
    )


def test_task_pair_uses_paper_template_and_preserves_both_word_spans() -> None:
    prompts = build_evaluation_prompts(_pair(task="mmlu_pro", answer="B"))

    assert "Step-by-step reasoning:" in prompts.clean.text
    assert "The following are multiple-choice questions" in prompts.clean.text
    assert prompts.clean.text[slice(*prompts.clean.spans[0])] == "airport"
    assert prompts.typo.text[slice(*prompts.typo.spans[0])] == "arport"
    assert prompts.task_for_prompt == "mmlu_pro"
    assert prompts.task_for_extractor == "mmlu_pro"


def test_general_text_pair_has_no_task_wrapper_or_coordinate_shift() -> None:
    prompts = build_evaluation_prompts(_pair(task=None, answer=None))

    assert prompts.clean.text == "The airport is open."
    assert prompts.typo.text == "The arport is open."
    assert prompts.clean.spans == ((4, 11),)
    assert prompts.typo.spans == ((4, 10),)
    assert prompts.task_for_prompt is None
    assert prompts.task_for_extractor is None


def test_empty_subject_is_not_replaced_by_category() -> None:
    prompts = build_evaluation_prompts(
        _pair(
            task="mmlu_pro",
            answer="B",
            metadata={"subject": "", "category": "business"},
        )
    )

    assert "questions about general knowledge." in prompts.clean.text
    assert "questions about business." not in prompts.clean.text


@pytest.mark.parametrize(
    ("clean", "typo", "expected"),
    [
        ((1,), (1,), "same-subtoken-count"),
        ((1,), (2,), "fragmentation-increased"),
        ((3,), (1,), "fragmentation-decreased"),
        ((1, 3), (2, 2), "mixed-subtoken-change"),
    ],
)
def test_tokenization_strata_are_edit_aligned(
    clean: tuple[int, ...], typo: tuple[int, ...], expected: str
) -> None:
    assert classify_tokenization_counts(clean, typo) == expected


def test_tokenization_strata_reject_mismatched_edit_cardinality() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        classify_tokenization_counts((1,), (1, 2))

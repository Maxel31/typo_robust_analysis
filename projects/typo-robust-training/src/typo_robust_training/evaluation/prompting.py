"""Paper-compatible prompts with exact evaluation edit coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from typo_robust_training.evaluation.data import EvaluationPair
from typo_robust_training.localization.prompting import PromptSide


_TASK_MAP = {"math_500": "math"}


@dataclass(frozen=True, slots=True)
class EvaluationPrompts:
    record_id: str
    answer: str | None
    task_for_prompt: str | None
    task_for_extractor: str | None
    clean: PromptSide
    typo: PromptSide


def _shifted_spans(pair: EvaluationPair, *, clean_offset: int, typo_offset: int):
    clean = tuple(
        (edit.clean_char_span[0] + clean_offset, edit.clean_char_span[1] + clean_offset)
        for edit in pair.edits
    )
    typo = tuple(
        (edit.typo_char_span[0] + typo_offset, edit.typo_char_span[1] + typo_offset)
        for edit in pair.edits
    )
    return clean, typo


def build_evaluation_prompts(pair: EvaluationPair) -> EvaluationPrompts:
    """Insert one held-out pair into the same task templates as the paper."""

    if not isinstance(pair, EvaluationPair):
        raise TypeError("evaluation prompting requires an EvaluationPair")
    if pair.task is None:
        clean_prompt, typo_prompt = pair.clean_text, pair.typo_text
        clean_offset = typo_offset = 0
        task = None
    else:
        from typo_cot.models.prompts import create_prompt_template

        task = _TASK_MAP.get(pair.task, pair.task)
        subject = pair.metadata.get("subject")
        if subject is None:
            subject = pair.metadata.get("category")
        if subject is not None and not isinstance(subject, str):
            raise ValueError("evaluation prompt subject/category metadata must be a string")
        template = create_prompt_template(task)
        clean_result = template.generate(question=pair.clean_text, subject=subject)
        typo_result = template.generate(question=pair.typo_text, subject=subject)
        clean_prompt = clean_result.get_full_prompt()
        typo_prompt = typo_result.get_full_prompt()
        clean_offset = int(clean_result.question_start_in_full)
        typo_offset = int(typo_result.question_start_in_full)
        if clean_prompt[clean_offset : clean_offset + len(pair.clean_text)] != pair.clean_text:
            raise ValueError("clean evaluation question coordinate differs from its paper prompt")
        if typo_prompt[typo_offset : typo_offset + len(pair.typo_text)] != pair.typo_text:
            raise ValueError("typo evaluation question coordinate differs from its paper prompt")
    clean_spans, typo_spans = _shifted_spans(
        pair,
        clean_offset=clean_offset,
        typo_offset=typo_offset,
    )
    for text, spans, words, side in (
        (clean_prompt, clean_spans, tuple(edit.clean_word for edit in pair.edits), "clean"),
        (typo_prompt, typo_spans, tuple(edit.typo_word for edit in pair.edits), "typo"),
    ):
        if len(set(spans)) != len(spans):
            raise ValueError(f"{side} evaluation prompt contains duplicate edit spans")
        if any(text[slice(*span)] != word for span, word in zip(spans, words, strict=True)):
            raise ValueError(f"{side} evaluation edit span differs after prompt rendering")
    return EvaluationPrompts(
        record_id=pair.record_id,
        answer=pair.answer,
        task_for_prompt=task,
        task_for_extractor=task,
        clean=PromptSide(clean_prompt, clean_spans),
        typo=PromptSide(typo_prompt, typo_spans),
    )


def classify_tokenization_counts(
    clean_counts: tuple[int, ...], typo_counts: tuple[int, ...]
) -> str:
    """Classify aligned edited words without collapsing mixed multi-edit cases."""

    if (
        not clean_counts
        or len(clean_counts) != len(typo_counts)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (*clean_counts, *typo_counts)
        )
    ):
        raise ValueError("tokenization edit cardinality and counts must align and be positive")
    differences = tuple(typo - clean for clean, typo in zip(clean_counts, typo_counts, strict=True))
    if all(difference == 0 for difference in differences):
        return "same-subtoken-count"
    if all(difference >= 0 for difference in differences):
        return "fragmentation-increased"
    if all(difference <= 0 for difference in differences):
        return "fragmentation-decreased"
    return "mixed-subtoken-change"


__all__ = [
    "EvaluationPrompts",
    "build_evaluation_prompts",
    "classify_tokenization_counts",
]

"""Submitted task-specific answer extraction for the Appendix E audit."""

from __future__ import annotations

from dataclasses import dataclass

from typo_cot.evaluation.extractor import create_extractor


@dataclass(frozen=True, slots=True)
class SubmittedAnswerExtraction:
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str


def extract_submitted_answer(
    text: str,
    *,
    benchmark: str,
    correct_answer: str,
) -> SubmittedAnswerExtraction:
    """Apply exactly the task extractor used by the submitted warning producer."""

    if not isinstance(text, str):
        raise TypeError("generated answer text must be a string")
    if not isinstance(correct_answer, str) or not correct_answer:
        raise ValueError("correct answer must be a non-empty string")
    extractor = create_extractor(benchmark)
    result = extractor.extract(text)
    value = result.extracted_answer
    return SubmittedAnswerExtraction(
        value=value,
        is_extracted=bool(value),
        is_correct=extractor.is_correct(value, correct_answer),
        method=result.extraction_method,
        primary_method=result.extraction_method,
    )


__all__ = ["SubmittedAnswerExtraction", "extract_submitted_answer"]

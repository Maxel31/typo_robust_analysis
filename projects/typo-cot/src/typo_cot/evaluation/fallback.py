"""Deterministic empty-result fallback used by the final-paper answer scans.

The benchmark extractor always runs first.  These rules are intentionally a
second pass for an empty primary result, never a competing parser that can
replace a non-empty primary answer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from typo_cot.evaluation.extractor import create_extractor

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_ANSWER_LINE = re.compile(r"\b(?:final\s+answer|answer)\b", re.IGNORECASE)
_UNIT = (
    r"(?:dollars?|cents?|liters?|litres?|hours?|hrs|minutes?|seconds?|days?|"
    r"weeks?|months?|years?|miles?|feet|foot|meters?|metres?|cm|centimeters?|"
    r"km|kg|grams?|pounds?|ounces?|eggs?|apples?|points?|people|students?|percent|%)"
)
_CHOICE_RANGE = {
    "mmlu": "A-Da-d",
    "mmlu_pro": "A-Ja-j",
    "arc": "A-Da-d",
    "commonsense_qa": "A-Ea-e",
}


@dataclass(frozen=True, slots=True)
class AnswerExtraction:
    """One primary-or-fallback answer with explicit provenance."""

    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str


def _clean_numeric(value: str) -> str:
    cleaned = value.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        number = float(cleaned)
    except (OverflowError, ValueError):
        return cleaned
    if not math.isfinite(number):
        return cleaned
    if number == int(number) and "e" not in cleaned.lower():
        return str(int(number))
    return repr(number)


def _last_nonempty_line(text: str) -> str:
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def _fallback_numeric(text: str) -> tuple[str, str]:
    for match in reversed(list(_BOXED.finditer(text))):
        candidate = match.group(1)
        candidate = re.sub(r"\\text\{[^{}]*\}|\\[,!;]|\\\$|\s", "", candidate)
        candidate = _clean_numeric(candidate)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
            return candidate, "N1_boxed"

    answer_lines = [line for line in text.splitlines() if _ANSWER_LINE.search(line)]
    if answer_lines:
        numbers = re.findall(r"\$?(" + _NUMBER + r")", answer_lines[-1])
        if numbers:
            return _clean_numeric(numbers[-1]), "N2_answer_line"

    for bold in reversed(re.findall(r"\*\*([^*\n]+)\*\*", text)):
        numbers = re.findall(r"\$?(" + _NUMBER + r")", bold)
        if numbers:
            return _clean_numeric(numbers[-1]), "N3_bold"

    final_line = _last_nonempty_line(text)
    if "=" in final_line:
        numbers = re.findall(r"\$?(" + _NUMBER + r")", final_line.rsplit("=", 1)[1])
        if numbers:
            return _clean_numeric(numbers[0]), "N4_equals_tail"
    tail = re.search(
        r"(-?\$?\d[\d,]*(?:\.\d+)?)\s*" + _UNIT + r"?\s*[.!)*]*\s*$",
        final_line,
    )
    if tail:
        return _clean_numeric(tail.group(1)), "N5_tail_number"
    return "", ""


def _fallback_choice(text: str, benchmark: str) -> tuple[str, str]:
    choice_range = _CHOICE_RANGE[benchmark]
    keyword_answers = re.findall(
        r"\b(?:answer|option|choice)\b(?:\s+is)?[:\s]*\(?([" + choice_range + r"])\)?\b",
        text,
        re.IGNORECASE,
    )
    if keyword_answers:
        return keyword_answers[-1].upper(), "C1_keyword"
    bold_answers = re.findall(
        r"\*\*\s*\(?([" + choice_range + r"])\)?\s*[.:]?\s*\*\*",
        text,
    )
    if bold_answers:
        return bold_answers[-1].upper(), "C2_bold"
    return "", ""


def fallback_answer(text: str, *, benchmark: str) -> tuple[str, str]:
    """Return ``(answer, rule)`` for an empty primary result."""

    if not text:
        return "", ""
    if benchmark in _CHOICE_RANGE:
        return _fallback_choice(text, benchmark)
    if benchmark == "gsm8k":
        return _fallback_numeric(text)
    raise ValueError(f"no final-paper fallback is registered for {benchmark!r}")


def canonical_answer(value: str, *, benchmark: str) -> str:
    """Canonicalize an already extracted answer for answer-change comparisons."""

    stripped = (value or "").strip()
    if not stripped:
        return ""
    if benchmark in _CHOICE_RANGE:
        match = re.fullmatch(r"\(?\s*([A-Ja-j])\s*\)?[.]?", stripped)
        return match.group(1).upper() if match else stripped.upper()
    if benchmark == "gsm8k":
        return _clean_numeric(stripped)
    raise ValueError(f"no final-paper answer canonicalizer is registered for {benchmark!r}")


def answers_equal(left: str, right: str, *, benchmark: str) -> bool:
    """Compare two non-empty extracted answers under the paper audit rules."""

    canonical_left = canonical_answer(left, benchmark=benchmark)
    canonical_right = canonical_answer(right, benchmark=benchmark)
    return bool(canonical_left and canonical_right and canonical_left == canonical_right)


def extract_with_fallback(
    text: str,
    *,
    benchmark: str,
    correct_answer: str,
) -> AnswerExtraction:
    """Apply the task extractor, then the fallback only when its value is empty."""

    extractor = create_extractor(benchmark)
    primary = extractor.extract(text)
    value = primary.extracted_answer
    if value:
        method = f"primary:{primary.extraction_method}"
    else:
        value, fallback_method = fallback_answer(text, benchmark=benchmark)
        method = f"fallback:{fallback_method}" if value else "unextractable"
    return AnswerExtraction(
        value=value,
        is_extracted=bool(value),
        is_correct=bool(value and extractor.is_correct(value, correct_answer)),
        method=method,
        primary_method=primary.extraction_method,
    )

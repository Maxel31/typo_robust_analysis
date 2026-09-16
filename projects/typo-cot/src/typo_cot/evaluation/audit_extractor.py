"""Gold-independent, exact-answer extraction for frozen rebuttal protocol v2.

This is deliberately independent of the historical, permissive extractors. Its
grammar is fixed in ``docs/rebuttal_v2/README.md`` section 18. Spans address the
original Python string (Unicode code points), with LF-delimited lines; CRLF and
other edge whitespace are stripped without changing the saved text.

Operational bounds are not model failures: input exceeding 1,000,000 code
points, valid numeric syntax containing more than 1,000 digits, or an absolute
exponent exceeding 1,000 raises ``ExtractionUnavailable``. Callers must record
parser-unavailable coverage, not an extraction/score row. No float conversion,
global integer-limit changes, model imports, or gold-dependent extraction occur.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Literal

PARSER_ID = "explicit_answer_v2"
PARSER_VERSION = "2.0.0"
MAX_TEXT_CODEPOINTS = 1_000_000
MAX_NUMERIC_DIGITS = 1_000
MAX_ABS_EXPONENT = 1_000

CanonicalValue = dict[str, str] | str
ExtractionStatus = Literal["extracted", "unextractable", "ambiguous"]

_CHOICE_TASKS = frozenset({"mmlu", "mmlu-pro", "mmlu_pro"})
_TASKS = _CHOICE_TASKS | {"gsm8k"}
_TERMINATIONS = frozenset({"eos", "length-cap", "unknown"})
_INTEGER = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_NUMBER = re.compile(
    rf"[+-]?(?:(?:{_INTEGER}(?:\.[0-9]+)?|\.[0-9]+)"
    rf"(?:[eE][+-]?[0-9]+)?|{_INTEGER}/[+-]?{_INTEGER})\Z"
)
_MARKER = re.compile(r"(?:(?ai:The answer is)\s+|(?ai:Answer:)\s*|(?ai:Final answer:)\s*)")
_BOX = re.compile(r"\\?boxed\{([^{}]*)\}\Z")
_LABEL = re.compile(r"(?:([A-Za-z])|\(([A-Za-z])\))\Z")
_FENCE = re.compile(r"(`{3,}|~{3,})(.*)\Z")
_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}
_OR = re.compile(r"\s+or\s+")


class ExtractionUnavailable(ValueError):
    """The selected parser could not execute; do not synthesize a failed answer."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class AuditExtraction:
    """Serializable extraction, with spans excluding markers and boxed wrappers."""

    task: str
    extraction_status: ExtractionStatus
    raw_value: str | None
    canonical_value: CanonicalValue | None
    candidate_spans: list[dict[str, object]]
    reasons: list[str]
    termination: str


@dataclass(frozen=True)
class Score:
    """A separate comparison under the fixed extraction-success definition."""

    score_status: Literal["scored", "not_scored"]
    gold_match: bool | None
    reasons: list[str]


def _validate_task_labels(task: str, valid_labels: Sequence[str] | None) -> frozenset[str]:
    if not isinstance(task, str) or task not in _TASKS:
        raise ValueError("task must be gsm8k, mmlu, or mmlu-pro (legacy alias: mmlu_pro)")
    if valid_labels is None:
        if task in _CHOICE_TASKS:
            raise ExtractionUnavailable("valid_labels_unavailable")
        return frozenset()
    if isinstance(valid_labels, (str, bytes)) or not isinstance(valid_labels, Sequence):
        raise ValueError("valid_labels must be a sequence of unique ASCII labels or null")
    if any(
        not isinstance(label, str) or not re.fullmatch(r"[A-Z]", label) for label in valid_labels
    ):
        raise ValueError("valid_labels must contain uppercase one-character ASCII labels")
    if len(set(valid_labels)) != len(valid_labels):
        raise ValueError("valid_labels must not contain duplicates")
    if task in _CHOICE_TASKS and not valid_labels:
        raise ExtractionUnavailable("valid_labels_unavailable")
    return frozenset(valid_labels)


def _canonical_number(value: str) -> dict[str, str] | None:
    if not _NUMBER.fullmatch(value):
        return None
    if sum("0" <= char <= "9" for char in value) > MAX_NUMERIC_DIGITS:
        raise ExtractionUnavailable("numeric_digit_limit_exceeded")
    numeric = value.replace(",", "")
    try:
        if "e" in numeric.lower():
            exponent = int(numeric.lower().partition("e")[2])
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise ExtractionUnavailable("numeric_exponent_limit_exceeded")
        # Fraction's string grammar does not accept a signed denominator.
        if "/" in numeric:
            numerator, denominator = numeric.split("/")
            result = Fraction(int(numerator), int(denominator))
        else:
            result = Fraction(numeric)
        return {"numerator": str(result.numerator), "denominator": str(result.denominator)}
    except ZeroDivisionError:
        return None
    except ExtractionUnavailable:
        raise
    except (ValueError, OverflowError, MemoryError) as error:
        # A stricter host integer-string limit is an execution limitation, not
        # evidence that this syntactically valid model answer was incorrect.
        raise ExtractionUnavailable("numeric_canonicalization_unavailable") from error


def _canonicalize(value: str, task: str, labels: frozenset[str]) -> CanonicalValue | None:
    if task == "gsm8k":
        return _canonical_number(value)
    match = _LABEL.fullmatch(value)
    if match is None:
        return None
    label = (match[1] or match[2]).upper()
    return label if label in labels else None


def canonicalize_answer(
    value: str, task: str, valid_labels: Sequence[str] | None
) -> CanonicalValue | None:
    """Normalize one whole raw value, without looking for markers or using gold.

    This helper also supports comparison with a separately executed public proxy.
    It does not strip syntax, punctuation, units, boxed wrappers, or whitespace.
    Invalid syntax yields null; operational inability raises unavailable.
    """
    labels = _validate_task_labels(task, valid_labels)
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    if len(value) > MAX_TEXT_CODEPOINTS:
        raise ExtractionUnavailable("text_codepoint_limit_exceeded")
    return _canonicalize(value, task, labels)


def _trim(value: str, start: int) -> tuple[str, int]:
    return value.strip(), start + len(value) - len(value.lstrip())


def _answer_body(line: str, start: int) -> tuple[str, int] | None:
    marker = _MARKER.match(line)
    if marker is not None:
        body, offset = _trim(line[marker.end() :], start + marker.end())
    elif line.startswith(("boxed{", "\\boxed{")):
        body, offset = line, start
    else:
        return None
    if body.endswith("."):
        body = body[:-1].rstrip()
    box = _BOX.fullmatch(body)
    if box is not None:
        return _trim(box[1], offset + box.start(1))
    # A malformed/nested box stays intact and cannot match the answer grammar.
    return body, offset


def _candidate(value: str, start: int, canonical: CanonicalValue) -> dict[str, object]:
    return {
        "start": start,
        "end": start + len(value),
        "raw_value": value,
        "canonical_value": canonical,
    }


def _value_key(canonical: object) -> tuple[str, ...]:
    if isinstance(canonical, dict):
        return ("number", canonical["numerator"], canonical["denominator"])
    return ("label", str(canonical))


def _has_unescaped(value: str, quote: str) -> bool:
    backslashes = 0
    for char in value:
        if char == quote and backslashes % 2 == 0:
            return True
        backslashes = backslashes + 1 if char == "\\" else 0
    return False


def _visible_lines(text: str) -> tuple[list[tuple[str, int, bool]], list[str], bool]:
    """Keep physical final-line identity while excluding quoted/code regions."""
    lines: list[tuple[str, int, bool]] = []
    reasons: list[str] = []
    fence: tuple[str, int] | None = None
    quote: str | None = None
    offset = 0
    for raw_line in text.split("\n"):
        line, start = _trim(raw_line, offset)
        offset += len(raw_line) + 1
        if not line:
            continue
        excluded = False
        fence_match = _FENCE.fullmatch(line)
        if fence is not None:
            excluded = True
            if (
                fence_match is not None
                and fence_match[1][0] == fence[0]
                and len(fence_match[1]) >= fence[1]
                and not fence_match[2].strip()
            ):
                fence = None
        elif quote is not None:
            excluded = True
            if _has_unescaped(line, quote):
                quote = None
        elif fence_match is not None:
            excluded = True
            fence = (fence_match[1][0], len(fence_match[1]))
            reasons.append("fenced_code_region")
        elif line.startswith(">"):
            excluded = True
            reasons.append("quoted_region")
        elif line[0] in _QUOTES:
            excluded = True
            reasons.append("quoted_region")
            end_quote = _QUOTES[line[0]]
            if not _has_unescaped(line[1:], end_quote):
                quote = end_quote
        lines.append((line, start, excluded))
    if fence is not None:
        reasons.append("unterminated_fence")
    if quote is not None:
        reasons.append("unterminated_quote")
    return lines, reasons, fence is not None or quote is not None


def extract_explicit_answer(
    text: str,
    task: str,
    valid_labels: Sequence[str] | None,
    termination: str = "unknown",
) -> AuditExtraction:
    """Apply the frozen full-line grammar, independently of gold and stopping.

    Complete, unquoted explicit answers on earlier lines only detect conflicts;
    a unique answer is extracted only from the final nonempty physical line.
    `` or `` alternatives detect ambiguity only, never select a candidate.
    Missing raw text must be handled by the caller without executing a parser.
    Task IDs are ``gsm8k``, ``mmlu``, and ``mmlu-pro``; legacy ``mmlu_pro`` is
    accepted explicitly without changing the task ID stored in the result.
    """
    labels = _validate_task_labels(task, valid_labels)
    if not isinstance(text, str):
        raise ValueError("text must be present and a string")
    if not isinstance(termination, str) or termination not in _TERMINATIONS:
        raise ValueError("termination must be eos, length-cap, or unknown")
    if len(text) > MAX_TEXT_CODEPOINTS:
        raise ExtractionUnavailable("text_codepoint_limit_exceeded")
    lines, reasons, uncertain_boundary = _visible_lines(text)
    candidates: list[dict[str, object]] = []
    final_candidate: dict[str, object] | None = None
    final_alternatives: list[dict[str, object]] = []
    for index, (line, start, excluded) in enumerate(lines):
        if excluded:
            continue
        body_info = _answer_body(line, start)
        if body_info is None:
            continue
        body, body_start = body_info
        canonical = _canonicalize(body, task, labels)
        if canonical is not None:
            candidate = _candidate(body, body_start, canonical)
            candidates.append(candidate)
            if index == len(lines) - 1:
                final_candidate = candidate
            continue
        reasons.append("unsupported_answer_syntax")
        if any(char in body for char in ('"', "'", "“", "”", "‘", "’")):
            reasons.append("quoted_answer_syntax")
        if re.search(r"\b(?:not|incorrect|never)\b", body, re.IGNORECASE | re.ASCII):
            reasons.append("negated_answer_syntax")
        if index != len(lines) - 1:
            continue
        splits = list(_OR.finditer(body))
        if splits:
            boundaries = [(0, splits[0].start())]
            boundaries.extend(
                (previous.end(), following.start())
                for previous, following in zip(splits, splits[1:])
            )
            boundaries.append((splits[-1].end(), len(body)))
            for first, last in boundaries:
                raw = body[first:last]
                value = _canonicalize(raw, task, labels)
                if value is None:
                    final_alternatives = []
                    break
                final_alternatives.append(_candidate(raw, body_start + first, value))
    distinct = {_value_key(item["canonical_value"]) for item in candidates}
    alternative_values = {_value_key(item["canonical_value"]) for item in final_alternatives}
    status: ExtractionStatus = "unextractable"
    raw_value: str | None = None
    canonical_value: CanonicalValue | None = None
    if uncertain_boundary:
        reasons.append("quoted_region_boundary_uncertain")
    if len(distinct) > 1:
        status = "ambiguous"
        reasons.append("conflicting_explicit_answers")
    elif uncertain_boundary:
        pass
    elif len(alternative_values) > 1:
        status = "ambiguous"
        reasons.append("multiple_distinct_final_candidates")
    elif final_candidate is not None:
        status = "extracted"
        raw_value = str(final_candidate["raw_value"])
        canonical_value = final_candidate["canonical_value"]  # type: ignore[assignment]
    elif not lines:
        reasons.append("no_nonempty_line")
    else:
        reasons.append("final_line_not_explicit_answer")
    return AuditExtraction(
        task=task,
        extraction_status=status,
        raw_value=raw_value,
        canonical_value=canonical_value,
        candidate_spans=candidates + final_alternatives,
        reasons=list(dict.fromkeys(reasons)),
        termination=termination,
    )


def _canonical_gold(gold: object, task: str) -> CanonicalValue | None:
    if task in _CHOICE_TASKS:
        return gold if isinstance(gold, str) and re.fullmatch(r"[A-Z]", gold) else None
    if not isinstance(gold, dict) or set(gold) != {"numerator", "denominator"}:
        return None
    numerator, denominator = gold["numerator"], gold["denominator"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        return None
    if max(len(numerator), len(denominator)) > MAX_NUMERIC_DIGITS + MAX_ABS_EXPONENT + 1:
        return None
    if not re.fullmatch(r"(?:0|-?[1-9][0-9]*)", numerator):
        return None
    if not re.fullmatch(r"[1-9][0-9]*", denominator):
        return None
    try:
        reduced = gcd(int(numerator), int(denominator)) == 1
    except (ValueError, OverflowError, MemoryError):
        return None
    if not reduced:
        return None
    return {"numerator": numerator, "denominator": denominator}


def score_answer(extraction: AuditExtraction, canonical_gold: object) -> Score:
    """Compare canonical values after extraction, without reparsing model text.

    Missing or noncanonical gold is not scored. With valid gold, ambiguous and
    unextractable outputs fail the fixed extraction-success metric, not a claim
    of semantic incorrectness. Item-specific gold label validity is checked by
    the input adapter, not inferred from a task-wide range here.
    """
    if not isinstance(extraction, AuditExtraction):
        raise ValueError("extraction must be an AuditExtraction")
    if extraction.task not in _TASKS:
        raise ValueError("unsupported extraction task")
    if extraction.extraction_status not in {"extracted", "unextractable", "ambiguous"}:
        raise ValueError("unsupported extraction status")
    gold = _canonical_gold(canonical_gold, extraction.task)
    if gold is None:
        return Score("not_scored", None, ["canonical_gold_unavailable"])
    return Score(
        "scored",
        extraction.extraction_status == "extracted" and extraction.canonical_value == gold,
        [],
    )

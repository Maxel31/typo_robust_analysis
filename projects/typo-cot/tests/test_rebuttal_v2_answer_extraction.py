"""Independent, outcome-free fixtures for the frozen explicit-answer grammar."""

from dataclasses import asdict
import inspect
import json

import pytest

from typo_cot.evaluation import audit_extractor
from typo_cot.evaluation.audit_extractor import (
    MAX_ABS_EXPONENT,
    MAX_NUMERIC_DIGITS,
    MAX_TEXT_CODEPOINTS,
    PARSER_ID,
    PARSER_VERSION,
    AuditExtraction,
    ExtractionUnavailable,
    canonicalize_answer,
    extract_explicit_answer,
    score_answer,
)
from typo_cot.evaluation.extractor import (
    GSM8KAnswerExtractor,
    MMLUAnswerExtractor,
    MMLUProAnswerExtractor,
)


def number(numerator: str, denominator: str = "1") -> dict[str, str]:
    return {"numerator": numerator, "denominator": denominator}


def extract(text: str, task: str = "gsm8k", termination: str = "unknown") -> AuditExtraction:
    labels = list("ABCD") if task == "mmlu" else list("ABCDEFGHIJ") if task == "mmlu_pro" else None
    return extract_explicit_answer(text, task, labels, termination)


@pytest.mark.parametrize(
    ("text", "status", "canonical"),
    [
        ("We compute 2 + 2 = 4\nStill calculating", "unextractable", None),
        ("Answer: -1,250.5", "extracted", number("-2501", "2")),
        ("Answer: 4/5", "extracted", number("4", "5")),
        ("Answer: 1e3", "extracted", number("1000")),
        ("Answer: 4/5\nFinal answer: 0.8.", "extracted", number("4", "5")),
        ("Answer: 4\nFinal answer: 5", "ambiguous", None),
        ("Answer: 4 or 5", "ambiguous", None),
        ("Answer: 4 followed by more reasoning", "unextractable", None),
        ('"Answer: 4"', "unextractable", None),
        ("> Answer: 4", "unextractable", None),
        ("Answer: 4 dollars", "unextractable", None),
        ("Answer: 4/0", "unextractable", None),
        (r"\boxed{-4/5}", "extracted", number("-4", "5")),
    ],
)
def test_protocol_numeric_acceptance_examples(text, status, canonical):
    result = extract(text)
    assert result.extraction_status == status
    assert result.canonical_value == canonical


@pytest.mark.parametrize(
    ("text", "task", "status", "canonical"),
    [
        ("The answer is definitely B.", "mmlu", "unextractable", None),
        ("The answer is incorrect; continue reasoning.", "mmlu_pro", "unextractable", None),
        ("Answer: (J).", "mmlu_pro", "extracted", "J"),
        ("Answer: J", "mmlu", "unextractable", None),
    ],
)
def test_protocol_choice_acceptance_examples(text, task, status, canonical):
    result = extract(text, task)
    assert result.extraction_status == status
    assert result.canonical_value == canonical


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("0", number("0")),
        ("-0", number("0")),
        ("+0004", number("4")),
        ("1,234,567", number("1234567")),
        ("+1,250.50", number("2501", "2")),
        (".5", number("1", "2")),
        ("-.125", number("-1", "8")),
        ("+1.25E-3", number("1", "800")),
        ("1,000e+2", number("100000")),
        ("-4/-5", number("4", "5")),
        ("+4/-5", number("-4", "5")),
        ("4/+5", number("4", "5")),
        ("1,000/2,000", number("1", "2")),
        ("0/-7", number("0")),
        ("9007199254740993", number("9007199254740993")),
        ("0.10000000000000001", number("10000000000000001", "100000000000000000")),
    ],
)
def test_complete_numeric_grammar_is_exact(spelling, canonical):
    result = extract(f"Answer: {spelling}.")
    assert result.extraction_status == "extracted"
    assert result.canonical_value == canonical
    assert result.raw_value == spelling


@pytest.mark.parametrize(
    "spelling",
    [
        "NaN",
        "Infinity",
        "inf",
        "-Infinity",
        "12%",
        "$12",
        "12 dollars",
        "€12",
        "1,25",
        "1234,567",
        "1,234,56",
        "1,,234",
        "1,234.",
        "1_234",
        "١٢",
        "１２",
        "−12",
        "4 /5",
        "4/ 5",
        "4 / 5",
        "4.0/5",
        "4/5.0",
        "4e1/5",
        "4/0",
        "0/0",
        "4/000",
        "4/-0",
        "4//5",
        "1e",
        "1e+",
        "1e-",
        "1.2.3",
        ".",
        "..5",
        "4..",
        "4 followed by more reasoning",
        "4;",
        "4!",
        "4?",
        "4,",
        "4: justification",
        "(4)",
        r"\frac{4}{5}",
        "4 or five",
        "4 and 5",
        "4 or 5 dollars",
        "4. or 5",
    ],
)
def test_numeric_partial_matches_are_never_adopted(spelling):
    # A terminal decimal dot may also be the one permitted sentence period.
    # Add the ordinary sentence period so unsupported dot forms remain visible.
    result = extract(f"Answer: {spelling}.")
    assert result.extraction_status == "unextractable"
    assert result.canonical_value is None
    assert result.raw_value is None


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 4",
        "THE ANSWER IS 4.",
        "tHe AnSwEr Is\t4",
        "The answer is\u00a04",
        "Answer:4",
        "ANSWER:\t4 . ",
        "Final answer:4.",
        "fInAl AnSwEr: 4",
        "boxed{4}",
        r"\boxed{4}.",
        "Answer: boxed{4}",
        r"Final answer: \boxed{  4  }.",
        " \tAnswer: 4\r\n \t\n",
    ],
)
def test_supported_markers_and_only_terminal_period(text):
    assert extract(text).canonical_value == number("4")


@pytest.mark.parametrize(
    "text",
    [
        "4",
        "4.",
        "= 4",
        "The final answer is 4",
        "The answer is: 4",
        "The answer is4",
        "Answer 4",
        "Final Answer is 4",
        "**Answer: 4**",
        "Some reasoning. Answer: 4",
        "Answer: 4\nTherefore done",
        "BOXED{4}",
        "boxed {4}",
        "boxed{4",
        "boxed4}",
        "boxed{{4}}",
        r"\boxed{\frac{4}{5}}",
        "$\\boxed{4}$",
        "boxed{4.}",
        "boxed{4}}",
        "boxed{4}{5}",
        "Answer: boxed{4} more reasoning",
        "Answer: 4..",
        "The anſwer is 4",
        "The answer ıS 4",
        "Ａnswer: 4",
    ],
)
def test_unsupported_markers_wrappers_and_nonfinal_answers(text):
    assert extract(text).extraction_status == "unextractable"


@pytest.mark.parametrize("raw", ["j", "J", "(j)", "(J)"])
def test_choice_normalization_respects_item_labels(raw):
    result = extract_explicit_answer(f"Final answer: {raw}.", "mmlu_pro", ["B", "J"])
    assert result.canonical_value == "J"
    assert result.raw_value == raw
    assert (
        extract_explicit_answer(f"Answer: {raw}", "mmlu_pro", ["A", "B"]).extraction_status
        == "unextractable"
    )


@pytest.mark.parametrize(
    "raw", ["definitely B", "incorrect", "Bingo", "A-B", "AB", "( B )", "B)", "(B", "Ｂ", "K", "ı"]
)
def test_choice_never_matches_prefix_unicode_fold_or_broken_parentheses(raw):
    assert extract(f"Answer: {raw}", "mmlu_pro").extraction_status == "unextractable"


def test_tasks_do_not_accept_each_others_value_types():
    assert extract("Answer: A").extraction_status == "unextractable"
    assert extract("Answer: 4", "mmlu").extraction_status == "unextractable"
    assert extract("B", "mmlu").extraction_status == "unextractable"


@pytest.mark.parametrize(
    "text",
    ["Answer: 4\nAnswer: 5\nStill thinking", "Answer: 4\nFinal answer: 5.", "Answer: 4\nboxed{5}"],
)
def test_conflicts_have_priority_even_without_a_final_answer(text):
    result = extract(text)
    assert result.extraction_status == "ambiguous"
    assert len(result.candidate_spans) == 2
    assert "conflicting_explicit_answers" in result.reasons


def test_equivalent_values_are_not_conflicts_and_final_raw_value_is_preserved():
    result = extract("Answer: 4/5\nThe answer is 8e-1\nFinal answer: .800.")
    assert result.extraction_status == "extracted"
    assert result.canonical_value == number("4", "5")
    assert result.raw_value == ".800"
    assert len(result.candidate_spans) == 3
    assert (
        extract("Answer: 4/5\nFinal answer: .8\nStill thinking").extraction_status
        == "unextractable"
    )


@pytest.mark.parametrize("text", ["Answer: 4 or 5", "Final answer: 4 or 5 or 6.", "boxed{4 or 5}"])
def test_or_alternatives_only_detect_ambiguity(text):
    result = extract(text)
    assert result.extraction_status == "ambiguous"
    assert result.canonical_value is None
    assert result.raw_value is None
    assert "multiple_distinct_final_candidates" in result.reasons
    for candidate in result.candidate_spans:
        assert text[candidate["start"] : candidate["end"]] == candidate["raw_value"]


def test_same_value_or_and_nonfinal_or_never_select_an_answer():
    assert extract("Answer: 4/5 or 0.8").extraction_status == "unextractable"
    assert extract("Answer: 4 or 5\nFinal answer: 4").canonical_value == number("4")
    assert extract("Answer: B or C", "mmlu").extraction_status == "ambiguous"
    assert extract("Answer: B or Z", "mmlu").extraction_status == "unextractable"


@pytest.mark.parametrize(
    "quoted",
    [
        '"Answer: 99"',
        "'Answer: 99'",
        "“Answer: 99”",
        "‘Answer: 99’",
        "«Answer: 99»",
        "> Answer: 99",
        ">>> Answer: 99",
        "```\nAnswer: 99\n```",
        "~~~text\nAnswer: 99\n~~~",
        "````python\nAnswer: 99\n```\nAnswer: 98\n````",
        '"quoted text\nAnswer: 99\n"',
        '"An escaped quote \\" is inside this quotation\nAnswer: 99\n"',
    ],
)
def test_quoted_or_fenced_candidates_are_excluded_from_conflict_detection(quoted):
    result = extract(f"{quoted}\nAnswer: 4")
    assert result.extraction_status == "extracted"
    assert result.canonical_value == number("4")
    assert len(result.candidate_spans) == 1
    assert any(reason in result.reasons for reason in ["quoted_region", "fenced_code_region"])
    assert extract(f"Answer: 4\n{quoted}").extraction_status == "unextractable"


@pytest.mark.parametrize(
    "text",
    [
        '"Answer: 4',
        "'quoted\nAnswer: 4",
        "“quoted\nAnswer: 4",
        "```\nAnswer: 4",
        "~~~\nAnswer: 4\n```",
    ],
)
def test_uncertain_quote_or_fence_boundary_is_unextractable(text):
    result = extract(text)
    assert result.extraction_status == "unextractable"
    assert result.candidate_spans == []
    assert "quoted_region_boundary_uncertain" in result.reasons


@pytest.mark.parametrize("tail", ['"unfinished', "```\nAnswer: 6", "~~~\nAnswer: 6"])
def test_known_unquoted_conflicts_precede_later_uncertain_boundary(tail):
    result = extract(f"Answer: 4\nFinal answer: 5\n{tail}")
    assert result.extraction_status == "ambiguous"
    assert result.canonical_value is None
    assert len(result.candidate_spans) == 2
    assert "conflicting_explicit_answers" in result.reasons
    assert "quoted_region_boundary_uncertain" in result.reasons


def test_doubts_are_retained_without_fabricating_semantic_judgments():
    result = extract('Answer: not 4\nAnswer: "4"\nFinal answer: 5')
    assert result.canonical_value == number("5")
    assert "quoted_answer_syntax" in result.reasons
    assert "negated_answer_syntax" in result.reasons
    assert "unsupported_answer_syntax" in result.reasons


def test_codepoint_spans_preserve_unicode_crlf_whitespace_and_wrappers():
    text = "😀 café e\u0301\r\n  Answer: 4/5\r\n\tFinal answer: \\boxed{ .800 }. \r\n"
    result = extract(text)
    assert result.canonical_value == number("4", "5")
    for candidate, raw in zip(result.candidate_spans, ["4/5", ".800"], strict=True):
        assert candidate["start"] == text.index(raw)
        assert candidate["end"] == text.index(raw) + len(raw)
        assert text[candidate["start"] : candidate["end"]] == raw
    assert result.candidate_spans[0]["start"] != len(text[: text.index("4/5")].encode())


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_do_not_silently_rewrite_physical_lines(separator):
    result = extract(f"Answer: 4{separator}Final answer: 5")
    assert result.extraction_status == "unextractable"
    assert result.candidate_spans == []


def test_serialization_is_stable_and_has_no_gold_input():
    result = extract("Answer: 4/5")
    assert json.loads(json.dumps(asdict(result))) == asdict(result)
    assert PARSER_ID == "explicit_answer_v2"
    assert PARSER_VERSION == "2.0.0"
    assert list(inspect.signature(extract_explicit_answer).parameters) == [
        "text",
        "task",
        "valid_labels",
        "termination",
    ]
    before = asdict(result)
    assert score_answer(result, number("4", "5")).gold_match is True
    assert score_answer(result, number("5", "4")).gold_match is False
    assert score_answer(result, None).score_status == "not_scored"
    assert asdict(result) == before


@pytest.mark.parametrize("termination", ["eos", "length-cap", "unknown"])
def test_stopping_does_not_change_extraction(termination):
    result = extract("Answer: 4", termination=termination)
    assert result.extraction_status == "extracted"
    assert result.canonical_value == number("4")
    assert result.termination == termination
    assert score_answer(result, number("4")).gold_match is True
    assert extract("We compute 4", termination=termination).extraction_status == "unextractable"


def test_old_numeric_partial_and_rounding_counterexamples_are_independent():
    old = GSM8KAnswerExtractor()
    assert old.extract("Answer: 4/5").extracted_answer == "4"
    assert extract("Answer: 4/5").canonical_value == number("4", "5")
    assert old.extract("Answer: 1e3").extracted_answer == "1"
    assert extract("Answer: 1e3").canonical_value == number("1000")
    assert old.extract("Answer: 4 dollars").extracted_answer == "4"
    assert extract("Answer: 4 dollars").extraction_status == "unextractable"
    assert old.is_correct("9007199254740993", "9007199254740992")
    assert (
        score_answer(extract("Answer: 9007199254740993"), number("9007199254740992")).gold_match
        is False
    )


def test_old_choice_word_prefix_counterexamples_are_independent():
    assert MMLUAnswerExtractor().extract("The answer is definitely B.").extracted_answer == "D"
    assert extract("The answer is definitely B.", "mmlu").extraction_status == "unextractable"
    assert (
        MMLUProAnswerExtractor()
        .extract("The answer is incorrect; continue reasoning.")
        .extracted_answer
        == "I"
    )
    assert (
        extract("The answer is incorrect; continue reasoning.", "mmlu_pro").extraction_status
        == "unextractable"
    )


@pytest.mark.parametrize("text", ["Answer: 5", "Answer: 4 or 5", "Still thinking", ""])
def test_valid_gold_scores_fixed_extraction_success_not_semantics(text):
    score = score_answer(extract(text), number("4"))
    assert score.score_status == "scored"
    assert score.gold_match is False


@pytest.mark.parametrize(
    "gold",
    [
        None,
        "4",
        4,
        4.0,
        {},
        number("04"),
        number("+4"),
        number("-0"),
        number("4", "0"),
        number("4", "-1"),
        number("8", "2"),
        {"numerator": 4, "denominator": "1"},
        {"numerator": "4", "denominator": "1", "extra": True},
    ],
)
def test_missing_or_noncanonical_gold_never_becomes_wrong_model_answer(gold):
    score = score_answer(extract("Answer: 4"), gold)
    assert score.score_status == "not_scored"
    assert score.gold_match is None
    assert score.reasons == ["canonical_gold_unavailable"]


def test_choice_gold_is_canonical_uppercase_and_separate():
    result = extract("Answer: b", "mmlu")
    assert score_answer(result, "B").gold_match is True
    assert score_answer(result, "C").gold_match is False
    for gold in [None, "b", "(B)", " B ", "Ｂ", "BB", {"label": "B"}]:
        assert score_answer(result, gold).score_status == "not_scored"


def test_public_proxy_helper_only_normalizes_whole_answer_values():
    assert canonicalize_answer("1,250.5", "gsm8k", None) == number("2501", "2")
    assert canonicalize_answer("(b)", "mmlu", ["A", "B"]) == "B"
    for value in ["Answer: 4", "boxed{4}", "4.", "4 dollars", " 4 "]:
        assert canonicalize_answer(value, "gsm8k", None) is None


@pytest.mark.parametrize("labels", [None, []])
def test_missing_item_labels_mean_parser_unavailable_not_failed_extraction(labels):
    with pytest.raises(ExtractionUnavailable, match="valid_labels_unavailable"):
        extract_explicit_answer("Answer: B", "mmlu_pro", labels)


@pytest.mark.parametrize("labels", ["ABCD", {"A", "B"}, ["A", "A"], ["a"], ["AB"], [1], ["Ｂ"]])
def test_invalid_labels_are_contract_errors(labels):
    with pytest.raises(ValueError):
        extract_explicit_answer("Answer: A", "mmlu", labels)


@pytest.mark.parametrize("task", ["MMLU", "mmlupro", "math", None, 1, []])
def test_unknown_tasks_are_contract_errors(task):
    with pytest.raises(ValueError):
        extract_explicit_answer("Answer: 4", task, None)


@pytest.mark.parametrize("termination", [None, "cap", "eos-at-512", 512, [], True])
def test_unsupported_termination_cannot_be_inferred(termination):
    with pytest.raises(ValueError):
        extract_explicit_answer("Answer: 4", "gsm8k", None, termination)


def test_absent_raw_text_is_not_an_unextractable_output():
    with pytest.raises(ValueError, match="present"):
        extract_explicit_answer(None, "gsm8k", None)
    assert extract("").extraction_status == "unextractable"


def test_resource_limits_are_unavailable_not_incorrect_results():
    for text, reason in [
        ("Answer: " + "9" * (MAX_NUMERIC_DIGITS + 1), "numeric_digit_limit_exceeded"),
        (f"Answer: 1e{MAX_ABS_EXPONENT + 1}", "numeric_exponent_limit_exceeded"),
        (f"Answer: 1e-{MAX_ABS_EXPONENT + 1}", "numeric_exponent_limit_exceeded"),
        ("x" * (MAX_TEXT_CODEPOINTS + 1), "text_codepoint_limit_exceeded"),
    ]:
        with pytest.raises(ExtractionUnavailable) as error:
            extract(text)
        assert error.value.reason_code == reason
    assert extract("Answer: " + "9" * MAX_NUMERIC_DIGITS).extraction_status == "extracted"
    assert extract(f"Answer: 1e{MAX_ABS_EXPONENT}").extraction_status == "extracted"
    assert extract(f"Answer: 1e-{MAX_ABS_EXPONENT}").extraction_status == "extracted"


@pytest.mark.parametrize("error_type", [ValueError, OverflowError, MemoryError])
def test_host_numeric_conversion_failure_is_parser_unavailable(monkeypatch, error_type):
    def unavailable_fraction(_value):
        raise error_type("host conversion unavailable")

    monkeypatch.setattr(audit_extractor, "Fraction", unavailable_fraction)
    with pytest.raises(ExtractionUnavailable, match="numeric_canonicalization_unavailable"):
        extract("Answer: 123")


@pytest.mark.parametrize("task", ["mmlu-pro", "mmlu_pro"])
def test_canonical_manifest_task_and_explicit_legacy_alias(task):
    result = extract_explicit_answer("Answer: (j).", task, list("ABCDEFGHIJ"))
    assert result.task == task
    assert result.extraction_status == "extracted"
    assert result.canonical_value == "J"
    assert score_answer(result, "J").gold_match is True
    assert canonicalize_answer("j", task, ["J"]) == "J"

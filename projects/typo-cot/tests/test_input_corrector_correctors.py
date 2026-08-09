"""Contracts for the Appendix E input-corrector implementations.

The neural backends are injected so these tests remain CPU-only.  The callback
arguments intentionally expose the submitted generation contract without
requiring ``torch`` or ``transformers`` during unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from typo_cot.experiments.input_corrector_audit.correctors import (
    CORRECTOR_IDS,
    QWEN_RETRY_REMINDER,
    QWEN_SYSTEM_PROMPT,
    QWEN_USER_TEMPLATE,
    PySpellCheckerCorrector,
    QwenCorrector,
    T5LargeSpellCorrector,
    apply_case,
    create_corrector,
)


class _FakeSpellChecker:
    """Small pyspellchecker-compatible dictionary for deterministic tests."""

    def __init__(
        self,
        *,
        frequencies: Mapping[str, int],
        candidates: Mapping[str, set[str]],
    ) -> None:
        self._frequencies = dict(frequencies)
        self._candidates = {word: set(values) for word, values in candidates.items()}
        self.candidate_calls: list[str] = []

    def __contains__(self, word: str) -> bool:
        return word in self._frequencies

    def __getitem__(self, word: str) -> int:
        return self._frequencies[word]

    def candidates(self, word: str) -> set[str] | None:
        self.candidate_calls.append(word)
        values = self._candidates.get(word)
        return None if values is None else set(values)

    def correction(self, word: str) -> str:
        raise AssertionError("the nondeterministic upstream correction() must not be used")


def _spellchecker() -> _FakeSpellChecker:
    return _FakeSpellChecker(
        frequencies={
            "can't": 20,
            "correct": 20,
            "matters": 20,
            "spelling": 20,
            "zza": 5,
            "zzb": 5,
            "zzc": 3,
        },
        candidates={
            "speling": {"spelling"},
            "zzq": {"zzc", "zzb", "zza"},
        },
    )


def test_public_corrector_ids_and_submitted_model_identifiers() -> None:
    assert CORRECTOR_IDS == (
        "pyspellchecker",
        "t5-large-spell",
        "qwen2.5-7b-instruct",
    )

    pyspell = PySpellCheckerCorrector(spellchecker=_spellchecker())
    t5 = T5LargeSpellCorrector(generate_fn=lambda prompt, **kwargs: prompt)
    qwen = QwenCorrector(generate_fn=lambda messages, **kwargs: "<corrected>x</corrected>")

    assert pyspell.corrector_id == "pyspellchecker"
    assert pyspell.dependency_requirement == "pyspellchecker==0.9.0"
    assert t5.corrector_id == "t5-large-spell"
    assert t5.model_id == "ai-forever/T5-large-spell"
    assert qwen.corrector_id == "qwen2.5-7b-instruct"
    assert qwen.model_id == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize(
    ("template", "corrected", "expected"),
    [
        ("SPELING", "spelling", "SPELLING"),
        ("Speling", "spelling", "Spelling"),
        ("speling", "spelling", "spelling"),
    ],
)
def test_apply_case_matches_the_submitted_pyspell_policy(
    template: str,
    corrected: str,
    expected: str,
) -> None:
    assert apply_case(template, corrected) == expected


def test_pyspell_uses_frequency_then_lexical_order_for_ties() -> None:
    corrector = PySpellCheckerCorrector(spellchecker=_spellchecker())

    assert corrector.correct("ZZQ zzq Zzq") == "ZZA zza Zza"


def test_pyspell_preserves_non_word_bytes_and_uses_the_submitted_word_regex() -> None:
    spellchecker = _spellchecker()
    corrector = PySpellCheckerCorrector(spellchecker=spellchecker)
    source = "Speling,\n  can't; (A) 12 + $5.00"

    assert corrector.correct(source) == "Spelling,\n  can't; (A) 12 + $5.00"
    # A one-character option label is excluded before dictionary candidate lookup.
    assert "a" not in spellchecker.candidate_calls


def test_pyspell_leaves_unknown_words_without_candidates_unchanged() -> None:
    corrector = PySpellCheckerCorrector(spellchecker=_spellchecker())

    assert corrector.correct("xyzzynocandidate") == "xyzzynocandidate"


def test_t5_prefixes_each_nonblank_line_and_preserves_blank_line_structure() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def generate(prompt: str, **generation: object) -> str:
        calls.append((prompt, dict(generation)))
        return f"  fixed-{len(calls)}  "

    corrector = T5LargeSpellCorrector(generate_fn=generate)
    source = "the qick fox\n \t \n(A) qick (B) slow\n"

    assert corrector.correct(source) == "fixed-1\n \t \nfixed-2\n"
    assert [prompt for prompt, _ in calls] == [
        "grammar: the qick fox",
        "grammar: (A) qick (B) slow",
    ]
    assert all(generation == {"max_new_tokens": 256, "do_sample": False} for _, generation in calls)
    assert corrector.prefix == "grammar: "
    assert corrector.max_input_length == 512


def test_qwen_builds_the_exact_conservative_chat_prompt_and_uses_greedy_generation() -> None:
    calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def generate(
        messages: Sequence[Mapping[str, str]],
        **generation: object,
    ) -> str:
        calls.append(([dict(message) for message in messages], dict(generation)))
        return "<corrected>fixed text</corrected>"

    corrector = QwenCorrector(generate_fn=generate)

    assert corrector.correct("some typo txt") == "fixed text"
    assert calls == [
        (
            [
                {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": QWEN_USER_TEMPLATE.format(text="some typo txt"),
                },
            ],
            {"max_new_tokens": 1024, "do_sample": False},
        )
    ]
    assert QWEN_SYSTEM_PROMPT == "You are a careful and conservative proofreader."
    assert "Fix ONLY the typos." in QWEN_USER_TEMPLATE
    assert "Do not rephrase" in QWEN_USER_TEMPLATE
    assert "<text>\n{text}\n</text>" in QWEN_USER_TEMPLATE


def test_qwen_tag_parser_preserves_spaces_and_strips_only_outer_newlines() -> None:
    corrector = QwenCorrector(
        generate_fn=lambda messages, **kwargs: (
            "prefix ignored<corrected>\n  line one  \n(A) x\n</corrected>suffix ignored"
        )
    )

    assert corrector.correct("line onn\n(A) x") == "  line one  \n(A) x"


def test_qwen_retries_once_with_a_changed_prompt_after_tag_parse_failure() -> None:
    responses = iter(["I fixed it for you", "<corrected>good text</corrected>"])
    calls: list[list[dict[str, str]]] = []

    def generate(
        messages: Sequence[Mapping[str, str]],
        **generation: object,
    ) -> str:
        assert generation == {"max_new_tokens": 1024, "do_sample": False}
        calls.append([dict(message) for message in messages])
        return next(responses)

    corrector = QwenCorrector(generate_fn=generate)
    corrected, metadata = corrector.correct_with_meta("good textt")

    assert corrected == "good text"
    assert metadata["parse_failed"] is False
    assert metadata["n_calls"] == 2
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert calls[1][1]["content"] == calls[0][1]["content"] + QWEN_RETRY_REMINDER


def test_qwen_fails_closed_to_the_original_after_two_parse_failures() -> None:
    responses = iter(["missing tags", "still missing tags"])
    calls = 0

    def generate(messages: Sequence[Mapping[str, str]], **generation: object) -> str:
        nonlocal calls
        calls += 1
        return next(responses)

    corrector = QwenCorrector(generate_fn=generate)
    source = "the qick fox"
    corrected, metadata = corrector.correct_with_meta(source)

    assert corrected == source
    assert calls == 2
    assert metadata == {
        "parse_failed": True,
        "n_calls": 2,
        "raw_response": "still missing tags",
    }


def test_create_corrector_dispatches_only_the_public_ids() -> None:
    assert isinstance(
        create_corrector("pyspellchecker", spellchecker=_spellchecker()),
        PySpellCheckerCorrector,
    )
    assert isinstance(
        create_corrector("t5-large-spell", generate_fn=lambda prompt, **kwargs: prompt),
        T5LargeSpellCorrector,
    )
    assert isinstance(
        create_corrector(
            "qwen2.5-7b-instruct",
            generate_fn=lambda messages, **kwargs: "<corrected>x</corrected>",
        ),
        QwenCorrector,
    )

    with pytest.raises(ValueError, match="unknown-corrector"):
        create_corrector("unknown-corrector")

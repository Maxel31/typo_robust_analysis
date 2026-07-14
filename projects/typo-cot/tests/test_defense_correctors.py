"""実験7: 校正器ラダー (defense/correctors.py) のユニットテスト.

GPU 不要。pyspellchecker は軽量辞書同梱 (==0.9.0 固定) なので実物を使い、
ニューラル校正器 / LLM 校正器は generate_fn 注入でモックする。
"""

import pytest

from typo_cot.defense.correctors import (
    LLMCorrector,
    PySpellCorrector,
    Seq2SeqCorrector,
    apply_case,
    create_corrector,
)


class TestApplyCase:
    def test_all_upper(self):
        assert apply_case("SPELING", "spelling") == "SPELLING"

    def test_title_case(self):
        assert apply_case("Speling", "spelling") == "Spelling"

    def test_lower_case(self):
        assert apply_case("speling", "spelling") == "spelling"


class TestPySpellCorrector:
    @pytest.fixture(scope="class")
    def corrector(self):
        return PySpellCorrector()

    def test_name(self, corrector):
        assert corrector.name == "pyspell"

    def test_corrects_misspelled_word(self, corrector):
        assert corrector.correct("correct speling matters") == "correct spelling matters"

    def test_leaves_dictionary_words_unchanged(self, corrector):
        text = "The quick brown fox jumps"
        assert corrector.correct(text) == text

    def test_preserves_case_of_corrected_word(self, corrector):
        assert corrector.correct("Speling is hard") == "Spelling is hard"

    def test_skips_single_char_words(self, corrector):
        # 選択肢ラベルや冠詞 (A, a) は訂正対象外
        text = "(A) x (B) y"
        assert corrector.correct(text) == text

    def test_leaves_numbers_and_symbols(self, corrector):
        text = "12 + 34 = 46, $5.00"
        assert corrector.correct(text) == text

    def test_correct_with_changes_records_positions(self, corrector):
        text = "I will recieve it"
        corrected, changes = corrector.correct_with_changes(text)
        assert corrected == "I will receive it"
        assert len(changes) == 1
        assert changes[0]["original"] == "recieve"
        assert changes[0]["corrected"] == "receive"
        assert changes[0]["start"] == text.index("recieve")

    def test_preserves_whitespace_structure(self, corrector):
        # 改行・複数スペースは語間でそのまま保存される
        text = "speling\n  test"
        assert corrector.correct(text) == "spelling\n  test"


class TestSeq2SeqCorrector:
    def test_corrects_line_by_line(self):
        calls = []

        def fake_fn(line: str) -> str:
            calls.append(line)
            return line.replace("qick", "quick")

        c = Seq2SeqCorrector(model_name="dummy", generate_fn=fake_fn)
        text = "the qick fox\n(A) qick (B) slow"
        assert c.correct(text) == "the quick fox\n(A) quick (B) slow"
        assert calls == ["the qick fox", "(A) qick (B) slow"]

    def test_empty_lines_passthrough_without_calling_fn(self):
        calls = []

        def fake_fn(line: str) -> str:
            calls.append(line)
            return line

        c = Seq2SeqCorrector(model_name="dummy", generate_fn=fake_fn)
        assert c.correct("a\n\nb") == "a\n\nb"
        assert calls == ["a", "b"]

    def test_name(self):
        c = Seq2SeqCorrector(model_name="dummy", generate_fn=lambda s: s)
        assert c.name == "neural"


class TestLLMCorrector:
    def test_prompt_contains_text_and_conservative_instruction(self):
        prompts = []

        def fake_fn(prompt: str) -> str:
            prompts.append(prompt)
            return "<corrected>fixed text</corrected>"

        c = LLMCorrector(model_name="dummy", generate_fn=fake_fn)
        c.correct("some typo txt")
        assert len(prompts) == 1
        assert "some typo txt" in prompts[0]
        # 保守的プロンプト: typo のみ修正・他は変更禁止の文言
        assert "ONLY" in prompts[0]

    def test_parses_corrected_tags(self):
        c = LLMCorrector(
            model_name="dummy",
            generate_fn=lambda p: "<corrected>the quick fox</corrected>",
        )
        assert c.correct("the qick fox") == "the quick fox"

    def test_preserves_internal_newlines_strips_outer(self):
        c = LLMCorrector(
            model_name="dummy",
            generate_fn=lambda p: "<corrected>\nline one\n(A) x\n</corrected>",
        )
        assert c.correct("line one\n(A) x") == "line one\n(A) x"

    def test_retries_once_on_parse_failure(self):
        responses = iter(["I fixed it for you!", "<corrected>good text</corrected>"])
        n_calls = []

        def fake_fn(prompt: str) -> str:
            n_calls.append(prompt)
            return next(responses)

        c = LLMCorrector(model_name="dummy", generate_fn=fake_fn)
        assert c.correct("good textt") == "good text"
        assert len(n_calls) == 2
        # リトライ時はプロンプトが同一でない (greedy 同一入力→同一出力のため)
        assert n_calls[0] != n_calls[1]

    def test_returns_original_when_both_parses_fail(self):
        c = LLMCorrector(model_name="dummy", generate_fn=lambda p: "no tags here")
        original = "the qick fox"
        corrected, meta = c.correct_with_meta(original)
        assert corrected == original
        assert meta["parse_failed"] is True

    def test_meta_reports_success(self):
        c = LLMCorrector(
            model_name="dummy",
            generate_fn=lambda p: "<corrected>ok</corrected>",
        )
        corrected, meta = c.correct_with_meta("okk")
        assert corrected == "ok"
        assert meta["parse_failed"] is False
        assert meta["n_calls"] == 1

    def test_name(self):
        c = LLMCorrector(model_name="dummy", generate_fn=lambda p: p)
        assert c.name == "llm"


class TestCreateCorrector:
    def test_pyspell(self):
        assert isinstance(create_corrector("pyspell"), PySpellCorrector)

    def test_neural_with_injected_fn(self):
        c = create_corrector("neural", model_name="dummy", generate_fn=lambda s: s)
        assert isinstance(c, Seq2SeqCorrector)

    def test_llm_with_injected_fn(self):
        c = create_corrector("llm", model_name="dummy", generate_fn=lambda s: s)
        assert isinstance(c, LLMCorrector)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_corrector("unknown")

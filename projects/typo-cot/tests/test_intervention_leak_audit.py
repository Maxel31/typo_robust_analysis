"""intervention.leak_audit のテスト (A2: restore「自明コピー」批判への反証).

GPU 不要・CPU のみ。3 点セットの共通部品を検証する:
  (i)   リーク層別: 強制 clean CoT prefix に最終答え文字列が現れるか
  (ii)  結論剥ぎ: prefix の最終行/最終文を除去する strip_conclusion
  (iii) 回復曲線: 先頭 p% prefix を切り出す cut_prefix_by_fraction
"""

import pytest

from typo_cot.intervention.leak_audit import (
    LeakResult,
    answer_leak,
    cut_prefix_by_fraction,
    letter_anywhere_leak,
    letter_marker_leak,
    numeric_leak,
    numeric_leak_lastline,
    option_text_leak,
    strip_conclusion,
)


class TestNumericLeak:
    """GSM8K: 金答え数値が prefix に現れるか."""

    def test_gold_number_present_anywhere(self):
        prefix = "She has 16 - 3 - 4 = 9 eggs left. 9 * 2 = 18 dollars.\n"
        assert numeric_leak(prefix, "18") is True

    def test_gold_number_absent(self):
        prefix = "She has 16 - 3 - 4 = 9 eggs left. 9 * 2 dollars.\n"
        assert numeric_leak(prefix, "42") is False

    def test_substring_not_matched(self):
        # 180 は 18 を含むが数値トークンとしては別 → False
        prefix = "The total was 180 units in stock.\n"
        assert numeric_leak(prefix, "18") is False

    def test_comma_thousands_normalized(self):
        prefix = "The revenue was 1,234 in total.\n"
        assert numeric_leak(prefix, "1234") is True

    def test_lastline_leak_true(self):
        prefix = "First 16 - 3 = 13.\nSo she makes 9 * 2 = 18 dollars.\n"
        assert numeric_leak_lastline(prefix, "18") is True

    def test_lastline_leak_false_when_only_earlier(self):
        prefix = "So she makes 18 total early.\nThen a final unrelated 7 step.\n"
        assert numeric_leak_lastline(prefix, "18") is False
        assert numeric_leak(prefix, "18") is True


class TestLetterLeak:
    """MMLU: 金答え選択肢文字が prefix に現れるか."""

    def test_marker_parenthesized(self):
        assert letter_marker_leak("Therefore the result is (B) clearly.", "B") is True

    def test_marker_option_word(self):
        assert letter_marker_leak("Option C is the strongest match.", "C") is True

    def test_marker_absent_no_article_false_positive(self):
        # "A" が冠詞として出るだけでは marker leak にしない
        assert letter_marker_leak("A free abelian group is infinite.", "A") is False

    def test_anywhere_catches_standalone(self):
        assert letter_anywhere_leak("We pick D over the rest.", "D") is True

    def test_anywhere_absent(self):
        assert letter_anywhere_leak("Both statements hold here.", "B") is False


class TestOptionTextLeak:
    def test_option_text_present(self):
        prefix = "Both statements are true here.\n"
        assert option_text_leak(prefix, "True, True") is True

    def test_option_text_absent(self):
        prefix = "The permutation order divides n factorial.\n"
        assert option_text_leak(prefix, "False, False") is False

    def test_short_option_skipped(self):
        # 極端に短い選択肢 (数字等) は誤マッチ回避のため leak 扱いしない
        assert option_text_leak("There are 6 apples on 6 tables.", "6") is False


class TestAnswerLeak:
    def test_gsm8k_dispatch(self):
        res = answer_leak("9 * 2 = 18 dollars.\n", "18", "gsm8k")
        assert isinstance(res, LeakResult)
        assert res.numeric_leak is True
        assert res.leaked is True

    def test_mmlu_no_leak(self):
        # 金答え文字も選択肢本文も現れない → leaked False
        res = answer_leak(
            "Both statements are false in this case.\n",
            "A",
            "mmlu",
            choices=["True, True", "False, False", "T", "F"],
        )
        assert res.letter_marker_leak is False
        assert res.option_text_leak is False
        assert res.leaked is False

    def test_mmlu_option_text_leak_sets_leaked(self):
        res = answer_leak(
            "Both statements are true.\n",
            "A",
            "mmlu",
            choices=["True, True", "False, False", "True, False", "False, True"],
        )
        assert res.option_text_leak is True
        assert res.leaked is True


class TestStripConclusion:
    def test_strip_last_line(self):
        prefix = "Step one is done.\nStep two gives 18.\n"
        out = strip_conclusion(prefix, mode="last_line")
        assert "18" not in out
        assert out.rstrip("\n") == "Step one is done."

    def test_strip_last_line_handles_trailing_blank(self):
        prefix = "Line A.\nFinal line B.\n\n  \n"
        out = strip_conclusion(prefix, mode="last_line")
        assert "Final line B" not in out
        assert "Line A." in out

    def test_strip_last_sentence(self):
        prefix = "First sentence. Second sentence gives 42."
        out = strip_conclusion(prefix, mode="last_sentence")
        assert "42" not in out
        assert "First sentence." in out

    def test_strip_single_line_returns_empty_ish(self):
        out = strip_conclusion("Only one line here.\n", mode="last_line")
        assert out.strip() == ""


class TestCutPrefixByFraction:
    def test_zero_is_empty(self):
        assert cut_prefix_by_fraction("hello world foo", 0) == ""

    def test_hundred_is_full(self):
        text = "hello world foo"
        assert cut_prefix_by_fraction(text, 100) == text

    def test_snaps_to_word_boundary(self):
        text = "aaaa bbbb cccc dddd"  # len 19
        out = cut_prefix_by_fraction(text, 50)
        # 語の途中で切らない: 空白で終わる
        assert out == "" or out.endswith(" ")
        assert text.startswith(out)

    def test_monotone_growth(self):
        text = "the quick brown fox jumps over the lazy dog again"
        lens = [len(cut_prefix_by_fraction(text, p)) for p in (0, 25, 50, 75, 100)]
        assert lens == sorted(lens)

"""実験4-MATH拡張: fixed-target の boxed (MATH-500) 対応のテスト.

MATH の最終回答は "The answer is \\boxed{...}" 形式 (LaTeX, ネスト中括弧あり) の
ため、正規表現 group(1) ベースの ANSWER_PATTERNS では回答スパンを特定できない。
- find_boxed_answer: 最後の「閉じた」\\boxed{...} の中身スパンを文字単位の
  括弧追跡で特定する (evaluation.extractor.MATHAnswerExtractor と同じ
  「最後の閉じた boxed 採用・未閉じは無視」規約)。
- plan_splice: どちらかのテキストに \\boxed があれば boxed 経路を優先し、
  片側にしか無い場合はスキップ (union 除外の strict=boxed 規約と整合)。
  \\boxed が無いテキスト同士は従来の ANSWER_PATTERNS 経路 (回帰なし)。
- map_answer_char_spans_to_tokens: lrp.analyzer._find_answer_pattern の
  トークン位置決定ループと同一規約の写像 (境界・overlap 判定を凍結)。
"""

from typo_cot.attribution.fixed_target import (
    find_answer_token_positions,
    find_boxed_answer,
    map_answer_char_spans_to_tokens,
    plan_splice,
)


class TestFindBoxedAnswer:
    """最後の閉じた \\boxed{...} の中身スパン検出."""

    def test_simple_number(self):
        text = "Some CoT here. The answer is \\boxed{42}."
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "42"
        assert text[b.content_start : b.content_end] == "42"
        # 回答フレーズ先頭 ("The answer is ...") をパターン開始とする
        assert text[b.pattern_start :].startswith("The answer is")

    def test_nested_braces_frac(self):
        text = "The answer is \\boxed{\\frac{\\pi}{2}}."
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "\\frac{\\pi}{2}"

    def test_deeply_nested(self):
        text = "The answer is \\boxed{\\sqrt{\\frac{1}{2}}}."
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "\\sqrt{\\frac{1}{2}}"

    def test_final_answer_dollar_form(self):
        """gemma/Llama 系の 'The final answer is $\\boxed{X}$' 形式."""
        text = "Final Answer: The final answer is $\\boxed{\\frac{1}{2}}$."
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "\\frac{1}{2}"
        assert text[b.pattern_start :].startswith("The final answer is")

    def test_colon_after_answer_is(self):
        text = "The final answer is: $\\boxed{7}$"
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "7"
        assert text[b.pattern_start :].startswith("The final answer is")

    def test_last_closed_boxed_wins(self):
        text = "We get \\boxed{1}. Wait. The answer is \\boxed{2}."
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "2"

    def test_trailing_unclosed_boxed_ignored(self):
        """途中打ち切り (未閉じ boxed) は無視し、直前の閉じた boxed を採用
        (extractor の「未閉じ→None→前の閉じた boxed 維持」規約と同一)."""
        text = "First \\boxed{1}. Then truncated \\boxed{\\frac{1"
        b = find_boxed_answer(text)
        assert b is not None
        assert b.content == "1"

    def test_all_unclosed_returns_none(self):
        assert find_boxed_answer("The answer is \\boxed{\\frac{1}{2") is None

    def test_no_boxed_returns_none(self):
        assert find_boxed_answer("The answer is 42") is None

    def test_no_trigger_phrase_uses_boxed_start(self):
        text = "Therefore \\boxed{10}."
        b = find_boxed_answer(text)
        assert b is not None
        assert text[b.pattern_start :].startswith("\\boxed")


class TestPlanSpliceBoxed:
    """boxed 経路の splice 計画 (MATH)."""

    BASE = "CoT base. The answer is \\boxed{\\frac{\\pi}{2}}."
    PERT = "CoT pert. The answer is \\boxed{\\frac{\\pi}{4}}."

    def test_flip_replaces_content_only(self):
        plan = plan_splice(self.BASE, self.PERT, "math_00000")
        assert plan.skip_reason is None
        assert plan.spliced is True
        assert plan.baseline_answer == "\\frac{\\pi}{2}"
        assert plan.perturbed_answer == "\\frac{\\pi}{4}"
        assert plan.baseline_pattern_type == "boxed"
        assert plan.perturbed_pattern_type == "boxed"
        assert (
            plan.spliced_text
            == "CoT pert. The answer is \\boxed{\\frac{\\pi}{2}}."
        )

    def test_non_flip_identical_content(self):
        pert_same = "CoT pert. The answer is \\boxed{\\frac{\\pi}{2}}."
        plan = plan_splice(self.BASE, pert_same, "math_00001")
        assert plan.skip_reason is None
        assert plan.spliced is False
        assert plan.spliced_text == pert_same

    def test_perturbed_without_boxed_is_skipped(self):
        """baseline が boxed のとき、摂動側に boxed が無ければスキップ
        (レガシー number パターンにはフォールバックしない —
        strict=boxed の union 除外と同じ母集団)."""
        pert = "CoT pert truncated. The answer is 42"
        plan = plan_splice(self.BASE, pert, "math_00002")
        assert plan.skip_reason == "no_perturbed_answer_pattern"
        assert plan.baseline_answer == "\\frac{\\pi}{2}"

    def test_baseline_without_boxed_is_skipped(self):
        base = "CoT base truncated. The answer is 42"
        plan = plan_splice(base, self.PERT, "math_00003")
        assert plan.skip_reason == "no_baseline_answer_pattern"

    def test_legacy_path_unchanged_when_no_boxed(self):
        """両側に \\boxed が無ければ従来経路 (GSM8K/MMLU の回帰なし)."""
        base = "So she has 18 left. The answer is 18"
        pert = "So she has 20 left. The answer is 20"
        plan = plan_splice(base, pert, "gsm8k_00000")
        assert plan.skip_reason is None
        assert plan.spliced is True
        assert plan.baseline_pattern_type == "number"
        assert plan.spliced_text == "So she has 20 left. The answer is 18"


class TestMapAnswerCharSpansToTokens:
    """lrp.analyzer._find_answer_pattern のトークン写像ループと同一規約."""

    def test_boundary_semantics(self):
        # トークン: [0,5) [5,10) | [10,14) [14,18) [18,24) [24,30)
        # (先頭2つはプロンプト)
        offset_list = [(0, 5), (5, 10), (10, 14), (14, 18), (18, 24), (24, 30)]
        ats, ate, acp = map_answer_char_spans_to_tokens(
            offset_list=offset_list,
            prompt_token_count=2,
            answer_char_start=14,
            answer_char_end=30,
            choice_char_start=24,
            choice_char_end=28,
        )
        # start: 最初の「end > answer_char_start」トークン (end=14 は不採用)
        assert ats == 3
        # choice: 最初の「start < choice_end かつ end > choice_start」トークン
        # ((18,24) は end==24 で不採用)
        assert acp == 5
        # end: 最後の「start < answer_char_end」トークン
        assert ate == 5

    def test_prompt_tokens_skipped(self):
        offset_list = [(0, 5), (5, 10), (10, 20)]
        ats, ate, acp = map_answer_char_spans_to_tokens(
            offset_list=offset_list,
            prompt_token_count=2,
            answer_char_start=2,
            answer_char_end=9,
            choice_char_start=3,
            choice_char_end=6,
        )
        # プロンプト範囲のトークンは候補にならない
        assert ats == 2
        assert acp == 2
        assert ate == 2


class TestFindAnswerTokenPositionsBoxed:
    """boxed 経路の一気通貫 (文字単位トークンで検証; analyzer 不要)."""

    def _char_tokens(self, full_text: str):
        return (
            list(full_text),
            [(i, i + 1) for i in range(len(full_text))],
        )

    def test_boxed_positions(self):
        prompt = "Q: p?\n"
        gen = "CoT. The answer is \\boxed{42}."
        full = prompt + gen
        tokens, offsets = self._char_tokens(full)
        ats, ate, acp = find_answer_token_positions(
            generated_text=gen,
            tokens=tokens,
            prompt_token_count=len(prompt),
            offset_list=offsets,
            prompt_length=len(prompt),
            analyzer=None,
        )
        # ターゲット = boxed 中身の先頭トークン ("4")
        assert full[acp] == "4"
        # 回答スパン開始 = "The answer is ..." の先頭 ("T")
        assert full[ats] == "T"
        assert ats == len(prompt) + gen.index("The answer is")
        # スパン終端 = 閉じ括弧 "}"
        assert full[ate] == "}"

    def test_no_boxed_without_analyzer_falls_back_to_none(self):
        prompt = "Q: p?\n"
        gen = "No final answer here"
        full = prompt + gen
        tokens, offsets = self._char_tokens(full)
        ats, ate, acp = find_answer_token_positions(
            generated_text=gen,
            tokens=tokens,
            prompt_token_count=len(prompt),
            offset_list=offsets,
            prompt_length=len(prompt),
            analyzer=None,
        )
        assert ats is None and ate is None and acp is None

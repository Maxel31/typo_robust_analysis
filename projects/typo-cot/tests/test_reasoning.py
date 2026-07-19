"""reasoning モデル(DeepSeek-R1蒸留系)サポートのテスト.

実験10③: R1蒸留系はチャットテンプレート+<think>タグのCoTを持つため、
- ゼロショットのユーザーメッセージ構築(質問スパンの追跡)
- チャットテンプレート適用後の全体プロンプト内の質問オフセット計算
- <think>...</think> の CoT / 最終回答セクション分離
- 回答抽出(ベンチマーク抽出器 + boxed フォールバックの連鎖)
を検証する。
"""

import pytest

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.models.reasoning import (
    R1_DISTILL_QWEN_7B,
    ReasoningSplit,
    build_full_prompt,
    build_user_message,
    extract_reasoning_answer,
    split_reasoning_output,
    think_prefix_end,
)


class StubTokenizer:
    """apply_chat_template を模したスタブ."""

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        assert not tokenize
        assert add_generation_prompt
        content = messages[0]["content"]
        return f"<BOS><User>{content}<Assistant><think>\n"


class TestBuildUserMessage:
    """build_user_message のテスト."""

    def test_gsm8k_contains_question_and_trigger(self) -> None:
        msg, q_start, q_end, q_choices_end = build_user_message(
            "gsm8k", "Tom has 3 apples. How many?"
        )
        assert "Tom has 3 apples. How many?" == msg[q_start:q_end]
        assert "The answer is" in msg
        assert q_choices_end == q_end

    def test_mmlu_contains_choices(self) -> None:
        msg, q_start, q_end, q_choices_end = build_user_message(
            "mmlu",
            "What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
        )
        assert msg[q_start:q_end] == "What is the capital of France?"
        # 選択肢は既存規約と同じ "(A) x (B) y" 空白区切り形式
        assert "(A) London (B) Paris (C) Berlin (D) Madrid" in msg
        assert "The answer is" in msg
        # 選択肢込み終端は選択肢の末尾
        assert msg[q_start:q_choices_end].endswith("Madrid")

    def test_mmlu_choices_embedded_in_question(self) -> None:
        # 摂動データセットでは選択肢が question に埋め込まれて渡される
        q = "What is X?\n(A) a (B) b (C) c (D) d"
        msg, q_start, q_end, q_choices_end = build_user_message("mmlu", q, choices=None)
        assert msg[q_start:q_choices_end] == q
        # 質問文のみの終端は最初の改行まで
        assert msg[q_start:q_end] == "What is X?"

    def test_math_boxed_trigger(self) -> None:
        msg, q_start, q_end, _ = build_user_message("math", "Compute $1+1$.")
        assert msg[q_start:q_end] == "Compute $1+1$."
        assert "\\boxed{}" in msg

    def test_unknown_benchmark_raises(self) -> None:
        with pytest.raises(ValueError):
            build_user_message("unknown_bench", "q")


class TestBuildFullPrompt:
    """build_full_prompt のテスト(チャットテンプレート適用+オフセット補正)."""

    def test_offsets_shifted_into_full_prompt(self) -> None:
        tokenizer = StubTokenizer()
        fp = build_full_prompt(
            tokenizer,
            "mmlu",
            "What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
        )
        assert fp.full_prompt.startswith("<BOS><User>")
        assert fp.full_prompt.endswith("<think>\n")
        assert (
            fp.full_prompt[fp.question_start : fp.question_end]
            == "What is the capital of France?"
        )
        assert fp.full_prompt[fp.question_start : fp.question_with_choices_end].endswith(
            "Madrid"
        )

    def test_model_constant(self) -> None:
        assert R1_DISTILL_QWEN_7B == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


class TestSplitReasoningOutput:
    """split_reasoning_output のテスト."""

    def test_split_with_think_tags(self) -> None:
        text = "<think>\nFirst I compute 1+1=2.\n</think>\n\nThe answer is 2."
        split = split_reasoning_output(text)
        assert isinstance(split, ReasoningSplit)
        assert split.cot_text == "First I compute 1+1=2."
        assert split.answer_text.strip() == "The answer is 2."
        assert split.has_think_close is True

    def test_split_without_opening_tag(self) -> None:
        # テンプレートが <think>\n を事前付与する場合、生成は本文から始まる
        text = "I reason here.\n</think>\n\nThe answer is (B)."
        split = split_reasoning_output(text)
        assert split.cot_text == "I reason here."
        assert split.answer_text.strip() == "The answer is (B)."
        assert split.has_think_close is True

    def test_split_truncated_cot(self) -> None:
        # max_new_tokens で切れて </think> が無い場合
        text = "<think>\nI keep reasoning forever..."
        split = split_reasoning_output(text)
        assert split.cot_text == "I keep reasoning forever..."
        assert split.answer_text == ""
        assert split.has_think_close is False


class TestThinkPrefixEnd:
    """think_prefix_end のテスト(R_Q の CoT 開始位置決定)."""

    def test_with_opening_tag(self) -> None:
        text = "<think>\nFirst step."
        idx = think_prefix_end(text)
        assert text[idx:].startswith("First step.")

    def test_without_opening_tag(self) -> None:
        text = "First step without tag."
        assert think_prefix_end(text) == 0


class TestExtractReasoningAnswer:
    """extract_reasoning_answer のテスト."""

    def test_gsm8k_answer_section(self) -> None:
        extractor = create_extractor("gsm8k")
        split = split_reasoning_output(
            "<think>\n5+6=11 so it is 11.\n</think>\n\nThe answer is 11."
        )
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "11"

    def test_gsm8k_boxed_fallback(self) -> None:
        # R1系は boxed で答えることが多い: GSM8K 抽出器で失敗したら boxed を試す
        extractor = create_extractor("gsm8k")
        split = split_reasoning_output(
            "<think>\ncompute...\n</think>\n\nFinal: \\(\\boxed{42}\\)"
        )
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "42"

    def test_mmlu_letter(self) -> None:
        extractor = create_extractor("mmlu")
        split = split_reasoning_output(
            "<think>\nParis is the capital.\n</think>\n\nThe answer is (B)."
        )
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "B"

    def test_math_boxed(self) -> None:
        extractor = create_extractor("math")
        split = split_reasoning_output(
            "<think>\n1+1=2\n</think>\n\nThe final answer is \\boxed{\\frac{1}{2}}."
        )
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "\\frac{1}{2}"

    def test_gsm8k_dollar_sign_fallback(self) -> None:
        # R1系は "The answer is $65,000." のように通貨記号を付けることがある
        extractor = create_extractor("gsm8k")
        split = split_reasoning_output(
            "<think>\ncompute...\n</think>\n\nThe answer is $65,000."
        )
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "65000"

    def test_truncated_falls_back_to_cot_tail(self) -> None:
        # </think> が無い場合は全文から抽出を試みる(答えが CoT 内に出ている場合の救済)
        extractor = create_extractor("gsm8k")
        split = split_reasoning_output("<think>\n... so The answer is 7.")
        result = extract_reasoning_answer(extractor, split)
        assert result.extracted_answer == "7"


class TestAllowedModels:
    """ModelWrapper が R1 蒸留モデルを許可することのテスト."""

    def test_r1_distill_allowed(self) -> None:
        from typo_cot.models.wrapper import ModelWrapper

        assert R1_DISTILL_QWEN_7B in ModelWrapper.ALLOWED_MODELS
        wrapper = ModelWrapper(model_name=R1_DISTILL_QWEN_7B)
        # Qwen2 アーキテクチャなので lxt サポート対象(R_Q計算に必要)
        assert wrapper.is_supported_for_lxt() is True

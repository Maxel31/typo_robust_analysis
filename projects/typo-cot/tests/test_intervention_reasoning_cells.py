"""intervention.reasoning_cells のテスト (実験1+3: R1蒸留系サポート).

GPU 不要。<think>...</think> 構造対応の CoT 切断、チャットテンプレート
prompt_builder の注入、R1 抽出チェーンの extract_fn を検証する。
基底5モデル (生プロンプト + "The answer is") の既存挙動は
test_intervention_cell_builder.py / test_intervention_runner.py が担保する。
"""

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.cell_builder import CellInputs, TruncationResult, build_cell_inputs
from typo_cot.intervention.reasoning_cells import (
    make_reasoning_extract_fn,
    make_reasoning_prompt_builder,
    truncate_reasoning_cot,
)
from typo_cot.intervention.records import PairRecord
from typo_cot.intervention.runner import run_cells


class FakeChatTokenizer:
    """apply_chat_template のみを持つ最小トークナイザー (プロンプト構築テスト用)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        # DeepSeek 風: BOS + User + 本文 + Assistant + <think> をテンプレートが付与
        return f"<｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜><think>\n"


# R1 生成テキストは <think> 本文 (先頭タグはプロンプト側) + </think> + 答え文。
GSM8K_GEN = (
    "Okay, she has 16 eggs. 16 - 3 - 4 = 9. 9 * 2 = 18. So the answer should be 18.\n"
    "</think>\n\nJanet sells 9 eggs at $2 each for $18.\n\nThe answer is 18."
)
GSM8K_GEN_TYPO = (
    "Okay, she has 16 eggs. 16 - 3 - 4 = 9. 9 * 2 = 17. So the answer should be 17.\n"
    "</think>\n\nShe sells for $17.\n\nThe answer is 17."
)


class TestTruncateReasoningCot:
    def test_cuts_before_post_think_declaration(self):
        res = truncate_reasoning_cot(GSM8K_GEN, benchmark="gsm8k")
        assert isinstance(res, TruncationResult)
        assert res.trigger_found is True
        # forced CoT は <think> 本文 + </think> + 宣言直前プロセを含む
        assert "</think>" in res.prefix
        assert res.prefix.endswith("The answer is 18.") is False
        assert "The answer is" not in res.prefix
        assert res.prefix.rstrip().endswith("$18.")

    def test_inner_think_answer_phrase_is_preserved(self):
        # <think> 本文内の "the answer should be 18" は切断トリガーにしない
        res = truncate_reasoning_cot(GSM8K_GEN, benchmark="gsm8k")
        assert "So the answer should be 18." in res.prefix

    def test_no_think_close_excluded(self):
        # </think> が生成されず途中切断 → trigger_found False
        cot = "Okay, reasoning that never closes the think block and no answer."
        res = truncate_reasoning_cot(cot, benchmark="gsm8k")
        assert res.trigger_found is False
        assert res.prefix == cot

    def test_no_declaration_in_answer_excluded(self):
        cot = "reasoning\n</think>\n\nSome closing remark without any declaration."
        res = truncate_reasoning_cot(cot, benchmark="gsm8k")
        assert res.trigger_found is False

    def test_multi_same_answer_identical(self):
        cot = (
            "reasoning\n</think>\n\nThe answer is 18.\nTo restate, The answer is 18."
        )
        res = truncate_reasoning_cot(cot, benchmark="gsm8k")
        assert res.trigger_count == 2
        assert res.trigger_answers == ["18", "18"]
        assert res.trigger_answers_identical is True

    def test_multi_different_answer_not_identical(self):
        cot = "reasoning\n</think>\n\nThe answer is 3. Wait. The answer is 5."
        res = truncate_reasoning_cot(cot, benchmark="gsm8k")
        assert res.trigger_answers == ["3", "5"]
        assert res.trigger_answers_identical is False

    def test_mmlu_answer_colon_declaration(self):
        # R1 MMLU は "Answer: A" / "ANSWER: A" 形も使う
        cot = "reasoning about options\n</think>\n\nBoth are true.\nANSWER: A"
        res = truncate_reasoning_cot(cot, benchmark="mmlu")
        assert res.trigger_found is True
        assert "ANSWER: A" not in res.prefix
        assert res.prefix.rstrip().endswith("Both are true.")

    def test_math_boxed_declaration(self):
        cot = "long derivation\n</think>\n\nThus the final result is \\boxed{42}."
        res = truncate_reasoning_cot(cot, benchmark="math")
        assert res.trigger_found is True
        assert "\\boxed{" not in res.prefix
        assert "</think>" in res.prefix

    def test_early_trigger_never_flagged(self):
        # <think> 本文は常に丸ごと移植 → early_trigger は概念上 False
        res = truncate_reasoning_cot(GSM8K_GEN, benchmark="gsm8k")
        assert res.early_trigger is False


class TestReasoningPromptBuilder:
    def test_builds_chat_template_prompt(self):
        builder = make_reasoning_prompt_builder(FakeChatTokenizer())
        prompt = builder("gsm8k", "What is 2+2?", None, "default")
        assert "<｜User｜>" in prompt
        assert "<｜Assistant｜><think>" in prompt
        assert "What is 2+2?" in prompt

    def test_mmlu_choices_formatted(self):
        builder = make_reasoning_prompt_builder(FakeChatTokenizer())
        prompt = builder("mmlu", "Pick one.", ["x", "y", "z", "w"], "abstract_algebra")
        assert "(A) x" in prompt and "(D) w" in prompt


def _r1_pair() -> PairRecord:
    return PairRecord(
        sample_id="gsm8k_00000",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        benchmark="gsm8k",
        question_clean="Janet has 16 eggs. How many dollars?",
        question_typo="Janeet has 16 egs. How many dollars?",
        choices_clean=None,
        choices_typo=None,
        subset="default",
        correct_answer="18",
        cot_clean=GSM8K_GEN,
        cot_typo=GSM8K_GEN_TYPO,
        answer_clean="18",
        answer_typo="17",
        is_correct_clean=True,
    )


class TestBuildCellInputsWithReasoning:
    def test_injected_builders_produce_four_cells(self):
        pair = _r1_pair()
        cells = build_cell_inputs(
            pair,
            prompt_builder=make_reasoning_prompt_builder(FakeChatTokenizer()),
            truncator=truncate_reasoning_cot,
            dedup_same_answer_triggers=True,
        )
        assert isinstance(cells, CellInputs)
        assert set(cells.prompts.keys()) == {"A", "B", "C", "D"}
        # forced CoT は </think> を含み、答え宣言は含まない
        for cot in cells.forced_cots.values():
            assert "</think>" in cot
            assert "The answer is" not in cot
        # チャットテンプレート prompt が使われている
        assert "<｜Assistant｜><think>" in cells.prompts["A"]
        assert cells.exclude is False

    def test_benign_restatement_not_excluded(self):
        pair = _r1_pair()
        pair.cot_clean = "reason\n</think>\n\nThe answer is 18.\nAgain, The answer is 18."
        cells = build_cell_inputs(
            pair,
            prompt_builder=make_reasoning_prompt_builder(FakeChatTokenizer()),
            truncator=truncate_reasoning_cot,
            dedup_same_answer_triggers=True,
        )
        assert "multi_trigger_clean" not in cells.exclude_reasons

    def test_no_think_close_excluded(self):
        pair = _r1_pair()
        pair.cot_typo = "reasoning that never closes think"
        cells = build_cell_inputs(
            pair,
            prompt_builder=make_reasoning_prompt_builder(FakeChatTokenizer()),
            truncator=truncate_reasoning_cot,
            dedup_same_answer_triggers=True,
        )
        assert cells.exclude is True
        assert "no_trigger_typo" in cells.exclude_reasons


class TestRunCellsWithReasoning:
    def test_answers_extracted_via_reasoning_chain(self):
        pair = _r1_pair()
        extractor = create_extractor("gsm8k")

        def fake_generate(prompts: list[str]) -> list[str]:
            # 強制 CoT に "= 17" があれば typo 側、なければ clean 側の答えを再生成
            out = []
            for p in prompts:
                out.append("The answer is 17." if "= 17" in p else "The answer is 18.")
            return out

        outcomes = run_cells(
            [pair],
            fake_generate,
            prompt_builder=make_reasoning_prompt_builder(FakeChatTokenizer()),
            truncator=truncate_reasoning_cot,
            extract_fn=make_reasoning_extract_fn(extractor),
            dedup_same_answer_triggers=True,
        )
        o = outcomes[0]
        assert o.answers["A"] == "18"  # clean q + clean cot
        assert o.answers["B"] == "17"  # typo q + typo cot
        assert o.answers["C"] == "18"  # typo q + clean cot (DE 復帰)
        assert o.answers["D"] == "17"  # clean q + typo cot (IE flip)
        assert o.te_match is True

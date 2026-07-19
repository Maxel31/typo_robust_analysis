"""実験14: no-CoT (直接回答) プロンプトビルダーのテスト.

no-CoT 版は既存 CoT プロンプト (models/prompts.py) の few-shot 例示から
reasoning を除去し「質問→答えのみ」にする。答えトリガー
("The answer is X" / "The answer is (X)") は既存と同一に保ち、抽出器を
そのまま流用できることを保証する。
"""

import pytest

from typo_cot.models.prompts import (
    GSM8KPromptTemplate,
    MATHPromptTemplate,
    MMLUPromptTemplate,
)
from typo_cot.models.nocot_prompts import (
    NoCoTARCPromptTemplate,
    NoCoTCommonsenseQAPromptTemplate,
    NoCoTGSM8KPromptTemplate,
    NoCoTMATHPromptTemplate,
    NoCoTMMLUProPromptTemplate,
    NoCoTMMLUPromptTemplate,
    create_nocot_prompt_template,
)


class TestNoCoTGSM8K:
    def test_examples_have_no_reasoning(self) -> None:
        """few-shot 例示から reasoning 文が除去されている."""
        cot = GSM8KPromptTemplate().generate(question="Q")
        nocot = NoCoTGSM8KPromptTemplate().generate(question="Q")
        # CoT 版に含まれる reasoning 文は no-CoT 版には無い
        assert "2 cans of 3 tennis balls each is 6" in cot.system_prompt
        assert "2 cans of 3 tennis balls each is 6" not in nocot.system_prompt

    def test_answer_trigger_preserved(self) -> None:
        """答えトリガー 'The answer is <n>.' は保持される."""
        nocot = NoCoTGSM8KPromptTemplate().generate(question="Q")
        assert "The answer is 11." in nocot.system_prompt
        assert "The answer is 6." in nocot.system_prompt

    def test_example_count_preserved(self) -> None:
        """例示数 (8) は CoT 版と同一."""
        nocot = NoCoTGSM8KPromptTemplate().generate(question="Q")
        assert nocot.system_prompt.count("Example") == 8

    def test_direct_answer_directly_after_label(self) -> None:
        """'Solution:' 直後に reasoning を挟まず答えが来る."""
        nocot = NoCoTGSM8KPromptTemplate().generate(question="Q")
        assert "Solution: The answer is 11." in nocot.system_prompt

    def test_system_instruction_is_direct(self) -> None:
        """system 指示が step-by-step ではなく直接回答を促す."""
        nocot = NoCoTGSM8KPromptTemplate().generate(question="Q")
        assert "step by step" not in nocot.system_prompt.lower()

    def test_user_prompt_structure_unchanged(self) -> None:
        """query 側 user_prompt は CoT 版と同一構造 (Solution: で終わる)."""
        cot = GSM8KPromptTemplate().generate(question="MyQ")
        nocot = NoCoTGSM8KPromptTemplate().generate(question="MyQ")
        assert cot.user_prompt == nocot.user_prompt
        assert nocot.user_prompt.endswith("Solution:")


class TestNoCoTMMLU:
    def test_examples_have_no_reasoning(self) -> None:
        nocot = NoCoTMMLUPromptTemplate().generate(
            question="Q", choices=["a", "b", "c", "d"], subject="math"
        )
        assert "France is a country" not in nocot.system_prompt

    def test_answer_trigger_preserved(self) -> None:
        nocot = NoCoTMMLUPromptTemplate().generate(
            question="Q", choices=["a", "b", "c", "d"], subject="math"
        )
        assert "The answer is (B)." in nocot.system_prompt

    def test_example_count_preserved(self) -> None:
        nocot = NoCoTMMLUPromptTemplate().generate(
            question="Q", choices=["a", "b", "c", "d"], subject="math"
        )
        assert nocot.system_prompt.count("Example") == 5

    def test_user_prompt_structure_unchanged(self) -> None:
        cot = MMLUPromptTemplate().generate(
            question="MyQ", choices=["a", "b", "c", "d"], subject="math"
        )
        nocot = NoCoTMMLUPromptTemplate().generate(
            question="MyQ", choices=["a", "b", "c", "d"], subject="math"
        )
        assert cot.user_prompt == nocot.user_prompt
        assert nocot.user_prompt.endswith("Step-by-step reasoning:")

    def test_system_instruction_is_direct(self) -> None:
        nocot = NoCoTMMLUPromptTemplate().generate(
            question="Q", choices=["a", "b", "c", "d"], subject="math"
        )
        assert "step by step" not in nocot.system_prompt.lower()


class TestNoCoTMATH:
    def test_boxed_answer_trigger_preserved(self) -> None:
        nocot = NoCoTMATHPromptTemplate().generate(question="Q")
        assert "The answer is \\boxed{2}." in nocot.system_prompt

    def test_no_reasoning(self) -> None:
        cot = MATHPromptTemplate().generate(question="Q")
        nocot = NoCoTMATHPromptTemplate().generate(question="Q")
        assert "Subtract 3 from both sides" in cot.system_prompt
        assert "Subtract 3 from both sides" not in nocot.system_prompt


class TestFactory:
    @pytest.mark.parametrize(
        "benchmark,cls",
        [
            ("gsm8k", NoCoTGSM8KPromptTemplate),
            ("mmlu", NoCoTMMLUPromptTemplate),
            ("mmlu_pro", NoCoTMMLUProPromptTemplate),
            ("arc", NoCoTARCPromptTemplate),
            ("commonsense_qa", NoCoTCommonsenseQAPromptTemplate),
            ("math", NoCoTMATHPromptTemplate),
        ],
    )
    def test_factory_returns_nocot(self, benchmark: str, cls: type) -> None:
        assert isinstance(create_nocot_prompt_template(benchmark), cls)

    def test_arc_and_csqa_subject_fixed(self) -> None:
        """ARC/CSQA は subject 固定で親 (MMLU no-CoT) の挙動を継承."""
        arc = create_nocot_prompt_template("arc").generate(
            question="Q", choices=["a", "b", "c", "d"]
        )
        assert "The answer is (A)." in arc.system_prompt  # ARC 例1の答えは A
        assert "science" in arc.system_prompt

    def test_unknown_benchmark_raises(self) -> None:
        with pytest.raises(ValueError, match="不明なベンチマーク"):
            create_nocot_prompt_template("nope")

    def test_extractor_compatibility(self) -> None:
        """no-CoT 出力想定文字列が既存抽出器でパースできる (トリガー同一の担保)."""
        from typo_cot.evaluation.extractor import create_extractor

        gsm = create_extractor("gsm8k")
        assert gsm.extract("The answer is 42.").extracted_answer == "42"
        mmlu = create_extractor("mmlu")
        assert mmlu.extract("The answer is (C).").extracted_answer == "C"

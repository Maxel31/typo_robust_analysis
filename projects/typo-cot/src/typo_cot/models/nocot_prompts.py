"""実験14: no-CoT (直接回答) few-shot プロンプトビルダー.

DE = ショートカット依存度 仮説 (H14) の検証用。既存 CoT プロンプト
(models/prompts.py) の few-shot 例示から reasoning を除去し「質問→答え
のみ」の直接回答形式にする。

設計方針 (交絡を最小化するため 1 変数のみ操作):
  - few-shot 例示の reasoning を除去 (_format_example を override)。
  - system 指示を「直接答える」旨に差し替え (step-by-step の記述を除去)。
  - **答えトリガー ("The answer is X" / "The answer is (X)") と query 側
    user_prompt の骨格 (ラベル "Solution:" / "Step-by-step reasoning:") は
    CoT 版と完全に同一に保つ** → evaluation.extractor をそのまま流用でき、
    CoT vs no-CoT の差は「例示に reasoning があるか否か」だけになる。

generate() は親クラスをそのまま継承する (self.SYSTEM_INSTRUCTION /
self._format_example / self.FEW_SHOT_EXAMPLES を参照するため、override 済み
のメンバがそのまま反映される)。
"""

from typo_cot.models.prompts import (
    ARCPromptTemplate,
    BasePromptTemplate,
    CommonsenseQAPromptTemplate,
    GSM8KPromptTemplate,
    MATHPromptTemplate,
    MMLUProPromptTemplate,
    MMLUPromptTemplate,
)


class NoCoTGSM8KPromptTemplate(GSM8KPromptTemplate):
    """GSM8K no-CoT: 数値を直接 'The answer is [number].' で回答."""

    SYSTEM_INSTRUCTION = """Solve the following math problems. For each problem, provide only the final answer directly in the format "The answer is [number]." Do not show any reasoning.

Here are some examples:"""

    def _format_example(self, example: dict, index: int) -> str:
        return f"""
---
Example {index + 1}:
Problem: {example["question"]}

Solution: The answer is {example["answer"]}.
---"""


class NoCoTMATHPromptTemplate(MATHPromptTemplate):
    """MATH no-CoT: \\boxed{...} を直接回答."""

    SYSTEM_INSTRUCTION = """Solve the following math problems. For each problem, provide only the final answer directly, enclosed in \\boxed{} in the format "The answer is \\boxed{...}." Do not show any reasoning.

Here are some examples:"""

    def _format_example(self, example: dict, index: int) -> str:
        return f"""
---
Example {index + 1}:
Problem: {example["question"]}

Solution: The answer is {example["answer"]}.
---"""


class NoCoTMMLUPromptTemplate(MMLUPromptTemplate):
    """MMLU no-CoT: 選択肢文字を直接 'The answer is (X)' で回答."""

    SYSTEM_INSTRUCTION = """The following are multiple-choice questions about {subject}. For each question, provide only the final answer directly in the format "The answer is (X)" where X is the correct letter choice. Do not show any reasoning.

Here are some examples:"""

    def _format_example(self, example: dict, index: int) -> str:
        options_str = self._format_options(example["options"])
        return f"""
---
Example {index + 1}:
Question: {example["question"]}
{options_str}

Step-by-step reasoning: The answer is ({example["answer"]}).
---"""


class NoCoTMMLUProPromptTemplate(MMLUProPromptTemplate):
    """MMLU-Pro no-CoT (A-J 選択肢)."""

    SYSTEM_INSTRUCTION = NoCoTMMLUPromptTemplate.SYSTEM_INSTRUCTION
    _format_example = NoCoTMMLUPromptTemplate._format_example


class NoCoTARCPromptTemplate(NoCoTMMLUPromptTemplate):
    """ARC no-CoT (subject=science 固定, 親 CoT ARC の few-shot を継承)."""

    SYSTEM_INSTRUCTION = """The following are multiple-choice science questions. For each question, provide only the final answer directly in the format "The answer is (X)" where X is the correct letter choice. Do not show any reasoning.

Here are some examples:"""
    FEW_SHOT_EXAMPLES = ARCPromptTemplate.FEW_SHOT_EXAMPLES

    def generate(self, question, choices=None, context=None, subject=None, answer_start=None):
        return super().generate(
            question=question,
            choices=choices,
            context=context,
            subject="science",
            answer_start=answer_start,
        )


class NoCoTCommonsenseQAPromptTemplate(NoCoTMMLUPromptTemplate):
    """CommonsenseQA no-CoT (subject=commonsense reasoning 固定)."""

    SYSTEM_INSTRUCTION = """The following are multiple-choice commonsense reasoning questions. For each question, provide only the final answer directly in the format "The answer is (X)" where X is the correct letter choice. Do not show any reasoning.

Here are some examples:"""
    FEW_SHOT_EXAMPLES = CommonsenseQAPromptTemplate.FEW_SHOT_EXAMPLES

    def generate(self, question, choices=None, context=None, subject=None, answer_start=None):
        return super().generate(
            question=question,
            choices=choices,
            context=context,
            subject="commonsense reasoning",
            answer_start=answer_start,
        )


def create_nocot_prompt_template(benchmark: str) -> BasePromptTemplate:
    """ベンチマーク名から no-CoT プロンプトテンプレートを作成する.

    Args:
        benchmark: ベンチマーク名

    Returns:
        対応する no-CoT プロンプトテンプレート

    Raises:
        ValueError: 不明なベンチマーク名の場合
    """
    templates: dict[str, type[BasePromptTemplate]] = {
        "gsm8k": NoCoTGSM8KPromptTemplate,
        "math": NoCoTMATHPromptTemplate,
        "mmlu": NoCoTMMLUPromptTemplate,
        "mmlu_pro": NoCoTMMLUProPromptTemplate,
        "arc": NoCoTARCPromptTemplate,
        "commonsense_qa": NoCoTCommonsenseQAPromptTemplate,
    }
    if benchmark not in templates:
        raise ValueError(
            f"不明なベンチマーク: {benchmark}. 利用可能: {list(templates.keys())}"
        )
    return templates[benchmark]()

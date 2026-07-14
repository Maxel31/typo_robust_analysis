"""プロンプトテンプレートモジュールのテスト."""

import pytest

from typo_cot.models.prompts import (
    GSM8KPromptTemplate,
    MMLUPromptTemplate,
    MMLUProPromptTemplate,
    PromptResult,
    SQuADv2PromptTemplate,
    create_prompt_template,
)


class TestPromptResult:
    """PromptResultのテスト."""

    def test_get_full_prompt_with_system(self) -> None:
        """システムプロンプトがある場合の完全なプロンプト生成を確認."""
        result = PromptResult(
            system_prompt="System prompt",
            user_prompt="User prompt",
        )
        full = result.get_full_prompt()
        assert "System prompt" in full
        assert "User prompt" in full
        assert full == "System prompt\n\nUser prompt"

    def test_get_full_prompt_no_system(self) -> None:
        """システムプロンプトがない場合のプロンプト生成を確認."""
        result = PromptResult(
            system_prompt="",
            user_prompt="User prompt",
        )
        full = result.get_full_prompt()
        assert full == "User prompt"

    def test_prompt_result_with_subject(self) -> None:
        """サブジェクト情報を含むPromptResultを確認."""
        result = PromptResult(
            system_prompt="System",
            user_prompt="User",
            subject="mathematics",
        )
        assert result.subject == "mathematics"

    def test_prompt_result_with_answer_positions(self) -> None:
        """回答位置情報を含むPromptResultを確認."""
        result = PromptResult(
            system_prompt="",
            user_prompt="User",
            answer_start=10,
            answer_end=20,
        )
        assert result.answer_start == 10
        assert result.answer_end == 20

    def test_prompt_result_with_question_range(self) -> None:
        """質問文範囲情報を含むPromptResultを確認."""
        result = PromptResult(
            system_prompt="System",
            user_prompt="User",
            question_text="Test question",
            question_start_in_full=100,
            question_end_in_full=113,
        )
        assert result.question_text == "Test question"
        assert result.question_start_in_full == 100
        assert result.question_end_in_full == 113


class TestGSM8KPromptTemplate:
    """GSM8KPromptTemplateのテスト."""

    def test_generate_basic(self) -> None:
        """基本的なプロンプト生成を確認."""
        template = GSM8KPromptTemplate()
        result = template.generate(
            question="If John has 5 apples and gives away 2, how many does he have left?",
        )

        assert isinstance(result, PromptResult)
        assert "Problem: If John has 5 apples" in result.user_prompt
        assert result.user_prompt.endswith("Solution:")

    def test_few_shot_examples_in_system_prompt(self) -> None:
        """8-Shot例示がシステムプロンプトに含まれることを確認."""
        template = GSM8KPromptTemplate()
        result = template.generate(question="Test question")

        # 8つの例示が含まれていることを確認（Example N:形式）
        assert result.system_prompt.count("Example") == 8
        assert result.system_prompt.count("Problem:") == 8
        assert result.system_prompt.count("Solution:") == 8

        # 例示の内容を確認
        assert "Roger has 5 tennis balls" in result.system_prompt
        assert "The answer is 11" in result.system_prompt

    def test_default_num_shots(self) -> None:
        """デフォルトのショット数が8であることを確認."""
        template = GSM8KPromptTemplate()
        assert template.get_default_num_shots() == 8

    def test_answer_format(self) -> None:
        """回答形式が "The answer is X." であることを確認."""
        template = GSM8KPromptTemplate()
        result = template.generate(question="Test")

        # 例示内の回答形式を確認
        assert "The answer is" in result.system_prompt

    def test_question_range_in_full_prompt(self) -> None:
        """質問文範囲が正しく計算されることを確認."""
        template = GSM8KPromptTemplate()
        question = "If John has 5 apples and gives away 2, how many does he have left?"
        result = template.generate(question=question)

        full_prompt = result.get_full_prompt()
        # 質問文範囲が正しく設定されていることを確認
        assert result.question_text == question
        assert result.question_start_in_full > 0
        assert result.question_end_in_full > result.question_start_in_full

        # 実際に抽出した文字列が質問文と一致することを確認
        extracted = full_prompt[result.question_start_in_full : result.question_end_in_full]
        assert extracted == question


class TestMMLUPromptTemplate:
    """MMLUPromptTemplateのテスト."""

    def test_generate_basic(self) -> None:
        """基本的なプロンプト生成を確認."""
        template = MMLUPromptTemplate()
        result = template.generate(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            subject="elementary_mathematics",
        )

        assert isinstance(result, PromptResult)
        assert "What is 2+2?" in result.user_prompt
        assert "(A) 3" in result.user_prompt
        assert "(B) 4" in result.user_prompt
        assert "(C) 5" in result.user_prompt
        assert "(D) 6" in result.user_prompt
        assert result.subject == "elementary_mathematics"

    def test_generate_requires_choices(self) -> None:
        """選択肢がない場合にエラーが発生することを確認."""
        template = MMLUPromptTemplate()
        with pytest.raises(ValueError, match="選択肢が必要"):
            template.generate(question="What is 2+2?")

    def test_system_prompt_contains_subject(self) -> None:
        """システムプロンプトにサブジェクトが含まれることを確認."""
        template = MMLUPromptTemplate()
        result = template.generate(
            question="Test?",
            choices=["A", "B", "C", "D"],
            subject="abstract_algebra",
        )

        assert "abstract algebra" in result.system_prompt

    def test_system_prompt_contains_answer_format(self) -> None:
        """システムプロンプトに回答形式が含まれることを確認."""
        template = MMLUPromptTemplate()
        result = template.generate(
            question="Test?",
            choices=["A", "B", "C", "D"],
        )

        assert "The answer is (X)" in result.system_prompt

    def test_five_shot_examples(self) -> None:
        """5-Shot例示が含まれることを確認."""
        template = MMLUPromptTemplate()
        result = template.generate(
            question="Test?",
            choices=["A", "B", "C", "D"],
        )

        # 5つの例示（Example N:形式）
        assert result.system_prompt.count("Example") == 5
        assert result.system_prompt.count("Question:") == 5

    def test_default_num_shots(self) -> None:
        """デフォルトのショット数が5であることを確認."""
        template = MMLUPromptTemplate()
        assert template.get_default_num_shots() == 5

    def test_options_format(self) -> None:
        """選択肢が (A), (B)... 形式でフォーマットされることを確認."""
        template = MMLUPromptTemplate()
        result = template.generate(
            question="Test?",
            choices=["Option1", "Option2", "Option3", "Option4"],
        )

        assert "(A) Option1" in result.user_prompt
        assert "(B) Option2" in result.user_prompt

    def test_question_range_in_full_prompt(self) -> None:
        """質問文範囲が正しく計算されることを確認."""
        template = MMLUPromptTemplate()
        question = "What is 2+2?"
        result = template.generate(
            question=question,
            choices=["3", "4", "5", "6"],
            subject="math",
        )

        full_prompt = result.get_full_prompt()
        # 質問文範囲が正しく設定されていることを確認
        assert result.question_text == question
        assert result.question_start_in_full > 0
        assert result.question_end_in_full > result.question_start_in_full

        # 実際に抽出した文字列が質問文と一致することを確認
        extracted = full_prompt[result.question_start_in_full : result.question_end_in_full]
        assert extracted == question


class TestMMLUProPromptTemplate:
    """MMLUProPromptTemplateのテスト."""

    def test_generate_basic(self) -> None:
        """基本的なプロンプト生成を確認."""
        template = MMLUProPromptTemplate()
        result = template.generate(
            question="What is the time complexity of binary search?",
            choices=["O(1)", "O(n)", "O(log n)", "O(n log n)", "O(n²)"],
            subject="computer_science",
        )

        assert isinstance(result, PromptResult)
        assert "binary search" in result.user_prompt
        assert "(A) O(1)" in result.user_prompt
        assert "(E) O(n²)" in result.user_prompt

    def test_generate_requires_choices(self) -> None:
        """選択肢がない場合にエラーが発生することを確認."""
        template = MMLUProPromptTemplate()
        with pytest.raises(ValueError, match="選択肢が必要"):
            template.generate(question="Test question?")

    def test_supports_ten_choices(self) -> None:
        """10個の選択肢をサポートすることを確認."""
        template = MMLUProPromptTemplate()
        choices = [f"Choice {i}" for i in range(10)]
        result = template.generate(
            question="Test?",
            choices=choices,
        )

        assert "(A) Choice 0" in result.user_prompt
        assert "(J) Choice 9" in result.user_prompt

    def test_five_shot_examples(self) -> None:
        """5-Shot例示が含まれることを確認."""
        template = MMLUProPromptTemplate()
        result = template.generate(
            question="Test?",
            choices=["A", "B", "C", "D", "E"],
        )

        # 5つの例示（Example N:形式）
        assert result.system_prompt.count("Example") == 5
        assert result.system_prompt.count("Question:") == 5

    def test_default_num_shots(self) -> None:
        """デフォルトのショット数が5であることを確認."""
        template = MMLUProPromptTemplate()
        assert template.get_default_num_shots() == 5


class TestSQuADv2PromptTemplate:
    """SQuADv2PromptTemplateのテスト."""

    def test_generate_basic(self) -> None:
        """基本的なプロンプト生成を確認."""
        template = SQuADv2PromptTemplate()
        result = template.generate(
            question="What is the capital?",
            context="Tokyo is the capital of Japan.",
        )

        assert isinstance(result, PromptResult)
        assert "What is the capital?" in result.user_prompt
        assert "Tokyo is the capital of Japan." in result.user_prompt

    def test_generate_requires_context(self) -> None:
        """コンテキストがない場合にエラーが発生することを確認."""
        template = SQuADv2PromptTemplate()
        with pytest.raises(ValueError, match="コンテキストが必要"):
            template.generate(question="What is the capital?")

    def test_no_cot_prompt(self) -> None:
        """CoT推論プロンプトがないことを確認（読解QAタスク）."""
        template = SQuADv2PromptTemplate()
        result = template.generate(
            question="Test?",
            context="Test context.",
        )

        # システムプロンプトが空であることを確認
        assert result.system_prompt == ""
        # シンプルな形式であることを確認
        assert "Context:" in result.user_prompt
        assert "Question:" in result.user_prompt
        assert "Answer:" in result.user_prompt

    def test_default_num_shots(self) -> None:
        """デフォルトのショット数が0であることを確認."""
        template = SQuADv2PromptTemplate()
        assert template.get_default_num_shots() == 0

    def test_answer_position_stored(self) -> None:
        """回答位置情報が保存されることを確認."""
        template = SQuADv2PromptTemplate()
        result = template.generate(
            question="What is the capital?",
            context="Tokyo is the capital of Japan.",
            answer_start=0,
        )

        assert result.answer_start == 0

    def test_simple_format(self) -> None:
        """シンプルな読解QA形式であることを確認."""
        template = SQuADv2PromptTemplate()
        result = template.generate(
            question="Who is the president?",
            context="The president of the company is John.",
        )

        expected_format = (
            "Context: The president of the company is John.\n\n"
            "Question: Who is the president?\n\n"
            "Answer:"
        )
        assert result.user_prompt == expected_format

    def test_question_range_in_full_prompt(self) -> None:
        """質問文範囲が正しく計算されることを確認."""
        template = SQuADv2PromptTemplate()
        question = "What is the capital of Japan?"
        context = "Tokyo is the capital of Japan."
        result = template.generate(question=question, context=context)

        full_prompt = result.get_full_prompt()
        # 質問文範囲が正しく設定されていることを確認
        assert result.question_text == question
        assert result.question_start_in_full > 0
        assert result.question_end_in_full > result.question_start_in_full

        # 実際に抽出した文字列が質問文と一致することを確認
        extracted = full_prompt[result.question_start_in_full : result.question_end_in_full]
        assert extracted == question

    def test_context_range_in_full_prompt(self) -> None:
        """コンテキスト範囲が正しく計算されることを確認."""
        template = SQuADv2PromptTemplate()
        question = "What is the capital of Japan?"
        context = "Tokyo is the capital of Japan."
        result = template.generate(question=question, context=context)

        full_prompt = result.get_full_prompt()
        # コンテキスト範囲が正しく設定されていることを確認
        assert result.context_start_in_full is not None
        assert result.context_end_in_full is not None
        assert result.context_start_in_full == len("Context: ")
        assert result.context_end_in_full == result.context_start_in_full + len(context)

        # 実際に抽出した文字列がコンテキストと一致することを確認
        extracted = full_prompt[result.context_start_in_full : result.context_end_in_full]
        assert extracted == context


class TestCreatePromptTemplate:
    """create_prompt_templateファクトリ関数のテスト."""

    def test_create_mmlu_template(self) -> None:
        """MMLUテンプレートが作成されることを確認."""
        template = create_prompt_template("mmlu")
        assert isinstance(template, MMLUPromptTemplate)

    def test_create_mmlu_pro_template(self) -> None:
        """MMLU-Proテンプレートが作成されることを確認."""
        template = create_prompt_template("mmlu_pro")
        assert isinstance(template, MMLUProPromptTemplate)

    def test_create_gsm8k_template(self) -> None:
        """GSM8Kテンプレートが作成されることを確認."""
        template = create_prompt_template("gsm8k")
        assert isinstance(template, GSM8KPromptTemplate)

    def test_create_squad_v2_template(self) -> None:
        """SQuAD v2テンプレートが作成されることを確認."""
        template = create_prompt_template("squad_v2")
        assert isinstance(template, SQuADv2PromptTemplate)

    def test_create_unknown_template(self) -> None:
        """不明なベンチマーク名でエラーが発生することを確認."""
        with pytest.raises(ValueError, match="不明なベンチマーク"):
            create_prompt_template("unknown")

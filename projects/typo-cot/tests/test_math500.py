"""MATH-500 (実験10②) のローダー・boxed 抽出・プロンプトのテスト.

実装は JSAI2026 v2 で移行済み(loader.MATHLoader / extractor.MATHAnswerExtractor /
prompts.MATHPromptTemplate)だがテストが無かったため、実験10で使う前に
挙動を凍結する検証テストを追加する。
"""

from unittest.mock import MagicMock, patch

from typo_cot.data.loader import MATHLoader, create_loader
from typo_cot.evaluation.extractor import MATHAnswerExtractor, create_extractor
from typo_cot.models.prompts import MATHPromptTemplate, create_prompt_template


class _FakeDataset:
    """datasets.Dataset の最小スタブ(len / iter / select)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        return _FakeDataset([self._rows[i] for i in indices])


class TestMATHLoader:
    """MATHLoader のテスト (HuggingFaceH4/MATH-500)."""

    @patch("typo_cot.data.loader.load_dataset")
    def test_load_all(self, mock_load_dataset: MagicMock) -> None:
        mock_load_dataset.return_value = _FakeDataset(
            [
                {"problem": "Compute $1+1$.", "answer": "2", "subject": "Algebra"},
                {
                    "problem": "Find $x$ if $2x=6$.",
                    "answer": "3",
                    "subject": "Prealgebra",
                },
            ]
        )
        loader = MATHLoader()
        samples = loader.load()

        mock_load_dataset.assert_called_once_with("HuggingFaceH4/MATH-500", split="test")
        assert len(samples) == 2
        assert samples[0].sample_id == "math_00000"
        assert samples[0].question == "Compute $1+1$."
        assert samples[0].correct_answer == "2"
        assert samples[0].choices is None
        assert samples[0].subset == "Algebra"

    @patch("typo_cot.data.loader.load_dataset")
    def test_create_loader_factory(self, mock_load_dataset: MagicMock) -> None:
        loader = create_loader("math")
        assert isinstance(loader, MATHLoader)


class TestMATHAnswerExtractor:
    """MATHAnswerExtractor (boxed 形式) のテスト."""

    def test_extract_simple_boxed(self) -> None:
        extractor = create_extractor("math")
        assert isinstance(extractor, MATHAnswerExtractor)
        result = extractor.extract("So we get $x=4$. The answer is \\boxed{4}.")
        assert result.extracted_answer == "4"
        assert result.extraction_method == "boxed"

    def test_extract_nested_braces(self) -> None:
        extractor = MATHAnswerExtractor()
        result = extractor.extract("Thus \\boxed{\\frac{\\pi}{2}} is the value.")
        assert result.extracted_answer == "\\frac{\\pi}{2}"

    def test_extract_last_boxed_wins(self) -> None:
        extractor = MATHAnswerExtractor()
        result = extractor.extract("First \\boxed{1}, but actually \\boxed{2}.")
        assert result.extracted_answer == "2"

    def test_is_correct_normalizes_latex(self) -> None:
        extractor = MATHAnswerExtractor()
        assert extractor.is_correct("\\dfrac{1}{2}", "\\frac{1}{2}")
        assert extractor.is_correct("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})")
        assert not extractor.is_correct("3", "4")

    def test_no_match(self) -> None:
        extractor = MATHAnswerExtractor()
        result = extractor.extract("I do not know.")
        assert result.extracted_answer == ""
        assert result.extraction_method == "no_match"

    def test_strict_mode_only_boxed(self) -> None:
        extractor = MATHAnswerExtractor()
        assert extractor.extract_strict("The answer is \\boxed{7}.") == "7"
        # "The answer is X" (boxedなし) は strict では拒否
        assert extractor.extract_strict("The answer is 7.") == ""


class TestMATHPromptTemplate:
    """MATHPromptTemplate のテスト."""

    def test_generate_tracks_question_span(self) -> None:
        template = create_prompt_template("math")
        assert isinstance(template, MATHPromptTemplate)
        q = "What is $7!/5!$?"
        result = template.generate(question=q)
        full = result.get_full_prompt()
        assert full[result.question_start_in_full : result.question_end_in_full] == q
        assert "\\boxed{}" in result.system_prompt
        assert result.question_with_choices_end == result.question_end_in_full

    def test_four_shot(self) -> None:
        template = MATHPromptTemplate()
        assert template.get_default_num_shots() == 4

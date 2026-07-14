"""データローダーモジュールのテスト."""

from unittest.mock import MagicMock, patch

import pytest

from typo_cot.data.loader import (
    GSM8KLoader,
    MMLULoader,
    MMLUProLoader,
    Sample,
    SQuADv2Loader,
    create_loader,
)


class TestSample:
    """Sampleデータクラスのテスト."""

    def test_sample_creation(self) -> None:
        """Sampleが正しく作成されることを確認."""
        sample = Sample(
            sample_id="test_001",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            correct_answer="B",
        )
        assert sample.sample_id == "test_001"
        assert sample.question == "What is 2+2?"
        assert sample.choices == ["3", "4", "5", "6"]
        assert sample.correct_answer == "B"
        assert sample.context is None
        assert sample.subset is None
        assert sample.answer_start is None
        assert sample.answer_end is None

    def test_sample_with_optional_fields(self) -> None:
        """オプションフィールドを含むSampleが正しく作成されることを確認."""
        sample = Sample(
            sample_id="squad_001",
            question="What is the capital?",
            choices=None,
            correct_answer="Tokyo",
            context="Japan is a country...",
            subset=None,
            answer_start=10,
            answer_end=15,
        )
        assert sample.context == "Japan is a country..."
        assert sample.answer_start == 10
        assert sample.answer_end == 15


class TestMMLULoader:
    """MMLULoaderのテスト."""

    def test_subsets_count(self) -> None:
        """57サブセットが定義されていることを確認."""
        assert len(MMLULoader.SUBSETS) == 57

    def test_get_subsets_default(self) -> None:
        """デフォルトで全サブセットが返されることを確認."""
        loader = MMLULoader()
        assert loader.get_subsets() == MMLULoader.SUBSETS

    def test_get_subsets_custom(self) -> None:
        """カスタムサブセットが正しく設定されることを確認."""
        custom_subsets = ["abstract_algebra", "anatomy"]
        loader = MMLULoader(subsets=custom_subsets)
        assert loader.get_subsets() == custom_subsets

    @patch("typo_cot.data.loader.load_dataset")
    def test_load_subset(self, mock_load_dataset: MagicMock) -> None:
        """サブセットの読み込みが正しく動作することを確認."""
        # モックデータセットの設定
        mock_dataset = [
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,  # B
            },
            {
                "question": "What is 3+3?",
                "choices": ["5", "6", "7", "8"],
                "answer": 1,  # B
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = MMLULoader(samples_per_subset=2, subsets=["abstract_algebra"])
        samples = loader.load_subset("abstract_algebra")

        assert len(samples) == 2
        assert samples[0].sample_id.startswith("mmlu_abstract_algebra_")
        assert samples[0].correct_answer == "B"
        assert samples[0].subset == "abstract_algebra"


class TestMMLUProLoader:
    """MMLUProLoaderのテスト."""

    def test_categories_count(self) -> None:
        """14カテゴリが定義されていることを確認."""
        assert len(MMLUProLoader.CATEGORIES) == 14

    def test_get_subsets_default(self) -> None:
        """デフォルトで全カテゴリが返されることを確認."""
        loader = MMLUProLoader()
        assert loader.get_subsets() == MMLUProLoader.CATEGORIES

    def test_get_subsets_custom(self) -> None:
        """カスタムカテゴリが正しく設定されることを確認."""
        custom_categories = ["math", "physics"]
        loader = MMLUProLoader(categories=custom_categories)
        assert loader.get_subsets() == custom_categories

    @patch("typo_cot.data.loader.load_dataset")
    def test_load(self, mock_load_dataset: MagicMock) -> None:
        """データの読み込みが正しく動作することを確認."""
        mock_dataset = [
            {
                "question": "What is the time complexity of binary search?",
                "options": ["O(1)", "O(n)", "O(log n)", "O(n log n)", "O(n²)"],
                "answer_index": 2,  # C
                "category": "computer science",
            },
            {
                "question": "What is 2+2?",
                "options": ["3", "4", "5", "6", "7"],
                "answer_index": 1,  # B
                "category": "math",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = MMLUProLoader(samples_per_category=10, categories=["computer science", "math"])
        samples = loader.load()

        assert len(samples) == 2
        # computer scienceのサンプルを確認
        cs_sample = next(s for s in samples if s.subset == "computer science")
        assert cs_sample.sample_id.startswith("mmlu_pro_computer_science_")
        assert cs_sample.correct_answer == "C"
        assert len(cs_sample.choices) == 5  # MMLU-Proは5個以上の選択肢

    @patch("typo_cot.data.loader.load_dataset")
    def test_load_ten_choices(self, mock_load_dataset: MagicMock) -> None:
        """10個の選択肢を持つ問題が正しく処理されることを確認."""
        mock_dataset = [
            {
                "question": "Test question?",
                "options": [f"Option {i}" for i in range(10)],
                "answer_index": 9,  # J
                "category": "other",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = MMLUProLoader(categories=["other"])
        samples = loader.load()

        assert len(samples) == 1
        assert len(samples[0].choices) == 10
        assert samples[0].correct_answer == "J"


class TestGSM8KLoader:
    """GSM8KLoaderのテスト."""

    def test_get_subsets(self) -> None:
        """GSM8Kにサブセットがないことを確認."""
        loader = GSM8KLoader()
        assert loader.get_subsets() == []

    @patch("typo_cot.data.loader.load_dataset")
    def test_load(self, mock_load_dataset: MagicMock) -> None:
        """データの読み込みが正しく動作することを確認."""
        mock_dataset = [
            {
                "question": "Janet has 10 apples...",
                "answer": "Janet has 10 apples. #### 5",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = GSM8KLoader()
        samples = loader.load()

        assert len(samples) == 1
        assert samples[0].sample_id == "gsm8k_00000"
        assert samples[0].correct_answer == "5"
        assert samples[0].choices is None


class TestSQuADv2Loader:
    """SQuADv2Loaderのテスト."""

    def test_get_subsets(self) -> None:
        """SQuAD v2にサブセットがないことを確認."""
        loader = SQuADv2Loader()
        assert loader.get_subsets() == []

    @patch("typo_cot.data.loader.load_dataset")
    def test_load(self, mock_load_dataset: MagicMock) -> None:
        """データの読み込みが正しく動作することを確認."""
        mock_dataset = [
            {
                "id": "test123",
                "question": "What is the capital of Japan?",
                "context": "Japan is a country in East Asia. Tokyo is its capital.",
                "answers": {"text": ["Tokyo"], "answer_start": [42]},
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = SQuADv2Loader()
        samples = loader.load()

        assert len(samples) == 1
        assert samples[0].sample_id == "squad_v2_test123"
        assert samples[0].correct_answer == "Tokyo"
        assert samples[0].context is not None

    @patch("typo_cot.data.loader.load_dataset")
    def test_load_with_answer_positions(self, mock_load_dataset: MagicMock) -> None:
        """回答位置情報が正しく保存されることを確認."""
        mock_dataset = [
            {
                "id": "test456",
                "question": "What is the capital?",
                "context": "The capital is Tokyo.",
                "answers": {"text": ["Tokyo"], "answer_start": [15]},
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = SQuADv2Loader()
        samples = loader.load()

        assert len(samples) == 1
        assert samples[0].answer_start == 15
        assert samples[0].answer_end == 15 + len("Tokyo")  # 20

    @patch("typo_cot.data.loader.load_dataset")
    def test_load_unanswerable(self, mock_load_dataset: MagicMock) -> None:
        """回答不可能な質問が正しく処理されることを確認."""
        mock_dataset = [
            {
                "id": "unanswerable123",
                "question": "What is the meaning of life?",
                "context": "This context doesn't contain the answer.",
                "answers": {"text": [], "answer_start": []},
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        loader = SQuADv2Loader()
        samples = loader.load()

        assert len(samples) == 1
        assert samples[0].correct_answer == ""  # 回答不可能
        assert samples[0].answer_start is None
        assert samples[0].answer_end is None


class TestCreateLoader:
    """create_loaderファクトリ関数のテスト."""

    def test_create_mmlu_loader(self) -> None:
        """MMLUローダーが作成されることを確認."""
        loader = create_loader("mmlu")
        assert isinstance(loader, MMLULoader)

    def test_create_mmlu_pro_loader(self) -> None:
        """MMLU-Proローダーが作成されることを確認."""
        loader = create_loader("mmlu_pro")
        assert isinstance(loader, MMLUProLoader)

    def test_create_gsm8k_loader(self) -> None:
        """GSM8Kローダーが作成されることを確認."""
        loader = create_loader("gsm8k")
        assert isinstance(loader, GSM8KLoader)

    def test_create_squad_v2_loader(self) -> None:
        """SQuAD v2ローダーが作成されることを確認."""
        loader = create_loader("squad_v2")
        assert isinstance(loader, SQuADv2Loader)

    def test_create_unknown_loader(self) -> None:
        """不明なベンチマーク名でエラーが発生することを確認."""
        with pytest.raises(ValueError, match="不明なベンチマーク"):
            create_loader("unknown_benchmark")

    def test_create_mmlu_with_options(self) -> None:
        """オプション付きでMMLUローダーが作成されることを確認."""
        loader = create_loader(
            "mmlu",
            split="validation",
            samples_per_subset=10,
            seed=123,
            subsets=["abstract_algebra"],
        )
        assert isinstance(loader, MMLULoader)
        assert loader.samples_per_subset == 10
        assert loader.seed == 123
        assert loader.subsets == ["abstract_algebra"]

    def test_create_mmlu_pro_with_options(self) -> None:
        """オプション付きでMMLU-Proローダーが作成されることを確認."""
        loader = create_loader(
            "mmlu_pro",
            split="validation",
            samples_per_subset=20,
            seed=456,
            subsets=["math", "physics"],
        )
        assert isinstance(loader, MMLUProLoader)
        assert loader.samples_per_category == 20
        assert loader.seed == 456
        assert loader.categories == ["math", "physics"]

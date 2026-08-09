"""AttnLRP分析モジュールのテスト."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from typo_cot.lrp.analyzer import (
    AttnLRPAnalyzer,
    ImportanceResult,
    WordScore,
    create_analyzer,
)


class TestWordScore:
    """WordScoreデータクラスのテスト."""

    def test_word_score_creation(self) -> None:
        """WordScoreが正しく作成されることを確認."""
        word_score = WordScore(
            word="important",
            score=0.85,
            token_indices=[1, 2],
            start_pos=10,
            end_pos=19,
        )

        assert word_score.word == "important"
        assert word_score.score == 0.85
        assert word_score.token_indices == [1, 2]
        assert word_score.start_pos == 10
        assert word_score.end_pos == 19

    def test_word_score_without_positions(self) -> None:
        """位置情報なしのWordScoreが正しく作成されることを確認."""
        word_score = WordScore(
            word="test",
            score=0.5,
            token_indices=[0],
        )

        assert word_score.start_pos is None
        assert word_score.end_pos is None


class TestImportanceResult:
    """ImportanceResultデータクラスのテスト."""

    def test_importance_result_creation(self) -> None:
        """ImportanceResultが正しく作成されることを確認."""
        word_scores = [
            WordScore(word="hello", score=0.8, token_indices=[1]),
            WordScore(word="world", score=0.6, token_indices=[2]),
        ]

        result = ImportanceResult(
            input_text="hello world",
            token_scores=[("hello", 0.8), ("world", 0.6)],
            word_scores=word_scores,
            top_k_words=word_scores[:1],
            raw_relevance=torch.tensor([0.0, 0.8, 0.6]),
        )

        assert result.input_text == "hello world"
        assert len(result.token_scores) == 2
        assert len(result.word_scores) == 2
        assert len(result.top_k_words) == 1


class TestAttnLRPAnalyzer:
    """AttnLRPAnalyzerのテスト."""

    def test_initialization(self) -> None:
        """初期化が正しく行われることを確認."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        analyzer = AttnLRPAnalyzer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            top_k=5,
            device=torch.device("cpu"),
        )

        assert analyzer.model == mock_model
        assert analyzer.tokenizer == mock_tokenizer
        assert analyzer.top_k == 5
        assert analyzer.device == torch.device("cpu")

    def test_tokens_to_words_basic(self) -> None:
        """トークンから単語への変換が正しく行われることを確認."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        analyzer = AttnLRPAnalyzer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            top_k=10,
            device=torch.device("cpu"),
        )

        tokens = ["<s>", " Hello", " world", "</s>"]
        relevance = torch.tensor([0.0, 0.8, 0.6, 0.0])

        word_scores = analyzer.tokens_to_words(tokens, relevance)

        assert len(word_scores) == 2
        assert word_scores[0].word == "Hello"
        assert word_scores[0].score == pytest.approx(0.8)
        assert word_scores[1].word == "world"
        assert word_scores[1].score == pytest.approx(0.6)

    def test_tokens_to_words_subwords(self) -> None:
        """サブワードトークンの結合が正しく行われることを確認."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        analyzer = AttnLRPAnalyzer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            top_k=10,
            device=torch.device("cpu"),
        )

        # "important" が "import" + "ant" に分割された場合
        tokens = ["<s>", " import", "ant", " word", "</s>"]
        relevance = torch.tensor([0.0, 0.5, 0.3, 0.2, 0.0])

        word_scores = analyzer.tokens_to_words(tokens, relevance)

        assert len(word_scores) == 2
        assert word_scores[0].word == "important"
        assert word_scores[0].score == pytest.approx(0.5 + 0.3)  # サブワードのスコアが合算
        assert word_scores[0].token_indices == [1, 2]

    @patch.object(AttnLRPAnalyzer, "compute_relevance")
    def test_analyze(self, mock_compute: MagicMock) -> None:
        """分析が正しく行われることを確認."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # トークナイザーのモック設定
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
        mock_tokenizer.decode.side_effect = lambda x: {
            (1,): "<s>",
            (2,): " Hello",
            (3,): " world",
            (4,): "</s>",
        }.get(tuple(x), "")

        # compute_relevanceのモック設定
        mock_compute.return_value = torch.tensor([0.0, 0.8, 0.6, 0.0])

        analyzer = AttnLRPAnalyzer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            top_k=2,
            device=torch.device("cpu"),
        )

        result = analyzer.analyze("Hello world")

        assert isinstance(result, ImportanceResult)
        assert result.input_text == "Hello world"
        assert len(result.top_k_words) <= 2


class TestCreateAnalyzer:
    """create_analyzerファクトリ関数のテスト."""

    def test_create_analyzer(self) -> None:
        """分析器が正しく作成されることを確認."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        analyzer = create_analyzer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            top_k=5,
            device=torch.device("cpu"),
        )

        assert isinstance(analyzer, AttnLRPAnalyzer)
        assert analyzer.top_k == 5

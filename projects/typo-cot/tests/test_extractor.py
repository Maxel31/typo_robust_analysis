"""回答抽出モジュールのテスト."""

import pytest

from typo_cot.evaluation.extractor import (
    ExtractionResult,
    GSM8KAnswerExtractor,
    MMLUAnswerExtractor,
    MMLUProAnswerExtractor,
    SQuADv2AnswerExtractor,
    create_extractor,
)


class TestExtractionResult:
    """ExtractionResultデータクラスのテスト."""

    def test_extraction_result_creation(self) -> None:
        """ExtractionResultが正しく作成されることを確認."""
        result = ExtractionResult(
            extracted_answer="42",
            raw_text="The answer is 42.",
            confidence=0.95,
            extraction_method="pattern_1",
        )

        assert result.extracted_answer == "42"
        assert result.raw_text == "The answer is 42."
        assert result.confidence == 0.95
        assert result.extraction_method == "pattern_1"


class TestGSM8KAnswerExtractor:
    """GSM8KAnswerExtractorのテスト."""

    def test_extract_standard_format(self) -> None:
        """標準形式の回答抽出を確認."""
        extractor = GSM8KAnswerExtractor()
        result = extractor.extract("Let me calculate. 5 + 6 = 11. The answer is 11.")

        assert result.extracted_answer == "11"
        assert result.confidence >= 0.8

    def test_extract_with_comma(self) -> None:
        """カンマ付き数値の抽出を確認."""
        extractor = GSM8KAnswerExtractor()
        result = extractor.extract("The total is 1,234. The answer is 1,234.")

        assert result.extracted_answer == "1234"

    def test_extract_negative_number(self) -> None:
        """負の数の抽出を確認."""
        extractor = GSM8KAnswerExtractor()
        result = extractor.extract("The answer is -15.")

        assert result.extracted_answer == "-15"

    def test_extract_decimal(self) -> None:
        """小数の抽出を確認."""
        extractor = GSM8KAnswerExtractor()
        result = extractor.extract("The answer is 3.14.")

        assert result.extracted_answer == "3.14"

    def test_extract_no_match(self) -> None:
        """マッチしない場合の処理を確認."""
        extractor = GSM8KAnswerExtractor()
        result = extractor.extract("I don't know the answer.")

        assert result.extracted_answer == ""
        assert result.confidence == 0.0

    def test_is_correct_exact_match(self) -> None:
        """完全一致の判定を確認."""
        extractor = GSM8KAnswerExtractor()

        assert extractor.is_correct("42", "42") is True
        assert extractor.is_correct("100", "100") is True

    def test_is_correct_with_decimal(self) -> None:
        """小数点を含む一致判定を確認."""
        extractor = GSM8KAnswerExtractor()

        assert extractor.is_correct("3.14", "3.14") is True
        assert extractor.is_correct("3.14", "3.15") is False


class TestMMLUAnswerExtractor:
    """MMLUAnswerExtractorのテスト."""

    def test_extract_standard_format(self) -> None:
        """標準形式の回答抽出を確認."""
        extractor = MMLUAnswerExtractor()
        result = extractor.extract("Let me think. The answer is (B).")

        assert result.extracted_answer == "B"
        assert result.confidence >= 0.8

    def test_extract_without_parentheses(self) -> None:
        """括弧なしの回答抽出を確認."""
        extractor = MMLUAnswerExtractor()
        result = extractor.extract("The answer is B.")

        assert result.extracted_answer == "B"

    def test_extract_lowercase(self) -> None:
        """小文字の回答抽出を確認."""
        extractor = MMLUAnswerExtractor()
        result = extractor.extract("the answer is c")

        assert result.extracted_answer == "C"

    def test_extract_last_letter(self) -> None:
        """最後の選択肢を抽出することを確認."""
        extractor = MMLUAnswerExtractor()
        result = extractor.extract("Option A is wrong, B is also wrong. C seems correct.")

        assert result.extracted_answer == "C"

    def test_extract_no_match(self) -> None:
        """マッチしない場合の処理を確認."""
        extractor = MMLUAnswerExtractor()
        result = extractor.extract("I cannot determine the answer.")

        assert result.extracted_answer == ""
        assert result.confidence == 0.0

    def test_is_correct(self) -> None:
        """正解判定を確認."""
        extractor = MMLUAnswerExtractor()

        assert extractor.is_correct("A", "A") is True
        assert extractor.is_correct("a", "A") is True
        assert extractor.is_correct("B", "A") is False


class TestMMLUProAnswerExtractor:
    """MMLUProAnswerExtractorのテスト."""

    def test_extract_extended_options(self) -> None:
        """拡張選択肢（A-J）の抽出を確認."""
        extractor = MMLUProAnswerExtractor()

        result = extractor.extract("The answer is (J).")
        assert result.extracted_answer == "J"

        result = extractor.extract("The answer is (E).")
        assert result.extracted_answer == "E"

    def test_extract_standard_format(self) -> None:
        """標準形式の回答抽出を確認."""
        extractor = MMLUProAnswerExtractor()
        result = extractor.extract("After analysis, the answer is (C).")

        assert result.extracted_answer == "C"


class TestSQuADv2AnswerExtractor:
    """SQuADv2AnswerExtractorのテスト."""

    def test_extract_simple_answer(self) -> None:
        """シンプルな回答の抽出を確認."""
        extractor = SQuADv2AnswerExtractor()
        result = extractor.extract("Tokyo")

        assert result.extracted_answer == "Tokyo"

    def test_extract_first_sentence(self) -> None:
        """最初の文の抽出を確認."""
        extractor = SQuADv2AnswerExtractor()
        result = extractor.extract("Tokyo is the capital. It is a large city.")

        assert result.extracted_answer == "Tokyo is the capital"

    def test_extract_unanswerable(self) -> None:
        """回答不可能パターンの検出を確認."""
        extractor = SQuADv2AnswerExtractor()

        result = extractor.extract("This question is unanswerable.")
        assert result.extracted_answer == ""

        result = extractor.extract("The answer cannot be answered from the context.")
        assert result.extracted_answer == ""

    def test_is_correct_exact(self) -> None:
        """完全一致の判定を確認."""
        extractor = SQuADv2AnswerExtractor()

        assert extractor.is_correct("Tokyo", "Tokyo") is True
        assert extractor.is_correct("tokyo", "Tokyo") is True  # 大文字小文字無視

    def test_is_correct_unanswerable(self) -> None:
        """回答不可能の判定を確認."""
        extractor = SQuADv2AnswerExtractor()

        assert extractor.is_correct("", "") is True  # 両方空
        assert extractor.is_correct("something", "") is False  # 回答ありだが正解は空

    def test_compute_f1(self) -> None:
        """F1スコアの計算を確認."""
        extractor = SQuADv2AnswerExtractor()

        # 完全一致
        assert extractor.compute_f1("Tokyo", "Tokyo") == 1.0

        # 部分一致
        f1 = extractor.compute_f1("the capital Tokyo", "Tokyo")
        assert 0.0 < f1 < 1.0

        # 不一致
        assert extractor.compute_f1("Osaka", "Tokyo") == 0.0

        # 回答不可能
        assert extractor.compute_f1("", "") == 1.0

    def test_compute_em(self) -> None:
        """EMスコアの計算を確認."""
        extractor = SQuADv2AnswerExtractor()

        # 完全一致
        assert extractor.compute_em("Tokyo", "Tokyo") == 1.0

        # 大文字小文字を無視
        assert extractor.compute_em("tokyo", "Tokyo") == 1.0

        # 冠詞を無視
        assert extractor.compute_em("the capital", "capital") == 1.0

        # 不一致
        assert extractor.compute_em("Osaka", "Tokyo") == 0.0

        # 回答不可能
        assert extractor.compute_em("", "") == 1.0
        assert extractor.compute_em("something", "") == 0.0

    def test_compute_scores(self) -> None:
        """EMとF1両方のスコア計算を確認."""
        extractor = SQuADv2AnswerExtractor()

        # 完全一致
        scores = extractor.compute_scores("Tokyo", "Tokyo")
        assert scores["em"] == 1.0
        assert scores["f1"] == 1.0

        # 部分一致（EMは0、F1は中間値）
        scores = extractor.compute_scores("the capital Tokyo", "Tokyo")
        assert scores["em"] == 0.0
        assert 0.0 < scores["f1"] < 1.0

        # 不一致
        scores = extractor.compute_scores("Osaka", "Tokyo")
        assert scores["em"] == 0.0
        assert scores["f1"] == 0.0


class TestCreateExtractor:
    """create_extractorファクトリ関数のテスト."""

    def test_create_mmlu_extractor(self) -> None:
        """MMLUの抽出器が作成されることを確認."""
        extractor = create_extractor("mmlu")
        assert isinstance(extractor, MMLUAnswerExtractor)

    def test_create_mmlu_pro_extractor(self) -> None:
        """MMLU-Proの抽出器が作成されることを確認."""
        extractor = create_extractor("mmlu_pro")
        assert isinstance(extractor, MMLUProAnswerExtractor)

    def test_create_gsm8k_extractor(self) -> None:
        """GSM8Kの抽出器が作成されることを確認."""
        extractor = create_extractor("gsm8k")
        assert isinstance(extractor, GSM8KAnswerExtractor)

    def test_create_squad_v2_extractor(self) -> None:
        """SQuAD v2の抽出器が作成されることを確認."""
        extractor = create_extractor("squad_v2")
        assert isinstance(extractor, SQuADv2AnswerExtractor)

    def test_create_unknown_extractor(self) -> None:
        """不明なベンチマーク名でエラーが発生することを確認."""
        with pytest.raises(ValueError, match="不明なベンチマーク"):
            create_extractor("unknown")

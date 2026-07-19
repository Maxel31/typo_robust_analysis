"""実験10③: R1蒸留(Qwen2トークナイザ)向け摂動データセット作成ポリシーのテスト.

背景: Qwen2 トークナイザは選択肢マーカー "(A)" を '(A' + ')' に分割するため、
共有 PerturbedDatasetCreator._should_skip_token のマーカー除外パターン
(閉じ括弧つき '(A)' / 単独 'A' / 括弧のみ) をすり抜け、'(A' 断片が
摂動標的に選ばれてしまう (mmlu LXT-4 で 5564/5700 サンプルが該当)。
アーカイブ (gemma/llama 系) では '(' と 'A' が別トークンで両方除外されるため
マーカー標的はゼロ = 摂動ポリシーは「選択肢マーカーは摂動しない」。
R1PerturbedDatasetCreator はこのポリシーを Qwen2 トークナイザでも保つ。
"""

import pytest

from typo_cot.perturbation.dataset import PerturbedDatasetCreator
from typo_cot.perturbation.r1_dataset import R1PerturbedDatasetCreator


def _skip(token: str) -> bool:
    """インスタンス生成なしで R1 の skip 判定を呼ぶ."""
    return R1PerturbedDatasetCreator._should_skip_token(
        object.__new__(R1PerturbedDatasetCreator), token
    )


def _skip_base(token: str) -> bool:
    return PerturbedDatasetCreator._should_skip_token(
        object.__new__(PerturbedDatasetCreator), token
    )


class TestR1MarkerFragmentSkip:
    """Qwen2 トークナイザで生じる選択肢マーカー断片の除外."""

    @pytest.mark.parametrize("token", ["(A", "(B", "(J", "(a", " (A", "(A "])
    def test_paren_letter_fragment_is_skipped(self, token: str) -> None:
        assert _skip(token) is True

    def test_base_creator_misses_fragment(self) -> None:
        # 共有クラスの現行挙動の回帰確認 (すり抜ける = 本サブクラスの存在理由)
        assert _skip_base("(A") is False

    @pytest.mark.parametrize(
        "token",
        ["(Apple", " Users", "(About", "A5", "(An", "(AB"],
    )
    def test_content_words_are_not_skipped(self, token: str) -> None:
        # 実単語や複数文字の括弧開始トークンは摂動対象のまま
        assert _skip(token) is False

    @pytest.mark.parametrize(
        "token",
        ["(A)", "A.", "B)", "A:", "A", "123", "3.14", "(", ")", "  "],
    )
    def test_base_skip_patterns_still_apply(self, token: str) -> None:
        # 共有クラスの既存除外パターンは継承される
        assert _skip(token) is True

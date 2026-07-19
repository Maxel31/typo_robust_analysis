"""実験5: 対応のある McNemar 検定とリスク差 CI のテスト (GPU 不要)."""

import math

import pytest

from typo_cot.analysis.matched_control import (
    mcnemar_exact_p,
    paired_condition_comparison,
)


class TestMcNemarExact:
    def test_symmetric_discordance_gives_p_one(self) -> None:
        assert mcnemar_exact_p(5, 5) == pytest.approx(1.0)

    def test_fully_asymmetric(self) -> None:
        # 二項両側検定: 2 * 0.5^10
        assert mcnemar_exact_p(0, 10) == pytest.approx(2 * 0.5**10)

    def test_no_discordance_gives_p_one(self) -> None:
        assert mcnemar_exact_p(0, 0) == pytest.approx(1.0)


class TestPairedComparison:
    def test_counts_and_risk_difference(self) -> None:
        # n=100: 両方正解 60, A のみ正解 10, B のみ正解 2, 両方不正解 28
        correct_a = [True] * 60 + [True] * 10 + [False] * 2 + [False] * 28
        correct_b = [True] * 60 + [False] * 10 + [True] * 2 + [False] * 28
        result = paired_condition_comparison(correct_a, correct_b)
        assert result["n"] == 100
        assert result["n01"] == 10  # A 正解 / B 不正解
        assert result["n10"] == 2  # A 不正解 / B 正解
        assert result["acc_a"] == pytest.approx(0.70)
        assert result["acc_b"] == pytest.approx(0.62)
        assert result["risk_diff"] == pytest.approx(0.08)

        # 対応のある Wald CI: se = sqrt(n01 + n10 - (n01-n10)^2/n) / n
        se = math.sqrt(10 + 2 - (10 - 2) ** 2 / 100) / 100
        assert result["risk_diff_ci95"][0] == pytest.approx(0.08 - 1.959963984540054 * se)
        assert result["risk_diff_ci95"][1] == pytest.approx(0.08 + 1.959963984540054 * se)

        # McNemar exact = binomtest(min, n01+n10, 0.5) 両側
        # p = 2 * sum_{i<=2} C(12,i) 0.5^12 = 2*(1+12+66)/4096
        assert result["mcnemar_p"] == pytest.approx(2 * (1 + 12 + 66) / 4096)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_condition_comparison([True], [True, False])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_condition_comparison([], [])

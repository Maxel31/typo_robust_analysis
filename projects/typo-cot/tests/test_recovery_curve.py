"""実験2 副実験: recovery_curve (p% prefix 強制→回復ジャンプ位置) のテスト.

計画 §4 実験2-2-4: セルC構成 (typo質問 + clean CoT 先頭 p% 強制、
p∈{0,25,50,75,100}) → 回復率のジャンプ位置と最上位 R_C 語の初出位置の
一致率を並べ替え検定。
"""

import pytest

from typo_cot.intervention.recovery_curve import (
    GRID,
    cut_prefix_by_fraction,
    find_jump,
    jump_match,
    match_rate_permutation_test,
    target_first_fraction,
)

COT = (
    "Janet's ducks lay 16 eggs per day. "
    "She eats 3 eggs for breakfast and bakes muffins. "
    "So she has 13 eggs left."
)


class TestCutPrefix:
    def test_p0_empty_and_p100_full(self):
        assert cut_prefix_by_fraction(COT, 0) == ""
        assert cut_prefix_by_fraction(COT, 100) == COT

    def test_cut_is_prefix_and_word_boundary(self):
        for p in (25, 50, 75):
            prefix = cut_prefix_by_fraction(COT, p)
            assert COT.startswith(prefix)
            assert len(prefix) <= len(COT) * p / 100 + 1
            # 語の途中で切らない (prefix 末尾は空白 or 空)
            assert prefix == "" or prefix[-1].isspace()

    def test_monotone_in_p(self):
        lens = [len(cut_prefix_by_fraction(COT, p)) for p in GRID]
        assert lens == sorted(lens)


class TestFindJump:
    def test_jump_in_middle(self):
        recovered = {0: False, 25: False, 50: True, 75: True, 100: True}
        assert find_jump(recovered) == (25, 50)

    def test_no_recovery_returns_none(self):
        assert find_jump({p: False for p in GRID}) is None

    def test_recovered_at_zero(self):
        recovered = {0: True, 25: True, 50: True, 75: True, 100: True}
        assert find_jump(recovered) == (None, 0)

    def test_first_true_wins_even_if_nonmonotone(self):
        recovered = {0: False, 25: True, 50: False, 75: True, 100: True}
        assert find_jump(recovered) == (0, 25)


class TestJumpMatch:
    def test_inside_interval(self):
        assert jump_match((25, 50), 0.40) is True

    def test_boundary_inclusive_upper_exclusive_lower(self):
        assert jump_match((25, 50), 0.50) is True
        assert jump_match((25, 50), 0.25) is False

    def test_zero_jump_never_matches(self):
        assert jump_match((None, 0), 0.10) is False


class TestTargetFirstFraction:
    def test_fraction_of_first_occurrence(self):
        frac = target_first_fraction(COT, "eggs")
        assert frac == pytest.approx(COT.index("eggs") / len(COT))

    def test_missing_word_returns_none(self):
        assert target_first_fraction(COT, "zebra") is None


class TestPermutationTest:
    def test_perfect_match_low_p(self):
        cases = [
            {
                "interval": (25, 50),
                "target_frac": 0.4,
                "candidate_fracs": [0.05, 0.1, 0.6, 0.7, 0.9, 0.95],
            }
            for _ in range(20)
        ]
        res = match_rate_permutation_test(cases, n_perm=500, seed=0)
        assert res["observed_match_rate"] == 1.0
        assert res["p_value"] < 0.05

    def test_random_target_high_p(self):
        # 標的位置が候補分布と同一なら有意にならない
        cases = [
            {
                "interval": (25, 50),
                "target_frac": 0.9,
                "candidate_fracs": [0.3, 0.4, 0.45, 0.9, 0.95],
            }
            for _ in range(20)
        ]
        res = match_rate_permutation_test(cases, n_perm=300, seed=0)
        assert res["p_value"] > 0.2

    def test_cases_without_jump_are_excluded(self):
        cases = [
            {"interval": None, "target_frac": 0.4, "candidate_fracs": [0.1]},
            {"interval": (25, 50), "target_frac": 0.4, "candidate_fracs": [0.1, 0.9]},
        ]
        res = match_rate_permutation_test(cases, n_perm=100, seed=0)
        assert res["n_cases"] == 1

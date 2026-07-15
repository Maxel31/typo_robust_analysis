"""実験2: deletion_stats (McNemar・リスク差CI・用量反応・腕別集計) のテスト.

共通規約 (§3.4-3): 対応比較 = McNemar + リスク差 CI / CI = bootstrap。
用量反応の単調性は並べ替え検定。集計は content / numeric 層を分離する。
"""

import pytest

from typo_cot.intervention.deletion_stats import (
    aggregate_results,
    dose_trend_test,
    mcnemar_exact,
    paired_risk_difference,
)


class TestMcNemar:
    def test_known_discordant_counts(self):
        a = [True, True, True, True, False, False, False, False]
        b = [True, False, False, False, False, False, False, True]
        res = mcnemar_exact(a, b)
        assert res["b"] == 3  # a のみ flip
        assert res["c"] == 1  # b のみ flip
        assert res["p_value"] == pytest.approx(0.625)

    def test_identical_lists(self):
        a = [True, False, True]
        res = mcnemar_exact(a, list(a))
        assert res["b"] == 0 and res["c"] == 0
        assert res["p_value"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_exact([True], [True, False])


class TestPairedRiskDifference:
    def test_point_estimate(self):
        a = [True] * 6 + [False] * 4
        b = [True] * 2 + [False] * 8
        res = paired_risk_difference(a, b, n_boot=200, seed=0)
        assert res["rd"] == pytest.approx(0.4)
        assert res["ci_low"] <= res["rd"] <= res["ci_high"]

    def test_degenerate_all_ones_vs_zeros(self):
        res = paired_risk_difference([True] * 10, [False] * 10, n_boot=100, seed=1)
        assert res["rd"] == 1.0
        assert res["ci_low"] == 1.0 and res["ci_high"] == 1.0

    def test_deterministic_with_seed(self):
        a = [True, False] * 10
        b = [False, False] * 10
        r1 = paired_risk_difference(a, b, n_boot=100, seed=7)
        r2 = paired_risk_difference(a, b, n_boot=100, seed=7)
        assert r1 == r2


class TestDoseTrend:
    def test_monotone_increase_significant(self):
        n = 40
        flips = {
            1: {f"s{i}": i < 4 for i in range(n)},
            2: {f"s{i}": i < 12 for i in range(n)},
            4: {f"s{i}": i < 28 for i in range(n)},
        }
        res = dose_trend_test(flips, n_perm=500, seed=0)
        assert res["slope"] > 0
        assert res["p_value"] < 0.05
        assert res["rates"][4] > res["rates"][1]

    def test_flat_not_significant(self):
        n = 30
        flips = {k: {f"s{i}": i < 10 for i in range(n)} for k in (1, 2, 4)}
        res = dose_trend_test(flips, n_perm=200, seed=0)
        assert res["p_value"] > 0.5


def _record(sample_id, clean_correct, arm_flips: dict, skip=None):
    arms = {}
    for name, flip in arm_flips.items():
        kind, op, k, stratum = {
            "top_rc_delete_k1": ("top_rc", "delete", 1, "content"),
            "matched_random_delete_k1": ("matched_random", "delete", 1, "content"),
            "numeric_top_rc_delete_k1": ("top_rc", "delete", 1, "numeric"),
        }[name]
        arms[name] = {
            "target_kind": kind, "op": op, "k": k, "stratum": stratum,
            "skip_reason": None, "flip": flip,
            "correct_to_incorrect": flip and clean_correct,
        }
    return {
        "sample_id": sample_id,
        "skip_reason": skip,
        "clean_correct": clean_correct,
        "residual_answer_in_prefix": False,
        "baseline": {"answer": "18", "matches_archive": True},
        "arms": arms,
    }


class TestAggregateResults:
    def _records(self):
        recs = []
        for i in range(10):
            recs.append(
                _record(
                    f"s{i}",
                    clean_correct=i < 8,  # 8 clean-correct / 2 not
                    arm_flips={
                        "top_rc_delete_k1": i % 2 == 0,        # 5/10 flip
                        "matched_random_delete_k1": i == 0,     # 1/10 flip
                        "numeric_top_rc_delete_k1": i < 9,      # 9/10 flip
                    },
                )
            )
        return recs

    def test_strata_are_separated(self):
        summary = aggregate_results(self._records())
        assert "content" in summary["strata"] and "numeric" in summary["strata"]
        assert "top_rc_delete_k1" in summary["strata"]["content"]["arms"]
        assert "numeric_top_rc_delete_k1" in summary["strata"]["numeric"]["arms"]
        assert "numeric_top_rc_delete_k1" not in summary["strata"]["content"]["arms"]

    def test_main_estimand_conditions_on_clean_correct(self):
        summary = aggregate_results(self._records())
        arm = summary["strata"]["content"]["arms"]["top_rc_delete_k1"]
        # clean-correct 8 サンプル中 flip は i∈{0,2,4,6} の 4 → 0.5
        assert arm["n"] == 8
        assert arm["flip_rate"] == pytest.approx(0.5)
        # 全サンプル版は副次
        assert arm["flip_rate_all"] == pytest.approx(0.5)

    def test_core_contrast_is_computed(self):
        summary = aggregate_results(self._records())
        contrasts = summary["contrasts"]
        assert len(contrasts) >= 1
        core = next(
            c
            for c in contrasts
            if c["arm_a"] == "top_rc_delete_k1" and c["arm_b"] == "matched_random_delete_k1"
        )
        assert core["risk_difference"] > 0
        assert 0 <= core["mcnemar_p"] <= 1

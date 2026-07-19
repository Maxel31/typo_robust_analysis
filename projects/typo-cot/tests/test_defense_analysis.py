"""実験7: 校正後評価の集計 (defense/analysis.py) のユニットテスト.

analyze_spellfix.py の集計ロジック (flip サブセット / R_Q 偏在 Mann-Whitney)
をライブラリ化したものの検証。GPU 不要・合成レコードのみ。
"""

import pytest

from typo_cot.defense.analysis import (
    flip_stats,
    restoration_subsets,
    token_rq_comparison,
)


def make_row(
    sample_id="s0",
    fully_restored=False,
    all_perturbed_restored=False,
    n_collateral=0,
    flip_perturbed=False,
    flip_corrected=False,
):
    return {
        "sample_id": sample_id,
        "fully_restored": fully_restored,
        "all_perturbed_restored": all_perturbed_restored,
        "n_collateral": n_collateral,
        "flip_perturbed": flip_perturbed,
        "flip_corrected": flip_corrected,
    }


class TestFlipStats:
    def test_counts_and_rates(self):
        rows = [
            make_row(flip_perturbed=True, flip_corrected=False),
            make_row(flip_perturbed=True, flip_corrected=True),
            make_row(flip_perturbed=False, flip_corrected=False),
            make_row(flip_perturbed=False, flip_corrected=False),
        ]
        st = flip_stats(rows)
        assert st["n"] == 4
        assert st["flips_perturbed"] == 2
        assert st["flips_corrected"] == 1
        assert st["flip_rate_perturbed"] == pytest.approx(0.5)
        assert st["flip_rate_corrected"] == pytest.approx(0.25)

    def test_empty_rows(self):
        st = flip_stats([])
        assert st["n"] == 0
        assert st["flip_rate_perturbed"] is None


class TestRestorationSubsets:
    def test_subset_partition(self):
        rows = [
            make_row("a", fully_restored=True, all_perturbed_restored=True),
            make_row("b", all_perturbed_restored=True, n_collateral=1),
            make_row("c"),  # 復元失敗
        ]
        subsets = restoration_subsets(rows)
        assert subsets["all"]["n"] == 3
        assert subsets["fully_restored"]["n"] == 1
        assert subsets["all_perturbed_restored_not_full"]["n"] == 1
        assert subsets["partially_or_not_restored"]["n"] == 1
        assert subsets["all_restored_with_collateral"]["n"] == 1


class TestTokenRQComparison:
    def test_mannwhitney_and_means(self):
        records = (
            [{"sample_id": "a", "importance_score": 1.0, "restored": False}] * 6
            + [{"sample_id": "a", "importance_score": 0.1, "restored": True}] * 6
        )
        out = token_rq_comparison(records)
        assert out["n_failed"] == 6
        assert out["n_restored"] == 6
        assert out["mean_rq_failed"] == pytest.approx(1.0)
        assert out["mean_rq_restored"] == pytest.approx(0.1)
        assert out["restoration_rate"] == pytest.approx(0.5)
        assert out["mannwhitney_p"] is not None
        assert 0.0 <= out["mannwhitney_p"] <= 1.0

    def test_empty_group_gives_none_p(self):
        records = [
            {"sample_id": "a", "importance_score": 0.5, "restored": True},
        ]
        out = token_rq_comparison(records)
        assert out["mannwhitney_p"] is None
        assert out["mean_rq_failed"] is None

    def test_fail_rate_by_rank(self):
        # サンプル内 R_Q 降順ランク: rank1 が最重要
        records = [
            {"sample_id": "a", "importance_score": 0.9, "restored": False},
            {"sample_id": "a", "importance_score": 0.1, "restored": True},
            {"sample_id": "b", "importance_score": 0.8, "restored": False},
            {"sample_id": "b", "importance_score": 0.2, "restored": True},
        ]
        out = token_rq_comparison(records)
        by_rank = out["fail_rate_by_within_sample_rank"]
        assert by_rank["rank1"]["fail_rate"] == pytest.approx(1.0)
        assert by_rank["rank2"]["fail_rate"] == pytest.approx(0.0)

    def test_no_records(self):
        out = token_rq_comparison([])
        assert out["n_matched_tokens"] == 0
        assert out["restoration_rate"] is None

"""実験7 本番シャード実行のマージロジックのテスト (GPU 不要)."""

import pytest

from typo_cot.defense.sharding import (
    merge_correction_shards,
    merge_generation_results,
    shard_ranges,
)


class TestShardRanges:
    def test_exact_division(self):
        assert shard_ranges(100, 50) == [(0, 50), (50, 100)]

    def test_remainder_goes_to_last_shard(self):
        assert shard_ranges(110, 50) == [(0, 50), (50, 100), (100, 110)]

    def test_single_shard_when_smaller_than_size(self):
        assert shard_ranges(30, 50) == [(0, 30)]

    def test_empty(self):
        assert shard_ranges(0, 50) == []


def _shard(start, end, prefix):
    n = end - start
    return {
        "start": start,
        "end": end,
        "samples": [
            {"sample_id": f"{prefix}_{i}", "perturbed_question": f"q{i}"}
            for i in range(start, end)
        ],
        "per_sample": [
            {"sample_id": f"{prefix}_{i}", "fully_restored": i % 2 == 0}
            for i in range(start, end)
        ],
        "aggregate": {
            "n_samples": n,
            "word_total": 4 * n,
            "word_restored": 2 * n,
            "fully_restored": sum(1 for i in range(start, end) if i % 2 == 0),
            "perturbed_words_all_restored": 0,
            "collateral_changes": n,
            "unalignable": 0,
            "llm_parse_failures": 0,
        },
    }


class TestMergeCorrectionShards:
    def test_merges_in_order_and_sums_aggregate(self):
        merged = merge_correction_shards([_shard(2, 4, "s"), _shard(0, 2, "s")])
        assert [s["sample_id"] for s in merged["samples"]] == [
            "s_0", "s_1", "s_2", "s_3",
        ]
        assert merged["aggregate"]["n_samples"] == 4
        assert merged["aggregate"]["word_total"] == 16
        assert merged["aggregate"]["word_restored"] == 8
        assert merged["rates"]["word_restoration_rate"] == pytest.approx(0.5)
        assert merged["rates"]["full_restoration_rate"] == pytest.approx(0.5)
        assert len(merged["per_sample"]) == 4

    def test_rejects_gap_between_shards(self):
        with pytest.raises(ValueError, match="連続していません"):
            merge_correction_shards([_shard(0, 2, "s"), _shard(3, 5, "s")])

    def test_rejects_overlap(self):
        with pytest.raises(ValueError, match="連続していません"):
            merge_correction_shards([_shard(0, 3, "s"), _shard(2, 5, "s")])

    def test_rejects_nonzero_first_start(self):
        with pytest.raises(ValueError, match="先頭"):
            merge_correction_shards([_shard(1, 3, "s")])


class TestMergeGenerationResults:
    def test_concatenates_and_recomputes_metrics(self):
        shard_a = {
            "start": 0,
            "end": 2,
            "results": [
                {"sample_id": "x_0", "is_correct": True, "subset": "algebra"},
                {"sample_id": "x_1", "is_correct": False, "subset": "algebra"},
            ],
        }
        shard_b = {
            "start": 2,
            "end": 3,
            "results": [
                {"sample_id": "x_2", "is_correct": True, "subset": "geometry"},
            ],
        }
        results, summary = merge_generation_results([shard_b, shard_a])
        assert [r["sample_id"] for r in results] == ["x_0", "x_1", "x_2"]
        assert summary["overall_metrics"]["accuracy"] == pytest.approx(2 / 3)
        assert summary["overall_metrics"]["total_correct"] == 2
        assert summary["overall_metrics"]["total_samples"] == 3
        assert summary["per_subset_metrics"]["algebra"]["accuracy"] == pytest.approx(0.5)
        assert summary["per_subset_metrics"]["geometry"]["total"] == 1

    def test_rejects_gap(self):
        a = {"start": 0, "end": 1, "results": [
            {"sample_id": "x_0", "is_correct": True, "subset": "s"}]}
        b = {"start": 2, "end": 3, "results": [
            {"sample_id": "x_2", "is_correct": True, "subset": "s"}]}
        with pytest.raises(ValueError, match="連続していません"):
            merge_generation_results([a, b])

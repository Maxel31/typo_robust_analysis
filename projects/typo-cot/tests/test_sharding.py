"""シャード分割/統合ヘルパー (実験10②: MATH-500 再生成) のテスト.

run_inference.py のシャード化(--start/--end/--merge)で使う共通ロジックを
src/typo_cot/sharding.py に置き、ここで挙動を凍結する。
- シャードファイルの命名規約は run_inference_reasoning.py と同一
  (shards/results_{start:05d}_{end:05d}.json、サンプルは後勝ちで統合)。
- summary スキーマはアーカイブ outputs/baseline/*_math/summary.json 互換。
"""

import json
from pathlib import Path

from typo_cot.sharding import (
    build_summary_from_results,
    load_shard_rows,
    merge_shard_results,
    shard_results_path,
)


def _write_shard(output_dir: Path, start: int, end: int, rows: list[dict]) -> Path:
    path = shard_results_path(output_dir, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return path


class TestShardResultsPath:
    def test_naming_convention(self, tmp_path: Path) -> None:
        path = shard_results_path(tmp_path, 0, 250)
        assert path == tmp_path / "shards" / "results_00000_00250.json"

    def test_five_digit_padding(self, tmp_path: Path) -> None:
        path = shard_results_path(tmp_path, 250, 500)
        assert path.name == "results_00250_00500.json"


class TestLoadShardRows:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_shard_rows(tmp_path / "shards" / "results_00000_00250.json") == []

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{ not json", encoding="utf-8")
        assert load_shard_rows(path) == []

    def test_loads_rows(self, tmp_path: Path) -> None:
        rows = [{"sample_id": "math_00000"}]
        path = _write_shard(tmp_path, 0, 1, rows)
        assert load_shard_rows(path) == rows


class TestMergeShardResults:
    def test_merges_and_sorts_by_sample_id(self, tmp_path: Path) -> None:
        _write_shard(
            tmp_path,
            250,
            500,
            [{"sample_id": "math_00251", "is_correct": True}],
        )
        _write_shard(
            tmp_path,
            0,
            250,
            [
                {"sample_id": "math_00001", "is_correct": False},
                {"sample_id": "math_00000", "is_correct": True},
            ],
        )
        results, covered = merge_shard_results(tmp_path)
        assert [r["sample_id"] for r in results] == [
            "math_00000",
            "math_00001",
            "math_00251",
        ]
        assert covered == [(0, 250), (250, 500)]

    def test_later_shard_file_wins_on_duplicate(self, tmp_path: Path) -> None:
        # ファイル名昇順で後のシャードが同一 sample_id を上書きする(再実行シャード優先)
        _write_shard(tmp_path, 0, 2, [{"sample_id": "math_00000", "is_correct": False}])
        _write_shard(tmp_path, 1, 2, [{"sample_id": "math_00000", "is_correct": True}])
        results, _ = merge_shard_results(tmp_path)
        assert len(results) == 1
        assert results[0]["is_correct"] is True

    def test_no_shards_returns_empty(self, tmp_path: Path) -> None:
        results, covered = merge_shard_results(tmp_path)
        assert results == []
        assert covered == []


class TestBuildSummaryFromResults:
    def test_archive_compatible_schema(self) -> None:
        results = [
            {"sample_id": "math_00000", "is_correct": True, "subset": "Algebra"},
            {"sample_id": "math_00001", "is_correct": False, "subset": "Algebra"},
            {"sample_id": "math_00002", "is_correct": True, "subset": "Geometry"},
        ]
        summary = build_summary_from_results(
            model="google/gemma-3-1b-it",
            benchmark="math",
            results=results,
            num_samples_per_subset=None,
            batch_size=1,
            merged_shards=[(0, 2), (2, 3)],
        )
        info = summary["experiment_info"]
        assert info["model"] == "google/gemma-3-1b-it"
        assert info["benchmark"] == "math"
        assert info["num_samples_per_subset"] is None
        assert info["batch_size"] == 1
        assert info["total_samples"] == 3
        assert "timestamp" in info
        assert info["merged_shards"] == [[0, 2], [2, 3]]

        overall = summary["overall_metrics"]
        assert overall["accuracy"] == 2 / 3
        assert overall["total_correct"] == 2
        assert overall["total_samples"] == 3

        per_subset = summary["per_subset_metrics"]
        assert per_subset["Algebra"] == {"accuracy": 0.5, "correct": 1, "total": 2}
        assert per_subset["Geometry"] == {"accuracy": 1.0, "correct": 1, "total": 1}

    def test_missing_subset_falls_back_to_default(self) -> None:
        summary = build_summary_from_results(
            model="m",
            benchmark="math",
            results=[{"sample_id": "math_00000", "is_correct": True}],
        )
        assert summary["per_subset_metrics"]["default"]["total"] == 1

    def test_empty_results(self) -> None:
        summary = build_summary_from_results(model="m", benchmark="math", results=[])
        assert summary["overall_metrics"]["accuracy"] == 0
        assert summary["overall_metrics"]["total_samples"] == 0

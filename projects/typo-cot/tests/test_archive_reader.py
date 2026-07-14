"""アーカイブ読み取り層 (archive_reader) のテスト.

tmp_path にアーカイブと同じディレクトリ構造の小さな JSON を作って検証する。
実アーカイブ・GPU は不要。
"""

from __future__ import annotations

import hashlib
import json

import pytest

from typo_cot.data.archive_reader import (
    analysis_condition_dir,
    baseline_dir,
    load_analysis_sample_results,
    load_json,
    perturbed_dir,
    sha256_file,
)


@pytest.fixture()
def archive(tmp_path):
    """アーカイブ構造のミニチュアを作る."""
    base = tmp_path / "outputs" / "baseline" / "gemma-3-4b-it_gsm8k"
    base.mkdir(parents=True)
    (base / "results.json").write_text(
        json.dumps([{"sample_id": "gsm8k_00000", "generated_text": "The answer is 2."}]),
        encoding="utf-8",
    )
    pert = tmp_path / "outputs" / "perturbed" / "gemma-3-4b-it_gsm8k_k4_importance"
    pert.mkdir(parents=True)
    (pert / "results.json").write_text("[]", encoding="utf-8")
    ana = tmp_path / "outputs" / "analysis" / "gsm8k" / "gemma-3-4b-it" / "k4_importance"
    ana.mkdir(parents=True)
    (ana / "full_results.json").write_text(
        json.dumps(
            {
                "metadata": {"total_samples": 1},
                "sample_results": [
                    {
                        "sample_id": "gsm8k_00000",
                        "pattern": "correct→correct",
                        "answer_changed": False,
                        "cot_metrics": {
                            "rouge_l": {"f1": 0.9},
                            "jaccard": {"top10": 0.5},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "outputs"


class TestPaths:
    def test_baseline_dir(self, archive):
        assert baseline_dir(archive, "gemma-3-4b-it", "gsm8k").name == "gemma-3-4b-it_gsm8k"

    def test_perturbed_dir_uses_condition_suffix(self, archive):
        p = perturbed_dir(archive, "gemma-3-4b-it", "gsm8k", "lxt4")
        assert p.name == "gemma-3-4b-it_gsm8k_k4_importance"
        assert p.exists()

    def test_analysis_condition_dir(self, archive):
        p = analysis_condition_dir(archive / "analysis", "gemma-3-4b-it", "gsm8k", "lxt4")
        assert p == archive / "analysis" / "gsm8k" / "gemma-3-4b-it" / "k4_importance"
        assert p.exists()

    def test_unknown_condition_raises(self, archive):
        with pytest.raises(KeyError):
            perturbed_dir(archive, "gemma-3-4b-it", "gsm8k", "clean")


class TestLoading:
    def test_load_json(self, archive):
        data = load_json(archive / "baseline" / "gemma-3-4b-it_gsm8k" / "results.json")
        assert data[0]["sample_id"] == "gsm8k_00000"

    def test_load_analysis_sample_results(self, archive):
        srs = load_analysis_sample_results(archive / "analysis", "gemma-3-4b-it", "gsm8k", "lxt4")
        assert len(srs) == 1
        assert srs[0]["sample_id"] == "gsm8k_00000"

    def test_load_analysis_missing_returns_none(self, archive):
        assert (
            load_analysis_sample_results(archive / "analysis", "gemma-3-4b-it", "gsm8k", "lxt8")
            is None
        )

    def test_sha256_file(self, archive):
        path = archive / "baseline" / "gemma-3-4b-it_gsm8k" / "results.json"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha256_file(path) == expected

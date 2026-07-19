"""アーカイブ読み取り層 (archive_reader) のテスト.

tmp_path にアーカイブと同じディレクトリ構造の小さな JSON を作って検証する。
実アーカイブ・GPU は不要。
"""

from __future__ import annotations

import hashlib
import json

import pytest

from pathlib import Path

from typo_cot.data.archive_reader import (
    QWEN_MODEL,
    R1_BENCHMARKS,
    R1_MODEL,
    V1_BENCHMARKS,
    V1_MODELS,
    analysis_condition_dir,
    baseline_dir,
    build_cell_plan,
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


def _plan_registry() -> dict:
    benches = ("gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math")
    return {
        "prompts": {b: {"prompt_id": f"{b}_cot_v1"} for b in benches},
        "reasoning_prompts": {
            b: {"prompt_id": f"{b}_r1_think_v1"} for b in ("gsm8k", "math", "mmlu")
        },
    }


def _plan_paths() -> dict:
    return {
        "archive_outputs": "/arc/outputs",
        "archive_analysis": "/arc/outputs/analysis",
        "exp10_outputs": "/exp10/outputs",
    }


class TestCellPlan:
    """build_cell_plan: 217 セル (v1 150 + anti 25 + math 18 + Qwen 15 + R1 9)."""

    @pytest.fixture(scope="class")
    def plan(self) -> list[dict]:
        return build_cell_plan(_plan_paths(), _plan_registry())

    def _cell(self, plan, model, bench, cond) -> dict:
        hits = [
            c
            for c in plan
            if (c["model"], c["benchmark"], c["condition"]) == (model, bench, cond)
        ]
        assert len(hits) == 1, (model, bench, cond, len(hits))
        return hits[0]

    def test_total_cells_and_unique_keys(self, plan):
        keys = {(c["model"], c["benchmark"], c["condition"]) for c in plan}
        assert len(keys) == len(plan) == 217

    def test_v1_cell_uses_archive_and_analysis(self, plan):
        c = self._cell(plan, "gemma-3-4b-it", "gsm8k", "lxt4")
        assert c["baseline_path"] == Path(
            "/arc/outputs/baseline/gemma-3-4b-it_gsm8k/results.json"
        )
        assert c["perturbed_path"] == Path(
            "/arc/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance/results.json"
        )
        assert c["analysis_root"] == Path("/arc/outputs/analysis")
        assert c["prompt_id"] == "gsm8k_cot_v1"

    def test_anti_lxt4_from_archive_without_analysis(self, plan):
        c = self._cell(plan, "gemma-3-4b-it", "gsm8k", "anti_lxt4")
        assert c["perturbed_path"] == Path(
            "/arc/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_bottom_k/results.json"
        )
        assert c["analysis_root"] is None
        # anti_lxt4 は v1 25 設定のみ (Qwen / R1 には無い)
        assert all(
            c["model"] in V1_MODELS and c["benchmark"] in V1_BENCHMARKS
            for c in plan
            if c["condition"] == "anti_lxt4"
        )

    def test_math_regen_from_exp10(self, plan):
        c = self._cell(plan, "gemma-3-4b-it", "math", "lxt4")
        assert c["baseline_path"] == Path(
            "/exp10/outputs/baseline/gemma-3-4b-it_math/results.json"
        )
        assert c["perturbed_path"] == Path(
            "/exp10/outputs/perturbed/gemma-3-4b-it_math_k4_importance/results.json"
        )
        assert c["analysis_root"] is None
        assert c["prompt_id"] == "math_cot_v1"
        # math は 3 条件のみ (lxt1 等は存在しない)
        math_conds = {c["condition"] for c in plan if c["benchmark"] == "math"}
        assert math_conds == {"clean", "lxt4", "random4"}

    def test_qwen_clean_from_archive_perturbed_from_exp10(self, plan):
        clean = self._cell(plan, QWEN_MODEL, "arc", "clean")
        assert clean["baseline_path"] == Path(
            "/arc/outputs/baseline/Qwen2.5-7B-Instruct_arc/results.json"
        )
        pert = self._cell(plan, QWEN_MODEL, "arc", "lxt4")
        assert pert["baseline_path"] == clean["baseline_path"]
        assert pert["perturbed_path"] == Path(
            "/exp10/outputs/perturbed/Qwen2.5-7B-Instruct_arc_k4_importance/results.json"
        )
        assert pert["analysis_root"] is None
        # Qwen math は exp10 の baseline
        qmath = self._cell(plan, QWEN_MODEL, "math", "clean")
        assert qmath["baseline_path"] == Path(
            "/exp10/outputs/baseline/Qwen2.5-7B-Instruct_math/results.json"
        )

    def test_r1_cells(self, plan):
        r1 = [c for c in plan if c["model"] == R1_MODEL]
        assert {c["benchmark"] for c in r1} == set(R1_BENCHMARKS)
        assert len(r1) == 9
        c = self._cell(plan, R1_MODEL, "gsm8k", "clean")
        assert c["baseline_path"] == Path(
            "/exp10/outputs/baseline/DeepSeek-R1-Distill-Qwen-7B_gsm8k/results.json"
        )
        assert c["prompt_id"] == "gsm8k_r1_think_v1"
        assert c["analysis_root"] is None

"""実験4: fixed-target 帰属のコアロジック (GPU 不要部分) のテスト.

scripts/rebuttal/run_fixed_target_attribution.py の実装済み参照と同一の規約
(回答パターン検出・splice・統計カウント) をモジュール化した
typo_cot.attribution.fixed_target を検証する。
"""

import json
from pathlib import Path

import pytest

from typo_cot.attribution.fixed_target import (
    ANSWER_PATTERNS,
    SplicePlan,
    find_answer_match,
    plan_run,
    plan_splice,
    top_k_token_set,
)


class TestFindAnswerMatch:
    """回答パターン検出 (rebuttal 実装・lrp/analyzer._find_answer_pattern と同一規約)."""

    def test_choice_paren(self):
        m, ptype = find_answer_match("Reasoning... The answer is (B).")
        assert ptype == "choice"
        assert m.group(1) == "B"

    def test_choice_without_paren(self):
        m, ptype = find_answer_match("So the answer is C.")
        assert ptype == "choice"
        assert m.group(1) == "C"

    def test_number_gsm8k(self):
        m, ptype = find_answer_match("She has 18 left. The answer is 18")
        assert ptype == "number"
        assert m.group(1) == "18"

    def test_number_gsm8k_hash_format(self):
        m, ptype = find_answer_match("blah blah\n#### 42")
        assert ptype == "number"
        assert m.group(1) == "42"

    def test_last_match_wins(self):
        """同一パターンが複数回出た場合は最後のマッチを採用 (rebuttal 規約)."""
        text = "The answer is (A). Wait, reconsider. The answer is (D)."
        m, _ = find_answer_match(text)
        assert m.group(1) == "D"

    def test_no_match(self):
        m, ptype = find_answer_match("no final answer here")
        assert m is None
        assert ptype is None

    def test_patterns_match_rebuttal_reference(self):
        """モジュール定数が rebuttal スクリプトの ANSWER_PATTERNS と一致すること."""
        import importlib.util

        script = (
            Path(__file__).parent.parent
            / "scripts"
            / "rebuttal"
            / "run_fixed_target_attribution.py"
        )
        spec = importlib.util.spec_from_file_location("rebuttal_ft", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert ANSWER_PATTERNS == mod.ANSWER_PATTERNS


class TestPlanSplice:
    """splice 計画 (flip 判定と固定ターゲットテキスト構築)."""

    def test_flip_choice_is_spliced(self):
        base = "Reasoning. The answer is (A)."
        pert = "Other reasoning. The answer is (B)."
        plan = plan_splice(base, pert, sample_id="s1")
        assert isinstance(plan, SplicePlan)
        assert plan.skip_reason is None
        assert plan.spliced is True
        assert plan.baseline_answer == "A"
        assert plan.perturbed_answer == "B"
        # 摂動側テキストの回答文字だけが元回答に置換される
        assert plan.spliced_text == "Other reasoning. The answer is (A)."

    def test_non_flip_identical_text(self):
        base = "Reasoning. The answer is (A)."
        pert = "Other reasoning. The answer is (A)."
        plan = plan_splice(base, pert, sample_id="s2")
        assert plan.spliced is False
        assert plan.spliced_text == pert

    def test_number_flip_multi_char(self):
        base = "Compute. The answer is 120"
        pert = "Compute. The answer is 7"
        plan = plan_splice(base, pert, sample_id="s3")
        assert plan.spliced is True
        assert plan.spliced_text == "Compute. The answer is 120"

    def test_skip_no_baseline_answer(self):
        plan = plan_splice("no answer", "The answer is (A).", sample_id="s4")
        assert plan.skip_reason == "no_baseline_answer_pattern"

    def test_skip_no_perturbed_answer(self):
        plan = plan_splice("The answer is (A).", "no answer", sample_id="s5")
        assert plan.skip_reason == "no_perturbed_answer_pattern"

    def test_flip_detection_uses_raw_span_not_upper(self):
        """flip 判定は生成テキスト中の生文字列比較 (rebuttal 規約: 大文字化しない)."""
        base = "The answer is (a)."
        pert = "The answer is (A)."
        plan = plan_splice(base, pert, sample_id="s6")
        # "a" != "A" なので splice される (rebuttal 実装と同値の挙動)
        assert plan.spliced is True
        assert plan.spliced_text == "The answer is (a)."


class TestPlanRun:
    """run 全体の計画と統計カウント (rebuttal fixed_target_stats.json と同キー)."""

    def _baseline(self):
        return {
            "flip": {"sample_id": "flip", "generated_text": "The answer is (A)."},
            "same": {"sample_id": "same", "generated_text": "The answer is (B)."},
            "nobase": {"sample_id": "nobase", "generated_text": "unfinished"},
            "nopert": {"sample_id": "nopert", "generated_text": "The answer is (C)."},
        }

    def _perturbed(self):
        return [
            {"sample_id": "flip", "generated_text": "The answer is (D)."},
            {"sample_id": "same", "generated_text": "The answer is (B)."},
            {"sample_id": "nobase", "generated_text": "The answer is (A)."},
            {"sample_id": "nopert", "generated_text": "unfinished"},
            {"sample_id": "orphan", "generated_text": "The answer is (A)."},
        ]

    def test_stats_counts(self):
        plans, stats = plan_run(self._baseline(), self._perturbed())
        assert stats["total"] == 5
        assert stats["processed"] == 2
        assert stats["spliced"] == 1
        assert stats["identical"] == 1
        assert stats["skip_no_baseline"] == 1
        assert stats["skip_no_base_answer"] == 1
        assert stats["skip_no_pert_answer"] == 1
        assert stats["errors"] == 0

    def test_plans_only_processed(self):
        plans, _ = plan_run(self._baseline(), self._perturbed())
        assert sorted(p.sample_id for p in plans) == ["flip", "same"]

    def test_skipped_ids_reasons(self):
        _, stats = plan_run(self._baseline(), self._perturbed())
        assert stats["skipped_ids"]["orphan"] == "no_baseline"
        assert stats["skipped_ids"]["nobase"] == "no_baseline_answer_pattern"
        assert stats["skipped_ids"]["nopert"] == "no_perturbed_answer_pattern"

    def test_limit(self):
        plans, stats = plan_run(self._baseline(), self._perturbed(), limit=2)
        assert stats["total"] == 2

    def test_sample_ids_filter(self):
        plans, stats = plan_run(
            self._baseline(), self._perturbed(), sample_ids={"flip"}
        )
        assert stats["total"] == 1
        assert plans[0].sample_id == "flip"


class TestTopKTokenSet:
    """rebuttal top_k_token_set と同じ規約 (トークン文字列 dedup, 最大スコア採用)."""

    def test_dedup_max(self):
        scores = [("a", 1.0), ("b", 5.0), ("a", 3.0), ("c", 2.0)]
        assert top_k_token_set(scores, 2) == {"b", "a"}

    def test_k_larger_than_unique(self):
        scores = [("a", 1.0), ("b", 2.0)]
        assert top_k_token_set(scores, 10) == {"a", "b"}


class TestRunIO:
    """薄いデータアクセス層 (後で master table に一行で差し替える前提)."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "model_bench_k4_importance"
        (d / "importance_scores").mkdir(parents=True)
        results = [
            {"sample_id": "x1", "generated_text": "The answer is (A)."},
            {"sample_id": "x2", "generated_text": "The answer is (B)."},
        ]
        with open(d / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f)
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "org/model",
                    "benchmark": "bench",
                    "perturbed_metadata": {"num_perturbations": 4},
                },
                f,
            )
        return d

    def test_load_results_by_id(self, run_dir: Path):
        from typo_cot.data.run_io import load_results_by_id

        by_id = load_results_by_id(run_dir)
        assert set(by_id) == {"x1", "x2"}
        assert by_id["x1"]["generated_text"] == "The answer is (A)."

    def test_load_results_list(self, run_dir: Path):
        from typo_cot.data.run_io import load_results_list

        rs = load_results_list(run_dir)
        assert [r["sample_id"] for r in rs] == ["x1", "x2"]

    def test_load_run_config(self, run_dir: Path):
        from typo_cot.data.run_io import load_run_config

        cfg = load_run_config(run_dir)
        assert cfg["model"] == "org/model"
        assert cfg["perturbed_metadata"]["num_perturbations"] == 4

    def test_cot_scores_path(self, run_dir: Path):
        from typo_cot.data.run_io import cot_scores_path, question_scores_path

        assert cot_scores_path(run_dir, "x1").name == "x1_cot.pt"
        assert question_scores_path(run_dir, "x1").name == "x1.pt"

    def test_load_cot_scores(self, run_dir: Path):
        import torch

        from typo_cot.data.run_io import cot_scores_path, load_cot_scores

        payload = {"token_scores": [("a", 1.0)], "cot_token_start": 3, "cot_token_end": 7}
        torch.save(payload, cot_scores_path(run_dir, "x1"))
        loaded = load_cot_scores(run_dir, "x1")
        assert loaded["cot_token_start"] == 3
        assert loaded["token_scores"] == [("a", 1.0)]

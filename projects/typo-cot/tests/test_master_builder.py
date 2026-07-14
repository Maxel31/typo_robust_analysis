"""Step 0 統合テーブル構築ロジック (master_builder) のテスト.

アーカイブの results.json / full_results.json(sample_results) を模した
小さな合成 dict のみで完結する。GPU・アーカイブ実体は不要。
"""

from __future__ import annotations

import pandas as pd
import pytest

from typo_cot.data.master_builder import (
    build_condition_df,
    derive_union_exclusion,
    sample_metrics_from_analysis,
)
from typo_cot.data.master_table import MASTER_COLUMNS, validate_master_df


def _baseline_result(sample_id: str = "gsm8k_00000", **overrides) -> dict:
    row = {
        "sample_id": sample_id,
        "question": "What is 1+1?",
        "correct_answer": "2",
        "choices": None,
        "context": None,
        "generated_text": "1+1=2.\nThe answer is 2.",
        "extracted_answer": "2",
        "is_correct": True,
        "subset": "default",
        "question_top_k_words": [{"word": "1", "score": 0.5}],
        "cot_top_k_words": [{"word": "2", "score": 0.3}],
    }
    row.update(overrides)
    return row


def _perturbed_result(sample_id: str = "gsm8k_00000", **overrides) -> dict:
    row = _baseline_result(sample_id)
    row.update(
        {
            "question": "Wht is 1+1?",
            "generated_text": "1+1=3.\nThe answer is 3.",
            "extracted_answer": "3",
            "is_correct": False,
            "original_question": "What is 1+1?",
            "perturbed_tokens": [
                {
                    "token_index": 1,
                    "original_token": "What",
                    "perturbed_token": "Wht",
                    "importance_score": 0.9,
                    "perturbation_type": "omission",
                }
            ],
        }
    )
    row.update(overrides)
    return row


def _analysis_sample_result(sample_id: str = "gsm8k_00000", **overrides) -> dict:
    row = {
        "sample_id": sample_id,
        "pattern": "correct→incorrect",
        "answer_changed": True,
        "before_correct": True,
        "after_correct": False,
        "token_count": {"before": 10, "after": 11, "diff": 1},
        "question_metrics": {"spearman_r": 0.5, "jaccard": {"top10": 0.4}},
        "cot_metrics": {
            "rouge_l": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "jaccard": {
                "top3": 0.1,
                "top5": 0.2,
                "top10": 0.3,
                "top15": 0.35,
                "top20": 0.4,
            },
        },
    }
    row.update(overrides)
    return row


class TestBuildCleanCondition:
    def test_clean_rows(self):
        df = build_condition_df(
            baseline_results=[_baseline_result()],
            perturbed_results=None,
            sample_metrics=None,
            model="gemma-3-4b-it",
            benchmark="gsm8k",
            condition="clean",
            seed=42,
            prompt_id="gsm8k_cot_v1",
            source_path="outputs/baseline/gemma-3-4b-it_gsm8k/results.json",
        )
        validate_master_df(df)
        assert list(df.columns) == list(MASTER_COLUMNS)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["condition"] == "clean"
        assert row["question_text"] == "What is 1+1?"
        assert row["cot_text"] == "1+1=2.\nThe answer is 2."
        assert row["answer_pred"] == "2"
        assert row["answer_gold"] == "2"
        assert bool(row["is_correct"]) is True
        # clean 行では flip / CoT 指標は定義されない
        assert pd.isna(row["flip"])
        assert pd.isna(row["cot_rouge_l_f1"])
        # strict span: "The answer is 2" は pattern_1 で strict 検出される
        assert bool(row["span_extract_ok"]) is True
        assert row["answer_span"] == "2"
        # R_Q / R_C は JSON 文字列
        assert '"word"' in row["r_q"]
        assert '"word"' in row["r_c"]
        assert row["seed"] == 42
        assert row["prompt_id"] == "gsm8k_cot_v1"

    def test_clean_span_failure(self):
        df = build_condition_df(
            baseline_results=[
                _baseline_result(generated_text="I do not know.", extracted_answer=None)
            ],
            perturbed_results=None,
            sample_metrics=None,
            model="gemma-3-4b-it",
            benchmark="gsm8k",
            condition="clean",
            seed=42,
            prompt_id="gsm8k_cot_v1",
            source_path="x/results.json",
        )
        row = df.iloc[0]
        assert bool(row["span_extract_ok"]) is False
        assert pd.isna(row["answer_span"])


class TestBuildPerturbedCondition:
    def test_lxt4_rows_join_metrics(self):
        metrics = sample_metrics_from_analysis([_analysis_sample_result()])
        df = build_condition_df(
            baseline_results=[_baseline_result()],
            perturbed_results=[_perturbed_result()],
            sample_metrics=metrics,
            model="gemma-3-4b-it",
            benchmark="gsm8k",
            condition="lxt4",
            seed=42,
            prompt_id="gsm8k_cot_v1",
            source_path="outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance/results.json",
        )
        validate_master_df(df)
        row = df.iloc[0]
        assert row["condition"] == "lxt4"
        assert row["question_text"] == "Wht is 1+1?"
        assert row["original_question"] == "What is 1+1?"
        assert bool(row["flip"]) is True
        assert row["pattern"] == "correct→incorrect"
        assert row["cot_rouge_l_f1"] == pytest.approx(0.85)
        assert row["cot_jaccard_top10"] == pytest.approx(0.3)
        assert row["cot_jaccard_top3"] == pytest.approx(0.1)
        assert '"perturbed_token"' in row["perturbed_tokens"]

    def test_excluded_sample_has_null_metrics(self):
        # 分析から除外されたサンプル (sample_metrics に無い) は flip=NA のまま
        df = build_condition_df(
            baseline_results=[_baseline_result()],
            perturbed_results=[_perturbed_result()],
            sample_metrics={},
            model="gemma-3-4b-it",
            benchmark="gsm8k",
            condition="lxt4",
            seed=42,
            prompt_id="gsm8k_cot_v1",
            source_path="x/results.json",
        )
        row = df.iloc[0]
        assert pd.isna(row["flip"])
        assert pd.isna(row["cot_rouge_l_f1"])

    def test_perturbed_requires_results(self):
        with pytest.raises(ValueError, match="perturbed_results"):
            build_condition_df(
                baseline_results=[_baseline_result()],
                perturbed_results=None,
                sample_metrics=None,
                model="m",
                benchmark="gsm8k",
                condition="lxt4",
                seed=42,
                prompt_id="p",
                source_path="x",
            )


class TestSampleMetricsFromAnalysis:
    def test_mapping(self):
        metrics = sample_metrics_from_analysis(
            [
                _analysis_sample_result(),
                _analysis_sample_result(
                    sample_id="gsm8k_00001",
                    answer_changed=False,
                    pattern="correct→correct",
                ),
            ]
        )
        assert set(metrics) == {"gsm8k_00000", "gsm8k_00001"}
        m = metrics["gsm8k_00000"]
        assert m["flip"] is True
        assert m["pattern"] == "correct→incorrect"
        assert m["cot_rouge_l_f1"] == pytest.approx(0.85)
        assert m["cot_jaccard_top10"] == pytest.approx(0.3)
        assert metrics["gsm8k_00001"]["flip"] is False


class TestUnionExclusion:
    def test_union_semantics(self):
        # analyzer.compute_unified_exclusion と同じ意味論:
        # excluded = {sid: clean 失敗} ∪ {sid: いずれかの摂動条件で失敗}
        clean_ok = {"a": True, "b": True, "c": False, "d": True}
        cond_ok = {
            "lxt4": {"a": True, "b": False, "c": True, "d": True},
            "random4": {"a": True, "b": True, "c": True, "d": True},
        }
        excluded = derive_union_exclusion(clean_ok, cond_ok)
        assert excluded == {"b", "c"}

    def test_all_ok(self):
        assert derive_union_exclusion({"a": True}, {"lxt4": {"a": True}}) == set()

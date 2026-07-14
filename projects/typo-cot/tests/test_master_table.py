"""Step 0 統合テーブル (master table) の io 層・スキーマのテスト.

GPU 不要。合成の小さな DataFrame のみで完結する。
"""

from __future__ import annotations

import pandas as pd
import pytest

from typo_cot.data.master_table import (
    CONDITION_TO_ARCHIVE_SUFFIX,
    CONDITIONS,
    MASTER_COLUMNS,
    METRIC_SCOPE,
    empty_master_df,
    master_parquet_path,
    read_master_table,
    validate_master_df,
    write_condition_parquet,
)


def _make_row(**overrides) -> dict:
    row = {
        "sample_id": "gsm8k_00000",
        "model": "gemma-3-4b-it",
        "benchmark": "gsm8k",
        "condition": "clean",
        "question_text": "1+1=?",
        "cot_text": "1+1=2. The answer is 2.",
        "answer_span": "2",
        "answer_pred": "2",
        "answer_gold": "2",
        "is_correct": True,
        "flip": None,
        "pattern": None,
        "cot_rouge_l_f1": None,
        "cot_jaccard_top3": None,
        "cot_jaccard_top5": None,
        "cot_jaccard_top10": None,
        "cot_jaccard_top15": None,
        "cot_jaccard_top20": None,
        "r_q": '[{"word": "1", "score": 0.5}]',
        "r_c": '[{"word": "2", "score": 0.3}]',
        "span_extract_ok": True,
        "seed": 42,
        "prompt_id": "gsm8k_cot_v1",
        "subset": None,
        "original_question": None,
        "perturbed_tokens": None,
        "source_path": "outputs/baseline/gemma-3-4b-it_gsm8k/results.json",
    }
    row.update(overrides)
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(MASTER_COLUMNS))


class TestSchema:
    def test_conditions_frozen(self):
        assert CONDITIONS == ("clean", "lxt1", "lxt2", "lxt4", "lxt8", "random4")

    def test_condition_archive_mapping(self):
        assert CONDITION_TO_ARCHIVE_SUFFIX["lxt4"] == "k4_importance"
        assert CONDITION_TO_ARCHIVE_SUFFIX["random4"] == "k4_random"
        assert CONDITION_TO_ARCHIVE_SUFFIX["lxt1"] == "k1_importance"
        assert "clean" not in CONDITION_TO_ARCHIVE_SUFFIX

    def test_master_columns_contains_spec_columns(self):
        # 仕様(§手法の詳細)の列が全て存在すること
        for col in (
            "sample_id",
            "model",
            "benchmark",
            "condition",
            "question_text",
            "cot_text",
            "answer_span",
            "answer_pred",
            "answer_gold",
            "flip",
            "cot_rouge_l_f1",
            "cot_jaccard_top10",
            "r_q",
            "r_c",
            "span_extract_ok",
            "seed",
            "prompt_id",
        ):
            assert col in MASTER_COLUMNS, col

    def test_metric_scope_labels(self):
        # 修正C: 本文は ROUGE-L・Jaccard@10・flip のみ
        assert METRIC_SCOPE["cot_rouge_l_f1"] == "main"
        assert METRIC_SCOPE["cot_jaccard_top10"] == "main"
        assert METRIC_SCOPE["flip"] == "main"
        assert METRIC_SCOPE["cot_jaccard_top3"] == "appendix"
        assert METRIC_SCOPE["cot_jaccard_top20"] == "appendix"

    def test_empty_master_df_valid(self):
        df = empty_master_df()
        assert list(df.columns) == list(MASTER_COLUMNS)
        validate_master_df(df)  # raiseしない

    def test_validate_rejects_missing_column(self):
        df = _make_df([_make_row()]).drop(columns=["flip"])
        with pytest.raises(ValueError, match="flip"):
            validate_master_df(df)

    def test_validate_rejects_unknown_condition(self):
        df = _make_df([_make_row(condition="k4_importance")])
        with pytest.raises(ValueError, match="condition"):
            validate_master_df(df)

    def test_validate_rejects_duplicate_key(self):
        df = _make_df([_make_row(), _make_row()])
        with pytest.raises(ValueError, match="duplicate"):
            validate_master_df(df)


class TestIO:
    def test_parquet_path_layout(self, tmp_path):
        p = master_parquet_path(tmp_path, "gemma-3-4b-it", "gsm8k", "lxt4")
        assert p == tmp_path / "gemma-3-4b-it" / "gsm8k" / "lxt4.parquet"

    def test_roundtrip(self, tmp_path):
        df = _make_df(
            [
                _make_row(),
                _make_row(sample_id="gsm8k_00001", is_correct=False),
            ]
        )
        path = write_condition_parquet(df, tmp_path)
        assert path.exists()
        back = read_master_table(tmp_path)
        assert len(back) == 2
        assert list(back.columns) == list(MASTER_COLUMNS)
        assert set(back["sample_id"]) == {"gsm8k_00000", "gsm8k_00001"}
        # nullable bool が保持される
        assert back["flip"].isna().all()

    def test_write_rejects_mixed_condition(self, tmp_path):
        df = _make_df(
            [
                _make_row(),
                _make_row(sample_id="gsm8k_00001", condition="lxt4"),
            ]
        )
        with pytest.raises(ValueError, match="single"):
            write_condition_parquet(df, tmp_path)

    def test_read_with_filters(self, tmp_path):
        clean = _make_df([_make_row()])
        lxt4 = _make_df(
            [
                _make_row(
                    condition="lxt4",
                    flip=True,
                    pattern="correct→incorrect",
                    cot_rouge_l_f1=0.5,
                    cot_jaccard_top10=0.2,
                )
            ]
        )
        write_condition_parquet(clean, tmp_path)
        write_condition_parquet(lxt4, tmp_path)
        only_lxt4 = read_master_table(tmp_path, conditions=["lxt4"])
        assert len(only_lxt4) == 1
        assert only_lxt4.iloc[0]["condition"] == "lxt4"
        assert bool(only_lxt4.iloc[0]["flip"]) is True
        none_df = read_master_table(tmp_path, models=["no-such-model"])
        assert none_df.empty

"""統合テーブルからの論文数値再現ロジック (analysis/reproduce) のテスト.

- 条件別精度 (論文 Table 3 / アーカイブ table5.csv 相当)
- flip との偏相関 (論文 Fig.3 系 / アーカイブ full_results.json の
  partial_correlations 相当。analyzer._compute_partial_correlation と
  同じ「線形回帰残差 + Pearson」で計算する)

小さな合成 DataFrame のみで完結する。GPU・アーカイブ実体は不要。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from typo_cot.analysis.reproduce import (
    accuracy_by_condition,
    partial_correlation_flip,
)
from typo_cot.data.master_table import MASTER_COLUMNS


def _row(sample_id, condition, *, is_correct=True, flip=None, rouge=None, j10=None):
    row = dict.fromkeys(MASTER_COLUMNS)
    row.update(
        {
            "sample_id": sample_id,
            "model": "m1",
            "benchmark": "gsm8k",
            "condition": condition,
            "is_correct": is_correct,
            "flip": flip,
            "cot_rouge_l_f1": rouge,
            "cot_jaccard_top10": j10,
            "span_extract_ok": True,
            "seed": 42,
            "prompt_id": "p",
        }
    )
    return row


def _master_df(rows):
    return pd.DataFrame(rows, columns=list(MASTER_COLUMNS))


class TestAccuracyByCondition:
    def test_accuracy_wide_table(self):
        rows = [
            _row("s1", "clean", is_correct=True),
            _row("s2", "clean", is_correct=False),
            _row("s1", "lxt4", is_correct=False),
            _row("s2", "lxt4", is_correct=False),
            _row("s1", "random4", is_correct=True),
            _row("s2", "random4", is_correct=True),
        ]
        df = accuracy_by_condition(_master_df(rows))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["model"] == "m1"
        assert row["benchmark"] == "gsm8k"
        assert row["clean"] == pytest.approx(0.5)
        assert row["lxt4"] == pytest.approx(0.0)
        assert row["random4"] == pytest.approx(1.0)
        # 存在しない条件は NaN
        assert np.isnan(row["lxt8"])

    def test_accuracy_counts(self):
        rows = [_row(f"s{i}", "clean", is_correct=(i % 2 == 0)) for i in range(10)]
        df = accuracy_by_condition(_master_df(rows))
        assert df.iloc[0]["n_clean"] == 10
        assert df.iloc[0]["clean"] == pytest.approx(0.5)


class TestPartialCorrelationFlip:
    def test_matches_analyzer_method(self):
        # analyzer._compute_partial_correlation と同一の
        # 「z を統制した線形回帰残差同士の Pearson 相関」に一致すること
        rng = np.random.default_rng(0)
        n = 200
        rouge = rng.uniform(0, 1, n)
        j10 = np.clip(rouge * 0.5 + rng.normal(0, 0.2, n), 0, 1)
        flip = (rng.uniform(0, 1, n) < 0.3 + 0.4 * (1 - j10)).astype(float)
        rows = [
            _row(
                f"s{i}",
                "lxt4",
                flip=bool(flip[i]),
                rouge=rouge[i],
                j10=j10[i],
            )
            for i in range(n)
        ]
        result = partial_correlation_flip(_master_df(rows), k=10)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["n"] == n

        def residual(a, b):
            slope, intercept, _, _, _ = stats.linregress(b, a)
            return a - (slope * b + intercept)

        exp_jr, _ = stats.pearsonr(residual(j10, rouge), residual(flip, rouge))
        exp_rj, _ = stats.pearsonr(residual(rouge, j10), residual(flip, j10))
        assert row["rho_J_given_R"] == pytest.approx(exp_jr, abs=1e-12)
        assert row["rho_R_given_J"] == pytest.approx(exp_rj, abs=1e-12)

    def test_excludes_rows_without_flip(self):
        rng = np.random.default_rng(1)
        rows = [
            _row(f"s{i}", "lxt4", flip=bool(i % 2), rouge=rng.uniform(), j10=rng.uniform())
            for i in range(30)
        ]
        # flip=NA の行 (分析から除外されたサンプル) は集計に入らない
        rows.append(_row("s_excluded", "lxt4", flip=None, rouge=0.5, j10=0.5))
        result = partial_correlation_flip(_master_df(rows), k=10)
        assert result.iloc[0]["n"] == 30

    def test_jaccard_na_becomes_zero(self):
        # analyzer は cot_jaccard.get(k, 0.0) と等価に NA→0.0 で計算する
        rng = np.random.default_rng(2)
        rows = [
            _row(f"s{i}", "lxt4", flip=bool(i % 2), rouge=rng.uniform(), j10=rng.uniform())
            for i in range(30)
        ]
        rows.append(_row("s_na_j10", "lxt4", flip=True, rouge=0.5, j10=None))
        result = partial_correlation_flip(_master_df(rows), k=10)
        assert result.iloc[0]["n"] == 31

    def test_too_few_samples_returns_empty(self):
        rows = [_row("s1", "lxt4", flip=True, rouge=0.5, j10=0.5)]
        result = partial_correlation_flip(_master_df(rows), k=10)
        assert result.empty

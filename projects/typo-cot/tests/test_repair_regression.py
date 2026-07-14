"""repair.regression のテスト (実験9: flip ~ 修復スコア + 統制のロジスティック回帰).

statsmodels GLM (Binomial) + クラスタロバスト SE (item 単位)。
GPU 不要・合成データのみ。
"""

import numpy as np
import pandas as pd
import pytest

from typo_cot.repair.regression import fit_flip_regression


def _make_synthetic(n_items: int = 120, words_per_item: int = 4, seed: int = 0) -> pd.DataFrame:
    """flip が repair_score に強く負に依存する合成データ."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        # item 内の語は同じ flip ラベルを共有 (flip はサンプル単位)
        repair_scores = rng.uniform(0.2, 1.0, size=words_per_item)
        mean_repair = repair_scores.mean()
        logit = 4.0 - 8.0 * mean_repair  # 修復スコアが高いほど flip しにくい
        p = 1.0 / (1.0 + np.exp(-logit))
        flip = bool(rng.uniform() < p)
        for w in range(words_per_item):
            rows.append(
                {
                    "sample_id": f"item_{i:04d}",
                    "flip": flip,
                    "repair_score": repair_scores[w],
                    "split_increment": int(rng.integers(0, 3)),
                    "zipf_freq": float(rng.uniform(1.0, 7.0)),
                    "r_q": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


FEATURES = ["repair_score", "split_increment", "zipf_freq", "r_q"]


class TestFitFlipRegression:
    def test_recovers_negative_repair_coefficient(self) -> None:
        df = _make_synthetic()
        result = fit_flip_regression(df, feature_cols=FEATURES, cluster_col="sample_id")
        coef = result.coefs
        # 出力形式: feature 行 x (coef, se, z, p) 列
        assert "repair_score" in coef.index
        assert {"coef", "se", "z", "p"} <= set(coef.columns)
        # 修復スコアの係数は負で有意
        assert coef.loc["repair_score", "coef"] < 0
        assert coef.loc["repair_score", "p"] < 0.01

    def test_standardization(self) -> None:
        df = _make_synthetic()
        r_std = fit_flip_regression(
            df, feature_cols=FEATURES, cluster_col="sample_id", standardize=True
        )
        r_raw = fit_flip_regression(
            df, feature_cols=FEATURES, cluster_col="sample_id", standardize=False
        )
        # 標準化しても符号は変わらない
        assert (
            np.sign(r_std.coefs.loc["repair_score", "coef"])
            == np.sign(r_raw.coefs.loc["repair_score", "coef"])
        )
        # 標準化時は係数の絶対値がスケールされる (単なる同値でない)
        assert not np.isclose(
            r_std.coefs.loc["zipf_freq", "coef"], r_raw.coefs.loc["zipf_freq", "coef"]
        )

    def test_n_reported(self) -> None:
        df = _make_synthetic(n_items=30)
        result = fit_flip_regression(df, feature_cols=FEATURES, cluster_col="sample_id")
        assert result.n_obs == len(df)
        assert result.n_clusters == 30

    def test_missing_rows_are_dropped(self) -> None:
        df = _make_synthetic(n_items=30)
        df.loc[0, "repair_score"] = np.nan
        result = fit_flip_regression(df, feature_cols=FEATURES, cluster_col="sample_id")
        assert result.n_obs == len(df) - 1

    def test_constant_flip_raises(self) -> None:
        df = _make_synthetic(n_items=10)
        df["flip"] = False
        with pytest.raises(ValueError):
            fit_flip_regression(df, feature_cols=FEATURES, cluster_col="sample_id")

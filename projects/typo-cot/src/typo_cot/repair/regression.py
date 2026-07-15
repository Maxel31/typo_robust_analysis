"""flip ~ 修復スコア + 表層統制 のロジスティック回帰 (実験9).

仕様の回帰式 flip ~ 修復スコア + 分割数増分 + Zipf頻度 + R_Q + (1|item) に対し、
statsmodels GLM (Binomial) + item (sample_id) 単位のクラスタロバスト SE で
近似する ((1|item) のランダム切片の代替。statsmodels の GLMM (BinomialBayesMixedGLM)
は p 値の扱いが異なるため、主報告はクラスタロバスト、GLMM は必要に応じて別途)。
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class FlipRegressionResult:
    """回帰結果.

    Attributes:
        coefs: 行 = const + 特徴量、列 = coef / se / z / p の DataFrame
        n_obs: 使用した観測数 (NaN 行除外後)
        n_clusters: クラスタ (item) 数
        standardized: 特徴量を z 標準化したか
    """

    coefs: pd.DataFrame
    n_obs: int
    n_clusters: int
    standardized: bool


def filter_clean_correct(df: pd.DataFrame, col: str = "clean_correct") -> pd.DataFrame:
    """主推定量の条件付け: clean 生成が正解のサンプルの語行のみ残す.

    本番計測は全量 (条件付けなし) で行い、分析側でこのフィルタを適用する
    (計画書の共通規約)。clean_correct が None (正誤判定不能) の行は落とす。

    Raises:
        KeyError: clean_correct 列が無い場合
    """
    if col not in df.columns:
        raise KeyError(f"列がありません: {col}")
    mask = df[col].map(lambda v: v is True or v == 1)
    return df[mask]


def fit_flip_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "sample_id",
    flip_col: str = "flip",
    standardize: bool = True,
) -> FlipRegressionResult:
    """flip を特徴量で予測するロジスティック回帰をあてはめる.

    Args:
        df: 語レベルの分析テーブル (1 行 = 1 摂動語)
        feature_cols: 説明変数の列名 (例: repair_score, split_increment, zipf_freq, r_q)
        cluster_col: クラスタロバスト SE のグループ列 (item = sample_id)
        flip_col: 目的変数 (bool)
        standardize: True なら特徴量を z 標準化 (係数の比較可能性のため)

    Returns:
        FlipRegressionResult

    Raises:
        ValueError: flip が定数 (全 True / 全 False) の場合
    """
    cols = [flip_col, cluster_col, *feature_cols]
    data = df[cols].dropna().copy()

    y = data[flip_col].astype(float)
    if y.nunique() < 2:
        raise ValueError("flip が定数のため回帰をあてはめられません")

    x = data[feature_cols].astype(float)
    if standardize:
        std = x.std(ddof=0).replace(0.0, 1.0)
        x = (x - x.mean()) / std
    x = sm.add_constant(x)

    model = sm.GLM(y, x, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col].to_numpy()}
    )

    coefs = pd.DataFrame(
        {
            "coef": result.params,
            "se": result.bse,
            "z": result.tvalues,
            "p": result.pvalues,
        }
    )
    return FlipRegressionResult(
        coefs=coefs,
        n_obs=int(len(data)),
        n_clusters=int(data[cluster_col].nunique()),
        standardized=standardize,
    )

"""統合テーブル (master table) から論文数値を再計算する集計ロジック.

- `accuracy_by_condition`: 条件別精度 (論文 Table 3 / アーカイブ table5.csv 相当)
- `partial_correlation_flip`: flip を目的変数とする偏相関
  (アーカイブ full_results.json の partial_correlations 相当)

偏相関は `typo_cot.analysis.analyzer.TypoCoTAnalyzer._compute_partial_correlation`
と同一の手続き (統制変数への線形回帰残差同士の Pearson 相関、NaN/Inf 除去、
n>=20、Jaccard 欠損は 0.0 扱い) を master table の列に対して適用する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from typo_cot.data.master_table import CONDITIONS

# 偏相関の最小サンプル数 (analyzer と同じ)
_MIN_N = 20


def accuracy_by_condition(master_df: pd.DataFrame) -> pd.DataFrame:
    """(model, benchmark) ごとの条件別精度を wide 形式で返す.

    Returns:
        columns = [model, benchmark, clean, lxt1, ..., random4,
                   n_clean, n_lxt1, ..., n_random4]
        条件の行が存在しない場合は NaN / 0。
    """
    rows: list[dict] = []
    for (model, benchmark), group in master_df.groupby(["model", "benchmark"], sort=True):
        row: dict = {"model": model, "benchmark": benchmark}
        for cond in CONDITIONS:
            sub = group[group["condition"] == cond]
            n = int(sub["is_correct"].notna().sum())
            row[f"n_{cond}"] = n
            row[cond] = (
                float(sub["is_correct"].astype("boolean").sum()) / n if n > 0 else np.nan
            )
        rows.append(row)
    columns = (
        ["model", "benchmark"]
        + list(CONDITIONS)
        + [f"n_{c}" for c in CONDITIONS]
    )
    return pd.DataFrame(rows, columns=columns)


def _partial_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, int]:
    """analyzer._compute_partial_correlation と同じ偏相関 (r, p, n).

    z を統制変数として x・y それぞれの線形回帰残差を取り、残差同士の
    Pearson 相関を返す。NaN/Inf は事前に除去。n < _MIN_N は (nan, nan, n)。
    """
    valid = ~(
        np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isinf(x) | np.isinf(y) | np.isinf(z)
    )
    x, y, z = x[valid], y[valid], z[valid]
    n = len(x)
    if n < _MIN_N:
        return (float("nan"), float("nan"), n)
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    residual_x = x - (slope_xz * z + intercept_xz)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    residual_y = y - (slope_yz * z + intercept_yz)
    r, p = stats.pearsonr(residual_x, residual_y)
    return (float(r), float(p), n)


def partial_correlation_flip(
    master_df: pd.DataFrame,
    k: int = 10,
    condition: str = "lxt4",
) -> pd.DataFrame:
    """flip を目的変数とする偏相関 ρ(J|R)・ρ(R|J) を (model, benchmark) ごとに計算.

    集計対象: 指定 condition の行のうち flip が非 NA のもの
    (= 旧 analyzer の集計対象サンプルと一致する)。
    Jaccard@k の NA は analyzer の `cot_jaccard.get(k, 0.0)` と等価に 0.0 に置換。

    Returns:
        columns = [model, benchmark, condition, k, n,
                   rho_J_given_R, p_J_given_R, rho_R_given_J, p_R_given_J,
                   n_flip, n_noflip]
    """
    j_col = f"cot_jaccard_top{k}"
    sub = master_df[
        (master_df["condition"] == condition) & master_df["flip"].notna()
    ]
    rows: list[dict] = []
    for (model, benchmark), group in sub.groupby(["model", "benchmark"], sort=True):
        flip = group["flip"].astype("boolean").astype(float).to_numpy(dtype=float)
        rouge = group["cot_rouge_l_f1"].astype(float).to_numpy(dtype=float)
        jaccard = group[j_col].fillna(0.0).astype(float).to_numpy(dtype=float)
        jr_r, jr_p, n = _partial_corr(jaccard, flip, rouge)
        rj_r, rj_p, _ = _partial_corr(rouge, flip, jaccard)
        if n < _MIN_N:
            continue
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "condition": condition,
                "k": k,
                "n": n,
                "rho_J_given_R": jr_r,
                "p_J_given_R": jr_p,
                "rho_R_given_J": rj_r,
                "p_R_given_J": rj_p,
                "n_flip": int(group["flip"].astype("boolean").sum()),
                "n_noflip": int(len(group) - group["flip"].astype("boolean").sum()),
            }
        )
    columns = [
        "model",
        "benchmark",
        "condition",
        "k",
        "n",
        "rho_J_given_R",
        "p_J_given_R",
        "rho_R_given_J",
        "p_R_given_J",
        "n_flip",
        "n_noflip",
    ]
    return pd.DataFrame(rows, columns=columns)

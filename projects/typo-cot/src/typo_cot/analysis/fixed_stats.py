"""実験4: fixed-target 統計層.

- ρ(J|R): flip (0/1) と CoT:Jaccard@k の偏相関 (ROUGE-L を統制)。
  規約は analyzer._compute_partial_correlation と同一の「z を線形回帰で除去した
  残差同士の Pearson 相関」= 一次偏相関。r は pingouin.partial_corr と厳密一致。
  p 値は偏相関の自由度 (n-3) の t 分布で計算する (pingouin と同一)。
- bootstrap 95%CI (percentile, 自前実装, デフォルト B=10,000)
- Holm 補正
- Δρ = ρ_fixed − ρ_default の paired bootstrap 検定
  (同一リサンプルで両条件の ρ を計算し差の分布を作る)
- token_scores ([(token, score), ...]) からの CoT:Jaccard@k 再計算
  (analysis/metrics.top_k_jaccard_by_token と同値)
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from typo_cot.analysis.metrics import top_k_jaccard_by_token

DEFAULT_KS = (5, 10, 20)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def _residual_pearson_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """z の線形効果を除去した残差同士の Pearson r (一次偏相関と同値)."""
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    residual_x = x - (slope_xz * z + intercept_xz)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    residual_y = y - (slope_yz * z + intercept_yz)
    r, _ = stats.pearsonr(residual_x, residual_y)
    return float(r)


def partial_corr_flip(
    jaccard: Sequence[float],
    flip: Sequence[float],
    rouge: Sequence[float],
) -> tuple[float, float, int]:
    """ρ(J|R): ROUGE-L を統制した Jaccard と flip の偏相関.

    Args:
        jaccard: CoT:Jaccard@k (独立変数)
        flip: 回答変化 (0/1, 従属変数)
        rouge: CoT ROUGE-L F1 (統制変数)

    Returns:
        (r, p, n)。非有限値の行は除外する。p は自由度 n-3 の t 分布
        (一次偏相関の正しい自由度; pingouin.partial_corr と一致)。
    """
    x = np.asarray(jaccard, dtype=np.float64)
    y = np.asarray(flip, dtype=np.float64)
    z = np.asarray(rouge, dtype=np.float64)
    mask = _finite_mask(x, y, z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), n

    r = _residual_pearson_r(x, y, z)
    dof = n - 3  # n - 2 - (統制変数の数=1)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt(dof / (1.0 - r * r))
        p = float(2 * stats.t.sf(abs(t_stat), dof))
    return r, p, n


def bootstrap_partial_corr_ci(
    jaccard: Sequence[float],
    flip: Sequence[float],
    rouge: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """ρ(J|R) の percentile bootstrap 信頼区間 (デフォルト 95%)."""
    x = np.asarray(jaccard, dtype=np.float64)
    y = np.asarray(flip, dtype=np.float64)
    z = np.asarray(rouge, dtype=np.float64)
    mask = _finite_mask(x, y, z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)

    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb, zb = x[idx], y[idx], z[idx]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0 or np.ptp(zb) == 0:
            continue  # 退化リサンプル (分散ゼロ) はスキップ
        rs.append(_residual_pearson_r(xb, yb, zb))
    rs_arr = np.asarray(rs)
    rs_arr = rs_arr[np.isfinite(rs_arr)]
    lo = float(np.percentile(rs_arr, 100 * alpha / 2))
    hi = float(np.percentile(rs_arr, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni 補正済み p 値を入力順で返す (1 でキャップ)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def paired_bootstrap_delta_rho(
    jaccard_default: Sequence[float],
    jaccard_fixed: Sequence[float],
    flip: Sequence[float],
    rouge: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Δρ = ρ(J_fixed|R) − ρ(J_default|R) の paired bootstrap 検定.

    同一サンプルのリサンプルごとに両条件の偏相関を計算し、差の分布から
    percentile CI と両側 p 値 ((count+1)/(B+1) 規約) を得る。

    Returns:
        {rho_default, rho_fixed, delta_rho, ci95, p_value, n, n_boot}
    """
    jd = np.asarray(jaccard_default, dtype=np.float64)
    jf = np.asarray(jaccard_fixed, dtype=np.float64)
    y = np.asarray(flip, dtype=np.float64)
    z = np.asarray(rouge, dtype=np.float64)
    mask = _finite_mask(jd, jf, y, z)
    jd, jf, y, z = jd[mask], jf[mask], y[mask], z[mask]
    n = len(jd)

    rho_default = _residual_pearson_r(jd, y, z)
    rho_fixed = _residual_pearson_r(jf, y, z)
    delta = rho_fixed - rho_default

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, zb = y[idx], z[idx]
        jdb, jfb = jd[idx], jf[idx]
        if (
            np.ptp(yb) == 0
            or np.ptp(zb) == 0
            or np.ptp(jdb) == 0
            or np.ptp(jfb) == 0
        ):
            continue
        d = _residual_pearson_r(jfb, yb, zb) - _residual_pearson_r(jdb, yb, zb)
        if np.isfinite(d):
            deltas.append(d)
    deltas_arr = np.asarray(deltas)
    b = len(deltas_arr)

    lo = float(np.percentile(deltas_arr, 100 * alpha / 2))
    hi = float(np.percentile(deltas_arr, 100 * (1 - alpha / 2)))
    # 両側 p 値: bootstrap 分布の 0 に対する符号検定 ((count+1)/(B+1) 規約)
    p_le = (np.sum(deltas_arr <= 0) + 1) / (b + 1)
    p_ge = (np.sum(deltas_arr >= 0) + 1) / (b + 1)
    p_value = float(min(1.0, 2 * min(p_le, p_ge)))

    return {
        "rho_default": float(rho_default),
        "rho_fixed": float(rho_fixed),
        "delta_rho": float(delta),
        "ci95": (lo, hi),
        "p_value": p_value,
        "n": int(n),
        "n_boot": int(b),
    }


def cot_jaccard_from_scores(
    clean_scores: Sequence[Sequence],
    other_scores: Sequence[Sequence],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """token_scores ペアから CoT:Jaccard@k を再計算する.

    Args:
        clean_scores: clean 条件の [(token, score), ...]
        other_scores: 比較条件 (default/fixed) の [(token, score), ...]
        ks: k の水準 (デフォルト 5/10/20)

    Returns:
        {"top{k}": jaccard} (metrics.top_k_jaccard_by_token と同値)
    """
    out: dict[str, float] = {}
    if not clean_scores or not other_scores:
        return {f"top{k}": 0.0 for k in ks}
    t1 = [ts[0] for ts in clean_scores]
    s1 = [float(ts[1]) for ts in clean_scores]
    t2 = [ts[0] for ts in other_scores]
    s2 = [float(ts[1]) for ts in other_scores]
    for k in ks:
        out[f"top{k}"] = top_k_jaccard_by_token(t1, s1, t2, s2, k=k)
    return out

"""実験8-fine: 1層分解プロファイルの集計と H8f-1〜5 の事前登録判定.

run_patching_fine.py が出力する per-pair の cell レコード列 (kind ∈
{single, cumulative, noising, sham_single}) を層ごとに集計し、
主指標 s2_kl_recovery (最初の CoT 語分布の KL 回復率) の層プロファイルから

    H8f-1 ピーク相対深さ li/L < 0.2
    H8f-2 プラトー vs 単層スパイク
    H8f-3 累積 patch の早期飽和 (>= 単層 max の 1.2倍)
    H8f-4 検証点 (第14/20/26層) の回復 ≈ 0
    H8f-5 noising の最良層±1 における十分性 (KL 乖離の過半再現)

を判定する。すべて純関数 (GPU / モデル不要)。主推定量は **median** (s2_kl_recovery は
1 - KL_patched/KL_base で下に非有界の重い左裾を持ち平均が外れ値に汚染されるため; 平均も副次保持)、
パーセンタイル bootstrap 95% CI 付き。反証は各 judge の "supported" フラグと補助量で透明化する。
"""

from __future__ import annotations

import random as _random
from collections import defaultdict
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# 層プロファイル集計
# ---------------------------------------------------------------------------


def collect_by_layer(
    cells: Sequence[dict],
    kind: str,
    direction: str,
    field: str,
) -> dict[int, list[float]]:
    """cell レコード列を {層 index → 値のリスト} に集計する (None/欠損は除外)."""
    out: dict[int, list[float]] = defaultdict(list)
    for c in cells:
        if c.get("kind") != kind or c.get("direction") != direction:
            continue
        v = c.get(field, None)
        if v is None:
            continue
        out[int(c["layer"])].append(float(v))
    return dict(out)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _boot_ci(
    values: Sequence[float],
    statfn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """任意の統計量 statfn のパーセンタイル bootstrap CI. 空なら (None,None,None)."""
    n = len(values)
    if n == 0:
        return (None, None, None)
    vals = [float(v) for v in values]
    point = statfn(vals)
    if n == 1:
        return (point, point, point)
    rng = _random.Random(seed)
    boot = sorted(statfn([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (point, boot[lo_idx], boot[hi_idx])


def mean_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """連続値の平均のパーセンタイル bootstrap CI. 空なら (None, None, None)."""
    return _boot_ci(values, lambda v: sum(v) / len(v), n_boot=n_boot, seed=seed, alpha=alpha)


def summarize_by_layer(
    by_layer: dict[int, list[float]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[int, dict]:
    """{層 → 値リスト} を {層 → {n, mean, median, ci_lo/hi, median_lo/hi}} に要約する.

    主推定量は **median** (s2_kl_recovery は 1 - KL_patched/KL_base で下に非有界の
    重い左裾を持ち平均が外れ値に汚染されるため)。mean は副次的に保持する。
    """
    summary: dict[int, dict] = {}
    for layer, values in by_layer.items():
        mean, mlo, mhi = mean_ci(values, n_boot=n_boot, seed=seed)
        median, dlo, dhi = _boot_ci(values, _median, n_boot=n_boot, seed=seed)
        summary[layer] = {
            "n": len(values),
            "median": median,
            "median_lo": dlo,
            "median_hi": dhi,
            "mean": mean,
            "ci_lo": mlo,
            "ci_hi": mhi,
        }
    return summary


# ---------------------------------------------------------------------------
# プロファイル形状の解析
# ---------------------------------------------------------------------------


def argmax_layer(
    summary: dict[int, dict],
    key: str = "median",
    restrict: Sequence[int] | None = None,
) -> int | None:
    """平均が最大の層 index を返す (restrict 指定でその部分集合に限定)."""
    layers = list(summary.keys()) if restrict is None else [li for li in restrict if li in summary]
    layers = [li for li in layers if summary[li].get(key) is not None]
    if not layers:
        return None
    return max(layers, key=lambda li: summary[li][key])


def plateau_layers(
    summary: dict[int, dict],
    best_layer: int,
    rel: float = 0.9,
    key: str = "median",
) -> list[int]:
    """best_layer を中心に、平均が max の rel 倍以上で連続する層 index を返す.

    連続 (層 index が隣接) であることを要求するため、孤立ピーク (スパイク) は
    [best_layer] のみを返す。プラトーなら複数層。
    """
    if best_layer not in summary or summary[best_layer].get(key) is None:
        return []
    peak = summary[best_layer][key]
    thresh = rel * peak
    plateau = [best_layer]
    # 左に拡張
    li = best_layer - 1
    while li in summary and summary[li].get(key) is not None and summary[li][key] >= thresh:
        plateau.append(li)
        li -= 1
    # 右に拡張
    r = best_layer + 1
    while r in summary and summary[r].get(key) is not None and summary[r][key] >= thresh:
        plateau.append(r)
        r += 1
    return sorted(plateau)


def saturation_layer(
    summary: dict[int, dict],
    frac: float = 0.9,
    key: str = "median",
) -> int | None:
    """累積プロファイルが max の frac 倍に最初に到達する層 index (層昇順)."""
    layers = sorted(li for li in summary if summary[li].get(key) is not None)
    if not layers:
        return None
    peak = max(summary[li][key] for li in layers)
    thresh = frac * peak
    for li in layers:
        if summary[li][key] >= thresh:
            return li
    return layers[-1]


# ---------------------------------------------------------------------------
# H8f-1〜5 の事前登録判定 (反証も透明化)
# ---------------------------------------------------------------------------


def judge_h8f1_peak_depth(
    single_summary: dict[int, dict],
    n_layers: int,
    restrict_early: Sequence[int] | None = None,
    depth_thresh: float = 0.2,
    stat: str = "median",
) -> dict:
    """H8f-1: 単層回復率ピークの相対深さ li/L < 0.2 か (主推定量 = median)."""
    best = argmax_layer(single_summary, key=stat, restrict=restrict_early)
    if best is None or n_layers <= 0:
        return {"best_layer": None, "rel_depth": None, "supported": None}
    rel = best / n_layers
    return {
        "best_layer": best,
        "peak_stat": stat,
        "peak_value": single_summary[best].get(stat),
        "peak_median": single_summary[best].get("median"),
        "peak_mean": single_summary[best].get("mean"),
        "rel_depth": rel,
        "depth_thresh": depth_thresh,
        "supported": bool(rel < depth_thresh),
    }


def judge_h8f2_plateau_vs_spike(
    single_summary: dict[int, dict],
    best_layer: int,
    rel: float = 0.9,
) -> dict:
    """H8f-2: プラトー型 (隣接>=2層が同水準) か単層スパイクか.

    プラトー → prediction supported。スパイク → 反証枝: より強い局在主張
    (特定層の語彙統合点) に切替 (shape='spike')。
    """
    plat = plateau_layers(single_summary, best_layer, rel=rel)
    width = len(plat)
    shape = "plateau" if width >= 2 else "spike"
    return {
        "plateau_layers": plat,
        "plateau_width": width,
        "shape": shape,
        "supported": bool(shape == "plateau"),
        "branch": None if shape == "plateau" else "spike -> stronger localization claim",
    }


def judge_h8f3_cumulative_saturation(
    single_summary: dict[int, dict],
    cumulative_summary: dict[int, dict],
    n_layers: int,
    frac: float = 0.9,
    ratio_thresh: float = 1.2,
    depth_thresh: float = 0.2,
    stat: str = "median",
) -> dict:
    """H8f-3: 累積 patch が早期 (li/L<=0.2) で飽和し単層 max の >=1.2倍か (median)."""
    single_vals = [s[stat] for s in single_summary.values() if s.get(stat) is not None]
    cum_vals = [s[stat] for s in cumulative_summary.values() if s.get(stat) is not None]
    if not single_vals or not cum_vals:
        return {"supported": None}
    single_max = max(single_vals)
    cum_max = max(cum_vals)
    ratio = cum_max / single_max if abs(single_max) > 1e-9 else None
    sat = saturation_layer(cumulative_summary, frac=frac, key=stat)
    sat_rel = sat / n_layers if (sat is not None and n_layers > 0) else None
    supported = bool(
        ratio is not None
        and ratio >= ratio_thresh
        and sat_rel is not None
        and sat_rel <= depth_thresh
    )
    return {
        "single_max": single_max,
        "cumulative_max": cum_max,
        "ratio_cum_over_single": ratio,
        "ratio_thresh": ratio_thresh,
        "saturation_layer": sat,
        "saturation_rel_depth": sat_rel,
        "depth_thresh": depth_thresh,
        "supported": supported,
    }


def judge_h8f4_late_null(
    single_summary: dict[int, dict],
    val_layers: Sequence[int] = (14, 20, 26),
    thresh: float = 0.1,
    stat: str = "median",
) -> dict:
    """H8f-4: 検証点 (第14/20/26層) が幅1でも回復 ≈ 0 か (median)."""
    val_means: dict[int, float] = {}
    for li in val_layers:
        if li in single_summary and single_summary[li].get(stat) is not None:
            val_means[li] = single_summary[li][stat]
    if not val_means:
        return {"val_means": {}, "supported": None}
    supported = all(abs(m) < thresh for m in val_means.values())
    return {
        "val_means": val_means,
        "thresh": thresh,
        "supported": bool(supported),
    }


def judge_h8f5_noising_sufficiency(
    noising_summary: dict[int, dict],
    best_layer: int,
    thresh: float = 0.5,
    stat: str = "median",
) -> dict:
    """H8f-5: noising の最良層±1 で KL 乖離の過半 (recovery>=0.5) を再現するか (median)."""
    candidate = [best_layer - 1, best_layer, best_layer + 1]
    layers = [
        li
        for li in candidate
        if li in noising_summary and noising_summary[li].get(stat) is not None
    ]
    if best_layer not in noising_summary or noising_summary[best_layer].get(stat) is None:
        return {"layers": layers, "supported": None, "mean_at_best": None}
    mean_at_best = noising_summary[best_layer][stat]
    band_means = {li: noising_summary[li][stat] for li in layers}
    return {
        "layers": layers,
        "band_means": band_means,
        "mean_at_best": mean_at_best,
        "thresh": thresh,
        "supported": bool(mean_at_best >= thresh),
    }

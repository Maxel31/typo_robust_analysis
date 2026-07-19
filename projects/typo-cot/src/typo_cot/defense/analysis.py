"""校正後評価の集計: flip サブセットと R_Q 偏在 (Mann-Whitney).

rebuttal の analyze_spellfix.py の集計部をライブラリ化・一般化したもの
(spellfix 固定の命名を corrected に一般化。3段の校正器で共用)。
"""

from collections import defaultdict

import numpy as np
from scipy import stats as sstats


def flip_stats(rows: list[dict]) -> dict:
    """flip 率の集計 (analyze_spellfix.py:flip_stats と同一、命名のみ一般化).

    Args:
        rows: {"flip_perturbed": bool, "flip_corrected": bool} を含むレコード
    """
    n = len(rows)
    f_pert = sum(r["flip_perturbed"] for r in rows)
    f_corr = sum(r["flip_corrected"] for r in rows)
    return {
        "n": n,
        "flips_perturbed": f_pert,
        "flips_corrected": f_corr,
        "flip_rate_perturbed": f_pert / n if n else None,
        "flip_rate_corrected": f_corr / n if n else None,
    }


def restoration_subsets(per_sample: list[dict]) -> dict:
    """復元状態別のサブセット flip 集計 (rebuttal と同一の5分割).

    Args:
        per_sample: classify_restoration 由来のフィールド
            (fully_restored / all_perturbed_restored / n_collateral) と
            flip_perturbed / flip_corrected を含むレコード
    """
    return {
        "all": flip_stats(per_sample),
        "fully_restored": flip_stats(
            [r for r in per_sample if r["fully_restored"]]
        ),
        "all_perturbed_restored_not_full": flip_stats(
            [
                r
                for r in per_sample
                if r["all_perturbed_restored"] and not r["fully_restored"]
            ]
        ),
        "partially_or_not_restored": flip_stats(
            [r for r in per_sample if not r["all_perturbed_restored"]]
        ),
        "all_restored_with_collateral": flip_stats(
            [
                r
                for r in per_sample
                if r["all_perturbed_restored"] and r["n_collateral"] > 0
            ]
        ),
    }


def token_rq_comparison(token_records: list[dict], max_rank: int = 4) -> dict:
    """復元失敗語 vs 成功語の R_Q 分布比較 (Mann-Whitney).

    Args:
        token_records: {"sample_id", "importance_score", "restored"} のレコード
        max_rank: サンプル内 R_Q 降順ランク別失敗率を出す最大ランク
    """
    rq_rest = [t["importance_score"] for t in token_records if t["restored"]]
    rq_fail = [t["importance_score"] for t in token_records if not t["restored"]]

    by_rank: dict[int, list[int]] = {}
    per_sid = defaultdict(list)
    for t in token_records:
        per_sid[t["sample_id"]].append(t)
    for ts in per_sid.values():
        ts_sorted = sorted(ts, key=lambda t: -t["importance_score"])
        for rank, t in enumerate(ts_sorted, 1):
            by_rank.setdefault(rank, [0, 0])
            by_rank[rank][0] += not t["restored"]
            by_rank[rank][1] += 1
    fail_by_rank = {
        f"rank{k}": {"fail": v[0], "n": v[1], "fail_rate": v[0] / v[1]}
        for k, v in sorted(by_rank.items())
        if k <= max_rank
    }

    mw = (
        sstats.mannwhitneyu(rq_fail, rq_rest, alternative="two-sided")
        if rq_fail and rq_rest
        else None
    )
    n_total = len(token_records)
    return {
        "n_matched_tokens": n_total,
        "n_restored": len(rq_rest),
        "n_failed": len(rq_fail),
        "restoration_rate": len(rq_rest) / n_total if n_total else None,
        "mean_rq_restored": float(np.mean(rq_rest)) if rq_rest else None,
        "mean_rq_failed": float(np.mean(rq_fail)) if rq_fail else None,
        "median_rq_restored": float(np.median(rq_rest)) if rq_rest else None,
        "median_rq_failed": float(np.median(rq_fail)) if rq_fail else None,
        "mannwhitney_p": float(mw.pvalue) if mw else None,
        "fail_rate_by_within_sample_rank": fail_by_rank,
    }

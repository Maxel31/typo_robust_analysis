"""実験2: 統計層 — McNemar・対応リスク差 CI・用量反応・腕別集計 (層分離).

共通規約 (§3.4-3): 対応比較 = McNemar + リスク差 CI / CI = bootstrap /
用量反応の単調性・回復曲線 = 並べ替え検定。
主推定量 (§3.4-2): clean 正解条件付き flip 率 (全サンプル版は副次)。
数値層 (numeric) は content 層と分離して集計する (§4 実験2-1)。
"""

import math
import random

# 対比ペア (arm_a の target_kind, arm_b の target_kind)。同一 (op, k, stratum)
# セル内で両腕が揃ったときに McNemar + リスク差 CI を計算する。
# 2ペア目は 2026-07-15 決定の主対比 (無制限 top vs 層内マッチランダム)。
CONTRAST_PAIRS = (
    ("top_rc", "matched_random"),
    ("top_rc_unrestricted", "stratum_matched_random"),
)


def mcnemar_exact(flips_a: list[bool], flips_b: list[bool]) -> dict:
    """対応のある2値系列の McNemar 厳密検定 (二項、両側).

    Args:
        flips_a / flips_b: 同一サンプル順の flip フラグ

    Returns:
        {"n", "b", "c", "p_value"} — b = a のみ flip、c = b のみ flip
    """
    if len(flips_a) != len(flips_b):
        raise ValueError("flips_a and flips_b must have the same length")
    b = sum(1 for x, y in zip(flips_a, flips_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(flips_a, flips_b, strict=True) if not x and y)
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / 2**n_disc
        p = min(1.0, 2 * tail)
    return {"n": len(flips_a), "b": b, "c": c, "p_value": p}


def paired_risk_difference(
    flips_a: list[bool],
    flips_b: list[bool],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """対応リスク差 (P_a - P_b) と percentile bootstrap 95% CI."""
    if len(flips_a) != len(flips_b):
        raise ValueError("flips_a and flips_b must have the same length")
    n = len(flips_a)
    if n == 0:
        return {"rd": None, "ci_low": None, "ci_high": None, "n": 0}
    diffs = [int(x) - int(y) for x, y in zip(flips_a, flips_b, strict=True)]
    rd = sum(diffs) / n
    rng = random.Random(seed)
    boots = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"rd": rd, "ci_low": lo, "ci_high": hi, "n": n}


def dose_trend_test(
    flips_by_dose: dict[int, dict[str, bool]],
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """用量反応の単調性の並べ替え検定 (片側: 傾き > 0).

    統計量 = 用量別 flip 率の k に対する最小二乗傾き。帰無分布はサンプル内で
    用量ラベルを並べ替えて生成 (全用量に存在するサンプルのみ使用)。

    Args:
        flips_by_dose: {k: {sample_id: flip}}
    """
    doses = sorted(flips_by_dose)
    if len(doses) < 2:
        raise ValueError("need at least 2 dose levels")
    common = sorted(set.intersection(*(set(flips_by_dose[d]) for d in doses)))
    matrix = [[bool(flips_by_dose[d][sid]) for d in doses] for sid in common]

    def _slope(mat: list[list[bool]]) -> float:
        rates = [sum(row[j] for row in mat) / len(mat) for j in range(len(doses))]
        xbar = sum(doses) / len(doses)
        ybar = sum(rates) / len(rates)
        num = sum((x - xbar) * (y - ybar) for x, y in zip(doses, rates, strict=True))
        den = sum((x - xbar) ** 2 for x in doses)
        return num / den

    if not matrix:
        return {"slope": None, "p_value": None, "rates": {}, "n": 0}

    observed = _slope(matrix)
    rates = {
        d: sum(row[j] for row in matrix) / len(matrix) for j, d in enumerate(doses)
    }
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        permuted = []
        for row in matrix:
            shuffled = list(row)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        if _slope(permuted) >= observed - 1e-12:
            count += 1
    return {
        "slope": observed,
        "p_value": (count + 1) / (n_perm + 1),
        "rates": rates,
        "n": len(matrix),
    }


def _arm_flip_maps(records: list[dict], arm_name: str) -> tuple[dict, dict]:
    """腕の flip マップ (全サンプル / clean 正解のみ) を作る."""
    all_flips: dict[str, bool] = {}
    cc_flips: dict[str, bool] = {}
    for rec in records:
        if rec.get("skip_reason") is not None:
            continue
        arm = rec.get("arms", {}).get(arm_name)
        if arm is None or arm.get("skip_reason") is not None or arm.get("flip") is None:
            continue
        sid = rec["sample_id"]
        all_flips[sid] = bool(arm["flip"])
        if rec.get("clean_correct"):
            cc_flips[sid] = bool(arm["flip"])
    return all_flips, cc_flips


def aggregate_results(
    records: list[dict],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """record リストから腕別集計 (層分離) とコア対比統計を作る.

    - 主推定量: clean 正解サンプル条件付き flip 率 (`flip_rate`)。
      全サンプル版は `flip_rate_all` (副次)。
    - strata.content / strata.numeric に腕を分離。
    - contrasts: 同一 (op, k, stratum) の top_rc vs matched_random について
      両腕評価済み ∩ clean 正解のサンプルで McNemar + リスク差 CI。
    """
    evaluated = [r for r in records if r.get("skip_reason") is None]
    n_skipped = len(records) - len(evaluated)

    baselines = [r["baseline"] for r in evaluated if r.get("baseline")]
    baseline_summary = {
        "n": len(baselines),
        "matches_archive_rate": (
            sum(1 for b in baselines if b.get("matches_archive")) / len(baselines)
            if baselines
            else None
        ),
        "correct_rate": (
            sum(1 for b in baselines if b.get("is_correct")) / len(baselines)
            if baselines
            else None
        ),
    }

    # 腕メタデータの収集 (records 側から復元 — 実行時の腕構成に依存しない)
    arm_meta: dict[str, dict] = {}
    for rec in evaluated:
        for name, arm in rec.get("arms", {}).items():
            if name not in arm_meta:
                arm_meta[name] = {
                    "target_kind": arm["target_kind"],
                    "op": arm["op"],
                    "k": arm["k"],
                    "stratum": arm["stratum"],
                }

    strata: dict[str, dict] = {}
    flip_maps: dict[str, tuple[dict, dict]] = {}
    for name, meta in arm_meta.items():
        all_flips, cc_flips = _arm_flip_maps(evaluated, name)
        flip_maps[name] = (all_flips, cc_flips)
        n_arm_skipped = sum(
            1
            for rec in evaluated
            if rec.get("arms", {}).get(name, {}).get("skip_reason") is not None
        )
        summary = {
            **meta,
            "n": len(cc_flips),
            "n_flip": sum(cc_flips.values()),
            "flip_rate": (
                sum(cc_flips.values()) / len(cc_flips) if cc_flips else None
            ),
            "n_all": len(all_flips),
            "n_flip_all": sum(all_flips.values()),
            "flip_rate_all": (
                sum(all_flips.values()) / len(all_flips) if all_flips else None
            ),
            "n_skipped": n_arm_skipped,
        }
        strata.setdefault(meta["stratum"], {"arms": {}})["arms"][name] = summary

    # 対比: 同一 (op, k, stratum) セル内で CONTRAST_PAIRS の両腕が揃うもの
    contrasts: list[dict] = []
    by_cell: dict[tuple, dict[str, str]] = {}
    for name, meta in arm_meta.items():
        cell = (meta["op"], meta["k"], meta["stratum"])
        by_cell.setdefault(cell, {})[meta["target_kind"]] = name
    for cell, kinds in sorted(by_cell.items(), key=str):
        for kind_a, kind_b in CONTRAST_PAIRS:
            if kind_a not in kinds or kind_b not in kinds:
                continue
            name_a, name_b = kinds[kind_a], kinds[kind_b]
            cc_a, cc_b = flip_maps[name_a][1], flip_maps[name_b][1]
            shared = sorted(set(cc_a) & set(cc_b))
            if not shared:
                continue
            flags_a = [cc_a[s] for s in shared]
            flags_b = [cc_b[s] for s in shared]
            mc = mcnemar_exact(flags_a, flags_b)
            rd = paired_risk_difference(flags_a, flags_b, n_boot=n_boot, seed=seed)
            contrasts.append(
                {
                    "arm_a": name_a,
                    "arm_b": name_b,
                    "op": cell[0],
                    "k": cell[1],
                    "stratum": cell[2],
                    "n_paired": len(shared),
                    "mcnemar_p": mc["p_value"],
                    "mcnemar_b": mc["b"],
                    "mcnemar_c": mc["c"],
                    "risk_difference": rd["rd"],
                    "rd_ci95": [rd["ci_low"], rd["ci_high"]],
                }
            )

    return {
        "n_records": len(records),
        "n_evaluated": len(evaluated),
        "n_skipped": n_skipped,
        "n_clean_correct": sum(1 for r in evaluated if r.get("clean_correct")),
        "n_residual_answer_in_prefix": sum(
            1 for r in evaluated if r.get("residual_answer_in_prefix")
        ),
        "baseline": baseline_summary,
        "strata": strata,
        "contrasts": contrasts,
    }

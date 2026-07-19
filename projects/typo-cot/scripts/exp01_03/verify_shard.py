#!/usr/bin/env python3
"""完了シャードの検証レポート (読み取り専用).

スモーク合格水準との比較用に以下を出力する:
  - TE 再現一致率: 非除外限定 (判定母集団の主基準) と全件比 (参考値)
  - 4セル flip 表 (TE/DE/IE, headline restore rate) と除外内訳
  - KL 集中度: KL 上位10%位置の KL 和が全体に占める割合 (サンプル平均/中央値/最小)
  - divergence の alignment 成功率・onset 率・precision@10 (summary 経由)

使い方:
    python scripts/exp01_03/verify_shard.py results/exp01_03/<shard> [<shard2> ...]
"""

import json
import math
import sys
from pathlib import Path


def kl_concentration(kl: list[float], frac: float = 0.1) -> float | None:
    """KL 上位 frac 位置の KL 和が全体に占める割合 (スモークの集中度指標)."""
    if not kl:
        return None
    total = sum(kl)
    if total <= 0:
        return None
    k = max(1, math.ceil(len(kl) * frac))
    top = sum(sorted(kl, reverse=True)[:k])
    return top / total


def verify(shard_dir: Path) -> dict:
    with open(shard_dir / "summary.json") as f:
        summary = json.load(f)
    with open(shard_dir / "outcomes.json") as f:
        outcomes = json.load(f)

    non_excluded = [o for o in outcomes if not o["exclude"]]
    te_known = [o for o in non_excluded if o.get("te_match") is not None]
    te_match_nonexcl = (
        sum(1 for o in te_known if o["te_match"]) / len(te_known) if te_known else None
    )
    te_all_known = [o for o in outcomes if o.get("te_match") is not None]
    te_match_all = (
        sum(1 for o in te_all_known if o["te_match"]) / len(te_all_known)
        if te_all_known
        else None
    )

    concentrations = []
    n_div_ok = 0
    div_dir = shard_dir / "divergence"
    if div_dir.is_dir():
        for p in div_dir.glob("*.json"):
            with open(p) as f:
                rec = json.load(f)
            if not rec.get("ok"):
                continue
            n_div_ok += 1
            if rec.get("n_positions", 0) >= 10:
                c = kl_concentration(rec.get("kl", []))
                if c is not None:
                    concentrations.append(c)

    concentrations.sort()
    conc_stats = None
    if concentrations:
        n = len(concentrations)
        conc_stats = {
            "n": n,
            "mean": sum(concentrations) / n,
            "median": concentrations[n // 2],
            "min": concentrations[0],
        }

    ft = summary["flip_table"]
    return {
        "shard": shard_dir.name,
        "n_total": ft["n_total"],
        "n_excluded": ft["n_excluded"],
        "n_a_incorrect": ft["n_a_incorrect"],
        "n_included": ft["n_included"],
        "te_match_rate_nonexcluded": te_match_nonexcl,
        "te_match_n_nonexcluded": len(te_known),
        "te_match_rate_all": te_match_all,
        "flip_rate": ft["flip_rate"],
        "flip_count": ft["flip_count"],
        "headline_restore_rate": ft.get("headline_restore_rate"),
        "ie_flip_rate_given_cot_changed": ft.get("ie_flip_rate_given_cot_changed"),
        "exclusion_reasons": summary.get("exclusion_reasons"),
        "kl_top10pct_concentration": conc_stats,
        "n_divergence_ok_files": n_div_ok,
        "divergence_summary": summary.get("divergence"),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        print(json.dumps(verify(Path(arg)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

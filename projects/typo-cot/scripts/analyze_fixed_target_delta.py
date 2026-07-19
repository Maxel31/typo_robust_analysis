#!/usr/bin/env python3
"""実験4: Δρ 全設定表 (付録用) の生成.

default (k4_importance) と fixed_target の analyzer 出力 full_results.json の
ペアから、CoT:Jaccard@{5,10,20} × flip の偏相関 ρ(J|R) を両条件で計算し、
- bootstrap 95%CI (B=10,000 デフォルト)
- Δρ = ρ_fixed − ρ_default の paired bootstrap 検定
- Holm 補正 (k 水準ごとに、全設定の fixed 側 p 値と Δρ p 値それぞれへ適用)
を含むテーブル (JSON + CSV) を出力する。

使用例 (rebuttal 4設定での検証):
  uv run python scripts/analyze_fixed_target_delta.py \
    --pair gemma-3-4b-it_gsm8k $A/gsm8k/gemma-3-4b-it/k4_importance/full_results.json \
                               $A/gsm8k/gemma-3-4b-it/k4_fixed_target/full_results.json \
    --pair ... \
    --output_dir results/fixed_target_delta
"""

import argparse
import csv
import json
from pathlib import Path

from typo_cot.analysis.fixed_stats import (
    bootstrap_partial_corr_ci,
    holm_adjust,
    join_fixed_default_records,
    paired_bootstrap_delta_rho,
    partial_corr_flip,
)

KS = ("top5", "top10", "top20")


def main() -> None:
    parser = argparse.ArgumentParser(description="Δρ 全設定表の生成 (実験4)")
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True,
        metavar=("NAME", "DEFAULT_JSON", "FIXED_JSON"),
        help="設定名, default 条件の full_results.json, fixed 条件の full_results.json",
    )
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for name, default_path, fixed_path in args.pair:
        with open(default_path, encoding="utf-8") as f:
            dflt = json.load(f)
        with open(fixed_path, encoding="utf-8") as f:
            fixd = json.load(f)
        for k in KS:
            joined = join_fixed_default_records(
                dflt["sample_results"], fixd["sample_results"], k=k
            )
            r_def, p_def, n = partial_corr_flip(
                joined["j_default"], joined["flip"], joined["rouge_default"]
            )
            r_fix, p_fix, _ = partial_corr_flip(
                joined["j_fixed"], joined["flip"], joined["rouge_fixed"]
            )
            ci_def = bootstrap_partial_corr_ci(
                joined["j_default"], joined["flip"], joined["rouge_default"],
                n_boot=args.n_boot, seed=args.seed,
            )
            ci_fix = bootstrap_partial_corr_ci(
                joined["j_fixed"], joined["flip"], joined["rouge_fixed"],
                n_boot=args.n_boot, seed=args.seed,
            )
            delta = paired_bootstrap_delta_rho(
                joined["j_default"], joined["j_fixed"], joined["flip"],
                joined["rouge_default"], rouge_fixed=joined["rouge_fixed"],
                n_boot=args.n_boot, seed=args.seed,
            )
            rows.append({
                "setting": name,
                "k": k,
                "n": n,
                "rho_default": r_def,
                "rho_default_p": p_def,
                "rho_default_ci95": list(ci_def),
                "rho_fixed": r_fix,
                "rho_fixed_p": p_fix,
                "rho_fixed_ci95": list(ci_fix),
                "delta_rho": delta["delta_rho"],
                "delta_rho_ci95": list(delta["ci95"]),
                "delta_rho_p": delta["p_value"],
            })
            print(
                f"{name} {k}: n={n} rho_def={r_def:+.4f} rho_fix={r_fix:+.4f} "
                f"delta={delta['delta_rho']:+.4f} "
                f"CI[{delta['ci95'][0]:+.3f},{delta['ci95'][1]:+.3f}] p={delta['p_value']:.4g}"
            )

    # Holm 補正: k 水準ごとに全設定へ適用
    for k in KS:
        idxs = [i for i, r in enumerate(rows) if r["k"] == k]
        for key_p, key_h in [
            ("rho_fixed_p", "rho_fixed_p_holm"),
            ("delta_rho_p", "delta_rho_p_holm"),
        ]:
            adj = holm_adjust([rows[i][key_p] for i in idxs])
            for i, a in zip(idxs, adj, strict=True):
                rows[i][key_h] = a

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n_boot": args.n_boot, "seed": args.seed, "ks": list(KS)}
    with open(out_dir / "delta_rho_table.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys())
    with open(out_dir / "delta_rho_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"saved: {out_dir}/delta_rho_table.json / .csv")


if __name__ == "__main__":
    main()

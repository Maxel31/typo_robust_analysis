#!/usr/bin/env python3
"""実験5: LXT-4 vs Matched-Rnd-4 の精度低下差の検定 (McNemar + リスク差 CI).

clean / LXT-4 / Matched-Rnd-4 の results.json (run_inference.py /
run_generation_only.py の出力) を sample_id で対応付け、
- 各条件の accuracy と対 clean の低下幅
- Matched-Rnd-4 vs LXT-4 の対応のある McNemar exact 検定とリスク差 95% CI
を計算する。主推定量の慣行に合わせ、clean 正解サンプルに条件付けた
correct->incorrect 版も併記する。

使用例:
  uv run python scripts/exp5/analyze_matched_control.py \
    --clean_results  <baseline_dir>/results.json \
    --lxt_results    <lxt4_output_dir>/results.json \
    --matched_results <matched_rnd_output_dir>/results.json \
    --output results/smoke/matched_control_gsm8k.json
"""

import argparse
import json
from pathlib import Path

from typo_cot.analysis.matched_control import paired_condition_comparison


def load_correct_map(path: Path) -> dict[str, bool]:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return {r["sample_id"]: bool(r.get("is_correct", False)) for r in results}


def main() -> None:
    parser = argparse.ArgumentParser(description="実験5 マッチド統制の統計")
    parser.add_argument("--clean_results", type=str, required=True)
    parser.add_argument("--lxt_results", type=str, required=True)
    parser.add_argument("--matched_results", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    clean = load_correct_map(Path(args.clean_results))
    lxt = load_correct_map(Path(args.lxt_results))
    matched = load_correct_map(Path(args.matched_results))

    common = [sid for sid in clean if sid in lxt and sid in matched]
    c = [clean[sid] for sid in common]
    lxt_flags = [lxt[sid] for sid in common]
    m = [matched[sid] for sid in common]

    def acc(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    # A=Matched-Rnd, B=LXT: risk_diff>0 なら LXT の方が精度を下げている
    comparison = paired_condition_comparison(m, lxt_flags)

    # clean 正解サンプルに条件付けた correct->incorrect (主推定量の慣行)
    cond_ids = [i for i, sid in enumerate(common) if c[i]]
    comparison_cond = (
        paired_condition_comparison(
            [m[i] for i in cond_ids], [lxt_flags[i] for i in cond_ids]
        )
        if cond_ids
        else None
    )

    report = {
        "clean_results": args.clean_results,
        "lxt_results": args.lxt_results,
        "matched_results": args.matched_results,
        "n_paired": len(common),
        "acc": {"clean": acc(c), "lxt4": acc(lxt_flags), "matched_rnd4": acc(m)},
        "drop_vs_clean": {
            "lxt4": acc(c) - acc(lxt_flags),
            "matched_rnd4": acc(c) - acc(m),
        },
        "matched_vs_lxt": comparison,
        "matched_vs_lxt_clean_correct_only": comparison_cond,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"保存: {out}")


if __name__ == "__main__":
    main()

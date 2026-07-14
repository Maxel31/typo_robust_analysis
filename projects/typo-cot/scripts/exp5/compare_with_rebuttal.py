#!/usr/bin/env python3
"""実験5: 新 Matched-Rnd-4 (5変数マッチ) と rebuttal 版 (品詞・文字長マッチ) の差分確認.

同一サンプルについて、両者が摂動したトークン集合の重なり・LXT-4 標的の
除外整合を集計し、目視確認用サンプルを表示する。

使用例:
  uv run python scripts/exp5/compare_with_rebuttal.py \
    --ours data/exp5/matched_rnd/gemma-3-4b-it_mmlu_k4_matched_rnd \
    --rebuttal /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/datasets/rebuttal/gemma-3-4b-it_mmlu_k4_matched_random \
    --output results/smoke/compare_mmlu.json --show 10
"""

import argparse
import json
from pathlib import Path

from typo_cot.perturbation.dataset import PerturbedDataset


def token_set(sample) -> set[str]:
    return {pt.original_token for pt in sample.perturbed_tokens}


def index_set(sample) -> set[int]:
    return {pt.token_index for pt in sample.perturbed_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description="rebuttal 版 matched-random との差分確認")
    parser.add_argument("--ours", type=str, required=True, help="新 Matched-Rnd データセットディレクトリ")
    parser.add_argument("--rebuttal", type=str, required=True, help="rebuttal 版データセットディレクトリ")
    parser.add_argument("--output", type=str, default=None, help="比較結果 JSON の出力先")
    parser.add_argument("--show", type=int, default=10, help="目視確認サンプル数")
    args = parser.parse_args()

    ours = PerturbedDataset.load(Path(args.ours) / "perturbed_dataset.json")
    rebuttal = PerturbedDataset.load(Path(args.rebuttal) / "perturbed_dataset.json")

    ours_by_id = {s.sample_id: s for s in ours.samples}
    reb_by_id = {s.sample_id: s for s in rebuttal.samples}
    common_ids = [sid for sid in ours_by_id if sid in reb_by_id]

    overlaps = []
    n_identical = 0
    for sid in common_ids:
        o, r = ours_by_id[sid], reb_by_id[sid]
        oi, ri = index_set(o), index_set(r)
        union = oi | ri
        j = len(oi & ri) / len(union) if union else 1.0
        overlaps.append(j)
        if oi == ri:
            n_identical += 1

    summary = {
        "ours": str(args.ours),
        "rebuttal": str(args.rebuttal),
        "ours_mode": ours.metadata.get("perturbation_mode"),
        "rebuttal_mode": rebuttal.metadata.get("perturbation_mode"),
        "n_ours": len(ours.samples),
        "n_rebuttal": len(rebuttal.samples),
        "n_common": len(common_ids),
        "mean_token_index_jaccard": (
            sum(overlaps) / len(overlaps) if overlaps else None
        ),
        "identical_selection_rate": (
            n_identical / len(common_ids) if common_ids else None
        ),
    }

    # SMD 表があれば併記
    for label, d in (("ours", args.ours), ("rebuttal", args.rebuttal)):
        stats_path = Path(d) / "matched_stats.json"
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                st = json.load(f)
            summary[f"{label}_stats"] = (
                st.get("smd_table") or st.get("aggregate") or None
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 目視確認サンプル
    print("\n=== 目視確認サンプル ===")
    for sid in common_ids[: args.show]:
        o, r = ours_by_id[sid], reb_by_id[sid]
        print(f"\n[{sid}]")
        print(f"  ours     ({len(o.perturbed_tokens)}): "
              + ", ".join(f"{pt.original_token!r}->{pt.perturbed_token!r}" for pt in o.perturbed_tokens))
        print(f"  rebuttal ({len(r.perturbed_tokens)}): "
              + ", ".join(f"{pt.original_token!r}->{pt.perturbed_token!r}" for pt in r.perturbed_tokens))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n比較結果を保存: {out}")


if __name__ == "__main__":
    main()

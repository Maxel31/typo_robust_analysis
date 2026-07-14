#!/usr/bin/env python3
"""実験7: 校正後評価の詳細集計 (rebuttal analyze_spellfix.py の3段一般化).

baseline / LXT-4 摂動 / 校正後 の3条件の results.json を突き合わせ、
(a) 復元率 (単語・全文) (b) 復元サブセット別の残存 flip 率
(c) 復元失敗の高 R_Q トークン集中 (Mann-Whitney)
(d) clean 語の誤訂正率と誤訂正起因 flip (e) 全体 flip 率 before/after
を算出する。flip = baseline の extracted_answer と当該条件の不一致。

使用例:
  uv run python scripts/exp7/analyze_correction.py \
    --baseline_dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
    --perturbed_dir <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
    --corrected_dir results/exp7/perturbed/gemma-3-4b-it_gsm8k_k4_llmfix \
    --corrected_dataset data/exp7/corrected/gemma-3-4b-it_gsm8k_k4_llmfix/perturbed_dataset.json \
    --output results/exp7/analysis/correction_analysis_llm_gemma-3-4b-it_gsm8k.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from typo_cot.defense.analysis import restoration_subsets, token_rq_comparison
from typo_cot.defense.restoration import build_reference, classify_restoration


def load_results(d: str) -> dict:
    with open(Path(d) / "results.json", encoding="utf-8") as f:
        return {r["sample_id"]: r for r in json.load(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description="校正後評価の詳細集計 (実験7)")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--perturbed_dir", required=True)
    parser.add_argument("--corrected_dir", required=True)
    parser.add_argument("--corrected_dataset", required=True,
                        help="make_corrected_dataset.py が出力した perturbed_dataset.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = load_results(args.baseline_dir)
    pert = load_results(args.perturbed_dir)
    corr = load_results(args.corrected_dir)
    with open(args.corrected_dataset, encoding="utf-8") as f:
        ds_data = json.load(f)
    ds = {s["sample_id"]: s for s in ds_data["samples"]}
    corrector = ds_data.get("metadata", {}).get("corrector", "unknown")

    common = sorted(set(base) & set(pert) & set(corr) & set(ds))

    per_sample = []
    token_records = []

    for sid in common:
        b, p, x, s = base[sid], pert[sid], corr[sid], ds[sid]
        ref = build_reference(s["original_question"], s.get("choices"))
        pert_q = p["question"]   # 摂動文 (選択肢込み)
        corr_q = x["question"]   # 校正後文

        r = classify_restoration(ref, pert_q, corr_q)

        flip_pert = p["extracted_answer"] != b["extracted_answer"]
        flip_corr = x["extracted_answer"] != b["extracted_answer"]

        per_sample.append(
            {
                "sample_id": sid,
                "n_perturbed_words": r.n_perturbed_words,
                "n_restored": r.n_restored,
                "fully_restored": r.fully_restored,
                "all_perturbed_restored": r.all_perturbed_restored,
                "n_collateral": r.n_collateral,
                "flip_perturbed": flip_pert,
                "flip_corrected": flip_corr,
                "base_correct": bool(b["is_correct"]),
                "pert_correct": bool(p["is_correct"]),
                "corr_correct": bool(x["is_correct"]),
            }
        )

        # トークンレベル: perturbed_tokens の R_Q と復元可否の対応付け
        # (analyze_spellfix.py:127-150 と同一のヒューリスティック)
        records = list(s.get("perturbed_tokens", []))
        used = set()
        for ow, _pw, restored in r.restored_flags:
            cand_idx = None
            best_len = -1
            for ri, rec in enumerate(records):
                if ri in used:
                    continue
                surf = (
                    rec["original_token"].replace("▁", "").replace("Ġ", "").strip()
                )
                if surf and surf.lower() in ow.lower() and len(surf) > best_len:
                    cand_idx = ri
                    best_len = len(surf)
            if cand_idx is not None:
                used.add(cand_idx)
                token_records.append(
                    {
                        "sample_id": sid,
                        "importance_score": records[cand_idx]["importance_score"],
                        "restored": restored,
                    }
                )

    acc = {
        "baseline": float(np.mean([r["base_correct"] for r in per_sample])),
        "perturbed": float(np.mean([r["pert_correct"] for r in per_sample])),
        "corrected": float(np.mean([r["corr_correct"] for r in per_sample])),
    }
    subsets = restoration_subsets(per_sample)
    token_level = token_rq_comparison(token_records)

    n_coll = sum(1 for r in per_sample if r["n_collateral"] > 0)
    collateral = {
        "samples_with_collateral": n_coll,
        "collateral_sample_rate": n_coll / len(per_sample) if per_sample else None,
        "total_collateral_words": int(sum(r["n_collateral"] for r in per_sample)),
        "new_flip_rate_when_all_restored_with_collateral":
            subsets["all_restored_with_collateral"]["flip_rate_corrected"],
    }

    out = {
        "corrector": corrector,
        "n_common_samples": len(per_sample),
        "accuracy": acc,
        "flip_subsets": subsets,
        "token_level_rq": token_level,
        "collateral": collateral,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "per_sample": per_sample}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuttal 実験②: Spell-Correction Restoration の詳細集計 (N5Yq #1).

baseline / LXT-4 摂動 / スペル訂正後 の 3 条件の results.json を突き合わせ、
rebuttal 要求の (a)-(e) を算出する:
(a) 復元率 (単語・全文) ... make_spellfix_dataset.py の restoration_stats を再計算・拡張
(b) 復元サブセット別の残存 flip 率 (fully_restored / all_perturbed_restored / partial)
(c) 復元失敗の高 R_Q トークンへの集中 (摂動 4 トークン内の R_Q 順位・スコア比較)
(d) clean 語の誤訂正率と、摂動語全復元かつ誤訂正ありサンプルの flip 率 (誤訂正起因 flip)
(e) 全体 flip 率 before (LXT-4) / after (spellfix)

flip = baseline の extracted_answer と当該条件の extracted_answer の不一致
(analysis/analyzer.py:887-895 の answer_changed と同一定義)。

使用例:
  uv run --no-sync python scripts/rebuttal/analyze_spellfix.py \
    --baseline_dir outputs/baseline/gemma-3-4b-it_mmlu \
    --perturbed_dir outputs/perturbed/gemma-3-4b-it_mmlu_k4_importance \
    --spellfix_dir outputs/rebuttal/perturbed/gemma-3-4b-it_mmlu_k4_spellfix \
    --spellfix_dataset datasets/rebuttal/gemma-3-4b-it_mmlu_k4_spellfix/perturbed_dataset.json \
    --output outputs/rebuttal/spellfix_analysis_gemma-3-4b-it_mmlu.json
"""

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

LETTERS = "ABCDEFGHIJ"


def build_reference(original_question: str, choices: list | None) -> str:
    """perturbed_question と同形式の参照テキスト (dataset.py:593-597 と同じ)."""
    if choices:
        options = " ".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
        return f"{original_question}\n{options}"
    return original_question


def aligned_changes(ref: str, hyp: str):
    """空白区切り語列の difflib 対応付け (同数 replace のみ位置対応)."""
    rw, hw = ref.split(), hyp.split()
    out = []
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for off in range(i2 - i1):
                out.append((j1 + off, rw[i1 + off], hw[j1 + off]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spell-fix 詳細集計")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--perturbed_dir", required=True)
    parser.add_argument("--spellfix_dir", required=True)
    parser.add_argument("--spellfix_dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def load_results(d):
        with open(Path(d) / "results.json", encoding="utf-8") as f:
            return {r["sample_id"]: r for r in json.load(f)}

    base = load_results(args.baseline_dir)
    pert = load_results(args.perturbed_dir)
    fix = load_results(args.spellfix_dir)
    with open(args.spellfix_dataset, encoding="utf-8") as f:
        ds = {s["sample_id"]: s for s in json.load(f)["samples"]}

    common = sorted(set(base) & set(pert) & set(fix) & set(ds))

    per_sample = []
    token_records = []

    for sid in common:
        b, p, x, s = base[sid], pert[sid], fix[sid], ds[sid]
        ref = build_reference(s["original_question"], s.get("choices"))
        pert_q = p["question"]          # 摂動文 (選択肢込み)
        fix_q = x["question"]           # 訂正後文

        # 摂動位置の対応付け (ref vs pert) と訂正後の復元判定
        changes = aligned_changes(ref, pert_q)
        fix_words = fix_q.split()
        ref_words = ref.split()

        n_pw = 0
        n_restored = 0
        tok_flags = []  # (orig_word, pert_word, restored)
        for j, ow, pw in changes:
            n_pw += 1
            restored = j < len(fix_words) and fix_words[j] == ow
            n_restored += restored
            tok_flags.append((ow, pw, restored))

        fully = " ".join(fix_words) == " ".join(ref_words)
        all_restored = n_pw > 0 and n_restored == n_pw

        # 訂正後テキストの clean 語誤訂正 (摂動位置以外での ref との相違)
        fix_changes = aligned_changes(ref, fix_q)
        pert_positions = {j for j, _, _ in changes}
        collateral = [c for c in fix_changes if c[0] not in pert_positions]

        flip_pert = p["extracted_answer"] != b["extracted_answer"]
        flip_fix = x["extracted_answer"] != b["extracted_answer"]

        per_sample.append(
            {
                "sample_id": sid,
                "n_perturbed_words": n_pw,
                "n_restored": n_restored,
                "fully_restored": fully,
                "all_perturbed_restored": all_restored,
                "n_collateral": len(collateral),
                "flip_perturbed": flip_pert,
                "flip_spellfix": flip_fix,
                "base_correct": bool(b["is_correct"]),
                "pert_correct": bool(p["is_correct"]),
                "fix_correct": bool(x["is_correct"]),
            }
        )

        # トークンレベル: perturbed_tokens の R_Q スコアと復元可否の対応付け
        # 各語変化 (ow -> pw) に対し、original_token (マーカー除去) が ow の部分文字列
        # で未使用の摂動レコードを対応付ける (1文字編集なので基本的に一意)
        records = list(s.get("perturbed_tokens", []))
        used = set()
        for ow, pw, restored in tok_flags:
            cand_idx = None
            best_len = -1
            for ri, rec in enumerate(records):
                if ri in used:
                    continue
                surf = rec["original_token"].replace("▁", "").replace("Ġ", "").strip()
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

    def flip_stats(rows):
        n = len(rows)
        f_pert = sum(r["flip_perturbed"] for r in rows)
        f_fix = sum(r["flip_spellfix"] for r in rows)
        return {
            "n": n,
            "flips_perturbed": f_pert,
            "flips_spellfix": f_fix,
            "flip_rate_perturbed": f_pert / n if n else None,
            "flip_rate_spellfix": f_fix / n if n else None,
        }

    subsets = {
        "all": flip_stats(per_sample),
        "fully_restored": flip_stats([r for r in per_sample if r["fully_restored"]]),
        "all_perturbed_restored_not_full": flip_stats(
            [r for r in per_sample
             if r["all_perturbed_restored"] and not r["fully_restored"]]
        ),
        "partially_or_not_restored": flip_stats(
            [r for r in per_sample if not r["all_perturbed_restored"]]
        ),
        "all_restored_with_collateral": flip_stats(
            [r for r in per_sample
             if r["all_perturbed_restored"] and r["n_collateral"] > 0]
        ),
    }

    # accuracy 3条件
    acc = {
        "baseline": float(np.mean([r["base_correct"] for r in per_sample])),
        "perturbed_lxt4": float(np.mean([r["pert_correct"] for r in per_sample])),
        "spellfix": float(np.mean([r["fix_correct"] for r in per_sample])),
    }

    # (c) トークンレベル: 復元失敗と R_Q
    rq_rest = [t["importance_score"] for t in token_records if t["restored"]]
    rq_fail = [t["importance_score"] for t in token_records if not t["restored"]]
    # サンプル内順位 (1=最高R_Q) 別の失敗率
    by_rank = {}
    per_sid = defaultdict(list)
    for t in token_records:
        per_sid[t["sample_id"]].append(t)
    for sid, ts in per_sid.items():
        ts_sorted = sorted(ts, key=lambda t: -t["importance_score"])
        for rank, t in enumerate(ts_sorted, 1):
            by_rank.setdefault(rank, [0, 0])
            by_rank[rank][0] += (not t["restored"])
            by_rank[rank][1] += 1
    fail_by_rank = {
        f"rank{k}": {"fail": v[0], "n": v[1], "fail_rate": v[0] / v[1]}
        for k, v in sorted(by_rank.items()) if k <= 4
    }
    mw = (
        sstats.mannwhitneyu(rq_fail, rq_rest, alternative="two-sided")
        if rq_fail and rq_rest else None
    )
    token_level = {
        "n_matched_tokens": len(token_records),
        "n_restored": len(rq_rest),
        "n_failed": len(rq_fail),
        "restoration_rate": len(rq_rest) / len(token_records) if token_records else None,
        "mean_rq_restored": float(np.mean(rq_rest)) if rq_rest else None,
        "mean_rq_failed": float(np.mean(rq_fail)) if rq_fail else None,
        "median_rq_restored": float(np.median(rq_rest)) if rq_rest else None,
        "median_rq_failed": float(np.median(rq_fail)) if rq_fail else None,
        "mannwhitney_p": float(mw.pvalue) if mw else None,
        "fail_rate_by_within_sample_rank": fail_by_rank,
    }

    # (d) 誤訂正
    n_coll_samples = sum(1 for r in per_sample if r["n_collateral"] > 0)
    collateral = {
        "samples_with_collateral": n_coll_samples,
        "collateral_sample_rate": n_coll_samples / len(per_sample),
        "total_collateral_words": int(sum(r["n_collateral"] for r in per_sample)),
        "new_flip_rate_when_all_restored_with_collateral":
            subsets["all_restored_with_collateral"]["flip_rate_spellfix"],
    }

    out = {
        "n_common_samples": len(per_sample),
        "accuracy": acc,
        "flip_subsets": subsets,
        "token_level_rq": token_level,
        "collateral": collateral,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "per_sample": per_sample}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

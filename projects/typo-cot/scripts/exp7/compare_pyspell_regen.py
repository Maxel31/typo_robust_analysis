#!/usr/bin/env python3
"""実験7: rebuttal 2設定の統一再生成の検証レポートを作成する.

3つの検証を1ファイルの JSON にまとめる:
  1. 再現性: 同一入力で2回実行した出力 (run1/run2) の samples が byte 一致するか
  2. 旧 rebuttal ログとの差分: 校正後テキストが変わったサンプル数と、
     摂動語位置における訂正語の変化率 (語単位)
  3. 復元統計の変化: fully_restored (byte-identical) 率と語復元率の新旧比較
     (旧値はアーカイブの校正後テキストから同一ロジックで再計算)

使用例:
  uv run python scripts/exp7/compare_pyspell_regen.py \
    --source <LXT-4 perturbed_dataset.json> \
    --new data/exp7/corrected/gemma-3-4b-it_gsm8k_k4_spellfix/perturbed_dataset.json \
    --repro data/exp7/repro_check/gemma-3-4b-it_gsm8k_k4_spellfix/perturbed_dataset.json \
    --archive <archive rebuttal spellfix perturbed_dataset.json> \
    --output results/prod/exp7/rebuttal_regen_diff_gsm8k.json
"""

import argparse
import json
from pathlib import Path

from typo_cot.defense.restoration import (
    build_reference,
    classify_restoration,
    diff_word_positions,
)


def load_samples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def canonical_bytes(samples: list[dict]) -> bytes:
    return json.dumps(samples, ensure_ascii=False, sort_keys=True).encode("utf-8")


def restoration_summary(sources: dict, corrected: dict) -> dict:
    """校正後テキスト集合の復元統計 (analyze_spellfix.py と同一ロジック)."""
    n = 0
    word_total = 0
    word_restored = 0
    fully = 0
    collateral_samples = 0
    for sid, s in sources.items():
        if sid not in corrected:
            continue
        n += 1
        ref = build_reference(s["original_question"], s.get("choices"))
        r = classify_restoration(
            ref, s["perturbed_question"], corrected[sid]["perturbed_question"]
        )
        word_total += r.n_perturbed_words
        word_restored += r.n_restored
        fully += r.fully_restored
        collateral_samples += r.n_collateral > 0
    return {
        "n_samples": n,
        "word_total": word_total,
        "word_restored": word_restored,
        "word_restoration_rate": word_restored / word_total if word_total else 0.0,
        "fully_restored": fully,
        "full_restoration_rate": fully / n if n else 0.0,
        "samples_with_collateral": collateral_samples,
        "collateral_sample_rate": collateral_samples / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="rebuttal 再生成の検証レポート")
    parser.add_argument("--source", required=True, help="LXT-4 摂動データセット")
    parser.add_argument("--new", required=True, help="再生成 run1 の校正済みデータセット")
    parser.add_argument("--repro", required=True, help="再生成 run2 (再現性確認用)")
    parser.add_argument("--archive", required=True, help="旧 rebuttal spellfix データセット")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = {s["sample_id"]: s for s in load_samples(args.source)}
    new_samples = load_samples(args.new)
    new = {s["sample_id"]: s for s in new_samples}
    repro_samples = load_samples(args.repro)
    old = {s["sample_id"]: s for s in load_samples(args.archive)}

    # 1. 再現性 (byte 一致)
    reproducible = canonical_bytes(new_samples) == canonical_bytes(repro_samples)

    # 2. 旧 rebuttal との差分 (摂動語位置における訂正語の変化率)
    n_common = 0
    n_text_changed = 0
    word_pos_total = 0           # 整列可能な摂動語位置の総数
    word_correction_changed = 0  # 同位置で訂正語が旧≠新
    changed_words = []
    for sid, s in src.items():
        if sid not in new or sid not in old:
            continue
        n_common += 1
        ref = build_reference(s["original_question"], s.get("choices"))
        nt = new[sid]["perturbed_question"]
        ot = old[sid]["perturbed_question"]
        if nt != ot:
            n_text_changed += 1
        nw, ow_ = nt.split(), ot.split()
        for j, orig, _pert in diff_word_positions(ref, s["perturbed_question"]):
            if orig is None:
                continue
            word_pos_total += 1
            a = nw[j] if j < len(nw) else None
            b = ow_[j] if j < len(ow_) else None
            if a != b:
                word_correction_changed += 1
                if len(changed_words) < 200:
                    changed_words.append(
                        {"sample_id": sid, "original": orig, "new": a, "archive": b}
                    )

    # 3. 復元統計の新旧比較
    stats_new = restoration_summary(src, new)
    stats_old = restoration_summary(src, old)

    result = {
        "source": args.source,
        "new": args.new,
        "archive": args.archive,
        "reproducible_byte_identical": reproducible,
        "n_common": n_common,
        "n_corrected_text_changed": n_text_changed,
        "corrected_text_change_rate": n_text_changed / n_common if n_common else 0.0,
        "perturbed_word_positions": word_pos_total,
        "word_correction_changed": word_correction_changed,
        "word_correction_change_rate": (
            word_correction_changed / word_pos_total if word_pos_total else 0.0
        ),
        "restoration_new": stats_new,
        "restoration_archive": stats_old,
        "changed_word_examples": changed_words,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "changed_word_examples"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

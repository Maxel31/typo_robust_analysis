#!/usr/bin/env python3
"""Rebuttal 実験③: Spell-Correction Restoration 用データセット作成 (N5Yq Weakness 1).

LXT-4 摂動済みデータセット (datasets/perturbed/{model}_{bench}_k4_with_choices) の
perturbed_question に既製スペル訂正 (pyspellchecker) を単語単位で適用し、
「復元後」データセットを既存 Phase 3 (run_inference.py --perturbed_data) 互換形式で出力する。

同時に入力復元率を計測する:
- word_restored / word_total: 摂動された語のうち訂正で原文に戻った語の数
- fully_restored: 全文が原文と完全一致したサンプル (greedy 生成では flip が起こり得ない対照群)
- perturbed_words_all_restored: 摂動語は全て復元されたが、訂正器が他の語を壊した等で
  全文一致には至らないサンプル
- collateral_changes: 摂動されていない語を訂正器が変更してしまった数 (訂正器の副作用)

使用例:
  uv run --no-sync python scripts/rebuttal/make_spellfix_dataset.py \
    --input datasets/perturbed/gemma-3-4b-it_mmlu_k4_with_choices/perturbed_dataset.json \
    --output_dir datasets/rebuttal
"""

import argparse
import difflib
import json
import re
from datetime import datetime
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def apply_case(template: str, corrected: str) -> str:
    """訂正結果に元の語のケースパターンを適用する."""
    if template.isupper():
        return corrected.upper()
    if template[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def correct_text(text: str, spell) -> tuple[str, list[dict]]:
    """テキストを単語単位でスペル訂正する.

    アルファベット語のみ対象 (数値・記号・選択肢ラベルはそのまま)。
    辞書に存在する語は変更しない。訂正候補が無い語もそのまま。

    Returns:
        (訂正後テキスト, 変更ログ [{original, corrected, start}])
    """
    changes = []
    out = []
    last = 0
    for m in WORD_RE.finditer(text):
        word = m.group(0)
        out.append(text[last : m.start()])
        last = m.end()

        lower = word.lower()
        # 1文字語 (選択肢ラベル A-D, 冠詞 a 等) は訂正対象外
        if len(word) <= 1 or lower in spell:
            out.append(word)
            continue
        cand = spell.correction(lower)
        if cand is None or cand == lower:
            out.append(word)
            continue
        fixed = apply_case(word, cand)
        changes.append({"original": word, "corrected": fixed, "start": m.start()})
        out.append(fixed)
    out.append(text[last:])
    return "".join(out), changes


def diff_word_positions(original: str, perturbed: str) -> list[tuple[int, str, str]]:
    """原文と摂動文の空白区切り語列を difflib で対応付け、変化した語の位置を返す.

    Returns:
        [(perturbed 側の語インデックス, 原語, 摂動後語)]
    """
    orig_words = original.split()
    pert_words = perturbed.split()
    sm = difflib.SequenceMatcher(a=orig_words, b=pert_words, autojunk=False)
    changed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for off in range(i2 - i1):
                changed.append((j1 + off, orig_words[i1 + off], pert_words[j1 + off]))
        elif tag != "equal":
            # 語数が変わるケース (稀): 位置対応が取れないので None 印
            for j in range(j1, j2):
                changed.append((j, None, pert_words[j]))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Spell-Correction 復元データセット作成")
    parser.add_argument("--input", type=str, required=True,
                        help="摂動済み perturbed_dataset.json のパス")
    parser.add_argument("--output_dir", type=str, default="datasets/rebuttal")
    args = parser.parse_args()

    from spellchecker import SpellChecker

    spell = SpellChecker(language="en")

    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    samples = data["samples"]

    model_short = metadata.get("source_model", "unknown").split("/")[-1]
    benchmark = metadata.get("benchmark", "unknown")
    k = metadata.get("num_perturbations", "unknown")
    out_dir = Path(args.output_dir) / f"{model_short}_{benchmark}_k{k}_spellfix"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_samples = []
    per_sample_stats = []
    agg = {
        "n_samples": 0,
        "word_total": 0,          # difflib で同定できた摂動語数
        "word_restored": 0,       # 訂正後に原語へ戻った摂動語数
        "fully_restored": 0,      # 全文完全一致
        "perturbed_words_all_restored": 0,
        "collateral_changes": 0,  # 摂動されていない語への副作用的変更数
        "unalignable": 0,         # 語数変化などで対応付け不能だった摂動語数
    }

    for s in samples:
        # 参照テキスト: perturbed_question と同じ形式に合わせる。
        # MMLU 等では perturbed_question に選択肢が含まれる (dataset.py:593-597 の形式)
        # 一方 original_question は質問文のみのため、選択肢を同形式で付加する。
        original_q = s["original_question"]
        choices = s.get("choices")
        if choices:
            letters = "ABCDEFGHIJ"
            options_str = " ".join(
                f"({letters[i]}) {c}" for i, c in enumerate(choices)
            )
            original_q = f"{original_q}\n{options_str}"
        perturbed_q = s["perturbed_question"]

        corrected_q, changes = correct_text(perturbed_q, spell)

        # 摂動語の位置 (perturbed 基準) を difflib で同定
        pert_positions = diff_word_positions(original_q, perturbed_q)
        corr_words = corrected_q.split()
        orig_words = original_q.split()

        word_total = 0
        word_restored = 0
        unalignable = 0
        for j, ow, _pw in pert_positions:
            if ow is None:
                unalignable += 1
                continue
            word_total += 1
            if j < len(corr_words) and corr_words[j] == ow:
                word_restored += 1

        fully = " ".join(corrected_q.split()) == " ".join(orig_words)
        all_restored = word_total > 0 and word_restored == word_total

        # 副作用: 訂正器が変更した語のうち、摂動位置に対応しないもの
        pert_word_set = {pw for _, _, pw in pert_positions}
        collateral = sum(1 for c in changes if c["original"] not in pert_word_set)

        agg["n_samples"] += 1
        agg["word_total"] += word_total
        agg["word_restored"] += word_restored
        agg["unalignable"] += unalignable
        agg["collateral_changes"] += collateral
        if fully:
            agg["fully_restored"] += 1
        if all_restored:
            agg["perturbed_words_all_restored"] += 1

        per_sample_stats.append(
            {
                "sample_id": s["sample_id"],
                "n_perturbed_words": word_total,
                "n_restored": word_restored,
                "fully_restored": fully,
                "all_perturbed_restored": all_restored,
                "n_collateral_changes": collateral,
                "n_corrections": len(changes),
            }
        )

        new_s = dict(s)
        new_s["perturbed_question"] = corrected_q
        new_samples.append(new_s)

    new_metadata = dict(metadata)
    new_metadata["perturbation_mode"] = "spellfix"
    new_metadata["spellfix_source"] = str(input_path)
    new_metadata["spellfix_tool"] = "pyspellchecker"
    new_metadata["created_at"] = datetime.now().isoformat()
    new_metadata["total_samples"] = len(new_samples)

    with open(out_dir / "perturbed_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": new_metadata, "samples": new_samples}, f,
                  ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)

    rates = {
        "word_restoration_rate": agg["word_restored"] / agg["word_total"]
        if agg["word_total"] else 0.0,
        "full_restoration_rate": agg["fully_restored"] / agg["n_samples"]
        if agg["n_samples"] else 0.0,
        "all_perturbed_restored_rate": agg["perturbed_words_all_restored"] / agg["n_samples"]
        if agg["n_samples"] else 0.0,
    }
    with open(out_dir / "restoration_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": agg, "rates": rates, "per_sample": per_sample_stats},
            f, ensure_ascii=False, indent=2,
        )

    print(f"出力: {out_dir}")
    print(json.dumps({"aggregate": agg, "rates": rates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""実験7 スモーク①: pyspell 段の自実装が rebuttal 済みログと出力一致するか検証する.

アーカイブの LXT-4 摂動データセット (spellfix の元入力) に自実装の
PySpellCorrector を適用し、アーカイブの spellfix データセット
(rebuttal の make_spellfix_dataset.py の出力) と perturbed_question を
サンプル単位で byte 比較する。GPU 不要。アーカイブは読み取りのみ。

使用例:
  uv run python scripts/exp7/verify_pyspell_parity.py \
    --source /home/.../datasets/perturbed/gemma-3-4b-it_gsm8k_k4_with_choices/perturbed_dataset.json \
    --archive /home/.../datasets/rebuttal/gemma-3-4b-it_gsm8k_k4_spellfix/perturbed_dataset.json \
    --output results/smoke/pyspell_parity_gsm8k.json
"""

import argparse
import json
import re
from pathlib import Path

from typo_cot.defense.correctors import PySpellCorrector

_STRIP_RE = re.compile(r"^[^A-Za-z]+|[^A-Za-z']+$")


def _core(word: str) -> str:
    """語の前後の記号を除いた比較用コア (小文字)."""
    return _STRIP_RE.sub("", word).lower()


def classify_mismatch(
    corrector: PySpellCorrector, perturbed: str, ours: str, theirs: str
) -> tuple[bool, list[dict]]:
    """不一致サンプルが「頻度同点候補の選択差」だけで説明できるか判定する.

    pyspellchecker の correction() は同点候補を PYTHONHASHSEED 依存の
    set 反復順で選ぶため、rebuttal 出力の同点ケースは byte 再現不能。
    摂動語の候補集合 (現行辞書) に差分語の双方が同頻度で入っていれば
    tie-explained、archive 側の語が候補に無い場合は dictionary-drift
    (rebuttal 実行時の旧辞書由来) として記録する。
    """
    spell = corrector._spell
    details = []
    explained = True
    pw, ow, tw = perturbed.split(), ours.split(), theirs.split()
    if not (len(pw) == len(ow) == len(tw)):
        return False, [{"reason": "word_count_differs"}]
    for p, a, b in zip(pw, ow, tw):
        if a == b:
            continue
        cp, ca, cb = _core(p), _core(a), _core(b)
        cands = spell.candidates(cp) or set()
        freqs = {c: spell[c] for c in cands}
        tie = ca in cands and cb in cands and freqs.get(ca) == freqs.get(cb)
        # 辞書差: archive 側の語が現行辞書の候補に無い (語彙差)、または
        # 候補にはあるが頻度が最大でない (correction() は常に最大頻度を選ぶ
        # ため、rebuttal 実行時の頻度表が現行と異なっていたことを意味する)
        drift = cb not in cands or (not tie and freqs[cb] < freqs.get(ca, 0))
        details.append(
            {
                "perturbed": p,
                "ours": a,
                "archive": b,
                "tie_in_current_dict": tie,
                "dictionary_drift": drift,
            }
        )
        if not (tie or drift):
            explained = False
    return explained, details


def main() -> None:
    parser = argparse.ArgumentParser(description="pyspell 段のアーカイブ一致検証")
    parser.add_argument("--source", required=True,
                        help="アーカイブの LXT-4 摂動 perturbed_dataset.json")
    parser.add_argument("--archive", required=True,
                        help="アーカイブの spellfix perturbed_dataset.json (正解)")
    parser.add_argument("--output", required=True, help="検証結果 JSON の出力先")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as f:
        source = json.load(f)
    with open(args.archive, encoding="utf-8") as f:
        archive = {s["sample_id"]: s for s in json.load(f)["samples"]}

    corrector = PySpellCorrector()

    samples = source["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]

    n_total = 0
    n_match = 0
    n_explained = 0
    mismatches = []
    for s in samples:
        sid = s["sample_id"]
        if sid not in archive:
            continue
        n_total += 1
        perturbed = s["perturbed_question"]
        ours = corrector.correct(perturbed)
        theirs = archive[sid]["perturbed_question"]
        if ours == theirs:
            n_match += 1
            continue
        explained, details = classify_mismatch(corrector, perturbed, ours, theirs)
        if explained:
            n_explained += 1
        mismatches.append(
            {"sample_id": sid, "tie_or_drift_explained": explained, "words": details}
        )

    result = {
        "source": args.source,
        "archive": args.archive,
        "n_compared": n_total,
        "n_byte_identical": n_match,
        "parity_rate": n_match / n_total if n_total else None,
        "n_mismatch": n_total - n_match,
        "n_mismatch_tie_or_drift_explained": n_explained,
        "n_mismatch_unexplained": n_total - n_match - n_explained,
        "mismatch_details": mismatches,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "mismatch_details"},
                     ensure_ascii=False, indent=2))
    if result["n_mismatch_unexplained"] > 0:
        print(f"警告: 説明不能な不一致 {result['n_mismatch_unexplained']} 件 "
              f"(詳細: {out_path})")


if __name__ == "__main__":
    main()

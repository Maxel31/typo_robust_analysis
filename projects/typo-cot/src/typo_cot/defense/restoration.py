"""校正後テキストの語単位復元判定 (論文 Appendix B の文字スパン整列を流用).

rebuttal の make_spellfix_dataset.py / analyze_spellfix.py に散在していた
difflib ベースの語整列ロジックをライブラリ化したもの。動作は同一:
- 参照テキスト = original_question (+ 選択肢を "(A) x (B) y" 形式で付加)
- 摂動位置 = 参照 vs 摂動文の空白区切り語列の同数 replace 対応
- 復元判定 = 校正後の同位置の語が原語と一致するか
- 誤修正 (collateral) = 摂動位置以外での参照 vs 校正後の相違
- fully_restored = 空白正規化後の全文一致 (rebuttal の "byte-identical" 判定)
"""

import difflib
from dataclasses import dataclass, field

LETTERS = "ABCDEFGHIJ"


def build_reference(original_question: str, choices: list[str] | None) -> str:
    """perturbed_question と同形式の参照テキストを構築 (dataset.py:593-597 と同じ)."""
    if choices:
        options = " ".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
        return f"{original_question}\n{options}"
    return original_question


def aligned_word_changes(ref: str, hyp: str) -> list[tuple[int, str, str]]:
    """空白区切り語列の difflib 対応付け (同数 replace のみ位置対応).

    Returns:
        [(hyp 側の語インデックス, ref 語, hyp 語)]
    """
    rw, hw = ref.split(), hyp.split()
    out: list[tuple[int, str, str]] = []
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for off in range(i2 - i1):
                out.append((j1 + off, rw[i1 + off], hw[j1 + off]))
    return out


def diff_word_positions(ref: str, pert: str) -> list[tuple[int, str | None, str]]:
    """参照と摂動文の語列を対応付け、変化した語の位置を返す.

    同数 replace は位置対応付きで返し、語数が変わるケース (稀) は
    原語 None の unalignable として返す (make_spellfix_dataset.py と同一)。

    Returns:
        [(pert 側の語インデックス, 原語 | None, 摂動後語)]
    """
    ref_words = ref.split()
    pert_words = pert.split()
    sm = difflib.SequenceMatcher(a=ref_words, b=pert_words, autojunk=False)
    changed: list[tuple[int, str | None, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for off in range(i2 - i1):
                changed.append((j1 + off, ref_words[i1 + off], pert_words[j1 + off]))
        elif tag != "equal":
            for j in range(j1, j2):
                changed.append((j, None, pert_words[j]))
    return changed


@dataclass
class RestorationResult:
    """1サンプルの復元判定結果.

    Attributes:
        n_perturbed_words: difflib で位置対応できた摂動語数
        n_restored: 校正後に原語へ戻った摂動語数
        n_unalignable: 語数変化などで対応付け不能だった摂動語数
        restored_flags: [(原語, 摂動後語, 復元されたか)]
        fully_restored: 空白正規化後の全文一致 (byte-identical 判定)
        all_perturbed_restored: 摂動語は全復元 (誤修正が残る場合を含む)
        n_collateral: 摂動位置以外で校正器が壊した clean 語の数
        collateral: [(語インデックス, 原語, 校正後語)]
    """

    n_perturbed_words: int
    n_restored: int
    n_unalignable: int
    restored_flags: list[tuple[str, str, bool]]
    fully_restored: bool
    all_perturbed_restored: bool
    n_collateral: int
    collateral: list[tuple[int, str, str]] = field(default_factory=list)


def classify_restoration(
    reference: str, perturbed: str, corrected: str
) -> RestorationResult:
    """参照・摂動・校正後の3テキストから語単位の復元/非復元/誤修正を分類する.

    analyze_spellfix.py:86-106 のロジックと同一。

    Args:
        reference: 原文 (選択肢込みの参照形式。build_reference で構築)
        perturbed: 摂動後テキスト
        corrected: 校正後テキスト
    """
    changes = diff_word_positions(reference, perturbed)
    corr_words = corrected.split()
    ref_words = reference.split()

    n_pw = 0
    n_restored = 0
    n_unalignable = 0
    restored_flags: list[tuple[str, str, bool]] = []
    for j, ow, pw in changes:
        if ow is None:
            n_unalignable += 1
            continue
        n_pw += 1
        restored = j < len(corr_words) and corr_words[j] == ow
        n_restored += restored
        restored_flags.append((ow, pw, restored))

    fully = " ".join(corr_words) == " ".join(ref_words)
    all_restored = n_pw > 0 and n_restored == n_pw

    # 誤修正: 校正後テキストの参照との相違のうち、摂動位置に対応しないもの
    fix_changes = aligned_word_changes(reference, corrected)
    pert_positions = {j for j, _, _ in changes}
    collateral = [c for c in fix_changes if c[0] not in pert_positions]

    return RestorationResult(
        n_perturbed_words=n_pw,
        n_restored=n_restored,
        n_unalignable=n_unalignable,
        restored_flags=restored_flags,
        fully_restored=fully,
        all_perturbed_restored=all_restored,
        n_collateral=len(collateral),
        collateral=collateral,
    )

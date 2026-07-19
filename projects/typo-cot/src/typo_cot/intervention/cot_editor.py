"""実験2: CoT 編集オペレータ (削除 / 「…」マスク / 同品詞・同頻度帯置換).

計画 §4 実験2-2 の操作3種を、答え句前 prefix (loo_scorer.split_generated_text
の cot_text) 上の**語タイプ全出現**に適用する (§4 実験2-1「全出現を操作」)。

- delete: 出現スパン (端句読点を除くコア) を削除。LOO の削除規約と同一
  (loo_scorer.delete_spans と同じスパン単位・多重スペース詰め)。
- mask: 出現スパンを「…」に置換 — 文法破壊の交絡を統制する対照操作。
- replace: 出現スパンを指定語 (同品詞・同頻度帯の別語) に置換。

設計メモ: docs/dev_notes_02_target_deletion.md
"""

import re
from dataclasses import dataclass, field

from typo_cot.intervention.loo_scorer import extract_word_types

MASK_TOKEN = "…"
OPERATIONS = ("delete", "mask", "replace")


@dataclass
class EditResult:
    """CoT 編集の結果.

    Attributes:
        edited_text: 編集後 prefix
        n_spans_edited: 編集された出現スパン数の合計
        edited_words: 実際に編集された標的語タイプ (prefix 中に存在したもの)
        missing_words: prefix 中に見つからなかった標的語タイプ
        replacements: replace 操作で使われた語 → 置換語の対応
        changed: 編集でテキストが変化したか
    """

    edited_text: str
    n_spans_edited: int
    edited_words: list[str] = field(default_factory=list)
    missing_words: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    changed: bool = False


def apply_edit(
    cot_text: str,
    target_words: list[str],
    op: str,
    replacement_map: dict[str, str] | None = None,
) -> EditResult:
    """標的語タイプの全出現に操作を適用した編集後 prefix を作る.

    Args:
        cot_text: 答え句前 prefix
        target_words: 標的語タイプ (extract_word_types のキーと同一表記)
        op: "delete" | "mask" | "replace"
        replacement_map: replace 時の 語 → 置換語 (全標的分が必須)

    Raises:
        ValueError: 未知の操作、または replace で置換語が不足している場合
    """
    if op not in OPERATIONS:
        raise ValueError(f"unknown op: {op!r} (expected one of {OPERATIONS})")
    if op == "replace":
        if replacement_map is None:
            raise ValueError("replace op requires replacement_map")
        lacking = [w for w in target_words if w not in replacement_map]
        if lacking:
            raise ValueError(f"replacement_map missing words: {lacking}")

    types = {wt.word: wt for wt in extract_word_types(cot_text)}
    edited_words: list[str] = []
    missing_words: list[str] = []
    span_edits: list[tuple[int, int, str]] = []  # (start, end, replacement)

    for word in target_words:
        wt = types.get(word)
        if wt is None:
            missing_words.append(word)
            continue
        edited_words.append(word)
        if op == "delete":
            rep = ""
        elif op == "mask":
            rep = MASK_TOKEN
        else:
            rep = replacement_map[word]  # type: ignore[index]
        for start, end in wt.spans:
            span_edits.append((start, end, rep))

    text = cot_text
    for start, end, rep in sorted(span_edits, reverse=True):
        text = text[:start] + rep + text[end:]
    if op == "delete":
        # loo_scorer.delete_spans と同じ多重スペース詰め
        text = re.sub(r"[ \t]{2,}", " ", text)

    replacements = (
        {w: replacement_map[w] for w in edited_words}  # type: ignore[index]
        if op == "replace"
        else {}
    )
    return EditResult(
        edited_text=text,
        n_spans_edited=len(span_edits),
        edited_words=edited_words,
        missing_words=missing_words,
        replacements=replacements,
        changed=text != cot_text,
    )

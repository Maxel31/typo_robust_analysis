"""clean/typo テキストの文字レベル整列による摂動語スパン特定.

実験9 (inner lexicon 修復スコア) では、typo 語スパン末尾トークンの隠れ状態と
対応する clean 語のそれとを比較する。そのために、
clean テキストと typo テキスト (質問のみでも few-shot 込みプロンプト全体でもよい)
の差分から「どの語がどの語に摂動されたか」の文字スパン対を復元する。

アーカイブの perturbed_tokens はトークン文字列 (例: " Janet" -> "Janeet") しか
持たず文字位置を持たないため、difflib による整列で位置を確定し、
perturbed_tokens のメタデータ (importance_score 等) を突合して引き継ぐ。
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

# 語境界の判定: 英数字のみを語構成文字とみなす。
# アポストロフィは含めない (アーカイブの摂動対象トークンは "Janet's" の
# "Janet" のようにアポストロフィ前のサブトークン単位のため)。


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


@dataclass
class AlignedSpan:
    """整列済み摂動語スパン.

    Attributes:
        clean_word: clean テキスト側の語
        typo_word: typo テキスト側の語
        clean_start: clean テキスト内の開始文字位置
        clean_end: clean テキスト内の終了文字位置 (排他的)
        typo_start: typo テキスト内の開始文字位置
        typo_end: typo テキスト内の終了文字位置 (排他的)
        importance_score: perturbed_tokens 由来の R_Q 重要度 (突合できた場合)
        perturbation_type: 摂動タイプ (proximity/double_typing/omission)
        token_index: アーカイブのトークンインデックス
    """

    clean_word: str
    typo_word: str
    clean_start: int
    clean_end: int
    typo_start: int
    typo_end: int
    importance_score: float | None = None
    perturbation_type: str | None = None
    token_index: int | None = None


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """[start, end) を語境界まで拡張する."""
    # 空領域 (挿入・削除点) はそのまま左右へ拡張
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    while end < len(text) and _is_word_char(text[end]):
        end += 1
    return start, end


def _strip_nonword_edges(text: str, start: int, end: int) -> tuple[int, int]:
    """区間端の非語文字 (空白・句読点) を取り除く."""
    while start < end and not _is_word_char(text[start]):
        start += 1
    while end > start and not _is_word_char(text[end - 1]):
        end -= 1
    return start, end


def align_typo_spans(
    clean_text: str,
    typo_text: str,
    perturbed_tokens: list[dict],
) -> list[AlignedSpan]:
    """clean/typo テキストを整列し、摂動語スパンの対を返す.

    Args:
        clean_text: 摂動前テキスト (質問のみ or プロンプト全体)
        typo_text: 摂動後テキスト (clean_text と共通の前後文脈を持つこと)
        perturbed_tokens: アーカイブ形式の摂動トークンリスト
            (original_token / perturbed_token / importance_score /
             perturbation_type / token_index)

    Returns:
        テキスト中の出現順に並んだ AlignedSpan のリスト。
        perturbed_tokens と突合できなかった差分領域・
        テキスト差分に現れなかった perturbed_tokens は落とす。
    """
    sm = SequenceMatcher(None, clean_text, typo_text, autojunk=False)

    # 1. 非一致領域を収集し、語境界まで拡張
    regions: list[tuple[int, int, int, int]] = []  # (c1, c2, t1, t2)
    for tag, c1, c2, t1, t2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ec1, ec2 = _expand_to_word(clean_text, c1, c2)
        et1, et2 = _expand_to_word(typo_text, t1, t2)
        regions.append((ec1, ec2, et1, et2))

    # 2. 同一語内の複数編集 (領域の重複) をマージ
    merged: list[list[int]] = []
    for r in sorted(regions):
        if merged and r[0] < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged[-1][2] = min(merged[-1][2], r[2])
            merged[-1][3] = max(merged[-1][3], r[3])
        else:
            merged.append(list(r))

    # 3. 領域端の非語文字を除去して語スパンへ
    word_pairs: list[tuple[int, int, int, int]] = []
    for c1, c2, t1, t2 in merged:
        c1, c2 = _strip_nonword_edges(clean_text, c1, c2)
        t1, t2 = _strip_nonword_edges(typo_text, t1, t2)
        if c1 >= c2 and t1 >= t2:
            continue
        word_pairs.append((c1, c2, t1, t2))

    # 4. perturbed_tokens と出現順で突合 (語文字列の包含で検証)
    spans: list[AlignedSpan] = []
    unused = list(perturbed_tokens)
    for c1, c2, t1, t2 in word_pairs:
        clean_word = clean_text[c1:c2]
        typo_word = typo_text[t1:t2]
        meta: dict | None = None
        for tok in unused:
            orig = str(tok.get("original_token", "")).strip()
            pert = str(tok.get("perturbed_token", "")).strip()
            # トークンがサブワードの場合もあるため包含で照合する
            orig_ok = not orig or orig in clean_word or clean_word in orig
            pert_ok = not pert or pert in typo_word or typo_word in pert
            if orig_ok and pert_ok:
                meta = tok
                break
        if meta is None:
            # テキスト差分はあるが perturbed_tokens に対応が無い → 落とす
            continue
        unused.remove(meta)
        spans.append(
            AlignedSpan(
                clean_word=clean_word,
                typo_word=typo_word,
                clean_start=c1,
                clean_end=c2,
                typo_start=t1,
                typo_end=t2,
                importance_score=meta.get("importance_score"),
                perturbation_type=meta.get("perturbation_type"),
                token_index=meta.get("token_index"),
            )
        )
    return spans


def char_span_to_last_token(
    offset_mapping: list[tuple[int, int]],
    span: tuple[int, int],
) -> int | None:
    """文字スパンに重なる最後のトークンのインデックスを返す.

    Args:
        offset_mapping: tokenizer(..., return_offsets_mapping=True) の
            (start, end) リスト。special token の (0, 0) は無視する。
        span: (start, end) 文字スパン (end は排他的)

    Returns:
        スパン末尾トークンのインデックス。重なりが無ければ None。
    """
    s, e = span
    last: int | None = None
    for i, (ts, te) in enumerate(offset_mapping):
        if ts == te:  # special token
            continue
        if ts < e and te > s:
            last = i
    return last

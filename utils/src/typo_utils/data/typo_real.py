"""実typo生成器 — 文字レベルの本物のtypo注入。

対応typo種別 (TYPO_TYPES):
  - ``sub_keyboard``: QWERTYキー隣接文字への置換。
  - ``insert``:      ランダムな小文字1文字の挿入。
  - ``delete``:      1文字の削除（長さ>1の位置のみ）。
  - ``transpose``:   隣接2文字の入れ替え。

``apply_typo`` は ``random.Random(seed)`` で決定論的に動作する。
"""

from __future__ import annotations

import math
import random
import string

TYPO_TYPES: tuple[str, ...] = ("sub_keyboard", "insert", "delete", "transpose")

# QWERTY 隣接キーマップ（アルファベット小文字のみ）
_QWERTY_NEIGHBORS: dict[str, str] = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


def _eligible_sub_keyboard(chars: list[str]) -> list[int]:
    """sub_keyboard の eligible positions: neighborを持つアルファ文字のインデックス。"""
    return [
        i
        for i, ch in enumerate(chars)
        if ch.isalpha() and _QWERTY_NEIGHBORS.get(ch.lower())
    ]


def _eligible_insert(chars: list[str]) -> list[int]:
    """insert の eligible positions: 全インデックス (0..len)。"""
    return list(range(len(chars) + 1))


def _eligible_delete(chars: list[str]) -> list[int]:
    """delete の eligible positions: 全インデックス（ただし文字列長>1 のときのみ有効）。"""
    if len(chars) <= 1:
        return []
    return list(range(len(chars)))


def _eligible_transpose(chars: list[str]) -> list[int]:
    """transpose の eligible positions: 隣接ペアの左インデックス (0..len-2)。"""
    return list(range(len(chars) - 1))


def _apply_one_sub_keyboard(chars: list[str], pos: int, rng: random.Random) -> None:
    """sub_keyboard を1箇所適用（in-place）。neighborがなければスキップ。"""
    ch = chars[pos]
    neighbors = _QWERTY_NEIGHBORS.get(ch.lower(), "")
    if not neighbors:
        return
    replacement = rng.choice(neighbors)
    # 元の大文字小文字を保持
    if ch.isupper():
        replacement = replacement.upper()
    chars[pos] = replacement


def _apply_one_insert(chars: list[str], pos: int, rng: random.Random) -> None:
    """insert を1箇所適用（in-place）。"""
    letter = rng.choice(string.ascii_lowercase)
    chars.insert(pos, letter)


def _apply_one_delete(chars: list[str], pos: int, rng: random.Random) -> None:
    """delete を1箇所適用（in-place）。"""
    del chars[pos]


def _apply_one_transpose(chars: list[str], pos: int, rng: random.Random) -> None:
    """transpose を1箇所適用（in-place）。pos と pos+1 を入れ替える。"""
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]


# typo_type ごとの (eligible関数, apply関数) テーブル
_HANDLERS: dict[
    str,
    tuple[
        object,
        object,
    ],
] = {
    "sub_keyboard": (_eligible_sub_keyboard, _apply_one_sub_keyboard),
    "insert": (_eligible_insert, _apply_one_insert),
    "delete": (_eligible_delete, _apply_one_delete),
    "transpose": (_eligible_transpose, _apply_one_transpose),
}


def apply_typo(
    text: str,
    typo_type: str,
    eps: int | float,
    *,
    seed: int,
) -> str:
    """テキストに実typoを注入して返す（決定論的）。

    Parameters
    ----------
    text:
        注入対象のテキスト。
    typo_type:
        ``TYPO_TYPES`` のいずれか。不明な値は ``ValueError`` を送出。
    eps:
        * ``eps == 1`` (int): eligible positions からランダムに **1箇所** を選んで適用。
        * ``0 < eps < 1`` (float): ``max(1, floor(eps * n_eligible))`` 箇所に適用。
    seed:
        ``random.Random`` のシード（決定論性保証）。

    Returns
    -------
    str
        typoを注入したテキスト。eligible positionがない場合は元のテキストをそのまま返す。

    Raises
    ------
    ValueError
        ``typo_type`` が ``TYPO_TYPES`` に含まれない場合。
    """
    if typo_type not in _HANDLERS:
        raise ValueError(
            f"Unknown typo_type {typo_type!r}. "
            f"Expected one of {TYPO_TYPES}."
        )

    get_eligible, apply_one = _HANDLERS[typo_type]  # type: ignore[misc]
    rng = random.Random(seed)
    chars = list(text)

    if eps == 1:
        eligible = get_eligible(chars)
        if not eligible:
            return text
        pos = rng.choice(eligible)
        apply_one(chars, pos, rng)
    else:
        # 0 < eps < 1: max(1, floor(eps * n_eligible)) 箇所に適用
        eligible = get_eligible(chars)
        if not eligible:
            return text
        n_apply = max(1, math.floor(eps * len(eligible)))
        # distinct positions をランダムにサンプリング
        chosen = rng.sample(eligible, min(n_apply, len(eligible)))
        # insert/delete はインデックスがずれないよう後ろから処理する
        for pos in sorted(chosen, reverse=True):
            apply_one(chars, pos, rng)

    return "".join(chars)


def available_types() -> tuple[str, ...]:
    """利用可能な typo 種別を返す。

    Returns
    -------
    tuple[str, ...]
        ``TYPO_TYPES`` と同じタプル。
    """
    return TYPO_TYPES

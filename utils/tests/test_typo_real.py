"""実 typo 生成器 (typo_real) の単体テスト。

TYPO_TYPES = ('sub_keyboard', 'insert', 'delete', 'transpose')
apply_typo(text, typo_type, eps, seed) -> str

* eps == 1 (int): 対象位置 1 箇所に exactly one typo を適用。
* 0 < eps < 1 (float): max(1, floor(eps * n_eligible)) 箇所に適用。
"""

from __future__ import annotations

import pytest

from typo_utils.data import TYPO_TYPES, apply_typo


# ---------------------------------------------------------------------------
# 基本: TYPO_TYPES の中身
# ---------------------------------------------------------------------------


def test_typo_types_contents() -> None:
    assert set(TYPO_TYPES) == {"sub_keyboard", "insert", "delete", "transpose"}


# ---------------------------------------------------------------------------
# sub_keyboard — 決定論性・非変化・長さ不変
# ---------------------------------------------------------------------------

_LONG_ALPHA = "abcdefghijklmnopqrstuvwxyz"  # 26文字、全てアルファベット


def test_sub_keyboard_deterministic() -> None:
    out1 = apply_typo(_LONG_ALPHA, "sub_keyboard", 1, seed=0)
    out2 = apply_typo(_LONG_ALPHA, "sub_keyboard", 1, seed=0)
    assert out1 == out2


def test_sub_keyboard_changes_input() -> None:
    # 26文字全てにneighborがあるはずなので必ず変化する
    out = apply_typo(_LONG_ALPHA, "sub_keyboard", 1, seed=0)
    assert out != _LONG_ALPHA


def test_sub_keyboard_length_invariant() -> None:
    out = apply_typo(_LONG_ALPHA, "sub_keyboard", 1, seed=0)
    assert len(out) == len(_LONG_ALPHA)


# ---------------------------------------------------------------------------
# insert — 決定論性・非変化・長さ +1
# ---------------------------------------------------------------------------


def test_insert_deterministic() -> None:
    out1 = apply_typo(_LONG_ALPHA, "insert", 1, seed=42)
    out2 = apply_typo(_LONG_ALPHA, "insert", 1, seed=42)
    assert out1 == out2


def test_insert_changes_input() -> None:
    out = apply_typo(_LONG_ALPHA, "insert", 1, seed=42)
    assert out != _LONG_ALPHA


def test_insert_length_plus_one() -> None:
    out = apply_typo(_LONG_ALPHA, "insert", 1, seed=42)
    assert len(out) == len(_LONG_ALPHA) + 1


# ---------------------------------------------------------------------------
# delete — 決定論性・非変化・長さ -1
# ---------------------------------------------------------------------------


def test_delete_deterministic() -> None:
    out1 = apply_typo(_LONG_ALPHA, "delete", 1, seed=7)
    out2 = apply_typo(_LONG_ALPHA, "delete", 1, seed=7)
    assert out1 == out2


def test_delete_changes_input() -> None:
    out = apply_typo(_LONG_ALPHA, "delete", 1, seed=7)
    assert out != _LONG_ALPHA


def test_delete_length_minus_one() -> None:
    out = apply_typo(_LONG_ALPHA, "delete", 1, seed=7)
    assert len(out) == len(_LONG_ALPHA) - 1


def test_delete_single_char_unchanged() -> None:
    # 長さ1の文字列はelibleポジションがないので変化しない
    single = "a"
    out = apply_typo(single, "delete", 1, seed=0)
    assert out == single


# ---------------------------------------------------------------------------
# transpose — 決定論性・非変化・長さ同一
# ---------------------------------------------------------------------------


def test_transpose_deterministic() -> None:
    out1 = apply_typo(_LONG_ALPHA, "transpose", 1, seed=3)
    out2 = apply_typo(_LONG_ALPHA, "transpose", 1, seed=3)
    assert out1 == out2


def test_transpose_changes_input() -> None:
    out = apply_typo(_LONG_ALPHA, "transpose", 1, seed=3)
    assert out != _LONG_ALPHA


def test_transpose_length_invariant() -> None:
    out = apply_typo(_LONG_ALPHA, "transpose", 1, seed=3)
    assert len(out) == len(_LONG_ALPHA)


# ---------------------------------------------------------------------------
# eps=0.5 — 複数文字に適用・決定論的
# ---------------------------------------------------------------------------

_LONG_TEXT = "the quick brown fox jumps over the lazy dog and runs away"


def test_eps_float_insert_deterministic() -> None:
    out1 = apply_typo(_LONG_TEXT, "insert", 0.5, seed=99)
    out2 = apply_typo(_LONG_TEXT, "insert", 0.5, seed=99)
    assert out1 == out2


def test_eps_float_insert_changes_multiple_chars() -> None:
    # eps=0.5, insert: floor(0.5 * n_eligible) >= 1 箇所以上変化し、
    # eps=1 (1箇所のみ) と結果が異なる（= 複数箇所に適用されている）
    out_half = apply_typo(_LONG_TEXT, "insert", 0.5, seed=99)
    out_one = apply_typo(_LONG_TEXT, "insert", 1, seed=99)
    # 複数挿入なので長さの差が eps=1 より大きいはず
    assert len(out_half) > len(out_one)


def test_eps_float_delete_multiple() -> None:
    out_half = apply_typo(_LONG_TEXT, "delete", 0.5, seed=5)
    out_one = apply_typo(_LONG_TEXT, "delete", 1, seed=5)
    assert len(out_half) < len(out_one)


def test_eps_float_transpose_deterministic() -> None:
    out1 = apply_typo(_LONG_TEXT, "transpose", 0.5, seed=11)
    out2 = apply_typo(_LONG_TEXT, "transpose", 0.5, seed=11)
    assert out1 == out2


def test_eps_float_sub_keyboard_deterministic() -> None:
    out1 = apply_typo(_LONG_TEXT, "sub_keyboard", 0.5, seed=77)
    out2 = apply_typo(_LONG_TEXT, "sub_keyboard", 0.5, seed=77)
    assert out1 == out2


# ---------------------------------------------------------------------------
# 不明な typo_type → ValueError
# ---------------------------------------------------------------------------


def test_unknown_typo_type_raises() -> None:
    with pytest.raises(ValueError):
        apply_typo("hello world", "unknown_type", 1, seed=0)


def test_unknown_typo_type_raises_leetspeak() -> None:
    with pytest.raises(ValueError):
        apply_typo("hello world", "leetspeak", 1, seed=0)


# ---------------------------------------------------------------------------
# スペース・句読点の保存 (sub_keyboard / delete など)
# ---------------------------------------------------------------------------

_SENTENCE = "hello, world! foo bar"


def test_sub_keyboard_preserves_punctuation_positions() -> None:
    # 句読点の位置が変化しないことを確認（アルファ文字のみ操作）
    out = apply_typo(_SENTENCE, "sub_keyboard", 1, seed=0)
    # 長さ不変
    assert len(out) == len(_SENTENCE)
    # カンマ・感嘆符・スペースは保持
    for i, ch in enumerate(_SENTENCE):
        if not ch.isalpha():
            assert out[i] == ch, f"Position {i}: expected '{ch}', got '{out[i]}'"


def test_transpose_preserves_length() -> None:
    out = apply_typo(_SENTENCE, "transpose", 1, seed=0)
    assert len(out) == len(_SENTENCE)

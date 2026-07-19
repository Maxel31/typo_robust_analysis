"""実験2: cot_editor (削除 / 「…」マスク / 置換オペレータ) のテスト.

計画 §4 実験2-2: 操作3種 {(a)削除 / (b)「…」置換 / (c)同品詞・同頻度帯の別語への置換}。
標的語タイプの**全出現**を操作する (§4 実験2-1)。
"""

import pytest

from typo_cot.intervention.cot_editor import (
    MASK_TOKEN,
    EditResult,
    apply_edit,
)

COT = (
    "\nJanet's ducks lay 16 eggs per day.\n"
    "She eats 3 eggs for breakfast.\n"
    "So she has 16 - 3 = 13 eggs left.\n"
)


class TestDelete:
    def test_deletes_all_occurrences(self):
        res = apply_edit(COT, ["eggs"], op="delete")
        assert isinstance(res, EditResult)
        assert "eggs" not in res.edited_text
        assert res.n_spans_edited == 3
        assert res.changed is True
        assert res.edited_words == ["eggs"]

    def test_delete_preserves_other_words(self):
        res = apply_edit(COT, ["eggs"], op="delete")
        assert "breakfast" in res.edited_text
        assert "Janet" in res.edited_text
        assert "16" in res.edited_text

    def test_delete_multiple_targets(self):
        res = apply_edit(COT, ["eggs", "breakfast"], op="delete")
        assert "eggs" not in res.edited_text
        assert "breakfast" not in res.edited_text
        assert res.n_spans_edited == 4

    def test_delete_does_not_break_substrings(self):
        # "day" は "day." のコアであり、他語の部分文字列を壊さないこと
        res = apply_edit(COT, ["day"], op="delete")
        assert "breakfast" in res.edited_text
        assert res.n_spans_edited == 1

    def test_missing_target_recorded(self):
        res = apply_edit(COT, ["zebra"], op="delete")
        assert res.changed is False
        assert res.n_spans_edited == 0
        assert res.missing_words == ["zebra"]


class TestMask:
    def test_mask_replaces_with_ellipsis(self):
        res = apply_edit(COT, ["eggs"], op="mask")
        assert "eggs" not in res.edited_text
        assert res.edited_text.count(MASK_TOKEN) == 3
        assert res.n_spans_edited == 3

    def test_mask_preserves_length_of_text_structure(self):
        # マスクは削除と違い語数 (空白区切りチャンク数) を保つ
        res = apply_edit(COT, ["eggs"], op="mask")
        assert len(res.edited_text.split()) == len(COT.split())


class TestReplace:
    def test_replace_uses_replacement_map(self):
        res = apply_edit(COT, ["eggs"], op="replace", replacement_map={"eggs": "cards"})
        assert "eggs" not in res.edited_text
        assert res.edited_text.count("cards") == 3
        assert res.replacements == {"eggs": "cards"}

    def test_replace_without_map_raises(self):
        with pytest.raises(ValueError):
            apply_edit(COT, ["eggs"], op="replace")

    def test_replace_missing_word_in_map_raises(self):
        with pytest.raises(ValueError):
            apply_edit(COT, ["eggs", "breakfast"], op="replace", replacement_map={"eggs": "cards"})


class TestValidation:
    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            apply_edit(COT, ["eggs"], op="shuffle")

    def test_empty_targets_no_change(self):
        res = apply_edit(COT, [], op="delete")
        assert res.changed is False
        assert res.edited_text == COT

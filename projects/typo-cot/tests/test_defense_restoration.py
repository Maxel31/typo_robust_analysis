"""実験7: 復元判定 (defense/restoration.py) のユニットテスト.

rebuttal スクリプト (make_spellfix_dataset.py / analyze_spellfix.py) の
文字スパン整列ロジックをライブラリ化したものの検証。GPU 不要・合成データのみ。
"""

from typo_cot.defense.restoration import (
    RestorationResult,
    aligned_word_changes,
    build_reference,
    classify_restoration,
    diff_word_positions,
)


class TestBuildReference:
    def test_without_choices(self):
        assert build_reference("What is 2+2?", None) == "What is 2+2?"

    def test_with_choices(self):
        ref = build_reference("Pick one.", ["cat", "dog"])
        assert ref == "Pick one.\n(A) cat (B) dog"

    def test_with_many_choices_uses_letter_sequence(self):
        ref = build_reference("Q", ["a", "b", "c", "d", "e"])
        assert "(E) e" in ref


class TestDiffWordPositions:
    def test_same_length_replace_is_aligned(self):
        changes = diff_word_positions(
            "The quick brown fox", "The qick brwn fox"
        )
        assert changes == [(1, "quick", "qick"), (2, "brown", "brwn")]

    def test_no_change_returns_empty(self):
        assert diff_word_positions("a b c", "a b c") == []

    def test_word_count_change_marks_unalignable(self):
        # 語の分裂 (1語→2語) は位置対応不能 → 原語 None
        changes = diff_word_positions("The quick fox", "The qu ick fox")
        assert all(ow is None for _, ow, _ in changes)


class TestAlignedWordChanges:
    def test_only_same_length_replacements(self):
        out = aligned_word_changes("a b c d", "a x c y")
        assert out == [(1, "b", "x"), (3, "d", "y")]


class TestClassifyRestoration:
    REF = "The quick brown fox jumps"
    PERT = "The qick brwn fox jumps"

    def test_full_restoration(self):
        r = classify_restoration(self.REF, self.PERT, self.REF)
        assert isinstance(r, RestorationResult)
        assert r.n_perturbed_words == 2
        assert r.n_restored == 2
        assert r.fully_restored is True
        assert r.all_perturbed_restored is True
        assert r.n_collateral == 0

    def test_partial_restoration(self):
        corrected = "The quick brwn fox jumps"
        r = classify_restoration(self.REF, self.PERT, corrected)
        assert r.n_perturbed_words == 2
        assert r.n_restored == 1
        assert r.fully_restored is False
        assert r.all_perturbed_restored is False

    def test_collateral_damage_detected(self):
        # 摂動語は全復元だが、無関係の clean 語 (fox→fax) を壊した
        corrected = "The quick brown fax jumps"
        r = classify_restoration(self.REF, self.PERT, corrected)
        assert r.n_restored == 2
        assert r.all_perturbed_restored is True
        assert r.fully_restored is False
        assert r.n_collateral == 1
        assert r.collateral == [(3, "fox", "fax")]

    def test_whitespace_normalized_equality(self):
        # rebuttal 実装と同じ: 空白正規化後の全文一致で fully_restored 判定
        corrected = "The quick  brown fox jumps"
        r = classify_restoration(self.REF, self.PERT, corrected)
        assert r.fully_restored is True

    def test_no_restoration(self):
        r = classify_restoration(self.REF, self.PERT, self.PERT)
        assert r.n_restored == 0
        assert r.all_perturbed_restored is False

    def test_restored_flags_alignment(self):
        corrected = "The quick brwn fox jumps"
        r = classify_restoration(self.REF, self.PERT, corrected)
        assert r.restored_flags == [
            ("quick", "qick", True),
            ("brown", "brwn", False),
        ]

    def test_unalignable_counted(self):
        # 摂動側で語数が変わったケース
        pert = "The qu ick brown fox jumps"
        r = classify_restoration(self.REF, pert, self.REF)
        assert r.n_unalignable > 0

"""repair.span_align のテスト (実験9: inner lexicon 修復スコア).

clean テキストと typo テキストの文字レベル整列により、
摂動語スパン (clean 側 / typo 側) を特定するユーティリティのテスト。
GPU 不要・合成テキストのみ。
"""

from typo_cot.repair.span_align import (
    AlignedSpan,
    align_typo_spans,
    char_span_to_last_token,
)


def _tok(original: str, perturbed: str, score: float = 1.0) -> dict:
    """アーカイブの perturbed_tokens 形式のエントリを合成する."""
    return {
        "token_index": 0,
        "original_token": original,
        "perturbed_token": perturbed,
        "importance_score": score,
        "perturbation_type": "proximity",
        "char_position": 0,
    }


class TestAlignTypoSpans:
    def test_single_proximity_typo(self) -> None:
        clean = "Janet has five ducks in the yard."
        typo = "Janet has five dicks in the yard."
        spans = align_typo_spans(clean, typo, [_tok(" ducks", "dicks")])
        assert len(spans) == 1
        s = spans[0]
        assert isinstance(s, AlignedSpan)
        assert s.clean_word == "ducks"
        assert s.typo_word == "dicks"
        assert clean[s.clean_start : s.clean_end] == "ducks"
        assert typo[s.typo_start : s.typo_end] == "dicks"

    def test_double_typing_typo(self) -> None:
        clean = "Janet sells eggs daily."
        typo = "Janeet sells eggs daily."
        spans = align_typo_spans(clean, typo, [_tok(" Janet", "Janeet")])
        assert len(spans) == 1
        assert spans[0].clean_word == "Janet"
        assert spans[0].typo_word == "Janeet"

    def test_omission_typo(self) -> None:
        clean = "The ducks lay 16 eggs."
        typo = "The ducks ly 16 eggs."
        spans = align_typo_spans(clean, typo, [_tok(" lay", "ly")])
        assert len(spans) == 1
        assert spans[0].clean_word == "lay"
        assert spans[0].typo_word == "ly"

    def test_multiple_typos_in_order(self) -> None:
        clean = "Janet's ducks lay 16 eggs per day. How many eggs?"
        typo = "Janeet's dicks ly 16 eggs per day. Yow many eggs?"
        toks = [
            _tok(" Janet", "Janeet", 7.3),
            _tok(" ducks", "dicks", 0.64),
            _tok(" lay", "ly", 0.11),
            _tok(" How", "Yow", 0.09),
        ]
        spans = align_typo_spans(clean, typo, toks)
        assert [s.clean_word for s in spans] == ["Janet", "ducks", "lay", "How"]
        assert [s.typo_word for s in spans] == ["Janeet", "dicks", "ly", "Yow"]
        # importance_score がエントリから引き継がれる
        assert spans[0].importance_score == 7.3
        # スパンが元テキストと一致
        for s in spans:
            assert clean[s.clean_start : s.clean_end] == s.clean_word
            assert typo[s.typo_start : s.typo_end] == s.typo_word

    def test_repeated_word_only_one_perturbed(self) -> None:
        clean = "the cat saw the dog and the bird"
        typo = "the cat saw teh dog and the bird"
        spans = align_typo_spans(clean, typo, [_tok(" the", "teh")])
        assert len(spans) == 1
        # 2番目の "the" (index 12) が摂動された
        assert spans[0].clean_start == 12
        assert spans[0].typo_word == "teh"

    def test_alignment_within_long_shared_prefix(self) -> None:
        # 実利用ではプロンプト全体 (few-shot 込み) を渡すため、長い共通接頭辞を模す
        prefix = "Example: Roger has 5 tennis balls. " * 30
        clean = prefix + "Problem: Janet sells fresh eggs.\n\nSolution:"
        typo = prefix + "Problem: Janet sells frsh eggs.\n\nSolution:"
        spans = align_typo_spans(clean, typo, [_tok(" fresh", "frsh")])
        assert len(spans) == 1
        assert clean[spans[0].clean_start : spans[0].clean_end] == "fresh"
        assert typo[spans[0].typo_start : spans[0].typo_end] == "frsh"

    def test_unmatched_token_is_dropped(self) -> None:
        # perturbed_tokens に対応する差分がテキスト中に無い場合は落とす
        clean = "The cat sat."
        typo = "The cat sat."  # 差分なし
        spans = align_typo_spans(clean, typo, [_tok(" cat", "cta")])
        assert spans == []

    def test_apostrophe_word(self) -> None:
        clean = "She sells the farmers' market eggs."
        typo = "She sells the farmres' market eggs."
        spans = align_typo_spans(clean, typo, [_tok(" farmers", "farmres")])
        assert len(spans) == 1
        assert spans[0].clean_word == "farmers"
        assert spans[0].typo_word == "farmres"


class TestCharSpanToLastToken:
    def test_last_token_overlapping_span(self) -> None:
        # トークン offset: [0,5),[5,9),[9,14)
        offsets = [(0, 5), (5, 9), (9, 14)]
        # スパン [5,9) → トークン1
        assert char_span_to_last_token(offsets, (5, 9)) == 1
        # スパン [3,12) → 最後に重なるのはトークン2
        assert char_span_to_last_token(offsets, (3, 12)) == 2

    def test_subword_split_word(self) -> None:
        # "Janeet" が "Jane"+"et" に分割された場合、スパン末尾トークンは後者
        offsets = [(0, 3), (3, 7), (7, 9), (9, 15)]
        assert char_span_to_last_token(offsets, (3, 9)) == 2

    def test_special_tokens_with_zero_span_are_ignored(self) -> None:
        # BOS 等の (0,0) は無視される
        offsets = [(0, 0), (0, 5), (5, 9)]
        assert char_span_to_last_token(offsets, (0, 5)) == 1

    def test_no_overlap_returns_none(self) -> None:
        offsets = [(0, 5), (5, 9)]
        assert char_span_to_last_token(offsets, (20, 25)) is None

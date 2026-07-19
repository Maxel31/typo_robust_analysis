"""実験2: replacement (同品詞・同頻度帯の置換語サンプラ) のテスト.

計画 §4 実験2-2 操作(c): 「同品詞・同頻度帯の別語への置換」。
POS タガーと語彙は注入可能 (ユニットテストはフェイクで駆動)。
"""

import random

from typo_cot.intervention.replacement import (
    ReplacementSampler,
    VocabEntry,
    heuristic_pos,
)

FAKE_VOCAB = [
    VocabEntry(word="cards", pos="NOUN", zipf=4.4),
    VocabEntry(word="books", pos="NOUN", zipf=4.6),
    VocabEntry(word="stones", pos="NOUN", zipf=4.0),
    VocabEntry(word="galaxy", pos="NOUN", zipf=3.0),
    VocabEntry(word="run", pos="VERB", zipf=5.2),
    VocabEntry(word="jump", pos="VERB", zipf=4.5),
    VocabEntry(word="eggs", pos="NOUN", zipf=4.5),  # 同一語は選ばれない
]


def fake_tagger(word: str) -> str:
    return {"eggs": "NOUN", "bakes": "VERB", "Eggs": "NOUN"}.get(word, "NOUN")


class TestReplacementSampler:
    def test_same_pos_and_close_zipf(self):
        sampler = ReplacementSampler(vocab=FAKE_VOCAB, tagger=fake_tagger)
        # eggs (zipf≈4.5) → 最初の緩和帯 |Δ|<=0.5 に cards/books が入る
        repl = sampler.sample("eggs", rng=random.Random(0))
        assert repl in {"cards", "books"}

    def test_never_returns_same_word(self):
        sampler = ReplacementSampler(vocab=FAKE_VOCAB, tagger=fake_tagger)
        for seed in range(20):
            assert sampler.sample("eggs", rng=random.Random(seed)) != "eggs"

    def test_verb_gets_verb(self):
        sampler = ReplacementSampler(vocab=FAKE_VOCAB, tagger=fake_tagger)
        repl = sampler.sample("bakes", rng=random.Random(1))
        assert repl in {"run", "jump"}

    def test_capitalization_preserved(self):
        sampler = ReplacementSampler(vocab=FAKE_VOCAB, tagger=fake_tagger)
        repl = sampler.sample("Eggs", rng=random.Random(0))
        assert repl[0].isupper()

    def test_deterministic_with_seeded_rng(self):
        sampler = ReplacementSampler(vocab=FAKE_VOCAB, tagger=fake_tagger)
        r1 = sampler.sample("eggs", rng=random.Random(5))
        r2 = sampler.sample("eggs", rng=random.Random(5))
        assert r1 == r2

    def test_no_candidate_returns_none(self):
        vocab = [VocabEntry(word="run", pos="VERB", zipf=5.0)]
        sampler = ReplacementSampler(vocab=vocab, tagger=lambda w: "NOUN")
        assert sampler.sample("eggs", rng=random.Random(0)) is None

    def test_default_vocab_from_wordfreq(self):
        # 既定語彙は wordfreq 上位語から構築され、十分な規模を持つ
        sampler = ReplacementSampler(top_n=2000)
        repl = sampler.sample("eggs", rng=random.Random(0))
        assert repl is not None
        assert repl.lower() != "eggs"


class TestHeuristicPos:
    def test_adverb_suffix(self):
        assert heuristic_pos("quickly") == "ADV"

    def test_verb_suffix(self):
        assert heuristic_pos("baking") == "VERB"

    def test_default_noun(self):
        assert heuristic_pos("breakfast") == "NOUN"

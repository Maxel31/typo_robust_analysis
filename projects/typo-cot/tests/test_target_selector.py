"""実験2: target_selector (層判定 + top/bottom/matched-random/LOO 標的選定) のテスト.

計画 §4 実験2-1/2:
- 標的は clean CoT の答え句の外にある最上位 R_C **内容語**タイプ
- 選択肢文字・答え句内トークンは対象外
- 数値トークン (計算結果) は内容語層と分けて別枠 (numeric 層)
- matched_random は頻度・文字長一致のランダム内容語
"""

import random

from typo_cot.intervention.target_selector import (
    Candidate,
    build_candidates,
    normalize_ranking,
    rng_for_sample,
    select_matched_random,
    select_top,
    word_stratum,
)

COT = (
    "\nJanet's ducks lay 16 eggs per day.\n"
    "She eats 3 eggs for breakfast and bakes muffins.\n"
    "So she has 16 - 3 = 13 eggs left, selling each for $2.\n"
)


class TestWordStratum:
    def test_content_words(self):
        assert word_stratum("eggs") == "content"
        assert word_stratum("breakfast") == "content"
        assert word_stratum("muffins") == "content"

    def test_numeric_words(self):
        assert word_stratum("16") == "numeric"
        assert word_stratum("$2") == "numeric"
        assert word_stratum("13.5") == "numeric"

    def test_operator_words_are_numeric_stratum(self):
        assert word_stratum("=") == "numeric"
        assert word_stratum("-") == "numeric"
        assert word_stratum("*") == "numeric"

    def test_stopwords_are_other(self):
        assert word_stratum("the") == "other"
        assert word_stratum("She") == "other"
        assert word_stratum("for") == "other"

    def test_choice_letters_are_other(self):
        # 単一文字 (選択肢文字 A-J など) は content 候補から除外
        assert word_stratum("B") == "other"
        assert word_stratum("a") == "other"


class TestBuildCandidates:
    def test_content_and_numeric_candidates(self):
        cands = build_candidates(COT)
        by_word = {c.word: c for c in cands}
        assert by_word["eggs"].stratum == "content"
        assert by_word["eggs"].n_occurrences == 3
        assert by_word["16"].stratum == "numeric"
        assert "the" not in by_word  # ストップワード除外
        assert all(isinstance(c, Candidate) for c in cands)

    def test_candidates_have_zipf_and_length(self):
        cands = build_candidates(COT)
        egg = next(c for c in cands if c.word == "eggs")
        assert egg.length == 4
        assert egg.zipf > 0

    def test_first_char_pos_points_to_first_occurrence(self):
        cands = build_candidates(COT)
        egg = next(c for c in cands if c.word == "eggs")
        assert COT[egg.first_char_pos : egg.first_char_pos + 4] == "eggs"
        assert COT.index("eggs") == egg.first_char_pos


class TestNormalizeRanking:
    def test_dedup_takes_max_score(self):
        scores = normalize_ranking(
            [
                {"word": "eggs", "score": 0.5},
                {"word": "eggs.", "score": 0.9},  # 端句読点は正規化で剥がれる
                {"word": "muffins", "score": 0.3},
            ]
        )
        assert scores["eggs"] == 0.9
        assert scores["muffins"] == 0.3

    def test_multiword_entries_are_expanded(self):
        # R_C 側は改行またぎで語が結合されることがある (例: "dollars.\nThe")
        scores = normalize_ranking([{"word": "dollars.\nThe", "score": 0.7}])
        assert scores["dollars"] == 0.7


class TestSelectTop:
    RANKING = [
        {"word": "16", "score": 3.7},  # numeric — content 層では飛ばす
        {"word": "eggs", "score": 1.2},
        {"word": "the", "score": 1.0},  # stopword — 飛ばす
        {"word": "muffins", "score": 0.8},
        {"word": "breakfast", "score": 0.4},
        {"word": "zebra", "score": 9.9},  # prefix に存在しない — 飛ばす
        {"word": "ducks", "score": 0.1},
    ]

    def test_top_content_selection(self):
        cands = build_candidates(COT)
        top = select_top(self.RANKING, cands, k=2, stratum="content")
        assert top == ["eggs", "muffins"]

    def test_top_numeric_selection_is_separate(self):
        cands = build_candidates(COT)
        top = select_top(self.RANKING, cands, k=1, stratum="numeric")
        assert top == ["16"]

    def test_bottom_selection(self):
        cands = build_candidates(COT)
        bottom = select_top(self.RANKING, cands, k=1, stratum="content", bottom=True)
        assert bottom == ["ducks"]

    def test_insufficient_candidates_returns_short_list(self):
        cands = build_candidates("Janet eggs")
        top = select_top(self.RANKING, cands, k=4, stratum="content")
        assert len(top) < 4


class TestSelectMatchedRandom:
    def test_matched_excludes_top_set_and_matches_length(self):
        cands = build_candidates(COT)
        rng = random.Random(0)
        matched = select_matched_random(["eggs"], cands, rng=rng)
        assert len(matched) == 1
        assert matched[0] != "eggs"
        # content 層から選ばれる
        by_word = {c.word: c for c in cands}
        assert by_word[matched[0]].stratum == "content"
        # 文字長は近い (緩和ステップ内)
        assert abs(by_word[matched[0]].length - 4) <= 3

    def test_deterministic_given_same_rng_seed(self):
        cands = build_candidates(COT)
        m1 = select_matched_random(["eggs", "muffins"], cands, rng=random.Random(7))
        m2 = select_matched_random(["eggs", "muffins"], cands, rng=random.Random(7))
        assert m1 == m2

    def test_without_replacement(self):
        cands = build_candidates(COT)
        matched = select_matched_random(["eggs", "muffins"], cands, rng=random.Random(3))
        assert len(matched) == len(set(matched)) == 2

    def test_exhausted_pool_returns_short_list(self):
        cands = build_candidates("Janet eggs muffins")
        matched = select_matched_random(
            ["eggs", "muffins", "Janet"], cands, rng=random.Random(0)
        )
        assert len(matched) < 3


class TestRngForSample:
    def test_deterministic_per_sample(self):
        r1 = rng_for_sample(42, "gsm8k_00001")
        r2 = rng_for_sample(42, "gsm8k_00001")
        assert r1.random() == r2.random()

    def test_differs_across_samples_and_seeds(self):
        assert rng_for_sample(42, "a").random() != rng_for_sample(42, "b").random()
        assert rng_for_sample(1, "a").random() != rng_for_sample(2, "a").random()

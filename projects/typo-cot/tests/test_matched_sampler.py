"""実験5(双子語統制)の層化マッチドサンプラのテスト.

GPU 不要: トークナイザは語彙ベースのフェイク、zipf・埋め込みは注入関数で代替する。
"""

import math

import pytest

from typo_cot.perturbation.matched_sampler import (
    FeatureExtractor,
    MatchedTwinSampler,
    MatchRecord,
    TokenFeatures,
    centrality_bin,
    compute_smd_table,
    function_word_class,
    strip_token,
    zipf_bin,
)


class VocabTokenizer:
    """語彙内の単語は1ピース、未知語は文字単位に分割するフェイクトークナイザ.

    typo が入った語は語彙から外れて文字分割される、というサブワード断片化の
    最小モデル。分割増分が決定的に計算できる。
    """

    def __init__(self, vocab: set[str]) -> None:
        self.vocab = set(vocab)

    def tokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        for word in text.strip().split():
            if word in self.vocab:
                pieces.append(word)
            else:
                pieces.extend(list(word))
        return pieces


VOCAB = {"machine", "network", "potato", "the", "ox", "learning", "banana", "quantum"}

ZIPF = {
    "machine": 4.0,
    "network": 4.1,
    "potato": 3.6,
    "the": 6.5,
    "ox": 3.0,
    "learning": 4.2,
    "banana": 3.9,
    "quantum": 4.05,
}


def make_extractor(embed_fn=None) -> FeatureExtractor:
    return FeatureExtractor(
        tokenizer=VocabTokenizer(VOCAB),
        zipf_fn=lambda w: ZIPF.get(w, 2.0),
        classify_fn=function_word_class,
        embed_fn=embed_fn,
        seed=42,
    )


class TestSurfaceHelpers:
    def test_strip_token_removes_markers_and_space(self) -> None:
        assert strip_token("▁infinite") == "infinite"
        assert strip_token(" Every") == "Every"
        assert strip_token("Ġword") == "word"

    def test_function_word_class(self) -> None:
        assert function_word_class(" the") == "function"
        assert function_word_class("▁machine") == "content"
        assert function_word_class(" The,") == "function"

    def test_zipf_bin_width_half(self) -> None:
        assert zipf_bin(4.16) == pytest.approx(4.0)
        assert zipf_bin(4.5) == pytest.approx(4.5)
        assert zipf_bin(3.99) == pytest.approx(3.5)

    def test_centrality_bin(self) -> None:
        assert centrality_bin(0.55) == 5
        assert centrality_bin(0.04) == 0


class TestFeatureExtractor:
    def test_basic_features(self) -> None:
        ex = make_extractor()
        f = ex.features("s1", (3, " machine", 0.9))
        assert isinstance(f, TokenFeatures)
        assert f.token_index == 3
        assert f.surface == "machine"
        assert f.word_class == "content"
        assert f.char_len == 7
        assert f.zipf == pytest.approx(4.0)
        assert f.zipf_bin == pytest.approx(4.0)
        # 埋め込み無効時は centrality は None
        assert f.centrality is None
        assert f.centrality_bin is None

    def test_split_increment_by_type(self) -> None:
        ex = make_extractor()
        # "machine" は語彙内 (1ピース)。typo後は未知語になり文字分割される。
        # omission: 7文字 -> 6文字 = 6ピース -> 増分 +5
        assert ex.split_increment("s1", " machine", "omission") == 5
        # proximity: 7文字のまま未知語化 -> 増分 +6
        assert ex.split_increment("s1", " machine", "proximity") == 6
        # double_typing: 8文字 -> 増分 +7
        assert ex.split_increment("s1", " machine", "double_typing") == 7

    def test_tentative_perturbation_matches_dataset_seed_formula(self) -> None:
        """仮適用の摂動は dataset.py と同じ token_seed 式で決定される."""
        from typo_cot.perturbation.generator import CharacterPerturbationGenerator

        ex = make_extractor()
        result = ex.perturbation_for("s1", " machine")
        token_seed = hash((42, "s1", " machine"))
        expected = CharacterPerturbationGenerator(seed=token_seed).perturb(" machine")
        assert result is not None and expected is not None
        assert result.perturbed == expected.perturbed
        assert result.perturbation_type == expected.perturbation_type

    def test_features_with_embedding(self) -> None:
        def embed(text: str) -> list[float]:
            # "machine" だけ質問と同方向、他は直交
            if "machine" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

        ex = make_extractor(embed_fn=embed)
        qvec = ex.question_embedding("a machine question")
        f = ex.features("s1", (3, " machine", 0.9), question_vec=qvec)
        assert f.centrality == pytest.approx(1.0)
        assert f.centrality_bin is not None


class TestMatchedTwinSampler:
    def test_exact_match_preferred(self) -> None:
        ex = make_extractor()
        sampler = MatchedTwinSampler(ex, num_perturbations=1, seed=42)
        tokens = [
            (3, " machine", 0.9),  # 標的 (top-1)
            (5, " network", 0.1),  # 完全一致候補 (content, len7, bin4.0)
            (6, " the", 0.05),  # 機能語 -> クラス不一致
            (7, " ox", 0.02),  # 長さが遠い
        ]
        candidates, records = sampler.select("s1", tokens)
        assert len(records) == 1
        rec = records[0]
        assert rec.target.token_index == 3
        assert rec.matched is not None
        assert rec.matched.token_index == 5
        assert rec.relaxation_label == "exact"
        assert rec.relaxation_level == 0
        # 摂動候補リストの先頭はマッチした双子語
        assert candidates[0][0] == 5

    def test_caliper_relaxation_logged(self) -> None:
        ex = make_extractor()
        sampler = MatchedTwinSampler(ex, num_perturbations=1, seed=42)
        tokens = [
            (3, " machine", 0.9),  # 標的 len7 bin4.0
            (5, " potato", 0.1),  # len6 bin3.5 -> exact 不成立, caliper で成立
        ]
        _, records = sampler.select("s1", tokens)
        rec = records[0]
        assert rec.matched is not None
        assert rec.matched.token_index == 5
        assert rec.relaxation_label == "caliper"

    def test_fallback_to_any(self) -> None:
        ex = make_extractor()
        sampler = MatchedTwinSampler(ex, num_perturbations=1, seed=42)
        tokens = [
            (3, " machine", 0.9),  # content 標的
            (6, " the", 0.05),  # プールには機能語しかない
        ]
        _, records = sampler.select("s1", tokens)
        rec = records[0]
        assert rec.matched is not None
        assert rec.matched.token_index == 6
        assert rec.relaxation_label == "any"

    def test_without_replacement_and_excludes_targets(self) -> None:
        ex = make_extractor()
        sampler = MatchedTwinSampler(ex, num_perturbations=2, seed=42)
        tokens = [
            (1, " machine", 0.9),
            (2, " learning", 0.8),
            (3, " network", 0.1),
            (4, " quantum", 0.05),
            (5, " banana", 0.02),
        ]
        candidates, records = sampler.select("s1", tokens)
        target_indices = {1, 2}
        matched_indices = [r.matched.token_index for r in records if r.matched]
        assert len(matched_indices) == 2
        # 非復元抽出
        assert len(set(matched_indices)) == 2
        # 標的 (top-k) 自身は双子語に選ばれない
        assert not target_indices & set(matched_indices)
        # 候補リストにも標的は含まれない (統制条件の純度)
        assert not target_indices & {c[0] for c in candidates}

    def test_small_pool_falls_back_to_all_tokens(self) -> None:
        ex = make_extractor()
        sampler = MatchedTwinSampler(ex, num_perturbations=2, seed=42)
        tokens = [
            (1, " machine", 0.9),
            (2, " learning", 0.8),
        ]  # トークン数 <= k: rebuttal 同様 全トークンをプールにフォールバック
        candidates, records = sampler.select("s1", tokens)
        assert len(records) == 2
        assert all(r.matched is not None for r in records)


class TestSMDTable:
    @staticmethod
    def _feat(idx: int, length: int, zipf: float, inc: int, cls: str = "content") -> TokenFeatures:
        return TokenFeatures(
            token_index=idx,
            token="t",
            surface="t",
            word_class=cls,
            char_len=length,
            zipf=zipf,
            zipf_bin=zipf_bin(zipf),
            split_increment=inc,
            perturbation_type="omission",
            centrality=None,
            centrality_bin=None,
        )

    def test_smd_zero_for_identical_groups(self) -> None:
        records = [
            MatchRecord("s1", self._feat(1, 7, 4.0, 5), self._feat(2, 7, 4.0, 5), 0, "exact"),
            MatchRecord("s1", self._feat(3, 5, 3.5, 4), self._feat(4, 5, 3.5, 4), 0, "exact"),
        ]
        table = compute_smd_table(records)
        assert table["n_targets"] == 2
        assert table["n_matched"] == 2
        assert table["smd"]["char_len"] == pytest.approx(0.0)
        assert table["smd"]["zipf"] == pytest.approx(0.0)
        assert table["smd"]["split_increment"] == pytest.approx(0.0)
        assert table["class_match_rate"] == pytest.approx(1.0)
        assert table["relaxation_rates"]["exact"] == pytest.approx(1.0)

    def test_smd_known_value(self) -> None:
        # 標的 char_len [8, 6] (mean 7, var 2), マッチ [7, 5] (mean 6, var 2)
        # SMD = 1 / sqrt((2+2)/2) = 1/sqrt(2)
        records = [
            MatchRecord("s1", self._feat(1, 8, 4.0, 5), self._feat(2, 7, 4.0, 5), 2, "caliper"),
            MatchRecord("s1", self._feat(3, 6, 4.0, 5), self._feat(4, 5, 4.0, 5), 2, "caliper"),
        ]
        table = compute_smd_table(records)
        assert table["smd"]["char_len"] == pytest.approx(1.0 / math.sqrt(2.0))
        assert table["relaxation_rates"]["caliper"] == pytest.approx(1.0)

    def test_unmatched_excluded_from_smd(self) -> None:
        records = [
            MatchRecord("s1", self._feat(1, 8, 4.0, 5), self._feat(2, 8, 4.0, 5), 0, "exact"),
            MatchRecord("s1", self._feat(3, 6, 4.0, 5), None, 5, "unmatched"),
        ]
        table = compute_smd_table(records)
        assert table["n_targets"] == 2
        assert table["n_matched"] == 1

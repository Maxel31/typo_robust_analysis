"""実験6-(iv) LOO スコアラのユニットテスト（GPU 不要・モックのみ）.

LOO の定義（docs/dev_notes_06_attribution.md / experiment_plan.md §4 実験6-(iv)）:
clean CoT の各語タイプについて全出現を削除した変種 CoT を作り、
(質問 + 変種 CoT + 答えトリガー) を teacher-forcing して
「元の答えトークン列の log-prob 合計」の低下量をその語の重要度とする。
"""

from types import SimpleNamespace

import pytest
import torch

from typo_cot.intervention.loo_scorer import (
    batched_answer_logprobs,
    build_loo_variants,
    delete_word_type,
    extract_word_types,
    loo_jaccard_topk,
    normalize_word,
    rc_word_ranking_from_cot_pt,
    score_sample_loo,
    sequence_logprob,
    split_generated_text,
)


# ============================================================
# モック（GPU / ネットワーク不要）
# ============================================================


class MockTokenizer:
    """空白区切りのトークナイザ. 同一インスタンス内で語彙は安定."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}
        self.pad_token_id = 0

    def _ids(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            ids.append(self.vocab[w])
        return ids

    def __call__(self, text: str, return_tensors=None, **kwargs):
        ids = self._ids(text)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class MockModel(torch.nn.Module):
    """位置文脈に依存しない決定的 logits を返すモデル.

    logits[b, t, :] = W[input_ids[b, t]] なのでパディングが
    非パディング位置のスコアに影響しない（バッチ一致検証に使う）。
    """

    def __init__(self, vocab_size: int = 64, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(vocab_size, vocab_size, generator=gen))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.W[input_ids])


@pytest.fixture()
def mock_lm():
    return MockModel(), MockTokenizer()


def manual_logprob(model: MockModel, ids: list[int], start: int) -> float:
    """テスト側の独立実装: ids[start:] の log-prob 合計."""
    logp = torch.log_softmax(model.W.float(), dim=-1)
    return float(sum(logp[ids[i - 1], ids[i]] for i in range(start, len(ids))))


# ============================================================
# split_generated_text
# ============================================================


class TestSplitGeneratedText:
    def test_gsm8k_number_answer(self):
        text = "\n16 - 3 - 4 = 9.\nSo she makes 18 dollars.\nThe answer is 18."
        split = split_generated_text(text)
        assert split is not None
        assert split.cot_text == "\n16 - 3 - 4 = 9.\nSo she makes 18 dollars.\n"
        assert split.trigger_text == "The answer is "
        assert split.answer_text == "18"
        assert split.pattern_type == "number"

    def test_mmlu_choice_answer(self):
        text = "Reasoning here.\nThe answer is (B)."
        split = split_generated_text(text)
        assert split is not None
        assert split.cot_text == "Reasoning here.\n"
        assert split.trigger_text == "The answer is ("
        assert split.answer_text == "B"
        assert split.pattern_type == "choice"

    def test_uses_last_match(self):
        text = "The answer is 5. Wait, recompute: 5 + 2 = 7.\nThe answer is 7."
        split = split_generated_text(text)
        assert split is not None
        assert split.answer_text == "7"
        assert split.cot_text.startswith("The answer is 5. Wait")

    def test_no_answer_pattern_returns_none(self):
        assert split_generated_text("no final answer here") is None

    def test_reconstruction_identity(self):
        """cot + trigger + answer が元テキストの接頭辞になっている."""
        text = "A step.\nThe answer is 42. Done."
        split = split_generated_text(text)
        assert text.startswith(split.cot_text + split.trigger_text + split.answer_text)


# ============================================================
# 語タイプ抽出と削除
# ============================================================


class TestExtractWordTypes:
    def test_groups_all_occurrences_of_type(self):
        wts = extract_word_types("16 eggs and 16 more eggs.")
        by_word = {wt.word: wt for wt in wts}
        assert len(by_word["16"].spans) == 2
        assert len(by_word["eggs"].spans) == 2  # "eggs" と "eggs." が同一タイプ

    def test_edge_punctuation_stripped_from_key(self):
        wts = extract_word_types("She has 9 eggs.")
        words = {wt.word for wt in wts}
        assert "eggs" in words
        assert "eggs." not in words

    def test_pure_punctuation_chunk_kept_as_operator(self):
        """GSM8K の演算語 (=, -, *) は語タイプとして残す."""
        wts = extract_word_types("16 - 3 = 13")
        words = {wt.word for wt in wts}
        assert {"16", "-", "3", "=", "13"} == words

    def test_case_sensitive(self):
        wts = extract_word_types("She said she left")
        words = [wt.word for wt in wts]
        assert "She" in words and "she" in words

    def test_spans_point_to_key_substring(self):
        text = "wait, eggs."
        for wt in extract_word_types(text):
            for s, e in wt.spans:
                assert text[s:e] == wt.word

    def test_word_count_scale(self):
        """語タイプ数 = ユニーク語数（変種数の根拠）."""
        text = "a b c a b a"
        assert len(extract_word_types(text)) == 3


class TestDeleteWordType:
    def _delete(self, text: str, word: str) -> str:
        wts = {wt.word: wt for wt in extract_word_types(text)}
        return delete_word_type(text, wts[word])

    def test_removes_all_occurrences(self):
        out = self._delete("16 minus 3 leaves 16 eggs", "16")
        assert "16" not in out
        assert "minus" in out and "eggs" in out

    def test_no_substring_deletion(self):
        out = self._delete("cat catalog cat", "cat")
        assert "catalog" in out
        assert out.split() == ["catalog"]

    def test_punctuation_preserved(self):
        out = self._delete("She has 9 eggs.", "eggs")
        assert "eggs" not in out
        assert "." in out

    def test_no_double_spaces(self):
        out = self._delete("a b c b a", "b")
        assert "  " not in out

    def test_other_words_untouched(self):
        out = self._delete("x y z", "y")
        assert "x" in out and "z" in out


class TestBuildLooVariants:
    def test_one_variant_per_word_type(self):
        cot = "x y x z"
        word_types, variants = build_loo_variants(cot)
        assert len(word_types) == 3
        assert len(variants) == len(word_types)
        for wt, var in zip(word_types, variants, strict=True):
            assert wt.word not in var.split()


# ============================================================
# teacher-forcing log-prob
# ============================================================


class TestSequenceLogprob:
    def test_matches_manual_computation(self, mock_lm):
        model, tok = mock_lm
        context = "q1 q2 cot1 cot2 trigger"
        target = " ans1 ans2"
        got = sequence_logprob(model, tok, context, target)
        full_ids = tok(context + target)["input_ids"]
        expected = manual_logprob(model, full_ids, start=5)  # 答え2トークン分
        assert got == pytest.approx(expected, rel=1e-5)

    def test_scores_only_answer_tokens(self, mock_lm):
        """context が違っても target トークン数分だけスコアされる.

        MockModel は直前トークンのみで logits が決まるので、直前トークン
        (trigger 末尾) と target が同じなら log-prob は一致するはず。
        """
        model, tok = mock_lm
        lp1 = sequence_logprob(model, tok, "a b trigger", " ans")
        lp2 = sequence_logprob(model, tok, "c d e f trigger", " ans")
        assert lp1 == pytest.approx(lp2, rel=1e-6)

    def test_boundary_merge_fallback(self, mock_lm):
        """トークン境界がマージされる場合（target 先頭に空白なし）でも落ちない."""
        model, tok = mock_lm
        got = sequence_logprob(model, tok, "a b c", "d")  # "cd" にマージされる
        full_ids = tok("a b cd")["input_ids"]
        expected = manual_logprob(model, full_ids, start=2)
        assert got == pytest.approx(expected, rel=1e-5)

    def test_logprob_is_negative(self, mock_lm):
        """log-prob は必ず負（確率 < 1）."""
        model, tok = mock_lm
        assert sequence_logprob(model, tok, "a b trigger", " ans") < 0


class TestBatchedAnswerLogprobs:
    def test_matches_sequential(self, mock_lm):
        """可変長 context のバッチ計算が逐次計算と一致する."""
        model, tok = mock_lm
        contexts = [
            "q a b c trigger",
            "q a c trigger",  # 短い
            "q b c b a extra words trigger",  # 長い
            "q trigger",
        ]
        target = " ans1 ans2"
        sequential = [sequence_logprob(model, tok, c, target) for c in contexts]
        batched = batched_answer_logprobs(model, tok, contexts, target, batch_size=3)
        assert len(batched) == len(contexts)
        for b, s in zip(batched, sequential, strict=True):
            assert b == pytest.approx(s, rel=1e-4)

    def test_empty_contexts(self, mock_lm):
        model, tok = mock_lm
        assert batched_answer_logprobs(model, tok, [], " ans") == []


# ============================================================
# サンプル単位の LOO スコアリング
# ============================================================


class TestScoreSampleLoo:
    PROMPT = "Problem: p1 p2\n\nSolution:"
    GENERATED = " x y x z\nThe answer is 18."

    def test_variant_count_equals_word_types(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(model, tok, self.PROMPT, self.GENERATED)
        assert result is not None
        assert result["n_word_types"] == 3  # x, y, z
        assert len(result["word_scores"]) == 3
        assert len(result["word_types"]) == 3

    def test_ranking_schema_matches_rc(self, mock_lm):
        """R_C ランキング (cot_top_k_words) と同じ word/score スキーマ・降順."""
        model, tok = mock_lm
        result = score_sample_loo(model, tok, self.PROMPT, self.GENERATED)
        scores = [ws["score"] for ws in result["word_scores"]]
        assert scores == sorted(scores, reverse=True)
        for ws in result["word_scores"]:
            assert set(ws.keys()) == {"word", "score"}
            assert isinstance(ws["word"], str)
            assert isinstance(ws["score"], float)

    def test_score_is_logprob_drop(self, mock_lm):
        """score = base_logprob - variant_logprob."""
        model, tok = mock_lm
        result = score_sample_loo(model, tok, self.PROMPT, self.GENERATED)
        split = split_generated_text(self.GENERATED)
        base_expected = sequence_logprob(
            model, tok, self.PROMPT + split.cot_text + split.trigger_text, split.answer_text
        )
        assert result["base_logprob"] == pytest.approx(base_expected, rel=1e-5)
        by_word = {wt["word"]: wt for wt in result["word_types"]}
        y_variant = delete_word_type(
            split.cot_text,
            next(wt for wt in extract_word_types(split.cot_text) if wt.word == "y"),
        )
        y_expected = base_expected - sequence_logprob(
            model, tok, self.PROMPT + y_variant + split.trigger_text, split.answer_text
        )
        assert by_word["y"]["score"] == pytest.approx(y_expected, rel=1e-4)

    def test_answer_metadata(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(model, tok, self.PROMPT, self.GENERATED)
        assert result["answer_text"] == "18"
        assert result["trigger_text"] == "The answer is "
        assert isinstance(result["base_logprob"], float)

    def test_no_answer_pattern_returns_none(self, mock_lm):
        model, tok = mock_lm
        assert score_sample_loo(model, tok, self.PROMPT, " no answer at all") is None


# ============================================================
# R_C ランキング互換と Jaccard@k
# ============================================================


class TestRcWordRanking:
    def test_restricts_to_cot_region_and_sorts(self):
        data = {
            "word_scores": [
                {"word": "prompt_word", "score": 0.0, "token_indices": [0, 1]},
                {"word": "eggs", "score": 2.0, "token_indices": [10, 11]},
                {"word": "16", "score": 5.0, "token_indices": [12]},
                {"word": "the", "score": 0.1, "token_indices": [13]},
                {"word": "answer_region", "score": 9.9, "token_indices": [20]},
            ],
            "cot_token_start": 10,
            "cot_token_end": 15,
        }
        ranking = rc_word_ranking_from_cot_pt(data)
        assert [r["word"] for r in ranking] == ["16", "eggs", "the"]
        for r in ranking:
            assert set(r.keys()) == {"word", "score"}


class TestLooJaccard:
    def test_known_value(self):
        loo = [
            {"word": "a", "score": 3.0},
            {"word": "b", "score": 2.0},
            {"word": "c", "score": 1.0},
        ]
        rc = [
            {"word": "b.", "score": 9.0},
            {"word": "c", "score": 8.0},
            {"word": "d", "score": 7.0},
        ]
        # 正規化後 top3 集合 {a,b,c} vs {b,c,d} → 2/4
        assert loo_jaccard_topk(loo, rc, k=3) == pytest.approx(0.5)

    def test_identical_rankings_give_one(self):
        r = [{"word": w, "score": float(10 - i)} for i, w in enumerate("abcde")]
        assert loo_jaccard_topk(r, r, k=5) == pytest.approx(1.0)

    def test_empty_ranking_gives_zero(self):
        r = [{"word": "a", "score": 1.0}]
        assert loo_jaccard_topk([], r, k=10) == 0.0


class TestNormalizeWord:
    def test_strips_edge_punctuation(self):
        assert normalize_word("18.") == "18"
        assert normalize_word("**4**") == "4"
        assert normalize_word("(B)") == "B"

    def test_keeps_pure_punctuation(self):
        assert normalize_word("=") == "="
        assert normalize_word("-") == "-"

    def test_keeps_inner_punctuation(self):
        assert normalize_word("3.5") == "3.5"
        assert normalize_word("1,000") == "1,000"

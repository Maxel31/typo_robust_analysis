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
    align_tokens_to_text,
    batched_answer_logprobs,
    build_loo_variants,
    delete_word_type,
    extract_word_types,
    loo_jaccard_topk,
    normalize_word,
    rc_word_ranking_from_cot_pt,
    rc_word_ranking_from_token_scores,
    score_sample_loo,
    sequence_logprob,
    split_generated_text,
    word_scores_degenerate,
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


class TestRcRankingRebuildFromTokenScores:
    """Mistral アーカイブ不良 (word_scores が全文1語に結合) の再構築フォールバック.

    アーカイブの tokens_to_words は先頭スペース/▁ で語境界を検出するが、
    Mistral の token_scores は空白マーカーなしのトークン文字列を持つため
    word_scores が全トークン結合の1語に潰れる。token_scores を既知の
    生成テキスト (prompt + generated_text) に貪欲整合して語ランキングを
    再構築する。
    """

    # "Q : add 3 and 4 .\n A : 3 + 4 = 7 . The answer is 7 ."
    FULL_TEXT = "Q: add 3 and 4.\nA: 3 + 4 = 7. The answer is 7."
    # Mistral 風: 空白マーカーなしトークン列 (<s> + 21 トークン)
    TOKENS = [
        "<s>", "Q", ":", "add", "3", "and", "4", ".", "\n", "A", ":",
        "3", "+", "4", "=", "7", ".", "The", "answer", "is", "7", ".",
    ]
    # CoT 領域 = "3 + 4 = 7." (トークン 11..16)
    COT_START, COT_END = 11, 16

    def _token_scores(self):
        by_index = {11: 0.5, 12: 1.0, 13: 2.0, 14: 3.0, 15: 5.0, 16: 0.25}
        return [(t, by_index.get(i, 0.1)) for i, t in enumerate(self.TOKENS)]

    def _degenerate_data(self):
        return {
            "word_scores": [
                {
                    "word": self.FULL_TEXT.replace(" ", "").replace("\n", ""),
                    "score": 1.23,
                    "token_indices": list(range(1, len(self.TOKENS))),
                }
            ],
            "token_scores": self._token_scores(),
            "cot_token_start": self.COT_START,
            "cot_token_end": self.COT_END,
        }

    def test_degenerate_detection_on_merged_word_scores(self):
        assert word_scores_degenerate(self._degenerate_data())

    def test_healthy_word_scores_not_degenerate(self):
        data = {
            "word_scores": [
                {"word": "eggs", "score": 2.0, "token_indices": [10, 11]},
                {"word": "16", "score": 5.0, "token_indices": [12]},
            ],
            "token_scores": [("a", 0.1)] * 20,
        }
        assert not word_scores_degenerate(data)

    def test_align_mistral_style_tokens(self):
        spans = align_tokens_to_text(self.TOKENS, self.FULL_TEXT)
        assert spans is not None
        assert spans[0] is None  # <s>
        assert self.FULL_TEXT[slice(*spans[3])] == "add"
        # 貪欲整合: 最初の "3" は質問側 (token 4)、CoT 側は token 11
        assert spans[4] == (7, 8)
        assert self.FULL_TEXT[slice(*spans[11])] == "3"
        assert spans[11][0] > spans[4][0]

    def test_align_space_and_sentencepiece_markers(self):
        # Gemma 風 (先頭スペース) と ▁ マーカーも同じ整合器で扱える
        text = "He ran.\nFast!"
        tokens = ["<bos>", "He", " ran", ".", "▁Fast", "!"]
        spans = align_tokens_to_text(tokens, text)
        assert spans is not None
        assert text[slice(*spans[2])] == " ran"
        assert text[slice(*spans[4])] == "Fast"

    def test_align_byte_fallback_token(self):
        text = "a\nb"
        tokens = ["a", "<0x0A>", "b"]
        spans = align_tokens_to_text(tokens, text)
        assert spans is not None
        assert spans[1] == (1, 2)

    def test_align_failure_returns_none(self):
        assert align_tokens_to_text(["zzz"], "abc") is None

    def test_rebuild_ranking_restricted_to_cot_region(self):
        ranking = rc_word_ranking_from_token_scores(
            self._degenerate_data(), self.FULL_TEXT
        )
        assert ranking is not None
        # 領域内チャンク: "7."(5.25) > "="(3.0) > "4"(2.0) > "+"(1.0) > "3"(0.5)
        assert [r["word"] for r in ranking] == ["7.", "=", "4", "+", "3"]
        assert ranking[0]["score"] == pytest.approx(5.25)
        assert ranking[1]["score"] == pytest.approx(3.0)
        for r in ranking:
            assert set(r.keys()) == {"word", "score"}

    def test_cot_pt_falls_back_to_rebuild_when_degenerate(self):
        ranking = rc_word_ranking_from_cot_pt(
            self._degenerate_data(), full_text=self.FULL_TEXT
        )
        assert [r["word"] for r in ranking] == ["7.", "=", "4", "+", "3"]

    def test_cot_pt_keeps_existing_path_when_healthy(self):
        data = {
            "word_scores": [
                {"word": "eggs", "score": 2.0, "token_indices": [10]},
                {"word": "outside", "score": 9.0, "token_indices": [99]},
            ],
            "token_scores": [("eggs", 2.0)],
            "cot_token_start": 5,
            "cot_token_end": 15,
        }
        ranking = rc_word_ranking_from_cot_pt(data, full_text="eggs outside")
        assert ranking == [{"word": "eggs", "score": 2.0}]

    def test_cot_pt_degenerate_without_full_text_uses_existing_path(self):
        # full_text なしでは従来挙動 (結合1語がそのまま返る) — 後方互換
        ranking = rc_word_ranking_from_cot_pt(self._degenerate_data())
        assert len(ranking) == 1

    def test_cot_pt_degenerate_align_failure_falls_back(self):
        data = self._degenerate_data()
        ranking = rc_word_ranking_from_cot_pt(data, full_text="unrelated text")
        assert len(ranking) == 1  # 再構築失敗 → 従来経路


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


# ============================================================
# clean vs perturbed の LOO 版 Jaccard@k ペアリング
# ============================================================


class TestComputeLooJaccardPairs:
    def _entry(self, sid, words):
        return {
            "sample_id": sid,
            "loo_word_scores": [
                {"word": w, "score": float(len(words) - i)} for i, w in enumerate(words)
            ],
        }

    def test_pairs_by_sample_id(self):
        from typo_cot.intervention.loo_scorer import compute_loo_jaccard_pairs

        clean = [self._entry("s1", ["a", "b", "c"]), self._entry("s2", ["x", "y"])]
        pert = [self._entry("s2", ["x", "z"]), self._entry("s1", ["a", "b", "c"])]
        pairs = compute_loo_jaccard_pairs(clean, pert, k=3)
        by_id = {p["sample_id"]: p for p in pairs}
        assert by_id["s1"]["loo_jaccard"] == pytest.approx(1.0)
        # {x,y} vs {x,z} -> 1/3
        assert by_id["s2"]["loo_jaccard"] == pytest.approx(1 / 3)

    def test_skips_unmatched_samples(self):
        from typo_cot.intervention.loo_scorer import compute_loo_jaccard_pairs

        clean = [self._entry("s1", ["a"]), self._entry("only_clean", ["b"])]
        pert = [self._entry("s1", ["a"]), self._entry("only_pert", ["c"])]
        pairs = compute_loo_jaccard_pairs(clean, pert, k=10)
        assert [p["sample_id"] for p in pairs] == ["s1"]


# ============================================================
# run_loo_scoring.py: GPU ID 解決 (run_with_gpu.sh 互換)
# ============================================================


def _load_run_loo_scoring_module():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_loo_scoring.py"
    )
    spec = importlib.util.spec_from_file_location("run_loo_scoring", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestResolveGpuId:
    """setup_device が CUDA_VISIBLE_DEVICES を gpu_id で上書きするため、
    run_with_gpu.sh が設定した CUDA_VISIBLE_DEVICES を優先する必要がある。"""

    def test_env_cvd_takes_precedence_over_cli_default(self):
        mod = _load_run_loo_scoring_module()
        assert mod.resolve_gpu_id("0", {"CUDA_VISIBLE_DEVICES": "3"}) == "3"

    def test_cli_used_when_env_unset(self):
        mod = _load_run_loo_scoring_module()
        assert mod.resolve_gpu_id("1", {}) == "1"

    def test_empty_env_cvd_falls_back_to_cli(self):
        mod = _load_run_loo_scoring_module()
        assert mod.resolve_gpu_id("0", {"CUDA_VISIBLE_DEVICES": ""}) == "0"


# ============================================================
# 案B: 出現ごと削除 → タイプへ集約（deletion_mode="occurrence"）
# ============================================================


class TestBuildLooVariantsOccurrence:
    def test_one_variant_per_occurrence(self):
        from typo_cot.intervention.loo_scorer import build_loo_variants_occurrence

        cot = "x y x z"
        word_types, occ_type_idx, variants = build_loo_variants_occurrence(cot)
        # 変種数 = 出現数合計（4）、タイプ数は3
        assert len(word_types) == 3
        assert len(variants) == 4
        assert len(occ_type_idx) == 4
        total_occ = sum(len(wt.spans) for wt in word_types)
        assert len(variants) == total_occ

    def test_each_variant_deletes_exactly_one_occurrence(self):
        from typo_cot.intervention.loo_scorer import build_loo_variants_occurrence

        cot = "x y x z"
        word_types, occ_type_idx, variants = build_loo_variants_occurrence(cot)
        for ti, var in zip(occ_type_idx, variants, strict=True):
            wt = word_types[ti]
            n_before = cot.split().count(wt.word)
            n_after = var.split().count(wt.word)
            assert n_after == n_before - 1  # ちょうど1出現だけ消える

    def test_other_words_untouched(self):
        from typo_cot.intervention.loo_scorer import build_loo_variants_occurrence

        cot = "cat catalog cat"
        word_types, occ_type_idx, variants = build_loo_variants_occurrence(cot)
        cat_idx = next(i for i, wt in enumerate(word_types) if wt.word == "cat")
        cat_variants = [
            v for ti, v in zip(occ_type_idx, variants, strict=True) if ti == cat_idx
        ]
        assert len(cat_variants) == 2
        for var in cat_variants:
            assert "catalog" in var.split()  # 部分文字列削除で壊れない
            assert var.split().count("cat") == 1


class TestScoreSampleLooOccurrenceMode:
    PROMPT = "Problem: p1 p2\n\nSolution:"
    GENERATED = " x y x z\nThe answer is 18."

    def test_default_mode_is_occurrence(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(model, tok, self.PROMPT, self.GENERATED)
        assert result["deletion_mode"] == "occurrence"
        assert result["aggregation"] == "mean"

    def test_n_variants_equals_total_occurrences(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="occurrence"
        )
        # x が2出現、y/z が各1出現 → 変種4
        assert result["n_variants"] == 4
        assert result["n_word_types"] == 3
        assert len(result["word_scores"]) == 3  # ランキングはタイプ単位のまま

    def test_type_score_is_mean_of_occurrence_scores(self, mock_lm):
        """タイプ重要度 = 出現スコアの平均（Li et al. 2016 準拠）."""
        model, tok = mock_lm
        result = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="occurrence"
        )
        by_word = {d["word"]: d for d in result["word_types"]}
        x = by_word["x"]
        assert x["n_occurrences"] == 2
        assert len(x["occurrence_scores"]) == 2
        assert x["score"] == pytest.approx(
            sum(x["occurrence_scores"]) / 2, rel=1e-6
        )
        assert x["score_max"] == pytest.approx(max(x["occurrence_scores"]), rel=1e-6)

    def test_occurrence_score_matches_manual_single_deletion(self, mock_lm):
        """各出現スコア = base - (その出現だけ削除した変種の log-prob)."""
        model, tok = mock_lm
        result = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="occurrence"
        )
        split = split_generated_text(self.GENERATED)
        wts = {wt.word: wt for wt in extract_word_types(split.cot_text)}
        x = wts["x"]
        base = sequence_logprob(
            model, tok, self.PROMPT + split.cot_text + split.trigger_text, split.answer_text
        )
        expected = []
        for s, e in x.spans:
            var = split.cot_text[:s] + split.cot_text[e:]
            import re as _re

            var = _re.sub(r"[ \t]{2,}", " ", var)
            lp = sequence_logprob(
                model, tok, self.PROMPT + var + split.trigger_text, split.answer_text
            )
            expected.append(base - lp)
        by_word = {d["word"]: d for d in result["word_types"]}
        for got, exp in zip(by_word["x"]["occurrence_scores"], expected, strict=True):
            assert got == pytest.approx(exp, rel=1e-4)

    def test_single_occurrence_words_match_type_mode(self, mock_lm):
        """1出現しかない語は案A/案Bでスコアが一致する."""
        model, tok = mock_lm
        r_occ = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="occurrence"
        )
        r_type = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="type"
        )
        occ = {d["word"]: d for d in r_occ["word_types"]}
        typ = {d["word"]: d for d in r_type["word_types"]}
        for w in ("y", "z"):
            assert occ[w]["score"] == pytest.approx(typ[w]["score"], rel=1e-4)

    def test_type_mode_metadata(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="type"
        )
        assert result["deletion_mode"] == "type"
        assert result["n_variants"] == result["n_word_types"] == 3

    def test_ranking_schema_unchanged_in_occurrence_mode(self, mock_lm):
        model, tok = mock_lm
        result = score_sample_loo(
            model, tok, self.PROMPT, self.GENERATED, deletion_mode="occurrence"
        )
        scores = [ws["score"] for ws in result["word_scores"]]
        assert scores == sorted(scores, reverse=True)
        for ws in result["word_scores"]:
            assert set(ws.keys()) == {"word", "score"}

    def test_invalid_mode_raises(self, mock_lm):
        model, tok = mock_lm
        with pytest.raises(ValueError):
            score_sample_loo(
                model, tok, self.PROMPT, self.GENERATED, deletion_mode="bogus"
            )


# ============================================================
# R_C 側の改行またぎ結合語の分解（Jaccard 計算パスの正規化）
# ============================================================


class TestExpandMultiwordEntries:
    def test_splits_newline_joined_word(self):
        from typo_cot.intervention.loo_scorer import expand_multiword_entries

        ranking = [{"word": "dollars.\nThe", "score": 3.0}, {"word": "eggs", "score": 1.0}]
        out = expand_multiword_entries(ranking)
        words = [d["word"] for d in out]
        assert "dollars." in words and "The" in words and "eggs" in words
        assert "dollars.\nThe" not in words
        # 分解された各語は親スコアを引き継ぐ
        by_word = {d["word"]: d["score"] for d in out}
        assert by_word["dollars."] == 3.0
        assert by_word["The"] == 3.0

    def test_noop_for_plain_words(self):
        from typo_cot.intervention.loo_scorer import expand_multiword_entries

        ranking = [{"word": "eggs", "score": 2.0}]
        assert expand_multiword_entries(ranking) == ranking


class TestLooJaccardMultiwordFix:
    def test_newline_joined_rc_word_matches_loo(self):
        """R_C 側 "dollars.\\nThe" が LOO 側 "dollars" / "The" と一致するようになる."""
        loo = [
            {"word": "dollars", "score": 3.0},
            {"word": "The", "score": 2.0},
        ]
        rc = [{"word": "dollars.\nThe", "score": 9.0}]
        # 分解 + 端句読点正規化後: {dollars, The} vs {dollars, The} → 1.0
        assert loo_jaccard_topk(loo, rc, k=10) == pytest.approx(1.0)

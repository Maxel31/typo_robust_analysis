"""intervention.free_generation のテスト (実験15: patch保持・CoT自由生成).

GPU 不要。純関数 (発散オンセット / スパン整列 / 語スパン探索 / CoT ROUGE-L の
薄いラッパ) と、小さなランダム Llama を用いた「恒等パッチ下の自由生成 bit 不変」
(sham 検証のユニット版) を検証する。実験15 の要は「早期層 patch を保持したまま
CoT 全体を greedy 生成する」ことなので、prefill でのパッチ注入が KV キャッシュ
経由で多数の decode ステップにわたり効果を保つこと・恒等パッチが生成を一切
変えないことをここで担保する。

ROUGE-L は既存の analysis.metrics.rouge_l_score (文字単位 LCS。論文 Table 6 の
cot_rouge_l_f1 と同一定義) を再利用する。free_generation はその薄いラッパのみ提供。
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from typo_cot.intervention.free_generation import (
    AlignedSpans,
    align_span_positions,
    cot_rouge_l,
    cot_rouge_l_f,
    divergence_index,
    generate_ids,
    generate_ids_patched,
    locate_word_char_spans,
)
from typo_cot.intervention.patching import capture_activations, find_decoder_layers


# ---------------------------------------------------------------------------
# CoT ROUGE-L (既存 analysis.metrics.rouge_l_score の再利用ラッパ)
# ---------------------------------------------------------------------------


class TestCotRougeL:
    def test_identical_is_one(self):
        out = cot_rouge_l("The answer is 5.", "The answer is 5.")
        assert out["f1"] == pytest.approx(1.0)
        assert out["precision"] == pytest.approx(1.0)
        assert out["recall"] == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        # 文字集合が完全に素
        assert cot_rouge_l("abc", "xyz")["f1"] == pytest.approx(0.0)

    def test_known_char_lcs(self):
        # reference="abc", hypothesis="axbyc": 文字 LCS = "abc" = 3
        out = cot_rouge_l("abc", "axbyc")
        assert out["recall"] == pytest.approx(3 / 3)
        assert out["precision"] == pytest.approx(3 / 5)
        assert out["f1"] == pytest.approx(2 * 0.6 * 1.0 / (0.6 + 1.0))

    def test_empty_is_zero(self):
        assert cot_rouge_l("", "abc")["f1"] == 0.0
        assert cot_rouge_l("abc", "")["f1"] == 0.0

    def test_f_convenience_matches_dict(self):
        ref, hyp = "Tom has 3 apples", "Tom has 5 apples"
        assert cot_rouge_l_f(ref, hyp) == pytest.approx(cot_rouge_l(ref, hyp)["f1"])


# ---------------------------------------------------------------------------
# 発散オンセット (生成トークン列の最初の分岐位置)
# ---------------------------------------------------------------------------


class TestDivergenceIndex:
    def test_mid_divergence(self):
        assert divergence_index([1, 2, 3, 4], [1, 2, 9, 4]) == 2

    def test_divergence_at_start(self):
        assert divergence_index([1], [2]) == 0

    def test_identical_is_none(self):
        assert divergence_index([1, 2, 3], [1, 2, 3]) is None

    def test_prefix_is_none(self):
        assert divergence_index([1, 2], [1, 2, 3]) is None
        assert divergence_index([1, 2, 3], [1, 2]) is None

    def test_empty(self):
        assert divergence_index([], []) is None
        assert divergence_index([], [1]) is None


# ---------------------------------------------------------------------------
# スパン整列 (exp8 の question_span 位置整列を自由生成用に流用)
# ---------------------------------------------------------------------------


class TestAlignSpanPositions:
    def test_drops_word_missing_on_either_side(self):
        aligned = align_span_positions([2, 5, 7], [3, 6, None])
        assert isinstance(aligned, AlignedSpans)
        assert aligned.clean_positions == [2, 5]
        assert aligned.pert_positions == [3, 6]
        assert aligned.n_words == 3
        assert aligned.n_dropped == 1

    def test_all_aligned(self):
        aligned = align_span_positions([1, 2], [4, 5])
        assert aligned.clean_positions == [1, 2]
        assert aligned.pert_positions == [4, 5]
        assert aligned.n_dropped == 0

    def test_all_dropped(self):
        aligned = align_span_positions([None, None], [1, 2])
        assert aligned.clean_positions == []
        assert aligned.pert_positions == []
        assert aligned.n_dropped == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            align_span_positions([1, 2], [1])


class TestLocateWordCharSpans:
    def test_finds_word_in_question_region(self):
        prompt = "Few-shot...\n\nQuestion: what is teh capital?\nAnswer:"
        spans = locate_word_char_spans(prompt, "what is teh capital?", ["teh"])
        assert len(spans) == 1
        s, e = spans[0]
        assert prompt[s:e] == "teh"

    def test_missing_word_is_none(self):
        prompt = "Question: what is the capital?"
        spans = locate_word_char_spans(prompt, "what is the capital?", ["zzz"])
        assert spans == [None]

    def test_empty_word_is_none(self):
        spans = locate_word_char_spans("Q: a b c", "a b c", [""])
        assert spans == [None]


# ---------------------------------------------------------------------------
# 生成ヘルパと恒等パッチ下の自由生成 bit 不変 (sham 検証のユニット版)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _input(seed: int, length: int = 12) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 99, (1, length), generator=g)


class TestGenerateHelpers:
    def test_generate_ids_returns_only_continuation(self, tiny_model):
        ids = _input(1)
        gen = generate_ids(tiny_model, ids, max_new_tokens=5, pad_id=0)
        assert isinstance(gen, list)
        assert len(gen) <= 5

    def test_generate_is_deterministic(self, tiny_model):
        ids = _input(1)
        a = generate_ids(tiny_model, ids, max_new_tokens=8, pad_id=0)
        b = generate_ids(tiny_model, ids, max_new_tokens=8, pad_id=0)
        assert a == b

    def test_identity_patch_keeps_free_generation_bit_identical(self, tiny_model):
        """早期層 residual の恒等パッチを prompt スパン位置に注入したまま長い自由
        生成をしても、無パッチ生成と bit 完全一致する (KV キャッシュ整合込み)."""
        ids = _input(2, length=10)
        span_positions = [3, 5, 7]  # 「摂動語スパン」相当 (prompt 内)
        layers = find_decoder_layers(tiny_model)
        early = [0, 1, 2]  # 早期窓 [0,3)
        cache = capture_activations(tiny_model, ids, span_positions, sites=("residual",))
        values = {li: cache.values("residual", li, span_positions) for li in early}

        base = generate_ids(tiny_model, ids, max_new_tokens=24, pad_id=0)
        patched = generate_ids_patched(
            tiny_model,
            layers,
            ids,
            site="residual",
            layer_indices=early,
            dst_positions=span_positions,
            values=values,
            max_new_tokens=24,
            pad_id=0,
        )
        assert patched == base
        assert len(base) > 4  # 実際に複数 decode ステップ回っている

    def test_cross_patch_changes_free_generation(self, tiny_model):
        """donor≠recipient のスパンパッチは自由生成を変える (効いていることの確認)."""
        recip = _input(2, length=10)
        donor = _input(7, length=10)
        span_positions = [3, 5, 7]
        layers = find_decoder_layers(tiny_model)
        early = [0, 1, 2]
        donor_cache = capture_activations(tiny_model, donor, span_positions, sites=("residual",))
        values = {li: donor_cache.values("residual", li, span_positions) for li in early}

        base = generate_ids(tiny_model, recip, max_new_tokens=24, pad_id=0)
        patched = generate_ids_patched(
            tiny_model,
            layers,
            recip,
            site="residual",
            layer_indices=early,
            dst_positions=span_positions,
            values=values,
            max_new_tokens=24,
            pad_id=0,
        )
        assert patched != base

"""intervention.divergence のテスト (実験3: forced-decoding divergence).

GPU 不要。小さな合成 logits テンソルのみで、位置別 KL / log-prob / rank の
スカラー化、CoT トークン列の 1:1 対応 (質問長差オフセット補正)、
発散オンセット、precision@k とそのシャッフル帰無分布を検証する。
"""

import math

import pytest
import torch

from typo_cot.intervention.divergence import (
    align_cot_targets,
    divergence_onset,
    positionwise_divergence,
    precision_at_k,
    shuffle_null_precision,
)


class TestAlignCotTargets:
    def test_alignment_with_offset(self):
        # prompt 長が異なる 2 run。CoT 部分のトークン ID は一致する
        clean_ids = [10, 11, 12, 5, 6, 7, 8]  # prompt=3 tokens, cot=4
        typo_ids = [20, 21, 22, 23, 24, 5, 6, 7, 8]  # prompt=5 tokens, cot=4
        aligned = align_cot_targets(clean_ids, 3, typo_ids, 5)
        assert aligned.ok is True
        assert aligned.cot_ids == [5, 6, 7, 8]
        assert aligned.start_clean == 3
        assert aligned.start_typo == 5

    def test_mismatch_flagged(self):
        clean_ids = [10, 11, 5, 6]
        typo_ids = [20, 21, 5, 99]
        aligned = align_cot_targets(clean_ids, 2, typo_ids, 2)
        assert aligned.ok is False

    def test_length_mismatch_flagged(self):
        aligned = align_cot_targets([1, 5, 6], 1, [2, 5], 1)
        assert aligned.ok is False


class TestPositionwiseDivergence:
    def test_identical_logits_zero_kl(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 8)
        prof = positionwise_divergence(logits, logits.clone(), [1, 2, 3, 0])
        assert len(prof.kl) == 4
        assert all(abs(v) < 1e-5 for v in prof.kl)
        assert prof.rank_clean == prof.rank_typo
        for a, b in zip(prof.logp_clean, prof.logp_typo, strict=True):
            assert a == pytest.approx(b, abs=1e-5)

    def test_matches_torch_reference(self):
        torch.manual_seed(1)
        t, v = 6, 10
        logits_c = torch.randn(t, v)
        logits_t = torch.randn(t, v)
        targets = [3, 7, 0, 9, 4, 4]
        prof = positionwise_divergence(logits_c, logits_t, targets, chunk_size=2)

        logp_c = torch.log_softmax(logits_c.float(), dim=-1)
        logp_t = torch.log_softmax(logits_t.float(), dim=-1)
        kl_ref = (logp_c.exp() * (logp_c - logp_t)).sum(-1)
        for i in range(t):
            assert prof.kl[i] == pytest.approx(kl_ref[i].item(), rel=1e-4, abs=1e-5)
            assert prof.logp_clean[i] == pytest.approx(logp_c[i, targets[i]].item(), rel=1e-4)
            assert prof.logp_typo[i] == pytest.approx(logp_t[i, targets[i]].item(), rel=1e-4)
            # rank は 1-indexed (最尤トークン = rank 1)
            rank_ref = int((logits_t[i] > logits_t[i, targets[i]]).sum().item()) + 1
            assert prof.rank_typo[i] == rank_ref

    def test_rank_one_for_argmax(self):
        logits = torch.tensor([[0.0, 5.0, 1.0]])
        prof = positionwise_divergence(logits, logits, [1])
        assert prof.rank_clean == [1]
        assert prof.rank_typo == [1]


class TestDivergenceOnset:
    def test_onset_first_position_over_threshold(self):
        ranks = [1, 2, 1, 8, 3, 20]
        assert divergence_onset(ranks, threshold=5) == 3

    def test_no_onset(self):
        assert divergence_onset([1, 1, 2], threshold=5) is None

    def test_empty(self):
        assert divergence_onset([], threshold=5) is None


class TestPrecisionAtK:
    def test_full_overlap(self):
        kl_words = [("alpha", 3.0), ("beta", 2.0), ("gamma", 1.0)]
        rc_words = [("beta", 9.0), ("alpha", 5.0), ("gamma", 0.5)]
        assert precision_at_k(kl_words, rc_words, k=3) == pytest.approx(1.0)

    def test_partial_overlap_case_insensitive(self):
        kl_words = [("Alpha", 3.0), ("delta", 2.0)]
        rc_words = [("alpha", 9.0), ("beta", 5.0)]
        assert precision_at_k(kl_words, rc_words, k=2) == pytest.approx(0.5)

    def test_k_larger_than_lists(self):
        kl_words = [("a", 1.0)]
        rc_words = [("a", 1.0)]
        assert precision_at_k(kl_words, rc_words, k=10) == pytest.approx(1.0)

    def test_abs_score_ranking(self):
        # R_C は負スコアも重要 (絶対値でランク)
        kl_words = [("x", 5.0), ("y", 1.0)]
        rc_words = [("x", -10.0), ("z", 2.0), ("y", 0.1)]
        assert precision_at_k(kl_words, rc_words, k=1) == pytest.approx(1.0)


class TestShuffleNull:
    def test_null_distribution_properties(self):
        # 10 語中 top-3 KL が top-3 R_C と完全一致 → 実測 precision=1.0、
        # シャッフル帰無分布の平均は k/n=0.3 付近
        words = [f"w{i}" for i in range(10)]
        kl_values = [10.0 - i for i in range(10)]  # w0..w2 が top-3
        rc_words = [("w0", 9.0), ("w1", 8.0), ("w2", 7.0)] + [
            (f"w{i}", 0.1) for i in range(3, 10)
        ]
        res = shuffle_null_precision(
            list(zip(words, kl_values, strict=True)), rc_words, k=3, n_shuffles=200, seed=0
        )
        assert res.observed == pytest.approx(1.0)
        assert 0.1 < res.null_mean < 0.5
        assert 0.0 <= res.p_value <= 0.05 + 1e-9
        assert not math.isnan(res.null_std)

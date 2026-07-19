"""実験13: 読み出し集中度(M3測定) — 集中度指標のTDDテスト.

LOO 重要度分布 / attention 質量分布の集中度 (Gini係数・top-k シェア・
有効語数) を合成データで凍結する。すべて合成 (in-memory) 入力で、
データファイルの読み書きは行わない。
"""

import math

import numpy as np
import pytest

from typo_cot.analysis.concentration import (
    answer_to_cot_distribution,
    attention_gini_per_layer,
    effective_count,
    gini,
    loo_sample_concentration,
    top1_share,
    topk_share,
)


# ---------------------------------------------------------------
# gini
# ---------------------------------------------------------------


def test_gini_uniform_is_zero():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([3.0, 3.0, 3.0]) == pytest.approx(0.0)


def test_gini_one_hot_is_n_minus_1_over_n():
    # 完全集中 (1語が全質量) の Gini は (n-1)/n
    assert gini([1, 0, 0, 0]) == pytest.approx(0.75)
    assert gini([0, 0, 5]) == pytest.approx(2.0 / 3.0)


def test_gini_monotone_more_concentrated_is_higher():
    g_uniform = gini([1, 1, 1, 1])
    g_mild = gini([4, 2, 1, 1])
    g_sharp = gini([10, 1, 1, 1])
    assert g_uniform < g_mild < g_sharp


def test_gini_clips_negative_by_default():
    # 負の LOO スコア (削除で logprob 上昇) は 0 にクリップして集中度を測る
    assert gini([5, -3, 0, 0]) == pytest.approx(gini([5, 0, 0, 0]))


def test_gini_edge_cases():
    assert math.isnan(gini([]))  # n==0
    assert gini([7.0]) == pytest.approx(0.0)  # 単一要素
    assert gini([0, 0, 0]) == pytest.approx(0.0)  # 全ゼロ
    assert gini([-1, -2]) == pytest.approx(0.0)  # 全負 → 全ゼロ扱い


def test_gini_scale_invariant():
    assert gini([1, 2, 3, 4]) == pytest.approx(gini([10, 20, 30, 40]))


# ---------------------------------------------------------------
# top1 / topk share
# ---------------------------------------------------------------


def test_top1_share():
    assert top1_share([1, 0, 0, 0]) == pytest.approx(1.0)
    assert top1_share([1, 1, 1, 1]) == pytest.approx(0.25)
    assert top1_share([8, 1, 1]) == pytest.approx(0.8)


def test_topk_share():
    assert topk_share([4, 3, 2, 1], 2) == pytest.approx(7.0 / 10.0)
    assert topk_share([1, 1, 1, 1], 4) == pytest.approx(1.0)
    # k がサイズ超過でも全和
    assert topk_share([2, 3], 10) == pytest.approx(1.0)


def test_share_negative_clip_and_zero_sum():
    assert top1_share([5, -5, 0]) == pytest.approx(1.0)
    assert top1_share([0, 0]) == pytest.approx(0.0)
    assert topk_share([], 3) == pytest.approx(0.0)


# ---------------------------------------------------------------
# effective count (participation ratio / inverse Simpson)
# ---------------------------------------------------------------


def test_effective_count():
    assert effective_count([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_count([1, 0, 0, 0]) == pytest.approx(1.0)
    # 有効語数は 1..n の間
    ec = effective_count([5, 1, 1, 1])
    assert 1.0 < ec < 4.0


# ---------------------------------------------------------------
# loo_sample_concentration
# ---------------------------------------------------------------


def test_loo_sample_concentration_schema_and_values():
    ws = [
        {"word": "a", "score": 8.0},
        {"word": "b", "score": 1.0},
        {"word": "c", "score": 1.0},
    ]
    out = loo_sample_concentration(ws)
    assert out["n_words"] == 3
    assert out["top1_share"] == pytest.approx(0.8)
    assert out["gini"] == pytest.approx(gini([8, 1, 1]))
    assert out["effective_count"] == pytest.approx(effective_count([8, 1, 1]))
    assert 0.0 <= out["gini"] <= 1.0


def test_loo_sample_concentration_empty():
    out = loo_sample_concentration([])
    assert out["n_words"] == 0
    assert math.isnan(out["gini"])


# ---------------------------------------------------------------
# attention: answer -> cot distribution
# ---------------------------------------------------------------


def test_answer_to_cot_distribution_extracts_rows_and_columns():
    # 5x5 attention, 答え位置=4, CoT位置=[1,2,3]
    A = np.array(
        [
            [1.0, 0, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0],
            [0.2, 0.3, 0.5, 0, 0],
            [0.1, 0.2, 0.3, 0.4, 0],
            [0.0, 0.6, 0.3, 0.1, 0.0],  # 答え行
        ]
    )
    dist = answer_to_cot_distribution(A, answer_positions=[4], cot_positions=[1, 2, 3])
    assert np.allclose(dist, [0.6, 0.3, 0.1])


def test_answer_to_cot_distribution_mean_over_answer_positions():
    A = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0.2, 0.4, 0],
            [0, 0.4, 0.2, 0],
        ]
    )
    dist = answer_to_cot_distribution(A, answer_positions=[2, 3], cot_positions=[1, 2])
    assert np.allclose(dist, [0.3, 0.3])


def test_attention_gini_per_layer_shapes_and_concentration():
    seq = 4
    # layer0: 答え行が1つのCoTトークンに集中, layer1: 一様
    layer0 = np.zeros((1, 1, seq, seq))
    layer0[0, 0, 3, :] = [0.0, 1.0, 0.0, 0.0]  # 答え位置3 -> cot位置1に集中
    layer1 = np.zeros((1, 1, seq, seq))
    layer1[0, 0, 3, :] = [0.0, 0.5, 0.5, 0.0]  # cot位置1,2に均等
    attns = (layer0, layer1)
    ginis = attention_gini_per_layer(
        attns, answer_positions=[3], cot_positions=[1, 2], batch_index=0
    )
    assert len(ginis) == 2
    assert ginis[0] > ginis[1]  # 集中している層のほうがGiniが高い
    assert ginis[0] == pytest.approx(0.5)  # one-hot over 2 -> (n-1)/n = 0.5

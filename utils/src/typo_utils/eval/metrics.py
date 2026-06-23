"""typo 頑健性評価のメトリクス。"""

from __future__ import annotations

import math
from collections.abc import Sequence


def accuracy(preds: Sequence, golds: Sequence) -> float:
    """正解率。"""
    if not golds:
        return 0.0
    correct = sum(int(p == g) for p, g in zip(preds, golds))
    return correct / len(golds)


def robustness_gap(clean_acc: float, typo_acc: float) -> float:
    """clean 入力と typo 入力の精度差（大きいほど脆弱）。"""
    return clean_acc - typo_acc


def relative_robustness(clean_acc: float, typo_acc: float) -> float:
    """typo 入力での精度維持率 typo_acc / clean_acc（1.0 で完全頑健）。"""
    if clean_acc == 0.0:
        return 0.0
    return typo_acc / clean_acc


def mean_logprob(logprobs: Sequence[float]) -> float:
    """トークン log-probability の平均。空なら 0.0。"""
    if not logprobs:
        return 0.0
    return sum(logprobs) / len(logprobs)


def perplexity(logprobs: Sequence[float]) -> float:
    """トークン log-probability 列から perplexity を計算。空なら inf。"""
    if not logprobs:
        return float("inf")
    return math.exp(-mean_logprob(logprobs))

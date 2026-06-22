"""perplexity / mean_logprob メトリクスのテスト。"""

import math

from typo_utils.eval.metrics import mean_logprob, perplexity


def test_mean_logprob_basic():
    assert mean_logprob([-1.0, -2.0, -3.0]) == -2.0


def test_mean_logprob_single():
    assert mean_logprob([-0.5]) == -0.5


def test_mean_logprob_empty():
    assert mean_logprob([]) == 0.0


def test_perplexity_basic():
    logprobs = [-1.0, -1.0, -1.0]
    expected = math.exp(1.0)
    assert abs(perplexity(logprobs) - expected) < 1e-6


def test_perplexity_zero_logprobs():
    assert abs(perplexity([0.0, 0.0]) - 1.0) < 1e-6


def test_perplexity_empty():
    assert perplexity([]) == float("inf")

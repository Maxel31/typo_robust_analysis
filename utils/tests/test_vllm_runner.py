"""vLLM 推論ラッパーのテスト。"""

from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

from typo_utils.models.vllm_runner import VLLMRunner, GenerationOutput


def test_generation_output_fields():
    out = GenerationOutput(text="hello", token_logprobs=[-0.5, -0.3])
    assert out.text == "hello"
    assert len(out.token_logprobs) == 2


def test_generation_output_defaults():
    out = GenerationOutput(text="hi")
    assert out.token_logprobs is None


@patch("typo_utils.models.vllm_runner.LLM")
def test_runner_init(mock_llm_cls):
    runner = VLLMRunner("test-model", tensor_parallel_size=2, gpu_memory_utilization=0.8)
    mock_llm_cls.assert_called_once()
    kwargs = mock_llm_cls.call_args.kwargs
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["gpu_memory_utilization"] == 0.8


@patch("typo_utils.models.vllm_runner.LLM")
def test_runner_sets_gpu_ids(mock_llm_cls):
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        VLLMRunner("test-model", gpu_ids=[2, 3])
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


@patch("typo_utils.models.vllm_runner.LLM")
def test_runner_no_gpu_ids_preserves_env(mock_llm_cls):
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        VLLMRunner("test-model")
        assert "CUDA_VISIBLE_DEVICES" not in os.environ or original is not None
    finally:
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


@patch("typo_utils.models.vllm_runner.LLM")
def test_score_log_likelihood_callable(mock_llm_cls):
    runner = VLLMRunner("test-model")
    assert callable(runner.score_log_likelihood)
    sig = inspect.signature(runner.score_log_likelihood)
    assert "prompts" in sig.parameters
    assert "continuations" in sig.parameters


@patch("typo_utils.models.vllm_runner.LLM")
def test_generate_callable(mock_llm_cls):
    runner = VLLMRunner("test-model")
    assert callable(runner.generate)
    sig = inspect.signature(runner.generate)
    assert "prompts" in sig.parameters


@patch("typo_utils.models.vllm_runner.LLM")
def test_compute_perplexity_callable(mock_llm_cls):
    runner = VLLMRunner("test-model")
    assert callable(runner.compute_perplexity)
    sig = inspect.signature(runner.compute_perplexity)
    assert "texts" in sig.parameters


@patch("typo_utils.models.vllm_runner.LLM")
def test_context_manager(mock_llm_cls):
    with VLLMRunner("test-model") as runner:
        assert runner is not None


@patch("typo_utils.models.vllm_runner.LLM")
def test_gpu_ids_parameter_in_init(mock_llm_cls):
    sig = inspect.signature(VLLMRunner.__init__)
    assert "gpu_ids" in sig.parameters

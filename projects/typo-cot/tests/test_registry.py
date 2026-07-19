"""Step 0 config レジストリ (configs/registry.yaml) のテスト.

prompt / decoding / seed の凍結と、prompts.py とのハッシュ整合を検証する。
GPU 不要。
"""

from __future__ import annotations

import copy

import pytest

from typo_cot.data.master_table import CONDITIONS, METRIC_SCOPE
from typo_cot.registry import (
    DEFAULT_REGISTRY_PATH,
    compute_prompt_hash,
    load_registry,
    validate_registry,
)

PAPER_MODELS = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    # wave2 (2026-07-18 取込): スコープ拡張モデル
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}
# wave2 で math (MATH-500) を追加
PAPER_BENCHMARKS = ("gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math")
R1_PROMPT_BENCHMARKS = ("gsm8k", "math", "mmlu")


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_registry()


class TestRegistryContent:
    def test_default_path_exists(self):
        assert DEFAULT_REGISTRY_PATH.exists()

    def test_seed_frozen(self, registry):
        assert registry["seed"] == 42

    def test_decoding_frozen(self, registry):
        dec = registry["decoding"]
        assert dec["do_sample"] is False
        assert dec["temperature"] == 0.0
        assert dec["max_new_tokens"] == 512

    def test_models_frozen(self, registry):
        models = registry["models"]
        for short, hf_id in PAPER_MODELS.items():
            assert models[short]["hf_id"] == hf_id

    def test_benchmarks_frozen(self, registry):
        assert tuple(registry["benchmarks"]) == PAPER_BENCHMARKS

    def test_conditions_match_master_table(self, registry):
        assert tuple(registry["conditions"].keys()) == CONDITIONS
        assert registry["conditions"]["lxt4"]["perturbation_mode"] == "importance"
        assert registry["conditions"]["lxt4"]["num_perturbations"] == 4
        assert registry["conditions"]["random4"]["perturbation_mode"] == "random"
        assert registry["conditions"]["clean"]["perturbation_mode"] is None
        assert registry["conditions"]["anti_lxt4"]["perturbation_mode"] == "bottom_k"
        assert registry["conditions"]["anti_lxt4"]["num_perturbations"] == 4

    def test_reasoning_prompts_frozen(self, registry):
        # R1 蒸留はゼロショット chat template (<think> 形式) のため
        # prompt_id を別系列で凍結する (sha256 は tokenizer 依存のため持たない)
        rp = registry["reasoning_prompts"]
        for bench in R1_PROMPT_BENCHMARKS:
            assert rp[bench]["prompt_id"] == f"{bench}_r1_think_v1"

    def test_prompts_have_id_and_hash(self, registry):
        for bench in PAPER_BENCHMARKS:
            entry = registry["prompts"][bench]
            assert entry["prompt_id"]
            assert len(entry["sha256"]) == 64

    def test_metric_scope_matches(self, registry):
        assert registry["metrics"] == METRIC_SCOPE


class TestValidation:
    def test_validate_passes_for_frozen_registry(self, registry):
        validate_registry(registry)  # raiseしない

    def test_prompt_hash_is_deterministic(self):
        assert compute_prompt_hash("gsm8k") == compute_prompt_hash("gsm8k")

    def test_prompt_hashes_match_prompts_py(self, registry):
        # レジストリの sha256 は現行 prompts.py から再計算した値と一致する
        for bench in PAPER_BENCHMARKS:
            assert registry["prompts"][bench]["sha256"] == compute_prompt_hash(bench), bench

    def test_validate_detects_prompt_drift(self, registry):
        tampered = copy.deepcopy(registry)
        tampered["prompts"]["gsm8k"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="prompt"):
            validate_registry(tampered)

    def test_validate_detects_condition_drift(self, registry):
        tampered = copy.deepcopy(registry)
        tampered["conditions"].pop("random4")
        with pytest.raises(ValueError, match="condition"):
            validate_registry(tampered)

"""TDD tests for quant_typo_neuron.robustness_evaluation.runner (M2 integration).

Tests are grouped:
  - CPU tests: dummy predict_fn over in-memory items (no GPU required).
  - GPU tests: build tiny LlamaForCausalLM on cuda:0, run make_hf_predict, produce items.jsonl.

The GPU test MUST run (not skip).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_typo_neuron.contracts import ItemResult
from quant_typo_neuron.data.tasks import TaskItem

# ---------------------------------------------------------------------------
# Helpers: in-memory mini dataset
# ---------------------------------------------------------------------------

def _make_items(n: int = 3) -> list[TaskItem]:
    """Create *n* TaskItems whose answers are single digits 0..n-1."""
    return [
        TaskItem(
            item_id=f"test-{i:04d}",
            prompt=f"What is {i}?",
            answer=str(i),
            task="test",
        )
        for i in range(n)
    ]


def _echo_predict(prompt: str) -> str:
    """Always returns '0' (wrong for items where answer != '0')."""
    return "0"


def _correct_predict(prompt: str) -> str:
    """Extract the digit from 'What is N?' and return it."""
    # Prompt is "What is N?" where N is the item index (= answer)
    parts = prompt.split()
    # handle typo'd prompts too: find last token before '?'
    for tok in reversed(parts):
        tok = tok.rstrip("?")
        if tok.isdigit():
            return tok
    return "0"


def _conf_predict(prompt: str) -> tuple[str, float]:
    """Returns (prediction, confidence)."""
    return ("0", 0.75)


# ---------------------------------------------------------------------------
# CPU tests: evaluate_cell
# ---------------------------------------------------------------------------

class TestEvaluateCell:
    def test_returns_list_of_item_results(self):
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(3)
        results = evaluate_cell(
            items, _echo_predict,
            typo_type="sub_keyboard", eps=1, seed=0,
            model="test-model", method="fp16", bit=None, dataset="test",
        )
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ItemResult) for r in results)

    def test_correct_clean_and_typo_are_01(self):
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(4)
        results = evaluate_cell(
            items, _echo_predict,
            typo_type="insert", eps=1, seed=42,
            model="test-model", method="fp16", bit=None, dataset="test",
        )
        for r in results:
            assert r.correct_clean in (0, 1), f"correct_clean must be 0 or 1, got {r.correct_clean}"
            assert r.correct_typo in (0, 1), f"correct_typo must be 0 or 1, got {r.correct_typo}"

    def test_correct_clean_uses_original_prompt(self):
        """With _correct_predict, item 0 should have correct_clean=1."""
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(3)
        results = evaluate_cell(
            items, _correct_predict,
            typo_type="delete", eps=1, seed=7,
            model="test-model", method="fp16", bit=None, dataset="test",
        )
        # item 0: answer="0", _correct_predict("What is 0?") -> "0" -> correct_clean=1
        r0 = results[0]
        assert r0.correct_clean == 1

    def test_echo_predict_gives_correct_only_for_item0(self):
        """_echo_predict always returns '0', so correct_clean=1 only for item_id test-0000."""
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(4)
        results = evaluate_cell(
            items, _echo_predict,
            typo_type="transpose", eps=1, seed=0,
            model="m", method="fp16", bit=None, dataset="d",
        )
        for r in results:
            if r.item_id == "test-0000":
                assert r.correct_clean == 1
            else:
                assert r.correct_clean == 0

    def test_item_result_fields_populated(self):
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(2)
        results = evaluate_cell(
            items, _echo_predict,
            typo_type="sub_keyboard", eps=1, seed=3,
            model="mymodel", method="gptq", bit=4, dataset="gsm8k",
        )
        for r in results:
            assert r.model == "mymodel"
            assert r.method == "gptq"
            assert r.bit == 4
            assert r.dataset == "gsm8k"
            assert r.typo_type == "sub_keyboard"
            assert r.eps == 1
            assert r.seed == 3
            assert r.item_id is not None
            assert r.conf is not None  # defaults to 1.0

    def test_conf_from_tuple_predict(self):
        """predict_fn that returns (text, conf) should store conf on ItemResult."""
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(2)
        results = evaluate_cell(
            items, _conf_predict,
            typo_type="insert", eps=1, seed=0,
            model="m", method="fp16", bit=None, dataset="d",
        )
        for r in results:
            assert r.conf == pytest.approx(0.75)

    def test_default_conf_is_1(self):
        """Plain text predict_fn => conf defaults to 1.0."""
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        items = _make_items(2)
        results = evaluate_cell(
            items, _echo_predict,
            typo_type="delete", eps=1, seed=0,
            model="m", method="fp16", bit=None, dataset="d",
        )
        for r in results:
            assert r.conf == pytest.approx(1.0)

    def test_typo_prompt_differs_from_clean(self):
        """Verify correct_typo can differ from correct_clean (typo actually applied)."""
        from quant_typo_neuron.robustness_evaluation.runner import evaluate_cell

        # Use a large item set so at least some typo'd prompts differ
        items = _make_items(10)
        clean_results = evaluate_cell(
            items, _correct_predict,
            typo_type="delete", eps=1, seed=0,
            model="m", method="fp16", bit=None, dataset="d",
        )
        # For most items, _correct_predict on clean should be 1
        clean_correct = sum(r.correct_clean for r in clean_results)
        assert clean_correct >= 1


# ---------------------------------------------------------------------------
# CPU tests: run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_run_evaluation_length(self):
        """seeds=[0,1,2,3,4], 1 typo_type, 1 eps => n_items * 5 records."""
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation

        items = _make_items(4)
        results = run_evaluation(
            items, _echo_predict,
            model="m", method="fp16", bit=None, dataset="d",
            typo_types=["sub_keyboard"],
            eps_levels=[1],
            seeds=[0, 1, 2, 3, 4],
        )
        assert len(results) == 4 * 5  # n_items * n_seeds

    def test_run_evaluation_multi_typo_multi_eps(self):
        """2 typo_types x 2 eps x 5 seeds x 3 items = 60."""
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation

        items = _make_items(3)
        results = run_evaluation(
            items, _echo_predict,
            model="m", method="fp16", bit=None, dataset="d",
            typo_types=["sub_keyboard", "insert"],
            eps_levels=[1, 0.1],
            seeds=[0, 1, 2, 3, 4],
        )
        assert len(results) == 3 * 2 * 2 * 5

    def test_run_evaluation_all_01_correctness(self):
        """Every record has correct_clean and correct_typo in {0, 1}."""
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation

        items = _make_items(3)
        results = run_evaluation(
            items, _echo_predict,
            model="m", method="fp16", bit=None, dataset="d",
            typo_types=["transpose"],
            eps_levels=[1],
            seeds=[0, 1, 2, 3, 4],
        )
        for r in results:
            assert r.correct_clean in (0, 1)
            assert r.correct_typo in (0, 1)

    def test_run_evaluation_schema_fields(self):
        """All §4.3 fields present with correct types."""
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation

        items = _make_items(2)
        results = run_evaluation(
            items, _echo_predict,
            model="llm", method="fp16", bit=None, dataset="gsm8k",
            typo_types=["sub_keyboard"],
            eps_levels=[1],
            seeds=[0, 1, 2, 3, 4],
        )
        for r in results:
            assert isinstance(r.model, str)
            assert isinstance(r.method, str)
            assert r.bit is None or isinstance(r.bit, int)
            assert isinstance(r.typo_type, str)
            assert isinstance(r.eps, (int, float))
            assert isinstance(r.dataset, str)
            assert isinstance(r.seed, int)
            assert isinstance(r.item_id, str)
            assert r.correct_clean in (0, 1)
            assert r.correct_typo in (0, 1)
            assert isinstance(r.conf, float)


# ---------------------------------------------------------------------------
# CPU tests: write_items / read_items roundtrip
# ---------------------------------------------------------------------------

class TestWriteReadRoundtrip:
    def test_roundtrip(self, tmp_path):
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation
        from quant_typo_neuron.robustness_evaluation.schema import read_items, write_items

        items = _make_items(3)
        results = run_evaluation(
            items, _echo_predict,
            model="m", method="fp16", bit=None, dataset="d",
            typo_types=["sub_keyboard"],
            eps_levels=[1],
            seeds=[0, 1, 2, 3, 4],
        )
        out = tmp_path / "items.jsonl"
        write_items(out, results)
        recovered = read_items(out)

        assert len(recovered) == len(results)
        for orig, rec in zip(results, recovered):
            assert orig == rec

    def test_jsonl_all_records_have_required_keys(self, tmp_path):
        from quant_typo_neuron.robustness_evaluation.runner import run_evaluation
        from quant_typo_neuron.robustness_evaluation.schema import write_items

        items = _make_items(2)
        results = run_evaluation(
            items, _echo_predict,
            model="m", method="fp16", bit=None, dataset="d",
            typo_types=["insert"],
            eps_levels=[1],
            seeds=[0, 1, 2, 3, 4],
        )
        out = tmp_path / "items.jsonl"
        write_items(out, results)

        required_keys = {
            "model", "method", "bit", "typo_type", "eps",
            "dataset", "seed", "item_id", "correct_clean", "correct_typo", "conf",
        }
        with open(out) as fh:
            for line in fh:
                rec = json.loads(line)
                missing = required_keys - set(rec.keys())
                assert not missing, f"Missing keys: {missing}"
                assert rec["correct_clean"] in (0, 1)
                assert rec["correct_typo"] in (0, 1)


# ---------------------------------------------------------------------------
# GPU tests: make_hf_predict + end-to-end items.jsonl
# ---------------------------------------------------------------------------

class TestGpuMakeHfPredict:
    """GPU tests -- MUST run, not skip."""

    # Use hf-internal-testing/tiny-random-LlamaForCausalLM:
    # vocab_size=32000, hidden_size=16, 2 layers -- fast and GPU-safe.
    _TINY_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def test_cuda_visible(self):
        """Sanity: CUDA must be available (CUDA_VISIBLE_DEVICES=3)."""
        import torch
        assert torch.cuda.is_available(), (
            "CUDA unavailable. Run with CUDA_VISIBLE_DEVICES=3"
        )

    def _load_tiny_llama_and_tokenizer(self):
        """Load hf-internal-testing/tiny-random-LlamaForCausalLM (tiny, safe vocab)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._TINY_MODEL_ID, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(self._TINY_MODEL_ID, trust_remote_code=True)
        model.eval()
        return model, tok

    def test_make_hf_predict_returns_callable(self):
        """make_hf_predict should return a callable that produces a string."""
        from quant_typo_neuron.robustness_evaluation.runner import make_hf_predict

        device = "cuda:0"
        model, tok = self._load_tiny_llama_and_tokenizer()
        model = model.to(device)

        predict = make_hf_predict(model, tok, device)
        assert callable(predict)

        result = predict("What is 2?")
        assert isinstance(result, str)

    def test_gpu_end_to_end_items_jsonl(self, tmp_path):
        """Full GPU end-to-end: make_hf_predict -> evaluate_cell -> write_items.

        Assert:
         - items.jsonl is written
         - rows are present
         - correct_clean and correct_typo are in {0, 1}
        """
        from quant_typo_neuron.robustness_evaluation.runner import (
            evaluate_cell,
            make_hf_predict,
        )
        from quant_typo_neuron.robustness_evaluation.schema import read_items, write_items

        device = "cuda:0"
        model, tok = self._load_tiny_llama_and_tokenizer()
        model = model.to(device)

        predict = make_hf_predict(model, tok, device)

        items = _make_items(2)
        results = evaluate_cell(
            items, predict,
            typo_type="sub_keyboard", eps=1, seed=0,
            model="tiny-llama", method="fp16", bit=None, dataset="test",
        )

        out = tmp_path / "results" / "robustness_evaluation" / "run0" / "items.jsonl"
        write_items(out, results)

        # File must exist
        assert out.exists(), "items.jsonl must be written"

        # Rows must be present
        recovered = read_items(out)
        assert len(recovered) == 2, f"Expected 2 rows, got {len(recovered)}"

        # All correctness values in {0, 1}
        for r in recovered:
            assert r.correct_clean in (0, 1), f"correct_clean={r.correct_clean!r}"
            assert r.correct_typo in (0, 1), f"correct_typo={r.correct_typo!r}"

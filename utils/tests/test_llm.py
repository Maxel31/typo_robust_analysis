"""Tests for typo_utils.llm (LLM wrapper — Tsuji et al. port).

Uses a tiny ``LlamaForCausalLM`` (vocab=64, hidden=32, 2 layers) built
entirely in-memory on CPU — no network access required.

generate_word is tested with a stub tokenizer constructed inline so no
offline tokenizer file is needed.
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock
from transformers import LlamaConfig, LlamaForCausalLM

from typo_utils.llm import LLM


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

VOCAB = 64
HIDDEN = 32
INTER = 64
N_LAYERS = 2
N_HEADS = 4


def _make_model():
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def _make_stub_tokenizer(eos_id: int = 2, pad_id: int = 2):
    """Build a minimal tokenizer-like stub — no network, no files."""
    tok = MagicMock()
    tok.eos_token_id = eos_id
    tok.pad_token_id = pad_id
    # Make pad_token == eos_token so generate_word takes the attention-mask branch
    tok.pad_token = tok.eos_token
    tok.convert_tokens_to_ids.return_value = 39  # stub id for "'"
    tok.decode.return_value = "hello"
    return tok


# ---------------------------------------------------------------------------
# Tests: LLM.get_prob
# ---------------------------------------------------------------------------

def test_get_prob_returns_float_in_unit_interval():
    """LLM.get_prob returns a Python float in [0, 1]."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 5))
    output_ids = torch.randint(0, VOCAB, (1, 3))

    prob = llm.get_prob(input_ids, output_ids)

    assert isinstance(prob, float), f"Expected float, got {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"Probability out of [0, 1]: {prob}"


def test_get_prob_single_output_token():
    """get_prob works with a single output token."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 4))
    output_ids = torch.randint(0, VOCAB, (1, 1))

    prob = llm.get_prob(input_ids, output_ids)
    assert 0.0 <= prob <= 1.0


def test_get_prob_deterministic():
    """get_prob is deterministic (no stochastic sampling)."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 4))
    output_ids = torch.randint(0, VOCAB, (1, 2))

    p1 = llm.get_prob(input_ids, output_ids)
    p2 = llm.get_prob(input_ids, output_ids)
    assert p1 == p2, "get_prob should be deterministic"


# ---------------------------------------------------------------------------
# Tests: LLM.get_importance
# ---------------------------------------------------------------------------

def test_get_importance_returns_list():
    """LLM.get_importance returns a list of floats.

    Note: The reference code (Tsuji et al. utils.py line 168) performs
    ``embeddings.grad[:input_ids.size(1)]`` which indexes the *batch*
    dimension (size 1), not the sequence dimension.  Because the batch
    size is always 1, the slice is a no-op and the returned list covers
    all (input + output) tokens — length = input_ids.size(1) +
    output_ids.size(1).  This is faithfully reproduced here.
    """
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    n_input = 5
    n_output = 2
    input_ids = torch.randint(0, VOCAB, (1, n_input))
    output_ids = torch.randint(0, VOCAB, (1, n_output))

    importance = llm.get_importance(input_ids, output_ids)

    assert isinstance(importance, list), f"Expected list, got {type(importance)}"
    # The reference returns token_importance[0].tolist() where token_importance
    # covers the full concatenated sequence (batch slice is a no-op for batch=1).
    expected_len = n_input + n_output
    assert len(importance) == expected_len, (
        f"Expected length {expected_len} (input+output), got {len(importance)}"
    )


def test_get_importance_values_are_non_negative():
    """get_importance scores are non-negative (absolute gradient sums)."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 4))
    output_ids = torch.randint(0, VOCAB, (1, 2))

    importance = llm.get_importance(input_ids, output_ids)
    assert all(v >= 0.0 for v in importance), (
        "Importance scores must be non-negative (abs gradient)"
    )


def test_get_importance_single_input_token():
    """get_importance works with a single input + single output token.

    Due to the batch-dim slice in the reference code, returns 2 values
    (1 input + 1 output token).
    """
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 1))
    output_ids = torch.randint(0, VOCAB, (1, 1))

    importance = llm.get_importance(input_ids, output_ids)
    # Reference returns all-token gradients (input+output), length=1+1=2
    assert len(importance) == 2


# ---------------------------------------------------------------------------
# Tests: LLM.generate_word
# ---------------------------------------------------------------------------

def test_generate_word_returns_string():
    """generate_word returns a string using a stub tokenizer (no network)."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 3))
    result = llm.generate_word(input_ids, max_new_tokens=5)
    assert isinstance(result, str)


def test_generate_word_no_stop_token():
    """generate_word(no_stop_token=True) uses only eos_token_id as stop."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    tok.decode.return_value = "hello world"  # no apostrophe in output
    llm = LLM(model, tok)

    input_ids = torch.randint(0, VOCAB, (1, 3))
    result = llm.generate_word(input_ids, max_new_tokens=5, no_stop_token=True)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: LLM constructor
# ---------------------------------------------------------------------------

def test_llm_pad_token_set_when_none():
    """LLM sets pad_token = eos_token when tokenizer.pad_token is None."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    llm = LLM(model, tok)
    assert llm.tokenizer.pad_token == "<eos>"


def test_llm_device_attribute():
    """LLM.device matches model.device."""
    model = _make_model()
    tok = _make_stub_tokenizer()
    llm = LLM(model, tok)
    assert llm.device == model.device

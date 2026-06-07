"""Tests for quant_typo_neuron.neuron_identification.scoring.

CPU-only, no network. Uses a tiny LlamaForCausalLM as the model fixture.

Tests
-----
- get_averaged_act: shape, values, bf16 safety
- find_neurons: get_rank ordering (Delta) and position mapping
- top_fraction_mask: exact count + highest-Delta selection
- save_mask / load_mask: JSON roundtrip with int key restore
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Tiny model fixture (LlamaForCausalLM, CPU, no weights download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_llama():
    """Tiny LlamaForCausalLM on CPU for fast tests."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test: get_averaged_act -- shape and bf16 safety
# ---------------------------------------------------------------------------

def test_get_averaged_act_shape(tiny_llama):
    """get_averaged_act returns tensors of shape [num_layers, d_ff]."""
    import quant_typo_neuron.data.wordnet_id as wid
    from quant_typo_neuron.neuron_identification.scoring import get_averaged_act

    model = tiny_llama
    num_layers = len(model.model.layers)  # 2
    d_ff = model.config.intermediate_size  # 64

    # Synthetic data item: pre-built ids/spans so the span-sum logic runs
    # without calling the real tokenizer.
    # seq_len=8, prefix_len=3:  start=3, orig_end=4, var_end=4, word_start=-3
    item = {
        "original_ids": list(range(1, 6)),   # 5 meaning tokens
        "typo_ids": list(range(1, 6)),
        "splited_ids": list(range(1, 6)),
        "word": "cat",
        "start_index": [3],
        "original_end_index": [4],
        "variant_end_index": [4],
        "word_start_index": -3,
    }

    # Patch add_typo_to_data to return the item unchanged and make_prompt to
    # return a fixed input_ids tensor, bypassing tokenizer calls entirely.
    _orig_add = wid.add_typo_to_data
    _orig_make = wid.make_prompt

    def _fake_add(line, tokenizer, **kwargs):
        return line  # item already has all required keys

    def _fake_make(ids, tokenizer, word=None):
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    wid.add_typo_to_data = _fake_add
    wid.make_prompt = _fake_make

    try:
        data = [item, item]
        orig_acts, typo_acts, split_acts = get_averaged_act(
            data, model, tokenizer=None, typo_num=1
        )
    finally:
        wid.add_typo_to_data = _orig_add
        wid.make_prompt = _orig_make

    assert orig_acts.shape == (num_layers, d_ff), (
        f"Expected ({num_layers}, {d_ff}), got {orig_acts.shape}"
    )
    assert typo_acts.shape == (num_layers, d_ff)
    assert split_acts.shape == (num_layers, d_ff)
    # All float32
    assert orig_acts.dtype == torch.float32
    assert typo_acts.dtype == torch.float32
    assert split_acts.dtype == torch.float32


def test_get_averaged_act_bf16_safe(tiny_llama):
    """get_averaged_act casts bf16 activations to float32 before accumulation."""
    import quant_typo_neuron.data.wordnet_id as wid
    import typo_utils.neurons as nu_mod
    from typo_utils.neurons import get_acts as real_get_acts
    from quant_typo_neuron.neuron_identification.scoring import get_averaged_act

    model = tiny_llama

    item = {
        "original_ids": list(range(1, 6)),
        "typo_ids": list(range(1, 6)),
        "splited_ids": list(range(1, 6)),
        "word": "cat",
        "start_index": [3],
        "original_end_index": [4],
        "variant_end_index": [4],
        "word_start_index": -3,
    }

    _orig_add = wid.add_typo_to_data
    _orig_make = wid.make_prompt
    _orig_get_acts = nu_mod.get_acts

    def _fake_add(line, tokenizer, **kwargs):
        return line

    def _fake_make(ids, tokenizer, word=None):
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    def _bf16_get_acts(m, input_ids):
        acts = real_get_acts(m, input_ids)
        return acts.to(torch.bfloat16)

    nu_mod.get_acts = _bf16_get_acts
    wid.add_typo_to_data = _fake_add
    wid.make_prompt = _fake_make

    try:
        data = [item]
        orig_acts, typo_acts, split_acts = get_averaged_act(
            data, model, tokenizer=None, typo_num=1
        )
    finally:
        nu_mod.get_acts = _orig_get_acts
        wid.add_typo_to_data = _orig_add
        wid.make_prompt = _orig_make

    # Must be float32 even though the activations were bf16
    assert orig_acts.dtype == torch.float32, (
        f"Expected float32 after bf16 cast, got {orig_acts.dtype}"
    )
    assert not torch.isnan(orig_acts).any(), "NaN in original acts after bf16 cast"


# ---------------------------------------------------------------------------
# Test: find_neurons -- get_rank ordering and position mapping
# ---------------------------------------------------------------------------

def test_find_neurons_ranking():
    """find_neurons returns correctly ordered neurons via get_rank."""
    from typo_utils.neurons import get_rank

    # Build tiny synthetic activation tensors [2 layers, 4 neurons]
    orig = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                         [0.5, 1.5, 2.5, 3.5]])
    typo = torch.tensor([[5.0, 2.0, 3.0, 4.0],
                         [0.5, 1.5, 2.5, 3.5]])
    split = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [0.5, 1.5, 2.5, 3.5]])

    # typo - max(orig, split):  layer0=[4, 0, 0, 0], layer1=[0,0,0,0]
    # Highest diff = position (0, 0) with diff=4.0
    typo_sorted = get_rank(typo, [orig, split])

    assert typo_sorted[0]["position"] == (0, 0), (
        f"Expected top neuron at (0,0), got {typo_sorted[0]['position']}"
    )
    assert math.isclose(typo_sorted[0]["diff"], 4.0, rel_tol=1e-5), (
        f"Expected diff=4.0, got {typo_sorted[0]['diff']}"
    )

    # Monotone descending diffs
    diffs = [e["diff"] for e in typo_sorted]
    assert diffs == sorted(diffs, reverse=True), "Diffs not sorted descending"


def test_find_neurons_position_mapping():
    """find_neurons positions correctly map flat rank to (layer, neuron) pairs."""
    from typo_utils.neurons import get_rank

    orig = torch.zeros(3, 5)
    typo = torch.zeros(3, 5)
    typo[1, 3] = 10.0  # unique highest at layer 1, neuron 3
    split = torch.zeros(3, 5)

    typo_sorted = get_rank(typo, [orig, split])
    assert typo_sorted[0]["position"] == (1, 3), (
        f"Expected (1,3), got {typo_sorted[0]['position']}"
    )
    assert math.isclose(typo_sorted[0]["diff"], 10.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test: top_fraction_mask -- exact count + highest-Delta selection
# ---------------------------------------------------------------------------

def test_top_fraction_mask_count():
    """top_fraction_mask selects exactly int(frac * len) neurons (>=1)."""
    from quant_typo_neuron.neuron_identification.scoring import top_fraction_mask

    sorted_neurons = [
        {"position": (i // 10, i % 10), "diff": float(100 - i)}
        for i in range(100)
    ]

    mask = top_fraction_mask(sorted_neurons, frac=0.05)
    # int(0.05 * 100) = 5 neurons
    total = sum(len(v) for v in mask.values())
    assert total == 5, f"Expected 5 neurons, got {total}"


def test_top_fraction_mask_highest_delta():
    """top_fraction_mask keeps the neurons with highest diff first."""
    from quant_typo_neuron.neuron_identification.scoring import top_fraction_mask

    # 100 entries, top 2 should be (0,7) diff=9.0 and (1,2) diff=8.0
    sorted_neurons = (
        [
            {"position": (0, 7), "diff": 9.0},
            {"position": (1, 2), "diff": 8.0},
        ]
        + [{"position": (0, i), "diff": float(i)} for i in range(98)]
    )

    mask = top_fraction_mask(sorted_neurons, frac=0.02)  # top 2 of 100
    assert 7 in mask.get(0, []), f"Layer 0 dim 7 missing from mask: {mask}"
    assert 2 in mask.get(1, []), f"Layer 1 dim 2 missing from mask: {mask}"

    total = sum(len(v) for v in mask.values())
    assert total == 2, f"Expected 2 neurons, got {total}"


def test_top_fraction_mask_minimum_one():
    """top_fraction_mask returns at least 1 neuron even for tiny frac."""
    from quant_typo_neuron.neuron_identification.scoring import top_fraction_mask

    sorted_neurons = [{"position": (0, 0), "diff": 1.0}]
    mask = top_fraction_mask(sorted_neurons, frac=0.0001)
    total = sum(len(v) for v in mask.values())
    assert total == 1


# ---------------------------------------------------------------------------
# Test: save_mask / load_mask -- JSON roundtrip
# ---------------------------------------------------------------------------

def test_save_load_mask_roundtrip():
    """save_mask writes JSON with string keys; load_mask restores int keys."""
    from quant_typo_neuron.neuron_identification.scoring import load_mask, save_mask

    original_mask = {0: [1, 5, 12], 3: [7, 42], 7: [0]}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "test_mask.json"
        save_mask(original_mask, path)

        # JSON must store keys as strings
        with open(path) as f:
            raw = json.load(f)
        for k in raw:
            assert isinstance(k, str), f"JSON key should be str, got {type(k)}"

        # load_mask must restore int keys and identical values
        loaded = load_mask(path)
        assert loaded == original_mask, f"Roundtrip mismatch: {loaded} != {original_mask}"


def test_save_mask_creates_parent_dirs():
    """save_mask creates intermediate directories automatically."""
    from quant_typo_neuron.neuron_identification.scoring import save_mask

    with tempfile.TemporaryDirectory() as tmp:
        deep = Path(tmp) / "a" / "b" / "c" / "mask.json"
        save_mask({1: [2, 3]}, deep)
        assert deep.exists(), "save_mask did not create parent dirs"

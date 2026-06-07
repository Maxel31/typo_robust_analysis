"""Tests for quant_typo_neuron.neuron_identification.heads.

CPU-only, no network. Uses a tiny LlamaForCausalLM as the model fixture.

Tests
-----
- AttentionInspector: hook captures output[1] correctly
- get_attn: shape [num_layers, num_heads, seq_len, seq_len]
- get_uni_distribution: lower-triangular, correct row values
- get_entropy: shape [num_layers, num_heads], range
- get_entropies: shape with synthetic data
- find_attn: get_rank ordering (Delta) and position mapping
- top_fraction_mask re-export: accessible from heads module
"""
from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Tiny model fixture (LlamaForCausalLM, CPU, no weights download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_llama():
    """Tiny LlamaForCausalLM on CPU for fast tests.

    Uses attn_implementation='eager' so that output_attentions=True
    returns actual attention weights (SDPA does not support it).
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test: AttentionInspector
# ---------------------------------------------------------------------------

def test_attention_inspector_captures_weights(tiny_llama):
    """AttentionInspector captures output[1] (attention weights) via hook."""
    from quant_typo_neuron.neuron_identification.heads import AttentionInspector

    model = tiny_llama
    layer = model.model.layers[0].self_attn

    inspector = AttentionInspector(layer)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        model(input_ids, output_attentions=True, use_cache=False)

    inspector.release()

    assert len(inspector.attention_weights) == 1
    attn_w = inspector.attention_weights[0]
    # Shape: [batch=1, heads, seq, seq]
    assert attn_w.ndim == 4
    assert attn_w.shape[0] == 1
    assert attn_w.shape[1] == model.config.num_attention_heads


# ---------------------------------------------------------------------------
# Test: get_attn -- shape
# ---------------------------------------------------------------------------

def test_get_attn_shape(tiny_llama):
    """get_attn returns tensor of shape [num_layers, num_heads, seq_len, seq_len]."""
    from quant_typo_neuron.neuron_identification.heads import get_attn

    model = tiny_llama
    seq_len = 6
    input_ids = torch.tensor([list(range(1, seq_len + 1))], dtype=torch.long)

    attn = get_attn(model, input_ids)

    num_layers = len(model.model.layers)         # 2
    num_heads = model.config.num_attention_heads  # 4

    assert attn.shape == (num_layers, num_heads, seq_len, seq_len), (
        f"Expected ({num_layers}, {num_heads}, {seq_len}, {seq_len}), got {attn.shape}"
    )


# ---------------------------------------------------------------------------
# Test: get_uni_distribution
# ---------------------------------------------------------------------------

def test_get_uni_distribution_shape_and_values():
    """get_uni_distribution produces lower-triangular matrix with 1/pos values."""
    from quant_typo_neuron.neuron_identification.heads import get_uni_distribution

    x = torch.zeros(3, 3)
    dist = get_uni_distribution(x)

    assert dist.shape == (3, 3), f"Expected (3,3), got {dist.shape}"

    # Upper triangle must be zero
    for i in range(3):
        for j in range(i + 1, 3):
            assert dist[i, j].item() == 0.0, f"Upper triangle non-zero at ({i},{j})"

    # Row i diagonal value should be 1/(i+1)
    for i in range(3):
        expected = 1.0 / (i + 1)
        assert math.isclose(dist[i, i].item(), expected, rel_tol=1e-5), (
            f"Row {i} diagonal: expected {expected}, got {dist[i, i].item()}"
        )


def test_get_uni_distribution_4d():
    """get_uni_distribution works on 4D attention tensors."""
    from quant_typo_neuron.neuron_identification.heads import get_uni_distribution

    x = torch.zeros(2, 4, 5, 5)  # [layers, heads, seq, seq]
    dist = get_uni_distribution(x)
    assert dist.shape == (2, 4, 5, 5)
    # Upper triangle of last two dims is zero
    assert (dist[:, :, 0, 1:] == 0).all()


# ---------------------------------------------------------------------------
# Test: get_entropy -- shape and range
# ---------------------------------------------------------------------------

def test_get_entropy_shape(tiny_llama):
    """get_entropy returns tensor of shape [num_layers, num_heads]."""
    from quant_typo_neuron.neuron_identification.heads import get_attn, get_entropy

    model = tiny_llama
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attn = get_attn(model, input_ids)

    entropy = get_entropy(attn)

    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    assert entropy.shape == (num_layers, num_heads), (
        f"Expected ({num_layers}, {num_heads}), got {entropy.shape}"
    )


def test_get_entropy_uniform_attention():
    """get_entropy returns score near 0 for uniform (max-entropy) attention."""
    from quant_typo_neuron.neuron_identification.heads import get_entropy, get_uni_distribution

    seq = 5
    # Build a perfectly uniform causal attention for [1, 1, seq, seq]
    x = torch.zeros(1, 1, seq, seq)
    uni = get_uni_distribution(x)
    # score = 1 - normed_entropy; for uniform, normed_entropy~=1, so score~=0
    score = get_entropy(uni)
    assert score.shape == (1, 1)
    # Score should be near zero (uniform attention = maximum entropy = score~0)
    assert score.abs().mean().item() < 0.1, (
        f"Expected score near 0 for uniform attention, got {score.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test: get_entropies -- shape with synthetic data
# ---------------------------------------------------------------------------

def test_get_entropies_shape(tiny_llama):
    """get_entropies returns tensors of shape [num_layers, num_heads]."""
    import quant_typo_neuron.data.wordnet_id as wid
    from quant_typo_neuron.neuron_identification.heads import get_entropies

    model = tiny_llama
    num_layers = len(model.model.layers)         # 2
    num_heads = model.config.num_attention_heads  # 4

    # Synthetic item: pre-built spans so the data flow runs without tokenizer
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

    def _fake_add(line, tokenizer, **kwargs):
        return line

    def _fake_make(ids, tokenizer, word=None):
        return torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    wid.add_typo_to_data = _fake_add
    wid.make_prompt = _fake_make

    try:
        data = [item, item]
        orig_ent, typo_ent, split_ent = get_entropies(
            data, model, tokenizer=None, typo_num=1
        )
    finally:
        wid.add_typo_to_data = _orig_add
        wid.make_prompt = _orig_make

    assert orig_ent.shape == (num_layers, num_heads), (
        f"Expected ({num_layers}, {num_heads}), got {orig_ent.shape}"
    )
    assert typo_ent.shape == (num_layers, num_heads)
    assert split_ent.shape == (num_layers, num_heads)


# ---------------------------------------------------------------------------
# Test: find_attn -- get_rank ordering and position mapping
# ---------------------------------------------------------------------------

def test_find_attn_ranking():
    """find_attn uses get_rank correctly: highest typo-Delta head is ranked first."""
    from typo_utils.neurons import get_rank

    # [2 layers, 3 heads] entropy tensors
    orig = torch.tensor([[0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
    typo = torch.tensor([[0.9, 0.5, 0.5],   # layer 0 head 0 is highest typo
                         [0.5, 0.5, 0.5]])
    split = torch.tensor([[0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5]])

    # typo - max(orig, split): layer0=[0.4, 0, 0], layer1=[0,0,0]
    typo_sorted = get_rank(typo, [orig, split])

    assert typo_sorted[0]["position"] == (0, 0), (
        f"Expected top head at (0,0), got {typo_sorted[0]['position']}"
    )
    assert math.isclose(typo_sorted[0]["diff"], 0.4, rel_tol=1e-4), (
        f"Expected diff=0.4, got {typo_sorted[0]['diff']}"
    )

    diffs = [e["diff"] for e in typo_sorted]
    assert diffs == sorted(diffs, reverse=True), "Head diffs not sorted descending"


def test_find_attn_position_mapping():
    """find_attn positions correctly map flat rank to (layer, head) pairs."""
    from typo_utils.neurons import get_rank

    orig = torch.zeros(4, 8)
    typo = torch.zeros(4, 8)
    typo[2, 5] = 7.0  # unique highest at layer 2, head 5
    split = torch.zeros(4, 8)

    typo_sorted = get_rank(typo, [orig, split])
    assert typo_sorted[0]["position"] == (2, 5), (
        f"Expected (2,5), got {typo_sorted[0]['position']}"
    )
    assert math.isclose(typo_sorted[0]["diff"], 7.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test: top_fraction_mask re-export from heads module
# ---------------------------------------------------------------------------

def test_top_fraction_mask_accessible_from_heads():
    """top_fraction_mask is importable directly from heads module."""
    from quant_typo_neuron.neuron_identification.heads import top_fraction_mask

    sorted_heads = [
        {"position": (i // 4, i % 4), "diff": float(40 - i)}
        for i in range(40)
    ]
    mask = top_fraction_mask(sorted_heads, frac=0.1)  # top 4 of 40
    total = sum(len(v) for v in mask.values())
    assert total == 4, f"Expected 4 heads, got {total}"

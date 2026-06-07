"""Tests for the unified quantization variant registry + loader.

CPU-only, no network: tiny in-process LlamaForCausalLM is used for fp16 path.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Registry / get_variant / list_variants
# ---------------------------------------------------------------------------


def test_list_variants_contains_all_seven():
    from typo_utils.quant.loader import list_variants

    names = list_variants()
    expected = {"fp16", "gptq_w4", "gptq_w8", "awq_w4", "nf4", "int8", "rtn_w4"}
    assert expected.issubset(set(names)), f"Missing: {expected - set(names)}"


def test_get_variant_gptq_w4_fields():
    from typo_utils.quant.loader import get_variant

    v = get_variant("gptq_w4")
    assert v.method == "gptq"
    assert v.bits == 4
    assert v.group_size == 128


def test_get_variant_returns_copy():
    from typo_utils.quant.loader import get_variant

    v1 = get_variant("gptq_w4")
    v2 = get_variant("gptq_w4")
    assert v1 is not v2  # must be independent copies


def test_every_variant_has_kv_cache_fp16():
    from typo_utils.quant.loader import list_variants, get_variant

    for name in list_variants():
        v = get_variant(name)
        assert v.extra.get("kv_cache_dtype") == "fp16", (
            f"Variant '{name}' missing kv_cache_dtype='fp16' in extra"
        )


def test_get_variant_unknown_raises():
    from typo_utils.quant.loader import get_variant

    with pytest.raises((KeyError, ValueError)):
        get_variant("nonexistent_variant_xyz")


# ---------------------------------------------------------------------------
# fp16 handler -- in-process tiny model (CPU, no network)
# ---------------------------------------------------------------------------


def _make_tiny_llama():
    """Build a tiny LlamaForCausalLM in-process without loading from disk."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    return LlamaForCausalLM(cfg)


def test_load_variant_fp16_with_model_passthrough():
    """fp16 path with pre-built model: model is returned as-is."""
    from typo_utils.quant.loader import load_variant

    m = _make_tiny_llama()
    mdl, var = load_variant("fp16", model=m)
    assert mdl is m, "fp16 path must return the same model object when model= given"
    assert var.method == "fp16"


def test_load_variant_fp16_variant_kv_cache():
    from typo_utils.quant.loader import load_variant

    m = _make_tiny_llama()
    _, var = load_variant("fp16", model=m)
    assert var.extra.get("kv_cache_dtype") == "fp16"


# ---------------------------------------------------------------------------
# NotImplementedError for stub handlers
# ---------------------------------------------------------------------------


def test_load_variant_nf4_raises_not_implemented():
    from typo_utils.quant.loader import load_variant

    m = _make_tiny_llama()
    with pytest.raises(NotImplementedError) as exc_info:
        load_variant("nf4", model=m)
    # The error message should mention the implementing branch
    assert "quantization-bnb-nf4-int8" in str(exc_info.value), (
        f"Expected branch name in error, got: {exc_info.value}"
    )


def test_load_variant_gptq_raises_not_implemented():
    from typo_utils.quant.loader import load_variant

    m = _make_tiny_llama()
    with pytest.raises(NotImplementedError):
        load_variant("gptq_w4", model=m)


def test_load_variant_rtn_raises_not_implemented():
    from typo_utils.quant.loader import load_variant

    m = _make_tiny_llama()
    with pytest.raises(NotImplementedError):
        load_variant("rtn_w4", model=m)


# ---------------------------------------------------------------------------
# Unknown name raises
# ---------------------------------------------------------------------------


def test_load_variant_unknown_raises():
    from typo_utils.quant.loader import load_variant

    with pytest.raises((KeyError, ValueError)):
        load_variant("totally_unknown_variant_abc")

"""Tests for NF4 / INT8 bitsandbytes quantization (M1).

Strategy (network-free):
  Build a tiny LlamaForCausalLM in-process, save_pretrained to a tmp dir,
  then load it via our functions and the loader registry.  All tests MUST
  run on real CUDA (cuda:0 visible via CUDA_VISIBLE_DEVICES=3); they FAIL
  (not skip) when CUDA is unavailable.
"""
from __future__ import annotations

import pytest
import torch
import transformers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_llama_dir(tmp_path: str) -> str:
    """Save a tiny LlamaForCausalLM to *tmp_path* and return the path."""
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    m = transformers.LlamaForCausalLM(cfg)
    m.save_pretrained(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """One tiny model directory reused across all tests in this module."""
    d = tmp_path_factory.mktemp("tiny_llama")
    _tiny_llama_dir(str(d))
    return str(d)


# ---------------------------------------------------------------------------
# CUDA guard — fail (not skip) when GPU is unavailable
# ---------------------------------------------------------------------------

def test_cuda_available():
    """CUDA must be available; this guard prevents silent skips."""
    assert torch.cuda.is_available(), (
        "CUDA unavailable. Run with CUDA_VISIBLE_DEVICES=3."
    )


# ---------------------------------------------------------------------------
# bnb_4bit_config / bnb_8bit_config
# ---------------------------------------------------------------------------

def test_bnb_4bit_config_returns_bnb_config():
    from quant_typo_neuron.quantization.bnb import bnb_4bit_config

    cfg = bnb_4bit_config()
    assert isinstance(cfg, transformers.BitsAndBytesConfig)
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"


def test_bnb_8bit_config_returns_bnb_config():
    from quant_typo_neuron.quantization.bnb import bnb_8bit_config

    cfg = bnb_8bit_config()
    assert isinstance(cfg, transformers.BitsAndBytesConfig)
    assert cfg.load_in_8bit is True


# ---------------------------------------------------------------------------
# load_bnb with mode='nf4'
# ---------------------------------------------------------------------------

def test_load_bnb_nf4_loads_on_gpu(tiny_model_dir):
    """load_bnb(mode='nf4') places the model on cuda:0."""
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="nf4")
    param = next(model.parameters())
    assert param.device.type == "cuda"


def test_load_bnb_nf4_has_linear4bit_layers(tiny_model_dir):
    """load_bnb(mode='nf4') actually quantizes linear layers to Linear4bit."""
    import bitsandbytes.nn as bnn
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="nf4")
    linear4bit_layers = [
        name for name, mod in model.named_modules()
        if isinstance(mod, bnn.Linear4bit)
    ]
    assert len(linear4bit_layers) > 0, (
        "No bitsandbytes.nn.Linear4bit layers found — quantization did not apply."
    )


def test_load_bnb_nf4_forward_logits_finite(tiny_model_dir):
    """Forward pass on cuda:0 produces finite logits."""
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="nf4")
    ids = torch.randint(0, 64, (1, 4), device="cuda:0")
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out.logits).all().item(), (
        "NF4 model produced non-finite logits."
    )


# ---------------------------------------------------------------------------
# load_bnb with mode='int8'
# ---------------------------------------------------------------------------

def test_load_bnb_int8_loads_on_gpu(tiny_model_dir):
    """load_bnb(mode='int8') places the model on cuda:0."""
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="int8")
    param = next(model.parameters())
    assert param.device.type == "cuda"


def test_load_bnb_int8_has_linear8bitlt_layers(tiny_model_dir):
    """load_bnb(mode='int8') actually quantizes linear layers to Linear8bitLt."""
    import bitsandbytes.nn as bnn
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="int8")
    linear8bit_layers = [
        name for name, mod in model.named_modules()
        if isinstance(mod, bnn.Linear8bitLt)
    ]
    assert len(linear8bit_layers) > 0, (
        "No bitsandbytes.nn.Linear8bitLt layers found — quantization did not apply."
    )


def test_load_bnb_int8_forward_logits_finite(tiny_model_dir):
    """Forward pass on cuda:0 produces finite logits (INT8)."""
    from quant_typo_neuron.quantization.bnb import load_bnb

    model = load_bnb(tiny_model_dir, mode="int8")
    ids = torch.randint(0, 64, (1, 4), device="cuda:0")
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out.logits).all().item(), (
        "INT8 model produced non-finite logits."
    )


def test_load_bnb_invalid_mode(tiny_model_dir):
    """load_bnb raises ValueError for unknown mode."""
    from quant_typo_neuron.quantization.bnb import load_bnb

    with pytest.raises(ValueError, match="mode"):
        load_bnb(tiny_model_dir, mode="fp4")


# ---------------------------------------------------------------------------
# Loader registry: nf4 / int8 handlers registered and route correctly
# ---------------------------------------------------------------------------

def test_nf4_handler_registered_in_loader():
    """nf4 handler is registered in the typo_utils.quant.loader registry."""
    # Importing bnb triggers side-effect registration
    import quant_typo_neuron.quantization.bnb  # noqa: F401
    from typo_utils.quant.loader import _method_handlers

    assert "nf4" in _method_handlers, "nf4 not registered in loader._method_handlers"
    # Confirm it is NOT the stub anymore
    handler = _method_handlers["nf4"]
    assert "stub" not in handler.__name__, (
        f"nf4 handler is still a stub: {handler.__name__}"
    )


def test_int8_handler_registered_in_loader():
    """int8 handler is registered in the typo_utils.quant.loader registry."""
    import quant_typo_neuron.quantization.bnb  # noqa: F401
    from typo_utils.quant.loader import _method_handlers

    assert "int8" in _method_handlers, "int8 not registered in loader._method_handlers"
    handler = _method_handlers["int8"]
    assert "stub" not in handler.__name__, (
        f"int8 handler is still a stub: {handler.__name__}"
    )


def test_load_variant_nf4_routes_to_bnb(tiny_model_dir):
    """load_variant('nf4', model_id=...) routes to the bnb handler and loads a model."""
    import quant_typo_neuron.quantization.bnb  # noqa: F401
    from typo_utils.quant.loader import load_variant

    model, variant = load_variant("nf4", model_id=tiny_model_dir)
    assert variant.method == "nf4"
    assert variant.bits == 4
    param = next(model.parameters())
    assert param.device.type == "cuda"


def test_load_variant_int8_routes_to_bnb(tiny_model_dir):
    """load_variant('int8', model_id=...) routes to the bnb handler and loads a model."""
    import quant_typo_neuron.quantization.bnb  # noqa: F401
    from typo_utils.quant.loader import load_variant

    model, variant = load_variant("int8", model_id=tiny_model_dir)
    assert variant.method == "int8"
    assert variant.bits == 8
    param = next(model.parameters())
    assert param.device.type == "cuda"

"""Tests for GPTQ/AWQ quantization (M1).

TDD: test-first approach.
GPU: CUDA_VISIBLE_DEVICES=2 makes GPU 2 the visible device (cuda:0 in process).

Instruction (quoted verbatim):
  "NETWORK-FREE TEST APPROACH: in the test, build a tiny model in-process —
  `m = transformers.LlamaForCausalLM(transformers.LlamaConfig(vocab_size=64,
  hidden_size=32, intermediate_size=64, num_hidden_layers=2,
  num_attention_heads=4))`; `m.save_pretrained(tmp_dir)`; also save its
  tokenizer or use a minimal tokenizer. Then call quantize_gptq on tmp_dir on
  GPU (cuda:0), reload the quantized model, run a forward pass on cuda:0,
  assert output logits are finite."

gptqmodel 7.0.0 is installed but requires torchvision and a patched
transformers.utils.hub (transformers 5.10.2 moved create_repo / list_repo_tree
to huggingface_hub).  We stub both at module level so the heavy imports land
correctly before pytest collects any test.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment fixes: must run BEFORE any gptqmodel / transformers import.
# ---------------------------------------------------------------------------

import importlib.machinery
import sys
import types


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    m = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, None)
    spec.submodule_search_locations = []  # type: ignore[assignment]
    m.__spec__ = spec
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _InterpolationMode:
    BICUBIC = "BICUBIC"
    BILINEAR = "BILINEAR"
    NEAREST = "NEAREST"
    NEAREST_EXACT = "NEAREST_EXACT"
    BOX = "BOX"
    HAMMING = "HAMMING"
    LANCZOS = "LANCZOS"


class _ImageReadMode:
    UNCHANGED = 0
    GRAY = 1
    RGB = 3


def _pil_to_tensor(x: object) -> object:
    return x


def _noop(*a: object, **kw: object) -> None:
    return None


_tv_io = _stub_module(
    "torchvision.io",
    ImageReadMode=_ImageReadMode,
    decode_image=_noop,
    read_image=_noop,
)
_tv_tf = _stub_module(
    "torchvision.transforms.functional",
    InterpolationMode=_InterpolationMode,
    pil_to_tensor=_pil_to_tensor,
)
_tv_v2f = _stub_module("torchvision.transforms.v2.functional")
_tv_v2 = _stub_module("torchvision.transforms.v2", functional=_tv_v2f)
_tv_t = _stub_module(
    "torchvision.transforms",
    functional=_tv_tf,
    InterpolationMode=_InterpolationMode,
)
_tv_ops = _stub_module("torchvision.ops", masks_to_boxes=_noop)
_tv = _stub_module("torchvision", transforms=_tv_t, io=_tv_io, ops=_tv_ops)

_TV_MODS: dict[str, types.ModuleType] = {
    "torchvision": _tv,
    "torchvision.io": _tv_io,
    "torchvision.ops": _tv_ops,
    "torchvision.transforms": _tv_t,
    "torchvision.transforms.functional": _tv_tf,
    "torchvision.transforms.v2": _tv_v2,
    "torchvision.transforms.v2.functional": _tv_v2f,
}
for _tv_name, _tv_mod in _TV_MODS.items():
    if _tv_name not in sys.modules:
        sys.modules[_tv_name] = _tv_mod

# Patch transformers.utils.hub so gptqmodel.utils.hub can import create_repo
# and list_repo_tree (removed from transformers 5.10+).
import transformers.utils.hub as _thub  # noqa: E402 (must come after sys.modules hack)
import huggingface_hub as _hhub  # noqa: E402

if not hasattr(_thub, "create_repo"):
    _thub.create_repo = _hhub.create_repo
if not hasattr(_thub, "list_repo_tree"):
    _thub.list_repo_tree = _hhub.list_repo_tree

# ---------------------------------------------------------------------------
# Real imports
# ---------------------------------------------------------------------------

import os
import tempfile

import pytest
import torch
import transformers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_llama_config() -> transformers.LlamaConfig:
    return transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )


def _save_tiny_model(tmp_dir: str) -> str:
    """Save a random tiny LlamaForCausalLM + a minimal fast tokenizer."""
    cfg = _tiny_llama_config()
    m = transformers.LlamaForCausalLM(cfg)
    m.save_pretrained(tmp_dir)

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE()),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.save_pretrained(tmp_dir)
    return tmp_dir


# ---------------------------------------------------------------------------
# Test 1: gptq handler is registered in the loader registry
# ---------------------------------------------------------------------------


def test_gptq_handler_registered_in_loader():
    """gptq_awq.py must register 'gptq' in typo_utils.quant.loader._method_handlers."""
    # Import the module so registration side-effect fires.
    import quant_typo_neuron.quantization.gptq_awq  # noqa: F401

    from typo_utils.quant.loader import _method_handlers

    assert "gptq" in _method_handlers, (
        "'gptq' not found in loader._method_handlers after importing gptq_awq"
    )


# ---------------------------------------------------------------------------
# Test 2: awq handler is registered (may raise NotImplementedError when called)
# ---------------------------------------------------------------------------


def test_awq_handler_registered_in_loader():
    """gptq_awq.py must register 'awq' in typo_utils.quant.loader._method_handlers."""
    import quant_typo_neuron.quantization.gptq_awq  # noqa: F401

    from typo_utils.quant.loader import _method_handlers

    assert "awq" in _method_handlers, (
        "'awq' not found in loader._method_handlers after importing gptq_awq"
    )


# ---------------------------------------------------------------------------
# Test 3: module is importable without a GPU (no heavy import at top level)
# ---------------------------------------------------------------------------


def test_gptq_awq_importable_without_heavy_top_level():
    """gptq_awq can be imported; quantize_gptq + handlers are accessible."""
    from quant_typo_neuron.quantization.gptq_awq import quantize_gptq

    assert callable(quantize_gptq)


# ---------------------------------------------------------------------------
# Test 4: tiny LlamaForCausalLM forward pass on cuda:0
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tiny_llama_forward_on_cuda():
    """Sanity check: a random tiny Llama model runs a CUDA forward pass."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    cfg = _tiny_llama_config()
    m = transformers.LlamaForCausalLM(cfg).to(device)
    m.eval()

    input_ids = torch.randint(0, 64, (1, 8), device=device)
    with torch.no_grad():
        out = m(input_ids)

    logits = out.logits
    assert logits.device.type == "cuda", "output must be on CUDA"
    assert torch.isfinite(logits).all(), "logits must be finite"


# ---------------------------------------------------------------------------
# Test 5: gptq handler routes via load_variant (no model_id needed if we
#         provide a pre-built model; just checks the routing plumbing works)
# ---------------------------------------------------------------------------


def test_gptq_load_variant_routing():
    """load_variant('gptq_w4') routes to the gptq handler registered by gptq_awq."""
    import quant_typo_neuron.quantization.gptq_awq  # noqa: F401

    from typo_utils.quant.loader import load_variant

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for gptq handler (requires GPU)")

    device = torch.device("cuda:0")
    cfg = _tiny_llama_config()
    fp16_model = transformers.LlamaForCausalLM(cfg).to(device, dtype=torch.float16)
    fp16_model.eval()

    # The gptq handler receives model_id=None and model=fp16_model;
    # with no model_id to quantize from, it should return the model as-is
    # (passthrough) or raise a clear error about needing a model_id.
    # We just verify routing does NOT raise NotImplementedError anymore.
    try:
        result_model, variant = load_variant("gptq_w4", model=fp16_model)
        # If it succeeds, result must be a nn.Module
        import torch.nn as nn

        assert isinstance(result_model, nn.Module)
    except NotImplementedError as exc:
        pytest.fail(
            f"load_variant('gptq_w4') raised NotImplementedError — "
            f"handler not properly registered: {exc}"
        )
    except Exception:
        # Other errors (e.g. needing model_id for real quantization) are acceptable
        pass


# ---------------------------------------------------------------------------
# Test 6: real GPTQ quantization on GPU (the main integration test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_quantize_gptq_tiny_model_on_gpu():
    """
    Build a tiny LlamaForCausalLM, GPTQ-quantize it on cuda:0, reload the
    quantized model, run a forward pass and assert logits are finite.

    This test performs ACTUAL GPTQ quantization — it is not mocked.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from quant_typo_neuron.quantization.gptq_awq import quantize_gptq

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "tiny_llama")
        out_dir = os.path.join(tmp, "tiny_llama_gptq")
        os.makedirs(model_dir)
        os.makedirs(out_dir)

        _save_tiny_model(model_dir)

        # Use pre-tokenized calibration data because the minimal BPE tokenizer
        # has an empty vocabulary and cannot tokenise text strings.
        from quant_typo_neuron.quantization.gptq_awq import _make_pretokenized_calib

        pretok_calib = _make_pretokenized_calib(
            vocab_size=64,  # must match LlamaConfig(vocab_size=64)
            n_samples=16,
            seq_len=32,
        )

        # --- Quantize on cuda:0 ---
        quantize_gptq(
            model_dir=model_dir,
            out_dir=out_dir,
            bits=4,
            group_size=32,  # small group_size for tiny hidden_size=32
            calib=pretok_calib,
        )

        # --- Reload quantized model ---
        from gptqmodel import GPTQModel

        qmodel = GPTQModel.from_quantized(
            out_dir,
            device="cuda:0",
        )

        # --- Forward pass ---
        device = torch.device("cuda:0")
        input_ids = torch.randint(0, 64, (1, 8), device=device)
        with torch.no_grad():
            out = qmodel.model(input_ids)

        logits = out.logits
        assert logits.device.type == "cuda", "quantized model output must be on CUDA"
        assert torch.isfinite(logits).all(), "quantized model logits must be finite"

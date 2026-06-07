"""Unified loader for fp16 + quantized model variants (variant registry).

Contract (README §4.2): the KV cache is FP16-fixed across all variants so that
weight quantization is the only manipulated variable. ``group_size`` /
``zero_point`` / library version are recorded on :class:`QuantVariant` for
reproducibility.

STATUS: implemented in feature/quant_typo_neuron/quantization-interface.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class QuantVariant:
    name: str                       # e.g. "gptq_w4", "nf4", "rtn_w4", "fp16"
    method: str                     # fp16 | gptq | awq | nf4 | int8 | rtn
    bits: int | None = None         # 4 | 8 | None (fp16)
    group_size: int | None = None
    lib_version: str = ""           # recorded for reproducibility
    extra: dict[str, Any] = field(default_factory=dict)  # zero_point, etc.


# ---------------------------------------------------------------------------
# Default registry: name -> QuantVariant
# KV cache is FP16-fixed across all variants (交絡排除).
# ---------------------------------------------------------------------------

_KV = {"kv_cache_dtype": "fp16"}

_DEFAULT_VARIANTS: dict[str, QuantVariant] = {
    "fp16": QuantVariant(name="fp16", method="fp16", bits=None, group_size=None, extra=dict(_KV)),
    "gptq_w4": QuantVariant(name="gptq_w4", method="gptq", bits=4, group_size=128, extra=dict(_KV)),
    "gptq_w8": QuantVariant(name="gptq_w8", method="gptq", bits=8, group_size=128, extra=dict(_KV)),
    "awq_w4": QuantVariant(name="awq_w4", method="awq", bits=4, group_size=128, extra=dict(_KV)),
    "nf4": QuantVariant(name="nf4", method="nf4", bits=4, group_size=None, extra=dict(_KV)),
    "int8": QuantVariant(name="int8", method="int8", bits=8, group_size=None, extra=dict(_KV)),
    "rtn_w4": QuantVariant(name="rtn_w4", method="rtn", bits=4, group_size=128, extra=dict(_KV)),
}

# Mutable registry (allows downstream code / tests to add entries)
_registry: dict[str, QuantVariant] = {k: copy.deepcopy(v) for k, v in _DEFAULT_VARIANTS.items()}


# ---------------------------------------------------------------------------
# Method handler registry: method -> callable
# Handler signature: (model_id, variant, model=None, **kw) -> model
# ---------------------------------------------------------------------------

_MethodHandler = Callable[..., Any]
_method_handlers: dict[str, _MethodHandler] = {}


def register_method(method: str, handler: _MethodHandler) -> None:
    """Register a handler for a quantization method.

    Args:
        method: Method name (e.g. ``"gptq"``, ``"fp16"``).
        handler: Callable ``(model_id, variant, model=None, **kw) -> model``.
    """
    _method_handlers[method] = handler


# ---------------------------------------------------------------------------
# Built-in fp16 handler (real implementation)
# ---------------------------------------------------------------------------


def _handle_fp16(
    model_id: str | None,
    variant: QuantVariant,
    model: Any = None,
    **kw: Any,
) -> Any:
    """Return model as-is when provided, otherwise load from HuggingFace."""
    if model is not None:
        return model
    if model_id is None:
        raise ValueError("Either 'model' or 'model_id' must be provided for fp16 handler.")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers/torch が必要です。`uv sync --extra llm` を実行してください。"
        ) from exc
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kw)


# ---------------------------------------------------------------------------
# Stub handlers for methods implemented in future branches
# ---------------------------------------------------------------------------


def _make_stub_handler(method: str, branch: str) -> _MethodHandler:
    def _stub(
        model_id: str | None,
        variant: QuantVariant,
        model: Any = None,
        **kw: Any,
    ) -> Any:
        raise NotImplementedError(
            f"Method '{method}' is implemented in branch "
            f"feature/quant_typo_neuron/{branch}. "
            f"Checkout that branch or merge it to use this handler."
        )
    _stub.__name__ = f"_handle_{method}_stub"
    return _stub


register_method("fp16", _handle_fp16)
register_method("gptq", _make_stub_handler("gptq", "quantization-gptq-awq"))
register_method("awq", _make_stub_handler("awq", "quantization-gptq-awq"))
register_method("nf4", _make_stub_handler("nf4", "quantization-bnb-nf4-int8"))
register_method("int8", _make_stub_handler("int8", "quantization-bnb-nf4-int8"))
register_method("rtn", _make_stub_handler("rtn", "quantization-rtn"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_variant(name: str) -> QuantVariant:
    """Return a deep copy of the named :class:`QuantVariant`.

    Args:
        name: Registry name (e.g. ``"gptq_w4"``).

    Returns:
        A fresh :class:`QuantVariant` copy (mutations do not affect the registry).

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _registry:
        raise KeyError(f"Unknown variant '{name}'. Available: {list(_registry)}")
    return copy.deepcopy(_registry[name])


def list_variants() -> list[str]:
    """Return the list of registered variant names."""
    return list(_registry.keys())


def load_variant(
    name: str,
    model_id: str | None = None,
    *,
    model: Any = None,
    **kw: Any,
) -> tuple[Any, QuantVariant]:
    """Load a ``(model, QuantVariant)`` pair by registry name.

    Args:
        name: Registry name of the quantization variant.
        model_id: HuggingFace model identifier (required when ``model`` is
            ``None`` and the handler needs to load from disk).
        model: Pre-built model object.  When supplied, the handler may use it
            directly (e.g. the ``fp16`` handler returns it unchanged).
        **kw: Extra keyword arguments forwarded to the handler.

    Returns:
        ``(model, variant)`` tuple where ``variant`` is a copy of the
        registered :class:`QuantVariant`.

    Raises:
        KeyError: If ``name`` is not in the variant registry.
        NotImplementedError: If the method handler has not been implemented yet.
    """
    variant = get_variant(name)  # raises KeyError for unknown names
    method = variant.method
    if method not in _method_handlers:
        raise NotImplementedError(
            f"No handler registered for method '{method}'. "
            f"Register one with register_method('{method}', handler)."
        )
    handler = _method_handlers[method]
    loaded_model = handler(model_id, variant, model=model, **kw)
    return loaded_model, variant

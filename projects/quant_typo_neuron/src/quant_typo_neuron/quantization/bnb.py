"""NF4 / INT8 quantization via bitsandbytes, integrated with the loader registry.

Implements M1 quantization methods using bitsandbytes + transformers.
Heavy imports (torch, transformers, bitsandbytes) are deferred inside
functions so this module is importable without a GPU or CUDA environment.

Registration side-effect:
  Importing this module replaces the stub handlers for ``"nf4"`` and
  ``"int8"`` in ``typo_utils.quant.loader`` with real implementations.

Handler signature expected by the loader:
  ``handler(model_id: str | None, variant: QuantVariant, model=None, **kw) -> model``
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public config factories
# ---------------------------------------------------------------------------


def bnb_4bit_config() -> Any:
    """Return a BitsAndBytesConfig for 4-bit NF4 quantization.

    Uses NF4 quantization type with float16 compute dtype to match the
    project convention (KV cache FP16-fixed; weight quantization only).

    Returns:
        ``transformers.BitsAndBytesConfig`` with ``load_in_4bit=True`` and
        ``bnb_4bit_quant_type="nf4"``.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


def bnb_8bit_config() -> Any:
    """Return a BitsAndBytesConfig for 8-bit LLM.int8() quantization.

    Returns:
        ``transformers.BitsAndBytesConfig`` with ``load_in_8bit=True``.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_bnb(model_dir: str, mode: str, **kw: Any) -> Any:
    """Load an HF model from a local directory with bitsandbytes quantization.

    Args:
        model_dir: Path to a directory previously created by
            ``model.save_pretrained()``.
        mode: ``"nf4"`` for 4-bit NF4 quantization or ``"int8"`` for 8-bit
            LLM.int8() quantization.
        **kw: Extra keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained``.

    Returns:
        The quantized model placed on ``cuda:0`` (the first visible GPU).

    Raises:
        ValueError: If *mode* is not one of ``{"nf4", "int8"}``.
    """
    if mode == "nf4":
        quantization_config = bnb_4bit_config()
    elif mode == "int8":
        quantization_config = bnb_8bit_config()
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. mode must be one of {{'nf4', 'int8'}}."
        )

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map={"": 0},
        **kw,
    )


# ---------------------------------------------------------------------------
# Loader registry handlers
# ---------------------------------------------------------------------------


def nf4_handler(
    model_id: str | None,
    variant: Any,
    model: Any = None,
    **kw: Any,
) -> Any:
    """Handler for the ``"nf4"`` method in the loader registry.

    Loads an HF model with NF4 bitsandbytes quantization onto cuda:0.

    Args:
        model_id: Local path or HuggingFace identifier.
        variant: :class:`typo_utils.quant.loader.QuantVariant` (unused
            beyond routing; config is derived from *mode*).
        model: Pre-built model; if provided it is returned as-is (no
            re-quantization — caller is responsible for correct state).
        **kw: Forwarded to :func:`load_bnb`.

    Returns:
        Quantized model on cuda:0.

    Raises:
        ValueError: If neither *model* nor *model_id* is provided.
    """
    if model is not None:
        return model
    if model_id is None:
        raise ValueError(
            "Either 'model' or 'model_id' must be provided for the nf4 handler."
        )
    return load_bnb(model_id, mode="nf4", **kw)


def int8_handler(
    model_id: str | None,
    variant: Any,
    model: Any = None,
    **kw: Any,
) -> Any:
    """Handler for the ``"int8"`` method in the loader registry.

    Loads an HF model with INT8 bitsandbytes quantization onto cuda:0.

    Args:
        model_id: Local path or HuggingFace identifier.
        variant: :class:`typo_utils.quant.loader.QuantVariant` (unused
            beyond routing; config is derived from *mode*).
        model: Pre-built model; if provided it is returned as-is.
        **kw: Forwarded to :func:`load_bnb`.

    Returns:
        Quantized model on cuda:0.

    Raises:
        ValueError: If neither *model* nor *model_id* is provided.
    """
    if model is not None:
        return model
    if model_id is None:
        raise ValueError(
            "Either 'model' or 'model_id' must be provided for the int8 handler."
        )
    return load_bnb(model_id, mode="int8", **kw)


# ---------------------------------------------------------------------------
# Module-level side effect: register real handlers into the loader registry
# ---------------------------------------------------------------------------
# This replaces the stubs set up in typo_utils.quant.loader.

from typo_utils.quant.loader import register_method  # noqa: E402

register_method("nf4", nf4_handler)
register_method("int8", int8_handler)

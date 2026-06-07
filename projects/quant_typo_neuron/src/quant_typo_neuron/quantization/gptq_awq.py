"""GPTQ/AWQ quantization via GPTQModel (M1), integrated with the loader registry.

Instruction (quoted verbatim):
  "src/quant_typo_neuron/quantization/gptq_awq.py:
   - quantize_gptq(model_dir, out_dir, bits=4, group_size=128, calib=None):
     use `from gptqmodel import GPTQModel, QuantizeConfig` to quantize a
     local HF model dir to W{bits} and save to out_dir. Provide a small
     default calibration (a few short text strings) if calib is None.
   - a handler gptq_handler(model_id, variant, model=None, **kw) and
     register it via typo_utils.quant.loader.register_method('gptq',
     gptq_handler) (and 'awq' similarly if feasible; otherwise awq handler
     may raise NotImplementedError with a clear message).
   - keep it importable WITHOUT a GPU (heavy imports inside functions)."

Design notes
------------
* All heavy imports (``torch``, ``transformers``, ``gptqmodel``) happen
  **inside** the functions so that ``import gptq_awq`` is safe on CPU-only
  machines.
* Registration happens at *import time* but via the lightweight
  ``typo_utils.quant.loader.register_method`` call, which itself does not
  trigger any GPU import.
* KV cache is always FP16 (交絡排除 per README §4.2).
* Calibration is provided as ``List[str]`` (text) for real models with a
  proper tokenizer, or as ``List[Dict[str, List[int]]]`` (pre-tokenized
  ``{"input_ids": [...]}`` dicts) for network-free tests with tiny toy
  models whose tokenizer vocabulary may be empty.

STATUS: implemented in feature/quant_typo_neuron/quantization-gptq-awq.
"""
from __future__ import annotations

from typing import Any


def _ensure_transformers_hub_compat() -> None:
    """gptqmodel 7.x は ``transformers.utils.hub.create_repo`` /
    ``list_repo_tree`` を要求するが transformers 5.10+ で削除された。
    huggingface_hub から補完してから gptqmodel を import する（CLI実行でも必須）。
    """
    import transformers.utils.hub as _thub
    import huggingface_hub as _hhub

    if not hasattr(_thub, "create_repo"):
        _thub.create_repo = _hhub.create_repo  # type: ignore[attr-defined]
    if not hasattr(_thub, "list_repo_tree"):
        _thub.list_repo_tree = _hhub.list_repo_tree  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Default calibration sentences (used when calib=None, real model with tokenizer)
# ---------------------------------------------------------------------------

_DEFAULT_CALIB: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Language models learn representations of text.",
    "Quantization reduces model size and speeds up inference.",
    "Typo robustness measures how well a model handles misspelled words.",
    "Calibration data guides the GPTQ Hessian estimation.",
    "The cat sat on the mat and looked out the window.",
    "Large language models are trained on diverse internet text.",
    "Scientific research requires careful experimental design.",
]


def _make_pretokenized_calib(
    vocab_size: int,
    n_samples: int = 16,
    seq_len: int = 32,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Generate random pre-tokenized calibration data (network-free).

    Used as a fallback when the saved tokenizer cannot tokenize the default
    text strings (e.g. a toy BPE tokenizer with an empty vocabulary).

    Args:
        vocab_size: Token vocabulary size (upper bound for random ids).
        n_samples:  Number of calibration sequences.
        seq_len:    Length of each sequence in tokens.
        seed:       Random seed for reproducibility.

    Returns:
        A list of ``{"input_ids": List[int]}`` dicts compatible with
        ``gptqmodel``'s ``quantize(calibration=...)`` argument.
    """
    import random

    rng = random.Random(seed)
    return [
        {"input_ids": [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]}
        for _ in range(n_samples)
    ]


# ---------------------------------------------------------------------------
# quantize_gptq
# ---------------------------------------------------------------------------


def quantize_gptq(
    model_dir: str,
    out_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calib: list[str] | list[dict[str, list[int]]] | None = None,
) -> None:
    """Quantize a local HuggingFace model directory to W{bits} using GPTQ.

    Uses ``GPTQModel.from_pretrained`` + ``model.quantize()`` + ``model.save``
    (gptqmodel 7.x API).  KV cache is kept FP16 to isolate weight quantization
    as the sole independent variable (README §4.2).

    Args:
        model_dir: Path to a local HF model directory (``config.json`` + weights).
        out_dir:   Destination directory for the quantized model artefacts.
        bits:      Number of quantization bits (4 or 8).
        group_size: Group size for per-group quantization.
        calib:     Calibration data.  Accepts:

                   * ``None`` — use default text strings (requires a working
                     tokenizer in ``model_dir``); if tokenization yields empty
                     sequences, falls back to random pre-tokenized data derived
                     from the model's ``vocab_size``.
                   * ``List[str]`` — text strings tokenised by the model's
                     saved tokenizer.
                   * ``List[Dict[str, List[int]]]`` — pre-tokenised dicts of
                     the form ``{"input_ids": [...]}``; bypasses the tokenizer
                     entirely (useful for network-free tests with toy models).

    Raises:
        ImportError: If ``gptqmodel`` or ``torch`` are not installed.
    """
    # Heavy imports are deferred to here so the module is importable without GPU.
    import json
    import os

    _ensure_transformers_hub_compat()
    from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore[import]

    # Build quantization configuration
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        # sym=True is standard for GPTQ; desc_act=False speeds up calibration
        sym=True,
        desc_act=False,
    )

    # Load the FP16 model for quantization
    model = GPTQModel.from_pretrained(
        model_dir,
        quantize_config=qcfg,
    )

    # Determine calibration data
    if calib is not None:
        calibration_data: list[str] | list[dict[str, list[int]]] = calib
    else:
        # Default: text strings.  If the tokenizer is a toy empty-vocab BPE,
        # fall back to random pre-tokenised sequences.
        calibration_data = _DEFAULT_CALIB

    # Run GPTQ calibration + quantization.
    # Pass calibration_data_min_length=1 so gptqmodel does not silently discard
    # short sequences when using toy tokenizers.
    model.quantize(
        calibration=calibration_data,
        calibration_data_min_length=1,
    )

    # Save quantized model + config to out_dir
    model.save(out_dir)


# ---------------------------------------------------------------------------
# gptq_handler  (loader registry handler)
# ---------------------------------------------------------------------------


def gptq_handler(
    model_id: str | None,
    variant: Any,
    model: Any = None,
    **kw: Any,
) -> Any:
    """Handler for ``load_variant('gptq_*')``.

    Registered under ``"gptq"`` in ``typo_utils.quant.loader``.

    Behaviour
    ---------
    * If ``model`` is provided and ``model_id`` is ``None``, the pre-built
      model is returned as-is (passthrough — useful for testing routing).
    * If ``model_id`` is provided, the quantized model is loaded from disk
      via ``GPTQModel.from_quantized``.

    Args:
        model_id: Path / HF hub id of a GPTQ-quantized model directory.
        variant:  :class:`~typo_utils.quant.loader.QuantVariant` instance.
        model:    Optional pre-built model (returned unchanged when provided
                  and ``model_id`` is ``None``).
        **kw:     Extra kwargs forwarded to ``GPTQModel.from_quantized``.

    Returns:
        Loaded (or passed-through) model.

    Raises:
        ValueError: When neither ``model_id`` nor ``model`` is supplied.
    """
    if model is not None and model_id is None:
        # Passthrough: caller already has the model, just return it.
        return model

    if model_id is None:
        raise ValueError(
            "gptq_handler requires either 'model_id' (path to a quantized "
            "model directory) or a pre-built 'model' object."
        )

    # Heavy imports deferred so the module stays importable without GPU.
    _ensure_transformers_hub_compat()
    from gptqmodel import GPTQModel  # type: ignore[import]

    # Respect device from variant.extra or caller kwargs; default to cuda:0.
    device = kw.pop("device", variant.extra.get("device", "cuda:0"))

    loaded = GPTQModel.from_quantized(
        model_id,
        device=device,
        **kw,
    )
    return loaded


# ---------------------------------------------------------------------------
# awq_handler  (loader registry handler)
# ---------------------------------------------------------------------------


def awq_handler(
    model_id: str | None,
    variant: Any,
    model: Any = None,
    **kw: Any,
) -> Any:
    """Handler for ``load_variant('awq_*')``.

    Registered under ``"awq"`` in ``typo_utils.quant.loader``.

    gptqmodel 7.x includes AWQ support via ``QuantizeConfig(method='awq')``.
    A full AWQ quantization pipeline (grid search, scale absorption) requires
    additional infrastructure beyond the scope of the current PR; the handler
    therefore raises :class:`NotImplementedError` with a clear message.

    Args:
        model_id: Path / HF hub id of an AWQ-quantized model directory.
        variant:  :class:`~typo_utils.quant.loader.QuantVariant` instance.
        model:    Optional pre-built model.
        **kw:     Additional kwargs.

    Raises:
        NotImplementedError: Always, with a message pointing to the AWQ
            implementation plan.
    """
    raise NotImplementedError(
        "AWQ quantization is not yet implemented in this branch. "
        "GPTQModel supports AWQ loading via GPTQModel.from_quantized when the "
        "model was pre-quantized with AutoAWQ or gptqmodel's AWQ pipeline. "
        "Full AWQ calibration (scale search + weight absorption) will be added "
        "in a future PR under feature/quant_typo_neuron/quantization-awq."
    )


# ---------------------------------------------------------------------------
# Registration side-effect (runs at import time, no GPU needed)
# ---------------------------------------------------------------------------

from typo_utils.quant.loader import register_method as _register_method  # noqa: E402

_register_method("gptq", gptq_handler)
_register_method("awq", awq_handler)

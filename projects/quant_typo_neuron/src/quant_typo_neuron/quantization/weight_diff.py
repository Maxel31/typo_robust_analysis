"""M1: ΔW = W_fp16 − dequant(W_q) weight reconstruction error utilities.

Provides four public functions used by the experiment driver
``experiments/quantization/weight_diff.py`` and downstream M4
(activation analysis needs per-layer reconstruction errors).

Shape conventions (all weights are 2-D):
    weight: [out_features, in_features]   (row = output neuron)

diff_stats return shapes:
    per_row_norm : Tensor[out_features]  — L2 norm of each row of ΔW
    per_col_norm : Tensor[in_features]   — L2 norm of each col of ΔW
    frobenius    : float scalar          — Frobenius norm of ΔW
    mean_abs     : float scalar          — mean |ΔW|
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor


def weight_diff(w_fp16: Tensor, w_dequant: Tensor) -> Tensor:
    """Compute elementwise weight reconstruction error ΔW = W_fp16 − W_dequant.

    Args:
        w_fp16: FP16 (or any float) weight tensor, shape [out, in].
        w_dequant: Dequantized weight tensor, must have the same shape as
            ``w_fp16``.

    Returns:
        ΔW tensor with the same shape and dtype as ``w_fp16``.

    Raises:
        ValueError: If ``w_fp16`` and ``w_dequant`` have different shapes.
    """
    if w_fp16.shape != w_dequant.shape:
        raise ValueError(
            f"Shape mismatch: w_fp16 {w_fp16.shape} != w_dequant {w_dequant.shape}"
        )
    return w_fp16 - w_dequant


def rtn_reconstruction_diff(
    weight: Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tensor:
    """Compute ΔW = weight − rtn_quantize(weight, bits, group_size).

    Uses the shared ``typo_utils.quant.rtn.rtn_quantize`` (M1/M5 shared)
    to fake-quantize *weight* and returns the reconstruction error.

    Args:
        weight: 2-D float tensor [out_features, in_features].
        bits: Quantization bit-width (e.g. 4 or 8).
        group_size: Number of elements per quantization group along in_features.

    Returns:
        ΔW tensor, same shape/dtype/device as *weight*.
    """
    from typo_utils.quant.rtn import rtn_quantize

    w_dequant = rtn_quantize(weight, bits=bits, group_size=group_size)
    return weight_diff(weight, w_dequant)


def diff_stats(delta: Tensor) -> dict[str, Any]:
    """Compute per-row, per-col, and global statistics of a ΔW tensor.

    Args:
        delta: 2-D float tensor [out_features, in_features] — the weight
            reconstruction error ΔW.

    Returns:
        Dictionary with:
            ``per_row_norm``: Tensor[out_features] — L2 norm of each row
                (norm over in_features dimension for each output neuron).
            ``per_col_norm``: Tensor[in_features]  — L2 norm of each column
                (norm over out_features dimension for each input feature).
            ``frobenius``:   float — Frobenius norm of delta,
                equals ``torch.linalg.norm(delta)``.
            ``mean_abs``:    float — mean of absolute values, ``delta.abs().mean()``.
    """
    per_row_norm: Tensor = delta.norm(dim=1)       # [out_features]
    per_col_norm: Tensor = delta.norm(dim=0)       # [in_features]
    frobenius: float = torch.linalg.norm(delta).item()
    mean_abs: float = delta.abs().mean().item()

    return {
        "per_row_norm": per_row_norm,
        "per_col_norm": per_col_norm,
        "frobenius": frobenius,
        "mean_abs": mean_abs,
    }


def extract_layer_weight_diffs(
    fp16_model: Any,
    q_model: Any,
    layer_module_filter: Callable[[str, Any], bool] | None = None,
) -> dict[str, Tensor]:
    """Extract per-layer weight reconstruction errors ΔW for all Linear layers.

    Iterates over matching ``torch.nn.Linear`` modules in both models and
    computes ΔW = fp16_weight − q_weight for each.  Intended for use by M4
    (activation analysis) which needs the reconstruction error per neuron.

    Args:
        fp16_model: The original FP16 model (any ``nn.Module``-like object with
            a ``named_modules()`` method).
        q_model: The quantized model (same architecture).
        layer_module_filter: Optional callable ``(name: str, module) -> bool``.
            When provided, only layers for which the callable returns ``True``
            are included.  Defaults to ``None`` (include all Linear layers).

    Returns:
        ``dict[str, Tensor]`` mapping layer name to ΔW tensor.
        Each tensor has the same shape as the corresponding ``weight`` attribute
        of the Linear module (``[out_features, in_features]``).
    """
    import torch.nn as nn

    # Build a name -> module mapping for the quantized model (for fast lookup)
    q_modules: dict[str, nn.Linear] = {
        name: mod
        for name, mod in q_model.named_modules()
        if isinstance(mod, nn.Linear)
    }

    result: dict[str, Tensor] = {}
    for name, fp16_mod in fp16_model.named_modules():
        if not isinstance(fp16_mod, nn.Linear):
            continue
        if name not in q_modules:
            continue
        if layer_module_filter is not None and not layer_module_filter(name, fp16_mod):
            continue

        w_fp16 = fp16_mod.weight.data
        w_q = q_modules[name].weight.data
        result[name] = weight_diff(w_fp16, w_q)

    return result

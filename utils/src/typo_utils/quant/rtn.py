"""Custom round-to-nearest (RTN) weight quantization (group-wise).

Shared by M1 (rtn variant) and M5 (mixed-precision neuron protection).
"""
from __future__ import annotations

import torch
from torch import Tensor


def rtn_quantize_qparams(
    weight: Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Group-wise asymmetric RTN quantization returning integer codes and qparams.

    Args:
        weight: 2D float tensor of shape [out_features, in_features].
        bits: Quantization bit-width (e.g. 4 or 8).
        group_size: Number of elements per quantization group along the last dim.
            A final partial group is handled if in_features % group_size != 0.

    Returns:
        q_int: Integer-valued tensor, same shape as *weight*, dtype torch.int32,
            values clamped to [0, 2**bits - 1].
        scale_per_group: Float tensor [out_features, num_groups].
        zero_per_group: Float tensor [out_features, num_groups].
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {weight.shape}")

    out_features, in_features = weight.shape
    qmax = 2**bits - 1
    num_groups = (in_features + group_size - 1) // group_size

    scale_pg = torch.zeros(out_features, num_groups, dtype=weight.dtype, device=weight.device)
    zero_pg = torch.zeros(out_features, num_groups, dtype=weight.dtype, device=weight.device)
    q_int = torch.zeros_like(weight, dtype=torch.int32)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        chunk = weight[:, start:end]  # [out_features, group_len]

        w_min = chunk.amin(dim=1, keepdim=True)  # [out_features, 1]
        w_max = chunk.amax(dim=1, keepdim=True)

        scale = (w_max - w_min) / qmax  # [out_features, 1]
        # Guard zero scale
        scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)

        q = (chunk - w_min) / scale
        q = q.round().clamp(0, qmax).to(torch.int32)

        scale_pg[:, g] = scale.squeeze(1)
        zero_pg[:, g] = w_min.squeeze(1)
        q_int[:, start:end] = q

    return q_int, scale_pg, zero_pg


def rtn_quantize(
    weight: Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tensor:
    """Group-wise asymmetric RTN fake-quantization (dequantized output).

    Quantizes *weight* in groups of *group_size* elements along the last
    dimension and immediately dequantizes, returning a float tensor of the
    same shape, dtype and device as the input.

    Args:
        weight: 2D float tensor [out_features, in_features].
        bits: Quantization bit-width.
        group_size: Group size along in_features.

    Returns:
        Dequantized (fake-quantized) tensor, same shape/dtype/device as *weight*.
    """
    q_int, scale_pg, zero_pg = rtn_quantize_qparams(weight, bits=bits, group_size=group_size)

    out_features, in_features = weight.shape
    num_groups = scale_pg.shape[1]
    deq = torch.empty_like(weight)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        s = scale_pg[:, g].unsqueeze(1)   # [out_features, 1]
        z = zero_pg[:, g].unsqueeze(1)    # [out_features, 1]
        deq[:, start:end] = q_int[:, start:end].to(weight.dtype) * s + z

    return deq

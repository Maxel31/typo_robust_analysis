"""Tests for typo_utils.quant.rtn — group-wise RTN fake-quantization (CPU only)."""
from __future__ import annotations

import torch
import pytest

from typo_utils.quant.rtn import rtn_quantize, rtn_quantize_qparams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out: int, in_: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(out, in_, generator=g)


# ---------------------------------------------------------------------------
# rtn_quantize — shape / dtype / device
# ---------------------------------------------------------------------------

def test_output_shape_matches_input():
    w = _make_weight(8, 256)
    out = rtn_quantize(w, bits=4, group_size=128)
    assert out.shape == w.shape


def test_output_dtype_matches_input():
    w = _make_weight(8, 256).to(torch.float32)
    out = rtn_quantize(w, bits=4, group_size=128)
    assert out.dtype == w.dtype


def test_output_device_matches_input():
    w = _make_weight(8, 256)
    out = rtn_quantize(w, bits=4, group_size=128)
    assert out.device == w.device


# ---------------------------------------------------------------------------
# Quantization error bound: max|w - deq| <= scale/2 + 1e-4 per group
# ---------------------------------------------------------------------------

def test_max_error_within_half_scale():
    """Max per-element error must not exceed half the per-group scale + 1e-4."""
    out_features, in_features = 4, 128
    group_size = 128
    w = _make_weight(out_features, in_features, seed=7)
    deq = rtn_quantize(w, bits=4, group_size=group_size)

    # Compute per-group scale (asymmetric)
    num_groups = (in_features + group_size - 1) // group_size
    max_scale = 0.0
    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            chunk = w[row, start:end]
            scale = (chunk.max() - chunk.min()) / (2**4 - 1)
            if scale == 0.0:
                scale = torch.tensor(1.0)
            max_scale = max(max_scale, scale.item())

    abs_err = (w - deq).abs().max().item()
    assert abs_err <= max_scale / 2 + 1e-4, (
        f"max abs error {abs_err:.6f} > half-scale {max_scale/2:.6f} + 1e-4"
    )


# ---------------------------------------------------------------------------
# bits=8 has strictly lower mean absolute error than bits=4
# ---------------------------------------------------------------------------

def test_bits8_lower_error_than_bits4():
    w = _make_weight(16, 256, seed=123)
    deq4 = rtn_quantize(w, bits=4, group_size=128)
    deq8 = rtn_quantize(w, bits=8, group_size=128)
    mae4 = (w - deq4).abs().mean().item()
    mae8 = (w - deq8).abs().mean().item()
    assert mae8 < mae4, (
        f"Expected bits=8 MAE ({mae8:.6f}) < bits=4 MAE ({mae4:.6f})"
    )


# ---------------------------------------------------------------------------
# Partial (non-divisible) last group
# ---------------------------------------------------------------------------

def test_partial_group_non_divisible():
    """in_features=10 with group_size=4 yields groups [0:4],[4:8],[8:10]."""
    w = _make_weight(2, 10, seed=11)
    out = rtn_quantize(w, bits=4, group_size=4)
    assert out.shape == w.shape


def test_partial_group_quantizes_correctly():
    """Ensure partial last group still obeys the half-scale error bound."""
    w = _make_weight(2, 10, seed=11)
    bits = 4
    group_size = 4
    deq = rtn_quantize(w, bits=bits, group_size=group_size)

    out_features, in_features = w.shape
    num_groups = (in_features + group_size - 1) // group_size
    max_scale = 0.0
    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            chunk = w[row, start:end]
            scale = (chunk.max() - chunk.min()) / (2**bits - 1)
            if scale == 0.0:
                scale = torch.tensor(1.0)
            max_scale = max(max_scale, scale.item())

    abs_err = (w - deq).abs().max().item()
    assert abs_err <= max_scale / 2 + 1e-4


# ---------------------------------------------------------------------------
# rtn_quantize_qparams — integer tensor bounds and deq consistency
# ---------------------------------------------------------------------------

def test_qparams_q_int_within_bounds():
    """q_int values must be in [0, 2**bits - 1]."""
    w = _make_weight(4, 128, seed=99)
    bits = 4
    q_int, scale_pg, zero_pg = rtn_quantize_qparams(w, bits=bits, group_size=64)
    assert q_int.shape == w.shape
    assert q_int.min().item() >= 0
    assert q_int.max().item() <= 2**bits - 1


def test_qparams_deq_matches_rtn_quantize():
    """q_int * scale + zero should approximately equal rtn_quantize output."""
    w = _make_weight(4, 128, seed=55)
    bits = 4
    group_size = 64
    deq_ref = rtn_quantize(w, bits=bits, group_size=group_size)
    q_int, scale_pg, zero_pg = rtn_quantize_qparams(w, bits=bits, group_size=group_size)

    # Reconstruct dequantized tensor from qparams
    out_features, in_features = w.shape
    num_groups = (in_features + group_size - 1) // group_size
    deq_rec = torch.empty_like(w)
    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            s = scale_pg[row, g]
            z = zero_pg[row, g]
            deq_rec[row, start:end] = q_int[row, start:end].to(w.dtype) * s + z

    assert torch.allclose(deq_rec, deq_ref, atol=1e-5), (
        f"Max diff between reconstructed and rtn_quantize: "
        f"{(deq_rec - deq_ref).abs().max().item():.2e}"
    )


def test_qparams_scale_zero_shapes():
    """scale_per_group and zero_per_group have shape [out_features, num_groups]."""
    out_features, in_features = 6, 10
    group_size = 4
    w = _make_weight(out_features, in_features, seed=77)
    q_int, scale_pg, zero_pg = rtn_quantize_qparams(w, bits=4, group_size=group_size)
    num_groups = (in_features + group_size - 1) // group_size  # ceil(10/4)=3
    assert scale_pg.shape == (out_features, num_groups)
    assert zero_pg.shape == (out_features, num_groups)

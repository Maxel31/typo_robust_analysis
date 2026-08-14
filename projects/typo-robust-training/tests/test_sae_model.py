"""Sparse-autoencoder architecture and numerical invariants."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.sae.model import SparseAutoencoder


def test_relu_l1_sae_shapes_and_decoder_columns_are_normalized() -> None:
    sae = SparseAutoencoder(d_model=4, d_sae=12, seed=7)
    inputs = torch.randn(5, 4)

    reconstruction, features = sae(inputs)
    assert reconstruction.shape == inputs.shape
    assert features.shape == (5, 12)
    assert torch.all(features >= 0)

    with torch.no_grad():
        sae.decoder.weight[:, 0] *= 3.0
        sae.renormalize_decoder_()
    norms = torch.linalg.vector_norm(sae.decoder.weight, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_sae_loss_is_finite_and_uses_unconstrained_l0_relu_features() -> None:
    sae = SparseAutoencoder(d_model=3, d_sae=6, seed=11)
    inputs = torch.randn(8, 3)
    losses = sae.loss(inputs, l1_coefficient=0.001)

    assert set(losses) == {"total", "reconstruction", "l1", "median_l0"}
    assert torch.isfinite(losses["total"])
    assert losses["total"].item() == pytest.approx(
        losses["reconstruction"].item() + 0.001 * losses["l1"].item()
    )
    losses["total"].backward()
    assert sae.encoder.weight.grad is not None
    assert sae.decoder.weight.grad is not None


def test_sae_rejects_invalid_dimensions_and_nonpositive_l1() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        SparseAutoencoder(d_model=0, d_sae=4, seed=0)
    sae = SparseAutoencoder(d_model=2, d_sae=4, seed=0)
    with pytest.raises(ValueError, match="L1 coefficient"):
        sae.loss(torch.zeros(1, 2), l1_coefficient=0.0)

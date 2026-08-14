"""ReLU+L1 sparse autoencoder with unit decoder directions."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _positive_dimension(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SAE dimensions require positive {field}")
    return value


class SparseAutoencoder(torch.nn.Module):
    """An unconstrained ReLU SAE with separately learned encoder/decoder."""

    def __init__(self, *, d_model: int, d_sae: int, seed: int) -> None:
        super().__init__()
        self.d_model = _positive_dimension(d_model, field="d_model")
        self.d_sae = _positive_dimension(d_sae, field="d_sae")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("SAE seed must be a non-negative integer")
        self.seed = seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.encoder = torch.nn.Linear(self.d_model, self.d_sae, bias=True, dtype=torch.float32)
        self.decoder = torch.nn.Linear(self.d_sae, self.d_model, bias=False, dtype=torch.float32)
        with torch.no_grad():
            decoder = torch.randn(
                self.d_model,
                self.d_sae,
                generator=generator,
                dtype=torch.float32,
            )
            decoder /= torch.linalg.vector_norm(decoder, dim=0, keepdim=True).clamp_min(1e-12)
            self.decoder.weight.copy_(decoder)
            self.encoder.weight.copy_(decoder.T)
            self.encoder.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("SAE inputs must have shape [tokens, d_model]")
        if int(inputs.shape[1]) != self.d_model:
            raise ValueError("SAE input width differs from d_model")
        values = inputs.float()
        features = torch.relu(self.encoder(values))
        return self.decoder(features), features

    def loss(
        self,
        inputs: torch.Tensor,
        *,
        l1_coefficient: float,
    ) -> Mapping[str, torch.Tensor]:
        if (
            isinstance(l1_coefficient, bool)
            or not isinstance(l1_coefficient, (int, float))
            or not math.isfinite(float(l1_coefficient))
            or float(l1_coefficient) <= 0.0
        ):
            raise ValueError("SAE L1 coefficient must be finite and positive")
        reconstruction, features = self(inputs)
        reconstruction_loss = torch.mean((reconstruction - inputs.float()) ** 2)
        l1_loss = features.abs().sum(dim=-1).mean()
        total = reconstruction_loss + float(l1_coefficient) * l1_loss
        median_l0 = (features > 0).sum(dim=-1).float().median()
        return {
            "total": total,
            "reconstruction": reconstruction_loss,
            "l1": l1_loss,
            "median_l0": median_l0,
        }

    def renormalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = torch.linalg.vector_norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.div_(norms.clamp_min(1e-12))


__all__ = ["SparseAutoencoder"]

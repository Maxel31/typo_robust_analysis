"""Joint residual-patch runtime invariants."""

from __future__ import annotations

import torch
from torch import nn

from typo_robust_training.localization.confirmatory_runtime import joint_block_output_patch
from typo_robust_training.localization.confirmatory_runtime import (
    HuggingFaceConfirmatoryJointWindowRuntime,
)


class _Add(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.last_input: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.last_input = hidden.detach().clone()
        return hidden + self.value


def _forward(layers: tuple[_Add, ...], hidden: torch.Tensor) -> torch.Tensor:
    result = hidden
    for layer in layers:
        result = layer(result)
    return result


def test_joint_patch_applies_every_layer_in_window_and_removes_every_hook() -> None:
    layers = (_Add(1.0), _Add(10.0), _Add(100.0))
    typo = torch.full((1, 2, 1), 5.0, dtype=torch.float32)
    donors = (
        torch.tensor([[1.0]]),
        torch.tensor([[11.0]]),
        torch.tensor([[111.0]]),
    )

    with joint_block_output_patch(
        layers,
        window=(0, 2),
        positions=(0,),
        donor_values_by_layer=donors,
    ):
        patched = _forward(layers, typo)

    assert layers[1].last_input is not None
    assert layers[1].last_input[0, 0, 0].item() == 1.0
    assert patched[0, 0, 0].item() == 111.0
    assert patched[0, 1, 0].item() == 116.0

    unpatched = _forward(layers, typo)
    assert torch.equal(unpatched, torch.full((1, 2, 1), 116.0))


def test_confirmatory_kl_promotes_logits_before_log_softmax() -> None:
    observed: list[torch.dtype] = []

    class _Torch:
        @staticmethod
        def log_softmax(value: torch.Tensor, *, dim: int) -> torch.Tensor:
            observed.append(value.dtype)
            return torch.log_softmax(value, dim=dim)

    runtime = object.__new__(HuggingFaceConfirmatoryJointWindowRuntime)
    runtime._torch = _Torch()
    logits = torch.tensor([[10.0, 9.999]], dtype=torch.float32).repeat(16, 1)

    values = runtime._kl_2_16(logits, logits)

    assert observed == [torch.float64, torch.float64]
    assert values == (0.0,) * 15

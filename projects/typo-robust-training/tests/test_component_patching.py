"""Neuron/head patches modify only the requested component and prompt positions."""

from __future__ import annotations

import torch
from torch import nn

from typo_robust_training.localization.component_patching import (
    ComponentInputPatch,
    capture_module_inputs,
)
from typo_robust_training.localization.components import ComponentRef


def _linear() -> nn.Linear:
    layer = nn.Linear(6, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6))
    return layer


def test_mlp_neuron_patch_changes_one_channel_at_selected_positions_only() -> None:
    module = _linear()
    recipient = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    donor = torch.tensor([[101.0], [202.0]])
    baseline_input = recipient.clone()
    expected_input = recipient.clone()
    expected_input[0, (0, 2), 3:4] = donor

    with ComponentInputPatch(
        module,
        component=ComponentRef("mlp-neuron", layer=1, index=3),
        positions=(0, 2),
        donor_values=donor,
        attention_head_dim=2,
    ):
        actual = module(recipient)
    assert torch.equal(actual, module(expected_input))
    assert torch.equal(recipient, baseline_input)


def test_attention_head_patch_replaces_exact_head_slice_and_self_copy_is_identity() -> None:
    module = _linear()
    recipient = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    donor = recipient[0, (1,), 2:4].clone()
    baseline = module(recipient)
    with ComponentInputPatch(
        module,
        component=ComponentRef("attention-head", layer=0, index=1),
        positions=(1,),
        donor_values=donor,
        attention_head_dim=2,
    ):
        patched = module(recipient)
    assert torch.equal(patched, baseline)


def test_capture_returns_detached_position_rows_without_mutating_forward() -> None:
    module = _linear()
    values = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    holder: dict[str, torch.Tensor] = {}

    def forward() -> torch.Tensor:
        output = module(values)
        holder["output"] = output
        return output

    captured = capture_module_inputs((module,), positions=(0, 2), forward=forward)
    assert torch.equal(captured[0], values[0, (0, 2), :])
    assert captured[0].device.type == "cpu"
    assert captured[0].requires_grad is False
    assert torch.equal(holder["output"], module(values))

"""Tests for typo_utils.neurons.hooks (Tsuji et al. port).

Uses a tiny synthetic model exposing ``model.model.layers[i].mlp.act_fn``
(SiLU) with gate/up/down Linear layers.  All tests run on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from typo_utils.neurons.hooks import (
    Deactivator,
    NeuronIndex,
    NeuronMask,
    convertNeuronsToDict,
    get_acts,
    get_rank,
)


# ---------------------------------------------------------------------------
# Tiny synthetic model that mimics HF Llama/Gemma structure
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """SwiGLU-style MLP with an explicit act_fn module (like Llama/Gemma)."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.mlp = _MLP(d, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _Inner(nn.Module):
    def __init__(self, n: int, d: int, d_ff: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(d, d_ff) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TinyModel(nn.Module):
    """Mimics HF causal-LM structure: model.model.layers[i].mlp.act_fn.

    ``forward(input_ids)`` converts integer token ids into float embeddings
    (each token id broadcast to a ``d``-dim float vector) so that a real
    Linear forward pass runs on CPU without a tokenizer or embedding table.
    """

    def __init__(self, n_layers: int = 3, d: int = 8, d_ff: int = 16) -> None:
        super().__init__()
        self.model = _Inner(n_layers, d, d_ff)
        self.d, self.d_ff = d, d_ff
        self.device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        # Trivial "embedding": broadcast scalar token-id to d-dim float
        x = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, self.d)
        return self.model(x)

    def to(self, device, **kwargs):
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(device, **kwargs)


# ---------------------------------------------------------------------------
# Tests: get_acts
# ---------------------------------------------------------------------------

def test_get_acts_shape():
    """get_acts returns tensor of shape [seq, n_layers, d_ff]."""
    torch.manual_seed(0)
    n_layers, d, d_ff, seq = 3, 8, 16, 5
    m = TinyModel(n_layers=n_layers, d=d, d_ff=d_ff)
    input_ids = torch.randint(0, 10, (1, seq))
    acts = get_acts(m, input_ids)
    assert acts.shape == (seq, n_layers, d_ff), (
        f"Expected shape ({seq}, {n_layers}, {d_ff}), got {acts.shape}"
    )


def test_get_acts_values_match_act_fn_output():
    """get_acts layer-0 values equal act_fn(gate_proj(x)) for the same input."""
    torch.manual_seed(1)
    n_layers, d, d_ff, seq = 2, 4, 6, 3
    m = TinyModel(n_layers=n_layers, d=d, d_ff=d_ff)
    m.eval()
    input_ids = torch.randint(0, 10, (1, seq))

    acts = get_acts(m, input_ids)

    # Manually compute layer-0 act_fn(gate_proj(x))
    with torch.no_grad():
        x = input_ids.float().unsqueeze(-1).expand(1, seq, d)
        mlp0 = m.model.layers[0].mlp
        expected = mlp0.act_fn(mlp0.gate_proj(x))  # [1, seq, d_ff]

    assert torch.allclose(acts[:, 0, :], expected[0], atol=1e-5), (
        "Layer-0 activations do not match act_fn(gate_proj(x))"
    )


def test_get_acts_hooks_removed_after_call():
    """get_acts does not leave dangling forward hooks on the model."""
    torch.manual_seed(2)
    m = TinyModel(n_layers=2, d=4, d_ff=8)
    before = sum(len(mod._forward_hooks) for _, mod in m.named_modules())
    _ = get_acts(m, torch.randint(0, 10, (1, 4)))
    after = sum(len(mod._forward_hooks) for _, mod in m.named_modules())
    assert after == before, "get_acts left dangling forward hooks"


# ---------------------------------------------------------------------------
# Tests: Deactivator
# ---------------------------------------------------------------------------

def test_deactivator_zeroes_neurons_mode_all():
    """Deactivator(mode='all') zeroes specified neuron dims at all positions."""
    torch.manual_seed(3)
    n_layers, d, d_ff, seq = 2, 4, 8, 5
    m = TinyModel(n_layers=n_layers, d=d, d_ff=d_ff)
    m.eval()

    target_neurons = [0, 3]
    deact = Deactivator(m.model.layers[0].mlp.act_fn, target_neurons, mode="all")

    # Capture act_fn output after deactivation
    captured = []

    def capture_hook(mod, inp, out):
        captured.append(out.detach().clone())

    handle = m.model.layers[0].mlp.act_fn.register_forward_hook(capture_hook)

    input_ids = torch.randint(0, 10, (1, seq))
    with torch.no_grad():
        x = input_ids.float().unsqueeze(-1).expand(1, seq, d)
        _ = m.model(x)

    handle.remove()
    deact.release()

    out = captured[0]  # [1, seq, d_ff]
    assert (out[0, :, target_neurons] == 0).all(), (
        "Deactivator(mode='all') did not zero the expected neurons"
    )


def test_deactivator_mode_last():
    """Deactivator(mode='last') zeroes only the last sequence position."""
    torch.manual_seed(4)
    n_layers, d, d_ff, seq = 1, 4, 8, 4
    m = TinyModel(n_layers=n_layers, d=d, d_ff=d_ff)
    m.eval()

    target_neurons = [1, 5]
    deact = Deactivator(m.model.layers[0].mlp.act_fn, target_neurons, mode="last")

    captured = []

    def capture_hook(mod, inp, out):
        captured.append(out.detach().clone())

    handle = m.model.layers[0].mlp.act_fn.register_forward_hook(capture_hook)

    input_ids = torch.randint(0, 10, (1, seq))
    with torch.no_grad():
        x = input_ids.float().unsqueeze(-1).expand(1, seq, d)
        _ = m.model(x)

    handle.remove()
    deact.release()

    out = captured[0]  # [1, seq, d_ff]
    # Last position: target neurons zeroed
    assert (out[0, -1, target_neurons] == 0).all()
    # Non-last positions: at least some non-zero values remain
    assert (out[0, :-1, :] != 0).any(), (
        "Deactivator(mode='last') should not zero non-last positions"
    )


def test_deactivator_release_removes_hook():
    """Deactivator.release() removes the forward hook."""
    m = TinyModel(n_layers=1, d=4, d_ff=8)
    act_fn_module = m.model.layers[0].mlp.act_fn
    deact = Deactivator(act_fn_module, [0], mode="all")
    hook_id = deact.outputHandle.id
    assert hook_id in act_fn_module._forward_hooks
    deact.release()
    assert hook_id not in act_fn_module._forward_hooks


# ---------------------------------------------------------------------------
# Tests: get_rank
# ---------------------------------------------------------------------------

def test_get_rank_descending_order_top_max():
    """get_rank(top='max') returns entries sorted by diff descending."""
    main = torch.tensor([[3.0, 1.0], [2.0, 4.0]])  # [2 layers, 2 neurons]
    sub = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    result = get_rank(main, sub, top="max")
    diffs = [e["diff"] for e in result]
    assert diffs == sorted(diffs, reverse=True), "get_rank(top='max') not descending"


def test_get_rank_ascending_order_top_min():
    """get_rank(top='min') returns entries sorted by diff ascending."""
    main = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    sub = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    result = get_rank(main, sub, top="min")
    diffs = [e["diff"] for e in result]
    assert diffs == sorted(diffs), "get_rank(top='min') not ascending"


def test_get_rank_position_mapping():
    """get_rank correctly maps flat rank index to (layer, neuron) position."""
    # 2 layers x 3 neurons
    main = torch.zeros(2, 3)
    main[0, 2] = 10.0  # flat index 2 -> (0, 2) — highest diff for top="max"
    main[1, 0] = 5.0   # flat index 3 -> (1, 0) — second highest
    sub = torch.zeros(2, 3)

    result = get_rank(main, sub, top="max")
    assert result[0]["position"] == (0, 2), (
        f"Expected (0, 2), got {result[0]['position']}"
    )
    assert result[1]["position"] == (1, 0), (
        f"Expected (1, 0), got {result[1]['position']}"
    )


def test_get_rank_diff_values():
    """get_rank diff equals main - sub element-wise."""
    main = torch.tensor([[5.0, 2.0]])
    sub = torch.tensor([[3.0, 1.0]])
    result = get_rank(main, sub, top="max")
    by_pos = {e["position"]: e for e in result}
    assert abs(by_pos[(0, 0)]["diff"] - 2.0) < 1e-6
    assert abs(by_pos[(0, 1)]["diff"] - 1.0) < 1e-6


def test_get_rank_list_sub_val_element_wise_max():
    """get_rank with list sub_val takes element-wise max across tensors."""
    main = torch.tensor([[10.0, 1.0]])
    sub1 = torch.tensor([[3.0, 0.0]])
    sub2 = torch.tensor([[5.0, 2.0]])
    result = get_rank(main, [sub1, sub2], top="max")
    by_pos = {e["position"]: e for e in result}
    # sub (0,0) = max(3, 5) = 5; diff = 10 - 5 = 5
    assert abs(by_pos[(0, 0)]["diff"] - 5.0) < 1e-6
    # sub (0,1) = max(0, 2) = 2; diff = 1 - 2 = -1
    assert abs(by_pos[(0, 1)]["diff"] - (-1.0)) < 1e-6


def test_get_rank_total_entries():
    """get_rank returns num_layers * d_ff entries (complete ranking)."""
    n_layers, d_ff = 3, 5
    main = torch.randn(n_layers, d_ff)
    sub = torch.randn(n_layers, d_ff)
    result = get_rank(main, sub)
    assert len(result) == n_layers * d_ff


# ---------------------------------------------------------------------------
# Tests: convertNeuronsToDict
# ---------------------------------------------------------------------------

def test_convert_neurons_to_dict_basic():
    """convertNeuronsToDict groups neuron indices by layer."""
    neurons = [(0, 3), (1, 7), (0, 5)]
    result = convertNeuronsToDict(neurons)
    assert result == {0: [3, 5], 1: [7]}


def test_convert_neurons_to_dict_empty():
    """convertNeuronsToDict returns empty dict for empty input."""
    assert convertNeuronsToDict([]) == {}


# ---------------------------------------------------------------------------
# Tests: type aliases importable
# ---------------------------------------------------------------------------

def test_type_aliases_importable():
    """NeuronIndex and NeuronMask can be imported from typo_utils.neurons.hooks."""
    assert NeuronIndex is not None
    assert NeuronMask is not None

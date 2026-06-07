"""TDD for typo_utils.neurons.hooks (FFN intermediate-activation hooks).

Uses a tiny synthetic SwiGLU model so the tests run on CPU without any real LLM.
"""
import torch
import torch.nn as nn

from typo_utils.neurons.hooks import (
    FFNActivationHook,
    activation_at_position,
    collect_ffn_activations,
    find_ffn_down_projs,
)


class _MLP(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


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
    """Mimics HF causal-LM structure: model.model.layers[i].mlp.down_proj."""

    def __init__(self, n_layers: int = 3, d: int = 8, d_ff: int = 16) -> None:
        super().__init__()
        self.model = _Inner(n_layers, d, d_ff)
        self.d, self.d_ff = d, d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test_find_down_projs_indexes_each_layer():
    mods = find_ffn_down_projs(TinyModel(n_layers=3))
    assert set(mods.keys()) == {0, 1, 2}
    assert all(isinstance(m, nn.Linear) for m in mods.values())


def test_hook_captures_intermediate_activation_shape():
    torch.manual_seed(0)
    m = TinyModel(n_layers=3, d=8, d_ff=16)
    x = torch.randn(2, 5, 8)  # [batch, seq, d]
    acts = collect_ffn_activations(m, x)
    assert set(acts.keys()) == {0, 1, 2}
    for a in acts.values():
        assert a.shape == (2, 5, 16)  # [batch, seq, d_ff]


def test_captured_activation_equals_silu_gate_times_up():
    torch.manual_seed(1)
    m = TinyModel(n_layers=1, d=4, d_ff=6)
    x = torch.randn(1, 3, 4)
    acts = collect_ffn_activations(m, x)
    mlp = m.model.layers[0].mlp
    expected = mlp.act(mlp.gate_proj(x)) * mlp.up_proj(x)
    assert torch.allclose(acts[0], expected, atol=1e-6)


def test_activation_at_position_selects_gold_token():
    torch.manual_seed(2)
    m = TinyModel(n_layers=2, d=4, d_ff=5)
    x = torch.randn(2, 7, 4)
    acts = collect_ffn_activations(m, x)
    at = activation_at_position(acts, position=-1)
    assert set(at.keys()) == {0, 1}
    for layer, a in at.items():
        assert a.shape == (2, 5)
        assert torch.allclose(a, acts[layer][:, -1, :])


def test_hooks_removed_after_context():
    m = TinyModel()
    with FFNActivationHook(m) as h:
        assert h._handles  # registered inside context
    assert h._handles == []  # cleaned up on exit

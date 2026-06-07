"""Tests for weight_diff module (M1 ΔW extraction).

TDD: these tests are written first and drive the implementation in
src/quant_typo_neuron/quantization/weight_diff.py.

All tests run on CPU with synthetic tensors — no real models needed.
"""
from __future__ import annotations

import torch
import pytest

from quant_typo_neuron.quantization.weight_diff import (
    weight_diff,
    rtn_reconstruction_diff,
    diff_stats,
    extract_layer_weight_diffs,
)


# ---------------------------------------------------------------------------
# weight_diff: elementwise ΔW = W_fp16 − W_dequant
# ---------------------------------------------------------------------------

class TestWeightDiff:
    def test_shape_preserved(self):
        """Output shape equals input shape."""
        w = torch.randn(8, 16)
        d = torch.randn(8, 16)
        delta = weight_diff(w, d)
        assert delta.shape == w.shape

    def test_values_correct(self):
        """ΔW = W_fp16 − W_dequant elementwise."""
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        d = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        delta = weight_diff(w, d)
        expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        assert torch.allclose(delta, expected)

    def test_zero_when_equal(self):
        """ΔW = 0 when fp16 == dequant."""
        w = torch.randn(4, 8)
        delta = weight_diff(w, w.clone())
        assert torch.allclose(delta, torch.zeros_like(w))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        w = torch.randn(4, 8)
        d = torch.randn(4, 9)
        with pytest.raises(ValueError):
            weight_diff(w, d)


# ---------------------------------------------------------------------------
# rtn_reconstruction_diff: ΔW = weight − rtn_quantize(weight, bits, group_size)
# ---------------------------------------------------------------------------

class TestRtnReconstructionDiff:
    def setup_method(self, method):
        torch.manual_seed(42)
        self.weight = torch.randn(16, 64)

    def test_shape_preserved(self):
        delta = rtn_reconstruction_diff(self.weight, bits=8, group_size=32)
        assert delta.shape == self.weight.shape

    def test_bits8_smaller_error_than_bits4(self):
        """Higher bits → smaller reconstruction error ‖ΔW‖_F."""
        delta4 = rtn_reconstruction_diff(self.weight, bits=4, group_size=32)
        delta8 = rtn_reconstruction_diff(self.weight, bits=8, group_size=32)
        norm4 = delta4.norm()
        norm8 = delta8.norm()
        assert norm8 < norm4, (
            f"Expected ‖ΔW‖_F(bits=8)={norm8:.4f} < ‖ΔW‖_F(bits=4)={norm4:.4f}"
        )

    def test_values_match_manual(self):
        """ΔW = weight − rtn_quantize(weight) holds elementwise."""
        from typo_utils.quant.rtn import rtn_quantize
        delta = rtn_reconstruction_diff(self.weight, bits=4, group_size=32)
        expected = self.weight - rtn_quantize(self.weight, bits=4, group_size=32)
        assert torch.allclose(delta, expected)

    def test_dtype_preserved(self):
        """Output dtype matches input dtype."""
        w = torch.randn(8, 32, dtype=torch.float32)
        delta = rtn_reconstruction_diff(w, bits=4, group_size=16)
        assert delta.dtype == w.dtype


# ---------------------------------------------------------------------------
# diff_stats: per-row norm, per-col norm, frobenius norm, mean abs
# ---------------------------------------------------------------------------

class TestDiffStats:
    def setup_method(self, method):
        torch.manual_seed(7)
        self.out_features = 12
        self.in_features = 20
        self.delta = torch.randn(self.out_features, self.in_features)

    def test_returns_dict_with_expected_keys(self):
        stats = diff_stats(self.delta)
        for key in ("per_row_norm", "per_col_norm", "frobenius", "mean_abs"):
            assert key in stats, f"Missing key '{key}' in diff_stats output"

    def test_per_row_norm_shape(self):
        """per_row_norm has length == out_features."""
        stats = diff_stats(self.delta)
        row_norm = stats["per_row_norm"]
        assert len(row_norm) == self.out_features, (
            f"per_row_norm length {len(row_norm)} != out_features {self.out_features}"
        )

    def test_per_col_norm_shape(self):
        """per_col_norm has length == in_features."""
        stats = diff_stats(self.delta)
        col_norm = stats["per_col_norm"]
        assert len(col_norm) == self.in_features, (
            f"per_col_norm length {len(col_norm)} != in_features {self.in_features}"
        )

    def test_frobenius_matches_torch(self):
        """frobenius norm matches torch.linalg.norm(delta)."""
        stats = diff_stats(self.delta)
        expected = torch.linalg.norm(self.delta).item()
        actual = float(stats["frobenius"])
        assert abs(actual - expected) < 1e-4, (
            f"frobenius={actual:.6f} != torch.linalg.norm={expected:.6f}"
        )

    def test_mean_abs_scalar(self):
        """mean_abs is a scalar matching delta.abs().mean()."""
        stats = diff_stats(self.delta)
        expected = self.delta.abs().mean().item()
        actual = float(stats["mean_abs"])
        assert abs(actual - expected) < 1e-5

    def test_per_row_norm_values(self):
        """per_row_norm[i] == L2 norm of row i."""
        stats = diff_stats(self.delta)
        for i in range(self.out_features):
            expected = self.delta[i].norm().item()
            actual = float(stats["per_row_norm"][i])
            assert abs(actual - expected) < 1e-5, f"Row {i}: {actual} != {expected}"

    def test_per_col_norm_values(self):
        """per_col_norm[j] == L2 norm of col j."""
        stats = diff_stats(self.delta)
        for j in range(self.in_features):
            expected = self.delta[:, j].norm().item()
            actual = float(stats["per_col_norm"][j])
            assert abs(actual - expected) < 1e-5, f"Col {j}: {actual} != {expected}"


# ---------------------------------------------------------------------------
# extract_layer_weight_diffs: iterate matching Linear weights
# ---------------------------------------------------------------------------

class TestExtractLayerWeightDiffs:
    def _make_fake_model(self, names_and_shapes: dict):
        """Create a minimal fake model-like object with named Linear modules."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                for name, shape in names_and_shapes.items():
                    # Use safe attribute names (replace dots)
                    attr = name.replace(".", "_")
                    layer = nn.Linear(shape[1], shape[0], bias=False)
                    layer.weight.data = torch.randn(*shape)
                    setattr(self, attr, layer)

            def named_modules(self):
                yield "", self
                for name in names_and_shapes:
                    attr = name.replace(".", "_")
                    yield name, getattr(self, attr)

        return FakeModel()

    def test_returns_dict_of_tensors(self):
        """Result is a dict mapping layer names to ΔW tensors."""
        shapes = {"layer1": (8, 16), "layer2": (4, 8)}
        fp16 = self._make_fake_model(shapes)
        q = self._make_fake_model(shapes)
        result = extract_layer_weight_diffs(fp16, q)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(val, torch.Tensor)

    def test_delta_shape_matches_weight(self):
        """Each ΔW has the same shape as the corresponding Linear weight."""
        shapes = {"fc1": (8, 16), "fc2": (4, 8)}
        fp16 = self._make_fake_model(shapes)
        q = self._make_fake_model(shapes)
        result = extract_layer_weight_diffs(fp16, q)
        for name, shape in shapes.items():
            assert name in result, f"Layer '{name}' missing from result"
            assert result[name].shape == torch.Size(shape), (
                f"Shape mismatch for '{name}': {result[name].shape} != {shape}"
            )

    def test_delta_values_correct(self):
        """ΔW = fp16_weight − q_weight for each matched layer."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.layer = nn.ModuleList([nn.Linear(8, 4, bias=False)])
                self.layer[0].weight.data = w.clone()

            def named_modules(self):
                yield "", self
                yield "layer.0", self.layer[0]

        w_fp16 = torch.randn(4, 8)
        w_q = torch.randn(4, 8)
        fp16_model = SimpleModel(w_fp16)
        q_model = SimpleModel(w_q)

        result = extract_layer_weight_diffs(fp16_model, q_model)
        assert "layer.0" in result
        assert torch.allclose(result["layer.0"], w_fp16 - w_q)

    def test_layer_filter_applied(self):
        """layer_module_filter restricts which layers are included."""
        import torch.nn as nn

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.keep = nn.Linear(8, 4, bias=False)
                self.skip = nn.Linear(8, 4, bias=False)

            def named_modules(self):
                yield "", self
                yield "keep", self.keep
                yield "skip", self.skip

        fp16 = TwoLayerModel()
        q = TwoLayerModel()

        result = extract_layer_weight_diffs(fp16, q, layer_module_filter=lambda n, m: "keep" in n)
        assert "keep" in result
        assert "skip" not in result

    def test_empty_when_no_linears(self):
        """Returns empty dict if there are no Linear layers."""
        import torch.nn as nn

        class NoLinear(nn.Module):
            def named_modules(self):
                yield "", self

        fp16 = NoLinear()
        q = NoLinear()
        result = extract_layer_weight_diffs(fp16, q)
        assert result == {}

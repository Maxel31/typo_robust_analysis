"""intervention.patching のテスト (実験8: activation patching).

GPU 不要。小さなランダム初期化 Llama (4層) で forward hook による活性化の
捕捉/注入の正確性を検証する:

- デコーダ層の発見 (get_decoder 経路と vision 除外フォールバック)
- 捕捉値が output_hidden_states と一致すること (residual site)
- 恒等パッチ (自分自身の値の注入) が logits / greedy 生成を変えないこと
- 交差パッチが下流位置の logits だけを変えること (因果マスク整合)
- 最終層 residual パッチが donor の logits をその位置で再現すること
- 純関数 (層窓列挙・答え分岐トークン・スパン末尾トークン・冪等スキップ)
"""

import json

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from typo_cot.intervention.patching import (
    ActivationCache,
    PatchInjector,
    SITES,
    capture_activations,
    find_decoder_layers,
    first_divergence,
    get_site_module,
    iter_patch_cells,
    kl_from_logits,
    layer_windows,
    result_is_current,
    span_end_token,
)


@pytest.fixture(scope="module")
def tiny_model():
    """4層の小型ランダム Llama (CPU, float32)."""
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _input(seed: int, length: int = 10) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 99, (1, length), generator=g)


class TestFindDecoderLayers:
    def test_llama_layers(self, tiny_model):
        layers = find_decoder_layers(tiny_model)
        assert len(layers) == 4
        for layer in layers:
            assert hasattr(layer, "self_attn")
            assert hasattr(layer, "mlp")

    def test_fallback_skips_vision_tower(self):
        """get_decoder が無いモデルでは vision 系 ModuleList を飛ばして探す."""

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Identity()
                self.mlp = nn.Identity()

        class FakeMultimodal(nn.Module):
            def __init__(self):
                super().__init__()
                # 先に見つかる位置に vision 側を置く
                self.vision_tower = nn.Module()
                self.vision_tower.layers = nn.ModuleList([FakeLayer() for _ in range(2)])
                self.text_model = nn.Module()
                self.text_model.layers = nn.ModuleList([FakeLayer() for _ in range(3)])

        layers = find_decoder_layers(FakeMultimodal())
        assert len(layers) == 3

    def test_site_modules(self, tiny_model):
        layers = find_decoder_layers(tiny_model)
        assert get_site_module(layers[0], "residual") is layers[0]
        assert get_site_module(layers[0], "attn") is layers[0].self_attn
        assert get_site_module(layers[0], "mlp") is layers[0].mlp
        with pytest.raises(ValueError):
            get_site_module(layers[0], "bogus")


class TestCaptureActivations:
    def test_residual_matches_hidden_states(self, tiny_model):
        ids = _input(1)
        positions = [2, 5, 9]
        cache = capture_activations(tiny_model, ids, positions, sites=("residual",))

        with torch.no_grad():
            out = tiny_model(input_ids=ids, output_hidden_states=True)
        # hidden_states[l+1] = 第 l 層通過後の残差ストリーム
        for layer_idx in range(4):
            captured = cache.values("residual", layer_idx, positions)
            expected = out.hidden_states[layer_idx + 1][0, positions, :]
            assert torch.allclose(captured, expected, atol=1e-6)

    def test_attn_mlp_shapes(self, tiny_model):
        ids = _input(1)
        positions = [0, 3]
        cache = capture_activations(tiny_model, ids, positions, sites=SITES)
        for site in SITES:
            for layer_idx in range(4):
                vals = cache.values(site, layer_idx, positions)
                assert vals.shape == (2, 32)

    def test_values_subset_of_positions(self, tiny_model):
        """捕捉位置の部分集合を絶対位置で取り出せる."""
        ids = _input(1)
        cache = capture_activations(tiny_model, ids, [2, 5, 9], sites=("residual",))
        full = cache.values("residual", 0, [2, 5, 9])
        sub = cache.values("residual", 0, [5])
        assert torch.equal(sub[0], full[1])

    def test_unknown_position_raises(self, tiny_model):
        ids = _input(1)
        cache = capture_activations(tiny_model, ids, [2], sites=("residual",))
        with pytest.raises(KeyError):
            cache.values("residual", 0, [3])


class TestPatchInjector:
    def _logits(self, model, ids):
        with torch.no_grad():
            return model(input_ids=ids).logits

    def test_identity_patch_is_noop(self, tiny_model):
        """自分自身の活性を同位置に注入しても logits が変わらない (合格判定aの根拠)."""
        ids = _input(2)
        positions = list(range(4, 10))
        base = self._logits(tiny_model, ids)
        cache = capture_activations(tiny_model, ids, positions, sites=SITES)
        layers = find_decoder_layers(tiny_model)

        for site in SITES:
            values = {li: cache.values(site, li, positions) for li in range(4)}
            with PatchInjector(layers, site, list(range(4)), positions, values):
                patched = self._logits(tiny_model, ids)
            assert torch.allclose(patched, base, atol=1e-6), f"site={site}"

    def test_hooks_removed_after_exit(self, tiny_model):
        ids = _input(2)
        positions = [4]
        cache = capture_activations(tiny_model, _input(3), positions, sites=("residual",))
        layers = find_decoder_layers(tiny_model)
        values = {0: cache.values("residual", 0, positions)}
        base = self._logits(tiny_model, ids)
        with PatchInjector(layers, "residual", [0], positions, values):
            pass
        after = self._logits(tiny_model, ids)
        assert torch.equal(after, base)

    def test_cross_patch_changes_downstream_only(self, tiny_model):
        """donor≠recipient のパッチは下流位置の logits のみ変える (因果マスク)."""
        ids_a = _input(2)
        ids_b = _input(3)
        positions = [4, 5]
        cache_b = capture_activations(tiny_model, ids_b, positions, sites=("residual",))
        layers = find_decoder_layers(tiny_model)
        values = {li: cache_b.values("residual", li, positions) for li in (1, 2)}

        base = self._logits(tiny_model, ids_a)
        with PatchInjector(layers, "residual", [1, 2], positions, values):
            patched = self._logits(tiny_model, ids_a)

        # パッチ位置より前 (位置 0..3) は不変
        assert torch.allclose(patched[0, :4], base[0, :4], atol=1e-6)
        # 最終位置は変化する
        assert not torch.allclose(patched[0, -1], base[0, -1], atol=1e-4)

    def test_last_layer_residual_patch_reproduces_donor_logits(self, tiny_model):
        """最終層 residual を donor 値で置換するとその位置の logits は donor と一致."""
        ids_a = _input(2)
        ids_b = _input(3)
        pos = [7]
        cache_b = capture_activations(tiny_model, ids_b, pos, sites=("residual",))
        layers = find_decoder_layers(tiny_model)
        last = len(layers) - 1
        values = {last: cache_b.values("residual", last, pos)}

        donor_logits = self._logits(tiny_model, ids_b)
        with PatchInjector(layers, "residual", [last], pos, values):
            patched = self._logits(tiny_model, ids_a)
        assert torch.allclose(patched[0, 7], donor_logits[0, 7], atol=1e-5)

    def test_short_sequence_skips_patch(self, tiny_model):
        """dst 位置が系列長を超える forward ではパッチせずエラーも出さない."""
        ids = _input(2, length=3)
        cache = capture_activations(tiny_model, _input(3), [5], sites=("residual",))
        layers = find_decoder_layers(tiny_model)
        values = {0: cache.values("residual", 0, [5])}
        base = self._logits(tiny_model, ids)
        with PatchInjector(layers, "residual", [0], [5], values):
            patched = self._logits(tiny_model, ids)
        assert torch.equal(patched, base)

    def test_identity_patch_keeps_greedy_generation(self, tiny_model):
        """恒等パッチ下の greedy 生成が無パッチと一致 (decode ステップ安全性込み)."""
        ids = _input(4)
        positions = list(range(2, 8))
        cache = capture_activations(tiny_model, ids, positions, sites=("attn",))
        layers = find_decoder_layers(tiny_model)
        values = {li: cache.values("attn", li, positions) for li in range(4)}

        with torch.no_grad():
            base = tiny_model.generate(
                ids, max_new_tokens=4, do_sample=False, pad_token_id=0
            )
            with PatchInjector(layers, "attn", list(range(4)), positions, values):
                patched = tiny_model.generate(
                    ids, max_new_tokens=4, do_sample=False, pad_token_id=0
                )
        assert torch.equal(base, patched)


class TestLayerWindows:
    def test_nonoverlapping_gemma34b(self):
        windows = layer_windows(34, width=3, stride=3)
        assert len(windows) == 12
        assert windows[0] == (0, 3)
        assert windows[-1] == (33, 34)
        covered = set()
        for s, e in windows:
            covered.update(range(s, e))
        assert covered == set(range(34))

    def test_sliding(self):
        windows = layer_windows(5, width=3, stride=1)
        assert windows[0] == (0, 3)
        assert all(e <= 5 for _, e in windows)
        covered = set()
        for s, e in windows:
            covered.update(range(s, e))
        assert covered == set(range(5))

    def test_width_larger_than_layers(self):
        assert layer_windows(2, width=3, stride=3) == [(0, 2)]


class TestIterPatchCells:
    def test_full_sweep_enumeration(self):
        cells = list(
            iter_patch_cells(
                n_layers=34,
                window_size=3,
                window_stride=3,
                sites=("residual", "attn", "mlp"),
                directions=("clean_to_pert", "pert_to_clean"),
            )
        )
        assert len(cells) == 3 * 12 * 2
        first = cells[0]
        assert first.site in ("residual", "attn", "mlp")
        assert first.direction in ("clean_to_pert", "pert_to_clean")
        assert first.window == (0, 3)
        # (site, window, direction) は一意
        keys = {(c.site, c.window, c.direction) for c in cells}
        assert len(keys) == len(cells)


class TestFirstDivergence:
    def test_basic_divergence(self):
        div = first_divergence([5, 6, 7, 8], [5, 6, 9])
        assert div is not None
        assert div.common == [5, 6]
        assert div.token_a == 7
        assert div.token_b == 9

    def test_divergence_at_start(self):
        div = first_divergence([1, 2], [3, 4])
        assert div.common == []
        assert (div.token_a, div.token_b) == (1, 3)

    def test_identical_returns_none(self):
        assert first_divergence([1, 2, 3], [1, 2, 3]) is None

    def test_prefix_relation_returns_none(self):
        assert first_divergence([1, 2], [1, 2, 3]) is None

    def test_limit(self):
        assert first_divergence([1] * 20 + [2], [1] * 20 + [3], limit=16) is None


class TestSpanEndToken:
    def test_last_overlapping_token(self):
        offsets = [(0, 4), (4, 7), (7, 11), (11, 14)]
        # 文字スパン [4, 11) は トークン 1, 2 に跨る → 末尾はトークン 2
        assert span_end_token(offsets, 4, 11) == 2

    def test_single_token(self):
        offsets = [(0, 4), (4, 7), (7, 11)]
        assert span_end_token(offsets, 5, 6) == 1

    def test_no_overlap_returns_none(self):
        offsets = [(0, 4), (4, 7)]
        assert span_end_token(offsets, 10, 12) is None

    def test_zero_width_special_tokens_ignored(self):
        offsets = [(0, 0), (0, 4), (4, 7)]
        assert span_end_token(offsets, 0, 4) == 1


class TestKlFromLogits:
    def test_zero_for_identical(self):
        torch.manual_seed(0)
        logits = torch.randn(50)
        assert kl_from_logits(logits, logits.clone()) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different(self):
        torch.manual_seed(0)
        p = torch.randn(50)
        q = torch.randn(50)
        assert kl_from_logits(p, q) > 0.0


class TestResultIsCurrent:
    def test_missing_file(self, tmp_path):
        assert result_is_current(tmp_path / "nope.json", "abc") is False

    def test_matching_hash(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"config_hash": "abc", "cells": []}))
        assert result_is_current(path, "abc") is True

    def test_stale_hash(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"config_hash": "old"}))
        assert result_is_current(path, "abc") is False

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text("{not json")
        assert result_is_current(path, "abc") is False


class TestActivationCacheApi:
    def test_manual_construction(self):
        cache = ActivationCache(
            positions=[3, 7],
            data={("residual", 0): torch.zeros(2, 8)},
        )
        assert cache.values("residual", 0, [7]).shape == (1, 8)

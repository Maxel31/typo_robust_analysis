"""実験8-fine: 1層分解スイープの位置指定・相対深さ整列・sham 恒等性テスト.

GPU 不要。粗い窓 (幅3) の最良窓 residual[0,6) を 1 層解像度に精密化する
実験のための追加ユーティリティを検証する:

- 単層窓 (li, li+1) の位置指定 (single_layer_windows)
- 累積窓 (0, li+1) の位置指定 (cumulative_windows)
- 相対深さ li/L の算出とモデル間整列 (relative_depth / align_by_relative_depth)
- sham patch の恒等性: recipient 自身の値を摂動語スパンに書き戻すと
  単層窓・累積窓のいずれでも logits がビット不変 (アーチファクト検出の根拠)
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from typo_cot.intervention.patching import (
    PatchInjector,
    align_by_relative_depth,
    capture_activations,
    cumulative_windows,
    find_decoder_layers,
    relative_depth,
    single_layer_windows,
)


@pytest.fixture(scope="module")
def tiny_model():
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _input(seed: int, length: int = 12) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 99, (1, length), generator=g)


class TestSingleLayerWindows:
    def test_maps_each_layer_to_width1_window(self):
        assert single_layer_windows([0, 1, 2, 11]) == [(0, 1), (1, 2), (2, 3), (11, 12)]

    def test_validation_layers(self):
        # 早期 0-11 + 検証 14/20/26
        layers = list(range(12)) + [14, 20, 26]
        windows = single_layer_windows(layers)
        assert len(windows) == 15
        assert windows[12] == (14, 15)
        assert windows[-1] == (26, 27)
        # すべて幅 1
        assert all(e - s == 1 for s, e in windows)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            single_layer_windows([-1])


class TestCumulativeWindows:
    def test_prefix_windows_from_layer0(self):
        assert cumulative_windows([0, 1, 2]) == [(0, 1), (0, 2), (0, 3)]

    def test_l0_to_l11(self):
        windows = cumulative_windows(list(range(12)))
        assert len(windows) == 12
        assert windows[0] == (0, 1)
        assert windows[-1] == (0, 12)
        # すべて第0層から始まる
        assert all(s == 0 for s, _ in windows)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            cumulative_windows([-2])


class TestRelativeDepth:
    def test_single_layer_depth_is_l_over_L(self):
        # Gemma 第6層 (L=34) → 6/34 ≈ 0.176 < 0.2
        assert relative_depth(6, 34) == pytest.approx(6 / 34)
        assert relative_depth(6, 34) < 0.2
        # Mistral 第2層 (L=32) → 2/32 = 0.0625 < 0.2
        assert relative_depth(2, 32) == pytest.approx(2 / 32)

    def test_zero_layer(self):
        assert relative_depth(0, 28) == 0.0

    def test_rejects_nonpositive_layers(self):
        with pytest.raises(ValueError):
            relative_depth(0, 0)
        with pytest.raises(ValueError):
            relative_depth(0, -3)


class TestAlignByRelativeDepth:
    def test_overlays_models_on_common_axis(self):
        # 3 モデルの層プロファイル (絶対層番号) を相対深さで整列
        profiles = {
            "gemma": {0: 0.1, 6: 0.9, 17: 0.5},
            "mistral": {0: 0.8, 2: 0.85, 16: 0.4},
        }
        n_layers = {"gemma": 34, "mistral": 32}
        aligned = align_by_relative_depth(profiles, n_layers)
        # 各モデルは (rel_depth, value) の昇順リスト
        assert [d for d, _ in aligned["gemma"]] == sorted(d for d, _ in aligned["gemma"])
        # 値が保持される (gemma 第6層 → 相対深さ 6/34)
        g = {round(d, 6): v for d, v in aligned["gemma"]}
        assert g[round(6 / 34, 6)] == 0.9

    def test_missing_n_layers_raises(self):
        with pytest.raises((KeyError, ValueError)):
            align_by_relative_depth({"m": {0: 1.0}}, {})


class TestShamPatchIdentity:
    """sham patch = recipient 自身の値を同位置に書き戻すダミー (効果ゼロのはず)."""

    def _logits(self, model, ids):
        with torch.no_grad():
            return model(input_ids=ids).logits

    def test_single_layer_sham_is_bit_identical(self, tiny_model):
        """摂動語スパン相当の位置で単層 sham が logits をビット不変に保つ."""
        ids = _input(2)
        span_positions = [3, 4, 7]  # 摂動語スパン相当
        base = self._logits(tiny_model, ids)
        cache = capture_activations(tiny_model, ids, span_positions, sites=("residual",))
        layers = find_decoder_layers(tiny_model)

        for s, e in single_layer_windows([0, 1, 5, 7]):
            layer_indices = list(range(s, e))
            values = {li: cache.values("residual", li, span_positions) for li in layer_indices}
            with PatchInjector(layers, "residual", layer_indices, span_positions, values):
                patched = self._logits(tiny_model, ids)
            assert torch.equal(patched, base), f"sham single window ({s},{e}) が非恒等"

    def test_cumulative_sham_is_bit_identical(self, tiny_model):
        """累積 sham (第0層〜第l層に自分の値を書き戻す) も logits をビット不変に保つ."""
        ids = _input(3)
        span_positions = [2, 6]
        base = self._logits(tiny_model, ids)
        cache = capture_activations(tiny_model, ids, span_positions, sites=("residual",))
        layers = find_decoder_layers(tiny_model)

        for s, e in cumulative_windows([0, 3, 6]):
            layer_indices = list(range(s, e))
            values = {li: cache.values("residual", li, span_positions) for li in layer_indices}
            with PatchInjector(layers, "residual", layer_indices, span_positions, values):
                patched = self._logits(tiny_model, ids)
            assert torch.equal(patched, base), f"sham cumulative window ({s},{e}) が非恒等"

    def test_real_donor_changes_downstream(self, tiny_model):
        """対照: 別 run (real donor) の値を単層注入すると下流位置が変化する."""
        ids_recip = _input(2)
        ids_donor = _input(9)
        span_positions = [3, 4]
        base = self._logits(tiny_model, ids_recip)
        donor_cache = capture_activations(
            tiny_model, ids_donor, span_positions, sites=("residual",)
        )
        layers = find_decoder_layers(tiny_model)
        (s, e) = single_layer_windows([2])[0]
        values = {2: donor_cache.values("residual", 2, span_positions)}
        with PatchInjector(layers, "residual", [2], span_positions, values):
            patched = self._logits(tiny_model, ids_recip)
        # スパン最大位置 (4) より下流 (最終位置) は変化
        assert not torch.allclose(patched[0, -1], base[0, -1], atol=1e-4)

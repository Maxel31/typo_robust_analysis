"""run_patching_fine.run_pair_fine の CPU 統合テスト (tiny Llama).

アーカイブ無しで合成 PreparedPair を組み、実際の実行経路
(両 run 捕捉 → 単層/累積/noising/sham セル生成 → S2 KL 回復率) を検証する:

- 4 系統 (single/cumulative/noising/sham_single) のセルが期待数だけ出る
- sham セルは generation_identical_to_recipient=True かつ s2_kl_recovery≈0
  (recipient 自身の値を書き戻すため恒等)
- 単層 denoising セルは s2_kl_recovery を持ち、恒等ではない
- GPU 不要
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from typo_cot.intervention.patching import find_decoder_layers

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "exp8"


@pytest.fixture(scope="module")
def rpf():
    """run_patching_fine モジュールをファイルパスからロード."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_patching_fine", SCRIPTS / "run_patching_fine.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = 0
    return model


class _StubExtracted:
    def __init__(self, ans):
        self.extracted_answer = ans


class _StubExtractor:
    def extract(self, text):
        return _StubExtracted(text.strip()[:4])


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _make_prepared(rpf):
    """合成 PreparedPair: clean/pert は span 位置のみ token が異なる."""
    prepared = rpf.PreparedPair()
    prompt_len = 8
    # 共通 suffix (プロンプト以降は両 run 同一 = teacher forcing)
    suffix = [40, 41, 42]
    clean_prompt = [5, 3, 9, 2, 7, 1, 4, 6]
    pert_prompt = [5, 3, 20, 2, 7, 1, 4, 6]  # span 位置 2 のみ差替
    prepared.input_ids = {
        "clean": clean_prompt + suffix,
        "pert": pert_prompt + suffix,
    }
    prepared.prompt_len = {"clean": prompt_len, "pert": prompt_len}
    prepared.trigger_start = {"clean": prompt_len, "pert": prompt_len}
    prepared.span_positions = {"clean": [2], "pert": [2]}
    prepared.readout_tokens = (40, 41)
    prepared.readout_prefix_text = ""
    prepared.suffix_len = len(suffix)
    prepared.meta = {"sample_id": "syn"}
    return prepared


def _args():
    return SimpleNamespace(
        single_layers="0-3",
        cumulative_layers="0-3",
        noising_layers="0-2",
        sham_layers="0-3",
        max_new_tokens=3,
    )


def test_run_pair_fine_produces_all_cell_families(rpf, tiny_model):
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), _args()
    )
    cells = result["cells"]
    kinds = [c["kind"] for c in cells]
    assert kinds.count("single") == 4          # 層 0-3
    assert kinds.count("cumulative") == 4       # (0,1)..(0,4)
    assert kinds.count("noising") == 3          # 層 0-2
    assert kinds.count("sham_single") == 4      # 層 0-3
    assert result["n_layers"] == 8
    assert "clean_to_pert" in result["s2_kl_unpatched"]


def test_sham_cells_are_identity(rpf, tiny_model):
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), _args()
    )
    sham = [c for c in result["cells"] if c["kind"] == "sham_single"]
    assert sham, "sham セルが無い"
    for c in sham:
        # recipient 自身の値 → 生成は recipient と一致、S2 KL 回復はほぼ 0
        assert c["generation_identical_to_recipient"] is True
        if "s2_kl_recovery" in c:
            assert abs(c["s2_kl_recovery"]) < 1e-4


def test_single_denoising_cells_have_recovery(rpf, tiny_model):
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), _args()
    )
    single = [c for c in result["cells"] if c["kind"] == "single"]
    assert all(c["direction"] == "clean_to_pert" for c in single)
    # denoising は s2_kl_recovery を持つ (質問スパンパッチ) 場合がある
    assert any("s2_kl_recovery" in c for c in single)


def test_a3_controls_present(rpf, tiny_model):
    """A3 統制 other_span / all_positions のセルが出る."""
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), _args()
    )
    kinds = {c["kind"] for c in result["cells"]}
    assert "other_span" in kinds
    assert "all_positions" in kinds


def test_all_positions_gives_full_recovery(rpf, tiny_model):
    """全位置を clean 値で patch すると c1 は完全に clean = s2_kl_recovery≈1 (重み非依存)."""
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), _args()
    )
    allp = [c for c in result["cells"] if c["kind"] == "all_positions" and "s2_kl_recovery" in c]
    assert allp, "all_positions セルが無い"
    for c in allp:
        # 全位置 clean → c1 分布は clean と一致 → KL 回復率 ≈ 1
        assert c["s2_kl_recovery"] > 0.99


def test_semantic_mode_single_only(rpf, tiny_model):
    """perturb_mode=semantic では単層 'semantic' セルのみ (他系統・統制なし)."""
    prepared = _make_prepared(rpf)
    layers = find_decoder_layers(tiny_model)
    args = _args()
    args.perturb_mode = "semantic"
    result = rpf.run_pair_fine(
        tiny_model, _StubTokenizer(), layers, prepared, _StubExtractor(), args
    )
    kinds = {c["kind"] for c in result["cells"]}
    assert kinds == {"semantic"}
    assert len([c for c in result["cells"] if c["kind"] == "semantic"]) == 4

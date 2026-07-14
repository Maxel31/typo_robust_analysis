"""repair.lexicon_probe のテスト (実験9: 層別 cos・修復スコア・logit lens).

GPU 不要。小さな合成テンソルとダミーモジュールのみで完結する。
"""

import torch
from torch import nn

from typo_cot.repair.lexicon_probe import (
    LogitLens,
    extract_span_hiddens,
    find_decoder_backbone,
    get_final_softcap,
    layerwise_cos,
    repair_score,
)


class TestLayerwiseCos:
    def test_identical_states_give_one(self) -> None:
        h = torch.randn(5, 8)  # [L+1, d]
        cos = layerwise_cos(h, h)
        assert cos.shape == (5,)
        assert torch.allclose(cos, torch.ones(5), atol=1e-5)

    def test_orthogonal_states_give_zero(self) -> None:
        h1 = torch.zeros(3, 4)
        h2 = torch.zeros(3, 4)
        h1[:, 0] = 1.0
        h2[:, 1] = 1.0
        cos = layerwise_cos(h1, h2)
        assert torch.allclose(cos, torch.zeros(3), atol=1e-6)

    def test_bfloat16_inputs_are_upcast(self) -> None:
        h = torch.randn(4, 8).to(torch.bfloat16)
        cos = layerwise_cos(h, h)
        assert cos.dtype == torch.float32
        assert torch.allclose(cos, torch.ones(4), atol=1e-3)


class TestRepairScore:
    def test_max_over_layers_excluding_embedding(self) -> None:
        # 層0 (埋め込み) が最大でも無視され、層2 の値が選ばれる
        curve = torch.tensor([0.99, 0.20, 0.80, 0.50])
        score, layer = repair_score(curve)
        assert abs(score - 0.80) < 1e-6
        assert layer == 2

    def test_include_embedding_option(self) -> None:
        curve = torch.tensor([0.99, 0.20, 0.80])
        score, layer = repair_score(curve, skip_input_embedding=False)
        assert abs(score - 0.99) < 1e-6
        assert layer == 0


class TestLogitLens:
    def _make_lens(self, softcap: float | None = None) -> LogitLens:
        # 単位行列 unembed: hidden = e_v なら token v が最大 logit
        d = 6
        unembed = torch.eye(d)  # [V=d, d]
        return LogitLens(norm=nn.Identity(), unembed_weight=unembed, softcap=softcap)

    def test_rank_of_target_token(self) -> None:
        lens = self._make_lens()
        hidden = torch.zeros(3, 6)  # [L+1, d]
        hidden[0, 4] = 1.0  # 層0 は token4 を指す
        hidden[1, 2] = 1.0  # 層1 は token2 を指す
        hidden[2, 2] = 1.0
        ranks = lens.layer_ranks(hidden, target_id=2)
        assert ranks[1] == 0 and ranks[2] == 0
        assert ranks[0] > 0

    def test_first_hit_layer(self) -> None:
        lens = self._make_lens()
        hidden = torch.zeros(3, 6)
        hidden[2, 5] = 1.0
        ranks = lens.layer_ranks(hidden, target_id=5)
        assert lens.first_hit_layer(ranks, top_k=1) == 2
        # どの層でも hit しない場合は None
        ranks_none = lens.layer_ranks(hidden, target_id=1)
        assert lens.first_hit_layer(ranks_none, top_k=1) is None

    def test_softcap_bounds_logits(self) -> None:
        lens = self._make_lens(softcap=0.5)
        hidden = torch.zeros(1, 6)
        hidden[0, 3] = 100.0
        logits = lens.project(hidden)
        assert logits.abs().max().item() <= 0.5 + 1e-6

    def test_topk_tokens(self) -> None:
        lens = self._make_lens()
        hidden = torch.zeros(2, 6)
        hidden[1, 0] = 2.0
        hidden[1, 3] = 1.0
        top = lens.layer_topk(hidden, k=2)
        assert top[1][0] == 0 and top[1][1] == 3


class _DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyDecoder()
        self.lm_head = nn.Linear(4, 10, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class TestFindDecoderBackbone:
    def test_finds_norm_and_unembed(self) -> None:
        m = _DummyModel()
        norm, unembed = find_decoder_backbone(m)
        assert norm is m.model.norm
        assert unembed is m.lm_head.weight


class _Cfg:
    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class TestGetFinalSoftcap:
    def test_direct_attribute(self) -> None:
        assert get_final_softcap(_Cfg(final_logit_softcapping=30.0)) == 30.0

    def test_nested_text_config(self) -> None:
        cfg = _Cfg(text_config=_Cfg(final_logit_softcapping=25.0))
        assert get_final_softcap(cfg) == 25.0

    def test_absent_returns_none(self) -> None:
        assert get_final_softcap(_Cfg()) is None
        assert get_final_softcap(_Cfg(final_logit_softcapping=None)) is None


class _FakeTokenizerForExtract:
    """offset_mapping 付き encode を模す."""

    def __call__(self, text: str, return_offsets_mapping: bool, return_tensors: str) -> dict:
        # 4文字=1トークンの固定分割
        offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
        ids = torch.arange(len(offsets)).unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "offset_mapping": torch.tensor(offsets).unsqueeze(0),
        }


class _FakeModelForExtract(nn.Module):
    """hidden_states = (層0, 層1) を返す。値は position 番号を埋め込む."""

    def __init__(self, d: int = 3) -> None:
        super().__init__()
        self.d = d
        self.dummy = nn.Linear(1, 1)  # device 判定用

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool) -> object:
        t = input_ids.shape[1]
        base = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(1, t, self.d)

        class Out:
            hidden_states = (base.clone(), base.clone() + 100.0)

        return Out()


class TestExtractSpanHiddens:
    def test_extracts_span_end_positions(self) -> None:
        model = _FakeModelForExtract()
        tok = _FakeTokenizerForExtract()
        text = "abcdefghijklmnop"  # 4 トークン
        # スパン (5, 11) はトークン1..2 に重なる → 末尾トークンは 2
        hiddens, positions = extract_span_hiddens(model, tok, text, [(5, 11)])
        assert positions == [2]
        assert hiddens.shape == (2, 1, 3)  # [L+1, n_spans, d]
        assert torch.allclose(hiddens[0, 0], torch.full((3,), 2.0))
        assert torch.allclose(hiddens[1, 0], torch.full((3,), 102.0))

    def test_unmatched_span_gives_none_position(self) -> None:
        model = _FakeModelForExtract()
        tok = _FakeTokenizerForExtract()
        text = "abcdefgh"
        hiddens, positions = extract_span_hiddens(model, tok, text, [(100, 105)])
        assert positions == [None]
        assert hiddens.shape[1] == 1
        # 未整列スパンの hidden は NaN で埋める
        assert torch.isnan(hiddens[:, 0]).all()

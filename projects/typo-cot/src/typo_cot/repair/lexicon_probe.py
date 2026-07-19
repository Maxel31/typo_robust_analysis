"""層別 hidden 抽出 + logit lens + 修復スコア (実験9).

clean/typo 質問の forward (output_hidden_states=True) から
摂動語スパン末尾トークンの層別隠れ状態を取り出し、
- 層別 cos 類似 (layerwise_cos) と 修復スコア = 最大層 cos (repair_score)
- logit lens (最終 LayerNorm + unembed を各層 hidden に適用) による
  clean 語先頭トークンの復号ランク (LogitLens)
を計算する。

モデル固有の注意点:
- Gemma-2 系は final_logit_softcapping を持つ (Gemma-3 は None)。
  get_final_softcap() で config から取得し、LogitLens が tanh cap を適用する。
- Gemma-3 のマルチモーダル構成 (language_model 内包) でも
  find_decoder_backbone() が `layers`+`norm` を持つデコーダを探し当てる。
"""

import math

import torch
from torch import nn

from typo_cot.repair.span_align import char_span_to_last_token


def layerwise_cos(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """層別 cos 類似を float32 で計算する.

    Args:
        h_a: [L+1, d] (または [L+1, n, d]) の隠れ状態
        h_b: 同形状の隠れ状態

    Returns:
        [L+1] (または [L+1, n]) の cos 類似 (float32)
    """
    a = h_a.to(torch.float32)
    b = h_b.to(torch.float32)
    return nn.functional.cosine_similarity(a, b, dim=-1)


def repair_score(
    cos_curve: torch.Tensor, skip_input_embedding: bool = True
) -> tuple[float, int]:
    """修復スコア = 最大層 cos とその層番号を返す.

    Args:
        cos_curve: [L+1] の層別 cos (index 0 = 入力埋め込み層)
        skip_input_embedding: True なら層0 (埋め込み) を最大値の探索から除外。
            typo 語と clean 語はトークン自体が違うため層0の類似はスコアに含めない。

    Returns:
        (最大 cos, その層番号)
    """
    curve = cos_curve.to(torch.float32)
    start = 1 if skip_input_embedding and curve.shape[0] > 1 else 0
    sub = curve[start:]
    idx = int(torch.argmax(sub).item())
    return float(sub[idx].item()), start + idx


class LogitLens:
    """各層 hidden を最終 norm + unembed で語彙に射影する logit lens.

    Attributes:
        norm: 最終正規化層 (RMSNorm / LayerNorm / Identity)
        unembed_weight: [V, d] の unembedding 行列
        softcap: final logit softcapping (Gemma-2 系)。None なら適用しない。
    """

    def __init__(
        self,
        norm: nn.Module,
        unembed_weight: torch.Tensor,
        softcap: float | None = None,
    ) -> None:
        self.norm = norm
        self.unembed_weight = unembed_weight
        self.softcap = softcap

    @classmethod
    def from_model(cls, model: nn.Module) -> "LogitLens":
        """HF CausalLM からノルム層・unembed 行列・softcap を抽出して構築する."""
        norm, unembed = find_decoder_backbone(model)
        softcap = get_final_softcap(getattr(model, "config", None))
        return cls(norm=norm, unembed_weight=unembed, softcap=softcap)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """[L+1, d] の hidden を [L+1, V] の logits (float32) に射影する."""
        with torch.no_grad():
            h = hidden.to(self.unembed_weight.device)
            params = list(self.norm.parameters())
            if params:
                h = h.to(params[0].dtype)
            normed = self.norm(h)
            logits = normed.to(self.unembed_weight.dtype) @ self.unembed_weight.T
            logits = logits.to(torch.float32)
            if self.softcap is not None:
                logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    def layer_ranks(self, hidden: torch.Tensor, target_id: int) -> list[int]:
        """各層で target_id トークンの logit が何位か (0=1位) を返す."""
        logits = self.project(hidden)  # [L+1, V]
        target = logits[:, target_id].unsqueeze(-1)  # [L+1, 1]
        # 同値 (tie) は標的の不利に数える (悲観的ランク)。
        # 全ロジット同値の退化ケースを「復号成功」と誤判定しないため。
        greater = (logits > target).sum(dim=-1)
        ties = (logits == target).sum(dim=-1) - 1  # 自分自身を除く
        ranks = greater + ties.clamp(min=0)
        return [int(r.item()) for r in ranks]

    def layer_topk(self, hidden: torch.Tensor, k: int = 5) -> list[list[int]]:
        """各層の上位 k トークン ID を返す."""
        logits = self.project(hidden)
        top = torch.topk(logits, k=k, dim=-1).indices
        return [[int(i) for i in row] for row in top]

    @staticmethod
    def first_hit_layer(ranks: list[int], top_k: int = 5) -> int | None:
        """rank < top_k となる最初の層番号 (無ければ None)."""
        for layer, r in enumerate(ranks):
            if r < top_k:
                return layer
        return None


def find_decoder_backbone(model: nn.Module) -> tuple[nn.Module, torch.Tensor]:
    """デコーダの最終 norm と unembedding 行列を探す.

    `layers` と `norm` の両属性を持つモジュールをデコーダ本体とみなす
    (Llama/Qwen/Mistral/Gemma のテキストデコーダ構造。
    Gemma-3 マルチモーダルの language_model 内包構成にも対応)。

    Returns:
        (最終 norm モジュール, unembed 重み [V, d])
    """
    decoder: nn.Module | None = None
    for module in model.modules():
        if hasattr(module, "layers") and hasattr(module, "norm"):
            decoder = module
            break
    if decoder is None:
        raise ValueError("デコーダ本体 (layers + norm を持つモジュール) が見つかりません")

    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise ValueError("get_output_embeddings() から unembed 重みを取得できません")
    return decoder.norm, out_emb.weight


def get_final_softcap(config: object | None) -> float | None:
    """config から final_logit_softcapping を取得する (Gemma-2 系).

    テキスト config が入れ子の場合 (text_config) も探索する。
    """
    if config is None:
        return None
    cap = getattr(config, "final_logit_softcapping", None)
    if cap is not None:
        return float(cap)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        cap = getattr(text_config, "final_logit_softcapping", None)
        if cap is not None:
            return float(cap)
    return None


@torch.no_grad()
def extract_span_hiddens(
    model: nn.Module,
    tokenizer: object,
    text: str,
    spans: list[tuple[int, int]],
) -> tuple[torch.Tensor, list[int | None]]:
    """テキストを 1 回 forward し、各文字スパン末尾トークンの層別 hidden を返す.

    Args:
        model: output_hidden_states=True に対応する CausalLM
        tokenizer: return_offsets_mapping に対応するトークナイザ (fast)
        text: 入力テキスト (プロンプト全体)
        spans: 文字スパン (start, end) のリスト

    Returns:
        hiddens: [L+1, n_spans, d] (cpu, float32)。
            スパンがトークンに整列できなかった場合は NaN 埋め。
        positions: 各スパンの末尾トークン位置 (整列不能なら None)
    """
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # hidden_states: (L+1) x [1, T, d]
    n_layers = len(out.hidden_states)
    d = out.hidden_states[0].shape[-1]

    positions = [char_span_to_last_token(offsets, span) for span in spans]
    hiddens = torch.full((n_layers, len(spans), d), math.nan, dtype=torch.float32)
    for j, pos in enumerate(positions):
        if pos is None:
            continue
        for li in range(n_layers):
            hiddens[li, j] = out.hidden_states[li][0, pos].to(torch.float32).cpu()
    return hiddens, positions

"""実験6-(i)〜(iii): 帰属ファミリー代替手法 (AttnLRP の R_C 比較用).

3手法とも「CoT 語ランキング」を R_C / LOO と同一スキーマ
(`[{"word": str, "score": float}]` 降順) で出力する:

- (i)   Gradient×Input: 答えトークン列 log-prob を目的関数とし、CoT トークン
        埋め込みへの勾配×入力 (backward 1回)。
- (ii)  Integrated Gradients: 同目的関数、ベースライン = ゼロ埋め込み、
        midpoint リーマン和 m ステップ (completeness 診断付き)。
- (iii) Attention rollout: 層ごとの head 平均 attention に残差 0.5I を混ぜて
        行正規化し層積 (Abnar & Zuidema 2020)。forward のみ。

共通規約 (exp6-iv LOO と同一):
- 3分割は `loo_scorer.split_generated_text` (ANSWER_PATTERNS 共有)。
- 目的関数の context = prompt + cot + trigger、target = answer トークン列。
  answer より後のテキストは入力から除外する。
- トークン→語の集約は `loo_scorer.rc_word_ranking_from_token_scores`
  (Mistral R_C 再構築ローダー) と同一の空白チャンク整合を再利用。

設計メモ: docs/dev_notes_06_attribution.md
"""

from dataclasses import dataclass

import torch

from typo_cot.intervention.loo_scorer import (
    _encode,
    _model_device,
    _target_start,
    rc_word_ranking_from_token_scores,
    split_generated_text,
)


@dataclass
class PreparedSample:
    """1サンプルの帰属計算入力 (3分割 + トークン範囲)."""

    full_text: str  # prompt + cot + trigger + answer (answer より後は除外)
    input_ids: list[int]
    cot_token_start: int  # CoT 先頭トークン (プロンプト直後)
    cot_token_end: int  # CoT 末尾トークン (trigger 直前)
    target_start: int  # 答えトークン列の開始インデックス
    answer_text: str
    trigger_text: str
    pattern_type: str


def prepare_sample(tokenizer, prompt: str, generated_text: str) -> PreparedSample | None:
    """生成テキストを3分割し、帰属対象のトークン範囲を確定する.

    Returns:
        PreparedSample。回答パターンが無い / CoT が空の場合は None。
    """
    split = split_generated_text(generated_text)
    if split is None:
        return None

    ctx = prompt + split.cot_text + split.trigger_text
    full_text = ctx + split.answer_text
    full_ids = _encode(tokenizer, full_text)
    prompt_ids = _encode(tokenizer, prompt)
    prompt_cot_ids = _encode(tokenizer, prompt + split.cot_text)
    ctx_ids = _encode(tokenizer, ctx)

    cot_token_start = _target_start(full_ids, prompt_ids)
    trigger_start = _target_start(full_ids, prompt_cot_ids)
    target_start = _target_start(full_ids, ctx_ids)
    cot_token_end = trigger_start - 1

    if cot_token_end < cot_token_start or target_start >= len(full_ids):
        return None

    return PreparedSample(
        full_text=full_text,
        input_ids=full_ids,
        cot_token_start=cot_token_start,
        cot_token_end=cot_token_end,
        target_start=target_start,
        answer_text=split.answer_text,
        trigger_text=split.trigger_text,
        pattern_type=split.pattern_type,
    )


def answer_logprob_from_logits(
    logits: torch.Tensor, input_ids: list[int], target_start: int
) -> torch.Tensor:
    """答えトークン列の log-prob 合計 (teacher forcing、バッチ対応).

    位置 t-1 の logits が位置 t のトークンを予測する。大語彙モデルの
    メモリ節約のため、答え位置の行のみ切り出してから log_softmax する。

    Args:
        logits: (B, S, V)
        input_ids: 長さ S のトークン列 (バッチ内で共通)
        target_start: 答えトークン列の開始インデックス (>=1)

    Returns:
        (B,) の log-prob 合計 (float32、微分可能)
    """
    n = len(input_ids)
    rows = logits[:, target_start - 1 : n - 1, :].float()  # (B, T, V)
    logp = torch.log_softmax(rows, dim=-1)
    targets = torch.tensor(
        input_ids[target_start:], dtype=torch.long, device=logits.device
    )
    picked = logp.gather(-1, targets.view(1, -1, 1).expand(logp.shape[0], -1, 1))
    return picked.squeeze(-1).sum(dim=-1)


def gradient_x_input_token_scores(
    model,
    input_ids: list[int],
    target_start: int,
    device: torch.device | None = None,
) -> tuple[list[float], float]:
    """(i) Gradient×Input: backward 1回でトークン別スコアを計算する.

    Returns:
        (token_scores, objective) — token_scores[i] = grad·embed の内積、
        objective = 答えトークン列 log-prob 合計 (base_logprob 相当)。
    """
    if device is None:
        device = _model_device(model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        obj = answer_logprob_from_logits(logits, input_ids, target_start)[0]
        obj.backward()

    scores = (embeds.grad.float() * embeds.float()).sum(-1)[0].detach().cpu()
    obj_value = float(obj.detach())
    del embeds, logits, obj
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores.tolist(), obj_value


def integrated_gradients_token_scores(
    model,
    input_ids: list[int],
    target_start: int,
    steps: int = 16,
    step_batch: int = 4,
    device: torch.device | None = None,
) -> tuple[list[float], dict]:
    """(ii) Integrated Gradients (ベースライン = ゼロ埋め込み、midpoint 則).

    IG_i = embed_i · (1/m) Σ_k grad F(α_k · embed), α_k = (k+0.5)/m。
    α ステップは step_batch 件ずつバッチ forward/backward する
    (系列長が同一なのでパディング不要)。

    Returns:
        (token_scores, info) — info は completeness 診断
        {"sum_attr", "f_x", "f_baseline", "completeness_ratio"}。
    """
    if device is None:
        device = _model_device(model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        base_embeds = model.get_input_embeddings()(ids).detach()  # (1, S, D)

    total_grad = torch.zeros(base_embeds.shape[1:], dtype=torch.float32, device=device)
    alphas = [(k + 0.5) / steps for k in range(steps)]
    for i in range(0, steps, step_batch):
        chunk = alphas[i : i + step_batch]
        a = torch.tensor(chunk, dtype=base_embeds.dtype, device=device).view(-1, 1, 1)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            e = (a * base_embeds).detach().requires_grad_(True)  # (b, S, D)
            logits = model(inputs_embeds=e, use_cache=False).logits
            obj = answer_logprob_from_logits(logits, input_ids, target_start)
            obj.sum().backward()
        total_grad += e.grad.float().sum(0)
        del e, logits, obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_grad = total_grad / steps
    scores = (avg_grad * base_embeds[0].float()).sum(-1).detach().cpu()

    with torch.no_grad():
        f_x = float(
            answer_logprob_from_logits(
                model(inputs_embeds=base_embeds, use_cache=False).logits,
                input_ids,
                target_start,
            )[0]
        )
        f_0 = float(
            answer_logprob_from_logits(
                model(inputs_embeds=torch.zeros_like(base_embeds), use_cache=False).logits,
                input_ids,
                target_start,
            )[0]
        )
    sum_attr = float(scores.sum())
    delta = f_x - f_0
    info = {
        "sum_attr": sum_attr,
        "f_x": f_x,
        "f_baseline": f_0,
        "completeness_ratio": (sum_attr / delta) if abs(delta) > 1e-9 else None,
    }
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores.tolist(), info


def rollout_from_attentions(
    attentions, residual: float = 0.5
) -> torch.Tensor:
    """(iii) attention 行列の層積 (Abnar & Zuidema 2020 の rollout).

    各層: head 平均 → residual·I + (1-residual)·A → 行正規化 → 左から積算。

    Args:
        attentions: 層ごとの attention。各要素は (H, S, S) または (1, H, S, S)。
        residual: 残差接続の混合率 (慣行値 0.5)。

    Returns:
        (S, S) の rollout 行列 (行和 1)。
    """
    r = None
    for a in attentions:
        if a.dim() == 4:
            a = a[0]
        a = a.float().mean(0)  # (S, S)
        s = a.shape[-1]
        eye = torch.eye(s, dtype=a.dtype, device=a.device)
        a = residual * eye + (1.0 - residual) * a
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
        r = a if r is None else a @ r
    return r


def attention_rollout_token_scores(
    model,
    input_ids: list[int],
    target_start: int,
    device: torch.device | None = None,
    residual: float = 0.5,
) -> list[float]:
    """(iii) rollout によるトークン別スコア (forward のみ).

    答えトークン列を予測する位置 (target_start-1 .. S-2) の rollout 行を
    平均し、各入力トークンへの attention 流を返す。

    Note:
        output_attentions=True が実 attention を返すよう、モデルは
        attn_implementation="eager" でロードしておくこと。
    """
    if device is None:
        device = _model_device(model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
        r = rollout_from_attentions(out.attentions, residual=residual)
        n = len(input_ids)
        rows = r[max(target_start - 1, 0) : n - 1]
        if rows.shape[0] == 0:
            rows = r[-1:]
        scores = rows.mean(0).detach().cpu()
    del out, r
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores.tolist()


def decode_tokens_for_alignment(tokenizer, input_ids: list[int]) -> list[str]:
    """整合用トークン文字列列を作る (特殊トークンは "<s>" に写像).

    `align_tokens_to_text` の _SPECIAL_TOKENS 集合は "<s>" を含むため、
    tokenizer 固有の特殊トークン ("<|begin_of_text|>" 等) を "<s>" に
    置き換えることで、モデルによらず整合対象外として扱われる。
    """
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    return [
        "<s>" if tid in special else tokenizer.decode([tid]) for tid in input_ids
    ]


def token_scores_to_word_ranking(
    tokens: list[str],
    scores: list[float],
    full_text: str,
    cot_token_start: int,
    cot_token_end: int,
) -> list[dict] | None:
    """トークンスコアを CoT 領域の語ランキングに集約する.

    R_C 再構築ローダー (`rc_word_ranking_from_token_scores`) をそのまま
    再利用し、語の単位 (空白チャンク)・CoT 領域フィルタ・スコア合計の
    規約を LOO vs R_C 比較経路と完全に一致させる。

    Returns:
        [{"word", "score"}] スコア降順。トークン整合失敗時は None。
    """
    data = {
        "token_scores": list(zip(tokens, scores, strict=True)),
        "cot_token_start": cot_token_start,
        "cot_token_end": cot_token_end,
    }
    return rc_word_ranking_from_token_scores(data, full_text)

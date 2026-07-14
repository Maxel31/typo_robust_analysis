"""実験6-(iv): leave-one-out (LOO) 重要度スコアラ.

帰属（attribution）という枠組みを使わない削除ベースの語重要度:
clean CoT の各語タイプについて全出現を削除した変種 CoT を作り、
(質問プロンプト + 変種 CoT + 答えトリガー) を teacher-forcing して
「元の答えトークン列の log-prob 合計」の低下量をその語の重要度とする。

- 1変種 = 1 forward（生成なし）。元の答えで測るため構成上 fixed-target。
- 出力ランキングは results.json の `cot_top_k_words` / `_cot.pt` の
  `word_scores` と同じ `{"word": str, "score": float}` スキーマ（降順）。

設計メモ: docs/dev_notes_06_attribution.md
"""

import re
import string
from dataclasses import dataclass, field

import torch

# lrp/analyzer.py:_find_answer_pattern と同一の回答パターン・同一の
# 「最初にマッチしたパターンの最後のマッチを採用」規約
# (scripts/rebuttal/run_fixed_target_attribution.py と同じコピー)。
ANSWER_PATTERNS: list[tuple[str, str]] = [
    (r"[Tt]he\s+answer\s+is[:\s]*\(([A-Ja-j])\)", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"[Aa]nswer[:\s]+\(([A-Ja-j])\)", "choice"),
    (r"[Aa]nswer[:\s]+([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"\*\*\(([A-Ja-j])\)\*\*", "choice"),
    (r"\*\*([A-Ja-j])\*\*", "choice"),
    (r"(?:correct|right)\s+(?:answer|option)\s+is[:\s]*\(?([A-Ja-j])\)?", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Tt]he\s+answer\s+is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"####\s*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Aa]nswer[:\s]+(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"(?:^|\n)\s*\(?([A-Ja-j])\)?\s*\.?\s*$", "choice"),
]

# 語タイプのキーから剥がす端句読点（ASCII + よく出る Unicode 引用符・ダッシュ）
EDGE_PUNCT = string.punctuation + "“”‘’…—–´`«»„"


@dataclass
class CotSplit:
    """生成テキストの3分割: CoT本文 / 答えトリガー / 答え文字列."""

    cot_text: str
    trigger_text: str
    answer_text: str
    pattern_type: str


@dataclass
class WordType:
    """CoT テキスト中の語タイプ（正規化キーと全出現スパン）."""

    word: str
    spans: list[tuple[int, int]] = field(default_factory=list)


def split_generated_text(generated_text: str) -> CotSplit | None:
    """生成テキストを (CoT, 答えトリガー, 答え) に分割する.

    lrp/analyzer.py と同じ ANSWER_PATTERNS で最終回答を特定し、
    - cot_text: 回答パターン開始位置より前の全テキスト
    - trigger_text: パターン開始〜答え文字列直前（例: "The answer is "）
    - answer_text: パターンの group(1)（例: "18", "B"）
    を返す。回答パターンが無い場合は None。
    """
    for pattern, ptype in ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, generated_text))
        if matches:
            m = matches[-1]
            s, e = m.span(1)
            return CotSplit(
                cot_text=generated_text[: m.start()],
                trigger_text=generated_text[m.start() : s],
                answer_text=generated_text[s:e],
                pattern_type=ptype,
            )
    return None


def normalize_word(word: str) -> str:
    """語タイプ比較用の正規化: 端句読点を剥がす（純句読点はそのまま）."""
    core = word.strip(EDGE_PUNCT)
    return core if core else word


def extract_word_types(cot_text: str) -> list[WordType]:
    """CoT テキストから語タイプ（ユニーク語 + 全出現スパン）を抽出する.

    空白区切りチャンクの端句読点を剥がしたものをキーとする（case-sensitive）。
    剥がすと空になる純句読点チャンク（"=", "-" 等の演算語）はチャンク全体をキーとする。
    スパンはキー部分のみを指す（削除時に句読点を保存するため）。
    """
    types: dict[str, WordType] = {}
    for m in re.finditer(r"\S+", cot_text):
        chunk = m.group(0)
        core = chunk.strip(EDGE_PUNCT)
        if core:
            lead = len(chunk) - len(chunk.lstrip(EDGE_PUNCT))
            span = (m.start() + lead, m.start() + lead + len(core))
            key = core
        else:
            span = (m.start(), m.end())
            key = chunk
        wt = types.get(key)
        if wt is None:
            wt = WordType(word=key)
            types[key] = wt
        wt.spans.append(span)
    return list(types.values())


def delete_word_type(cot_text: str, word_type: WordType) -> str:
    """語タイプの全出現を削除した変種 CoT を作る.

    出現スパン（チャンク単位で記録済み）のみを削除するため、
    他の語の部分文字列を壊すことはない。削除後の多重スペースは1つに詰める。
    """
    text = cot_text
    for s, e in sorted(word_type.spans, reverse=True):
        text = text[:s] + text[e:]
    return re.sub(r"[ \t]{2,}", " ", text)


def build_loo_variants(cot_text: str) -> tuple[list[WordType], list[str]]:
    """全語タイプの LOO 変種を作る（変種数 = 語タイプ数）."""
    word_types = extract_word_types(cot_text)
    variants = [delete_word_type(cot_text, wt) for wt in word_types]
    return word_types, variants


# ============================================================
# teacher-forcing log-prob
# ============================================================


def _encode(tokenizer, text: str) -> list[int]:
    """トークン ID のリストを返す（tensor 戻りにも対応）."""
    ids = tokenizer(text)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.flatten().tolist()
    return list(ids)


def _model_device(model) -> torch.device:
    """モデルのデバイスを推定する（パラメータ→バッファ→CPU の順）."""
    for tensors in (model.parameters(), model.buffers()):
        for t in tensors:
            return t.device
    return torch.device("cpu")


def _target_start(full_ids: list[int], ctx_ids: list[int]) -> int:
    """答えトークン列の開始インデックス（最長共通接頭辞、最低1）.

    トークン境界がマージされる場合は保守的に共通接頭辞の直後から
    （= マージされた境界トークンも答え側としてスコア）。
    """
    n = min(len(full_ids), len(ctx_ids))
    i = 0
    while i < n and full_ids[i] == ctx_ids[i]:
        i += 1
    return max(1, i)


def sequence_logprob(
    model,
    tokenizer,
    context: str,
    target: str,
    device: torch.device | None = None,
) -> float:
    """context を条件とした target トークン列の log-prob 合計（1 forward）."""
    return batched_answer_logprobs(
        model, tokenizer, [context], target, batch_size=1, device=device
    )[0]


def batched_answer_logprobs(
    model,
    tokenizer,
    contexts: list[str],
    target: str,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> list[float]:
    """複数 context に対する同一 target の log-prob 合計をバッチ計算する.

    context ごとに長さが異なるため右パディング + attention_mask で処理する
    （causal LM では右パディングは非パッド位置の logits に影響しない）。
    """
    if not contexts:
        return []
    if device is None:
        device = _model_device(model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    encoded: list[tuple[list[int], int]] = []
    for ctx in contexts:
        full_ids = _encode(tokenizer, ctx + target)
        ctx_ids = _encode(tokenizer, ctx)
        encoded.append((full_ids, _target_start(full_ids, ctx_ids)))

    results: list[float] = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i : i + batch_size]
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, (ids, _) in enumerate(batch):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        for j, (ids, start) in enumerate(batch):
            if start >= len(ids):
                results.append(0.0)
                continue
            # 位置 t-1 の logits が位置 t のトークンを予測する
            rows = logits[j, start - 1 : len(ids) - 1].float()
            logp = torch.log_softmax(rows, dim=-1)
            targets = torch.tensor(ids[start:], dtype=torch.long, device=logp.device)
            results.append(float(logp.gather(1, targets[:, None]).sum()))
        del logits
    return results


# ============================================================
# サンプル単位の LOO スコアリング
# ============================================================


def score_sample_loo(
    model,
    tokenizer,
    prompt: str,
    generated_text: str,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict | None:
    """1サンプルの全語タイプ LOO スコアリング.

    Returns:
        {
          "word_scores": [{"word", "score"}, ...]  # 降順・R_C ランキング互換,
          "word_types": [{"word", "score", "n_occurrences", "variant_logprob"}, ...],
          "base_logprob": float,
          "answer_text" / "trigger_text" / "pattern_type": str,
          "n_word_types": int,
        }
        回答パターンが無い場合は None。
    """
    split = split_generated_text(generated_text)
    if split is None:
        return None

    word_types, variants = build_loo_variants(split.cot_text)
    base_context = prompt + split.cot_text + split.trigger_text
    contexts = [base_context] + [prompt + v + split.trigger_text for v in variants]
    logprobs = batched_answer_logprobs(
        model, tokenizer, contexts, split.answer_text, batch_size=batch_size, device=device
    )
    base_logprob, variant_logprobs = logprobs[0], logprobs[1:]

    details = [
        {
            "word": wt.word,
            "score": float(base_logprob - lp),
            "n_occurrences": len(wt.spans),
            "variant_logprob": float(lp),
        }
        for wt, lp in zip(word_types, variant_logprobs, strict=True)
    ]
    details_sorted = sorted(details, key=lambda d: d["score"], reverse=True)
    return {
        "word_scores": [
            {"word": d["word"], "score": d["score"]} for d in details_sorted
        ],
        "word_types": details_sorted,
        "base_logprob": float(base_logprob),
        "answer_text": split.answer_text,
        "trigger_text": split.trigger_text,
        "pattern_type": split.pattern_type,
        "n_word_types": len(word_types),
    }


# ============================================================
# R_C ランキング互換ユーティリティ
# ============================================================


def rc_word_ranking_from_cot_pt(data: dict) -> list[dict]:
    """`{id}_cot.pt` の word_scores から CoT 領域の語ランキングを抽出する.

    Args:
        data: torch.load した `_cot.pt` 辞書
            (word_scores: [{"word","score","token_indices"}], cot_token_start/end)

    Returns:
        [{"word": str, "score": float}, ...] スコア降順（CoT 領域内のみ）
    """
    start = data.get("cot_token_start")
    end = data.get("cot_token_end")
    ranking = []
    for ws in data.get("word_scores", []):
        indices = ws.get("token_indices") or []
        if start is not None and end is not None:
            if not any(start <= i <= end for i in indices):
                continue
        ranking.append({"word": ws["word"], "score": float(ws["score"])})
    ranking.sort(key=lambda d: d["score"], reverse=True)
    return ranking


def loo_jaccard_topk(
    ranking1: list[dict],
    ranking2: list[dict],
    k: int = 10,
) -> float:
    """2つの word/score ランキング間の Top-k Jaccard（語タイプ正規化つき）.

    LOO vs R_C、または clean-LOO vs perturbed-LOO（LOO版 Jaccard@10）の
    どちらの比較にも使える。既存 metrics.top_k_jaccard_by_token と同一規約
    （タイプ dedup は最大スコア採用）。
    """
    from typo_cot.analysis.metrics import top_k_jaccard_by_token

    if not ranking1 or not ranking2:
        return 0.0
    tokens1 = [normalize_word(d["word"]) for d in ranking1]
    scores1 = [float(d["score"]) for d in ranking1]
    tokens2 = [normalize_word(d["word"]) for d in ranking2]
    scores2 = [float(d["score"]) for d in ranking2]
    return top_k_jaccard_by_token(tokens1, scores1, tokens2, scores2, k=k)

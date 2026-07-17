"""実験6-(iv): leave-one-out (LOO) 重要度スコアラ.

帰属（attribution）という枠組みを使わない削除ベースの語重要度:
CoT 中の語を削除した変種 CoT を作り、(質問プロンプト + 変種 CoT + 答えトリガー)
を teacher-forcing して「元の答えトークン列の log-prob 合計」の低下量を
その語の重要度とする。

削除単位（deletion_mode、ユーザー決定 2026-07-14）:
- "occurrence"（主定義・デフォルト、案B）: 出現1つを削除して1変種。
  語タイプの重要度 = 出現スコアの平均（Li et al. 2016 準拠。max も副次保存）。
- "type"（感度分析、案A）: 同一タイプの全出現を一括削除して1変種
  （type-level erasure。冗長な再言及を遮断する反実仮想）。

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


def delete_spans(cot_text: str, spans: list[tuple[int, int]]) -> str:
    """指定スパン群を削除した変種 CoT を作る.

    出現スパン（チャンク単位で記録済み）のみを削除するため、
    他の語の部分文字列を壊すことはない。削除後の多重スペースは1つに詰める。
    """
    text = cot_text
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + text[e:]
    return re.sub(r"[ \t]{2,}", " ", text)


def delete_word_type(cot_text: str, word_type: WordType) -> str:
    """語タイプの全出現を一括削除した変種 CoT を作る（案A: type-level erasure）."""
    return delete_spans(cot_text, word_type.spans)


def build_loo_variants(cot_text: str) -> tuple[list[WordType], list[str]]:
    """全語タイプの一括削除 LOO 変種を作る（案A、変種数 = 語タイプ数）."""
    word_types = extract_word_types(cot_text)
    variants = [delete_word_type(cot_text, wt) for wt in word_types]
    return word_types, variants


def build_loo_variants_occurrence(
    cot_text: str,
) -> tuple[list[WordType], list[int], list[str]]:
    """出現ごとの LOO 変種を作る（案B、変種数 = 出現数合計）.

    Returns:
        (word_types, occ_type_idx, variants)
        occ_type_idx[i] = variants[i] が属する word_types のインデックス。
        タイプ内の変種順は出現スパン順。
    """
    word_types = extract_word_types(cot_text)
    occ_type_idx: list[int] = []
    variants: list[str] = []
    for ti, wt in enumerate(word_types):
        for span in wt.spans:
            occ_type_idx.append(ti)
            variants.append(delete_spans(cot_text, [span]))
    return word_types, occ_type_idx, variants


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
    deletion_mode: str = "occurrence",
) -> dict | None:
    """1サンプルの全語タイプ LOO スコアリング.

    Args:
        deletion_mode: "occurrence"（案B・デフォルト。出現ごと削除→タイプ平均、
            max は score_max に併記）または "type"（案A。全出現一括削除）。

    Returns:
        {
          "word_scores": [{"word", "score"}, ...]  # 降順・R_C ランキング互換,
          "word_types": [{"word", "score", "score_max", "n_occurrences",
                          "occurrence_scores", "variant_logprobs"}, ...],
          "base_logprob": float,
          "answer_text" / "trigger_text" / "pattern_type": str,
          "n_word_types": int,
          "n_variants": int,          # occurrence: 出現数合計 / type: タイプ数
          "deletion_mode": str,
          "aggregation": "mean" | "whole_type",
        }
        回答パターンが無い場合は None。
    """
    if deletion_mode not in ("occurrence", "type"):
        raise ValueError(f"unknown deletion_mode: {deletion_mode!r}")
    split = split_generated_text(generated_text)
    if split is None:
        return None

    if deletion_mode == "occurrence":
        word_types, occ_type_idx, variants = build_loo_variants_occurrence(
            split.cot_text
        )
        aggregation = "mean"
    else:
        word_types, variants = build_loo_variants(split.cot_text)
        occ_type_idx = list(range(len(word_types)))
        aggregation = "whole_type"

    base_context = prompt + split.cot_text + split.trigger_text
    contexts = [base_context] + [prompt + v + split.trigger_text for v in variants]
    logprobs = batched_answer_logprobs(
        model, tokenizer, contexts, split.answer_text, batch_size=batch_size, device=device
    )
    base_logprob, variant_logprobs = logprobs[0], logprobs[1:]

    # タイプごとに変種スコアを集約（type モードでは1タイプ=1変種）
    per_type_scores: list[list[float]] = [[] for _ in word_types]
    per_type_logprobs: list[list[float]] = [[] for _ in word_types]
    for ti, lp in zip(occ_type_idx, variant_logprobs, strict=True):
        per_type_scores[ti].append(float(base_logprob - lp))
        per_type_logprobs[ti].append(float(lp))

    details = [
        {
            "word": wt.word,
            "score": sum(scores) / len(scores),  # occurrence: 出現平均 / type: 単一値
            "score_max": max(scores),
            "n_occurrences": len(wt.spans),
            "occurrence_scores": scores,
            "variant_logprobs": lps,
        }
        for wt, scores, lps in zip(
            word_types, per_type_scores, per_type_logprobs, strict=True
        )
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
        "n_variants": len(variants),
        "deletion_mode": deletion_mode,
        "aggregation": aggregation,
    }


# ============================================================
# R_C ランキング互換ユーティリティ
# ============================================================


# tokens_to_words (lrp/analyzer.py) と同じ特殊トークン + 主要モデルの BOS/EOS
_SPECIAL_TOKENS = frozenset(
    {"<s>", "</s>", "<pad>", "<unk>", "<bos>", "<eos>", "[CLS]", "[SEP]", "[PAD]"}
)
_BYTE_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def word_scores_degenerate(data: dict) -> bool:
    """`_cot.pt` の word_scores がトークン結合不良で潰れているかを判定する.

    アーカイブ生成側 (lrp/analyzer.py tokens_to_words) は先頭スペース / "▁" で
    語境界を検出するが、Mistral アーカイブの token_scores は空白マーカーの無い
    トークン文字列のため全トークンが1語に結合されている (2026-07 本番で確認)。
    実語が 16 トークンを超えることは無いため、それを閾値に検出する。
    """
    word_scores = data.get("word_scores") or []
    if not word_scores:
        return bool(data.get("token_scores"))
    max_span = max(len(w.get("token_indices") or []) for w in word_scores)
    return max_span >= 16


def align_tokens_to_text(
    tokens: list[str], text: str
) -> list[tuple[int, int] | None] | None:
    """トークン文字列列を text に貪欲整合し、各トークンの文字スパンを返す.

    - 特殊トークン (<s> 等) は None
    - `<0xNN>` バイトトークンは該当文字に展開 (ASCII のみ)、"▁" はスペース扱い
    - トークンが空白マーカーを持たない場合 (Mistral アーカイブ) は text 側の
      空白をスキップして照合する
    - 整合に失敗したら None (呼び出し側でフォールバック)
    """
    spans: list[tuple[int, int] | None] = []
    pos = 0
    for tok in tokens:
        if tok in _SPECIAL_TOKENS:
            spans.append(None)
            continue
        m = _BYTE_TOKEN_RE.fullmatch(tok)
        piece = chr(int(m.group(1), 16)) if m else tok.replace("▁", " ")
        if text.startswith(piece, pos):
            spans.append((pos, pos + len(piece)))
            pos += len(piece)
            continue
        j = pos
        while j < len(text) and text[j].isspace():
            j += 1
        if text.startswith(piece, j):
            spans.append((j, j + len(piece)))
            pos = j + len(piece)
            continue
        stripped = piece.lstrip()
        if stripped and text.startswith(stripped, j):
            spans.append((j, j + len(stripped)))
            pos = j + len(stripped)
            continue
        if not stripped:
            # 空白のみのトークンが text と一致しない場合は幅0で読み飛ばす
            spans.append((pos, pos))
            continue
        return None
    return spans


def rc_word_ranking_from_token_scores(
    data: dict, full_text: str
) -> list[dict] | None:
    """token_scores を full_text に整合し、CoT 領域の語ランキングを再構築する.

    word_scores が結合不良のアーカイブ (Mistral) 用のフォールバック。語の単位は
    full_text の空白区切りチャンク (extract_word_types と同じ規約)、スコアは
    チャンクに重なる CoT 領域内トークンの relevance 合計 (tokens_to_words の
    合計規約 + cot_filtered_relevance の領域ゼロ埋めと同値)。

    Args:
        data: torch.load した `_cot.pt` 辞書 (token_scores: [(token, score)])
        full_text: トークン列が表すテキスト (prompt + generated_text)

    Returns:
        [{"word","score"}] スコア降順。整合失敗時は None。
    """
    token_scores = data.get("token_scores") or []
    tokens = [t for t, _ in token_scores]
    scores = [float(s) for _, s in token_scores]
    spans = align_tokens_to_text(tokens, full_text)
    if spans is None:
        return None
    start = data.get("cot_token_start")
    end = data.get("cot_token_end")
    aligned = [(i, sp) for i, sp in enumerate(spans) if sp is not None]
    ranking = []
    ti = 0
    for m in re.finditer(r"\S+", full_text):
        s, e = m.span()
        while ti < len(aligned) and aligned[ti][1][1] <= s:
            ti += 1
        indices = []
        tj = ti
        while tj < len(aligned) and aligned[tj][1][0] < e:
            if aligned[tj][1][1] > s:
                indices.append(aligned[tj][0])
            tj += 1
        if not indices:
            continue
        if start is not None and end is not None:
            indices = [i for i in indices if start <= i <= end]
            if not indices:
                continue
        ranking.append(
            {"word": m.group(0), "score": float(sum(scores[i] for i in indices))}
        )
    ranking.sort(key=lambda d: d["score"], reverse=True)
    return ranking


def rc_word_ranking_from_cot_pt(data: dict, full_text: str | None = None) -> list[dict]:
    """`{id}_cot.pt` の word_scores から CoT 領域の語ランキングを抽出する.

    Args:
        data: torch.load した `_cot.pt` 辞書
            (word_scores: [{"word","score","token_indices"}], cot_token_start/end)
        full_text: トークン列が表すテキスト (prompt + generated_text)。指定時、
            word_scores が結合不良 (Mistral アーカイブ) なら token_scores から
            語ランキングを再構築する。

    Returns:
        [{"word": str, "score": float}, ...] スコア降順（CoT 領域内のみ）
    """
    if full_text is not None and word_scores_degenerate(data):
        rebuilt = rc_word_ranking_from_token_scores(data, full_text)
        if rebuilt is not None:
            return rebuilt
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


def expand_multiword_entries(ranking: list[dict]) -> list[dict]:
    """空白（改行含む）を内包する結合語エントリを構成語に分解する.

    R_C 側 `word_scores` は `tokens_to_words` の仕様で改行をまたいで語が
    結合されることがある（例: "dollars.\\nThe"）。LOO 側の空白区切り語タイプ
    とは単位が合わず Jaccard の偽不一致になるため、比較前に空白で分解し、
    各構成語に親エントリのスコアを引き継がせる。単一語エントリはそのまま。
    """
    out: list[dict] = []
    for d in ranking:
        parts = str(d["word"]).split()
        if len(parts) <= 1:
            out.append(d)
        else:
            out.extend({"word": p, "score": d["score"]} for p in parts)
    return out


def loo_jaccard_topk(
    ranking1: list[dict],
    ranking2: list[dict],
    k: int = 10,
) -> float:
    """2つの word/score ランキング間の Top-k Jaccard（語タイプ正規化つき）.

    LOO vs R_C、または clean-LOO vs perturbed-LOO（LOO版 Jaccard@10）の
    どちらの比較にも使える。既存 metrics.top_k_jaccard_by_token と同一規約
    （タイプ dedup は最大スコア採用）。比較前に両側とも
    `expand_multiword_entries`（改行またぎ結合語の分解）と
    `normalize_word`（端句読点剥がし）で正規化する。
    """
    from typo_cot.analysis.metrics import top_k_jaccard_by_token

    ranking1 = expand_multiword_entries(ranking1)
    ranking2 = expand_multiword_entries(ranking2)
    if not ranking1 or not ranking2:
        return 0.0
    tokens1 = [normalize_word(d["word"]) for d in ranking1]
    scores1 = [float(d["score"]) for d in ranking1]
    tokens2 = [normalize_word(d["word"]) for d in ranking2]
    scores2 = [float(d["score"]) for d in ranking2]
    return top_k_jaccard_by_token(tokens1, scores1, tokens2, scores2, k=k)


def compute_loo_jaccard_pairs(
    clean_entries: list[dict],
    perturbed_entries: list[dict],
    k: int = 10,
) -> list[dict]:
    """clean / perturbed の LOO ランキングを sample_id で対応付けて Jaccard@k を計算する.

    LOO 版 CoT:Jaccard@k（内的軸の帰属フリー再構成）の本体。
    run_loo_scoring.py の results.json エントリ
    （sample_id / loo_word_scores）をそのまま受け取れる。

    Returns:
        [{"sample_id": str, "loo_jaccard": float}, ...]（clean 側の順序、
        片側にしか無いサンプルはスキップ）
    """
    perturbed_by_id = {e["sample_id"]: e for e in perturbed_entries}
    pairs: list[dict] = []
    for clean in clean_entries:
        pert = perturbed_by_id.get(clean["sample_id"])
        if pert is None:
            continue
        pairs.append(
            {
                "sample_id": clean["sample_id"],
                "loo_jaccard": loo_jaccard_topk(
                    clean.get("loo_word_scores", []),
                    pert.get("loo_word_scores", []),
                    k=k,
                ),
            }
        )
    return pairs

"""実験15: 早期層 patch を保持したまま CoT 全体を自由生成する介入の中核.

実験8 (`patching.py`) の `PatchInjector` は「答えトークンでの測定」用に、
recipient run の prefill 時に患部 (摂動語スパン) 位置へ donor 活性を注入し、
generate の decode ステップ (系列長 1) では自動的に無効化する設計だった。

本実験はこの hook を **CoT 自由生成** に転用する:

    recipient = typo 質問プロンプト (質問のみ、CoT は与えない)
    早期窓の residual を摂動語スパン位置で clean 値に patch
    → prefill で 1 回注入され、以降の greedy decode は KV キャッシュ経由で
      その効果を保持したまま自然生成される (CoT 全体を生成)

つまり「hook を生成ループ全体で維持」する要件は、`PatchInjector` を
`model.generate` 全体を囲うコンテキストマネージャとして使い、患部が prompt 内に
あるため prefill での 1 回注入で足りる、という形で満たされる (decode 追加注入は
不要 — 患部は新規生成トークンではない)。

このモジュールは:

- **純関数** (GPU 非依存): 発散オンセット `divergence_index`、質問スパンの
  文字位置探索 `locate_word_char_spans`、clean/typo スパンのトークン位置整列
  `align_span_positions`、CoT ROUGE-L の薄いラッパ `cot_rouge_l` (定義は既存の
  `analysis.metrics.rouge_l_score` = 文字単位 LCS、論文 Table 6 と同一)。
- **生成ヘルパ**: モデルを引数注入する `generate_ids` / `generate_ids_patched`。

設計の詳細は docs/dev_notes_15_patch_freegen.md を参照。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from typo_cot.analysis.metrics import rouge_l_score
from typo_cot.intervention.patching import PatchInjector


# ---------------------------------------------------------------------------
# CoT ROUGE-L (既存実装の再利用)
# ---------------------------------------------------------------------------


def cot_rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """CoT の ROUGE-L (precision/recall/f1) を返す.

    既存の `analysis.metrics.rouge_l_score` (文字単位 LCS。論文 Table 6 の
    `cot_rouge_l_f1` と同一定義) をそのまま用いる。新定義は導入しない。
    """
    return rouge_l_score(reference, hypothesis)


def cot_rouge_l_f(reference: str, hypothesis: str) -> float:
    """CoT ROUGE-L の F1 のみを返す便宜関数."""
    return cot_rouge_l(reference, hypothesis)["f1"]


# ---------------------------------------------------------------------------
# 発散オンセット (生成トークン列の最初の分岐)
# ---------------------------------------------------------------------------


def divergence_index(a: Sequence[int], b: Sequence[int]) -> int | None:
    """2 つの生成トークン列 a, b が最初に分岐するインデックスを返す.

    一方が他方の接頭辞 (同一含む) の場合は None (= 分岐なし)。自由生成では
    a = clean 生成, b = patched/typo 生成 とし、`None` は「clean と完全一致 =
    発散オンセット消失」を意味する。
    """
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None


# ---------------------------------------------------------------------------
# 摂動語スパンの位置整列 (exp8 の question_span 整列を自由生成用に純化)
# ---------------------------------------------------------------------------


def locate_word_char_spans(
    prompt: str, question: str, words: Sequence[str]
) -> list[tuple[int, int] | None]:
    """プロンプト内の質問領域で words を順に探し、文字スパンのリストを返す.

    exp8 `run_patching._locate_spans` と同じ規約: 質問先頭 80 文字の最終出現を
    起点に語を順探索し、見つからない語 (空語含む) は None。長さは len(words) 維持。
    """
    q_start = prompt.rfind(question[:80]) if question else -1
    search_from = q_start if q_start >= 0 else 0
    spans: list[tuple[int, int] | None] = []
    cursor = search_from
    for w in words:
        if not w:
            spans.append(None)
            continue
        pos = prompt.find(w, cursor)
        if pos < 0:
            pos = prompt.find(w, search_from)  # 順序制約を緩める
        if pos < 0:
            spans.append(None)
        else:
            spans.append((pos, pos + len(w)))
            cursor = pos + len(w)
    return spans


@dataclass
class AlignedSpans:
    """clean/typo の摂動語スパンのトークン位置対応 (語 i ↔ 語 i).

    Attributes:
        clean_positions: 両側で見つかった語の clean 側トークン位置
        pert_positions:  同 typo 側トークン位置 (clean_positions と同長・同順)
        n_words: 入力語数
        n_dropped: どちらか一方で見つからず落とした語数
    """

    clean_positions: list[int]
    pert_positions: list[int]
    n_words: int
    n_dropped: int


def align_span_positions(
    clean_tok_positions: Sequence[int | None],
    pert_tok_positions: Sequence[int | None],
) -> AlignedSpans:
    """語ごとの (clean 側, typo 側) トークン位置を突き合わせ、両側で有効な語のみ残す.

    どちらかが None の語は落とす (donor/recipient で行数を一致させるため)。

    Raises:
        ValueError: 2 つのリスト長が異なる場合
    """
    if len(clean_tok_positions) != len(pert_tok_positions):
        raise ValueError(
            f"語数不一致: clean={len(clean_tok_positions)} pert={len(pert_tok_positions)}"
        )
    clean_pos: list[int] = []
    pert_pos: list[int] = []
    n_dropped = 0
    for c, p in zip(clean_tok_positions, pert_tok_positions, strict=True):
        if c is None or p is None:
            n_dropped += 1
            continue
        clean_pos.append(c)
        pert_pos.append(p)
    return AlignedSpans(
        clean_positions=clean_pos,
        pert_positions=pert_pos,
        n_words=len(clean_tok_positions),
        n_dropped=n_dropped,
    )


# ---------------------------------------------------------------------------
# 生成ヘルパ (モデル引数注入)
# ---------------------------------------------------------------------------


def _resolve_pad_id(model: nn.Module, pad_id: int | None) -> int | None:
    if pad_id is not None:
        return pad_id
    gc = getattr(model, "generation_config", None)
    pad = getattr(gc, "pad_token_id", None) if gc is not None else None
    if pad is None and gc is not None:
        pad = getattr(gc, "eos_token_id", None)
    if isinstance(pad, (list, tuple)):
        pad = pad[0] if pad else None
    return pad


def generate_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    pad_id: int | None = None,
) -> list[int]:
    """greedy 生成し、**継続部分** (プロンプト以降) のトークン ID 列を返す (batch=1)."""
    pad = _resolve_pad_id(model, pad_id)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
    return out[0, input_ids.shape[1] :].tolist()


def generate_ids_patched(
    model: nn.Module,
    layers: Sequence[nn.Module],
    input_ids: torch.Tensor,
    site: str,
    layer_indices: Sequence[int],
    dst_positions: Sequence[int],
    values: dict[int, torch.Tensor],
    max_new_tokens: int,
    pad_id: int | None = None,
) -> list[int]:
    """patch (donor 活性の注入) を **生成ループ全体で維持** したまま greedy 生成する.

    `PatchInjector` は `model.generate` 全体を囲い、prefill (系列長 > max(dst))
    で患部位置に donor 値を注入する。decode ステップ (系列長 1) では自動無効で、
    患部の効果は KV キャッシュ経由で保持される。継続部分のトークン ID 列を返す。
    """
    with PatchInjector(layers, site, layer_indices, dst_positions, values):
        return generate_ids(model, input_ids, max_new_tokens, pad_id=pad_id)

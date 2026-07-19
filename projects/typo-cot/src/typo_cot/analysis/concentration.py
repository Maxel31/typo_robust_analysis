"""実験13: 読み出し集中度(M3測定) — 集中度指標.

答え段が CoT のどれだけ狭い集合に依存するかを測る集中度メトリクス:

- LOO 集中度: 1サンプルの LOO 重要度分布 (loo_word_scores の score) から
  Gini係数 / top-1シェア / top-4シェア / 有効語数 (participation ratio) を算出。
  負の LOO スコア (削除で答え logprob が上昇 = 重要でない) は 0 にクリップして
  「正の重要度質量」の集中度を測る (clip_negative=True がデフォルト)。

- attention 集中度代理: 答えトークン位置の行 → CoT トークン列への attention 質量の
  分布 (answer_to_cot_distribution) の Gini。forward 1回で得られる安価な代理で、
  全設定に適用して LOO Gini と突合し妥当性を検証する。

集中度が高い (Gini/top1 が大きい) ほど答え段が少数の CoT 語に依存し、
削除介入が効きやすい、というのが H13。

入出力は in-memory の list / numpy 配列 / attention テンソルのみで、
データファイルの読み書きは行わない (ドライバスクリプト側で JSON 入出力する)。
"""

from __future__ import annotations

import string

import numpy as np

# loo_scorer.normalize_word と同一の端句読点集合 (ASCII + Unicode 引用符・ダッシュ)
EDGE_PUNCT = string.punctuation + "“”‘’…—–´`«»„"


def _as_nonneg(values, clip_negative: bool = True) -> np.ndarray:
    """有限化 (nan/inf→0) し、clip_negative なら負を 0 にクリップした配列を返す."""
    a = np.asarray(list(values), dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    if clip_negative:
        a = np.clip(a, 0.0, None)
    return a


def gini(values, clip_negative: bool = True) -> float:
    """非負分布の Gini係数.

    一様分布=0、完全集中 (1要素に全質量) = (n-1)/n。スケール不変。
    空配列は nan、単一要素・全ゼロ・全負(クリップ後全ゼロ)は 0。
    ソート済み配列に対する標準式:
        G = 2*Σ(i*x_(i)) / (n*Σx) − (n+1)/n   (x昇順, i=1..n)
    """
    a = _as_nonneg(values, clip_negative)
    n = a.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    s = a.sum()
    if s <= 0:
        return 0.0
    a_sorted = np.sort(a)
    idx = np.arange(1, n + 1)
    g = (2.0 * float(np.sum(idx * a_sorted)) / (n * s)) - (n + 1.0) / n
    # 数値誤差で [0,1] をわずかに外れることがあるためクリップ
    return float(min(1.0, max(0.0, g)))


def top1_share(values, clip_negative: bool = True) -> float:
    """最大要素が占めるシェア max/Σ. 空・全ゼロは 0."""
    a = _as_nonneg(values, clip_negative)
    s = a.sum()
    if a.size == 0 or s <= 0:
        return 0.0
    return float(a.max() / s)


def topk_share(values, k: int, clip_negative: bool = True) -> float:
    """上位 k 要素が占めるシェア. k がサイズ超過なら全和 (=1)."""
    a = _as_nonneg(values, clip_negative)
    s = a.sum()
    if a.size == 0 or s <= 0:
        return 0.0
    top = np.sort(a)[::-1][: max(0, k)]
    return float(top.sum() / s)


def effective_count(values, clip_negative: bool = True) -> float:
    """有効語数 (participation ratio / inverse Simpson): (Σx)^2 / Σx^2.

    一様n語=n、完全集中=1。集中度の逆向き指標 (小さいほど集中)。
    """
    a = _as_nonneg(values, clip_negative)
    s = a.sum()
    if a.size == 0 or s <= 0:
        return 0.0
    ssq = float(np.sum(a * a))
    if ssq <= 0:
        return 0.0
    return float((s * s) / ssq)


def loo_sample_concentration(
    loo_word_scores: list[dict], clip_negative: bool = True
) -> dict:
    """1サンプルの LOO 重要度分布から集中度メトリクスを算出する.

    Args:
        loo_word_scores: [{"word": str, "score": float}, ...] (run_loo_scoring の
            results.json エントリの loo_word_scores)。
        clip_negative: 負の重要度を 0 にクリップするか (デフォルト True)。

    Returns:
        {"n_words", "gini", "top1_share", "top4_share", "effective_count"}
        空入力では gini=nan, その他 0/0.0。
    """
    scores = [float(w["score"]) for w in loo_word_scores]
    return {
        "n_words": len(scores),
        "gini": gini(scores, clip_negative),
        "top1_share": top1_share(scores, clip_negative),
        "top4_share": topk_share(scores, 4, clip_negative),
        "effective_count": effective_count(scores, clip_negative),
    }


# ---------------------------------------------------------------
# 内容語限定の集中度 (exp2 削除RD の content 層と同一の語区分)
# ---------------------------------------------------------------

# run_loo_scoring.py と同一の演算語集合 (numeric 層に区分)
_OPERATOR_WORDS = frozenset({"=", "+", "-", "*", "/", "×", "÷", "%"})

# exp2 target_selector.STOPWORDS と同一 (機能語判定用の固定リスト)
_STOPWORDS = frozenset(
    """
    a an the this that these those it its he she him his her hers they them their
    theirs i you we me my mine your yours our ours us
    is are was were be been being am do does did done has have had having
    will would shall should can could may might must
    of in on at by for with to from as and or but nor not no yes than
    if then else so too very each per every all any some most more less much many
    few both either neither there here when where which who whom whose what why how
    also just only even still again once about into over under after before between
    during through against out up down off above below further
    s t d ll ve re m don didn doesn isn aren wasn weren won shan hasn haven hadn
    """.split()
)


def _strip_edge_punct(word: str) -> str:
    core = word.strip(EDGE_PUNCT)
    return core if core else word


def word_stratum(word: str) -> str:
    """語の層を判定する: "numeric" / "content" / "other" (exp2 と同一規約).

    - numeric: 数字を含む語、または演算語 (= + - * / × ÷ %)
    - content: 正規化コアが 2 文字以上 (選択肢文字 A-J を自動排除)、英字を含み、
      ストップワードでない
    - other: 上記以外 (機能語・1文字・非英字)
    """
    core = _strip_edge_punct(word)
    if word in _OPERATOR_WORDS or core in _OPERATOR_WORDS:
        return "numeric"
    if any(ch.isdigit() for ch in core):
        return "numeric"
    if len(core) < 2:
        return "other"
    if not any(ch.isalpha() for ch in core):
        return "other"
    if core.lower() in _STOPWORDS:
        return "other"
    return "content"


def loo_content_concentration(
    loo_word_scores: list[dict], clip_negative: bool = True
) -> dict:
    """内容語 (content 層) に限定した LOO 集中度 + 内容語質量シェア.

    exp2 の content 層削除RD と同じ語区分で、内容語のみの Gini/top1 と、
    全正質量に占める内容語の質量シェアを算出する。Mistral の「削除に強いが
    観察相関は強い」= 読み出しが内容語に集中していない (分散 + 非内容語追跡)
    を定量化するための指標。

    Returns:
        {"n_content_words", "content_gini", "content_top1_share",
         "content_mass_share", "top1_stratum"}
        top1_stratum = 全語で最大 LOO スコアの語の stratum。
        content_mass_share = Σ(content の正スコア) / Σ(全語の正スコア)。
    """
    all_scores = [(w["word"], float(w["score"])) for w in loo_word_scores]
    content = [(w, s) for w, s in all_scores if word_stratum(w) == "content"]
    content_scores = [s for _, s in content]

    # 全正質量 (clip 済み) に占める content の正質量シェア
    def _pos_sum(scores):
        return float(sum(x for x in scores if x > 0)) if clip_negative else float(sum(scores))

    total_pos = _pos_sum([s for _, s in all_scores])
    content_pos = _pos_sum(content_scores)
    mass_share = (content_pos / total_pos) if total_pos > 0 else 0.0

    top1_stratum = None
    if all_scores:
        top_word = max(all_scores, key=lambda ws: ws[1])[0]
        top1_stratum = word_stratum(top_word)

    return {
        "n_content_words": len(content_scores),
        "content_gini": gini(content_scores, clip_negative),
        "content_top1_share": top1_share(content_scores, clip_negative),
        "content_mass_share": float(mass_share),
        "top1_stratum": top1_stratum,
    }


def answer_to_cot_distribution(
    attn_2d,
    answer_positions,
    cot_positions,
    reduce_answer: str = "mean",
) -> np.ndarray:
    """head-reduce 済み [seq, seq] attention から 答え行→CoT列 の分布を取り出す.

    Args:
        attn_2d: [seq, seq] の attention 行列 (attn_2d[q, k] = query q が key k に
            置く attention 質量)。head 方向は事前に平均済みを想定。
        answer_positions: 答えトークンの query 位置 (複数可)。
        cot_positions: CoT トークンの key 位置。
        reduce_answer: 複数答え位置の集約 ("mean" | "sum")。

    Returns:
        cot_positions 上の 1次元分布 (長さ len(cot_positions))。
    """
    A = np.asarray(attn_2d, dtype=float)
    ans = np.asarray(list(answer_positions), dtype=int)
    cot = np.asarray(list(cot_positions), dtype=int)
    if ans.size == 0 or cot.size == 0:
        return np.zeros(cot.size, dtype=float)
    rows = A[ans][:, cot]  # [n_ans, n_cot]
    if reduce_answer == "sum":
        return rows.sum(axis=0)
    return rows.mean(axis=0)


def attention_gini_per_layer(
    attentions,
    answer_positions,
    cot_positions,
    batch_index: int = 0,
    reduce_answer: str = "mean",
    clip_negative: bool = False,
) -> list[float]:
    """全層の 答え→CoT attention 分布の Gini を層ごとに返す.

    Args:
        attentions: HuggingFace `output_attentions=True` の戻り値と同形の
            レイヤ列。各要素は [batch, heads, seq, seq] (numpy か torch)。
        answer_positions / cot_positions: query / key の位置インデックス。
        batch_index: バッチ内のどのサンプルか。
        reduce_answer: 複数答え位置の集約。
        clip_negative: attention は非負なので通常 False。

    Returns:
        層数と同じ長さの Gini リスト。
    """
    ginis: list[float] = []
    for layer in attentions:
        if hasattr(layer, "detach"):
            arr = layer.detach().float().cpu().numpy()
        else:
            arr = np.asarray(layer)
        # [batch, heads, seq, seq] -> head 平均 -> [seq, seq]
        head_mean = arr[batch_index].mean(axis=0)
        dist = answer_to_cot_distribution(
            head_mean, answer_positions, cot_positions, reduce_answer=reduce_answer
        )
        ginis.append(gini(dist, clip_negative=clip_negative))
    return ginis

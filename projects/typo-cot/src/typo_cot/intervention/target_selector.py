"""実験2: 標的選定 (top/bottom R_C・matched-random・LOO) と語の層判定.

計画 §4 実験2-1/2:
- 標的候補 = 答え句前 prefix の語タイプのうち content 層
  (選択肢文字=単一文字・ストップワード・数値/演算語を除く)
- 数値・演算語は numeric 層として**選定・集計とも分離** (準自明性の別枠報告)
- matched_random は「頻度・文字長一致のランダム内容語」— top 標的1語ごとに
  文字長 (≤0→≤1→≤2→≤3→∞) × Zipf 頻度帯 (<0.25→<0.5→<1→<2→∞) を緩和しながら
  非復元抽出。seed は (global_seed, sample_id) から決定論的に導出する (冪等性)。

ランキング入力は R_C (`_cot.pt` word_scores / results.json cot_top_k_words) と
LOO (`loo_word_scores`) のどちらも [{"word","score"}] スキーマで受ける。
"""

import hashlib
import random
from dataclasses import dataclass

from wordfreq import zipf_frequency

from typo_cot.intervention.loo_scorer import (
    expand_multiword_entries,
    extract_word_types,
    normalize_word,
)

# run_loo_scoring.py と同一の演算語集合 (数値層に区分)
OPERATOR_WORDS = frozenset({"=", "+", "-", "*", "/", "×", "÷", "%"})

# 最小限の英語ストップワード (機能語)。内容語判定用 — 依存追加を避けた固定リスト
STOPWORDS = frozenset(
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

# matched-random の緩和スケジュール: (文字長許容 [≤, None=∞], Zipf 帯半幅 [<, None=∞])
MATCH_SCHEDULE: list[tuple[int | None, float | None]] = [
    (0, 0.25),
    (1, 0.5),
    (2, 1.0),
    (3, 2.0),
    (None, None),
]


@dataclass
class Candidate:
    """prefix 中の標的候補語タイプ.

    Attributes:
        word: 語タイプ (extract_word_types のキーと同一表記)
        stratum: "content" | "numeric"
        n_occurrences: prefix 中の出現数
        zipf: wordfreq の Zipf 頻度 (正規化コアの小文字)
        length: 正規化コアの文字長
        first_char_pos: 初出コアスパンの開始文字位置 (回復曲線の初出位置に使用)
    """

    word: str
    stratum: str
    n_occurrences: int
    zipf: float
    length: int
    first_char_pos: int


def word_stratum(word: str) -> str:
    """語の層を判定する: "numeric" / "content" / "other".

    - numeric: 数字を含む語、または演算語 (= + - * / × ÷ %)
    - content: 正規化コアが 2 文字以上 (→ 選択肢文字 A–J を自動排除)、
      英字を含み、ストップワードでない
    - other: 上記以外 (主要因計画の標的にしない)
    """
    core = normalize_word(word)
    if word in OPERATOR_WORDS or core in OPERATOR_WORDS:
        return "numeric"
    if any(ch.isdigit() for ch in core):
        return "numeric"
    if len(core) < 2:
        return "other"
    if not any(ch.isalpha() for ch in core):
        return "other"
    if core.lower() in STOPWORDS:
        return "other"
    return "content"


def build_candidates(cot_text: str) -> list[Candidate]:
    """prefix の語タイプから標的候補 (content + numeric 層) を構築する."""
    out: list[Candidate] = []
    for wt in extract_word_types(cot_text):
        stratum = word_stratum(wt.word)
        if stratum == "other":
            continue
        core = normalize_word(wt.word)
        out.append(
            Candidate(
                word=wt.word,
                stratum=stratum,
                n_occurrences=len(wt.spans),
                zipf=zipf_frequency(core.lower(), "en"),
                length=len(core),
                first_char_pos=min(s for s, _ in wt.spans),
            )
        )
    return out


def normalize_ranking(ranking: list[dict]) -> dict[str, float]:
    """[{"word","score"}] ランキングを正規化語 → 最大スコアの辞書に潰す.

    結合語 (改行またぎ等) は expand_multiword_entries で分解し、端句読点を
    剥がしてタイプ化。タイプ重複は最大スコア採用 (既存 top_k_jaccard_by_token
    と同一規約)。
    """
    scores: dict[str, float] = {}
    for d in expand_multiword_entries(ranking):
        w = normalize_word(str(d["word"]))
        s = float(d["score"])
        if w not in scores or s > scores[w]:
            scores[w] = s
    return scores


def select_top(
    ranking: list[dict] | dict[str, float],
    candidates: list[Candidate],
    k: int,
    stratum: str | None = "content",
    bottom: bool = False,
) -> list[str]:
    """ランキング上位 (bottom=True で下位 = Anti) k 語タイプを層内から選ぶ.

    候補はランキングにスコアを持つものに限る (prefix に存在しない語・層違いは
    除外)。stratum=None は**無制限選定** (数値・演算語を含む R_C 純粋上位 k 語;
    "other" 層は候補構築時点で除外済み)。返り値は選定順。候補不足時は k 未満の
    短いリストを返す (呼び出し側で腕 skip を判断する)。
    """
    scores = ranking if isinstance(ranking, dict) else normalize_ranking(ranking)
    scored: list[tuple[float, str]] = []
    for c in candidates:
        if stratum is not None and c.stratum != stratum:
            continue
        key = normalize_word(c.word)
        if key in scores:
            scored.append((scores[key], c.word))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=not bottom)
    return [w for _, w in scored[:k]]


def select_matched_random(
    top_words: list[str],
    candidates: list[Candidate],
    rng: random.Random,
    stratum: str = "content",
    exclude: set[str] | None = None,
) -> list[str]:
    """top 標的と頻度・文字長を一致させたランダム語を非復元抽出する.

    top_words の各語について MATCH_SCHEDULE を緩和しながら候補を探し、
    band 内から rng で一様抽選する。プール枯渇時は短いリストを返す。
    """
    excluded = {normalize_word(w).lower() for w in top_words}
    if exclude:
        excluded |= {normalize_word(w).lower() for w in exclude}
    pool = [
        c
        for c in candidates
        if c.stratum == stratum and normalize_word(c.word).lower() not in excluded
    ]
    by_word = {c.word: c for c in candidates}

    matched: list[str] = []
    for word in top_words:
        target = by_word.get(word)
        core = normalize_word(word)
        t_len = target.length if target else len(core)
        t_zipf = target.zipf if target else zipf_frequency(core.lower(), "en")

        chosen: Candidate | None = None
        for len_tol, zipf_tol in MATCH_SCHEDULE:
            band = [
                c
                for c in pool
                if (len_tol is None or abs(c.length - t_len) <= len_tol)
                and (zipf_tol is None or abs(c.zipf - t_zipf) < zipf_tol)
            ]
            if band:
                chosen = rng.choice(sorted(band, key=lambda c: c.word))
                break
        if chosen is None:
            continue
        matched.append(chosen.word)
        pool = [c for c in pool if c.word != chosen.word]
    return matched


def select_stratum_matched_random(
    top_words: list[str],
    candidates: list[Candidate],
    rng: random.Random,
    exclude: set[str] | None = None,
) -> list[str]:
    """無制限 top 標的の**層内マッチ**ランダム統制を非復元抽出する.

    top_words の各語について、その語自身の層 (content / numeric) と同じ層の
    プールから MATCH_SCHEDULE (文字長×Zipf 帯) を緩和しながら候補を探す —
    数値標的には数値語を、内容語には内容語をマッチさせる (2026-07-15 決定)。
    どこかの標的でプールが枯渇した場合は短いリストを返す (腕 skip 判断は
    呼び出し側)。
    """
    excluded = {normalize_word(w).lower() for w in top_words}
    if exclude:
        excluded |= {normalize_word(w).lower() for w in exclude}
    pool = [
        c
        for c in candidates
        if c.stratum in ("content", "numeric")
        and normalize_word(c.word).lower() not in excluded
    ]
    by_word = {c.word: c for c in candidates}

    matched: list[str] = []
    for word in top_words:
        target = by_word.get(word)
        core = normalize_word(word)
        t_stratum = target.stratum if target else word_stratum(word)
        t_len = target.length if target else len(core)
        t_zipf = target.zipf if target else zipf_frequency(core.lower(), "en")

        chosen: Candidate | None = None
        for len_tol, zipf_tol in MATCH_SCHEDULE:
            band = [
                c
                for c in pool
                if c.stratum == t_stratum
                and (len_tol is None or abs(c.length - t_len) <= len_tol)
                and (zipf_tol is None or abs(c.zipf - t_zipf) < zipf_tol)
            ]
            if band:
                chosen = rng.choice(sorted(band, key=lambda c: c.word))
                break
        if chosen is None:
            continue
        matched.append(chosen.word)
        pool = [c for c in pool if c.word != chosen.word]
    return matched


def rng_for_sample(seed: int, sample_id: str) -> random.Random:
    """(global_seed, sample_id) から決定論的な RNG を導出する (再実行で冪等)."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))

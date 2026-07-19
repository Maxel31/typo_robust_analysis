"""実験12: R_C top-10 語の 4 カテゴリ分類 (操作的定義).

カテゴリ (M2 測定):
    conclusion  結論句定型  … 答え定型 ("The answer is" 周辺 / 選択肢ラベル)
    numeric     数値
    content     内容語 (名詞・動詞・形容詞・副詞の代理: 機能語ストップリスト外の英字語)
    function    機能語 (冠詞・前置詞・接続詞・代名詞・助動詞・句読点のみ)

操作的定義 (POS + テンプレートマッチ):
  R_C ランキング (`rc_word_ranking_from_cot_pt`) の 1 エントリ "word" は
  tokens_to_words の仕様で空白/改行をまたいで結合されることがある
  (例 "dollars.\\nThe", "(C).\\nThe")。エントリを空白分割して構成語ごとに
  判定し、エントリの最終カテゴリは優先順位 conclusion > numeric > content >
  function で 1 つに割り当てる (各エントリはちょうど 1 カテゴリ、
  シェアは 10 スロット上で和 1.0)。

  構成語 core = 前後の句読点・記号を除去した文字列。
  1. conclusion:
     - core.lower() ∈ CONCLUSION_LEX (答え宣言語彙)
     - or MC 選択肢ラベル `^\\(?[A-Ea-e]\\)?[.:) ]*$`
     - or 答え行リード: エントリ raw に `\\n\\s*(The|So|Therefore)` を含み
       当該構成語が The/So/Therefore (改行直後の定型行開始)
  2. numeric: 数字のみ (通貨・桁区切り・小数・% 許容、英字なし)
  3. content: 英字 core (len>=2) かつ FUNCTION_STOP 外
  4. function: FUNCTION_STOP または句読点のみ/空
"""
from __future__ import annotations

import re
import string

CONCLUSION_LEX = {
    "answer", "answers", "final", "correct", "therefore", "thus", "hence",
    "conclusion", "boxed", "option", "choice", "conclude",
}

# 改行直後に来る答え行の定型リード語
LEADIN_WORDS = {"the", "so", "therefore", "thus", "hence", "final"}

# 英語機能語ストップリスト (冠詞/前置詞/接続詞/代名詞/助動詞/限定詞など)
FUNCTION_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "into", "onto", "over", "under", "as", "and", "or", "but", "nor", "so", "yet",
    "if", "then", "than", "that", "this", "these", "those", "there", "here",
    "is", "are", "was", "were", "be", "been", "being", "am", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "do", "does",
    "did", "has", "have", "had", "not", "no", "yes", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "we", "us", "our", "you", "your",
    "i", "me", "my", "him", "who", "whom", "which", "what", "when", "where",
    "why", "how", "all", "any", "each", "both", "some", "such", "only", "own",
    "up", "down", "out", "off", "about", "because", "while", "during", "per",
    "s", "t", "d", "m", "re", "ve", "ll",
}

_PUNCT = string.punctuation + "“”‘’—–…·•\t\n\r "
# MC 選択肢ラベル: 括弧付き ("(C)", "(c).") か 大文字+句読点 ("C.", "D)")。
# 素の 1 文字 ("a"/"A"/"C") は冠詞・変数と区別できないため除外。
_MC_LABEL_RE = re.compile(r"^(?:\([A-Ea-e]\)[.:)]*|[A-E][.):]+)$")
_LEADIN_RE = re.compile(r"\n\s*(The|So|Therefore|Thus|Hence|Final)\b")
_DIGIT_RE = re.compile(r"[0-9]")
_NUM_ALLOWED_RE = re.compile(r"^[$€£¥]?[-+]?[0-9][0-9,.\/:$%°]*$")

CATEGORIES = ["conclusion", "numeric", "content", "function"]
_PRIORITY = {"conclusion": 3, "numeric": 2, "content": 1, "function": 0}


def _core(sub: str) -> str:
    return sub.strip(_PUNCT)


def _is_numeric(core: str) -> bool:
    if not _DIGIT_RE.search(core):
        return False
    if re.search(r"[A-Za-z]", core):  # 英字混在は数値でない (例 "2x", "12th")
        return False
    return bool(_NUM_ALLOWED_RE.match(core)) or bool(re.fullmatch(r"[0-9.,/:$%+\-]+", core))


def classify_subword(sub: str, entry_raw: str) -> str:
    core = _core(sub)
    low = core.lower()
    # 1. conclusion
    if low in CONCLUSION_LEX:
        return "conclusion"
    if core and _MC_LABEL_RE.match(sub.strip()):
        return "conclusion"  # 単独選択肢ラベル ((C). / (A) / C.) 等)
    if low in LEADIN_WORDS and _LEADIN_RE.search(entry_raw):
        return "conclusion"  # 改行直後の答え行リード ("...\nThe answer is")
    # 2. numeric
    if _is_numeric(core):
        return "numeric"
    # 3 / 4
    if low and re.search(r"[a-z]", low):
        if low in FUNCTION_STOP:
            return "function"
        return "content" if len(low) >= 2 else "function"
    return "function"


def classify_entry(entry_word: str) -> str | None:
    """R_C ランキング 1 エントリを 4 カテゴリの 1 つに割り当てる (優先順位付き)."""
    subs = entry_word.split() or [entry_word]
    best, best_pri = None, -1
    for sub in subs:
        if not _core(sub) and not re.search(r"[0-9A-Za-z]", sub):
            continue
        cat = classify_subword(sub, entry_word)
        if _PRIORITY[cat] > best_pri:
            best_pri, best = _PRIORITY[cat], cat
    return best


def compose_top10(ranking_words: list[str]) -> dict[str, int]:
    """top-10 エントリ列 → カテゴリ別カウント (エントリ単位)."""
    counts = {c: 0 for c in CATEGORIES}
    for w in ranking_words[:10]:
        cat = classify_entry(w)
        if cat is not None:
            counts[cat] += 1
    return counts

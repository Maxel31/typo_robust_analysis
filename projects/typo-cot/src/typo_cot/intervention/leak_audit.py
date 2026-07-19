"""A2: restore「自明コピー」批判への反証のための共通部品.

攻撃: GSM8K の CoT 末尾は最終数値を含む。セル C (typo 質問 + clean CoT 強制) の
restore は「答え抽出段が CoT 末尾の数値を書き写すだけ」の自明な帰結ではないか。

本モジュールは 3 点セット共通の純関数を提供する (全て CPU・モデル不要):

  (i)   リーク層別 — 強制 clean CoT prefix に「最終答え文字列」が現れるか判定
        (numeric_leak / letter_marker_leak / option_text_leak / answer_leak)。
        リークなし事例で restore が高ければ「答えは CoT に書かれておらず、
        再導出された」ことになり自明コピー説を否定する。
  (ii)  結論剥ぎ — prefix の最終行/最終文を除去 (strip_conclusion)。
        末尾の結論行を消しても restore が保たれれば、テキストが運ぶのは
        結論の丸写しでなく再導出可能な推論内容であることを支持する。
  (iii) 回復曲線 — 先頭 p% prefix を切り出す (cut_prefix_by_fraction)。
        部分プレフィックスで段階的に復帰するなら丸写しでは説明不能。

切断・抽出の規約は cell_builder / evaluation.extractor と整合させてある
(数値正規化はカンマ除去、選択肢文字は A-D 大文字化)。
"""

import re
from dataclasses import dataclass, field

# 数値トークン (カンマ区切り・小数・符号に対応)。部分文字列マッチを避けるため
# 「数値トークン単位」で抽出してから正規化比較する (180 が 18 に誤マッチしない)。
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# 選択肢文字リーク (marker) の文脈パターン。冠詞 "A" の誤検出を避けるため、
# 括弧・リストマーカ・option/choice/answer 等の語に隣接する場合のみ leak とする。
_LETTER_MARKER_TEMPLATES = (
    r"\(\s*{L}\s*\)",  # (A)
    r"\b{L}\s*[\).:]",  # A) / A. / A:
    r"(?:option|choice|answer|select(?:ed)?|correct)\s+(?:is\s+)?\(?{L}\b",
)

# 選択肢本文リークで「短すぎて誤マッチしうる」選択肢を除外する下限文字数
_MIN_OPTION_CHARS = 4


@dataclass
class LeakResult:
    """強制 clean CoT prefix への「最終答え文字列」リーク判定結果.

    Attributes:
        leaked: ベンチマーク別の主リーク信号 (自明コピーが可能か)。
            数値系は numeric_leak、選択式は letter_marker_leak OR option_text_leak。
        numeric_leak: 金答え数値が prefix のどこかに数値トークンとして現れるか
        numeric_leak_lastline: 金答え数値が prefix の最終非空行に現れるか
        letter_marker_leak: 金答え選択肢文字が marker 文脈で現れるか
        letter_anywhere_leak: 金答え選択肢文字が単独トークンとして現れるか (寛容上限)
        option_text_leak: 金答え選択肢の本文が prefix に部分一致で現れるか
        signals: 上記フラグの辞書 (集計・デバッグ用)
    """

    leaked: bool
    numeric_leak: bool = False
    numeric_leak_lastline: bool = False
    letter_marker_leak: bool = False
    letter_anywhere_leak: bool = False
    option_text_leak: bool = False
    signals: dict = field(default_factory=dict)


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for tok in _NUMBER_RE.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _gold_float(correct_answer: str) -> float | None:
    try:
        return float(str(correct_answer).replace(",", ""))
    except (ValueError, TypeError):
        return None


def numeric_leak(prefix: str, correct_answer: str) -> bool:
    """金答え数値が prefix のどこかに数値トークンとして現れるか."""
    g = _gold_float(correct_answer)
    if g is None:
        return str(correct_answer).strip() in prefix
    return any(abs(x - g) < 1e-6 for x in _extract_numbers(prefix))


def _last_nonempty_line(prefix: str) -> str:
    for line in reversed(prefix.splitlines()):
        if line.strip():
            return line
    return ""


def numeric_leak_lastline(prefix: str, correct_answer: str) -> bool:
    """金答え数値が prefix の最終非空行 (読み上げ行) に現れるか."""
    g = _gold_float(correct_answer)
    last = _last_nonempty_line(prefix)
    if g is None:
        return str(correct_answer).strip() in last
    return any(abs(x - g) < 1e-6 for x in _extract_numbers(last))


def letter_marker_leak(prefix: str, letter: str) -> bool:
    """金答え選択肢文字が marker 文脈 (括弧・リスト・option 等) で現れるか."""
    L = str(letter).strip().upper()
    if not L:
        return False
    return any(
        re.search(tmpl.format(L=re.escape(L)), prefix, re.IGNORECASE)
        for tmpl in _LETTER_MARKER_TEMPLATES
    )


def letter_anywhere_leak(prefix: str, letter: str) -> bool:
    """金答え選択肢文字が単独トークン \\b<L>\\b として現れるか (寛容な上限判定)."""
    L = str(letter).strip().upper()
    if not L:
        return False
    return re.search(rf"\b{re.escape(L)}\b", prefix) is not None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def option_text_leak(prefix: str, option_text: str | None) -> bool:
    """金答え選択肢の本文が prefix に (正規化して) 現れるか.

    まず正規化した完全部分一致を試し、外れたら選択肢を内容語 (>=2 文字) に分解し
    全語が prefix に語単位で現れるかを見る。MMLU の "True, True" 等の構造化
    選択肢を prose ("...are true") とも突き合わせられる寛容判定 (リーク側に寛容 =
    「リークなし」層を厳しく取るので反証には保守的)。
    極端に短い選択肢 (単一数字・単語) は誤マッチ回避のため leak 扱いしない。
    """
    if not option_text:
        return False
    opt = _normalize_text(str(option_text))
    if len(opt) < _MIN_OPTION_CHARS:
        return False
    ptext = _normalize_text(prefix)
    if opt in ptext:
        return True
    tokens = [t for t in re.split(r"[,\s]+", opt) if len(t) >= 2]
    if not tokens:
        return False
    return all(re.search(rf"\b{re.escape(t)}\b", ptext) for t in tokens)


def _option_text_for_letter(letter: str, choices: list[str] | None) -> str | None:
    if not choices:
        return None
    idx = ord(str(letter).strip().upper()[:1]) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return None


_NUMERIC_BENCHMARKS = ("gsm8k", "math")


def answer_leak(
    prefix: str,
    correct_answer: str,
    benchmark: str,
    choices: list[str] | None = None,
) -> LeakResult:
    """強制 clean CoT prefix への「最終答え文字列」リークを総合判定する.

    Args:
        prefix: セル C の強制 clean CoT (答え句直前で切断済み)
        correct_answer: 金答え (数値系は数値文字列、選択式は選択肢文字 A-D)
        benchmark: ベンチマーク名 (gsm8k/math は数値、mmlu 等は選択式)
        choices: 選択式の選択肢本文リスト (option_text_leak 用)

    Returns:
        LeakResult。leaked は「自明コピーが可能な最終答え文字列が prefix に
        存在するか」の主信号。
    """
    if benchmark in _NUMERIC_BENCHMARKS:
        nl = numeric_leak(prefix, correct_answer)
        nll = numeric_leak_lastline(prefix, correct_answer)
        return LeakResult(
            leaked=nl,
            numeric_leak=nl,
            numeric_leak_lastline=nll,
            signals={"numeric_leak": nl, "numeric_leak_lastline": nll},
        )

    # 選択式 (mmlu / mmlu_pro / arc / commonsense_qa 等)
    marker = letter_marker_leak(prefix, correct_answer)
    anywhere = letter_anywhere_leak(prefix, correct_answer)
    opt_text = _option_text_for_letter(correct_answer, choices)
    opt_leak = option_text_leak(prefix, opt_text)
    leaked = marker or opt_leak
    return LeakResult(
        leaked=leaked,
        letter_marker_leak=marker,
        letter_anywhere_leak=anywhere,
        option_text_leak=opt_leak,
        signals={
            "letter_marker_leak": marker,
            "letter_anywhere_leak": anywhere,
            "option_text_leak": opt_leak,
        },
    )


# ---------------------------------------------------------------------------
# (ii) 結論剥ぎ: prefix の末尾を除去する
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def strip_conclusion(prefix: str, mode: str = "last_line") -> str:
    """強制 clean CoT prefix の末尾 (最終行 / 最終文) を除去する.

    GSM8K では最終行が「= N と読み上げる計算行」であることが多く、これを除くと
    答え数値のリークが消える。restore が保たれれば丸写しでない証拠になる。

    Args:
        prefix: セル C の強制 clean CoT prefix
        mode: "last_line" (最終非空行を除去) / "last_sentence" (最終文を除去)

    Returns:
        末尾を除去した prefix。末尾に改行を 1 つ残し、続きを新行から生成させる。
    """
    if mode == "last_line":
        lines = prefix.split("\n")
        idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                idx = i
                break
        if idx is None:
            return ""
        kept = lines[:idx]
        joined = "\n".join(kept)
        return joined + "\n" if joined.strip() else ""

    if mode == "last_sentence":
        stripped = prefix.rstrip()
        parts = _SENTENCE_SPLIT_RE.split(stripped)
        if len(parts) <= 1:
            return ""
        joined = " ".join(parts[:-1]).rstrip()
        return joined + " " if joined else ""

    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# (iii) 回復曲線: 先頭 p% prefix を切り出す
# ---------------------------------------------------------------------------

RECOVERY_GRID: tuple[int, ...] = (0, 25, 50, 75, 100)


def cut_prefix_by_fraction(cot_text: str, p: int) -> str:
    """CoT の先頭 p% (文字基準・語境界スナップ) の prefix を返す.

    exp-02 recovery_curve と同一規約。p=0 は空文字列、p=100 は全文。
    それ以外は目標文字位置以前の最後の空白まで (語の途中で切らない)。
    """
    if p <= 0:
        return ""
    if p >= 100:
        return cot_text
    target = int(len(cot_text) * p / 100)
    last_ws = -1
    for i, ch in enumerate(cot_text):
        if i > target:
            break
        if ch.isspace():
            last_ws = i
    return cot_text[: last_ws + 1] if last_ws >= 0 else ""

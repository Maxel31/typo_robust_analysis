"""実験2 副実験: 回復曲線 — p% prefix 強制と回復ジャンプ位置の並べ替え検定.

計画 §4 実験2-2-4: セルC構成 (typo 質問 + clean CoT 先頭 p% 強制、
p∈{0,25,50,75,100}) で続きを自由生成し、回復 (= clean 答えへの復帰) の
ジャンプ位置と最上位 R_C 語の初出位置の一致率を permutation 検定する。
帰無分布は同一 CoT の content 候補語から一様抽選した語で同判定。
"""

import random

from typo_cot.intervention.loo_scorer import extract_word_types

GRID: tuple[int, ...] = (0, 25, 50, 75, 100)


def cut_prefix_by_fraction(cot_text: str, p: int) -> str:
    """CoT の先頭 p% (文字基準・語境界スナップ) の prefix を返す.

    p=0 は空文字列、p=100 は全文。それ以外は目標文字位置以前の最後の空白まで
    (prefix は空白で終わる = 語の途中で切らない)。
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


def find_jump(
    recovered: dict[int, bool],
    grid: tuple[int, ...] = GRID,
) -> tuple[int | None, int] | None:
    """回復が最初に真になる格子点 p* とその直前点 p_prev を返す.

    Returns:
        (p_prev, p_star)。最初の格子点で回復していた場合は (None, p_star)。
        どの点でも回復しなければ None。
    """
    prev: int | None = None
    for p in grid:
        if recovered.get(p):
            return (prev, p)
        prev = p
    return None


def jump_match(jump: tuple[int | None, int] | None, target_frac: float | None) -> bool:
    """標的語の初出位置比がジャンプ区間 (p_prev/100, p_star/100] に入るか.

    p_prev が None (CoT なしで回復) の場合は語に帰属できないため False。
    """
    if jump is None or target_frac is None:
        return False
    p_prev, p_star = jump
    if p_prev is None:
        return False
    return p_prev / 100 < target_frac <= p_star / 100


def target_first_fraction(cot_text: str, word: str) -> float | None:
    """語タイプの初出コアスパン開始位置を CoT 長で正規化した比を返す."""
    if not cot_text:
        return None
    for wt in extract_word_types(cot_text):
        if wt.word == word:
            return min(s for s, _ in wt.spans) / len(cot_text)
    return None


def match_rate_permutation_test(
    cases: list[dict],
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """ジャンプ位置と標的語初出位置の一致率の並べ替え検定 (片側: 一致率が高い).

    Args:
        cases: [{"interval": (p_prev, p_star) | None,
                 "target_frac": float | None,
                 "candidate_fracs": [float, ...]}]  — candidate_fracs は同一 CoT の
                 content 候補語の初出位置比 (帰無分布の抽選元)
        n_perm: 並べ替え回数
        seed: RNG seed

    Returns:
        {"n_cases", "observed_match_rate", "null_mean_match_rate", "p_value"}
    """
    valid = [c for c in cases if c.get("interval") is not None]
    n = len(valid)
    if n == 0:
        return {
            "n_cases": 0,
            "observed_match_rate": None,
            "null_mean_match_rate": None,
            "p_value": None,
        }
    observed = sum(jump_match(c["interval"], c.get("target_frac")) for c in valid) / n

    rng = random.Random(seed)
    count = 0
    null_total = 0.0
    for _ in range(n_perm):
        rate = 0
        for c in valid:
            fracs = c.get("candidate_fracs") or []
            frac = rng.choice(fracs) if fracs else None
            rate += jump_match(c["interval"], frac)
        rate /= n
        null_total += rate
        if rate >= observed - 1e-12:
            count += 1
    return {
        "n_cases": n,
        "observed_match_rate": observed,
        "null_mean_match_rate": null_total / n_perm,
        "p_value": (count + 1) / (n_perm + 1),
    }

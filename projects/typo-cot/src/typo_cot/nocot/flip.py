"""実験14: no-CoT flip 集計 (純粋関数, GPU 不要).

生成スクリプト (scripts/exp14_nocot/run_nocot_shard.py) が clean / typo 条件
それぞれで no-CoT の答えスパンを生成・抽出し、sample_id → {"answer","is_correct"}
のレコードを保存する。本モジュールはそれらを結合して flip 指標を計算する。

flip の定義 (実験1 の DE=C セルと整合):
  no-CoT flip = clean 質問で正解 かつ typo 質問で不正解 (clean正解→摂動誤答)。
  clean 正解に条件付けると「答えが変わる」= 「不正解になる」と同値
  (correct_answer は一意)。よって DE の C セル (answers[C] != answers[A]) の
  no-CoT アナログになっている。
"""

from __future__ import annotations


def join_records(
    clean: dict[str, dict],
    typo: dict[str, dict],
) -> list[dict]:
    """clean / typo レコードを sample_id で結合する (両方に存在するもののみ).

    Args:
        clean: sample_id → {"answer": str, "is_correct": bool, ...}
        typo: 同上 (摂動条件)

    Returns:
        結合済みレコードのリスト。各要素:
            sample_id, clean_answer, typo_answer, clean_correct, typo_correct,
            answer_changed (clean と typo の抽出答えが異なるか),
            flip_correct_to_wrong (clean 正解 かつ typo 不正解)
    """
    out: list[dict] = []
    for sid, c in clean.items():
        t = typo.get(sid)
        if t is None:
            continue
        ca = str(c.get("answer", "")).strip()
        ta = str(t.get("answer", "")).strip()
        cc = bool(c.get("is_correct", False))
        tc = bool(t.get("is_correct", False))
        out.append(
            {
                "sample_id": sid,
                "clean_answer": ca,
                "typo_answer": ta,
                "clean_correct": cc,
                "typo_correct": tc,
                "answer_changed": ca != ta,
                "flip_correct_to_wrong": cc and not tc,
            }
        )
    return out


def flip_summary(joined: list[dict]) -> dict:
    """結合済みレコードから no-CoT flip 指標を集計する.

    Returns:
        n_joined / n_clean_correct / n_flip / nocot_flip_rate
        (= clean正解→摂動誤答 率, clean 正解に条件付け) と
        n_answer_changed / answer_change_rate (全結合上, 参考値)。
    """
    n = len(joined)
    clean_correct = [r for r in joined if r["clean_correct"]]
    ncc = len(clean_correct)
    n_flip = sum(1 for r in clean_correct if r["flip_correct_to_wrong"])
    n_changed = sum(1 for r in joined if r["answer_changed"])
    return {
        "n_joined": n,
        "n_clean_correct": ncc,
        "n_flip": n_flip,
        "nocot_flip_rate": (n_flip / ncc) if ncc > 0 else None,
        "n_answer_changed": n_changed,
        "answer_change_rate": (n_changed / n) if n > 0 else None,
    }


def odds_ratio(
    a: float,
    b: float,
    c: float,
    d: float,
    haldane: bool = True,
) -> dict:
    """2x2 分割表のオッズ比.

    表の並び (露出=C セル DE flip, 帰結=no-CoT flip):
        a = 両方で flip
        b = DE flip のみ (no-CoT では flip せず)
        c = no-CoT flip のみ (DE では flip せず)
        d = どちらも flip せず

    Args:
        haldane: いずれかのセルが 0 のとき全セルに 0.5 を加える
            (Haldane-Anscombe 補正)。

    Returns:
        {"odds_ratio", "log_odds_ratio", "table", "haldane_applied"}
    """
    import math

    cells = [a, b, c, d]
    haldane_applied = False
    if haldane and any(x == 0 for x in cells):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        haldane_applied = True

    denom = b * c
    if denom == 0:
        return {
            "odds_ratio": None,
            "log_odds_ratio": None,
            "table": {"a": a, "b": b, "c": c, "d": d},
            "haldane_applied": haldane_applied,
        }
    orr = (a * d) / denom
    return {
        "odds_ratio": orr,
        "log_odds_ratio": math.log(orr) if orr > 0 else None,
        "table": {"a": a, "b": b, "c": c, "d": d},
        "haldane_applied": haldane_applied,
    }

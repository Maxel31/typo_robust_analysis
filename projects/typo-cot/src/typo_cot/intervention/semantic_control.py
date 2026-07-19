"""実験8-fine A3 統制(c): 意味置換 (semantic replacement) 摂動ペアの生成.

敵対的レビュー A3 への対応。typo ではなく「同義でない実語」に標的語を置換した
摂動ペアを作る。これに対して同じ 1 層 residual スイープを回し、深さプロファイルが
typo と同形なら「読み出しダイナミクスは入力摂動一般の性質であり、typo 固有寄与は
LXT vs Random の倍率差に局在する」と正直に書ける。異形なら typo 固有の局在を主張できる。

置換は決定論的 (seed + sample_id + 標的語) で、既存の flip ペア (typo で選定済み) の
標的語を実語プール REAL_WORDS からランダムに引く。生成は不要 (patching 側は clean CoT を
teacher-forcing するため semantic 側の CoT を必要としない)。
"""

from __future__ import annotations

import random as _random
from dataclasses import replace

from typo_cot.intervention.records import PairRecord

# 同義になりにくい、具体的で頻度の高い実語プール (名詞/形容詞中心)。
# 標的語の同義語を避けるためドメイン横断でランダムに引く。
REAL_WORDS: tuple[str, ...] = (
    "planet", "garden", "copper", "violin", "harbor", "lantern", "meadow", "cactus",
    "pillow", "anchor", "marble", "sandal", "tunnel", "kettle", "pebble", "sparrow",
    "canyon", "velvet", "acorn", "bison", "cabin", "domino", "ember", "fossil",
    "granite", "hammock", "iceberg", "jungle", "kayak", "ladder", "magnet", "nectar",
    "orchid", "puzzle", "quartz", "ribbon", "saddle", "teapot", "umbrella", "walnut",
    "yogurt", "zephyr", "beacon", "compass", "dolphin", "engine", "feather", "glacier",
    "hazel", "island", "jacket", "kernel", "lemon", "mirror", "needle", "otter",
    "parrot", "quilt", "rocket", "shovel", "trumpet", "urchin", "vessel", "wagon",
    "willow", "cinder", "basket", "candle", "drawer", "falcon", "gravel", "helmet",
    "ivory", "jasmine", "locket", "muffin", "nutmeg", "onion", "prism", "raven",
)


def _pick_word(rng: _random.Random, original: str) -> str:
    """元語と異なる実語を選ぶ (元語の大文字化を継承)."""
    orig_low = original.strip().lower()
    w = rng.choice(REAL_WORDS)
    for _ in range(20):
        if w.lower() != orig_low:
            break
        w = rng.choice(REAL_WORDS)
    if original[:1].isupper():
        return w[:1].upper() + w[1:]
    return w


def make_semantic_pair(pair: PairRecord, seed: int = 1234) -> PairRecord | None:
    """標的語を実語ランダム置換した semantic-perturbation ペアを返す.

    question_clean 中の各 original_token を実語に置換し、question_typo に格納する。
    perturbed_token を置換語に更新 (original_token は不変)。teacher-forcing する
    clean CoT は不変なので、生成は不要で S2 KL 回復率をそのまま比較できる。

    Args:
        pair: typo で選定済みの flip ペア
        seed: 置換の乱数シード (sample_id/標的語と合わせて決定論的)

    Returns:
        semantic ペア。標的語が無い / 質問中に見つからない場合は None (除外)。
    """
    tokens = pair.extra.get("perturbed_tokens", []) if pair.extra else []
    if not tokens:
        return None

    question = pair.question_clean
    new_tokens: list[dict] = []
    replaced_any = False
    cursor = 0
    for i, tok in enumerate(tokens):
        orig = str(tok.get("original_token", "")).strip()
        if not orig:
            new_tokens.append(dict(tok))
            continue
        rng = _random.Random(f"{seed}|{pair.sample_id}|{i}|{orig}")
        rep = _pick_word(rng, orig)
        pos = question.find(orig, cursor)
        if pos < 0:
            pos = question.find(orig)  # 順序制約を緩めて全体探索
        if pos < 0:
            new_tokens.append(dict(tok))  # 置換不能 (質問に無い)
            continue
        question = question[:pos] + rep + question[pos + len(orig):]
        cursor = pos + len(rep)
        replaced_any = True
        nt = dict(tok)
        nt["perturbed_token"] = rep
        new_tokens.append(nt)

    if not replaced_any:
        return None

    new_extra = dict(pair.extra)
    new_extra["perturbed_tokens"] = new_tokens
    new_extra["perturb_mode"] = "semantic"
    return replace(
        pair,
        question_typo=question,
        choices_typo=pair.choices_clean,
        extra=new_extra,
    )

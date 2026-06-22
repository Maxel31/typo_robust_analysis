"""文字レベルの typo（誤字）注入。

再現実験・提案手法の双方で使う中核ユーティリティ。決定論的に動かすため
``random.Random`` インスタンスを使い、シードを明示できるようにしている。
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Literal

TypoType = Literal[
    "swap", "insert", "delete", "substitute", "keyboard", "replace", "random"
]

_RANDOM_POOL: list[TypoType] = ["swap", "insert", "delete", "substitute"]

# 近接キー（QWERTY）— keyboard typo 用の簡易マップ
_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}


@dataclass
class TypoAnnotation:
    """1 箇所の typo 適用記録。"""

    word_index: int
    original_word: str
    typo_word: str
    typo_type: TypoType


@dataclass
class TypoConfig:
    """typo 注入の設定。"""

    rate: float = 0.1          # 単語あたり typo を入れる確率
    type: TypoType = "swap"    # typo の種類
    seed: int = 42


def _apply_one(word: str, typo_type: TypoType, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word))
    chars = list(word)
    if typo_type == "swap":
        j = min(i + 1, len(chars) - 1)
        chars[i], chars[j] = chars[j], chars[i]
    elif typo_type == "insert":
        chars.insert(i, rng.choice(string.ascii_lowercase))
    elif typo_type == "delete":
        del chars[i]
    elif typo_type == "substitute":
        chars[i] = rng.choice(string.ascii_lowercase)
    elif typo_type == "keyboard":
        neighbors = _KEYBOARD_NEIGHBORS.get(chars[i].lower())
        if neighbors:
            chars[i] = rng.choice(neighbors)
    return "".join(chars)


def inject_typos(text: str, config: TypoConfig | None = None) -> str:
    """文を単語単位で走査し、確率 ``rate`` で各単語に typo を 1 つ入れる。"""
    config = config or TypoConfig()
    rng = random.Random(config.seed)
    words = text.split(" ")
    out = [
        _apply_one(w, config.type, rng) if (w and rng.random() < config.rate) else w
        for w in words
    ]
    return " ".join(out)


def inject_typos_by_count(
    text: str,
    num_typos: int,
    typo_type: TypoType,
    seed: int = 42,
    exclude_indices: set[int] | None = None,
) -> tuple[str, list[TypoAnnotation]]:
    """指定個数の単語に typo を注入し、変更履歴を返す。"""
    rng = random.Random(seed)
    words = text.split(" ")
    exclude = exclude_indices or set()

    candidates = [
        i for i, w in enumerate(words) if len(w) >= 2 and i not in exclude
    ]
    rng.shuffle(candidates)
    targets = candidates[:num_typos]
    targets.sort()

    annotations: list[TypoAnnotation] = []
    for idx in targets:
        original = words[idx]
        if typo_type == "random":
            chosen: TypoType = rng.choice(_RANDOM_POOL)
        elif typo_type == "replace":
            chosen = "substitute"
        else:
            chosen = typo_type
        modified = _apply_one(original, chosen, rng)
        for _ in range(5):
            if modified != original:
                break
            modified = _apply_one(original, chosen, rng)
        if modified == original:
            continue
        words[idx] = modified
        ann_type: TypoType = "replace" if typo_type == "replace" else chosen
        annotations.append(
            TypoAnnotation(
                word_index=idx,
                original_word=original,
                typo_word=modified,
                typo_type=ann_type,
            )
        )

    return " ".join(words), annotations

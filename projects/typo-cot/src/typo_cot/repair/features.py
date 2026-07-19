"""実験9 の flip 回帰に用いる表層統制特徴量.

- split_increment: 摂動によるサブワード分割数の増分
- zipf_frequency: clean 語の Zipf 頻度 (wordfreq)
- first_token_id: 語の先頭トークン ID (logit lens の標的)
"""

from typing import Protocol

import wordfreq


class _Encoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def _encode_word(tokenizer: _Encoder, word: str, with_leading_space: bool = True) -> list[int]:
    text = f" {word}" if with_leading_space else word
    return tokenizer.encode(text, add_special_tokens=False)


def split_increment(tokenizer: _Encoder, clean_word: str, typo_word: str) -> int:
    """摂動後 piece 数 - 摂動前 piece 数.

    文中出現を模して先頭スペース付きでトークナイズする。
    """
    n_clean = len(_encode_word(tokenizer, clean_word))
    n_typo = len(_encode_word(tokenizer, typo_word))
    return n_typo - n_clean


def zipf_frequency(word: str, lang: str = "en") -> float:
    """clean 語の Zipf 頻度 (大文字小文字を無視)。未知語は 0.0."""
    return wordfreq.zipf_frequency(word.lower(), lang)


def first_token_id(
    tokenizer: _Encoder, word: str, with_leading_space: bool = True
) -> int:
    """語の先頭トークン ID を返す (logit lens の復号標的)."""
    ids = _encode_word(tokenizer, word, with_leading_space=with_leading_space)
    return ids[0]

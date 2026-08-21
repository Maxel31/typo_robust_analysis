"""Deterministic UCC-Inj variation-selector text noise.

UCC-Inj's GSM8K encoder appends a randomly selected Unicode variation selector
after every character.  These code points are usually invisible but are still
tokenized by modern LLM tokenizers.  ``noise_level`` is exactly the number of
selectors appended per original character, matching the reference encoder's
``insert_random_code(text, level, level)`` behaviour.
"""

from __future__ import annotations

import random


def variation_selector_for_byte(value: int) -> str:
    """Map an integer in ``[0, 255]`` to UCC-Inj's variation selector."""
    if not 0 <= value <= 255:
        raise ValueError("variation-selector byte must be in [0, 255]")
    return chr(0xFE00 + value) if value < 16 else chr(0xE0100 + value - 16)


def inject_variation_selector_noise(text: str, *, noise_level: int, seed: int) -> str:
    """Return a deterministic variation-selector-corrupted copy of ``text``.

    A level of zero is deliberately identity-preserving, which makes clean
    controls byte-comparable.  Each positive level appends exactly that many
    independently sampled selectors to every Unicode character.
    """
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    if noise_level == 0:
        return text
    rng = random.Random(seed)
    pieces: list[str] = []
    for character in text:
        pieces.append(character)
        pieces.extend(variation_selector_for_byte(rng.randrange(256)) for _ in range(noise_level))
    return "".join(pieces)

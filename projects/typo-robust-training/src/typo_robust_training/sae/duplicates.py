"""Bounded-memory character-ngram duplicate guard for SAE corpus extension."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from functools import cache

import numpy as np


_WHITESPACE = re.compile(r"\s+")
_HASH_MASK = (1 << 64) - 1
_SHINGLE_BASE = 257
_SIGNATURE_COMPONENTS = 32
_BANDS = 8
_ROWS_PER_BAND = 4


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("duplicate-guard text must be a string")
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, size: int) -> frozenset[int]:
    normalized = normalize_text(text)
    if not normalized:
        return frozenset()
    width = min(size, len(normalized))
    high = pow(_SHINGLE_BASE, width - 1, 1 << 64)
    value = 0
    for character in normalized[:width]:
        value = (value * _SHINGLE_BASE + ord(character) + 1) & _HASH_MASK
    hashes = {value}
    for index in range(width, len(normalized)):
        outgoing = ord(normalized[index - width]) + 1
        incoming = ord(normalized[index]) + 1
        value = ((value - outgoing * high) * _SHINGLE_BASE + incoming) & _HASH_MASK
        hashes.add(value)
    return frozenset(hashes)


@cache
def _minhash_coefficients() -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for component in range(_SIGNATURE_COMPONENTS):
        digest = hashlib.blake2b(component.to_bytes(2, "big"), digest_size=16).digest()
        values.append((int.from_bytes(digest[:8], "big") | 1, int.from_bytes(digest[8:], "big")))
    return tuple(values)


def _signature(shingles: frozenset[int]) -> tuple[int, ...]:
    if not shingles:
        return (0,) * _SIGNATURE_COMPONENTS
    coefficients = _minhash_coefficients()
    multipliers = np.asarray([value[0] for value in coefficients], dtype=np.uint64)
    offsets = np.asarray([value[1] for value in coefficients], dtype=np.uint64)
    values = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    signature = (multipliers[:, None] * values[None, :] + offsets[:, None]).min(axis=1)
    return tuple(int(value) for value in signature)


def _band_key(band: int, values: tuple[int, ...]) -> tuple[int, bytes]:
    payload = b"".join(value.to_bytes(8, "big") for value in values)
    return band, hashlib.blake2b(payload, digest_size=8).digest()


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class CharacterNgramDuplicateGuard:
    """Reject exact or LSH-candidate near duplicates as records stream in."""

    def __init__(self, *, shingle_size: int, threshold: float) -> None:
        if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size <= 0:
            raise ValueError("duplicate-guard shingle size must be positive")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 < float(threshold) <= 1.0
        ):
            raise ValueError("duplicate-guard threshold must be in (0, 1]")
        self.shingle_size = shingle_size
        self.threshold = float(threshold)
        self._normalized: dict[str, str] = {}
        self._texts: dict[str, str] = {}
        self._buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)

    def _find_duplicate(
        self,
        text: str,
        *,
        shingles: frozenset[int],
        signature: tuple[int, ...],
    ) -> str | None:
        exact = self._normalized.get(normalized_sha256(text))
        if exact is not None:
            return exact
        candidates: set[str] = set()
        for band in range(_BANDS):
            start = band * _ROWS_PER_BAND
            candidates.update(
                self._buckets.get(
                    _band_key(band, signature[start : start + _ROWS_PER_BAND]),
                    (),
                )
            )
        for identity in sorted(candidates):
            if (
                _jaccard(shingles, _shingles(self._texts[identity], self.shingle_size))
                >= self.threshold
            ):
                return identity
        return None

    def find_duplicate(self, text: str) -> str | None:
        shingles = _shingles(text, self.shingle_size)
        return self._find_duplicate(
            text,
            shingles=shingles,
            signature=_signature(shingles),
        )

    def add(self, identity: str, text: str) -> str | None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("duplicate-guard identity must be non-empty")
        if identity in self._texts:
            raise ValueError("duplicate-guard identity is duplicated")
        shingles = _shingles(text, self.shingle_size)
        signature = _signature(shingles)
        duplicate = self._find_duplicate(
            text,
            shingles=shingles,
            signature=signature,
        )
        if duplicate is not None:
            return duplicate
        self._normalized[normalized_sha256(text)] = identity
        self._texts[identity] = text
        for band in range(_BANDS):
            start = band * _ROWS_PER_BAND
            self._buckets[_band_key(band, signature[start : start + _ROWS_PER_BAND])].append(
                identity
            )
        return None

    @property
    def records(self) -> int:
        return len(self._texts)


__all__ = ["CharacterNgramDuplicateGuard", "normalize_text", "normalized_sha256"]

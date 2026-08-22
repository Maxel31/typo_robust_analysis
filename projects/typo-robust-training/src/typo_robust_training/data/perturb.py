"""Order-independent on-the-fly character typo generation."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence

from typo_robust_training.data.records import CleanRecord, TypoEdit, TypoPair


TRAINING_OPERATIONS = frozenset(
    {
        "keyboard-neighbor-substitution",
        "deletion",
        "insertion",
        "duplication",
        "natural-statistics-substitution",
    }
)
HELD_OUT_OPERATIONS = frozenset({"adjacent-transposition"})

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_FORBIDDEN = (
    re.compile(r"https?://\S+", flags=re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][A-Za-z0-9]*)+\b"),
)
_KEYBOARD_NEIGHBORS: Mapping[str, str] = {
    "q": "wa",
    "w": "qase",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _overlaps(span: tuple[int, int], forbidden: tuple[int, int]) -> bool:
    return span[0] < forbidden[1] and forbidden[0] < span[1]


def eligible_word_spans(text: str, *, minimum_letters: int = 2) -> tuple[tuple[int, int], ...]:
    """Return alphabetic word spans while excluding URLs, email, and identifiers."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if (
        isinstance(minimum_letters, bool)
        or not isinstance(minimum_letters, int)
        or minimum_letters < 2
    ):
        raise ValueError("minimum_letters must be an integer of at least two")
    forbidden = tuple(match.span() for pattern in _FORBIDDEN for match in pattern.finditer(text))
    spans: list[tuple[int, int]] = []
    for match in _WORD.finditer(text):
        span = match.span()
        word = match.group()
        if (span[0] and text[span[0] - 1].isalnum()) or (
            span[1] < len(text) and text[span[1]].isalnum()
        ):
            continue
        if sum(character.isalpha() for character in word) < minimum_letters:
            continue
        if any(_overlaps(span, blocked) for blocked in forbidden):
            continue
        spans.append(span)
    return tuple(spans)


def _preserve_case(replacement: str, original: str) -> str:
    return replacement.upper() if original.isupper() else replacement.lower()


def _keyboard_substitution(word: str, rng: random.Random) -> str:
    candidates = [
        index for index, character in enumerate(word) if character.lower() in _KEYBOARD_NEIGHBORS
    ]
    if not candidates:
        return _general_substitution(word, rng)
    index = rng.choice(candidates)
    original = word[index]
    replacement = rng.choice(_KEYBOARD_NEIGHBORS[original.lower()])
    return word[:index] + _preserve_case(replacement, original) + word[index + 1 :]


def _general_substitution(word: str, rng: random.Random) -> str:
    candidates = [index for index, character in enumerate(word) if character.isalpha()]
    index = rng.choice(candidates)
    original = word[index]
    alphabet = "abcdefghijklmnopqrstuvwxyz".replace(original.lower(), "")
    replacement = _preserve_case(rng.choice(alphabet), original)
    return word[:index] + replacement + word[index + 1 :]


def _natural_substitution(
    word: str,
    rng: random.Random,
    substitutions: Mapping[str, Mapping[str, float]],
) -> str:
    global_replacements: dict[str, float] = {}
    for replacements in substitutions.values():
        for replacement, weight in replacements.items():
            global_replacements[replacement] = global_replacements.get(replacement, 0.0) + weight
    candidates = [
        index
        for index, character in enumerate(word)
        if character.isalpha()
        and any(
            replacement != character.lower() and weight > 0.0
            for replacement, weight in substitutions.get(
                character.lower(), global_replacements
            ).items()
        )
    ]
    if not candidates:
        raise ValueError("word has no replacement supported by natural typo statistics")
    index = rng.choice(candidates)
    original = word[index]
    conditional = substitutions.get(original.lower(), global_replacements)
    eligible = {
        replacement: weight
        for replacement, weight in conditional.items()
        if replacement != original.lower() and weight > 0.0
    }
    total = sum(eligible.values())
    normalized = {key: value / total for key, value in sorted(eligible.items())}
    replacement = _weighted_choice(rng, normalized)
    return word[:index] + _preserve_case(replacement, original) + word[index + 1 :]


def _delete(word: str, rng: random.Random) -> str:
    if len(word) <= 1:
        return _general_substitution(word, rng)
    index = rng.randrange(len(word))
    return word[:index] + word[index + 1 :]


def _insert(word: str, rng: random.Random) -> str:
    anchor = rng.randrange(len(word))
    original = word[anchor]
    neighbors = _KEYBOARD_NEIGHBORS.get(original.lower(), "abcdefghijklmnopqrstuvwxyz")
    inserted = _preserve_case(rng.choice(neighbors), original)
    position = anchor + rng.randrange(2)
    return word[:position] + inserted + word[position:]


def _duplicate(word: str, rng: random.Random) -> str:
    index = rng.randrange(len(word))
    return word[: index + 1] + word[index] + word[index + 1 :]


def _apply_operation(
    word: str,
    operation: str,
    rng: random.Random,
    *,
    natural_substitutions: Mapping[str, Mapping[str, float]],
) -> str:
    if operation == "keyboard-neighbor-substitution":
        return _keyboard_substitution(word, rng)
    if operation == "natural-statistics-substitution":
        return _natural_substitution(word, rng, natural_substitutions)
    if operation == "deletion":
        return _delete(word, rng)
    if operation == "insertion":
        return _insert(word, rng)
    if operation == "duplication":
        return _duplicate(word, rng)
    raise ValueError(f"unsupported training typo operation: {operation}")


def apply_typo_operation_to_word(
    word: str,
    operation: str,
    rng: random.Random,
    *,
    natural_substitutions: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Apply one named character edit for a caller with an explicit RNG."""

    if not isinstance(word, str) or not word or not any(character.isalpha() for character in word):
        raise ValueError("typo operation requires a non-empty alphabetic word")
    if not isinstance(operation, str) or not operation:
        raise ValueError("typo operation name must be non-empty")
    if not isinstance(rng, random.Random):
        raise TypeError("typo operation rng must be random.Random")
    if operation == "adjacent-transposition":
        candidates = [index for index in range(len(word) - 1) if word[index] != word[index + 1]]
        if not candidates:
            raise ValueError("word has no non-identity adjacent transposition")
        index = rng.choice(candidates)
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    return _apply_operation(
        word,
        operation,
        rng,
        natural_substitutions=natural_substitutions or {},
    )


def _derived_rng(*, seed: int, epoch: int, variant: int, record_id: str) -> random.Random:
    material = f"typo-generator/v1\0{seed}\0{epoch}\0{variant}\0{record_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    ordered_weights = tuple(sorted(weights.items()))
    value = rng.random()
    cumulative = 0.0
    for name, weight in ordered_weights:
        cumulative += float(weight)
        if value < cumulative:
            return name
    return ordered_weights[-1][0]


class TypoGenerator:
    """Generate a record-local typo variant without shared mutable RNG state."""

    def __init__(
        self,
        *,
        seed: int,
        operation_weights: Mapping[str, float] | None = None,
        edit_count_weights: Mapping[str, float] | None = None,
        natural_substitutions: Mapping[str, Mapping[str, float]] | None = None,
        explicit_clean_pair_probability: float = 0.0,
        minimum_word_letters: int = 2,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("generator seed must be a non-negative integer")
        self.seed = seed
        self.operation_weights = dict(
            operation_weights
            or {
                operation: 1.0 / len(TRAINING_OPERATIONS)
                for operation in sorted(TRAINING_OPERATIONS)
            }
        )
        self.edit_count_weights = dict(edit_count_weights or {"1": 0.50, "2": 0.30, "3-4": 0.20})
        self.natural_substitutions: dict[str, dict[str, float]] = {}
        for clean, replacements in sorted((natural_substitutions or {}).items()):
            if (
                not isinstance(clean, str)
                or len(clean) != 1
                or not clean.isascii()
                or not clean.isalpha()
            ):
                raise ValueError("natural substitution source keys must be ASCII letters")
            if not isinstance(replacements, Mapping) or not replacements:
                raise ValueError("natural substitution targets must be non-empty objects")
            normalized: dict[str, float] = {}
            for replacement, weight in sorted(replacements.items()):
                if (
                    not isinstance(replacement, str)
                    or len(replacement) != 1
                    or not replacement.isascii()
                    or not replacement.isalpha()
                    or isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or float(weight) < 0.0
                ):
                    raise ValueError("natural substitution entries are invalid")
                normalized[replacement.lower()] = float(weight)
            if not any(weight > 0.0 for weight in normalized.values()):
                raise ValueError("natural substitution targets must include positive weight")
            self.natural_substitutions[clean.lower()] = normalized
        for label, weights in (
            ("operation", self.operation_weights),
            ("edit count", self.edit_count_weights),
        ):
            if not weights:
                raise ValueError(f"{label} weights cannot be empty")
            if label == "operation" and any(key not in TRAINING_OPERATIONS for key in weights):
                raise ValueError(f"{label} weights contain an unsupported value")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
                raise ValueError(f"{label} weights must sum to one")
            if any(float(value) < 0.0 for value in weights.values()):
                raise ValueError(f"{label} weights must be non-negative")
        if (
            self.operation_weights.get("natural-statistics-substitution", 0.0) > 0.0
            and not self.natural_substitutions
        ):
            raise ValueError("natural substitution statistics are required")
        if set(self.edit_count_weights) != {"1", "2", "3-4"}:
            raise ValueError("edit count weights must use 1, 2, and 3-4")
        if not 0.0 <= explicit_clean_pair_probability <= 1.0:
            raise ValueError("explicit clean-pair probability must be in [0, 1]")
        self.explicit_clean_pair_probability = float(explicit_clean_pair_probability)
        self.minimum_word_letters = minimum_word_letters

    def generate(
        self,
        record: CleanRecord,
        *,
        epoch: int,
        variant: int = 0,
        force_operations: Sequence[str] | None = None,
        force_edit_count: int | None = None,
        force_noop: bool | None = None,
        maximum_target_stop: int | None = None,
    ) -> TypoPair:
        if not isinstance(record, CleanRecord):
            raise TypeError("record must be a CleanRecord")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (epoch, variant)
        ):
            raise ValueError("epoch and variant must be non-negative integers")
        if force_noop is not None and type(force_noop) is not bool:
            raise TypeError("force_noop must be boolean or None")
        if maximum_target_stop is not None and (
            isinstance(maximum_target_stop, bool)
            or not isinstance(maximum_target_stop, int)
            or maximum_target_stop <= 0
        ):
            raise ValueError("maximum_target_stop must be a positive integer")
        if force_noop is True and (force_operations is not None or force_edit_count is not None):
            raise ValueError("forced clean pairs cannot also force typo edits")
        if force_operations is not None:
            if not force_operations:
                raise ValueError("force_operations cannot be empty")
            held_out = set(force_operations) & HELD_OUT_OPERATIONS
            if held_out:
                raise ValueError(
                    f"held-out operations cannot be used for training: {sorted(held_out)}"
                )
            unsupported = set(force_operations) - TRAINING_OPERATIONS
            if unsupported:
                raise ValueError(f"unsupported forced operations: {sorted(unsupported)}")
        rng = _derived_rng(seed=self.seed, epoch=epoch, variant=variant, record_id=record.record_id)
        if force_noop is True or (
            force_noop is None
            and force_edit_count is None
            and rng.random() < self.explicit_clean_pair_probability
        ):
            return TypoPair(
                record_id=record.record_id,
                source_id=record.source_id,
                clean_text=record.text,
                typo_text=record.text,
                seed=self.seed,
                epoch=epoch,
                variant=variant,
                edits=(),
                is_noop=True,
            )
        spans = eligible_word_spans(record.text, minimum_letters=self.minimum_word_letters)
        if maximum_target_stop is not None:
            spans = tuple(span for span in spans if span[1] <= maximum_target_stop)
        if not spans:
            raise ValueError(f"record contains no eligible typo target: {record.source_id}")
        if force_edit_count is not None:
            if (
                isinstance(force_edit_count, bool)
                or not isinstance(force_edit_count, int)
                or force_edit_count <= 0
            ):
                raise ValueError("force_edit_count must be a positive integer")
            count = force_edit_count
        else:
            bucket = _weighted_choice(rng, self.edit_count_weights)
            count = int(bucket) if bucket != "3-4" else rng.choice((3, 4))
        count = min(count, len(spans))
        selected = sorted(rng.sample(list(spans), k=count))
        operations = [
            force_operations[index % len(force_operations)]
            if force_operations is not None
            else _weighted_choice(rng, self.operation_weights)
            for index in range(count)
        ]

        pieces: list[str] = []
        edits: list[TypoEdit] = []
        cursor = 0
        typo_length = 0
        for (start, stop), operation in zip(selected, operations, strict=True):
            unchanged = record.text[cursor:start]
            pieces.append(unchanged)
            typo_length += len(unchanged)
            clean_word = record.text[start:stop]
            typo_word = _apply_operation(
                clean_word,
                operation,
                rng,
                natural_substitutions=self.natural_substitutions,
            )
            typo_start = typo_length
            pieces.append(typo_word)
            typo_length += len(typo_word)
            edits.append(
                TypoEdit(
                    operation=operation,
                    clean_word=clean_word,
                    typo_word=typo_word,
                    clean_char_span=(start, stop),
                    typo_char_span=(typo_start, typo_length),
                )
            )
            cursor = stop
        pieces.append(record.text[cursor:])
        typo_text = "".join(pieces)
        return TypoPair(
            record_id=record.record_id,
            source_id=record.source_id,
            clean_text=record.text,
            typo_text=typo_text,
            seed=self.seed,
            epoch=epoch,
            variant=variant,
            edits=tuple(edits),
        )


def classify_character_edit(*, clean: str, typo: str) -> str | None:
    """Classify one character edit, with clean text supplied first."""

    if not isinstance(clean, str) or not isinstance(typo, str) or clean == typo:
        return None
    prefix = 0
    while prefix < min(len(clean), len(typo)) and clean[prefix] == typo[prefix]:
        prefix += 1
    clean_stop, typo_stop = len(clean), len(typo)
    while (
        clean_stop > prefix and typo_stop > prefix and clean[clean_stop - 1] == typo[typo_stop - 1]
    ):
        clean_stop -= 1
        typo_stop -= 1
    clean_change = clean[prefix:clean_stop]
    typo_change = typo[prefix:typo_stop]
    if len(clean_change) == 2 and len(typo_change) == 2 and clean_change == typo_change[::-1]:
        return "adjacent-transposition"
    if len(clean_change) == 1 and len(typo_change) == 1:
        return "natural-statistics-substitution"
    if len(clean_change) == 1 and not typo_change:
        return "deletion"
    if not clean_change and len(typo_change) == 1:
        inserted = typo_change[0]
        adjacent = {
            character
            for character in (
                typo[prefix - 1] if prefix > 0 else "",
                typo[typo_stop] if typo_stop < len(typo) else "",
            )
            if character
        }
        return "duplication" if inserted in adjacent else "insertion"
    return None


def is_keyboard_neighbor_substitution(*, clean: str, typo: str) -> bool:
    """Return whether one case-preserving QWERTY-neighbor substitution occurred."""

    if not isinstance(clean, str) or not isinstance(typo, str) or len(clean) != len(typo):
        return False
    differences = [
        (clean_character, typo_character)
        for clean_character, typo_character in zip(clean, typo, strict=True)
        if clean_character != typo_character
    ]
    if len(differences) != 1:
        return False
    original, replacement = differences[0]
    return (
        original.lower() in _KEYBOARD_NEIGHBORS
        and replacement.lower() in _KEYBOARD_NEIGHBORS[original.lower()]
        and original.isupper() == replacement.isupper()
        and original.islower() == replacement.islower()
    )


__all__ = [
    "HELD_OUT_OPERATIONS",
    "TRAINING_OPERATIONS",
    "TypoGenerator",
    "classify_character_edit",
    "eligible_word_spans",
    "is_keyboard_neighbor_substitution",
]

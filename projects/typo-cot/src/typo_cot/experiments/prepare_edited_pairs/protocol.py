"""Pure implementation of the paper's token targeting and edit protocol.

AttnLRP ranks tokens, while activation patching operates on the distinct words
that actually changed.  This module keeps those two levels separate.  It also
retains the cumulative-offset behavior used by the reported experiment: an
earlier edit's length change is applied to every later-ranked target, even when
that target occurs earlier in the text.  The resulting landing flag makes the
paper's targeting-fidelity audit reproducible instead of silently correcting
the experimental inputs.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

Targeting = Literal["attribution-4", "random-4"]
Operation = Literal["substitution", "duplication", "deletion"]

_TARGETING_VALUES = frozenset(("attribution-4", "random-4"))
_ATTRIBUTION_TARGET_COUNT = 4
_NUMERIC_TOKEN = re.compile(r"[\d,.]+")
_OPTION_MARKER = re.compile(r"\([A-Ja-j]\)")
_WORD = re.compile(r"\S+")

_QWERTY_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "q": ("w", "a"),
    "w": ("q", "e", "a", "s"),
    "e": ("w", "r", "s", "d"),
    "r": ("e", "t", "d", "f"),
    "t": ("r", "y", "f", "g"),
    "y": ("t", "u", "g", "h"),
    "u": ("y", "i", "h", "j"),
    "i": ("u", "o", "j", "k"),
    "o": ("i", "p", "k", "l"),
    "p": ("o", "l"),
    "a": ("q", "w", "s", "z"),
    "s": ("a", "w", "e", "d", "z", "x"),
    "d": ("s", "e", "r", "f", "x", "c"),
    "f": ("d", "r", "t", "g", "c", "v"),
    "g": ("f", "t", "y", "h", "v", "b"),
    "h": ("g", "y", "u", "j", "b", "n"),
    "j": ("h", "u", "i", "k", "n", "m"),
    "k": ("j", "i", "o", "l", "m"),
    "l": ("k", "o", "p"),
    "z": ("a", "s", "x"),
    "x": ("z", "s", "d", "c"),
    "c": ("x", "d", "f", "v"),
    "v": ("c", "f", "g", "b"),
    "b": ("v", "g", "h", "n"),
    "n": ("b", "h", "j", "m"),
    "m": ("n", "j", "k"),
}


class PairProtocolError(ValueError):
    """Raised when a pair cannot satisfy the frozen preparation protocol."""


@dataclass(frozen=True, slots=True)
class CharSpan:
    """Half-open character span."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid character span: [{self.start}, {self.end})")


@dataclass(frozen=True, slots=True)
class CandidateToken:
    """One eligible prompt token and its signed AttnLRP relevance."""

    token_index: int
    text: str
    relevance: float
    prompt_start: int
    prompt_end: int
    attribution_rank: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateOrder:
    """Ordered edit candidates and the Attribution-k exclusion set."""

    candidates: tuple[CandidateToken, ...]
    excluded_top: tuple[CandidateToken, ...]


@dataclass(frozen=True, slots=True)
class CharacterEdit:
    """One applicable Table 4 character edit."""

    original: str
    edited: str
    operation: Operation
    character_index: int
    original_character: str
    new_character: str | None


@dataclass(frozen=True, slots=True)
class EditAttempt:
    """Provenance for one successfully applied ranked target attempt."""

    selection_rank: int
    attribution_rank: int
    target_token_index: int
    target_token_text: str
    relevance: float
    intended_prompt_span: CharSpan
    intended_editable_span: CharSpan
    landed_editable_span_before: CharSpan
    landed_text_before: str
    landed_origin_index: int | None
    landed_on_intended_token: bool
    intended_word_index: int
    landed_word_index: int
    operation: Operation
    character_index: int
    original_character: str
    new_character: str | None
    edited_token_text: str


@dataclass(frozen=True, slots=True)
class EditApplication:
    """Final edited text and its successful target attempts."""

    edited_text: str
    attempts: tuple[EditAttempt, ...]


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One actual distinct edited word aligned at its final prompt token."""

    word_index: int
    clean_text: str
    edited_text: str
    clean_editable_span: CharSpan
    edited_editable_span: CharSpan
    clean_prompt_span: CharSpan
    edited_prompt_span: CharSpan
    target_ranks: tuple[int, ...]
    target_token_indices: tuple[int, ...]
    clean_token_indices: tuple[int, ...]
    edited_token_indices: tuple[int, ...]
    clean_final_token: int
    edited_final_token: int


def _stable_seed(*parts: object) -> int:
    """Return a process-independent seed for one sample/selection/edit."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _is_ascii_letter(character: str) -> bool:
    return character.isascii() and character.isalpha()


def _is_eligible_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _NUMERIC_TOKEN.fullmatch(stripped):
        return False
    return any(_is_ascii_letter(character) for character in text)


def _is_option_marker_span(
    prompt_text: str,
    start: int,
    end: int,
    choice_list_start: int | None,
) -> bool:
    """Recognize a possibly split marker only inside the choice-list region."""
    if choice_list_start is None:
        return False
    word_start = start
    while word_start > 0 and not prompt_text[word_start - 1].isspace():
        word_start -= 1
    word_end = end
    while word_end < len(prompt_text) and not prompt_text[word_end].isspace():
        word_end += 1
    if word_start < choice_list_start:
        return False
    return _OPTION_MARKER.fullmatch(prompt_text[word_start:word_end]) is not None


def eligible_candidates(
    *,
    prompt_text: str,
    token_texts: Sequence[str],
    relevances: Sequence[float],
    offsets: Sequence[tuple[int, int]],
    editable_prompt_start: int,
    editable_prompt_end: int,
    choice_list_start: int | None = None,
) -> list[CandidateToken]:
    """Return nonzero, editable tokens fully inside the item input span."""
    if not (len(token_texts) == len(relevances) == len(offsets)):
        raise PairProtocolError("token texts, relevances, and offsets must have identical lengths")
    if editable_prompt_start < 0 or editable_prompt_end <= editable_prompt_start:
        raise PairProtocolError("editable prompt span must be non-empty")
    if choice_list_start is not None and not (
        editable_prompt_start <= choice_list_start <= editable_prompt_end
    ):
        raise PairProtocolError("choice-list start must lie inside the editable prompt span")

    candidates: list[CandidateToken] = []
    for token_index, (text, relevance, offset) in enumerate(
        zip(token_texts, relevances, offsets, strict=True)
    ):
        start, end = offset
        if start < 0 or end < start or end > len(prompt_text):
            raise PairProtocolError(f"token {token_index} has an invalid prompt offset: {offset}")
        score = float(relevance)
        if not math.isfinite(score):
            raise PairProtocolError(f"token {token_index} has non-finite relevance: {score}")
        if start < editable_prompt_start or end > editable_prompt_end or end <= start:
            continue
        if (
            abs(score) <= 1e-9
            or not _is_eligible_token(text)
            or _is_option_marker_span(prompt_text, start, end, choice_list_start)
        ):
            continue
        candidates.append(CandidateToken(token_index, text, score, start, end))
    return candidates


def order_candidates(
    candidates: Sequence[CandidateToken],
    *,
    targeting: Targeting,
    num_edits: int,
    seed: int,
    sample_id: str,
) -> CandidateOrder:
    """Apply Attribution-4 or the within-item Random-4 control ordering."""
    if targeting not in _TARGETING_VALUES:
        raise PairProtocolError(f"unknown targeting condition: {targeting!r}")
    if num_edits <= 0:
        raise PairProtocolError("num_edits must be positive")

    ranked = sorted(candidates, key=lambda item: (-abs(item.relevance), item.token_index))
    ranked = [replace(item, attribution_rank=rank) for rank, item in enumerate(ranked, 1)]
    if targeting == "attribution-4":
        return CandidateOrder(tuple(ranked), ())

    excluded = tuple(ranked[:_ATTRIBUTION_TARGET_COUNT])
    controls = ranked[_ATTRIBUTION_TARGET_COUNT:]
    rng = random.Random(_stable_seed(seed, sample_id, "random-4-selection"))
    rng.shuffle(controls)
    return CandidateOrder(tuple(controls), excluded)


def seeded_character_edit(text: str, seed: int) -> CharacterEdit | None:
    """Choose one applicable Table 4 operation with the reported seeded order."""
    letter_positions = [i for i, character in enumerate(text) if _is_ascii_letter(character)]
    if not letter_positions:
        return None

    operations: list[Operation] = ["substitution", "duplication", "deletion"]
    if len(letter_positions) == 1:
        operations.remove("deletion")
    rng = random.Random(seed)
    rng.shuffle(operations)

    # Substitution and duplication apply to every ASCII letter; deletion was
    # removed above when the token has only one letter. Thus the first shuffled
    # operation is always applicable, and no fallback loop is necessary.
    operation = operations[0]
    character_index = rng.choice(letter_positions)
    original_character = text[character_index]
    if operation == "substitution":
        neighbors = _QWERTY_NEIGHBORS[original_character.lower()]
        new_character = rng.choice(neighbors)
        if original_character.isupper():
            new_character = new_character.upper()
        edited = text[:character_index] + new_character + text[character_index + 1 :]
    elif operation == "duplication":
        new_character = original_character
        edited = text[: character_index + 1] + original_character + text[character_index + 1 :]
    else:
        new_character = None
        edited = text[:character_index] + text[character_index + 1 :]

    return CharacterEdit(
        original=text,
        edited=edited,
        operation=operation,
        character_index=character_index,
        original_character=original_character,
        new_character=new_character,
    )


def _word_spans(text: str) -> list[CharSpan]:
    return [CharSpan(match.start(), match.end()) for match in _WORD.finditer(text)]


def _word_index_at(text: str, position: int) -> int:
    for index, span in enumerate(_word_spans(text)):
        if span.start <= position < span.end:
            return index
    raise PairProtocolError(f"character {position} does not belong to a word")


def _first_letter_position(text: str, span: CharSpan) -> int:
    for position in range(span.start, span.end):
        if _is_ascii_letter(text[position]):
            return position
    raise PairProtocolError(
        f"eligible token has no editable letter: {text[span.start : span.end]!r}"
    )


def _updated_origins(
    origins: list[int | None], start: int, end: int, edit: CharacterEdit
) -> list[int | None]:
    local = origins[start:end]
    index = edit.character_index
    if edit.operation == "substitution":
        if len(edit.edited) != len(edit.original):
            raise PairProtocolError("substitution changed token length")
        replacement = local
    elif edit.operation == "duplication":
        if len(edit.edited) != len(edit.original) + 1:
            raise PairProtocolError("duplication did not add one character")
        replacement = local[: index + 1] + [None] + local[index + 1 :]
    else:
        if len(edit.edited) != len(edit.original) - 1:
            raise PairProtocolError("deletion did not remove one character")
        replacement = local[:index] + local[index + 1 :]
    return origins[:start] + replacement + origins[end:]


def apply_paper_edits(
    *,
    editable_text: str,
    editable_prompt_start: int,
    candidate_order: Sequence[CandidateToken],
    num_edits: int,
    seed: int,
    sample_id: str,
    edit_token: Callable[[str, int], CharacterEdit | None] = seeded_character_edit,
) -> EditApplication:
    """Apply up to ``num_edits`` ranked attempts with paper-compatible offsets."""
    if num_edits <= 0:
        raise PairProtocolError("num_edits must be positive")

    current_text = editable_text
    origins: list[int | None] = list(range(len(editable_text)))
    cumulative_shift = 0
    attempts: list[EditAttempt] = []

    for candidate in candidate_order:
        if len(attempts) >= num_edits:
            break
        if candidate.attribution_rank is None or candidate.attribution_rank <= 0:
            raise PairProtocolError("candidate attribution_rank must be a positive integer")
        intended_start = candidate.prompt_start - editable_prompt_start
        intended_end = candidate.prompt_end - editable_prompt_start
        if (
            intended_start < 0
            or intended_end > len(editable_text)
            or intended_end <= intended_start
        ):
            continue

        landed_start = intended_start + cumulative_shift
        landed_end = intended_end + cumulative_shift
        if landed_start < 0 or landed_end > len(current_text) or landed_end <= landed_start:
            continue
        landed_text = current_text[landed_start:landed_end]
        edit_seed = _stable_seed(seed, sample_id, candidate.token_index, candidate.text)
        edit = edit_token(landed_text, edit_seed)
        if edit is None:
            continue
        if edit.original != landed_text or edit.edited == landed_text:
            raise PairProtocolError("character edit returned inconsistent text")
        if not (0 <= edit.character_index < len(landed_text)):
            raise PairProtocolError("character edit index is outside the landed token")

        affected_position = landed_start + edit.character_index
        landed_origin = origins[affected_position]
        intended_span = CharSpan(intended_start, intended_end)
        landed_span = CharSpan(landed_start, landed_end)
        intended_word = _word_index_at(
            editable_text, _first_letter_position(editable_text, intended_span)
        )
        landed_word = _word_index_at(current_text, affected_position)

        attempts.append(
            EditAttempt(
                selection_rank=len(attempts) + 1,
                attribution_rank=candidate.attribution_rank,
                target_token_index=candidate.token_index,
                target_token_text=candidate.text,
                relevance=candidate.relevance,
                intended_prompt_span=CharSpan(candidate.prompt_start, candidate.prompt_end),
                intended_editable_span=intended_span,
                landed_editable_span_before=landed_span,
                landed_text_before=landed_text,
                landed_origin_index=landed_origin,
                landed_on_intended_token=(
                    landed_origin is not None and intended_start <= landed_origin < intended_end
                ),
                intended_word_index=intended_word,
                landed_word_index=landed_word,
                operation=edit.operation,
                character_index=edit.character_index,
                original_character=edit.original_character,
                new_character=edit.new_character,
                edited_token_text=edit.edited,
            )
        )

        origins = _updated_origins(origins, landed_start, landed_end, edit)
        current_text = current_text[:landed_start] + edit.edited + current_text[landed_end:]
        cumulative_shift += len(edit.edited) - len(landed_text)

    return EditApplication(current_text, tuple(attempts))


def _overlapping_tokens(offsets: Sequence[tuple[int, int]], span: CharSpan) -> tuple[int, ...]:
    return tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span.end and span.start < end
    )


def build_aligned_words(
    *,
    clean_editable: str,
    edited_editable: str,
    editable_prompt_start: int,
    attempts: Sequence[EditAttempt],
    clean_token_offsets: Sequence[tuple[int, int]],
    edited_token_offsets: Sequence[tuple[int, int]],
) -> tuple[AlignedWord, ...]:
    """Align each actual distinct changed word at its clean/edited final token."""
    clean_words = _word_spans(clean_editable)
    edited_words = _word_spans(edited_editable)
    if len(clean_words) != len(edited_words):
        raise PairProtocolError(
            "character operations changed whitespace-delimited word count; cannot align"
        )

    aligned: list[AlignedWord] = []
    for word_index, (clean_span, edited_span) in enumerate(
        zip(clean_words, edited_words, strict=True)
    ):
        clean_text = clean_editable[clean_span.start : clean_span.end]
        edited_text = edited_editable[edited_span.start : edited_span.end]
        if clean_text == edited_text:
            continue

        matching_attempts = [item for item in attempts if item.landed_word_index == word_index]
        if not matching_attempts:
            raise PairProtocolError(f"changed word {word_index} has no target-attempt provenance")
        clean_prompt_span = CharSpan(
            editable_prompt_start + clean_span.start,
            editable_prompt_start + clean_span.end,
        )
        edited_prompt_span = CharSpan(
            editable_prompt_start + edited_span.start,
            editable_prompt_start + edited_span.end,
        )
        clean_tokens = _overlapping_tokens(clean_token_offsets, clean_prompt_span)
        edited_tokens = _overlapping_tokens(edited_token_offsets, edited_prompt_span)
        if not clean_tokens or not edited_tokens:
            raise PairProtocolError(f"changed word {word_index} has no overlapping prompt token")

        aligned.append(
            AlignedWord(
                word_index=word_index,
                clean_text=clean_text,
                edited_text=edited_text,
                clean_editable_span=clean_span,
                edited_editable_span=edited_span,
                clean_prompt_span=clean_prompt_span,
                edited_prompt_span=edited_prompt_span,
                target_ranks=tuple(item.selection_rank for item in matching_attempts),
                target_token_indices=tuple(item.target_token_index for item in matching_attempts),
                clean_token_indices=clean_tokens,
                edited_token_indices=edited_tokens,
                clean_final_token=clean_tokens[-1],
                edited_final_token=edited_tokens[-1],
            )
        )
    return tuple(aligned)

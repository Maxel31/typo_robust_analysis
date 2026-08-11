"""Frozen within-word coordinate plans for subword patch comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

SubwordMode = Literal["first", "final", "all"]

_PRIMARY = "equal-count-primary"
_SECONDARY = "mismatch-monotone-secondary"
_EXACT = "exact-within-word-ordinal/v1"
_MONOTONE = "nearest-normalized-position-half-up-endpoints/v1"


@dataclass(frozen=True, slots=True)
class SubwordPatchPlan:
    """One mode's donor and recipient coordinates for a single pair."""

    mode: SubwordMode
    analysis_subset: str
    alignment: str
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"first", "final", "all"}:
            raise ValueError(f"unknown subword mode: {self.mode!r}")
        if self.analysis_subset not in {_PRIMARY, _SECONDARY}:
            raise ValueError("unknown subword analysis subset")
        if not self.source_positions or not self.destination_positions:
            raise ValueError("subword patch coordinates must be non-empty")
        if len(self.source_positions) != len(self.destination_positions):
            raise ValueError("subword patch coordinate cardinalities differ")
        if len(self.destination_positions) != len(set(self.destination_positions)):
            raise ValueError("subword destination coordinates contain duplicates")


def _positions(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be an integer sequence")
    positions = tuple(value)
    if not positions or any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0
        for position in positions
    ):
        raise ValueError(f"{field} must contain non-negative integers")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError(f"{field} must be strictly increasing and unique")
    return positions


def _words(
    edits: Sequence[Mapping[str, object]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes)) or not edits:
        raise ValueError("edits must contain one or more aligned words")
    words: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise TypeError(f"edits[{index}] must be an object")
        clean = _positions(
            edit.get("clean_token_indices"), field=f"edits[{index}].clean_token_indices"
        )
        typo = _positions(
            edit.get("typo_token_indices"), field=f"edits[{index}].typo_token_indices"
        )
        if edit.get("clean_word_final_token") != clean[-1]:
            raise ValueError(f"edits[{index}] clean final token differs")
        if edit.get("typo_word_final_token") != typo[-1]:
            raise ValueError(f"edits[{index}] typo final token differs")
        words.append((clean, typo))
    clean_union = tuple(position for clean, _typo in words for position in clean)
    typo_union = tuple(position for _clean, typo in words for position in typo)
    if len(clean_union) != len(set(clean_union)) or len(typo_union) != len(set(typo_union)):
        raise ValueError("aligned words must not share token coordinates")
    return tuple(words)


def _nearest_clean_index(*, typo_index: int, clean_count: int, typo_count: int) -> int:
    """Map normalized positions with integer half-up rounding and fixed endpoints."""

    if clean_count == 1 or typo_count == 1:
        return clean_count - 1
    numerator = 2 * typo_index * (clean_count - 1) + (typo_count - 1)
    denominator = 2 * (typo_count - 1)
    return numerator // denominator


def plan_subword_patch(edits: Sequence[Mapping[str, object]], *, mode: str) -> SubwordPatchPlan:
    """Plan first, final, or all-subword transfer without data-adaptive choices."""

    if mode not in {"first", "final", "all"}:
        raise ValueError("mode must be first, final, or all")
    words = _words(edits)
    equal = all(len(clean) == len(typo) for clean, typo in words)
    # The confirmatory comparison is paired across all three modes.  Keep one
    # pair-level subset label even though first/final coordinates themselves do
    # not require equal token counts; otherwise the modes would use different
    # denominators and cease to be a paired first/final/all comparison.
    subset = _PRIMARY if equal else _SECONDARY
    source: list[int] = []
    destination: list[int] = []
    if mode == "first":
        for clean, typo in words:
            source.append(clean[0])
            destination.append(typo[0])
        alignment = "word-first-endpoint/v1"
    elif mode == "final":
        for clean, typo in words:
            source.append(clean[-1])
            destination.append(typo[-1])
        alignment = "word-final-endpoint/v1"
    else:
        for clean, typo in words:
            destination.extend(typo)
            if len(clean) == len(typo):
                source.extend(clean)
            else:
                source.extend(
                    clean[
                        _nearest_clean_index(
                            typo_index=index,
                            clean_count=len(clean),
                            typo_count=len(typo),
                        )
                    ]
                    for index in range(len(typo))
                )
        alignment = _EXACT if equal else _MONOTONE
    return SubwordPatchPlan(
        mode=mode,  # type: ignore[arg-type]
        analysis_subset=subset,
        alignment=alignment,
        source_positions=tuple(source),
        destination_positions=tuple(destination),
    )


__all__ = ["SubwordMode", "SubwordPatchPlan", "plan_subword_patch"]

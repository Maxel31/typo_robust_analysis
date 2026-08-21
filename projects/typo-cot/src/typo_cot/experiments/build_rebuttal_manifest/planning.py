"""Outcome-independent coordinate and held-out plans for rebuttal pairs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _positions(values: Sequence[int], *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of integer token indices") from exc
    if not normalized and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in normalized
    ):
        raise ValueError(f"{field} must contain only non-negative integer token indices")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicate token indices")
    return normalized


def _prompt_length(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 2:
        raise ValueError(f"{field} must be an integer greater than two")
    return value


@dataclass(frozen=True, slots=True)
class StrictOffsetPlan:
    """All-or-nothing +N source/write coordinates for one pair."""

    valid: bool
    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]
    input_pairs: int
    offset_tokens: int
    invalid_reason: str | None


def plan_strict_offset_control(
    clean_positions: Sequence[int],
    typo_positions: Sequence[int],
    clean_edited_tokens: Sequence[int],
    typo_edited_tokens: Sequence[int],
    clean_prompt_length: int,
    typo_prompt_length: int,
    *,
    offset: int = 2,
) -> StrictOffsetPlan:
    """Shift every aligned endpoint or invalidate the complete offset arm.

    Unlike the historical coordinate helper, this confirmatory plan never
    drops only some edited words. Every shifted coordinate must remain in the
    prompt interior and outside every token occupied by an edited word. This
    preserves intervention cardinality across the correct and offset arms.
    """

    clean = _positions(clean_positions, field="clean positions", allow_empty=True)
    typo = _positions(typo_positions, field="typo positions", allow_empty=True)
    clean_edited = frozenset(
        _positions(clean_edited_tokens, field="clean edited tokens", allow_empty=not clean)
    )
    typo_edited = frozenset(
        _positions(typo_edited_tokens, field="typo edited tokens", allow_empty=not typo)
    )
    clean_length = _prompt_length(clean_prompt_length, field="clean prompt length")
    typo_length = _prompt_length(typo_prompt_length, field="typo prompt length")
    if len(clean) != len(typo):
        raise ValueError("clean and typo positions must have the same cardinality")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if offset == 0:
        raise ValueError("offset control requires a non-zero offset")
    if not clean:
        if clean_edited or typo_edited:
            raise ValueError("empty endpoints require empty edited-token sets")
        return StrictOffsetPlan(False, (), (), 0, offset, "no-aligned-edited-words")

    shifted_clean = tuple(position + offset for position in clean)
    shifted_typo = tuple(position + offset for position in typo)
    invalid_reason: str | None = None
    if any(not 1 <= position < clean_length - 1 for position in shifted_clean):
        invalid_reason = "clean-offset-outside-prompt-interior"
    elif any(not 1 <= position < typo_length - 1 for position in shifted_typo):
        invalid_reason = "typo-offset-outside-prompt-interior"
    elif any(position in clean_edited for position in shifted_clean):
        invalid_reason = "clean-offset-overlaps-edited-token"
    elif any(position in typo_edited for position in shifted_typo):
        invalid_reason = "typo-offset-overlaps-edited-token"

    if invalid_reason is not None:
        return StrictOffsetPlan(False, (), (), len(clean), offset, invalid_reason)
    return StrictOffsetPlan(
        True,
        shifted_clean,
        shifted_typo,
        len(clean),
        offset,
        None,
    )


@dataclass(frozen=True, slots=True)
class WindowSplitCandidate:
    """One restoration pair and its model/task/target-rule split identity."""

    pair_id: str
    stratum: tuple[str, ...]
    sample_group: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("pair_id must be a non-empty string")
        if not isinstance(self.stratum, tuple) or not self.stratum:
            raise ValueError("split stratum must be a non-empty tuple")
        if any(not isinstance(value, str) or not value for value in self.stratum):
            raise ValueError("split stratum must contain only non-empty strings")
        if len(self.stratum) < 2:
            raise ValueError("split stratum must identify model and shared task strata")
        if not isinstance(self.sample_group, str) or not self.sample_group:
            raise ValueError("sample_group must be a non-empty string")


@dataclass(frozen=True, slots=True)
class WindowSplitPlan:
    """Canonical diagnostic-selection and untouched-evaluation pair IDs."""

    selection_pair_ids: tuple[str, ...]
    evaluation_pair_ids: tuple[str, ...]


def _split_key(group_key: str, *, seed: int) -> tuple[str, str]:
    payload = f"held-out-window-split/v2\0{seed}\0{group_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), group_key


def plan_window_split(
    candidates: Iterable[WindowSplitCandidate],
    *,
    seed: int,
) -> WindowSplitPlan:
    """Create a balanced, stratified split without reading intervention outcomes."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("split seed must be an integer")
    groups: dict[tuple[str, ...], dict[str, list[WindowSplitCandidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, WindowSplitCandidate):
            raise TypeError("split candidates must be WindowSplitCandidate instances")
        if candidate.pair_id in seen:
            raise ValueError(f"duplicate pair_id in window split: {candidate.pair_id!r}")
        seen.add(candidate.pair_id)
        # The benchmark item, not the model/targeting realization, is the
        # independent split unit. Keeping only task here prevents the same
        # clean question from informing selection through another model or
        # typo-targeting condition.
        shared_stratum = candidate.stratum[1:2]
        group = groups[shared_stratum][candidate.sample_group]
        if any(existing.stratum == candidate.stratum for existing in group):
            raise ValueError(
                "window split contains duplicate model/sample membership: "
                f"{candidate.stratum!r}/{candidate.sample_group!r}"
            )
        group.append(candidate)

    selection: list[str] = []
    evaluation: list[str] = []
    for stratum, sample_groups in sorted(groups.items()):
        if len(sample_groups) < 2:
            raise ValueError(
                "every window-split shared stratum requires at least two sample groups: "
                f"stratum={stratum!r}"
            )
        ordered = sorted(
            sample_groups,
            key=lambda sample_group: _split_key("\0".join((*stratum, sample_group)), seed=seed),
        )
        split_at = len(ordered) // 2
        for sample_group in ordered[:split_at]:
            selection.extend(candidate.pair_id for candidate in sample_groups[sample_group])
        for sample_group in ordered[split_at:]:
            evaluation.extend(candidate.pair_id for candidate in sample_groups[sample_group])

    return WindowSplitPlan(tuple(sorted(selection)), tuple(sorted(evaluation)))


__all__ = [
    "StrictOffsetPlan",
    "WindowSplitCandidate",
    "WindowSplitPlan",
    "plan_strict_offset_control",
    "plan_window_split",
]

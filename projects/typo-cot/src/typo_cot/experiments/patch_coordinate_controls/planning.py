"""Deterministic coordinate and donor plans for the paper controls.

The final paper compares the edited-word patch with two controls.  The offset
control shifts both sides of every aligned endpoint by the same number of
tokens.  The cross-item control keeps the recipient coordinates but replaces
the clean donor with a different item from the same targeting/word-count
stratum.  These helpers construct both plans without loading a model so their
invariants can be checked before GPU work starts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OffsetPlan:
    """Paired source/destination coordinates retained by an offset control."""

    source_positions: tuple[int, ...]
    destination_positions: tuple[int, ...]
    input_pairs: int
    dropped_pairs: int


@dataclass(frozen=True, slots=True)
class DonorItem:
    """Minimal metadata used to select a matched non-self clean donor."""

    targeting: str
    sample_id: str
    n_positions: int


def _positions(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    """Materialize and validate a sequence of token coordinates."""

    try:
        positions = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of integer coordinates") from exc
    if any(isinstance(position, bool) or not isinstance(position, int) for position in positions):
        raise ValueError(f"{field} must contain only integer coordinates")
    return positions


def _prompt_length(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def plan_offset_positions(
    clean_positions: Sequence[int],
    edited_positions: Sequence[int],
    clean_prompt_length: int,
    edited_prompt_length: int,
    *,
    offset: int = 2,
) -> OffsetPlan:
    """Shift aligned clean/edited endpoints and retain valid non-edited pairs.

    A shifted pair is retained only when both coordinates are in
    ``[1, prompt_length - 1)`` and neither lands on an original edited-word
    endpoint on its respective side.  This is the exact validity rule used by
    the historical experiment: index zero and the final prompt token are not
    patch targets.  Input pairing, order, and duplicate occurrences are
    preserved.
    """

    clean = _positions(clean_positions, field="clean positions")
    edited = _positions(edited_positions, field="edited positions")
    clean_length = _prompt_length(clean_prompt_length, field="clean prompt length")
    edited_length = _prompt_length(edited_prompt_length, field="edited prompt length")
    if len(clean) != len(edited):
        raise ValueError(
            "clean and edited positions must have the same cardinality: "
            f"{len(clean)} != {len(edited)}"
        )
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if offset == 0:
        raise ValueError("offset control requires a non-zero offset")

    clean_endpoints = set(clean)
    edited_endpoints = set(edited)
    source: list[int] = []
    destination: list[int] = []
    for clean_position, edited_position in zip(clean, edited, strict=True):
        shifted_clean = clean_position + offset
        shifted_edited = edited_position + offset
        if (
            1 <= shifted_clean < clean_length - 1
            and 1 <= shifted_edited < edited_length - 1
            and shifted_clean not in clean_endpoints
            and shifted_edited not in edited_endpoints
        ):
            source.append(shifted_clean)
            destination.append(shifted_edited)

    retained = len(source)
    return OffsetPlan(
        source_positions=tuple(source),
        destination_positions=tuple(destination),
        input_pairs=len(clean),
        dropped_pairs=len(clean) - retained,
    )


def assign_cross_item_donors(items: Iterable[DonorItem]) -> dict[tuple[str, str], str]:
    """Assign a deterministic non-self donor within every matching stratum.

    Items are grouped by ``(targeting, n_positions)``.  Sorted sample IDs are
    then mapped to the next ID with a cyclic shift of one.  Sorting makes the
    plan independent of source iteration order.  A singleton is a hard error
    because silently dropping it would change the paired denominator.
    """

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, DonorItem):
            raise TypeError("cross-item donor entries must be DonorItem instances")
        if not item.targeting:
            raise ValueError("cross-item donor targeting must not be empty")
        if not item.sample_id:
            raise ValueError("cross-item donor sample_id must not be empty")
        if (
            isinstance(item.n_positions, bool)
            or not isinstance(item.n_positions, int)
            or item.n_positions <= 0
        ):
            raise ValueError(
                "cross-item donor position count must be positive: "
                f"targeting={item.targeting!r}, sample_id={item.sample_id!r}, "
                f"n_positions={item.n_positions!r}"
            )
        recipient_key = (item.targeting, item.sample_id)
        if recipient_key in seen:
            raise ValueError(f"duplicate cross-item donor recipient: {recipient_key!r}")
        seen.add(recipient_key)
        groups[(item.targeting, item.n_positions)].append(item.sample_id)

    donors: dict[tuple[str, str], str] = {}
    for (targeting, n_positions), sample_ids in sorted(groups.items()):
        sample_ids.sort()
        if len(sample_ids) == 1:
            raise ValueError(
                "cross-item donor stratum is singleton: "
                f"targeting={targeting!r}, n_positions={n_positions}, "
                f"sample_id={sample_ids[0]!r}"
            )
        for index, sample_id in enumerate(sample_ids):
            donors[(targeting, sample_id)] = sample_ids[(index + 1) % len(sample_ids)]
    return donors


__all__ = [
    "DonorItem",
    "OffsetPlan",
    "assign_cross_item_donors",
    "plan_offset_positions",
]

"""Training-only natural typo statistics for synthetic perturbations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from typo_robust_training.data.perturb import (
    HELD_OUT_OPERATIONS,
    TRAINING_OPERATIONS,
    classify_character_edit,
)
from typo_robust_training.data.records import NaturalTypoRecord


def _changed_fragments(clean: str, typo: str) -> tuple[str, str]:
    prefix = 0
    while prefix < min(len(clean), len(typo)) and clean[prefix] == typo[prefix]:
        prefix += 1
    clean_stop, typo_stop = len(clean), len(typo)
    while (
        clean_stop > prefix and typo_stop > prefix and clean[clean_stop - 1] == typo[typo_stop - 1]
    ):
        clean_stop -= 1
        typo_stop -= 1
    return clean[prefix:clean_stop], typo[prefix:typo_stop]


def derive_natural_typo_statistics(
    records: Sequence[NaturalTypoRecord],
) -> dict[str, object]:
    """Derive character counts from already partitioned training repositories."""

    if not records:
        raise ValueError("natural training repositories contain no eligible records")
    operation_counts = Counter({name: 0 for name in (*TRAINING_OPERATIONS, *HELD_OUT_OPERATIONS)})
    substitutions: dict[str, Counter[str]] = defaultdict(Counter)
    contributing_ids: list[str] = []
    for record in sorted(records, key=lambda candidate: candidate.record_id):
        if not isinstance(record, NaturalTypoRecord):
            raise TypeError("natural typo statistics require NaturalTypoRecord values")
        observed = classify_character_edit(clean=record.clean_text, typo=record.typo_text)
        if observed != record.operation:
            raise ValueError(f"natural typo operation differs for {record.record_id}")
        if observed in HELD_OUT_OPERATIONS:
            raise ValueError("held-out typo operation reached training statistics")
        if observed not in TRAINING_OPERATIONS:
            continue
        operation_counts[observed] += 1
        contributing_ids.append(record.record_id)
        if observed != "natural-statistics-substitution":
            continue
        clean_change, typo_change = _changed_fragments(record.clean_text, record.typo_text)
        clean_character = clean_change.casefold()
        typo_character = typo_change.casefold()
        if (
            len(clean_character) == 1
            and len(typo_character) == 1
            and clean_character.isascii()
            and typo_character.isascii()
            and clean_character.isalpha()
            and typo_character.isalpha()
            and clean_character != typo_character
        ):
            substitutions[clean_character][typo_character] += 1
    if not substitutions:
        raise ValueError("natural training repositories contain no character substitutions")
    identity_material = "\n".join(contributing_ids).encode("utf-8")
    return {
        "schema_version": "natural-typo-statistics/v1",
        "source_role": "natural-training-repositories-only",
        "input_records": len(records),
        "contributing_records": len(contributing_ids),
        "contributing_record_ids_sha256": hashlib.sha256(identity_material).hexdigest(),
        "held_out_operations": sorted(HELD_OUT_OPERATIONS),
        "operation_counts": dict(sorted(operation_counts.items())),
        "substitutions": {
            clean: dict(sorted(replacements.items()))
            for clean, replacements in sorted(substitutions.items())
        },
    }


def substitutions_from_statistics(
    payload: Mapping[str, object],
) -> Mapping[str, Mapping[str, int]]:
    """Validate and return the nested replacement-count table."""

    if payload.get("schema_version") != "natural-typo-statistics/v1":
        raise ValueError("natural typo statistics schema differs")
    substitutions = payload.get("substitutions")
    if not isinstance(substitutions, Mapping):
        raise ValueError("natural typo substitutions must be an object")
    try:
        canonical = json.loads(json.dumps(substitutions, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("natural typo substitutions are not JSON-compatible") from exc
    if not isinstance(canonical, dict):
        raise ValueError("natural typo substitutions must be an object")
    return canonical


__all__ = ["derive_natural_typo_statistics", "substitutions_from_statistics"]

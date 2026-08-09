"""Frozen Appendix E protocol for edited-word restoration order accuracy."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PAPER_MODELS = (
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
PAPER_BENCHMARKS = ("gsm8k", "mmlu")
PAPER_ORDERS = (
    "high-relevance-first",
    "seeded-random",
    "low-relevance-first",
)
PAPER_BUDGETS = (0, 1, 2, 3, 4)
INTERMEDIATE_BUDGETS = (1, 2, 3)
PAPER_SEED = 42
PAPER_BATCH_SIZE = 8
PAPER_SOURCE_RECORD_COUNTS = {"gsm8k": 1319, "mmlu": 2850}
PAPER_PROMPT_TEMPLATES = {
    "gsm8k": {
        "prompt_id": "gsm8k_cot_v1",
        "probe_sha256": (
            "b4e6d6087551c55558c1afe236c68bbcc83a00f69864d41641fa1a45d77ca22f"
        ),
    },
    "mmlu": {
        "prompt_id": "mmlu_cot_v1",
        "probe_sha256": (
            "eb9d417152748c6eebef118a8e5717919c8f10fac601e12a0376a10022538606"
        ),
    },
}

ALL_CONDITION_IDS = (
    "edited:k0",
    *(f"{order}:k{budget}" for order in PAPER_ORDERS for budget in INTERMEDIATE_BUDGETS),
    "clean:k4",
)

GENERATION = {
    "strategy": "greedy",
    "dtype": "bfloat16",
    "padding_side": "left",
    "max_new_tokens": 512,
    "batch_size": PAPER_BATCH_SIZE,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
}

PROTOCOL = {
    "version": "restoration-order-accuracy-protocol/v1",
    "paper_defined": {
        "models": list(PAPER_MODELS),
        "benchmarks": list(PAPER_BENCHMARKS),
        "orders": list(PAPER_ORDERS),
        "budgets": list(PAPER_BUDGETS),
        "cohort": "source-clean-correct-and-four-edit-wrong-before-fresh-generation",
        "pooling": "micro-by-model-task-sample-identity",
        "paired_test": "two-sided-exact-mcnemar-binomial-unadjusted",
    },
    "legacy_backed": {
        "prompt_templates": PAPER_PROMPT_TEMPLATES,
        "restoration_unit": "contiguous-difflib-non-equal-edit-group/v1",
        "alignment": "left-to-right-groups-to-target-attempts-sorted-by-token-index/v1",
        "relevance": "absolute-attnlrp-relevance-with-left-to-right-ties/v1",
        "random_order": "md5-seed-colon-sample-id-python-random-shuffle/v1",
        "generation": GENERATION,
    },
    "public_reproduction": {
        "answer_extraction": (
            "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
        ),
        "source": "completed-prepare-edited-pairs-attribution-4/v1",
        "source_artifact_binding": "completed-manifest-pairs-sha256/v1",
        "source_records": PAPER_SOURCE_RECORD_COUNTS,
        "source_prompt_revalidation": "archived-probe-plus-record-byte-match/v1",
        "source_outcome_revalidation": (
            "stored-continuation-final-pdf-extraction-match/v1"
        ),
        "source_generation_termination": "effective-eos-vs-length-cap/v1",
        "cross_setting_source_identity": (
            "model-revision-per-model-and-dataset-plus-ordered-samples-per-task/v1"
        ),
        "fresh_endpoint_refilter": False,
    },
}


def canonical_sha256(payload: object) -> str:
    """Hash finite canonical JSON for protocol and artifact identities."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


class EditGroupingError(ValueError):
    """Raised when submitted edit events cannot map to character edit groups."""


@dataclass(frozen=True, slots=True)
class EditGroup:
    """One contiguous non-equal character span and its stored edit event."""

    index: int
    clean_start: int
    clean_end: int
    edited_start: int
    edited_end: int
    clean_text: str
    edited_text: str
    selection_rank: int
    target_token_index: int
    relevance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "clean_span": {"start": self.clean_start, "end": self.clean_end},
            "edited_span": {"start": self.edited_start, "end": self.edited_end},
            "clean_text": self.clean_text,
            "edited_text": self.edited_text,
            "selection_rank": self.selection_rank,
            "target_token_index": self.target_token_index,
            "relevance": self.relevance,
        }


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise EditGroupingError(f"target_attempt {field} must be an integer >= {minimum}")
    return value


def _attempts(
    attempts: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int, float], ...]:
    parsed: list[tuple[int, int, float]] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise EditGroupingError(f"target_attempt {index} must be an object")
        token_index = _integer(
            attempt.get("target_token_index"), field="target_token_index"
        )
        selection_rank = _integer(
            attempt.get("selection_rank"), field="selection_rank", minimum=1
        )
        relevance_value = attempt.get("relevance")
        if (
            not isinstance(relevance_value, (int, float))
            or isinstance(relevance_value, bool)
            or not math.isfinite(float(relevance_value))
        ):
            raise EditGroupingError("target_attempt relevance must be finite")
        parsed.append((token_index, selection_rank, float(relevance_value)))
    parsed.sort(key=lambda item: (item[0], item[1]))
    if len({item[0] for item in parsed}) != len(parsed):
        raise EditGroupingError(
            "target_attempt token indices must be unique for one-to-one edit grouping"
        )
    return tuple(parsed)


def build_edit_groups(
    clean: str,
    edited: str,
    target_attempts: Sequence[Mapping[str, object]],
) -> tuple[EditGroup, ...]:
    """Build submitted-compatible difflib groups and bind events left-to-right."""
    if not isinstance(clean, str) or not isinstance(edited, str):
        raise TypeError("clean and edited text must be strings")
    parsed_attempts = _attempts(target_attempts)
    raw_groups: list[list[int]] = []
    current: list[int] | None = None
    matcher = difflib.SequenceMatcher(a=clean, b=edited, autojunk=False)
    for tag, clean_start, clean_end, edited_start, edited_end in matcher.get_opcodes():
        if tag == "equal":
            if current is not None:
                raw_groups.append(current)
                current = None
            continue
        if current is None:
            current = [clean_start, clean_end, edited_start, edited_end]
        else:
            current[1] = clean_end
            current[3] = edited_end
    if current is not None:
        raw_groups.append(current)
    if not raw_groups or len(raw_groups) != len(parsed_attempts):
        raise EditGroupingError(
            "edit groups and target_attempts must have a non-empty one-to-one mapping"
        )

    groups = tuple(
        EditGroup(
            index=index,
            clean_start=span[0],
            clean_end=span[1],
            edited_start=span[2],
            edited_end=span[3],
            clean_text=clean[span[0] : span[1]],
            edited_text=edited[span[2] : span[3]],
            target_token_index=attempt[0],
            selection_rank=attempt[1],
            relevance=attempt[2],
        )
        for index, (span, attempt) in enumerate(zip(raw_groups, parsed_attempts, strict=True))
    )
    if restore_edit_groups(clean, edited, groups, ()) != edited:
        raise EditGroupingError("zero-group restoration does not reproduce the edited endpoint")
    if restore_edit_groups(clean, edited, groups, range(len(groups))) != clean:
        raise EditGroupingError("all-group restoration does not reproduce the clean endpoint")
    return groups


def restore_edit_groups(
    clean: str,
    edited: str,
    groups: Sequence[EditGroup],
    restore_group_indices: Sequence[int] | range,
) -> str:
    """Copy exact clean segments for selected groups and edited segments otherwise."""
    restore = tuple(restore_group_indices)
    if any(not isinstance(index, int) or isinstance(index, bool) for index in restore):
        raise ValueError("restoration group indices must be integers")
    if len(set(restore)) != len(restore):
        raise ValueError("restoration group indices contain a duplicate")
    if any(index < 0 or index >= len(groups) for index in restore):
        raise ValueError("restoration group index is outside the valid range")
    restored = set(restore)
    output: list[str] = []
    edited_cursor = 0
    clean_cursor = 0
    for expected_index, group in enumerate(groups):
        if group.index != expected_index:
            raise ValueError("edit group indices must be contiguous and ordered")
        clean_equal = clean[clean_cursor : group.clean_start]
        edited_equal = edited[edited_cursor : group.edited_start]
        if clean_equal != edited_equal:
            raise ValueError("text between edit groups is not equal")
        output.append(edited_equal)
        output.append(group.clean_text if group.index in restored else group.edited_text)
        clean_cursor = group.clean_end
        edited_cursor = group.edited_end
    clean_tail = clean[clean_cursor:]
    edited_tail = edited[edited_cursor:]
    if clean_tail != edited_tail:
        raise ValueError("text after the final edit group is not equal")
    output.append(edited_tail)
    return "".join(output)


def condition_id(order: str | None, budget: int) -> str:
    """Return the stable public arm identifier for one paper condition."""
    if budget == 0:
        return "edited:k0"
    if budget == 4:
        return "clean:k4"
    if order not in PAPER_ORDERS or budget not in INTERMEDIATE_BUDGETS:
        raise ValueError(f"unsupported restoration condition: order={order!r}, budget={budget}")
    return f"{order}:k{budget}"


__all__ = [
    "ALL_CONDITION_IDS",
    "EditGroup",
    "EditGroupingError",
    "GENERATION",
    "INTERMEDIATE_BUDGETS",
    "PAPER_BATCH_SIZE",
    "PAPER_BENCHMARKS",
    "PAPER_BUDGETS",
    "PAPER_MODELS",
    "PAPER_ORDERS",
    "PAPER_PROMPT_TEMPLATES",
    "PAPER_SEED",
    "PAPER_SOURCE_RECORD_COUNTS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "canonical_sha256",
    "build_edit_groups",
    "condition_id",
    "restore_edit_groups",
]

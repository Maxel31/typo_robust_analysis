"""Pure construction of the final-paper complete-text CoT-swap cells."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

CELL_ORDER = ("A", "B", "C", "D")
CELL_SIDES: dict[str, tuple[str, str]] = {
    "A": ("clean", "clean"),
    "B": ("edited", "edited"),
    "C": ("edited", "clean"),
    "D": ("clean", "edited"),
}

_TRIGGER = re.compile(r"[Tt]he answer is")
_RESIDUAL_PATTERNS = (
    re.compile(r"[Tt]he answer is"),
    re.compile(r"[Aa]nswer\s*[:=]"),
    re.compile(r"[Ff]inal [Aa]nswer"),
)
_EARLY_TRIGGER_RATIO = 0.25
_BOUNDARY_METHOD = "submitted-first-[Tt]he-answer-is-filter/v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _string(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PreAnswerBoundary:
    """Submitted-implementation answer-template boundary for one continuation."""

    text: str
    method: str
    trigger_found: bool
    trigger_count: int
    trigger_char_start: int | None
    trigger_ratio: float | None
    early_trigger: bool
    residual_fragment: bool
    continuation_sha256: str
    continuation_char_count: int
    text_sha256: str

    @property
    def eligible(self) -> bool:
        """Return whether this side passes every frozen template check."""

        return bool(
            self.trigger_found
            and self.trigger_count == 1
            and not self.early_trigger
            and not self.residual_fragment
        )

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        """Return stable public diagnostics, optionally omitting supplied text."""

        payload: dict[str, object] = {
            "method": self.method,
            "trigger_found": self.trigger_found,
            "trigger_count": self.trigger_count,
            "trigger_char_start": self.trigger_char_start,
            "trigger_ratio": self.trigger_ratio,
            "early_trigger": self.early_trigger,
            "residual_fragment": self.residual_fragment,
            "continuation_sha256": self.continuation_sha256,
            "continuation_char_count": self.continuation_char_count,
            "pre_answer_text_sha256": self.text_sha256,
            "pre_answer_char_count": len(self.text),
            "eligible": self.eligible,
        }
        if include_text:
            payload["pre_answer_text"] = self.text
        return payload


def locate_pre_answer(continuation: str) -> PreAnswerBoundary:
    """Cut immediately before the first submitted ``[Tt]he answer is`` trigger."""

    if not isinstance(continuation, str):
        raise TypeError("continuation must be a string")
    matches = list(_TRIGGER.finditer(continuation))
    if not matches:
        text = continuation
        return PreAnswerBoundary(
            text=text,
            method=_BOUNDARY_METHOD,
            trigger_found=False,
            trigger_count=0,
            trigger_char_start=None,
            trigger_ratio=None,
            early_trigger=False,
            residual_fragment=False,
            continuation_sha256=_sha256(continuation),
            continuation_char_count=len(continuation),
            text_sha256=_sha256(text),
        )
    start = matches[0].start()
    text = continuation[:start]
    ratio = start / len(continuation) if continuation else 0.0
    return PreAnswerBoundary(
        text=text,
        method=_BOUNDARY_METHOD,
        trigger_found=True,
        trigger_count=len(matches),
        trigger_char_start=start,
        trigger_ratio=ratio,
        early_trigger=ratio < _EARLY_TRIGGER_RATIO,
        residual_fragment=any(pattern.search(text) is not None for pattern in _RESIDUAL_PATTERNS),
        continuation_sha256=_sha256(continuation),
        continuation_char_count=len(continuation),
        text_sha256=_sha256(text),
    )


@dataclass(frozen=True, slots=True)
class CellPlan:
    """One fixed question/pre-answer combination in the A/B/C/D crossing."""

    cell: str
    question_side: str
    cot_side: str
    prompt: str
    pre_answer_text: str
    prompt_token_count: int

    def __post_init__(self) -> None:
        if self.cell not in CELL_ORDER:
            raise ValueError(f"unsupported CoT-swap cell: {self.cell!r}")
        if CELL_SIDES[self.cell] != (self.question_side, self.cot_side):
            raise ValueError(f"cell {self.cell} has the wrong question/CoT source mapping")
        if not self.prompt:
            raise ValueError("cell prompt must not be empty")
        if self.prompt_token_count <= 0:
            raise ValueError("cell prompt_token_count must be positive")

    @property
    def full_input(self) -> str:
        return self.prompt + self.pre_answer_text

    @property
    def prompt_sha256(self) -> str:
        return _sha256(self.prompt)

    @property
    def pre_answer_sha256(self) -> str:
        return _sha256(self.pre_answer_text)

    @property
    def full_input_sha256(self) -> str:
        return _sha256(self.full_input)

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "cell": self.cell,
            "question_side": self.question_side,
            "cot_side": self.cot_side,
            "prompt_sha256": self.prompt_sha256,
            "pre_answer_text_sha256": self.pre_answer_sha256,
            "full_input_text_sha256": self.full_input_sha256,
            "prompt_char_count": len(self.prompt),
            "pre_answer_char_count": len(self.pre_answer_text),
            "full_input_char_count": len(self.full_input),
            "prompt_token_count": self.prompt_token_count,
        }
        if include_text:
            payload.update(
                {
                    "prompt": self.prompt,
                    "pre_answer_text": self.pre_answer_text,
                    "full_input": self.full_input,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class CotSwapPlan:
    """Deterministic four-cell plan and pre-model exclusion diagnostics."""

    sample_id: str
    clean_boundary: PreAnswerBoundary
    edited_boundary: PreAnswerBoundary
    cells: tuple[CellPlan, ...]
    exclusion_reasons: tuple[str, ...]

    @property
    def edit_valid(self) -> bool:
        """Return whether the source record contains an applied prompt edit."""

        return "no-applied-edit" not in self.exclusion_reasons

    @property
    def template_eligible(self) -> bool:
        """Return answer-template eligibility after the edit-validity gate."""

        return self.clean_boundary.eligible and self.edited_boundary.eligible

    @property
    def eligible(self) -> bool:
        return self.edit_valid and self.template_eligible

    @property
    def fingerprint(self) -> str:
        import json

        payload = json.dumps(
            self.to_dict(include_text=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "template_filter": {
                "clean": self.clean_boundary.to_dict(include_text=include_text),
                "edited": self.edited_boundary.to_dict(include_text=include_text),
            },
            "cells": [cell.to_dict(include_text=include_text) for cell in self.cells],
            "exclusion_reasons": list(self.exclusion_reasons),
            "edit_valid": self.edit_valid,
            "template_eligible": self.template_eligible,
            "eligible": self.eligible,
        }


def _boundary_reasons(boundary: PreAnswerBoundary, side: str) -> list[str]:
    reasons: list[str] = []
    if not boundary.trigger_found:
        reasons.append(f"no-trigger-{side}")
        return reasons
    if boundary.trigger_count != 1:
        reasons.append(f"multiple-trigger-{side}")
    if boundary.early_trigger:
        reasons.append(f"early-trigger-{side}")
    if boundary.residual_fragment:
        reasons.append(f"residual-fragment-{side}")
    return reasons


def build_cell_plan(record: Mapping[str, object]) -> CotSwapPlan:
    """Cross the exact stored prompts and submitted-rule pre-answer prefixes."""

    sample_id = _string(record.get("sample_id"), field="sample_id")
    sides = {side: _mapping(record.get(side), field=side) for side in ("clean", "edited")}
    prompts = {side: _string(sides[side].get("prompt"), field=f"{side}.prompt") for side in sides}
    prompt_counts = {
        side: _positive_int(
            sides[side].get("prompt_token_count"),
            field=f"{side}.prompt_token_count",
        )
        for side in sides
    }
    continuations = {
        side: _string(
            sides[side].get("continuation"),
            field=f"{side}.continuation",
            allow_empty=True,
        )
        for side in sides
    }
    boundaries = {side: locate_pre_answer(continuations[side]) for side in sides}

    reasons: list[str] = []
    attempts = record.get("num_target_attempts")
    if prompts["clean"] == prompts["edited"] or attempts == 0:
        reasons.append("no-applied-edit")
    reasons.extend(_boundary_reasons(boundaries["clean"], "clean"))
    reasons.extend(_boundary_reasons(boundaries["edited"], "edited"))

    cells = tuple(
        CellPlan(
            cell=cell,
            question_side=CELL_SIDES[cell][0],
            cot_side=CELL_SIDES[cell][1],
            prompt=prompts[CELL_SIDES[cell][0]],
            pre_answer_text=boundaries[CELL_SIDES[cell][1]].text,
            prompt_token_count=prompt_counts[CELL_SIDES[cell][0]],
        )
        for cell in CELL_ORDER
    )
    return CotSwapPlan(
        sample_id=sample_id,
        clean_boundary=boundaries["clean"],
        edited_boundary=boundaries["edited"],
        cells=cells,
        exclusion_reasons=tuple(reasons),
    )

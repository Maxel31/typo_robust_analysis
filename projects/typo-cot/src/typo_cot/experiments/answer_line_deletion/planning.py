"""Pure construction of the final answer-line deletion intervention."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from typo_cot.experiments.answer_line_deletion.protocol import ARM_ORDER
from typo_cot.experiments.cot_swap.planning import build_cell_plan

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DELETION_METHOD = "submitted-final-nonempty-line/v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FinalLineDeletion:
    """Exact legacy-backed character deletion and its diagnostics."""

    original_text: str
    deleted_text: str
    deleted_line: str | None
    deleted_line_index: int | None
    original_line_count: int
    prefix_became_empty: bool
    method: str = _DELETION_METHOD

    @property
    def original_sha256(self) -> str:
        return _sha256(self.original_text)

    @property
    def deleted_sha256(self) -> str:
        return _sha256(self.deleted_text)

    @property
    def deleted_line_sha256(self) -> str | None:
        return _sha256(self.deleted_line) if self.deleted_line is not None else None

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "original_text_sha256": self.original_sha256,
            "deleted_text_sha256": self.deleted_sha256,
            "deleted_line_sha256": self.deleted_line_sha256,
            "original_char_count": len(self.original_text),
            "deleted_char_count": len(self.deleted_text),
            "deleted_line_char_count": (
                len(self.deleted_line) if self.deleted_line is not None else None
            ),
            "deleted_line_index": self.deleted_line_index,
            "original_line_count": self.original_line_count,
            "prefix_became_empty": self.prefix_became_empty,
        }
        if include_text:
            payload.update(
                {
                    "original_text": self.original_text,
                    "deleted_text": self.deleted_text,
                    "deleted_line": self.deleted_line,
                }
            )
        return payload


def strip_final_nonempty_line(prefix: str) -> FinalLineDeletion:
    """Delete the last non-empty line exactly as the submitted producer did."""

    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    lines = prefix.split("\n")
    index = next(
        (position for position in range(len(lines) - 1, -1, -1) if lines[position].strip()),
        None,
    )
    if index is None:
        deleted_text = ""
        deleted_line = None
    else:
        joined = "\n".join(lines[:index])
        deleted_text = joined + "\n" if joined.strip() else ""
        deleted_line = lines[index]
    return FinalLineDeletion(
        original_text=prefix,
        deleted_text=deleted_text,
        deleted_line=deleted_line,
        deleted_line_index=index,
        original_line_count=len(lines),
        prefix_became_empty=bool(prefix.strip()) and not bool(deleted_text.strip()),
    )


@dataclass(frozen=True, slots=True)
class AnswerLineArmPlan:
    """One fixed edited-question/clean-prefix input."""

    arm: str
    prompt: str
    pre_answer_text: str
    prompt_token_count: int

    def __post_init__(self) -> None:
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unsupported answer-line deletion arm: {self.arm!r}")
        if not self.prompt:
            raise ValueError("arm prompt must not be empty")
        if (
            not isinstance(self.prompt_token_count, int)
            or isinstance(self.prompt_token_count, bool)
            or self.prompt_token_count <= 0
        ):
            raise ValueError("arm prompt_token_count must be positive")

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
            "arm": self.arm,
            "prompt_text_sha256": self.prompt_sha256,
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
class AnswerLineDeletionPlan:
    """Deterministic two-arm plan bound to one validated CoT-swap case."""

    sample_id: str
    source_a_answer: str
    source_c_answer: str
    source_record_sha256: str
    prepared_record_sha256: str
    deletion: FinalLineDeletion
    arms: tuple[AnswerLineArmPlan, ...]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("plan sample_id must not be empty")
        if not self.source_a_answer:
            raise ValueError("source A answer must not be empty")
        _require_sha256(self.source_record_sha256, field="source_record_sha256")
        _require_sha256(self.prepared_record_sha256, field="prepared_record_sha256")
        if tuple(arm.arm for arm in self.arms) != ARM_ORDER:
            raise ValueError("answer-line deletion arms must be in protocol order")
        if self.deletion.original_text != self.arms[0].pre_answer_text:
            raise ValueError("complete arm does not match the deletion source text")
        if self.deletion.deleted_text != self.arms[1].pre_answer_text:
            raise ValueError("deleted arm does not match the deletion result")

    @property
    def fingerprint(self) -> str:
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
            "source_a_answer": self.source_a_answer,
            "source_c_answer": self.source_c_answer,
            "source_record_sha256": self.source_record_sha256,
            "prepared_record_sha256": self.prepared_record_sha256,
            "deletion": self.deletion.to_dict(include_text=include_text),
            "arms": [arm.to_dict(include_text=include_text) for arm in self.arms],
        }


def build_answer_line_deletion_plan(
    pair: dict[str, object],
    *,
    source_a_answer: str,
    source_c_answer: str,
    source_record_sha256: str,
    prepared_record_sha256: str,
) -> AnswerLineDeletionPlan:
    """Reconstruct condition C and apply the submitted final-line rule."""

    cot_plan = build_cell_plan(pair)
    if not cot_plan.eligible:
        raise ValueError("answer-line deletion received a CoT-swap-ineligible pair")
    cell_c = next(cell for cell in cot_plan.cells if cell.cell == "C")
    deletion = strip_final_nonempty_line(cell_c.pre_answer_text)
    arms = (
        AnswerLineArmPlan(
            arm="complete",
            prompt=cell_c.prompt,
            pre_answer_text=cell_c.pre_answer_text,
            prompt_token_count=cell_c.prompt_token_count,
        ),
        AnswerLineArmPlan(
            arm="answer-line-deleted",
            prompt=cell_c.prompt,
            pre_answer_text=deletion.deleted_text,
            prompt_token_count=cell_c.prompt_token_count,
        ),
    )
    return AnswerLineDeletionPlan(
        sample_id=cot_plan.sample_id,
        source_a_answer=source_a_answer,
        source_c_answer=source_c_answer,
        source_record_sha256=source_record_sha256,
        prepared_record_sha256=prepared_record_sha256,
        deletion=deletion,
        arms=arms,
    )

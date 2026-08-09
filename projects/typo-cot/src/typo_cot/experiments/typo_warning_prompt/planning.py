"""Pure construction of the paired typo-warning prompt intervention."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from typo_cot.experiments.typo_warning_prompt.protocol import (
    ARM_ORDER,
    PROMPT_MARKERS,
    WARNING_TEXT,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inject_typo_warning(prompt: str, *, benchmark: str, enabled: bool = True) -> str:
    """Insert the frozen warning at the unique verified task-final boundary."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    try:
        marker = PROMPT_MARKERS[benchmark]
    except KeyError:
        raise ValueError(f"unsupported typo-warning benchmark: {benchmark!r}") from None
    if not enabled:
        return prompt
    if prompt.count(marker) != 1:
        raise ValueError(f"typo-warning prompt must contain {marker!r} exactly once")
    marker_index = prompt.index(marker)
    return prompt[:marker_index] + WARNING_TEXT + "\n\n" + prompt[marker_index:]


@dataclass(frozen=True, slots=True)
class WarningPromptArmPlan:
    """One fixed prompt in the paired warning comparison."""

    arm: str
    prompt: str
    warning_enabled: bool
    marker: str
    marker_index: int

    def __post_init__(self) -> None:
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unsupported typo-warning arm: {self.arm!r}")
        if type(self.warning_enabled) is not bool:
            raise TypeError("warning_enabled must be boolean")
        if not self.prompt or self.prompt.count(self.marker) != 1:
            raise ValueError("arm prompt must contain its marker exactly once")
        if self.prompt.index(self.marker) != self.marker_index:
            raise ValueError("marker_index does not identify the arm marker")
        if self.warning_enabled != (self.arm == "with-warning"):
            raise ValueError("arm and warning_enabled disagree")

    @property
    def prompt_sha256(self) -> str:
        return _sha256(self.prompt)

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "arm": self.arm,
            "warning_enabled": self.warning_enabled,
            "prompt_text_sha256": self.prompt_sha256,
            "prompt_char_count": len(self.prompt),
            "marker": self.marker,
            "marker_index": self.marker_index,
        }
        if include_text:
            payload["prompt"] = self.prompt
        return payload


@dataclass(frozen=True, slots=True)
class WarningPromptPlan:
    """Deterministic paired plan bound to one Attribution-4 source row."""

    sample_id: str
    gold_answer: str
    benchmark: str
    source_record_sha256: str
    source_prompt_sha256: str
    arms: tuple[WarningPromptArmPlan, ...]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.gold_answer:
            raise ValueError("plan sample_id and gold_answer must be non-empty")
        if self.benchmark not in PROMPT_MARKERS:
            raise ValueError("plan benchmark is unsupported")
        for field in ("source_record_sha256", "source_prompt_sha256"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"{field} must be a full lowercase SHA-256")
        if tuple(arm.arm for arm in self.arms) != ARM_ORDER:
            raise ValueError("typo-warning arms must be in protocol order")
        source_prompt = self.arms[0].prompt
        if _sha256(source_prompt) != self.source_prompt_sha256:
            raise ValueError("without-warning arm differs from the source prompt")
        marker = PROMPT_MARKERS[self.benchmark]
        if any(arm.marker != marker for arm in self.arms):
            raise ValueError("warning arm marker differs from the benchmark marker")
        insertion = WARNING_TEXT + "\n\n"
        warned = self.arms[1].prompt
        warned_index = warned.index(marker)
        if (
            warned_index < len(insertion)
            or warned[warned_index - len(insertion) : warned_index] != insertion
        ):
            raise ValueError("with-warning arm does not contain the exact warning insertion")
        if warned[: warned_index - len(insertion)] + warned[warned_index:] != source_prompt:
            raise ValueError("with-warning arm is not an exact insertion into the source prompt")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_text=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "gold_answer": self.gold_answer,
            "benchmark": self.benchmark,
            "source_record_sha256": self.source_record_sha256,
            "source_prompt_sha256": self.source_prompt_sha256,
            "arms": [arm.to_dict(include_text=include_text) for arm in self.arms],
        }


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def build_warning_prompt_plan(
    pair: Mapping[str, object], *, source_record_sha256: str
) -> WarningPromptPlan:
    """Build both arms from a validated prepared Attribution-4 record."""

    sample_id = _nonempty_string(pair.get("sample_id"), field="pair.sample_id")
    benchmark = _nonempty_string(pair.get("benchmark"), field="pair.benchmark")
    if benchmark not in PROMPT_MARKERS:
        raise ValueError(f"unsupported typo-warning benchmark: {benchmark!r}")
    gold_answer = _nonempty_string(pair.get("gold_answer"), field="pair.gold_answer")
    edited = _mapping(pair.get("edited"), field="pair.edited")
    source_prompt = _nonempty_string(edited.get("prompt"), field="pair.edited.prompt")
    marker = PROMPT_MARKERS[benchmark]
    warned = inject_typo_warning(source_prompt, benchmark=benchmark)
    arms = (
        WarningPromptArmPlan(
            arm="without-warning",
            prompt=source_prompt,
            warning_enabled=False,
            marker=marker,
            marker_index=source_prompt.index(marker),
        ),
        WarningPromptArmPlan(
            arm="with-warning",
            prompt=warned,
            warning_enabled=True,
            marker=marker,
            marker_index=warned.index(marker),
        ),
    )
    return WarningPromptPlan(
        sample_id=sample_id,
        gold_answer=gold_answer,
        benchmark=benchmark,
        source_record_sha256=source_record_sha256,
        source_prompt_sha256=_sha256(source_prompt),
        arms=arms,
    )

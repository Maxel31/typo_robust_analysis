"""Pure planning helpers for the Table 2 complete-text intervention."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

PRE_ANSWER_BOUNDARY_METHOD = "first-[Tt]he-answer-is-or-entire-continuation/v1"
_ANSWER_TRIGGER = re.compile(r"[Tt]he answer is")
_RESIDUAL_ANSWER_FRAGMENTS = (
    re.compile(r"[Aa]nswer\s*[:=]"),
    re.compile(r"[Ff]inal [Aa]nswer"),
)


@dataclass(frozen=True, slots=True)
class CompletePreAnswer:
    """Legacy-backed complete clean pre-answer boundary for one pair."""

    text: str
    source_char_count: int
    trigger_found: bool
    trigger_count: int
    trigger_char_start: int | None
    residual_fragment: bool
    method: str = PRE_ANSWER_BOUNDARY_METHOD

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("complete pre-answer text must be a string")
        if (
            not isinstance(self.source_char_count, int)
            or isinstance(self.source_char_count, bool)
            or self.source_char_count < len(self.text)
        ):
            raise ValueError("source_char_count must cover the complete pre-answer text")
        if not isinstance(self.trigger_found, bool):
            raise TypeError("trigger_found must be Boolean")
        if (
            not isinstance(self.trigger_count, int)
            or isinstance(self.trigger_count, bool)
            or self.trigger_count < 0
        ):
            raise ValueError("trigger_count must be a non-negative integer")
        if self.trigger_found is not (self.trigger_count > 0):
            raise ValueError("trigger_found and trigger_count disagree")
        if self.trigger_found:
            if (
                not isinstance(self.trigger_char_start, int)
                or isinstance(self.trigger_char_start, bool)
                or self.trigger_char_start != len(self.text)
            ):
                raise ValueError("trigger_char_start must equal the truncation boundary")
        elif self.trigger_char_start is not None or len(self.text) != self.source_char_count:
            raise ValueError("a no-trigger plan must retain the entire continuation")
        if not isinstance(self.residual_fragment, bool):
            raise TypeError("residual_fragment must be Boolean")
        if self.method != PRE_ANSWER_BOUNDARY_METHOD:
            raise ValueError("complete pre-answer boundary method is not supported")

    @property
    def sha256(self) -> str:
        """Return the full content fingerprint of the supplied text."""

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the stable public boundary record."""

        return {
            "method": self.method,
            "text": self.text,
            "text_sha256": self.sha256,
            "source_char_count": self.source_char_count,
            "supplied_char_count": len(self.text),
            "trigger_found": self.trigger_found,
            "trigger_count": self.trigger_count,
            "trigger_char_start": self.trigger_char_start,
            "residual_fragment": self.residual_fragment,
        }


def locate_complete_pre_answer(continuation: str) -> CompletePreAnswer:
    """Cut a clean continuation before its first historical answer trigger.

    The final PDF does not specify a character locator. This function preserves
    the submitted implementation's literal ``[Tt]he answer is`` rule and keeps
    the entire continuation when no trigger is present. Diagnostics never
    exclude a pair from the shared Table 2 denominator.
    """

    if not isinstance(continuation, str):
        raise TypeError("clean continuation must be a string")
    matches = tuple(_ANSWER_TRIGGER.finditer(continuation))
    if matches:
        boundary = matches[0].start()
        text = continuation[:boundary]
        trigger_start: int | None = boundary
    else:
        text = continuation
        trigger_start = None
    residual = any(pattern.search(text) is not None for pattern in _RESIDUAL_ANSWER_FRAGMENTS)
    return CompletePreAnswer(
        text=text,
        source_char_count=len(continuation),
        trigger_found=bool(matches),
        trigger_count=len(matches),
        trigger_char_start=trigger_start,
        residual_fragment=residual,
    )

"""Deterministic on-the-fly pairs and exact unchanged-token alignment."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.records import (
    CleanRecord,
    TypoEdit,
    infer_single_word_typo_edit,
)
from typo_robust_training.data.splits import normalized_content_sha256


_SOURCE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_CLEAN_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "text",
    "task",
    "answer",
    "content_sha256",
    "normalized_content_sha256",
    "metadata",
    "token_count",
}
_NATURAL_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "clean_text",
    "typo_text",
    "task",
    "answer",
    "operation",
    "training_eligible",
    "repository",
    "repository_license",
    "clean_sha256",
    "typo_sha256",
    "metadata",
    "token_count",
}


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


@dataclass(frozen=True, slots=True)
class TrainingSource:
    """One hash-checked row from ``training_sources.jsonl``."""

    kind: str
    record_id: str
    source: str
    source_revision: str
    source_split: str
    source_id: str
    group_id: str
    clean_text: str
    typo_text: str | None
    task: str | None
    answer: str | None
    operation: str | None
    metadata: Mapping[str, object]
    token_count: int

    @classmethod
    def from_dict(cls, value: object) -> TrainingSource:
        if not isinstance(value, Mapping):
            raise ValueError("training source must be an object")
        kind = value.get("kind")
        expected = _CLEAN_FIELDS if kind == "clean" else _NATURAL_FIELDS
        schema = "robustness-clean-record/v1" if kind == "clean" else "robustness-natural-pair/v1"
        if kind not in {"clean", "natural"} or set(value) != expected:
            raise ValueError("training source fields differ")
        if value.get("schema_version") != schema or value.get("split") != "train":
            raise ValueError("training source schema or split differs")
        record_id = _text(value.get("record_id"), field="record_id")
        revision = _text(value.get("source_revision"), field="source_revision")
        if _SHA64.fullmatch(record_id) is None or _SOURCE_REVISION.fullmatch(revision) is None:
            raise ValueError("training source identity hashes differ")
        token_count = value.get("token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("training source token_count must be positive")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("training source metadata must be an object")
        try:
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("training source metadata must be canonical JSON") from exc
        if kind == "clean":
            clean_text = _text(value.get("text"), field="text")
            if value.get("content_sha256") != hashlib.sha256(clean_text.encode()).hexdigest():
                raise ValueError("clean training source content hash differs")
            if value.get("normalized_content_sha256") != normalized_content_sha256(clean_text):
                raise ValueError("clean training source normalized hash differs")
            typo_text = None
            operation = None
        else:
            clean_text = _text(value.get("clean_text"), field="clean_text")
            typo_text = _text(value.get("typo_text"), field="typo_text")
            if clean_text == typo_text:
                raise ValueError("natural training source must contain a typo")
            if value.get("clean_sha256") != hashlib.sha256(clean_text.encode()).hexdigest():
                raise ValueError("natural clean source hash differs")
            if value.get("typo_sha256") != hashlib.sha256(typo_text.encode()).hexdigest():
                raise ValueError("natural typo source hash differs")
            if value.get("training_eligible") is not True:
                raise ValueError("natural training source is not training eligible")
            operation = _text(value.get("operation"), field="operation")
            _text(value.get("repository"), field="repository")
            _text(value.get("repository_license"), field="repository_license")
        task = _optional_text(value.get("task"), field="task")
        answer = _optional_text(value.get("answer"), field="answer")
        if (task is None) != (answer is None):
            raise ValueError("training source task and answer must be jointly present")
        return cls(
            kind=str(kind),
            record_id=record_id,
            source=_text(value.get("source"), field="source"),
            source_revision=revision,
            source_split=_text(value.get("source_split"), field="source_split"),
            source_id=_text(value.get("source_id"), field="source_id"),
            group_id=_text(value.get("group_id"), field="group_id"),
            clean_text=clean_text,
            typo_text=typo_text,
            task=task,
            answer=answer,
            operation=operation,
            metadata=MappingProxyType(dict(metadata)),
            token_count=token_count,
        )

    def clean_record(self) -> CleanRecord:
        return CleanRecord(
            source=self.source,
            source_revision=self.source_revision,
            source_split=self.source_split,
            source_id=self.source_id,
            group_id=self.group_id,
            text=self.clean_text,
            task=self.task,
            answer=self.answer,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class TrainingPair:
    """One clean/typo example after deterministic epoch materialization."""

    record_id: str
    clean_text: str
    typo_text: str
    task: str | None
    answer: str | None
    metadata: Mapping[str, object]
    edits: tuple[TypoEdit, ...]
    is_noop: bool
    epoch: int


def stable_epoch_sources(
    sources: Sequence[TrainingSource], *, seed: int, epoch: int
) -> tuple[TrainingSource, ...]:
    """Return an order independent of input iteration and safe to resume by index."""

    rows = tuple(sources)
    if len({source.record_id for source in rows}) != len(rows):
        raise ValueError("training source IDs are duplicated")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (seed, epoch)
    ):
        raise ValueError("training seed and epoch must be non-negative integers")

    def key(source: TrainingSource) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"training-epoch-order/v1\0{seed}\0{epoch}\0{source.record_id}".encode()
        ).hexdigest()
        return digest, source.record_id

    return tuple(sorted(rows, key=key))


def _natural_edit(source: TrainingSource) -> TypoEdit:
    clean, typo = source.clean_text, source.typo_text
    if typo is None or source.operation is None:
        raise ValueError("natural training source is incomplete")
    return infer_single_word_typo_edit(
        clean,
        typo,
        operation=source.operation,
    )


def materialize_training_pair(
    source: TrainingSource,
    *,
    generator: TypoGenerator,
    epoch: int,
    force_noop: bool | None = None,
) -> TrainingPair:
    """Use fixed natural text or a record-local synthetic typo for this epoch."""

    if not isinstance(source, TrainingSource) or not isinstance(generator, TypoGenerator):
        raise TypeError("materialization requires a TrainingSource and TypoGenerator")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("training pair epoch must be non-negative")
    if force_noop is not None and type(force_noop) is not bool:
        raise TypeError("force_noop must be boolean or None")
    if force_noop is True:
        typo_text, edits, is_noop = source.clean_text, (), True
    elif source.kind == "natural":
        edit = _natural_edit(source)
        typo_text = source.typo_text
        if typo_text is None:
            raise RuntimeError("validated natural source lost typo text")
        edits, is_noop = (edit,), False
    else:
        pair = generator.generate(
            source.clean_record(),
            epoch=epoch,
            force_noop=force_noop,
        )
        typo_text, edits, is_noop = pair.typo_text, pair.edits, pair.is_noop
    return TrainingPair(
        record_id=source.record_id,
        clean_text=source.clean_text,
        typo_text=typo_text,
        task=source.task,
        answer=source.answer,
        metadata=source.metadata,
        edits=edits,
        is_noop=is_noop,
        epoch=epoch,
    )


def _offsets(
    values: Sequence[Sequence[int]], *, text: str, field: str
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"{field}[{index}] is invalid")
        start, stop = value
        if not 0 <= start <= stop <= len(text):
            raise ValueError(f"{field}[{index}] is outside its text")
        while start < stop and text[start].isspace():
            start += 1
        while start < stop and text[stop - 1].isspace():
            stop -= 1
        normalized.append((start, stop))
    return tuple(normalized)


def _spans(
    values: Sequence[tuple[int, int]], *, text: str, field: str
) -> tuple[tuple[int, int], ...]:
    spans = tuple(values)
    previous = 0
    for index, span in enumerate(spans):
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in span)
            or not previous <= span[0] < span[1] <= len(text)
        ):
            raise ValueError(f"{field}[{index}] is invalid or overlapping")
        previous = span[1]
    return spans


def _unchanged_segments(
    clean_text: str,
    typo_text: str,
    clean_spans: tuple[tuple[int, int], ...],
    typo_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    if len(clean_spans) != len(typo_spans):
        raise ValueError("clean and typo edit counts differ")
    if not clean_spans:
        if clean_text != typo_text:
            raise ValueError("different pair text requires edit spans")
        return ((0, len(clean_text), 0, len(typo_text)),)
    result: list[tuple[int, int, int, int]] = []
    clean_cursor = typo_cursor = 0
    for clean_span, typo_span in zip(clean_spans, typo_spans, strict=True):
        if clean_text[clean_cursor : clean_span[0]] != typo_text[typo_cursor : typo_span[0]]:
            raise ValueError("pair differs outside declared edited words")
        result.append((clean_cursor, clean_span[0], typo_cursor, typo_span[0]))
        clean_cursor, typo_cursor = clean_span[1], typo_span[1]
    if clean_text[clean_cursor:] != typo_text[typo_cursor:]:
        raise ValueError("pair suffix differs outside declared edited words")
    result.append((clean_cursor, len(clean_text), typo_cursor, len(typo_text)))
    return tuple(result)


def align_unchanged_token_positions(
    *,
    clean_text: str,
    typo_text: str,
    clean_edit_spans: Sequence[tuple[int, int]],
    typo_edit_spans: Sequence[tuple[int, int]],
    clean_offsets: Sequence[Sequence[int]],
    typo_offsets: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Align exact token spans wholly inside unchanged character segments."""

    clean_spans = _spans(clean_edit_spans, text=clean_text, field="clean_edit_spans")
    typo_spans = _spans(typo_edit_spans, text=typo_text, field="typo_edit_spans")
    segments = _unchanged_segments(clean_text, typo_text, clean_spans, typo_spans)
    clean_tokens = _offsets(clean_offsets, text=clean_text, field="clean_offsets")
    typo_tokens = _offsets(typo_offsets, text=typo_text, field="typo_offsets")
    typo_by_span: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, span in enumerate(typo_tokens):
        if span[0] == span[1]:
            continue
        typo_by_span[span].append(index)
    aligned: list[tuple[int, int]] = []
    mapped_occurrences: dict[tuple[int, int], int] = defaultdict(int)
    for clean_index, (token_start, token_stop) in enumerate(clean_tokens):
        if token_start == token_stop:
            continue
        for clean_start, clean_stop, typo_start, typo_stop in segments:
            if clean_start <= token_start and token_stop <= clean_stop:
                shift = typo_start - clean_start
                mapped = token_start + shift, token_stop + shift
                if not typo_start <= mapped[0] <= mapped[1] <= typo_stop:
                    break
                occurrence = mapped_occurrences[mapped]
                mapped_occurrences[mapped] += 1
                candidates = typo_by_span.get(mapped, [])
                if occurrence < len(candidates):
                    typo_index = candidates[occurrence]
                    if clean_text[token_start:token_stop] != typo_text[slice(*mapped)]:
                        raise ValueError("aligned token text differs")
                    aligned.append((clean_index, typo_index))
                break
    return tuple(aligned)


def edited_word_final_token_positions(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[tuple[int, int]],
    *,
    text: str | None = None,
) -> tuple[int, ...]:
    """Return the last token covering each edited word's final character."""

    text_was_provided = text is not None
    if text is None:
        maximum = max((int(value[1]) for value in offsets), default=0)
        text = "x" * maximum
    tokens = _offsets(offsets, text=text, field="offsets")
    normalized_spans = _spans(spans, text=text, field="spans")
    positions: list[int] = []
    for index, (start, stop) in enumerate(normalized_spans):
        if text_was_provided and (
            (start > 0 and text[start - 1].isalnum()) or (stop < len(text) and text[stop].isalnum())
        ):
            raise ValueError(f"spans[{index}] starts or ends inside an alphanumeric word")
        overlapping = [
            token_index
            for token_index, (token_start, token_stop) in enumerate(tokens)
            if token_stop > token_start and token_start < stop and start < token_stop
        ]
        cursor = start
        for token_index in overlapping:
            token_start, token_stop = tokens[token_index]
            covered_start = max(start, token_start)
            covered_stop = min(stop, token_stop)
            if covered_start > cursor:
                break
            cursor = max(cursor, covered_stop)
        if not overlapping or cursor < stop:
            raise ValueError(f"spans[{index}] is not fully covered by tokenizer offsets")
        positions.append(overlapping[-1])
    if len(set(positions)) != len(positions):
        raise ValueError("edited words resolve to duplicate token positions")
    return tuple(positions)


__all__ = [
    "TrainingPair",
    "TrainingSource",
    "align_unchanged_token_positions",
    "edited_word_final_token_positions",
    "materialize_training_pair",
    "stable_epoch_sources",
]

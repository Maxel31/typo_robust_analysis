"""Hash-bound checkpoints and exact next-sample training cursors."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.training.pairs import TrainingSource, stable_epoch_sources


_BINDINGS_V1 = {
    "config_sha256",
    "training_data_sha256",
    "localization_sha256",
    "seed",
}
_BINDINGS_V2 = _BINDINGS_V1 | {
    "monitor_protocol_sha256",
    "monitor_data_sha256",
}


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    epoch: int
    source_index: int
    micro_steps: int
    optimizer_steps: int
    student_tokens: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.epoch,
                self.source_index,
                self.micro_steps,
                self.optimizer_steps,
                self.student_tokens,
            )
        ):
            raise ValueError("training cursor fields must be non-negative integers")

    def as_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "source_index": self.source_index,
            "micro_steps": self.micro_steps,
            "optimizer_steps": self.optimizer_steps,
            "student_tokens": self.student_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainingCursor:
        fields = {
            "epoch",
            "source_index",
            "micro_steps",
            "optimizer_steps",
            "student_tokens",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("training checkpoint cursor fields differ")
        try:
            return cls(**{field: value[field] for field in fields})  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("training checkpoint cursor is invalid") from exc


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    cursor: TrainingCursor
    state_path: Path
    state_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bindings(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or (
        set(value) != _BINDINGS_V1 and set(value) != _BINDINGS_V2
    ):
        raise ValueError("training checkpoint bindings fields differ")
    result = dict(value)
    for field in ("config_sha256", "training_data_sha256"):
        digest = result[field]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"training checkpoint {field} differs")
    localization = result["localization_sha256"]
    if localization is not None and (not isinstance(localization, str) or len(localization) != 64):
        raise ValueError("training checkpoint localization_sha256 differs")
    seed = result["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training checkpoint seed differs")
    for field in ("monitor_protocol_sha256", "monitor_data_sha256"):
        if field in result:
            digest = result[field]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"training checkpoint {field} differs")
    return result


def next_training_source(
    sources: Sequence[TrainingSource],
    *,
    cursor: TrainingCursor,
    seed: int,
) -> tuple[TrainingSource, int, TrainingCursor]:
    """Consume one source and return a cursor pointing to the exact successor."""

    rows = tuple(sources)
    if not rows or any(not isinstance(source, TrainingSource) for source in rows):
        raise ValueError("training cursor requires non-empty validated sources")
    if cursor.source_index >= len(rows):
        raise ValueError("training cursor source_index is outside the epoch")
    epoch = cursor.epoch
    ordered = stable_epoch_sources(rows, seed=seed, epoch=epoch)
    source = ordered[cursor.source_index]
    next_index = cursor.source_index + 1
    next_epoch = epoch
    if next_index == len(ordered):
        next_epoch += 1
        next_index = 0
    return (
        source,
        epoch,
        TrainingCursor(
            epoch=next_epoch,
            source_index=next_index,
            micro_steps=cursor.micro_steps + 1,
            optimizer_steps=cursor.optimizer_steps,
            student_tokens=cursor.student_tokens,
        ),
    )


def write_training_checkpoint(
    path: Path,
    *,
    cursor: TrainingCursor,
    state_path: Path,
    bindings: Mapping[str, object],
) -> None:
    """Atomically bind an opaque optimizer/adapter/RNG state file to its cursor."""

    if not isinstance(cursor, TrainingCursor):
        raise TypeError("training checkpoint cursor must be TrainingCursor")
    checkpoint = Path(path).resolve()
    state = Path(state_path).resolve()
    if not state.is_file():
        raise ValueError("training checkpoint runtime state is not a file")
    try:
        relative_state = state.relative_to(checkpoint.parent)
    except ValueError as exc:
        raise ValueError("training checkpoint runtime state must be inside its output") from exc
    normalized_bindings = _bindings(bindings)
    payload = {
        "schema_version": (
            "robustness-training-checkpoint/v2"
            if set(normalized_bindings) == _BINDINGS_V2
            else "robustness-training-checkpoint/v1"
        ),
        "bindings": normalized_bindings,
        "cursor": cursor.as_dict(),
        "runtime_state_file": relative_state.as_posix(),
        "runtime_state_sha256": _sha256_file(state),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: Path,
    *,
    expected_bindings: Mapping[str, object],
) -> LoadedTrainingCheckpoint:
    """Load only a checkpoint whose data, config, localization, and state still match."""

    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise ValueError("training checkpoint is not a file")
    payload = strict_loads(checkpoint.read_text(encoding="utf-8"), context=str(checkpoint))
    fields = {
        "schema_version",
        "bindings",
        "cursor",
        "runtime_state_file",
        "runtime_state_sha256",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != fields
        or payload.get("schema_version")
        not in {"robustness-training-checkpoint/v1", "robustness-training-checkpoint/v2"}
    ):
        raise ValueError("training checkpoint fields or schema differ")
    actual_bindings = _bindings(payload["bindings"])  # type: ignore[arg-type]
    if actual_bindings != _bindings(expected_bindings):
        raise ValueError("training checkpoint bindings differ")
    state_file = payload["runtime_state_file"]
    if not isinstance(state_file, str) or not state_file or Path(state_file).is_absolute():
        raise ValueError("training checkpoint runtime state path differs")
    state_path = (checkpoint.parent / state_file).resolve()
    if not state_path.is_relative_to(checkpoint.parent):
        raise ValueError("training checkpoint runtime state path differs")
    expected_sha = payload["runtime_state_sha256"]
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or not state_path.is_file()
        or _sha256_file(state_path) != expected_sha
    ):
        raise ValueError("training checkpoint runtime state hash differs")
    return LoadedTrainingCheckpoint(
        cursor=TrainingCursor.from_dict(payload["cursor"]),
        state_path=state_path,
        state_sha256=expected_sha,
    )


__all__ = [
    "LoadedTrainingCheckpoint",
    "TrainingCursor",
    "load_training_checkpoint",
    "next_training_source",
    "write_training_checkpoint",
]

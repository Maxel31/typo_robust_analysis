"""Checkpointed cursors resume at the exact next epoch-local sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.training.checkpoint import (
    TrainingCursor,
    load_training_checkpoint,
    next_training_source,
    write_training_checkpoint,
)
from typo_robust_training.training.pairs import TrainingSource
from typo_robust_training.data.splits import normalized_content_sha256


def _clean_source(record_id: str) -> TrainingSource:
    text = "Stable educational training text."
    return TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": record_id,
            "source": "fineweb_edu",
            "source_revision": "a" * 40,
            "source_split": "train",
            "source_id": record_id,
            "group_id": record_id,
            "split": "train",
            "text": text,
            "task": None,
            "answer": None,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalized_content_sha256": normalized_content_sha256(text),
            "metadata": {},
            "token_count": 6,
        }
    )


def _consume(
    sources: tuple[TrainingSource, ...],
    cursor: TrainingCursor,
    *,
    count: int,
) -> tuple[tuple[tuple[int, str], ...], TrainingCursor]:
    consumed: list[tuple[int, str]] = []
    current = cursor
    for _ in range(count):
        source, epoch, current = next_training_source(sources, cursor=current, seed=42)
        consumed.append((epoch, source.record_id))
    return tuple(consumed), current


def test_checkpoint_resume_reproduces_the_exact_next_sample_sequence(tmp_path: Path) -> None:
    sources = tuple(_clean_source(f"{index:064x}") for index in range(7))
    start = TrainingCursor(
        epoch=0, source_index=0, micro_steps=0, optimizer_steps=0, student_tokens=0
    )
    uninterrupted, _ = _consume(sources, start, count=31)
    prefix, cursor = _consume(sources, start, count=13)
    assert prefix == uninterrupted[:13]

    checkpoint = tmp_path / "checkpoint.json"
    state = tmp_path / "training-state.pt"
    state.write_bytes(b"opaque-runtime-state")
    write_training_checkpoint(
        checkpoint,
        cursor=cursor,
        state_path=state,
        bindings={
            "config_sha256": "a" * 64,
            "training_data_sha256": "b" * 64,
            "localization_sha256": "c" * 64,
            "seed": 42,
        },
    )
    resumed = load_training_checkpoint(
        checkpoint,
        expected_bindings={
            "config_sha256": "a" * 64,
            "training_data_sha256": "b" * 64,
            "localization_sha256": "c" * 64,
            "seed": 42,
        },
    )
    suffix, _ = _consume(sources, resumed.cursor, count=18)
    assert prefix + suffix == uninterrupted
    assert resumed.state_path == state.resolve()


def test_checkpoint_rejects_binding_or_runtime_state_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    state = tmp_path / "state.pt"
    state.write_bytes(b"state")
    bindings = {
        "config_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "localization_sha256": None,
        "seed": 43,
    }
    write_training_checkpoint(
        checkpoint,
        cursor=TrainingCursor(0, 0, 0, 0, 0),
        state_path=state,
        bindings=bindings,
    )
    changed = dict(bindings)
    changed["seed"] = 44
    with pytest.raises(ValueError, match="bindings differ"):
        load_training_checkpoint(checkpoint, expected_bindings=changed)

    state.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="runtime state hash differs"):
        load_training_checkpoint(checkpoint, expected_bindings=bindings)

    state.write_bytes(b"state")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["cursor"]["source_index"] = -1
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cursor"):
        load_training_checkpoint(checkpoint, expected_bindings=bindings)


def test_cycle2_checkpoint_binds_frozen_monitor_protocol_and_data(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    state = tmp_path / "state.pt"
    state.write_bytes(b"cycle-2-state")
    bindings = {
        "config_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "localization_sha256": "c" * 64,
        "monitor_protocol_sha256": "d" * 64,
        "monitor_data_sha256": "e" * 64,
        "seed": 42,
    }
    write_training_checkpoint(
        checkpoint,
        cursor=TrainingCursor(0, 0, 0, 0, 0),
        state_path=state,
        bindings=bindings,
    )
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["schema_version"].endswith(
        "/v2"
    )
    loaded = load_training_checkpoint(checkpoint, expected_bindings=bindings)
    assert loaded.state_path == state.resolve()
    changed = {**bindings, "monitor_data_sha256": "f" * 64}
    with pytest.raises(ValueError, match="bindings differ"):
        load_training_checkpoint(checkpoint, expected_bindings=changed)

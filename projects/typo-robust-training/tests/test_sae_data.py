"""Clean-only, leak-resistant input contract for SAE training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.sae.data import (
    load_clean_fineweb_sources,
    record_id_sha256,
    reserve_confirmatory_training_prefix,
)


def _row(index: int, *, source: str = "fineweb_edu", kind: str = "clean") -> dict[str, object]:
    text = f"clean educational document {index} with enough content"
    record_id = hashlib.sha256(f"record-{index}".encode()).hexdigest()
    return {
        "schema_version": "robustness-clean-record/v1",
        "kind": kind,
        "record_id": record_id,
        "source": source,
        "source_revision": "f" * 40,
        "source_split": "train",
        "source_id": f"fineweb_edu:{index}",
        "group_id": f"fineweb_edu:{index}",
        "split": "train",
        "text": text,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_content_sha256": hashlib.sha256(
            " ".join(text.lower().split()).encode()
        ).hexdigest(),
        "metadata": {},
        "token_count": 8,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sae_input_accepts_only_clean_fineweb_train_records(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    _write(path, [_row(index) for index in range(5)])

    rows = load_clean_fineweb_sources(path)
    reserved, eligible = reserve_confirmatory_training_prefix(
        rows,
        seed=42,
        epoch=0,
        reserved_records=2,
    )
    assert len(reserved) == 2
    assert len(eligible) == 3
    assert not ({row.record_id for row in reserved} & {row.record_id for row in eligible})
    assert len(record_id_sha256(eligible)) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("source", "gsm8k", "FineWeb-Edu"), ("kind", "natural", "clean")),
)
def test_sae_input_rejects_reasoning_or_typo_rows(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = tmp_path / "bad.jsonl"
    row = _row(0)
    row[field] = value
    _write(path, [row])
    with pytest.raises(ValueError, match=message):
        load_clean_fineweb_sources(path)

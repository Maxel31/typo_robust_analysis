"""Clean-only, leak-resistant input contract for SAE training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.sae.data import (
    load_clean_fineweb_sources,
    prepare_sae_sources,
    record_id_sha256,
    reserve_confirmatory_training_prefix,
    sha256_file,
)
from typo_robust_training.sae.registry import validate_sae_prepared_sources


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
            " ".join(text.casefold().split()).encode()
        ).hexdigest(),
        "metadata": {},
        "token_count": 8,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _replace_text(row: dict[str, object], text: str) -> None:
    row["text"] = text
    row["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    row["normalized_content_sha256"] = hashlib.sha256(
        " ".join(text.casefold().split()).encode()
    ).hexdigest()


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


def test_protected_manifest_deduplicates_only_eligible_content(tmp_path: Path) -> None:
    path = tmp_path / "protected.jsonl"
    rows = [_row(index) for index in range(8)]
    _write(path, rows)
    loaded = load_clean_fineweb_sources(path)
    expected_reserved, expected_eligible = reserve_confirmatory_training_prefix(
        loaded,
        seed=42,
        epoch=0,
        reserved_records=2,
    )
    by_id = {str(row["record_id"]): row for row in rows}
    reserved_text = expected_reserved[0].clean_text
    _replace_text(by_id[expected_eligible[0].record_id], reserved_text)
    duplicate_text = "eligible duplicate content retained by deterministic order"
    duplicate_ids = (expected_eligible[1].record_id, expected_eligible[2].record_id)
    for record_id in duplicate_ids:
        _replace_text(by_id[record_id], duplicate_text)
    _write(path, rows)

    prepared = prepare_sae_sources(
        [path],
        protected_manifest_sha256=sha256_file(path),
        reserved_seed=42,
        reserved_epoch=0,
        reserved_records=2,
    )

    assert [row.record_id for row in prepared.reserved] == [
        row.record_id for row in expected_reserved
    ]
    eligible_ids = {row.record_id for row in prepared.sources}
    assert expected_eligible[0].record_id not in eligible_ids
    assert len(eligible_ids & set(duplicate_ids)) == 1
    assert len(prepared.sources) == len(rows) - len(expected_reserved) - 2
    assert prepared.protected_normalized_duplicates_removed == 2
    assert expected_eligible[0].record_id in prepared.input_record_ids
    assert len(prepared.input_record_ids) == len(rows)
    assert len({row.clean_text.casefold() for row in prepared.sources}) == len(prepared.sources)

    class Preregistration:
        initial_eligible_records = prepared.protected_eligible_records
        initial_eligible_source_tokens = prepared.protected_eligible_source_tokens
        initial_eligible_record_ids_sha256 = prepared.protected_eligible_record_ids_sha256
        eligible_records_removed = prepared.protected_normalized_duplicates_removed

    validate_sae_prepared_sources(prepared, preregistration=Preregistration())
    Preregistration.initial_eligible_records += 1
    with pytest.raises(ValueError, match="differs from preregistration"):
        validate_sae_prepared_sources(prepared, preregistration=Preregistration())


def test_protected_manifest_still_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "protected.jsonl"
    rows = [_row(index) for index in range(4)]
    rows[3]["source_id"] = rows[2]["source_id"]
    _write(path, rows)

    with pytest.raises(ValueError, match="duplicates source ID"):
        prepare_sae_sources(
            [path],
            protected_manifest_sha256=sha256_file(path),
            reserved_seed=42,
            reserved_epoch=0,
            reserved_records=1,
        )


def test_supplement_cannot_overlap_protected_normalized_content(tmp_path: Path) -> None:
    protected = tmp_path / "protected.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    protected_rows = [_row(index) for index in range(4)]
    supplement_row = _row(10)
    _replace_text(supplement_row, str(protected_rows[2]["text"]))
    _write(protected, protected_rows)
    _write(supplement, [supplement_row])

    with pytest.raises(ValueError, match="overlap by normalized content"):
        prepare_sae_sources(
            [protected, supplement],
            protected_manifest_sha256=sha256_file(protected),
            reserved_seed=42,
            reserved_epoch=0,
            reserved_records=1,
        )

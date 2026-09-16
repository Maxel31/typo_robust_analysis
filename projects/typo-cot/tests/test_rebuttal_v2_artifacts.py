"""Independent tamper, source-binding, and exact-byte tests for manifest consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2 import artifacts, source
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError, canonical_json


PROTOCOL = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    empty: bool = False,
    raw_text: str | None = "Answer: 1",
    pair_changes: dict | None = None,
    generation_changes: dict | None = None,
    unknown_source: bool = False,
) -> Path:
    pair = {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": "p",
        "source_pair_key": "p",
        "task": "gsm8k",
        "model_id": "fixture-model",
        "clean_text": "one",
        "typo_text": "oen",
        "canonical_gold": {"numerator": "1", "denominator": "1"},
        "gold_canonicalization_version": "fixture/v1",
        "expected_generations": [
            {"arm": "clean", "window": None, "source_generation_key": "g"},
            {"arm": "typo", "window": None},
        ],
        **(pair_changes or {}),
    }
    generation = {
        "schema_version": "rebuttal-source-generation/v2",
        "source_record_id": "g",
        "source_generation_key": "g",
        "source_pair_key": "p",
        "arm": "clean",
        "window": None,
        "raw_text": raw_text,
        **(generation_changes or {}),
    }
    sources = []
    for role, schema, rows in (
        ("pairs", "rebuttal-source-pair/v2", [] if empty else [pair]),
        ("generations", "rebuttal-source-generation/v2", [] if empty else [generation]),
    ):
        path = tmp_path / f"{role}.jsonl"
        _write_rows(path, rows)
        sources.append(
            {
                "source_id": role,
                "source_kind": "submitted-archive",
                "role": role,
                "path": path.name,
                "expected_sha256": _hash(path),
                "format": "jsonl",
                "adapter_version": schema,
                "source_commit": None,
                "model_revision": None,
                "tokenizer_revision": None,
                "prompt_template_revision": None,
                "generation_config": None,
                "availability": "available",
                "reason_codes": [],
            }
        )
    if unknown_source:
        sources.append(
            {
                **sources[0],
                "source_id": "unlocated",
                "role": "prompts",
                "path": None,
                "expected_sha256": None,
                "format": None,
                "adapter_version": None,
                "availability": "unknown",
                "reason_codes": ["owner_location_unknown"],
            }
        )
    index = tmp_path / "index.json"
    _write(
        index,
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": "fixture-archive",
            "sources": sources,
            "cohorts": [],
            "identity_aliases_ref": None,
            "notes": "consumer fixture",
        },
    )
    output = tmp_path / "intake"
    source.run_intake(index, PROTOCOL, output)
    return output / "pair_manifest.jsonl"


def _rehash_meta(manifest: Path, key: str = "manifest_ref") -> None:
    path = manifest.with_name("pair_manifest.meta.json")
    meta = json.loads(path.read_text())
    target = path.parent / meta[key]["path"]
    meta[key]["sha256"] = _hash(target)
    _write(path, meta)


def _replace_pair(manifest: Path, update) -> None:
    rows = _rows(manifest)
    update(rows[0])
    _write_rows(manifest, rows)
    _rehash_meta(manifest)


def _replace_generation(manifest: Path, update) -> None:
    path = manifest.with_name("archive_generation_records.jsonl")
    rows = _rows(path)
    update(rows[0])
    _write_rows(path, rows)

    def rehash(pair):
        for link in pair["archived_generations"]:
            if link["record_ref"] is not None:
                link["record_ref"]["artifact"]["sha256"] = _hash(path)

    _replace_pair(manifest, rehash)


def test_loads_verified_slot_and_expected_missing_slot(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    bundle = artifacts.load_manifest(manifest)
    assert bundle.manifest_ref == {"path": str(manifest), "sha256": _hash(manifest)}
    assert bundle.metadata_ref["sha256"] == _hash(manifest.with_name("pair_manifest.meta.json"))
    assert bundle.protocol_bytes == PROTOCOL.read_bytes()
    assert len(bundle.pairs) == 1
    assert len(bundle.slots) == 2
    clean, typo = bundle.slots
    assert clean.link["arm"] == "clean" and clean.raw_text == "Answer: 1"
    assert clean.generation is not None
    assert typo.generation is None and typo.raw_text is None
    artifacts.revalidate_inputs(bundle)


def test_empty_manifest_still_validates_companion_and_source_coverage(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, empty=True)
    bundle = artifacts.load_manifest(manifest)
    assert bundle.pairs == bundle.slots == []
    assert bundle.source_audit["recovered_pair_count"] == 0
    manifest.with_name("pair_manifest.meta.json").unlink()
    with pytest.raises(IntakeError, match="cannot read"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize(
    "filename",
    ["pair_manifest.jsonl", "source_audit.json", "archive_inventory.jsonl", "protocol.json"],
)
def test_rejects_tampered_immediate_artifact(tmp_path: Path, filename: str) -> None:
    manifest = _fixture(tmp_path)
    target = manifest.with_name(filename)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize("filename", ["index.json", "pairs.jsonl", "generations.jsonl"])
def test_rejects_tampered_transitive_input(tmp_path: Path, filename: str) -> None:
    manifest = _fixture(tmp_path)
    target = tmp_path / filename
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        artifacts.load_manifest(manifest)


def test_metadata_cannot_bind_another_manifest_even_with_identical_bytes(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    other = manifest.with_name("other.jsonl")
    other.write_bytes(manifest.read_bytes())
    with pytest.raises(IntakeError, match="different manifest path"):
        artifacts.load_manifest(other)


def test_changed_protocol_is_not_accepted_after_rehashing(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, empty=True)
    protocol = manifest.with_name("protocol.json")
    payload = json.loads(protocol.read_text())
    payload["protocol_revision"] = "2.0.1"
    _write(protocol, payload)
    _rehash_meta(manifest, "protocol_ref")
    with pytest.raises(IntakeError, match="frozen protocol"):
        artifacts.load_manifest(manifest)


def test_original_generation_without_raw_text_stays_missing_not_fake_answer(tmp_path: Path) -> None:
    bundle = artifacts.load_manifest(_fixture(tmp_path, raw_text=None))
    clean = bundle.slots[0]
    assert clean.generation is not None
    assert clean.raw_text is None and clean.link["availability"] == "missing"


def test_unknown_source_does_not_require_fabricated_hash_or_path(tmp_path: Path) -> None:
    bundle = artifacts.load_manifest(_fixture(tmp_path, unknown_source=True))
    unknown = next(row for row in bundle.inventory if row["source_id"] == "unlocated")
    assert unknown["availability"] == "unknown" and unknown["observed_sha256"] is None


def test_exact_unicode_and_newlines_survive_inline_and_referenced_generation(
    tmp_path: Path,
) -> None:
    raw = "r\u2028s\u2029t\u0085u e\u0301\r\nAnswer: 1\r\n"
    text_path = tmp_path / "answer.txt"
    text_path.write_bytes(raw.encode("utf-8"))
    manifest = _fixture(
        tmp_path,
        raw_text=None,
        generation_changes={
            "raw_text_ref": {"path": text_path.name, "sha256": _hash(text_path)},
        },
    )
    bundle = artifacts.load_manifest(manifest)
    assert bundle.slots[0].raw_text == raw
    assert text_path in bundle.snapshots
    text_path.write_bytes(b"Answer: 2")
    with pytest.raises(IntakeError, match="input changed"):
        artifacts.revalidate_inputs(bundle)


def test_missing_referenced_text_is_error_not_output_missing(tmp_path: Path) -> None:
    text_path = tmp_path / "answer.txt"
    text_path.write_bytes(b"Answer: 1")
    manifest = _fixture(
        tmp_path,
        raw_text=None,
        generation_changes={
            "raw_text_ref": {"path": text_path.name, "sha256": _hash(text_path)},
        },
    )
    text_path.unlink()
    with pytest.raises(IntakeError, match="cannot read referenced"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "source_identity_provenance",
        "additional_source_refs",
        "cohort_membership",
        "clean_prompt_sha256",
    ],
)
def test_all_normalized_pair_fields_are_required(tmp_path: Path, field: str) -> None:
    manifest = _fixture(tmp_path)
    _replace_pair(manifest, lambda row: row.pop(field))
    with pytest.raises(IntakeError, match="missing required fields"):
        artifacts.load_manifest(manifest)


def test_undeclared_pair_extension_is_not_silently_ignored(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    _replace_pair(manifest, lambda row: row.update(unversioned_answer="1"))
    with pytest.raises(IntakeError, match="unknown fields"):
        artifacts.load_manifest(manifest)


def test_duplicate_pair_ids_are_not_deduplicated_by_consumer(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    rows = _rows(manifest)
    _write_rows(manifest, rows + rows)
    _rehash_meta(manifest)
    with pytest.raises(IntakeError, match="duplicate pair_id"):
        artifacts.load_manifest(manifest)


def test_generation_recordref_wrong_pair_rejected_after_hash_update(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    _replace_generation(manifest, lambda row: row.update(pair_id="0" * 64))
    with pytest.raises(IntakeError, match="generation source binding"):
        artifacts.load_manifest(manifest)


def test_generation_recordref_unknown_id_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def update(pair):
        pair["archived_generations"][0]["record_ref"]["record_id"] = "nonexistent"

    _replace_pair(manifest, update)
    with pytest.raises(IntakeError, match="record_id.*not found"):
        artifacts.load_manifest(manifest)


def test_duplicate_generation_ids_in_referenced_file_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    path = manifest.with_name("archive_generation_records.jsonl")
    rows = _rows(path)
    _write_rows(path, rows + rows)

    def update(pair):
        pair["archived_generations"][0]["record_ref"]["artifact"]["sha256"] = _hash(path)

    _replace_pair(manifest, update)
    with pytest.raises(IntakeError, match="duplicate generation_id"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize(
    "change",
    [
        {"raw_text_sha256": "0" * 64},
        {"raw_text": "Answer: 2"},
        {"termination": "length-cap"},
        {"generated_token_ids": [True]},
    ],
)
def test_generation_content_cannot_change_under_valid_outer_hash(
    tmp_path: Path, change: dict
) -> None:
    manifest = _fixture(tmp_path)
    _replace_generation(manifest, lambda row: row.update(change))
    with pytest.raises(IntakeError):
        artifacts.load_manifest(manifest)


def test_manifest_gold_cannot_drift_from_verified_acquisition(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    _replace_pair(
        manifest, lambda row: row.update(canonical_gold={"numerator": "2", "denominator": "1"})
    )
    with pytest.raises(IntakeError, match="differs from acquisition source: canonical_gold"):
        artifacts.load_manifest(manifest)


def test_cohort_membership_reference_is_artifact_not_recordref(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def update(pair):
        pair["cohort_membership"] = [
            {"cohort_id": "invalid", "membership": "unknown", "source_ref": pair["source_ref"]}
        ]

    _replace_pair(manifest, update)
    with pytest.raises(IntakeError, match="missing required fields"):
        artifacts.load_manifest(manifest)


def test_snapshot_aba_never_parses_unrecorded_reopened_bytes(tmp_path: Path, monkeypatch) -> None:
    manifest = _fixture(tmp_path)
    original_read = Path.read_bytes
    manifest_reads = 0
    original = manifest.read_bytes()
    poison = original.replace(b'"task":"gsm8k"', b'"task":"unrecorded-task"')

    def swapping_read(path):
        nonlocal manifest_reads
        if path == manifest:
            manifest_reads += 1
            # A broken verify-then-reopen reader would parse these B bytes.
            if manifest_reads == 2:
                return poison
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", swapping_read)
    with pytest.raises(IntakeError, match="input changed"):
        artifacts.load_manifest(manifest)
    assert manifest_reads == 2  # capture A; final revalidation sees B, never parses B.


def test_snapshot_object_parsing_uses_captured_bytes_even_after_replacement(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    _write(path, {"value": "A"})
    snapshots = artifacts.ArtifactSnapshots()
    snapshots.capture(path, _hash(path))
    _write(path, {"value": "B"})
    assert snapshots.object(path) == {"value": "A"}


def test_orphan_source_generation_evidence_survives_without_invented_pair(tmp_path: Path) -> None:
    manifest = _fixture(
        tmp_path,
        generation_changes={
            "source_pair_key": "unrecovered",
            "source_generation_key": "orphan-generation",
        },
    )
    bundle = artifacts.load_manifest(manifest)
    assert len(bundle.pairs) == 1
    assert all(slot.generation is None for slot in bundle.slots)
    assert bundle.source_audit["raw_generation_record_count"] == 1
    assert any(
        "generation_pair_missing" in finding["reason_codes"]
        for finding in bundle.source_audit["findings"]
    )


def test_strict_duplicate_json_key_rejected_in_metadata(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    meta = manifest.with_name("pair_manifest.meta.json")
    meta.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(IntakeError, match="duplicate JSON key"):
        artifacts.load_manifest(manifest)


def test_observed_generation_cannot_be_disguised_as_expected_missing(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)

    def update(pair):
        clean = pair["archived_generations"][0]
        clean["record_ref"] = None
        clean["availability"] = "missing"
        clean["reason_codes"] = ["expected_generation_missing"]

    _replace_pair(manifest, update)
    with pytest.raises(IntakeError, match="omits or invents an observed generation"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize(
    "field,value", [("available_raw_generation_count", 0), ("raw_generation_record_count", 0)]
)
def test_source_audit_counts_checked_against_captured_records(
    tmp_path: Path, field: str, value: int
) -> None:
    manifest = _fixture(tmp_path)
    audit_path = manifest.with_name("source_audit.json")
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    _write(audit_path, audit)
    _rehash_meta(manifest, "source_audit_ref")
    with pytest.raises(IntakeError, match="generation.*(count|coverage)"):
        artifacts.load_manifest(manifest)


def test_normalized_record_cannot_bind_an_unsupported_adapter(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    inventory_path = manifest.with_name("archive_inventory.jsonl")
    inventory = _rows(inventory_path)
    inventory[0]["adapter_status"] = "unsupported"
    _write_rows(inventory_path, inventory)
    _rehash_meta(manifest, "inventory_ref")
    with pytest.raises(IntakeError, match="matching available inventory source"):
        artifacts.load_manifest(manifest)


def _with_alias_and_cohort(tmp_path: Path) -> tuple[Path, Path]:
    _fixture(tmp_path)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs = _rows(pairs_path)
    pairs[0]["cohort_ids"] = ["fixture-cohort"]
    _write_rows(pairs_path, pairs)
    evidence = tmp_path / "alias-evidence.txt"
    evidence.write_text(
        "Producer ledger confirms p and canonical-p name one acquisition.", encoding="utf-8"
    )
    alias_path = tmp_path / "aliases.json"
    _write(
        alias_path,
        {
            "schema_version": "rebuttal-identity-aliases/v2",
            "pair_aliases": [
                {
                    "alias": {"source_namespace": "fixture-archive", "source_pair_key": "p"},
                    "canonical": {
                        "source_namespace": "fixture-archive",
                        "source_pair_key": "canonical-p",
                    },
                    "evidence_ref": {"path": evidence.name, "sha256": _hash(evidence)},
                }
            ],
            "group_aliases": [],
        },
    )
    ids_path = tmp_path / "ids.json"
    _write(
        ids_path,
        {
            "schema_version": "rebuttal-cohort-ids/v2",
            "cohort_id": "fixture-cohort",
            "source_namespace": "fixture-archive",
            "source_pair_keys": ["p", "unrecovered"],
        },
    )
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    index["sources"][0]["expected_sha256"] = _hash(pairs_path)
    index["identity_aliases_ref"] = {"path": alias_path.name, "sha256": _hash(alias_path)}
    index["cohorts"] = [
        {
            "cohort_id": "fixture-cohort",
            "setting": "fixture-setting",
            "historical_n_reference": 2,
            "expected_ids_ref": {"path": ids_path.name, "sha256": _hash(ids_path)},
            "id_namespace": "fixture-archive",
            "membership_provenance_ref": None,
        }
    ]
    _write(index_path, index)
    output = tmp_path / "aliased-intake"
    source.run_intake(index_path, PROTOCOL, output)
    return output / "pair_manifest.jsonl", evidence


def test_alias_evidence_and_exact_partial_cohort_are_verified(tmp_path: Path) -> None:
    manifest, evidence = _with_alias_and_cohort(tmp_path)
    bundle = artifacts.load_manifest(manifest)
    assert bundle.pairs[0]["source_pair_key"] == "canonical-p"
    assert bundle.pairs[0]["cohort_membership"][0]["membership"] == "member"
    assert bundle.source_audit["cohorts"][0]["coverage_status"] == "partial"
    assert evidence in bundle.snapshots
    evidence.write_bytes(b"changed evidence")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        artifacts.load_manifest(manifest)


def test_rehashed_cohort_audit_cannot_change_frozen_expected_id_list(tmp_path: Path) -> None:
    manifest, _ = _with_alias_and_cohort(tmp_path)
    audit_path = manifest.with_name("source_audit.json")
    audit = json.loads(audit_path.read_text())
    audit["cohorts"][0]["expected_source_pair_keys"] = ["p"]
    _write(audit_path, audit)
    _rehash_meta(manifest, "source_audit_ref")
    with pytest.raises(IntakeError, match="ID list differs from frozen"):
        artifacts.load_manifest(manifest)


@pytest.mark.parametrize("labels", [["AA"], ["a"], ["Ａ"], ["A", "1"]])
def test_pr01_item_labels_preserved_independently_of_parser_support(
    tmp_path: Path, labels: list[str]
) -> None:
    manifest = _fixture(tmp_path, pair_changes={"valid_labels": labels, "canonical_gold": None})
    bundle = artifacts.load_manifest(manifest)
    assert bundle.pairs[0]["valid_labels"] == labels


def test_pr01_choice_gold_preserved_independently_of_parser_support(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, pair_changes={"task": "mmlu-pro", "canonical_gold": "AA"})
    bundle = artifacts.load_manifest(manifest)
    assert bundle.pairs[0]["canonical_gold"] == "AA"

"""Integration fixtures for strict, independently audited donor-bank inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers

from typo_cot.experiments.rebuttal_v2.donor_artifacts import load_donor_bank
from typo_cot.experiments.rebuttal_v2.donors import assign_donors
from typo_cot.experiments.rebuttal_v2.input_audit import load_input_audit, run_input_audit
from typo_cot.experiments.rebuttal_v2.schemas import (
    IntakeError,
    canonical_json,
    make_ref,
)
from typo_cot.experiments.rebuttal_v2.source import run_intake


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/rebuttal_v2/protocol.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _public_audit_fixture(root: Path, *, count: int = 1, shared_group: bool = False) -> dict:
    clean = "red cats eat 12 fish."
    typo = "red cats eat 13 fish."
    prefix = "Example: red cats eat 12 fish.\nQuestion: "
    source_rows = []
    for index in range(count):
        source_rows.append(
            {
                "schema_version": "rebuttal-source-pair/v2",
                "source_record_id": f"source-{index}",
                "source_pair_key": f"donor-{index}",
                "task": "gsm8k",
                "model_id": "google/gemma-3-4b-it",
                "dataset_id": "gsm8k",
                "split": "test",
                "original_problem_id": "shared" if shared_group else f"question-{index}",
                "target_rule": "rule-a",
                "clean_text": clean,
                "typo_text": typo,
                "clean_prompt": prefix + clean,
                "typo_prompt": prefix + typo,
                "clean_query_span": [len(prefix), len(prefix + clean)],
                "typo_query_span": [len(prefix), len(prefix + typo)],
                "tokenizer_revision": "a" * 40,
                "intended_edits": [],
                "archived_token_positions": [],
                "prompt_regions": {
                    side: [
                        {
                            "kind": "stem",
                            "start": len(prefix),
                            "end": len(prefix + clean),
                        }
                    ]
                    for side in ("clean", "typo")
                },
            }
        )
    source_path = _write_rows(root / "pairs.jsonl", source_rows)
    index_path = _write_json(
        root / "index.json",
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": "donor-fixture",
            "sources": [
                {
                    "source_id": "pairs",
                    "source_kind": "submitted-archive",
                    "role": "pairs",
                    "path": source_path.name,
                    "expected_sha256": _sha(source_path),
                    "format": "jsonl",
                    "adapter_version": "rebuttal-source-pair/v2",
                    "source_commit": None,
                    "model_revision": None,
                    "tokenizer_revision": None,
                    "prompt_template_revision": None,
                    "generation_config": None,
                    "availability": "available",
                    "reason_codes": [],
                }
            ],
            # Donors are separately verified intake pairs, not members of the
            # archived recipient cohorts.
            "cohorts": [],
            "identity_aliases_ref": None,
            "notes": "Public-schema donor reader fixture.",
        },
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest_path = root / "intake/pair_manifest.jsonl"
    run_intake(index_path, PROTOCOL_PATH, manifest_path.parent)

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                word: index
                for index, word in enumerate(
                    ["[UNK]", "Example:", "red", "cats", "eat", "12", "fish.", "Question:", "13"]
                )
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer_path = root / "tokenizer.json"
    tokenizer_path.write_text(tokenizer.to_str(), encoding="utf-8")
    lock_path = _write_json(
        root / "tokenizer_lock.json",
        {
            "schema_version": "rebuttal-tokenizer-lock/v2",
            "models": {
                "google/gemma-3-4b-it": {
                    "tokenizer_id": "google/gemma-3-4b-it",
                    "revision": "a" * 40,
                    "tokenizer_files": {
                        "tokenizer.json": {
                            "path": tokenizer_path.name,
                            "sha256": _sha(tokenizer_path),
                        }
                    },
                    "library": {"name": "tokenizers", "version": tokenizers.__version__},
                    "special_token_ids": {},
                    "chat_template": None,
                    "add_special_tokens": False,
                    "normalization": None,
                    "padding_side": "left",
                    "offset_convention": "unicode-codepoint-half-open",
                }
            },
        },
    )
    audit_path = root / "audit/input_audit_records.jsonl"
    run_input_audit(manifest_path, lock_path, audit_path.parent)
    audit = load_input_audit(audit_path)

    bank_path = root / "donor_bank.jsonl"
    audit_by_pair = {row["pair_id"]: row for row in audit.rows}
    bank_rows = []
    for index, pair in enumerate(audit.pairs):
        pair_id = pair["pair_id"]
        audit_row = audit_by_pair[pair_id]
        prompt_path = root / f"clean_prompt_{index}.txt"
        prompt_path.write_text(pair["clean_prompt"], encoding="utf-8")
        endpoints = [edit["clean_endpoint"] for edit in audit_row["alignment"]["aligned_edits"]]
        bank_rows.append(
            {
                "schema_version": "rebuttal-donor-bank/v2",
                "pair_id": pair_id,
                "original_problem_group_id": pair["original_problem_group_id"],
                "task": pair["task"],
                "model_id": pair["model_id"],
                "target_rule": pair["target_rule"],
                "clean_prompt_ref": make_ref(prompt_path, bank_path),
                "clean_prompt_sha256": _sha(prompt_path),
                "aligned_word_count": len(endpoints),
                "clean_endpoints": endpoints,
                "tokenizer_lock_ref": make_ref(lock_path, bank_path),
                "input_audit_ref": {
                    "artifact": make_ref(audit_path, bank_path),
                    "record_id": audit_row["input_audit_id"],
                },
                "provenance": {
                    "manifest_ref": {
                        "artifact": make_ref(manifest_path, bank_path),
                        "record_id": pair_id,
                    },
                    "source_ref": {
                        "artifact": make_ref(source_path, bank_path),
                        "record_id": pair["source_record_id"],
                    },
                },
            }
        )
    _write_rows(bank_path, bank_rows)
    return {
        "bank": bank_path,
        "protocol": protocol,
        "audit": audit_path,
        "manifest": manifest_path,
        "source": source_path,
        "lock": lock_path,
    }


def test_loads_full_verified_chain_and_exposes_assignment_rows(tmp_path: Path) -> None:
    fixture = _public_audit_fixture(tmp_path, count=2)
    bundle = load_donor_bank(fixture["bank"], fixture["protocol"])

    assert bundle.path == fixture["bank"].resolve()
    assert bundle.reference == {"path": str(bundle.path), "sha256": _sha(fixture["bank"])}
    assert len(bundle.rows) == len(bundle.pairs) == 2
    assert set(bundle.audit_rows_by_pair_id) == {row["pair_id"] for row in bundle.rows}
    assert set(bundle.tokenizer_descriptors_by_pair_id) == set(bundle.audit_rows_by_pair_id)
    assert all(pair["clean_prompt"] is not None for pair in bundle.pairs)
    assert fixture["audit"].resolve() in bundle.snapshots
    assert fixture["manifest"].resolve() in bundle.snapshots
    assert fixture["source"].resolve() in bundle.snapshots
    assert fixture["lock"].resolve() in bundle.snapshots

    recipients = [
        {
            "pair_id": "recipient",
            "original_problem_group_id": "recipient-group",
            "model_id": bundle.rows[0]["model_id"],
            "task": bundle.rows[0]["task"],
            "target_rule": bundle.rows[0]["target_rule"],
            "aligned_word_count": bundle.rows[0]["aligned_word_count"],
            "gold": "must-not-be-read",
        }
    ]
    forward = assign_donors(recipients, bundle.rows, fixture["protocol"])
    reverse = assign_donors(recipients, list(reversed(bundle.rows)), fixture["protocol"])
    assert forward == reverse
    assert forward["assignments"][0]["eligibility"] == "valid"


def test_empty_bank_is_valid_and_retains_its_reference(tmp_path: Path) -> None:
    bank = tmp_path / "donor_bank.jsonl"
    bank.write_bytes(b"")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bundle = load_donor_bank(bank, protocol)
    assert bundle.rows == []
    assert bundle.pairs == []
    assert bundle.audit_rows_by_pair_id == {}
    assert bundle.tokenizer_descriptors_by_pair_id == {}
    assert bundle.reference["sha256"] == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("field", ["clean_endpoints", "original_problem_group_id", "target_rule"])
def test_tampered_scientific_fields_are_rejected(tmp_path: Path, field: str) -> None:
    fixture = _public_audit_fixture(tmp_path)
    rows = _read_rows(fixture["bank"])
    if field == "clean_endpoints":
        rows[0][field] = [rows[0][field][0] + 1]
    else:
        rows[0][field] = "tampered"
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="differs"):
        load_donor_bank(fixture["bank"], fixture["protocol"])


def test_hash_rebound_prompt_is_rejected_by_verified_pair_content(tmp_path: Path) -> None:
    fixture = _public_audit_fixture(tmp_path)
    rows = _read_rows(fixture["bank"])
    prompt = tmp_path / "attacker_prompt.txt"
    prompt.write_text("outcome-selected prompt", encoding="utf-8")
    rows[0]["clean_prompt_ref"] = make_ref(prompt, fixture["bank"])
    rows[0]["clean_prompt_sha256"] = _sha(prompt)
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="clean donor prompt differs"):
        load_donor_bank(fixture["bank"], fixture["protocol"])


def test_hash_rebound_audit_and_provenance_are_rejected(tmp_path: Path) -> None:
    audit_fixture = _public_audit_fixture(tmp_path / "audit-case")
    audit_rows = _read_rows(audit_fixture["audit"])
    audit_rows[0]["alignment"]["aligned_edits"][0]["clean_endpoint"] += 1
    _write_rows(audit_fixture["audit"], audit_rows)
    bank_rows = _read_rows(audit_fixture["bank"])
    bank_rows[0]["input_audit_ref"]["artifact"]["sha256"] = _sha(audit_fixture["audit"])
    _write_rows(audit_fixture["bank"], bank_rows)
    with pytest.raises(IntakeError, match="SHA-256 mismatch|differs from the verified"):
        load_donor_bank(audit_fixture["bank"], audit_fixture["protocol"])

    source_fixture = _public_audit_fixture(tmp_path / "source-case")
    attacker = source_fixture["bank"].parent / "attacker_source.jsonl"
    _write_rows(
        attacker,
        [
            {
                "schema_version": "rebuttal-source-pair/v2",
                "source_record_id": "source-0",
                "source_pair_key": "selected-after-outcome",
                "task": "gsm8k",
                "model_id": "google/gemma-3-4b-it",
            }
        ],
    )
    bank_rows = _read_rows(source_fixture["bank"])
    bank_rows[0]["provenance"]["source_ref"]["artifact"] = make_ref(
        attacker, source_fixture["bank"]
    )
    _write_rows(source_fixture["bank"], bank_rows)
    with pytest.raises(IntakeError, match="source_ref differs from verified provenance"):
        load_donor_bank(source_fixture["bank"], source_fixture["protocol"])


def test_provenance_record_ref_allows_only_existing_byte_identical_alias(
    tmp_path: Path,
) -> None:
    fixture = _public_audit_fixture(tmp_path)
    original_source = fixture["source"].resolve()
    alias = tmp_path / "source_copy.jsonl"
    alias.write_bytes(original_source.read_bytes())
    rows = _read_rows(fixture["bank"])
    rows[0]["provenance"]["source_ref"]["artifact"] = make_ref(alias, fixture["bank"])
    _write_rows(fixture["bank"], rows)

    bundle = load_donor_bank(fixture["bank"], fixture["protocol"])
    authoritative_ref = bundle.pairs[0]["source_ref"]["artifact"]
    assert Path(authoritative_ref["path"]).resolve() == original_source
    assert authoritative_ref["sha256"] == _sha(alias) == _sha(original_source)

    rows[0]["provenance"]["source_ref"]["artifact"]["path"] = "missing_copy.jsonl"
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="cannot read referenced artifact"):
        load_donor_bank(fixture["bank"], fixture["protocol"])

    alias.write_bytes(alias.read_bytes() + b"\n")
    rows[0]["provenance"]["source_ref"]["artifact"] = make_ref(alias, fixture["bank"])
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="source_ref differs from verified provenance"):
        load_donor_bank(fixture["bank"], fixture["protocol"])


def test_protocol_and_tokenizer_lock_mismatches_are_rejected(tmp_path: Path) -> None:
    fixture = _public_audit_fixture(tmp_path)
    changed_protocol = copy.deepcopy(fixture["protocol"])
    changed_protocol["seed"] = 43
    with pytest.raises(IntakeError, match="frozen revision"):
        load_donor_bank(fixture["bank"], changed_protocol)

    rows = _read_rows(fixture["bank"])
    attacker_lock = tmp_path / "attacker_lock.json"
    _write_json(attacker_lock, {"not": "the audited lock"})
    rows[0]["tokenizer_lock_ref"] = make_ref(attacker_lock, fixture["bank"])
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="tokenizer lock differs"):
        load_donor_bank(fixture["bank"], fixture["protocol"])


def test_duplicate_pairs_fail_but_shared_original_group_variants_are_allowed(
    tmp_path: Path,
) -> None:
    fixture = _public_audit_fixture(tmp_path, count=2, shared_group=True)
    shared = load_donor_bank(fixture["bank"], fixture["protocol"])
    assert len({row["original_problem_group_id"] for row in shared.rows}) == 1
    assert len({row["pair_id"] for row in shared.rows}) == 2

    rows = _read_rows(fixture["bank"])
    rows.append(copy.deepcopy(rows[0]))
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="duplicate donor pair_id"):
        load_donor_bank(fixture["bank"], fixture["protocol"])


def test_exact_schema_and_strict_integer_endpoints_are_enforced(tmp_path: Path) -> None:
    fixture = _public_audit_fixture(tmp_path)
    rows = _read_rows(fixture["bank"])
    rows[0]["gold"] = "forbidden outcome-bearing extension"
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="unknown fields: gold"):
        load_donor_bank(fixture["bank"], fixture["protocol"])

    fixture = _public_audit_fixture(tmp_path / "boolean")
    rows = _read_rows(fixture["bank"])
    rows[0]["clean_endpoints"] = [True]
    _write_rows(fixture["bank"], rows)
    with pytest.raises(IntakeError, match="not a boolean"):
        load_donor_bank(fixture["bank"], fixture["protocol"])

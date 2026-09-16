"""Adversarial source-adapter and publication checks independent of outcomes."""

from __future__ import annotations

import hashlib
import csv
import io
import itertools
import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2 import source


PROTOCOL = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def _fixture(
    tmp_path: Path, *, pairs: list[dict] | None = None, generations: list[dict] | None = None
) -> Path:
    pair = {
        "schema_version": source.PAIR_SCHEMA,
        "source_record_id": "p",
        "source_pair_key": "p",
        "task": "gsm8k",
        "model_id": "test-model",
        "clean_text": "clean",
        "typo_text": "typo",
    }
    sources = []
    for role, records in (
        ("pairs", [pair] if pairs is None else pairs),
        ("generations", generations),
    ):
        if records is None:
            continue
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        sources.append(
            {
                "source_id": role,
                "source_kind": "submitted-archive",
                "role": role,
                "path": path.name,
                "expected_sha256": _hash(path),
                "format": "jsonl",
                "adapter_version": source.PAIR_SCHEMA
                if role == "pairs"
                else source.GENERATION_SCHEMA,
                "source_commit": None,
                "model_revision": None,
                "tokenizer_revision": None,
                "prompt_template_revision": None,
                "generation_config": None,
                "availability": "available",
                "reason_codes": [],
            }
        )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-archive-index/v2",
                "created_at": "2026-09-16T00:00:00Z",
                "source_namespace": "archive",
                "sources": sources,
                "cohorts": [],
                "identity_aliases_ref": None,
                "notes": "test",
            }
        ),
        encoding="utf-8",
    )
    return index


def _generation(**changes) -> dict:
    return {
        "schema_version": source.GENERATION_SCHEMA,
        "source_record_id": "g",
        "source_generation_key": "g",
        "source_pair_key": "p",
        "arm": "clean",
        "window": None,
        "raw_text": "Answer: 1",
        **changes,
    }


def _update_pair(tmp_path: Path, **changes) -> Path:
    _fixture(tmp_path)
    pair = _rows(tmp_path / "pairs.jsonl")[0]
    pair.update(changes)
    return _fixture(tmp_path, pairs=[pair])


def test_unicode_line_separators_are_text_not_jsonl_record_boundaries(tmp_path: Path) -> None:
    exact = "a\u2028b\u2029c\u0085d"
    index = _update_pair(tmp_path, clean_text=exact)
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert _rows(tmp_path / "out/pair_manifest.jsonl")[0]["clean_text"] == exact


def test_transitive_text_reference_is_verified_and_preserves_exact_bytes(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes("before\r\nquery e\u0301\r\nafter".encode("utf-8"))
    index = _update_pair(tmp_path, clean_prompt_ref={"path": prompt.name, "sha256": _hash(prompt)})
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    row = _rows(tmp_path / "out/pair_manifest.jsonl")[0]
    assert row["clean_prompt"] is None
    assert row["clean_prompt_ref"] == {"path": str(prompt), "sha256": _hash(prompt)}
    assert row["clean_prompt_sha256"] == _hash(prompt)
    prompt.write_bytes(b"changed")
    with pytest.raises(source.IntakeError, match="SHA256 mismatch"):
        source.run_intake(index, PROTOCOL, tmp_path / "rejected")


def test_both_inline_and_referenced_text_are_rejected(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"same")
    index = _update_pair(
        tmp_path,
        clean_prompt="same",
        clean_prompt_ref={"path": prompt.name, "sha256": _hash(prompt)},
    )
    with pytest.raises(source.IntakeError, match="exactly one"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_unknown_identity_stays_null_and_has_reason(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    row = _rows(tmp_path / "out/pair_manifest.jsonl")[0]
    assert row["original_problem_group_id"] is None
    assert "original_problem_identity_unknown" in row["reason_codes"]


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": 1},
        {"generated_token_ids": [True]},
        {"window": [0, 0]},
        {"archived_scoring": {"gold_match": 1}},
        {"archived_scoring": {"parser_id": 9}},
        {"stopping_metadata": {"eos_observed": 1}},
    ],
)
def test_malformed_generation_adapter_fields_fail_closed(tmp_path: Path, changes: dict) -> None:
    index = _fixture(tmp_path, generations=[_generation(**changes)])
    with pytest.raises(source.IntakeError):
        source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert not (tmp_path / "out/run.json").exists()


def test_duplicate_generation_slot_not_chosen_by_saved_score(tmp_path: Path) -> None:
    index = _fixture(
        tmp_path,
        generations=[
            _generation(archived_scoring={"gold_match": False}),
            _generation(
                source_record_id="other",
                source_generation_key="other",
                archived_scoring={"gold_match": True},
            ),
        ],
    )
    with pytest.raises(source.IntakeError, match="Duplicate generation pair/arm/window"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (None, "unknown"),
        ({}, "unknown"),
        ({"eos_observed": True, "length_cap_reached": True}, "eos"),
        ({"eos_observed": False, "length_cap_reached": True}, "length-cap"),
        ({"length_cap_reached": True}, "unknown"),
    ],
)
def test_termination_requires_observed_metadata(
    tmp_path: Path, metadata: dict | None, expected: str
) -> None:
    index = _fixture(tmp_path, generations=[_generation(stopping_metadata=metadata)])
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert _rows(tmp_path / "out/archive_generation_records.jsonl")[0]["termination"] == expected


def test_expected_missing_arm_is_not_fabricated(tmp_path: Path) -> None:
    index = _update_pair(
        tmp_path,
        expected_generations=[
            {"arm": "correct", "window": [0, 6], "source_generation_key": "missing-g"}
        ],
    )
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert _rows(tmp_path / "out/archive_generation_records.jsonl") == []
    (link,) = _rows(tmp_path / "out/pair_manifest.jsonl")[0]["archived_generations"]
    assert link["availability"] == "missing"
    assert link["record_ref"] is None
    assert link["generation_id"] is not None


def test_inputs_revalidated_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _fixture(tmp_path)
    original = source._json_bytes
    modified = False

    def modify_after_read(value):
        nonlocal modified
        if not modified:
            modified = True
            (tmp_path / "pairs.jsonl").write_bytes(b"changed during intake")
        return original(value)

    monkeypatch.setattr(source, "_json_bytes", modify_after_read)
    with pytest.raises(source.IntakeError, match="SHA256 mismatch|changed"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert not (tmp_path / "out/run.json").exists()
    assert not list(tmp_path.glob(".out.intake-*"))


@pytest.mark.parametrize("span", [[0, 5], [8, 100], [1, 6]])
def test_explicit_query_span_must_match_full_prompt(tmp_path: Path, span: list[int]) -> None:
    index = _update_pair(tmp_path, clean_prompt="example clean\nquery clean", clean_query_span=span)
    with pytest.raises(source.IntakeError, match="exact query"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_explicit_valid_repeated_query_span_is_preserved(tmp_path: Path) -> None:
    index = _update_pair(tmp_path, clean_prompt="clean\nclean", clean_query_span=[6, 11])
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert _rows(tmp_path / "out/pair_manifest.jsonl")[0]["clean_query_span"] == [6, 11]


@pytest.mark.parametrize(
    "gold",
    [
        42,
        {"numerator": "01", "denominator": "1"},
        {"numerator": "2", "denominator": "2"},
        {"numerator": "1", "denominator": "0"},
        {"numerator": "-0", "denominator": "1"},
    ],
)
def test_canonical_gold_cannot_claim_unvalidated_numeric_form(tmp_path: Path, gold) -> None:
    index = _update_pair(tmp_path, canonical_gold=gold, gold_canonicalization_version="archived-v1")
    with pytest.raises(source.IntakeError):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_valid_supplied_canonical_gold_is_preserved_not_computed(tmp_path: Path) -> None:
    gold = {"numerator": "-2501", "denominator": "2"}
    index = _update_pair(tmp_path, canonical_gold=gold, gold_canonicalization_version="archived-v1")
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    row = _rows(tmp_path / "out/pair_manifest.jsonl")[0]
    assert row["canonical_gold"] == gold
    assert row["gold"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"arm": "unsupported"},
        {"arm": "clean", "window": [0, 6]},
        {"arm": "correct", "window": None},
    ],
)
def test_arm_window_contract_rejects_unusable_slots(tmp_path: Path, changes: dict) -> None:
    index = _fixture(tmp_path, generations=[_generation(**changes)])
    with pytest.raises(source.IntakeError):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_unsupported_archived_stop_reason_is_preserved_as_unknown(tmp_path: Path) -> None:
    index = _fixture(
        tmp_path,
        generations=[
            _generation(termination="vendor-specific-stop", stopping_metadata={"reason": "custom"})
        ],
    )
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    row = _rows(tmp_path / "out/archive_generation_records.jsonl")[0]
    assert row["termination"] == "unknown"
    assert row["archived_termination"] == "vendor-specific-stop"


def test_source_kind_cannot_silently_mix_archive_with_regeneration(tmp_path: Path) -> None:
    index = _fixture(tmp_path, generations=[_generation()])
    payload = json.loads(index.read_text())
    payload["sources"][1]["source_kind"] = "public-regeneration"
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="source_kind"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_r3_expected_identity_cannot_claim_a_different_task_model(tmp_path: Path) -> None:
    index = _update_pair(tmp_path, task="mmlu-pro", model_id="Qwen/Qwen2.5-3B-Instruct")
    ids = tmp_path / "ids.json"
    ids.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": "R3",
                "source_namespace": "archive",
                "source_pair_keys": ["p"],
            }
        )
    )
    payload = json.loads(index.read_text())
    payload["cohorts"] = [
        {
            "cohort_id": "R3",
            "setting": "R3",
            "historical_n_reference": 172,
            "expected_ids_ref": {"path": ids.name, "sha256": _hash(ids)},
            "id_namespace": "archive",
            "membership_provenance_ref": None,
        }
    ]
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="frozen setting R3"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_structurally_unused_source_is_not_reported_missing(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    unused = {
        **payload["sources"][0],
        "source_id": "unused-donor",
        "role": "donor-bank",
        "path": None,
        "expected_sha256": None,
        "availability": "not_applicable",
        "reason_codes": ["cross_not_part_of_this_archive"],
        "adapter_version": None,
        "format": None,
    }
    payload["sources"].append(unused)
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    entry = next(
        row
        for row in _rows(tmp_path / "out/archive_inventory.jsonl")
        if row["source_id"] == "unused-donor"
    )
    assert entry["availability"] == "not_applicable"
    assert entry["reason_codes"] == ["cross_not_part_of_this_archive"]
    audit = json.loads((tmp_path / "out/source_audit.json").read_text())
    assert not any(row["source_id"] == "unused-donor" for row in audit["findings"])


def test_structurally_unused_source_cannot_hide_a_present_path(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    payload["sources"][0]["availability"] = "not_applicable"
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="not_applicable"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_malformed_alias_array_raises_intake_error_not_typeerror(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-identity-aliases/v2",
                "pair_aliases": None,
                "group_aliases": [],
            }
        )
    )
    payload = json.loads(index.read_text())
    payload["identity_aliases_ref"] = {"path": aliases.name, "sha256": _hash(aliases)}
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="arrays"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_unknown_and_known_empty_expected_generation_grids_stay_distinct(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    source.run_intake(index, PROTOCOL, tmp_path / "unknown")
    unknown = _rows(tmp_path / "unknown/pair_manifest.jsonl")[0]
    assert unknown["expected_generations"] is None
    assert "expected_generation_grid_unknown" in unknown["reason_codes"]
    index = _update_pair(tmp_path, expected_generations=[])
    source.run_intake(index, PROTOCOL, tmp_path / "empty")
    empty = _rows(tmp_path / "empty/pair_manifest.jsonl")[0]
    assert empty["expected_generations"] == []
    assert "expected_generation_grid_unknown" not in empty["reason_codes"]


def test_observed_generation_outside_explicit_grid_is_retained_and_flagged(tmp_path: Path) -> None:
    _fixture(tmp_path)
    pair = _rows(tmp_path / "pairs.jsonl")[0]
    pair["expected_generations"] = []
    index = _fixture(tmp_path, pairs=[pair], generations=[_generation()])
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert len(_rows(tmp_path / "out/archive_generation_records.jsonl")) == 1
    (link,) = _rows(tmp_path / "out/pair_manifest.jsonl")[0]["archived_generations"]
    assert "unexpected_generation_slot" in link["reason_codes"]
    audit = json.loads((tmp_path / "out/source_audit.json").read_text())
    assert any("unexpected_generation_slot" in row["reason_codes"] for row in audit["findings"])


def test_expected_generation_key_cannot_identify_two_missing_arms(tmp_path: Path) -> None:
    index = _update_pair(
        tmp_path,
        expected_generations=[
            {"arm": "clean", "window": None, "source_generation_key": "shared"},
            {"arm": "typo", "window": None, "source_generation_key": "shared"},
        ],
    )
    with pytest.raises(source.IntakeError, match="Conflicting generation identity"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")
    assert not (tmp_path / "out/run.json").exists()


@pytest.mark.parametrize("observed", [False, True])
def test_generation_key_cannot_identify_slots_on_two_pairs(tmp_path: Path, observed: bool) -> None:
    _fixture(tmp_path)
    pair = _rows(tmp_path / "pairs.jsonl")[0]
    slot = {"arm": "clean", "window": None, "source_generation_key": "g"}
    first = {**pair, "expected_generations": [] if observed else [slot]}
    second = {
        **pair,
        "source_record_id": "q",
        "source_pair_key": "q",
        "expected_generations": [slot],
    }
    index = _fixture(
        tmp_path, pairs=[first, second], generations=[_generation()] if observed else None
    )
    with pytest.raises(source.IntakeError, match="Conflicting generation identity"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_expected_identity_matching_its_observed_slot_is_accepted(tmp_path: Path) -> None:
    _fixture(tmp_path)
    pair = _rows(tmp_path / "pairs.jsonl")[0]
    pair["expected_generations"] = [{"arm": "clean", "window": None, "source_generation_key": "g"}]
    index = _fixture(tmp_path, pairs=[pair], generations=[_generation()])
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (link,) = _rows(tmp_path / "out/pair_manifest.jsonl")[0]["archived_generations"]
    assert link["availability"] == "available"


@pytest.mark.parametrize("setting,wrong_reference", [("R3", 171), ("R4", 98)])
def test_known_setting_historical_reference_is_frozen_not_observed_quota(
    tmp_path: Path, setting: str, wrong_reference: int
) -> None:
    index = _fixture(tmp_path, pairs=[])
    payload = json.loads(index.read_text())
    payload["cohorts"] = [
        {
            "cohort_id": setting,
            "setting": setting,
            "historical_n_reference": wrong_reference,
            "expected_ids_ref": None,
            "id_namespace": "archive",
            "membership_provenance_ref": None,
        }
    ]
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="historical_n_reference.*frozen setting"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")
    payload["cohorts"][0]["historical_n_reference"] = 172 if setting == "R3" else 97
    index.write_text(json.dumps(payload))
    # Zero observations remain a valid audited run, not a fabricated size quota.
    assert source.run_intake(index, PROTOCOL, tmp_path / "accepted")["run_status"] == "complete"


def test_wrong_setting_extra_is_retained_excluded_and_reported(tmp_path: Path) -> None:
    _fixture(tmp_path)
    pair = _rows(tmp_path / "pairs.jsonl")[0]
    pair.update(model_id="google/gemma-3-4b-it", cohort_ids=["R3"])
    extra = {
        **pair,
        "source_record_id": "q",
        "source_pair_key": "q",
        "task": "mmlu-pro",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
    }
    index = _fixture(tmp_path, pairs=[pair, extra])
    ids = tmp_path / "ids.json"
    ids.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": "R3",
                "source_namespace": "archive",
                "source_pair_keys": ["p"],
            }
        )
    )
    payload = json.loads(index.read_text())
    payload["cohorts"] = [
        {
            "cohort_id": "R3",
            "setting": "R3",
            "historical_n_reference": 172,
            "expected_ids_ref": {"path": ids.name, "sha256": _hash(ids)},
            "id_namespace": "archive",
            "membership_provenance_ref": None,
        }
    ]
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    audit = json.loads((tmp_path / "out/source_audit.json").read_text())
    (cohort,) = audit["cohorts"]
    assert cohort["recovered_count"] == 1
    assert cohort["extra_count"] == 1
    assert "extra_pairs_excluded" in cohort["reason_codes"]
    rows = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert (
        next(row for row in rows if row["source_pair_key"] == "q")["cohort_membership"][0][
            "membership"
        ]
        == "extra"
    )


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"eos_observed": True, "length_cap_reached": True}, "eos"),
        ({"eos_observed": False, "length_cap_reached": True}, "length-cap"),
    ],
)
def test_observed_stop_flags_win_over_unknown_raw_termination_label(
    tmp_path: Path, metadata: dict, expected: str
) -> None:
    index = _fixture(
        tmp_path, generations=[_generation(termination="stop", stopping_metadata=metadata)]
    )
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/archive_generation_records.jsonl")
    assert row["termination"] == expected
    assert row["archived_termination"] == "stop"
    assert "archived_termination_unsupported" in row["reason_codes"]


@pytest.mark.parametrize("explicit_null", [False, True])
def test_pair_runtime_inherits_header_only_when_omitted(
    tmp_path: Path, explicit_null: bool
) -> None:
    index = _update_pair(tmp_path, **({"archived_runtime": None} if explicit_null else {}))
    payload = json.loads(index.read_text())
    header = {"max_new_tokens": 512}
    payload["sources"][0]["generation_config"] = header
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert row["archived_runtime"] == (None if explicit_null else header)
    assert ("archived_runtime_unknown" in row["reason_codes"]) is explicit_null


def test_duplicate_pair_across_files_with_different_hashes_preserves_both_sources(
    tmp_path: Path,
) -> None:
    index = _fixture(tmp_path)
    second = tmp_path / "second.jsonl"
    (pair,) = _rows(tmp_path / "pairs.jsonl")
    pair["source_record_id"] = "another-record"
    second.write_text(json.dumps(pair) + "\n\n")
    payload = json.loads(index.read_text())
    payload["sources"].append(
        {
            **payload["sources"][0],
            "source_id": "second",
            "path": second.name,
            "expected_sha256": _hash(second),
        }
    )
    index.write_text(json.dumps(payload))
    assert _hash(second) != _hash(tmp_path / "pairs.jsonl")
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert row["source_ref"]["artifact"]["sha256"] == _hash(tmp_path / "pairs.jsonl")
    assert row["additional_source_refs"][0]["artifact"]["sha256"] == _hash(second)


def test_alias_identity_uses_verified_captured_object_not_reopened_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _fixture(tmp_path)
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Owner-confirmed alias equivalence")
    alias = {
        "schema_version": "rebuttal-identity-aliases/v2",
        "pair_aliases": [
            {
                "alias": {"source_namespace": "archive", "source_pair_key": "p"},
                "canonical": {"source_namespace": "archive", "source_pair_key": "verified"},
                "evidence_ref": {"path": evidence.name, "sha256": _hash(evidence)},
            }
        ],
        "group_aliases": [],
    }
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps(alias))
    verified_bytes = alias_path.read_bytes()
    payload = json.loads(index.read_text())
    payload["identity_aliases_ref"] = {"path": alias_path.name, "sha256": _hash(alias_path)}
    index.write_text(json.dumps(payload))
    parser = source.parse_identity_aliases

    def mutate_path_while_parsing(captured, source_path):
        altered = json.loads(verified_bytes)
        altered["pair_aliases"][0]["canonical"]["source_pair_key"] = "unverified"
        alias_path.write_text(json.dumps(altered))
        try:
            return parser(captured, source_path=source_path)
        finally:
            alias_path.write_bytes(verified_bytes)

    monkeypatch.setattr(source, "parse_identity_aliases", mutate_path_while_parsing)
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert row["source_pair_key"] == "verified"


@pytest.mark.parametrize("dangerous", ["=1+1", " +1", "\t-1", "\x01@SUM(1)", "\u2000=2"])
def test_csv_formula_cells_are_escaped_but_numeric_types_are_unchanged(dangerous: str) -> None:
    exported = source._csv_bytes([{"raw": dangerous, "numeric": -1}], ["raw", "numeric"])
    (row,) = list(csv.DictReader(io.StringIO(exported.decode())))
    assert row["raw"] == "'" + dangerous
    assert row["numeric"] == "-1"


def test_dangerous_missing_source_id_remains_exact_in_json_only_csv_is_escaped(
    tmp_path: Path,
) -> None:
    index = _fixture(tmp_path, pairs=[])
    ids = tmp_path / "ids.json"
    ids.write_text(
        json.dumps(
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": "R3",
                "source_namespace": "archive",
                "source_pair_keys": ["=1+1"],
            }
        )
    )
    payload = json.loads(index.read_text())
    payload["cohorts"] = [
        {
            "cohort_id": "R3",
            "setting": "R3",
            "historical_n_reference": 172,
            "expected_ids_ref": {"path": ids.name, "sha256": _hash(ids)},
            "id_namespace": "archive",
            "membership_provenance_ref": None,
        }
    ]
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    audit = json.loads((tmp_path / "out/source_audit.json").read_text())
    assert audit["cohorts"][0]["missing_source_pair_keys"] == ["=1+1"]
    rows = list(csv.DictReader(io.StringIO((tmp_path / "out/missing_inputs.csv").read_text())))
    assert any(row["component"] == "'=1+1" for row in rows)


def _separate_pair_sources(tmp_path: Path, pairs: list[dict]) -> Path:
    index = _fixture(tmp_path, pairs=[])
    payload = json.loads(index.read_text())
    template = payload["sources"][0]
    payload["sources"] = []
    for number, pair in enumerate(pairs):
        path = tmp_path / f"pair-source-{number}.jsonl"
        path.write_text(json.dumps(pair) + "\n")
        payload["sources"].append(
            {
                **template,
                "source_id": f"pairs-{number}",
                "path": path.name,
                "expected_sha256": _hash(path),
            }
        )
    index.write_text(json.dumps(payload))
    return index


def test_duplicate_historical_unknown_and_known_merge_without_source_order_dependence(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    (pair,) = _rows(tmp_path / "pairs.jsonl")
    index = _separate_pair_sources(
        tmp_path,
        [pair, {**pair, "source_record_id": "known", "historical_pair_id": "historical-p"}],
    )
    source.run_intake(index, PROTOCOL, tmp_path / "forward")
    payload = json.loads(index.read_text())
    payload["sources"].reverse()
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "reverse")
    (first,) = _rows(tmp_path / "forward/pair_manifest.jsonl")
    (second,) = _rows(tmp_path / "reverse/pair_manifest.jsonl")
    assert first == second
    assert first["historical_pair_id"] == "historical-p"
    assert first["source_pair_key_kind"] == "historical-id"
    assert "historical_pair_id_unknown" not in first["reason_codes"]
    assert {item["historical_pair_id"] for item in first["source_identity_provenance"]} == {
        None,
        "historical-p",
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        ({"historical_pair_id": "first"}, {"historical_pair_id": "second"}),
        ({"source_pair_key_kind": "historical-id"}, {"source_pair_key_kind": "producer-id"}),
    ],
)
def test_same_original_pair_rejects_conflicting_known_identity_metadata(
    tmp_path: Path, reverse: bool, metadata: tuple[dict, dict]
) -> None:
    _fixture(tmp_path)
    (pair,) = _rows(tmp_path / "pairs.jsonl")
    rows = [{**pair, **metadata[0]}, {**pair, "source_record_id": "other", **metadata[1]}]
    index = _separate_pair_sources(tmp_path, list(reversed(rows)) if reverse else rows)
    with pytest.raises(
        source.IntakeError,
        match="Conflicting.*historical_pair_id|Conflicting.*source_pair_key_kind",
    ):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_explicit_producer_kind_is_preserved_when_only_inferred_default_changes(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    (pair,) = _rows(tmp_path / "pairs.jsonl")
    index = _separate_pair_sources(
        tmp_path,
        [
            {**pair, "source_pair_key_kind": "producer-id"},
            {**pair, "source_record_id": "other", "historical_pair_id": "paper-id"},
        ],
    )
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert row["historical_pair_id"] == "paper-id"
    assert row["source_pair_key_kind"] == "producer-id"


def _three_alias_sources(tmp_path: Path) -> Path:
    _fixture(tmp_path)
    (base,) = _rows(tmp_path / "pairs.jsonl")
    pairs = [
        {
            **base,
            "source_pair_key": key,
            "source_record_id": key,
            "historical_pair_id": f"history-{key}",
            "source_pair_key_kind": "producer-id" if key == "p" else "historical-id",
            "dataset_id": "dataset",
            "split": "test",
            "original_problem_id": key,
        }
        for key in ("p", "q", "canonical")
    ]
    index = _separate_pair_sources(tmp_path, pairs)
    evidence = tmp_path / "aliases-evidence.txt"
    evidence.write_text("Owner-confirmed equivalent identities")
    ref = {"path": evidence.name, "sha256": _hash(evidence)}
    aliases = {
        "schema_version": "rebuttal-identity-aliases/v2",
        "pair_aliases": [
            {
                "alias": {"source_namespace": "archive", "source_pair_key": key},
                "canonical": {"source_namespace": "archive", "source_pair_key": "canonical"},
                "evidence_ref": ref,
            }
            for key in ("p", "q")
        ],
        "group_aliases": [
            {
                "alias": {"dataset_id": "dataset", "split": "test", "original_problem_id": key},
                "canonical": {
                    "dataset_id": "dataset",
                    "split": "test",
                    "original_problem_id": "canonical",
                },
                "evidence_ref": ref,
            }
            for key in ("p", "q")
        ],
    }
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(aliases))
    payload = json.loads(index.read_text())
    payload["identity_aliases_ref"] = {"path": path.name, "sha256": _hash(path)}
    index.write_text(json.dumps(payload))
    return index


def test_three_explicit_alias_acquisitions_preserve_all_provenance_in_any_order(
    tmp_path: Path,
) -> None:
    index = _three_alias_sources(tmp_path)
    payload = json.loads(index.read_text())
    sources = payload["sources"]
    expected = None
    for number, permutation in enumerate(itertools.permutations(sources)):
        payload["sources"] = list(permutation)
        index.write_text(json.dumps(payload))
        output = tmp_path / f"order-{number}"
        source.run_intake(index, PROTOCOL, output)
        (row,) = _rows(output / "pair_manifest.jsonl")
        if expected is None:
            expected = row
        assert row == expected
    assert expected["historical_pair_id"] == "history-canonical"
    assert expected["source_pair_key_kind"] == "historical-id"
    provenance = expected["source_identity_provenance"]
    assert len(provenance) == 3
    assert {item["historical_pair_id"] for item in provenance} == {
        "history-p",
        "history-q",
        "history-canonical",
    }
    assert sum(len(item["pair_identity"]["alias_evidence_refs"]) for item in provenance) == 2
    assert sum(len(item["group_identity"]["alias_evidence_refs"]) for item in provenance) == 2


def test_same_original_conflict_is_not_excused_by_an_alias_mapping(tmp_path: Path) -> None:
    index = _three_alias_sources(tmp_path)
    payload = json.loads(index.read_text())
    source_entry = payload["sources"][0]
    path = tmp_path / source_entry["path"]
    (pair,) = _rows(path)
    path.write_text(
        json.dumps(pair)
        + "\n"
        + json.dumps(
            {**pair, "source_record_id": "conflict", "historical_pair_id": "different-p-history"}
        )
        + "\n"
    )
    source_entry["expected_sha256"] = _hash(path)
    index.write_text(json.dumps(payload))
    with pytest.raises(source.IntakeError, match="Conflicting historical_pair_id"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")


def test_alias_representative_without_canonical_acquisition_is_lexically_stable(
    tmp_path: Path,
) -> None:
    index = _three_alias_sources(tmp_path)
    payload = json.loads(index.read_text())
    payload["sources"] = payload["sources"][:2]
    rows = []
    for number in range(2):
        payload["sources"].reverse()
        index.write_text(json.dumps(payload))
        output = tmp_path / f"without-canonical-{number}"
        source.run_intake(index, PROTOCOL, output)
        rows.extend(_rows(output / "pair_manifest.jsonl"))
    assert rows[0] == rows[1]
    assert rows[0]["historical_pair_id"] == "history-p"
    assert rows[0]["source_pair_key_kind"] == "producer-id"


def test_single_and_duplicate_manifest_rows_share_provenance_keys(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (base,) = _rows(tmp_path / "pairs.jsonl")
    index = _separate_pair_sources(
        tmp_path,
        [
            base,
            {**base, "source_record_id": "same-pair"},
            {**base, "source_pair_key": "solo", "source_record_id": "solo"},
        ],
    )
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    rows = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert set(rows[0]) == set(rows[1])
    solo = next(row for row in rows if row["source_pair_key"] == "solo")
    assert solo["additional_source_refs"] == []
    assert len(solo["source_identity_provenance"]) == 1


@pytest.mark.parametrize("reference_mode", ["claimant", "expected_ids", "membership_provenance"])
def test_cohort_reference_tracks_actual_claimant_not_selected_pair_representative(
    tmp_path: Path, reference_mode: str
) -> None:
    _fixture(tmp_path)
    (base,) = _rows(tmp_path / "pairs.jsonl")
    base["model_id"] = "google/gemma-3-4b-it"
    known = {**base, "historical_pair_id": "known-history", "cohort_ids": []}
    claimant = {**base, "source_record_id": "claimant", "cohort_ids": ["R3"]}
    index = _separate_pair_sources(tmp_path, [known, claimant])
    payload = json.loads(index.read_text())
    cohort = {
        "cohort_id": "R3",
        "setting": "R3",
        "historical_n_reference": 172,
        "expected_ids_ref": None,
        "id_namespace": "archive",
        "membership_provenance_ref": None,
    }
    authoritative_ref = None
    if reference_mode != "claimant":
        evidence = tmp_path / "membership.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": "rebuttal-cohort-ids/v2",
                    "cohort_id": "R3",
                    "source_namespace": "archive",
                    "source_pair_keys": ["p"],
                }
            )
        )
        authoritative_ref = {"path": str(evidence), "sha256": _hash(evidence)}
        cohort[
            "expected_ids_ref" if reference_mode == "expected_ids" else "membership_provenance_ref"
        ] = authoritative_ref
    payload["cohorts"] = [cohort]
    rows = []
    for number in range(2):
        payload["sources"].reverse()
        index.write_text(json.dumps(payload))
        output = tmp_path / f"claimant-order-{number}"
        source.run_intake(index, PROTOCOL, output)
        rows.extend(_rows(output / "pair_manifest.jsonl"))
    assert rows[0] == rows[1]
    row = rows[0]
    assert row["source_ref"]["artifact"]["path"] == str(tmp_path / "pair-source-0.jsonl")
    membership_ref = row["cohort_membership"][0]["source_ref"]
    assert set(membership_ref) == {"path", "sha256"}
    if authoritative_ref is not None:
        assert membership_ref == authoritative_ref
    else:
        claimant_path = tmp_path / "pair-source-1.jsonl"
        assert membership_ref == {"path": str(claimant_path), "sha256": _hash(claimant_path)}
    claimants = [item for item in row["source_identity_provenance"] if item["cohort_ids"] == ["R3"]]
    assert len(claimants) == 1
    assert claimants[0]["source_ref"]["record_id"] == "claimant"


@pytest.mark.parametrize("field", ["clean_text", "typo_text", "clean_prompt", "typo_prompt"])
@pytest.mark.parametrize(
    "storage", [("inline", "ref"), ("ref", "ref"), ("ref", "inline", "ref"), ("ref", "ref", "ref")]
)
def test_duplicate_exact_text_storage_has_order_independent_manifest_bytes(
    tmp_path: Path, field: str, storage: tuple[str, ...]
) -> None:
    _fixture(tmp_path)
    (base,) = _rows(tmp_path / "pairs.jsonl")
    exact = "  identical é\r\n e\u0301\u2028tail\n"
    pairs = []
    text_refs = []
    for number, mode in enumerate(storage):
        pair = {**base, "source_record_id": f"record-{number}", field: None}
        if mode == "inline":
            pair[field] = exact
        else:
            # Reverse path order relative to source order to exercise the
            # reference minimum independently of the selected acquisition.
            path = tmp_path / f"text-{len(storage) - number}.txt"
            path.write_bytes(exact.encode("utf-8"))
            pair[f"{field}_ref"] = {"path": path.name, "sha256": _hash(path)}
            text_refs.append({"path": str(path), "sha256": _hash(path)})
        pairs.append(pair)
    index = _separate_pair_sources(tmp_path, pairs)
    payload = json.loads(index.read_text())
    sources = payload["sources"]
    expected_bytes = None
    expected_hash = None
    for number, permutation in enumerate(itertools.permutations(sources)):
        payload["sources"] = list(permutation)
        index.write_text(json.dumps(payload))
        output = tmp_path / f"storage-order-{number}"
        source.run_intake(index, PROTOCOL, output)
        manifest = output / "pair_manifest.jsonl"
        observed_bytes = manifest.read_bytes()
        observed_hash = _hash(manifest)
        if expected_bytes is None:
            expected_bytes, expected_hash = observed_bytes, observed_hash
        assert observed_bytes == expected_bytes
        assert observed_hash == expected_hash
        (row,) = _rows(manifest)
        if "inline" in storage:
            assert row[field] == exact
            assert row[f"{field}_ref"] is None
        else:
            assert row[field] is None
            assert row[f"{field}_ref"] == min(text_refs, key=source.canonical_json)
        assert row[f"{field}_sha256"] == hashlib.sha256(exact.encode("utf-8")).hexdigest()
        assert len(row["source_identity_provenance"]) == len(storage)


def test_repeated_source_declarations_do_not_add_self_references(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    payload["sources"].append(
        {**payload["sources"][0], "source_id": "same-file-second-declaration"}
    )
    index.write_text(json.dumps(payload))
    source.run_intake(index, PROTOCOL, tmp_path / "out")
    (row,) = _rows(tmp_path / "out/pair_manifest.jsonl")
    assert row["additional_source_refs"] == []
    assert len(row["source_identity_provenance"]) == 2
    inventory = _rows(tmp_path / "out/archive_inventory.jsonl")
    assert {item["source_id"] for item in inventory} == {"pairs", "same-file-second-declaration"}


def test_duplicate_expected_generation_disagreement_still_fails_generic_comparison(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    (base,) = _rows(tmp_path / "pairs.jsonl")
    first = {**base, "expected_generations": []}
    second = {
        **base,
        "source_record_id": "other",
        "expected_generations": [{"arm": "clean", "window": None}],
    }
    index = _separate_pair_sources(tmp_path, [first, second])
    with pytest.raises(source.IntakeError, match="fields: expected_generations"):
        source.run_intake(index, PROTOCOL, tmp_path / "out")

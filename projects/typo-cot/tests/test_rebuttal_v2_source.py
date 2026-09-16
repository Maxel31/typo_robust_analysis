"""Adversarial source-adapter and publication checks independent of outcomes."""

from __future__ import annotations

import hashlib
import csv
import io
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

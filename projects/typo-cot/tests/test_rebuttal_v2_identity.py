"""Independent byte, identity and provenance fixtures for v2 intake primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2.identity import (
    identity_hash,
    load_identity_aliases,
    original_problem_group_id_for,
    pair_id_for,
)
from typo_cot.experiments.rebuttal_v2.schemas import (
    IntakeError,
    canonical_json,
    canonical_sha256,
    iter_jsonl_objects,
    load_json_object,
    load_protocol,
    make_ref,
    require_bool,
    require_date,
    require_enum,
    require_int,
    require_keys,
    require_nullable_string,
    require_sha256,
    require_string_list,
    require_timestamp,
    sha256_file,
    sha256_text,
    strict_loads,
    validate_record_ref,
    verify_ref,
)


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"


def test_canonical_bytes_and_hash_match_fixed_utf8_vector() -> None:
    value = {"z": [-0.0, 0, "é\n"], "a": 1}
    assert canonical_json(value) == '{"a":1,"z":[0.0,0,"é\\n"]}'
    assert canonical_sha256(value) == "00d871b2d43e3cda9a387eaf76e73fba89a31abef669b5e76382880dd5c9bb87"
    assert canonical_json({"nested": {"zero": -0.0}, "integer": 0}) == (
        '{"integer":0,"nested":{"zero":0.0}}'
    )
    assert canonical_json({"n": 1}) != canonical_json({"n": 1.0})
    assert canonical_sha256("é") != canonical_sha256("e\u0301")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"x": [float("-inf")]}, (1, 2), {1: "x"}, {True: "x"}, "\ud800"])
def test_canonical_rejects_non_json_or_nonfinite_values(value: object) -> None:
    with pytest.raises(IntakeError):
        canonical_json(value)


def test_canonical_rejects_cyclic_containers() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(IntakeError):
        canonical_json(value)


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1,"x":2}',
        '{"outer":{"x":1,"x":1}}',
        r'{"a":1,"\u0061":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
        '{"x":[-1e999]}',
        r'{"x":"\ud800"}',
    ],
)
def test_strict_json_rejects_ambiguous_or_unrepresentable_input(text: str) -> None:
    with pytest.raises(IntakeError, match="fixture.json"):
        strict_loads(text, context="fixture.json")


def test_jsonl_keeps_physical_line_numbers_and_allows_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'\n{"x":1}\r\n \t\r\n{"x":2}\n')
    assert list(iter_jsonl_objects(path)) == [(2, {"x": 1}), (4, {"x": 2})]
    path.write_text('\n{"x":1}\n\n[2]\n', encoding="utf-8")
    with pytest.raises(IntakeError, match=r"records.jsonl:4"):
        list(iter_jsonl_objects(path))


def test_json_object_loader_rejects_non_object_and_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(IntakeError, match="JSON object"):
        load_json_object(path)
    path.write_bytes(b'"\xff"')
    with pytest.raises(IntakeError, match="cannot read JSON"):
        load_json_object(path)


def test_pair_and_original_group_match_independent_fixed_identity_vectors() -> None:
    assert pair_id_for("submitted-2026", "pair-7") == (
        "c402e56a6bf5d90f3f629916fb2c7f9ca1d5ec87a12bb1aa5e116f6e3339e3d9"
    )
    group = original_problem_group_id_for("gsm8k", "test", "7")
    assert group == "ed68a1326066570edd2a830ca9d3cd6ed44d1418de8c71effb7b1e9a7810430d"
    # Model, target, perturbation and dataset revision are not grouping inputs.
    records = [
        {"model": "gemma", "target_rule": "random", "dataset_revision": None},
        {"model": "qwen", "target_rule": "attribution", "dataset_revision": "new-snapshot"},
    ]
    assert {
        original_problem_group_id_for("gsm8k", "test", "7")
        for _record in records
    } == {group}
    assert original_problem_group_id_for("gsm8k", "train", "7") != group
    assert identity_hash("different-domain/v2", {"source_namespace": "submitted-2026", "source_pair_key": "pair-7"}) != pair_id_for("submitted-2026", "pair-7")


@pytest.mark.parametrize("value", [True, False, 0, 7, 1.0, "", [], {}])
def test_ids_do_not_coerce_booleans_numbers_or_empty_values(value: object) -> None:
    with pytest.raises(IntakeError):
        pair_id_for("archive", value)  # type: ignore[arg-type]
    with pytest.raises(IntakeError):
        original_problem_group_id_for("gsm8k", "test", value)  # type: ignore[arg-type]
    # An unknown component does not excuse a malformed different component.
    with pytest.raises(IntakeError):
        original_problem_group_id_for(None, value, "7")  # type: ignore[arg-type]


def test_unknown_original_identity_stays_unknown_and_identity_payloads_are_integer_only() -> None:
    assert original_problem_group_id_for("gsm8k", "test", None) is None
    assert original_problem_group_id_for(None, "test", "7") is None
    with pytest.raises(IntakeError, match="without floats"):
        identity_hash("record/v2", {"id": 1.0})
    with pytest.raises(IntakeError, match="single line"):
        identity_hash("record\nv2", {"id": 1})


def test_text_and_file_hashes_preserve_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "exact.txt"
    content = "é\r\n e\u0301\n"
    path.write_bytes(content.encode("utf-8"))
    expected = hashlib.sha256(b"\xc3\xa9\r\n e\xcc\x81\n").hexdigest()
    assert sha256_file(path) == expected
    assert sha256_text(content) == expected
    assert sha256_text(content.replace("\r\n", "\n")) != expected


def test_artifact_refs_are_relative_to_containing_file_and_detect_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.bin"
    data.write_bytes(b"archived generation\r\n")
    manifest = tmp_path / "out" / "manifest.jsonl"
    ref = make_ref(data, relative_to_file=manifest)
    assert ref == {
        "path": "../data.bin",
        "sha256": hashlib.sha256(b"archived generation\r\n").hexdigest(),
    }
    monkeypatch.chdir(tmp_path.parent)
    assert verify_ref(ref, relative_to_file=manifest) == data
    assert validate_record_ref({"artifact": ref, "record_id": "historical-7"}) == {
        "artifact": ref, "record_id": "historical-7"
    }
    data.write_bytes(b"modified")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        verify_ref(ref, relative_to_file=manifest)


@pytest.mark.parametrize(
    "ref",
    [
        {"path": "missing"},
        {"path": True, "sha256": "0" * 64},
        {"path": "missing", "sha256": "A" * 64},
        {"path": "missing", "sha256": "0" * 64, "extra": "ignored?"},
    ],
)
def test_artifact_refs_reject_bad_shape(tmp_path: Path, ref: object) -> None:
    with pytest.raises(IntakeError):
        verify_ref(ref, relative_to_file=tmp_path / "manifest.json")


def test_missing_artifact_and_invalid_record_reference_fail(tmp_path: Path) -> None:
    with pytest.raises(IntakeError, match="cannot hash artifact"):
        verify_ref({"path": "missing", "sha256": "0" * 64}, tmp_path / "manifest.json")
    with pytest.raises(IntakeError, match="record_id"):
        validate_record_ref({"artifact": {"path": "x", "sha256": "0" * 64}, "record_id": True})


def test_frozen_protocol_matches_independent_digest_and_relocated_format(tmp_path: Path) -> None:
    actual = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_sha = "d13794e9d29f1c1750c66f996475ba05f7932e239768963ef1014b562868e70b"
    assert hashlib.sha256(json.dumps(actual, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest() == expected_sha
    assert canonical_sha256(actual) == expected_sha
    relocated = tmp_path / "arbitrary-name.json"
    relocated.write_text(json.dumps(actual, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    assert load_protocol(relocated) == actual
    assert load_protocol(PROTOCOL_PATH) == actual


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("schema_version", "rebuttal-protocol/v1", "schema_version"),
        ("protocol_revision", "2.1.0", "protocol_revision"),
        ("seed", 43, "frozen revision"),
        ("seed", True, "frozen revision"),
        ("unrecognized", 1, "frozen revision"),
    ],
)
def test_protocol_rejects_changed_contracts_even_at_expected_filename(tmp_path: Path, key: str, value: object, error: str) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload[key] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntakeError, match=error):
        load_protocol(path)


def test_protocol_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    original = PROTOCOL_PATH.read_text(encoding="utf-8")
    path.write_text(original.replace('"seed": 42', '"seed": 42, "seed": 42', 1), encoding="utf-8")
    with pytest.raises(IntakeError, match="duplicate JSON key"):
        load_protocol(path)
    path.write_text(original.replace('"seed": 42', '"seed": 1e999', 1), encoding="utf-8")
    with pytest.raises(IntakeError, match="nonfinite"):
        load_protocol(path)


def _alias_artifact(tmp_path: Path, pair_edges: list[tuple[str, str]], group_edges: list[tuple[str, str]] | None = None) -> Path:
    evidence = tmp_path / "equivalence.txt"
    evidence.write_bytes(b"archival identity crosswalk")
    ref = {"path": "equivalence.txt", "sha256": hashlib.sha256(b"archival identity crosswalk").hexdigest()}
    pairs = [
        {"alias": {"source_namespace": "archive", "source_pair_key": alias},
         "canonical": {"source_namespace": "archive", "source_pair_key": target},
         "evidence_ref": ref}
        for alias, target in pair_edges
    ]
    groups = [
        {"alias": {"dataset_id": dataset, "split": "test", "original_problem_id": "7"},
         "canonical": {"dataset_id": canonical, "split": "test", "original_problem_id": "7"},
         "evidence_ref": ref}
        for dataset, canonical in (group_edges or [])
    ]
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"schema_version": "rebuttal-identity-aliases/v2", "pair_aliases": pairs, "group_aliases": groups}), encoding="utf-8")
    return path


def test_alias_chains_preserve_original_identity_and_verified_evidence(tmp_path: Path) -> None:
    path = _alias_artifact(tmp_path, [("old", "middle"), ("middle", "canonical")], [("reissued", "gsm8k")])
    aliases = load_identity_aliases(path)
    resolved = aliases.resolve_pair("archive", "old")
    assert resolved.original == {"source_namespace": "archive", "source_pair_key": "old"}
    assert resolved.canonical == {"source_namespace": "archive", "source_pair_key": "canonical"}
    assert len(resolved.evidence_refs) == 2
    assert all(ref["path"] == str(tmp_path / "equivalence.txt") for ref in resolved.evidence_refs)
    resolved.canonical["source_pair_key"] = "caller-modification"
    assert aliases.resolve_pair("archive", "old").canonical["source_pair_key"] == "canonical"
    group = aliases.resolve_group("reissued", "test", "7")
    assert group.original["dataset_id"] == "reissued"
    assert group.canonical == {"dataset_id": "gsm8k", "split": "test", "original_problem_id": "7"}
    assert original_problem_group_id_for(**group.canonical) == original_problem_group_id_for("gsm8k", "test", "7")
    unknown = aliases.resolve_group("reissued", "test", None)
    assert unknown.canonical == unknown.original
    assert unknown.evidence_refs == ()


@pytest.mark.parametrize("edges", [[("x", "x")], [("x", "y"), ("y", "x")], [("unused-x", "unused-y"), ("unused-y", "unused-x")]])
def test_alias_cycles_are_rejected_even_before_resolution(tmp_path: Path, edges: list[tuple[str, str]]) -> None:
    with pytest.raises(IntakeError, match="alias cycle"):
        load_identity_aliases(_alias_artifact(tmp_path, edges))


@pytest.mark.parametrize("edges", [[("x", "y"), ("x", "y")], [("x", "y"), ("x", "z")]])
def test_duplicate_alias_keys_are_not_deduplicated(tmp_path: Path, edges: list[tuple[str, str]]) -> None:
    with pytest.raises(IntakeError, match="duplicate alias"):
        load_identity_aliases(_alias_artifact(tmp_path, edges))


def test_alias_evidence_must_exist_and_match_hash(tmp_path: Path) -> None:
    path = _alias_artifact(tmp_path, [("old", "new")])
    evidence = tmp_path / "equivalence.txt"
    evidence.write_text("different evidence", encoding="utf-8")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        load_identity_aliases(path)
    evidence.unlink()
    with pytest.raises(IntakeError, match="cannot hash artifact"):
        load_identity_aliases(path)


def test_aliases_reject_unknown_schema_extra_fields_and_malformed_ids(tmp_path: Path) -> None:
    path = _alias_artifact(tmp_path, [("old", "new")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "rebuttal-identity-aliases/v9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntakeError, match="schema_version"):
        load_identity_aliases(path)
    payload["schema_version"] = "rebuttal-identity-aliases/v2"
    payload["pair_aliases"][0]["alias"]["source_pair_key"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntakeError, match="source_pair_key"):
        load_identity_aliases(path)
    payload["pair_aliases"][0]["alias"]["source_pair_key"] = "old"
    payload["extra"] = "silent schema extension"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntakeError, match="unknown fields"):
        load_identity_aliases(path)


def test_absent_alias_file_does_not_invent_equivalence() -> None:
    aliases = load_identity_aliases(None)
    result = aliases.resolve_pair("archive", "7")
    assert result.original == result.canonical == {"source_namespace": "archive", "source_pair_key": "7"}
    assert result.evidence_refs == ()


def test_scalar_helpers_enforce_types_dates_and_declared_fields() -> None:
    assert require_int(0, "count", minimum=0) == 0
    assert require_bool(False, "flag") is False
    assert require_nullable_string(None, "revision") is None
    assert require_enum("unknown", ("known", "unknown"), "state") == "unknown"
    assert require_sha256("0" * 64, "digest") == "0" * 64
    assert require_timestamp("2026-09-16T06:47:07Z", "created_at") == "2026-09-16T06:47:07Z"
    assert require_date("2024-02-29", "date") == "2024-02-29"
    assert require_string_list([], "empty-known-list", unique=True) == []
    require_keys({"a": 1, "b": 2}, ("a",), optional=("b",))
    for value in [True, 1.0, "1", -1]:
        with pytest.raises(IntakeError):
            require_int(value, "count", minimum=0)
    with pytest.raises(IntakeError):
        require_bool(1, "flag")
    with pytest.raises(IntakeError, match="duplicate"):
        require_string_list(["x", "x"], "ids", unique=True)
    with pytest.raises(IntakeError, match="missing"):
        require_keys({}, ("a",))
    with pytest.raises(IntakeError, match="unknown"):
        require_keys({"a": 1, "extra": 2}, ("a",))
    for timestamp in ["2026-09-16T06:47Z", "2026-09-16T06:47:07+00:00", "2026-02-30T06:47:07Z"]:
        with pytest.raises(IntakeError):
            require_timestamp(timestamp, "created_at")
    with pytest.raises(IntakeError):
        require_date("2025-02-29", "date")

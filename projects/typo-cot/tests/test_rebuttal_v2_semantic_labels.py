"""Read-only semantic-label intake through the complete public pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_cot.experiments.rebuttal_v2 import annotations
from typo_cot.experiments.rebuttal_v2.input_audit import load_input_audit, run_input_audit
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError
from typo_cot.experiments.rebuttal_v2.semantic_labels import load_semantic_labels


TIME = "2026-09-16T12:00:00Z"

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_semantic_labels_input_fixture", Path(__file__).with_name("test_rebuttal_v2_input_audit.py")
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)
_fixture = _FIXTURE_MODULE._fixture


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _write(path: Path, value, *, rows: bool = False) -> Path:
    text = (
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
        if rows
        else json.dumps(value, ensure_ascii=False) + "\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pipeline(root: Path, *, pair_changes=None, label="unique_preserved"):
    source = root / "source"
    manifest, lock = _fixture(source, pair_changes=pair_changes)
    audit_dir = root / "audit"
    run_input_audit(manifest, lock, audit_dir)
    audit = load_input_audit(audit_dir / "input_audit_records.jsonl")

    stage_a = root / "stage-a"
    annotations.export_stage_a(audit.path, stage_a)
    batch_a = _read(stage_a / "annotation_batch.json")
    ratings_a = [
        {
            "schema_version": annotations.A_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "interpretations": ["Synthetic interpretation."],
            "ambiguity": "unique",
            "rationale": "Synthetic Stage A rationale.",
        }
        for blind in batch_a["selected_blind_ids"]
        for rater in batch_a["required_raters"]
    ]
    labels_a = _write(root / "returned-a.jsonl", ratings_a, rows=True)

    stage_b = root / "stage-b"
    annotations.export_stage_b(audit.path, labels_a, stage_a / "annotation_batch.json", stage_b)
    batch_b = _read(stage_b / "annotation_batch.json")
    ratings_b = [
        {
            "schema_version": annotations.B_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "stage_a_lock_sha256": batch_b["stage_a_lock_ref"]["sha256"],
            "label": label,
            "rationale": "Synthetic Stage B rationale.",
        }
        for blind in batch_b["selected_blind_ids"]
        for rater in batch_b["required_raters"]
    ]
    labels_b = _write(root / "returned-b.jsonl", ratings_b, rows=True)
    imported = root / "imported"
    annotations.import_adjudicated_labels(
        audit.path, labels_b, stage_b / "annotation_batch.json", imported
    )
    return SimpleNamespace(
        audit=audit,
        stage_a=stage_a,
        stage_b=stage_b,
        labels_b=labels_b,
        imported=imported,
        semantic=imported / "semantic_labels.jsonl",
        run=imported / "run.json",
    )


def _update_output_hash(case, name: str) -> None:
    run = _read(case.run)
    run["output_refs"][name]["sha256"] = hashlib.sha256(
        (case.imported / name).read_bytes()
    ).hexdigest()
    _write(case.run, run)


def test_loads_verified_semantics_with_absolute_references(tmp_path):
    case = _pipeline(tmp_path)
    bundle = load_semantic_labels(case.semantic, case.audit)
    assert len(bundle.rows) == 1
    assert bundle.labels_by_pair_id == {bundle.rows[0]["pair_id"]: "unique_preserved"}
    assert bundle.path == case.semantic.resolve()
    assert bundle.run_path == case.run.resolve()
    assert bundle.reference == _ref(case.semantic)
    assert bundle.run_ref == _ref(case.run)


def test_empty_labels_remain_bound_by_annotation_import_run(tmp_path):
    case = _pipeline(
        tmp_path,
        pair_changes={"typo_text": None, "typo_query_span": None},
    )
    bundle = load_semantic_labels(case.semantic, case.audit)
    assert bundle.rows == []
    assert bundle.labels_by_pair_id == {}
    run = _read(case.run)
    run["command"] = "annotation-export-b"
    _write(case.run, run)
    with pytest.raises(IntakeError, match="run command"):
        load_semantic_labels(case.semantic, case.audit)


def test_rehashed_forged_semantic_label_is_reconstructed_and_rejected(tmp_path):
    case = _pipeline(tmp_path)
    rows = _rows(case.semantic)
    rows[0]["label"] = "task_changed"
    _write(case.semantic, rows, rows=True)
    _update_output_hash(case, "semantic_labels.jsonl")
    with pytest.raises(IntakeError, match="differ from verified annotation judgments"):
        load_semantic_labels(case.semantic, case.audit)


def test_mixed_batch_and_returned_labels_are_rejected(tmp_path):
    first = _pipeline(tmp_path / "one")
    second = _pipeline(tmp_path / "two")
    run = _read(first.run)
    run["invocation"]["annotation_batch"] = str(second.stage_b / "annotation_batch.json")
    run["invocation"]["annotation_batch_ref"] = _ref(second.stage_b / "annotation_batch.json")
    _write(first.run, run)
    with pytest.raises(IntakeError, match="SHA-256 mismatch|inventory|binding"):
        load_semantic_labels(first.semantic, first.audit)


def test_boolean_is_not_accepted_as_an_integer_run_count(tmp_path):
    case = _pipeline(tmp_path)
    run = _read(case.run)
    run["invocation"]["completed_count"] = True
    _write(case.run, run)
    with pytest.raises(IntakeError, match="integer.*not a boolean"):
        load_semantic_labels(case.semantic, case.audit)


def test_typed_canonical_comparison_rejects_boolean_derived_count(tmp_path):
    case = _pipeline(tmp_path)
    agreement = _read(case.imported / "annotation_agreement.json")
    agreement["selected_pair_count"] = True
    _write(case.imported / "annotation_agreement.json", agreement)
    _update_output_hash(case, "annotation_agreement.json")
    with pytest.raises(IntakeError, match="agreement differs"):
        load_semantic_labels(case.semantic, case.audit)


def test_missing_sibling_run_is_rejected_even_for_existing_labels(tmp_path):
    case = _pipeline(tmp_path)
    case.run.unlink()
    with pytest.raises(IntakeError, match="cannot read referenced artifact"):
        load_semantic_labels(case.semantic, case.audit)


def test_modified_returned_rating_fails_its_captured_hash(tmp_path):
    case = _pipeline(tmp_path)
    rows = _rows(case.labels_b)
    rows[0]["rationale"] = "Changed after import."
    _write(case.labels_b, rows, rows=True)
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        load_semantic_labels(case.semantic, case.audit)


def test_historical_code_hashes_are_bound_but_not_compared_to_upgraded_code(tmp_path):
    case = _pipeline(tmp_path)
    run = _read(case.run)
    code_ref = next(
        ref for ref in run["code_identity"]["source_refs"] if ref["path"].endswith("annotations.py")
    )
    code_ref["sha256"] = "0" * 64
    next(ref for ref in run["input_refs"] if ref["path"] == code_ref["path"])["sha256"] = "0" * 64
    _write(case.run, run)
    assert load_semantic_labels(case.semantic, case.audit).rows


def test_semantics_cannot_be_loaded_against_a_different_audit(tmp_path):
    case = _pipeline(tmp_path / "one")
    other = _pipeline(tmp_path / "two")
    with pytest.raises(IntakeError, match="different input audit"):
        load_semantic_labels(case.semantic, other.audit)


def test_valid_explicit_unassessable_label_loads(tmp_path):
    case = _pipeline(
        tmp_path,
        pair_changes={"clean_text": None, "clean_query_span": None},
        label="unassessable",
    )
    bundle = load_semantic_labels(case.semantic, case.audit)
    assert list(bundle.labels_by_pair_id.values()) == ["unassessable"]

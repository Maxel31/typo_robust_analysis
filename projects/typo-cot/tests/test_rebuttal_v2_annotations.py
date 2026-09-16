"""Independent human-workflow contracts; these fixtures are not real judgments."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_cot.experiments.rebuttal_v2 import annotations as annotation
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError, canonical_sha256


PROTOCOL = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"
TIME = "2026-09-16T12:00:00Z"


def _write(path, value, *, rows=False):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in value) if rows else json.dumps(value),
        encoding="utf-8",
    )
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def audit(tmp_path, monkeypatch):
    from typo_cot.experiments.rebuttal_v2 import input_audit

    manifest_path = _write(tmp_path / "pair_manifest.jsonl", [])
    audit_path = _write(tmp_path / "input_audit_records.jsonl", [])
    metadata_path = _write(tmp_path / "input_audit.meta.json", {})
    clean = "How many apples? A. 4 B. 5"
    typo = "How many applse? A. 4 B. 5"
    prefix = "PRIVATE FEW-SHOT ANSWER 42\nUse arithmetic.\nCount fruit.\n"
    pairs, records = [], []
    for index in range(2):
        pair = {
            "pair_id": f"private-pair-{index}",
            "model_id": "PRIVATE MODEL",
            "gold": "PRIVATE RAW GOLD",
            "canonical_gold": {"numerator": "4", "denominator": "1"},
            "cohort_membership": [{"cohort_id": "R3", "membership": "member"}],
            "source_ref": {"artifact": _ref(manifest_path), "record_id": f"private-source-{index}"},
            "prompt_regions": {
                side: [
                    {
                        "kind": "instructions",
                        "start": prefix.index("Use arithmetic."),
                        "end": prefix.index("Use arithmetic.") + len("Use arithmetic."),
                    },
                    {
                        "kind": "context",
                        "start": prefix.index("Count fruit."),
                        "end": prefix.index("Count fruit.") + len("Count fruit."),
                    },
                    {
                        "kind": "option-content",
                        "start": len(prefix) + len(typo) - 1,
                        "end": len(prefix) + len(typo),
                        "option_label": "B",
                    },
                ]
                for side in ("clean", "typo")
            },
        }
        for name, value in (
            ("clean_text", clean),
            ("typo_text", typo),
            ("clean_prompt", prefix + clean),
            ("typo_prompt", prefix + typo),
        ):
            pair[name] = value
            pair[f"{name}_ref"] = None
        pairs.append(pair)
        records.append(
            {
                "pair_id": pair["pair_id"],
                "alignment": {"eligibility": "valid" if index == 0 else "unknown"},
                "risk_flags": {"flags": {"option_content": False, "option_label": False}},
            }
        )
    manifest = SimpleNamespace(
        pairs=pairs,
        path=manifest_path,
        manifest_ref=_ref(manifest_path),
        protocol_bytes=PROTOCOL.read_bytes(),
        protocol=_read(PROTOCOL),
        source_audit={
            "cohorts": [
                {
                    "cohort_id": "R3",
                    "setting": "R3",
                    "missing_source_pair_keys": ["missing-original"],
                    "missing_count": 1,
                }
            ]
        },
    )
    bundle = SimpleNamespace(
        path=audit_path,
        metadata_path=metadata_path,
        manifest=manifest,
        rows=records,
        snapshots={
            path: _ref(path)["sha256"] for path in (manifest_path, audit_path, metadata_path)
        },
    )
    monkeypatch.setattr(input_audit, "load_input_audit", lambda path: bundle)
    return bundle


def _a_export(audit, tmp_path, **kwargs):
    out = tmp_path / "stage-a"
    annotation.export_stage_a(audit.path, out, **kwargs)
    return out


def _a_labels(stage_a, tmp_path):
    batch = _read(stage_a / "annotation_batch.json")
    ratings = [
        {
            "schema_version": annotation.A_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "interpretations": [f"own interpretation by {rater}"],
            "ambiguity": "unique",
            "rationale": f"PRIVATE A RATIONALE {rater}",
        }
        for blind in batch["selected_blind_ids"]
        for rater in batch["required_raters"]
    ]
    return _write(tmp_path / "returned-a.jsonl", ratings, rows=True)


def _b_export(audit, tmp_path, **kwargs):
    stage_a = _a_export(audit, tmp_path, **kwargs)
    labels = _a_labels(stage_a, tmp_path)
    stage_b = tmp_path / "stage-b"
    annotation.export_stage_b(audit.path, labels, stage_a / "annotation_batch.json", stage_b)
    return stage_a, labels, stage_b


def _b_labels(stage_b, tmp_path, *, disagreement=False, adjudicate=False):
    batch = _read(stage_b / "annotation_batch.json")
    ratings = [
        {
            "schema_version": annotation.B_SCHEMA,
            "blind_id": blind,
            "rater_id": rater,
            "timestamp": TIME,
            "stage_a_lock_sha256": batch["stage_a_lock_ref"]["sha256"],
            "label": "task_changed"
            if disagreement and rater == batch["required_raters"][-1]
            else "unique_preserved",
            "rationale": "Synthetic independent semantic judgment.",
        }
        for blind in batch["selected_blind_ids"]
        for rater in batch["required_raters"]
    ]
    if adjudicate:
        ratings.extend(
            {
                "schema_version": annotation.ADJ_SCHEMA,
                "blind_id": blind,
                "adjudicator_id": "third-reader",
                "timestamp": TIME,
                "stage_a_lock_sha256": batch["stage_a_lock_ref"]["sha256"],
                "final_label": "ambiguous",
                "disagreement_reason": "Independent interpretations differ.",
            }
            for blind in batch["selected_blind_ids"]
        )
    return _write(tmp_path / "returned-b.jsonl", ratings, rows=True)


def test_stage_a_allowlist_and_outcome_independent_selection(audit, tmp_path):
    stage_a = _a_export(audit, tmp_path)
    rows = _rows(stage_a / "annotation_stage_a.jsonl")
    assert len(rows) == 2  # Unknown alignment is not an annotation exclusion.
    assert set(rows[0]) == {
        "schema_version",
        "blind_id",
        "stage",
        "typo_query",
        "typo_context",
        "task_instructions",
        "presented_options",
    }
    assert rows[0]["typo_context"] == ["Count fruit."]
    assert rows[0]["task_instructions"] == ["Use arithmetic."]
    assert rows[0]["presented_options"] == [{"kind": "option-content", "label": "B", "text": "5"}]
    rendered = json.dumps(rows)
    for forbidden in (
        "private-pair",
        "PRIVATE",
        "apples",
        "model",
        "gold",
        "source_ref",
        str(tmp_path),
    ):
        assert forbidden not in rendered
    assert len({row["blind_id"] for row in rows}) == 2
    assert (stage_a.stat().st_mode & 0o777) == 0o700
    coverage = _read(stage_a / "annotation_coverage.json")
    assert coverage["missing_original_source_pair_keys"] == {"R3": ["missing-original"]}


def test_blind_ids_are_random_between_identical_exports(audit, tmp_path):
    one = _a_export(audit, tmp_path)
    two = tmp_path / "second-export"
    annotation.export_stage_a(audit.path, two)
    assert {row["blind_id"] for row in _rows(one / "annotation_stage_a.jsonl")}.isdisjoint(
        row["blind_id"] for row in _rows(two / "annotation_stage_a.jsonl")
    )


def test_missing_and_unknown_membership_are_coverage_not_fake_questions(audit, tmp_path):
    audit.manifest.pairs[0]["typo_text"] = None
    audit.manifest.pairs[1]["cohort_membership"][0]["membership"] = "unknown"
    out = _a_export(audit, tmp_path)
    assert _rows(out / "annotation_stage_a.jsonl") == []
    coverage = _read(out / "annotation_coverage.json")["pair_coverage"]
    assert all(row["semantic_status"] == "unknown" for row in coverage)
    assert "typo_query_unavailable" in coverage[0]["reason_codes"]
    assert "original_cohort_membership_not_verified" in coverage[1]["reason_codes"]


def test_unknown_regions_do_not_export_fewshot_prompt(audit, tmp_path):
    for pair in audit.manifest.pairs:
        pair["prompt_regions"] = None
    rows = _rows(_a_export(audit, tmp_path) / "annotation_stage_a.jsonl")
    assert all(row["typo_context"] == [] and row["presented_options"] == [] for row in rows)
    assert "FEW-SHOT" not in json.dumps(rows)


@pytest.mark.parametrize("missing_sides", [("clean",), ("typo",), ("clean", "typo")])
def test_missing_full_prompts_preserve_queries_and_complete_annotation_pipeline(
    audit, tmp_path, missing_sides
):
    for pair in audit.manifest.pairs:
        for side in missing_sides:
            pair[f"{side}_prompt"] = None
    stage_a, _, stage_b = _b_export(audit, tmp_path)
    forms = _rows(stage_a / "annotation_stage_a.jsonl")
    assert len(forms) == 2
    assert all(form["typo_query"] == "How many applse? A. 4 B. 5" for form in forms)
    if "typo" in missing_sides:
        assert all(
            form["typo_context"] == form["task_instructions"] == form["presented_options"] == []
            for form in forms
        )
    forms_b = _rows(stage_b / "annotation_stage_b.jsonl")
    assert all(form["clean_query"] == "How many apples? A. 4 B. 5" for form in forms_b)
    if "clean" in missing_sides:
        assert all(
            form["clean_context"]
            == form["clean_task_instructions"]
            == form["clean_presented_options"]
            == []
            for form in forms_b
        )
    coverage = _read(stage_a / "annotation_coverage.json")["pair_coverage"]
    for side in missing_sides:
        assert all(
            f"{side}_prompt_unavailable_for_declared_regions" in row["reason_codes"]
            for row in coverage
        )
    labels = _b_labels(stage_b, tmp_path)
    out = tmp_path / "imported"
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    assert len(_rows(out / "semantic_labels.jsonl")) == 2


def test_reference_only_prompts_still_supply_verified_ancillary_regions(audit, tmp_path):
    for side in ("clean", "typo"):
        path = tmp_path / f"private-{side}-prompt.txt"
        path.write_text(audit.manifest.pairs[0][f"{side}_prompt"], encoding="utf-8")
        audit.manifest.pairs[0][f"{side}_prompt"] = None
        audit.manifest.pairs[0][f"{side}_prompt_ref"] = _ref(path)
    stage_a, _, stage_b = _b_export(audit, tmp_path)
    assert all(
        row["typo_context"] == ["Count fruit."]
        for row in _rows(stage_a / "annotation_stage_a.jsonl")
    )
    assert all(
        row["clean_context"] == ["Count fruit."]
        for row in _rows(stage_b / "annotation_stage_b.jsonl")
    )
    assert all(
        not any("prompt_unavailable" in reason for reason in row["reason_codes"])
        for row in _read(stage_a / "annotation_coverage.json")["pair_coverage"]
    )


def test_reference_only_clean_query_allows_assessable_labels(audit, tmp_path):
    text = tmp_path / "private-clean-query.txt"
    text.write_text("Reference-only clean question", encoding="utf-8")
    for pair in audit.manifest.pairs:
        pair["clean_text"] = None
        pair["clean_text_ref"] = _ref(text)
    stage_a, _, stage_b = _b_export(audit, tmp_path)
    forms = _rows(stage_b / "annotation_stage_b.jsonl")
    assert all(
        row["clean_query"] == "Reference-only clean question"
        and row["allowed_labels"] == sorted(annotation.LABELS)
        for row in forms
    )
    assert all(
        "clean_query_unavailable" not in row["reason_codes"]
        for row in _read(stage_a / "annotation_coverage.json")["pair_coverage"]
    )
    labels = _b_labels(stage_b, tmp_path)
    out = tmp_path / "imported"
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    assert all(row["label"] == "unique_preserved" for row in _rows(out / "semantic_labels.jsonl"))


@pytest.mark.parametrize("clean_query", [None, " \t\n"])
@pytest.mark.parametrize("label", ["unique_preserved", "ambiguous", "task_changed"])
def test_unavailable_clean_query_preserves_stage_a_but_rejects_assessable_b_labels(
    audit, tmp_path, clean_query, label
):
    for pair in audit.manifest.pairs:
        pair["clean_text"] = clean_query
    stage_a, _, stage_b = _b_export(audit, tmp_path)
    assert len(_rows(stage_a / "annotation_stage_a.jsonl")) == 2
    forms_b = _rows(stage_b / "annotation_stage_b.jsonl")
    assert all(
        row["clean_query"] == clean_query and row["allowed_labels"] == ["unassessable"]
        for row in forms_b
    )
    assert all(
        "clean_query_unavailable" in row["reason_codes"]
        for row in _read(stage_a / "annotation_coverage.json")["pair_coverage"]
    )
    labels = _b_labels(stage_b, tmp_path)
    rows = _rows(labels)
    for row in rows:
        row["label"] = "unassessable"
    rows[0]["label"] = label
    _write(labels, rows, rows=True)
    with pytest.raises(
        IntakeError, match="clean query unavailable; explicit unassessable rating required"
    ):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", tmp_path / "imported"
        )
    assert not (tmp_path / "imported").exists()


@pytest.mark.parametrize("clean_query", [None, " \t\n"])
def test_unavailable_clean_query_requires_actual_unassessable_ratings_not_missing_rows(
    audit, tmp_path, clean_query
):
    for pair in audit.manifest.pairs:
        pair["clean_text"] = clean_query
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    rows = _rows(labels)
    for row in rows:
        row["label"] = "unassessable"
    _write(labels, rows[:-1], rows=True)
    out = tmp_path / "imported"
    with pytest.raises(IntakeError, match="missing Stage B ratings"):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", out
        )
    assert not out.exists()
    _write(labels, rows, rows=True)
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    semantic = _rows(out / "semantic_labels.jsonl")
    assert len(semantic) == 2
    assert all(row["label"] == "unassessable" for row in semantic)
    flow = {
        row["set"]: row
        for row in csv.DictReader((out / "cohort_flow.csv").read_text().splitlines())
    }
    assert flow["C_semantic"]["n"] == "0"


def test_reference_only_query_is_resolved_without_exposing_path(audit, tmp_path):
    text = tmp_path / "private-query.txt"
    text.write_text("Reference-only typo queestion", encoding="utf-8")
    audit.manifest.pairs[0]["typo_text"] = None
    audit.manifest.pairs[0]["typo_text_ref"] = _ref(text)
    rows = _rows(_a_export(audit, tmp_path) / "annotation_stage_a.jsonl")
    assert any(row["typo_query"] == "Reference-only typo queestion" for row in rows)
    assert str(text) not in json.dumps(rows)


@pytest.mark.parametrize(
    "raters,reason",
    [
        ([], None),
        (["a", "a"], None),
        ([" "], "exception"),
        (["a"], None),
        (["a"], " "),
        (["a", "b"], "inapplicable"),
    ],
)
def test_invalid_rater_contract_fails(audit, tmp_path, raters, reason):
    with pytest.raises(IntakeError):
        _a_export(audit, tmp_path, raters=raters, single_rater_reason=reason)
    assert not (tmp_path / "stage-a").exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows.pop(), "missing Stage A"),
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate Stage A"),
        (lambda rows: rows[0].update(rater_id="unknown"), "unknown or duplicate"),
        (lambda rows: rows[0].update(blind_id=" "), "whitespace"),
        (lambda rows: rows[0].update(timestamp="yesterday"), "timestamp"),
        (lambda rows: rows[0].update(interpretations=[]), "interpretation"),
        (lambda rows: rows[0].update(rationale=""), "rationale"),
        (lambda rows: rows[0].update(gold="4"), "unknown fields"),
    ],
)
def test_whole_batch_a_lock_rejects_incomplete_or_invalid_forms(audit, tmp_path, mutation, match):
    stage_a = _a_export(audit, tmp_path)
    labels = _a_labels(stage_a, tmp_path)
    rows = _rows(labels)
    mutation(rows)
    _write(labels, rows, rows=True)
    with pytest.raises(IntakeError, match=match):
        annotation.export_stage_b(
            audit.path, labels, stage_a / "annotation_batch.json", tmp_path / "stage-b"
        )
    assert not (tmp_path / "stage-b").exists()


def test_explicit_unassessable_stage_a_counts_as_completed(audit, tmp_path):
    stage_a = _a_export(audit, tmp_path)
    labels = _a_labels(stage_a, tmp_path)
    rows = _rows(labels)
    rows[0].update(interpretations=[], ambiguity="unassessable")
    _write(labels, rows, rows=True)
    annotation.export_stage_b(
        audit.path, labels, stage_a / "annotation_batch.json", tmp_path / "stage-b"
    )
    assert len(_read(tmp_path / "stage-b/stage_a_lock.json")["completed_keys"]) == 4


def test_stage_b_binds_own_a_but_discloses_no_other_rater_answers(audit, tmp_path):
    _, a_labels, stage_b = _b_export(audit, tmp_path)
    rows = _rows(stage_b / "annotation_stage_b.jsonl")
    a = {(row["blind_id"], row["rater_id"]): row for row in _rows(a_labels)}
    for row in rows:
        assert row["stage_a_response_sha256"] == canonical_sha256(
            a[(row["blind_id"], row["rater_id"])]
        )
        assert row["clean_query"] == "How many apples? A. 4 B. 5"
        assert row["gold"] == {"numerator": "4", "denominator": "1"}
        assert "stage_a_response" not in row
    assert "PRIVATE" not in json.dumps(rows)
    assert "interpretations" not in json.dumps(rows)
    assert "source_ref" not in json.dumps(rows)


def test_consensus_import_retains_both_stages_and_unknown_cohort_sets(audit, tmp_path):
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    out = tmp_path / "imported"
    result = annotation.import_adjudicated_labels(
        audit.path, labels, stage_b / "annotation_batch.json", out
    )
    assert result["model_execution_performed"] is False
    assert result["human_judgments_generated"] is False
    assert result["experiment_status"] == "partial"
    assert result["annotation_stage_status"] == "complete"
    semantic = _rows(out / "semantic_labels.jsonl")
    assert len(semantic) == 2
    assert all(
        row["label"] == "unique_preserved" and row["adjudication_status"] == "unanimous"
        for row in semantic
    )
    assert all(
        len(row["stage_a_judgments"]) == len(row["stage_b_judgments"]) == 2 for row in semantic
    )
    agreement = _read(out / "annotation_agreement.json")
    assert agreement["pre_adjudication_agreement_rate"] == 1
    assert agreement["missing_judgments"] == []
    flow = {
        row["set"]: row
        for row in csv.DictReader((out / "cohort_flow.csv").read_text().splitlines())
    }
    assert flow["C_archive_recovered"]["n"] == "2"
    assert flow["C_semantic"]["n"] == "2"
    assert flow["C_aligned"]["n"] == "1"
    assert flow["C_correct"]["n"] == "NA"
    assert flow["C_fresh_failure"]["status"] == "not_evaluated"


def test_disagreement_requires_explicit_adjudication_and_retains_reason(audit, tmp_path):
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path, disagreement=True)
    out = tmp_path / "imported"
    with pytest.raises(IntakeError, match="unresolved annotation disagreement"):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", out
        )
    assert not out.exists()
    labels = _b_labels(stage_b, tmp_path, disagreement=True, adjudicate=True)
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    agreement = _read(out / "annotation_agreement.json")
    assert agreement["pre_adjudication_agreement_rate"] == 0
    assert agreement["adjudicated_count"] == 2
    assert agreement["disagreement_count"] == 2
    assert all(
        row["label"] == "ambiguous" and row["adjudication"]["adjudicator_id"] == "third-reader"
        for row in _rows(out / "semantic_labels.jsonl")
    )


def test_single_rater_mode_never_claims_agreement(audit, tmp_path):
    _, _, stage_b = _b_export(
        audit, tmp_path, raters=["only-reader"], single_rater_reason="Second reader unavailable."
    )
    labels = _b_labels(stage_b, tmp_path)
    out = tmp_path / "imported"
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    agreement = _read(out / "annotation_agreement.json")
    assert agreement["rater_count"] == 1
    assert agreement["pre_adjudication_agreement_rate"] is None
    assert agreement["pre_adjudication_agreement_n"] is None
    assert agreement["agreement_status"] == "not_applicable_single_rater"
    assert all(row["rating_mode"] == "single" for row in _rows(out / "semantic_labels.jsonl"))


@pytest.mark.parametrize("single_rater", [False, True])
@pytest.mark.parametrize("final_label", ["ambiguous", "unique_preserved"])
def test_adjudication_cannot_override_or_recount_non_disagreements(
    audit, tmp_path, single_rater, final_label
):
    options = (
        {"raters": ["only-reader"], "single_rater_reason": "Second reader unavailable."}
        if single_rater
        else {}
    )
    _, _, stage_b = _b_export(audit, tmp_path, **options)
    labels = _b_labels(stage_b, tmp_path, adjudicate=True)
    rows = _rows(labels)
    for row in rows:
        if row["schema_version"] == annotation.ADJ_SCHEMA:
            row["final_label"] = final_label
    _write(labels, rows, rows=True)
    out = tmp_path / "imported"
    with pytest.raises(IntakeError, match="adjudication requires independent rater disagreement"):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", out
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows.pop(), "missing Stage B"),
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate Stage B"),
        (lambda rows: rows[0].update(label="correct"), "semantic label"),
        (lambda rows: rows[0].update(stage_a_lock_sha256="0" * 64), "lock hash mismatch"),
        (lambda rows: rows[0].update(arm="correct"), "unknown fields"),
        (lambda rows: rows[0].update(rationale=" "), "whitespace"),
    ],
)
def test_import_rejects_incomplete_or_unbound_b_ratings(audit, tmp_path, mutation, match):
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    rows = _rows(labels)
    mutation(rows)
    _write(labels, rows, rows=True)
    with pytest.raises(IntakeError, match=match):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", tmp_path / "imported"
        )


@pytest.mark.parametrize(
    "artifact", ["annotation_mapping.jsonl", "annotation_stage_a.jsonl", "annotation_coverage.json"]
)
def test_stage_a_referenced_artifact_tampering_is_rejected(audit, tmp_path, artifact):
    stage_a = _a_export(audit, tmp_path)
    labels = _a_labels(stage_a, tmp_path)
    path = stage_a / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        annotation.export_stage_b(
            audit.path, labels, stage_a / "annotation_batch.json", tmp_path / "stage-b"
        )


def test_rehashed_stage_a_export_cannot_add_clean_text(audit, tmp_path):
    stage_a = _a_export(audit, tmp_path)
    labels = _a_labels(stage_a, tmp_path)
    path = stage_a / "annotation_stage_a.jsonl"
    rows = _rows(path)
    rows[0]["clean_query"] = "Forbidden clean text"
    _write(path, rows, rows=True)
    batch = _read(stage_a / "annotation_batch.json")
    batch["export_ref"] = _ref(path)
    _write(stage_a / "annotation_batch.json", batch)
    with pytest.raises(IntakeError, match="safe allowlisted"):
        annotation.export_stage_b(
            audit.path, labels, stage_a / "annotation_batch.json", tmp_path / "stage-b"
        )


def test_stage_a_cannot_change_after_stage_b_exposure(audit, tmp_path):
    _, a_labels, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    rows = _rows(a_labels)
    rows[0]["rationale"] = "Revised after seeing clean material."
    _write(a_labels, rows, rows=True)
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", tmp_path / "imported"
        )


def test_publication_revalidates_returned_judgments_after_capture(audit, tmp_path, monkeypatch):
    from typo_cot.experiments.rebuttal_v2 import artifact_io

    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    publish = artifact_io.publish_artifacts

    def mutate_then_publish(output, files, snapshots):
        labels.write_bytes(labels.read_bytes() + b"\n")
        return publish(output, files, snapshots)

    monkeypatch.setattr(artifact_io, "publish_artifacts", mutate_then_publish)
    with pytest.raises(IntakeError, match="changed after capture"):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", tmp_path / "imported"
        )
    assert not (tmp_path / "imported").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("adjudicator_id", " "),
        ("timestamp", "unknown"),
        ("disagreement_reason", ""),
        ("final_label", "correct"),
    ],
)
def test_adjudication_requires_identity_timestamp_reason_and_legal_label(
    audit, tmp_path, field, value
):
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path, disagreement=True, adjudicate=True)
    rows = _rows(labels)
    rows[-1][field] = value
    _write(labels, rows, rows=True)
    with pytest.raises(IntakeError):
        annotation.import_adjudicated_labels(
            audit.path, labels, stage_b / "annotation_batch.json", tmp_path / "imported"
        )


def test_empty_batch_does_not_claim_perfect_agreement(audit, tmp_path):
    for pair in audit.manifest.pairs:
        pair["typo_text"] = None
    _, _, stage_b = _b_export(audit, tmp_path)
    labels = _b_labels(stage_b, tmp_path)
    out = tmp_path / "imported"
    annotation.import_adjudicated_labels(audit.path, labels, stage_b / "annotation_batch.json", out)
    assert _rows(out / "semantic_labels.jsonl") == []
    agreement = _read(out / "annotation_agreement.json")
    assert agreement["pre_adjudication_agreement_rate"] is None
    assert agreement["agreement_status"] == "empty_batch"


def test_publication_never_overwrites_existing_artifacts(audit, tmp_path):
    stage_a = _a_export(audit, tmp_path)
    before = {path: path.read_bytes() for path in stage_a.iterdir()}
    with pytest.raises(IntakeError, match="absent or empty"):
        annotation.export_stage_a(audit.path, stage_a)
    assert before == {path: path.read_bytes() for path in stage_a.iterdir()}


@pytest.mark.parametrize("field,value", [("start", -1), ("end", 9999), ("option_label", "PRIVATE")])
def test_unusable_prompt_region_is_rejected_not_exported(audit, tmp_path, field, value):
    audit.manifest.pairs[0]["prompt_regions"]["typo"][0][field] = value
    with pytest.raises(IntakeError):
        _a_export(audit, tmp_path)


@pytest.mark.parametrize(
    "layout",
    [{"legacy": "PRIVATE"}, {"clean": [], "typo": [{"kind": "fewshot", "start": 0, "end": 25}]}],
)
def test_unknown_region_adapter_preserves_query_and_records_ancillary_unknown(
    audit, tmp_path, layout
):
    for pair in audit.manifest.pairs:
        pair["prompt_regions"] = layout
    out = _a_export(audit, tmp_path)
    assert len(_rows(out / "annotation_stage_a.jsonl")) == 2
    assert "PRIVATE" not in (out / "annotation_stage_a.jsonl").read_text()
    assert all(
        "ancillary_prompt_regions_unknown" in row["reason_codes"]
        for row in _read(out / "annotation_coverage.json")["pair_coverage"]
    )

"""R1 fixtures use tiny invented archives, never the historical success counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2.answer_audit import (
    PRIMARY,
    PROXY,
    compare_parsers,
    export_blind_disagreements,
    run_answer_audit,
)
from typo_cot.experiments.rebuttal_v2.source import run_intake


PROTOCOL = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pair(key="p1", **changes) -> dict:
    return {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": key,
        "source_pair_key": key,
        "task": "gsm8k",
        "model_id": "test-model",
        "dataset_id": "gsm8k",
        "split": "test",
        "original_problem_id": key,
        "cohort_ids": ["original"],
        "clean_text": "How many apples?",
        "typo_text": "How many aplpes?",
        "gold": "4",
        "canonical_gold": {"numerator": "4", "denominator": "1"},
        "gold_canonicalization_version": "fixture/v1",
        **changes,
    }


def _generation(arm="correct", key="p1", raw_text="Answer: 4.", **changes) -> dict:
    return {
        "schema_version": "rebuttal-source-generation/v2",
        "source_record_id": f"{key}-{arm}",
        "source_generation_key": f"{key}-{arm}",
        "source_pair_key": key,
        "arm": arm,
        "window": None if arm in {"clean", "typo"} else [0, 6],
        "raw_text": raw_text,
        **changes,
    }


def _manifest(root: Path, *, pairs=None, generations=None, expected=None) -> Path:
    pair_file = _jsonl(root / "pairs.jsonl", [_pair()] if pairs is None else pairs)
    gen_file = _jsonl(
        root / "generations.jsonl", [_generation()] if generations is None else generations
    )
    ids = _json(
        root / "ids.json",
        {
            "schema_version": "rebuttal-cohort-ids/v2",
            "cohort_id": "original",
            "source_namespace": "r1-fixture",
            "source_pair_keys": ["p1"] if expected is None else expected,
        },
    )
    sources = []
    for path, role, adapter in (
        (pair_file, "pairs", "pair"),
        (gen_file, "generations", "generation"),
    ):
        sources.append(
            {
                "source_id": role,
                "source_kind": "submitted-archive",
                "role": role,
                "path": str(path),
                "expected_sha256": _ref(path)["sha256"],
                "format": "jsonl",
                "adapter_version": f"rebuttal-source-{adapter}/v2",
                "source_commit": None,
                "model_revision": None,
                "tokenizer_revision": None,
                "prompt_template_revision": None,
                "generation_config": None,
                "availability": "available",
                "reason_codes": [],
            }
        )
    index = _json(
        root / "archive_index.json",
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": "r1-fixture",
            "sources": sources,
            "cohorts": [
                {
                    "cohort_id": "original",
                    "setting": "fixture-only",
                    "historical_n_reference": 1,
                    "expected_ids_ref": _ref(ids),
                    "id_namespace": "r1-fixture",
                    "membership_provenance_ref": None,
                }
            ],
            "identity_aliases_ref": None,
            "notes": "Synthetic parser audit fixture.",
        },
    )
    run_intake(index, PROTOCOL, root / "intake")
    return root / "intake/pair_manifest.jsonl"


def _compare(text="Answer: 4", **changes):
    return compare_parsers(
        {
            "generation_id": "g1",
            "raw_text": text,
            "task": "gsm8k",
            "valid_labels": None,
            "termination": "unknown",
            "gold": "4",
            "canonical_gold": {"numerator": "4", "denominator": "1"},
            **changes,
        }
    )


@pytest.mark.parametrize("text", ["The answer is definitely B.", "The answer is incorrect"])
def test_proxy_preserves_partial_letter_bug_without_primary_regression(text) -> None:
    result = _compare(
        text, task="mmlu-pro", valid_labels=list("ABCDEFGHIJ"), gold="D", canonical_gold="D"
    )["parsers"]
    assert result[PRIMARY]["extraction_status"] == "unextractable"
    assert result[PROXY]["raw_value"] == ("D" if "definitely" in text else "I")
    assert result[PROXY]["extraction_status"] == "extracted"


def test_proxy_legacy_and_auxiliary_exact_equality_are_distinct() -> None:
    result = _compare(
        "Answer: 0.5", gold="1/2", canonical_gold={"numerator": "1", "denominator": "2"}
    )["parsers"]
    assert result[PRIMARY]["gold_match"] is True
    assert result[PROXY]["gold_match"] is False
    assert result[PROXY]["canonical_gold_match"] is True


def test_missing_raw_gold_does_not_invent_legacy_gold_from_canonical() -> None:
    result = _compare(gold=None)["parsers"]
    assert result[PRIMARY]["gold_match"] is True
    assert result[PROXY]["score_status"] == "not_scored"
    assert result[PROXY]["gold_match"] is None
    assert result[PROXY]["canonical_gold_match"] is True


def test_missing_canonical_gold_keeps_primary_extraction_not_false_score() -> None:
    result = _compare(canonical_gold=None)["parsers"]
    assert result[PRIMARY]["extraction_status"] == "extracted"
    assert result[PRIMARY]["score_status"] == "not_scored"
    assert result[PRIMARY]["gold_match"] is None
    assert result[PROXY]["gold_match"] is True


def test_proxy_can_run_without_item_labels_but_primary_is_unavailable() -> None:
    result = _compare(
        "Answer: B", task="mmlu-pro", valid_labels=None, gold="B", canonical_gold="B"
    )["parsers"]
    assert result[PRIMARY]["available"] is False
    assert result[PROXY]["available"] is True
    assert result[PROXY]["raw_value"] == "B"
    assert result[PROXY]["gold_match"] is True
    assert result[PROXY]["canonical_gold_match"] is None


def test_same_raw_all_arms_and_declared_parser_code_hashes(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        generations=[
            _generation(arm) for arm in ("clean", "typo", "correct", "offset", "cross", "self")
        ],
    )
    output = tmp_path / "audit"
    run = run_answer_audit(manifest, output)
    assert run["run_status"] == "complete"
    assert run["model_execution_performed"] is False
    rows = _rows(output / "answer_audit_records.jsonl")
    assert len(rows) == 12
    for arm in ("clean", "typo", "correct", "offset", "cross", "self"):
        compared = [row for row in rows if row["arm"] == arm]
        assert {row["parser_id"] for row in compared} == {PRIMARY, PROXY}
        assert len({row["raw_text_sha256"] for row in compared}) == 1
        assert len({json.dumps(row["generation_ref"], sort_keys=True) for row in compared}) == 1
    proxy = next(row for row in rows if row["parser_id"] == PROXY)
    assert set(proxy["parser_code_files"]) == {"extractor.py", "fallback.py"}
    assert proxy["score_rule"] == "fallback.answers_equal/public-v1"
    assert "archive_parser" not in {row["parser_id"] for row in rows}


def test_missing_expected_output_has_coverage_but_no_audit_row(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        pairs=[
            _pair(
                expected_generations=[
                    {"arm": "clean", "window": None},
                    {"arm": "typo", "window": None},
                ]
            )
        ],
        generations=[_generation("clean")],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    rows = _rows(output / "answer_audit_records.jsonl")
    coverage = _rows(output / "scoring_coverage.jsonl")
    assert len(rows) == 2
    assert len(coverage) == 4
    missing = [row for row in coverage if row["source_slot"]["arm"] == "typo"]
    assert {row["scoring_status"] for row in missing} == {"output_missing"}
    assert all(row["score_ref"] is None and row["generation_ref"] is None for row in missing)
    assert all("extraction_status" not in row for row in missing)
    (baseline,) = _read(output / "baseline_transition_table.json")
    assert baseline["primary_state"] == baseline["proxy_state"] == "unknown"


def test_present_generation_record_without_raw_text_is_not_scored(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, generations=[_generation(raw_text=None)])
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    assert _rows(output / "answer_audit_records.jsonl") == []
    coverage = _rows(output / "scoring_coverage.jsonl")
    assert len(coverage) == 2
    assert all(
        row["generation_id"] is not None
        and row["score_ref"] is None
        and row["generation_ref"] is None
        for row in coverage
    )


def test_baseline_transition_does_not_replace_fixed_cohort(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        generations=[
            _generation("clean", raw_text="Answer: 4"),
            _generation("typo", raw_text="Working = 4"),
            _generation("correct", raw_text="Answer: 4"),
        ],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    (baseline,) = _read(output / "baseline_transition_table.json")
    assert baseline["primary_state"] == "eligible"
    assert baseline["proxy_state"] == "ineligible"
    assert baseline["cohort_membership"][0]["membership"] == "member"
    primary = _read(output / "primary_parser_archived_baseline_failure_ids.json")
    proxy = _read(output / "public_v1_proxy_archived_baseline_failure_ids.json")
    assert primary["pair_ids"] == [baseline["pair_id"]]
    assert proxy["pair_ids"] == []
    summary = _read(output / "parser_audit_summary.json")
    counts = [row for row in summary["arm_parser_counts"] if row["arm"] == "correct"]
    assert {row["parser_id"]: row["recovery_count"] for row in counts} == {PRIMARY: 1, PROXY: 0}
    assert {row["known_slot_count"] for row in counts} == {1}


def test_extra_and_unknown_membership_not_pooled_with_original(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        pairs=[_pair(), _pair("extra")],
        generations=[_generation(), _generation(key="extra")],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    counts = _read(output / "parser_audit_summary.json")["arm_parser_counts"]
    assert {(row["membership"], row["known_slot_count"]) for row in counts} == {
        ("member", 1),
        ("extra", 1),
    }


def test_empty_archive_retains_cohort_missing_evidence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, pairs=[], generations=[], expected=["missing-original"])
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    summary = _read(output / "parser_audit_summary.json")
    assert summary["audit_record_count"] == 0
    assert summary["source_cohort_coverage"][0]["missing_source_pair_keys"] == ["missing-original"]
    assert summary["parser_independent_semantic_claim"] == "inconclusive"


def test_blind_allowlist_sampling_caps_and_order_independent_of_gold() -> None:
    records = []
    for i in range(50):
        raw = "Working = 4" if i < 40 else "Answer: 4"
        comparison = _compare(raw)
        records.append(
            {
                **comparison,
                "generation_id": f"opaque-{i}",
                "raw_text": raw,
                "pair_id": f"p{i}",
                "gold": "4",
                "arm": "correct",
                "model_id": "hidden",
            }
        )
    batch = export_blind_disagreements(records)
    assert len(batch["annotation_records"]) == 40  # 32 disagreement, 8 agreement controls.
    assert all(
        set(row) == {"schema_version", "blind_id", "raw_text"}
        for row in batch["annotation_records"]
    )
    for record in records:
        record["gold"] = "changed"
        for result in record["parsers"].values():
            result["gold_match"] = not result["gold_match"]
    repeated = export_blind_disagreements(list(reversed(records)))
    assert repeated["annotation_records"] == batch["annotation_records"]
    assert repeated["sampling_policy"] == batch["sampling_policy"]
    assert sum(row["unaudited_count"] for row in batch["sampling_policy"]["strata"]) == 10


def test_termination_and_exact_unicode_text_are_unchanged(tmp_path: Path) -> None:
    raw = "Café\u0085quoted\u2028line\r\nAnswer: 4.\n"
    manifest = _manifest(
        tmp_path,
        generations=[
            _generation(
                raw_text=raw, stopping_metadata={"eos_observed": True, "length_cap_reached": True}
            )
        ],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    assert {row["termination"] for row in _rows(output / "answer_audit_records.jsonl")} == {"eos"}
    assert _rows(output / "parser_disagreement_blind.jsonl")[0]["raw_text"] == raw


def test_nonempty_output_is_never_replaced(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    saved = (output / "run.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        run_answer_audit(manifest, output)
    assert (output / "run.json").read_bytes() == saved


def test_reference_hashes_verify_from_each_output_location(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)

    def check(value, containing):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = containing.parent / value["path"]
                assert _ref(path)["sha256"] == value["sha256"]
            for item in value.values():
                check(item, containing)
        elif isinstance(value, list):
            for item in value:
                check(item, containing)

    for path in output.glob("*.json"):
        check(_read(path), path)
    for path in output.glob("*.jsonl"):
        for row in _rows(path):
            check(row, path)


def test_unsupported_extra_task_retains_parser_unavailable_coverage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, pairs=[_pair(task="unsupported-task", canonical_gold=None)])
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    assert _rows(output / "answer_audit_records.jsonl") == []
    coverage = _rows(output / "scoring_coverage.jsonl")
    assert len(coverage) == 2
    assert {row["scoring_status"] for row in coverage} == {"parser_unavailable"}
    assert all(row["score_ref"] is None and row["generation_ref"] is not None for row in coverage)
    summary = _read(output / "parser_audit_summary.json")
    assert all(
        row["present_output_count"] == 1 and row["termination_counts"] == {"unknown": 1}
        for row in summary["arm_parser_counts"]
    )


def test_generation_outside_known_grid_is_retained_as_separate_extra(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, pairs=[_pair(expected_generations=[])])
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    rows = _rows(output / "answer_audit_records.jsonl")
    coverage = _rows(output / "scoring_coverage.jsonl")
    assert len(rows) == len(coverage) == 2
    assert {row["source_slot_membership"] for row in rows + coverage} == {"extra"}
    assert all("unexpected_generation_slot" in row["reason_codes"] for row in coverage)
    summary = _read(output / "parser_audit_summary.json")
    assert {row["source_slot_membership"] for row in summary["arm_parser_counts"]} == {"extra"}


def test_failure_ids_do_not_admit_extra_pairs_to_original_cohort(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        pairs=[_pair(), _pair("extra")],
        generations=[_generation("clean", key=key, raw_text="Answer: 4") for key in ("p1", "extra")]
        + [_generation("typo", key=key, raw_text="Answer: 5") for key in ("p1", "extra")],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    artifact = _read(output / "primary_parser_archived_baseline_failure_ids.json")
    assert len(artifact["pair_ids"]) == len(artifact["extra_pair_ids"]) == 1
    assert not set(artifact["pair_ids"]) & set(artifact["extra_pair_ids"])
    assert artifact["unverified_pair_ids"] == []


def test_publication_rejects_input_changed_after_scoring(tmp_path: Path, monkeypatch) -> None:
    from typo_cot.experiments.rebuttal_v2 import answer_audit

    manifest = _manifest(tmp_path)
    original = answer_audit._reports

    def mutate(*args):
        source = tmp_path / "generations.jsonl"
        source.write_bytes(source.read_bytes() + b"\n")
        return original(*args)

    monkeypatch.setattr(answer_audit, "_reports", mutate)
    with pytest.raises(ValueError, match="changed|hash|SHA"):
        run_answer_audit(manifest, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
    assert not list(tmp_path.glob(".audit.answer-audit-*"))


def test_run_binds_all_executed_code_and_manifest_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "audit"
    result = run_answer_audit(manifest, output)
    names = {Path(ref["path"]).name for ref in result["code_identity"]["source_refs"]}
    assert {
        "answer_audit.py",
        "artifacts.py",
        "identity.py",
        "schemas.py",
        "cli.py",
        "audit_extractor.py",
        "extractor.py",
        "fallback.py",
    } <= names
    assert result["manifest_metadata_ref"] == _ref(manifest.with_name("pair_manifest.meta.json"))
    assert result["experiment_status"] == "complete"
    assert result["experiment_id"] == "R1"


@pytest.mark.parametrize("labels", [["AA"], ["a"]])
def test_pr01_legal_but_unsupported_labels_are_parser_unavailable(tmp_path: Path, labels) -> None:
    manifest = _manifest(
        tmp_path,
        pairs=[_pair(task="mmlu-pro", valid_labels=labels, gold="B", canonical_gold=None)],
        generations=[_generation(raw_text="Answer: B")],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    rows = _rows(output / "answer_audit_records.jsonl")
    assert len(rows) == 1
    assert rows[0]["parser_id"] == PROXY
    assert rows[0]["gold_match"] is True
    assert rows[0]["canonical_gold_match"] is None
    coverage = {row["parser_id"]: row for row in _rows(output / "scoring_coverage.jsonl")}
    assert coverage[PRIMARY]["scoring_status"] == "parser_unavailable"
    assert "unsupported_item_label_set" in coverage[PRIMARY]["reason_codes"]


def test_rate_denominators_expose_missing_outputs_without_imputation(tmp_path: Path) -> None:
    expected_slots = [{"arm": "correct", "window": [0, 6]}]
    manifest = _manifest(
        tmp_path,
        pairs=[_pair(key, expected_generations=expected_slots) for key in ("p1", "p2")],
        generations=[_generation()],
        expected=["p1", "p2"],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    summary = _read(output / "parser_audit_summary.json")
    primary = next(row for row in summary["arm_parser_counts"] if row["parser_id"] == PRIMARY)
    rates = primary["rates"]
    assert rates["answer_success"]["value"] == 1.0
    assert rates["answer_success"]["denominator"] == 1
    assert rates["scoring_coverage"]["value"] == 0.5
    assert rates["scoring_coverage"]["denominator"] == 2
    assert rates["full_original_cohort_answer_success"]["value"] is None
    assert rates["full_original_cohort_answer_success"]["denominator"] == 2
    assert rates["recovery"]["value"] is None
    (difference,) = summary["paired_parser_differences"]
    assert difference["paired_scored_count"] == 1
    assert difference["unknown_count"] == 1


def test_full_original_rate_only_exists_for_complete_expected_scored_grid(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path, pairs=[_pair(expected_generations=[{"arm": "correct", "window": [0, 6]}])]
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    for row in _read(output / "parser_audit_summary.json")["arm_parser_counts"]:
        rate = row["rates"]["full_original_cohort_answer_success"]
        assert rate["complete"] is True
        assert rate["denominator"] == 1
        assert rate["value"] == 1.0


def test_paired_parser_difference_preserves_declared_scoring_rules(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, generations=[_generation(raw_text="Working = 4")])
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    (difference,) = _read(output / "parser_audit_summary.json")["paired_parser_differences"]
    assert difference["primary_minus_proxy_success_count"] == -1
    assert difference["primary_minus_proxy_success_rate"] == -1.0
    assert difference["paired_scored_count"] == 1
    assert difference["primary_score_rule"] != difference["proxy_score_rule"]


def test_extra_pairs_do_not_suppress_complete_original_cohort_rate(tmp_path: Path) -> None:
    slots = [{"arm": "correct", "window": [0, 6]}]
    manifest = _manifest(
        tmp_path,
        pairs=[_pair(key, expected_generations=slots) for key in ("p1", "extra")],
        generations=[_generation(key=key) for key in ("p1", "extra")],
    )
    output = tmp_path / "audit"
    run_answer_audit(manifest, output)
    summary = _read(output / "parser_audit_summary.json")
    assert summary["source_cohort_coverage"][0]["coverage_status"] == "partial"
    for row in summary["arm_parser_counts"]:
        rate = row["rates"]["full_original_cohort_answer_success"]
        assert rate["value"] == (1.0 if row["membership"] == "member" else None)

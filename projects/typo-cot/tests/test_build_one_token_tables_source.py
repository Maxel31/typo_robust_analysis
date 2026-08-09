"""Fail-closed input contracts for the one-token CPU artifact builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.build_one_token_tables import (
    OneTokenTablesInputError,
    discover_and_validate_runs,
)
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.one_token_prefix_replacement.metrics import (
    aggregate_one_token_events,
    classify_one_token_events,
)
from typo_cot.experiments.one_token_prefix_replacement.planning import (
    OneTokenInputPlan,
    OneTokenProfile,
    build_arm_specs,
    choose_adjacent_position,
    choose_distant_positions,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    LEGACY_SETTING_IDS,
    PROTOCOL,
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _generation(
    *,
    correct: bool,
    answer: str,
    benchmark: str = "gsm8k",
    gold_answer: str = "2",
) -> dict[str, object]:
    benchmark_name = {"gsm8k": "gsm8k", "mmlu": "mmlu", "arc": "arc_challenge"}[benchmark]
    text = f"The answer is {answer}."
    extraction = extract_with_fallback(
        text,
        benchmark=benchmark_name,
        correct_answer=gold_answer,
        allow_positional=True,
    )
    assert extraction.is_correct is correct
    return {
        "token_ids": [99],
        "text": text,
        "value": extraction.value,
        "is_extracted": extraction.is_extracted,
        "is_correct": extraction.is_correct,
        "method": extraction.method,
        "primary_method": extraction.primary_method,
        "stop_reason": "eos_token",
        "stop_token_id": 99,
    }


def _record(
    *,
    model: str,
    benchmark: str,
    cohort: str,
    adjacent: bool,
    sample_id: str = "sample-001",
) -> dict[str, object]:
    gold_answer = "2" if benchmark == "gsm8k" else "A"
    wrong_answer = "3" if benchmark == "gsm8k" else "B"
    clean_cot_ids = (30, 1, 3, 4, 5, 2, 6, 7)
    plan = OneTokenInputPlan(
        clean_prompt_ids=(100, 101),
        edited_prompt_ids=(200, 201),
        clean_full_ids=(100, 101, *clean_cot_ids),
        edited_full_ids=(200, 201, *clean_cot_ids),
        clean_cot_ids=clean_cot_ids,
    )
    profile = OneTokenProfile(
        clean_to_edited_kl=(0.0, 8.0, 2.0, 1.0, 0.5, 0.25, 0.0, 0.0),
        clean_token_rank_under_clean=(1, 1, 1, 1, 1, 1, 1, 1),
        clean_token_rank_under_edited=(1, 8, 2, 2, 2, 2, 1, 1),
        edited_top1_ids=(30, 10, 31, 32, 33, 20, 6, 7),
        edited_top1_is_admissible=(True, True, True, True, True, True, True, True),
    )
    selection = choose_distant_positions(profile)
    assert selection.selected_position == 1
    assert selection.distant_position == 5
    adjacent_position = None
    if adjacent:
        setting_id = LEGACY_SETTING_IDS[(model, benchmark)]
        adjacent_position = choose_adjacent_position(
            profile.clean_to_edited_kl,
            selected_position=selection.selected_position,
            tie_key=f"{setting_id}|lxt4|{sample_id}",
        )
        assert adjacent_position == 2
    specs = build_arm_specs(
        plan,
        profile,
        selection,
        adjacent_position=adjacent_position,
    )
    correctness = {
        "selected_keep": True,
        "selected_from_selected": False,
        "selected_from_distant": True,
        "distant_keep": True,
        "distant_from_selected": False,
        "distant_from_distant": True,
        "adjacent_keep": True,
        "adjacent_from_selected": False,
    }
    arms = [
        {
            **spec.to_dict(),
            "input_ids": list(plan.generation_input_ids(spec.position, spec.forced_token_id)),
            "generation": _generation(
                correct=correctness[spec.name],
                answer=gold_answer if correctness[spec.name] else wrong_answer,
                benchmark=benchmark,
                gold_answer=gold_answer,
            ),
        }
        for spec in specs
    ]
    by_name = {str(arm["name"]): arm for arm in arms}
    return {
        "schema_version": "one-token-prefix-replacement-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "model": model,
        "benchmark": benchmark,
        "cohort": cohort,
        "targeting": "attribution-4",
        "sample_id": sample_id,
        "gold_answer": gold_answer,
        "source": {
            "cohort_sha256": "c" * 64,
            "source_record_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
        },
        "input_plan": plan.to_dict(),
        "input_plan_sha256": _canonical_sha256(plan.to_dict()),
        "positions": {
            "selected": 1,
            "distant": 5,
            "adjacent": adjacent_position,
        },
        "profile": profile.to_dict(),
        "profile_sha256": _canonical_sha256(profile.to_dict()),
        "selection": selection.to_dict(),
        "adjacent_unavailable_reason": None,
        "arms": arms,
        "events": classify_one_token_events(
            by_name,
            selected_before_control=True,
            adjacent_requested=adjacent,
        ),
    }


def _frozen_plan(record: dict[str, object], *, target_count: int) -> dict[str, object]:
    targeting = str(record["targeting"])
    sample_ids = [
        str(record["sample_id"]),
        *[f"planned-{index:03d}" for index in range(1, target_count)],
    ]
    rows = [
        {
            "cohort": record["cohort"],
            "targeting": targeting,
            "sample_id": sample_id,
            "source_record_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
            "candidate_eligible": True,
            "boundary_valid": True,
            "cot_token_count": 8,
            "exclusion_reason": None,
            "input_plan_sha256": record["input_plan_sha256"],
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    selected = [[targeting, sample_id] for sample_id in sample_ids]
    return {
        "algorithm": "shared-clean-prefix-cohort-selection-before-limit/v1",
        "cases": rows,
        "cases_sha256": _canonical_sha256(rows),
        "source_case_count": target_count,
        "eligible_case_count": target_count,
        "selected_full": selected,
        "selected_full_sha256": _canonical_sha256(selected),
        "selected_for_execution": selected,
        "selected_for_execution_sha256": _canonical_sha256(selected),
    }


def _position_exclusion_evidence(
    record: dict[str, object],
    plan_case: dict[str, object],
    *,
    adjacent: bool,
) -> dict[str, object]:
    profile = OneTokenProfile(
        clean_to_edited_kl=(0.0, 8.0, 2.0, 1.0, 0.5, 0.25, 0.0, 0.0),
        clean_token_rank_under_clean=(1, 1, 1, 1, 1, 1, 1, 1),
        clean_token_rank_under_edited=(1, 1, 1, 1, 1, 1, 1, 1),
        edited_top1_ids=(30, 1, 3, 4, 5, 2, 6, 7),
        edited_top1_is_admissible=(True, True, True, True, True, True, True, True),
    ).to_dict()
    input_plan = record["input_plan"]
    return {
        "schema_version": "one-token-prefix-replacement-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "source": {
            "cohort": plan_case["cohort"],
            "targeting": plan_case["targeting"],
            "sample_id": plan_case["sample_id"],
            "source_record_sha256": plan_case["source_record_sha256"],
        },
        "input_plan": input_plan,
        "input_plan_sha256": _canonical_sha256(input_plan),
        "profile": profile,
        "profile_sha256": _canonical_sha256(profile),
        "selection": None,
        "positions": {"selected": None, "distant": None, "adjacent": None},
        "adjacent_requested": adjacent,
        "adjacent_unavailable_reason": (
            "case-excluded-before-adjacent-position-selection" if adjacent else None
        ),
        "position_exclusion_reason": "no-position-with-clean-token-below-edited-top1",
        "arm_specs": [],
        "arm_specs_sha256": _canonical_sha256([]),
        "arms": [],
    }


def _comparability(*, target_count: int) -> dict[str, object]:
    return {
        "status": "fresh-paper-protocol-run",
        "requirements": {
            "paper_setting": True,
            "paper_source_protocol": True,
            "paper_source_cohort_identity": False,
            "expected_selected_target_count": target_count,
            "selected_target_count_matches": True,
            "selected_exact_boundary_valid_count": target_count,
            "selected_exact_boundaries_all_valid": True,
            "prespecified_position_controls": True,
            "unlimited": True,
        },
        "limitations": [],
        "primary_in_fourteen_extension_aggregate": False,
        "fresh_public_preparation_is_historical_identity_proof": False,
        "single_setting_runner_computes_cross_setting_interval": False,
    }


def _write_run(
    run_dir: Path,
    *,
    model: str = "google/gemma-3-1b-it",
    benchmark: str = "gsm8k",
    cohort: str = "extension",
    adjacent: bool = True,
    code_sha: str = "a" * 64,
) -> Path:
    run_dir.mkdir(parents=True)
    record = _record(
        model=model,
        benchmark=benchmark,
        cohort=cohort,
        adjacent=adjacent,
    )
    records = [record]
    controls = ["distant", "adjacent"] if adjacent else ["distant"]
    target_count = 172 if cohort == "primary" else 150
    comparability = _comparability(target_count=target_count)
    plan = _frozen_plan(record, target_count=target_count)
    statuses = []
    for index, raw_row in enumerate(plan["cases"]):  # type: ignore[union-attr]
        row = dict(raw_row)
        evidence = (
            None if index == 0 else _position_exclusion_evidence(record, row, adjacent=adjacent)
        )
        positions = record["positions"] if index == 0 else evidence["positions"]
        statuses.append(
            {
                "schema_version": "one-token-prefix-replacement-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": model,
                "benchmark": benchmark,
                "cohort": cohort,
                "targeting": row["targeting"],
                "sample_id": row["sample_id"],
                "source_record_sha256": row["source_record_sha256"],
                "candidate_eligible": True,
                "boundary_valid": True,
                "selected_full": True,
                "selected_for_execution": True,
                "positions": positions,
                "adjacent_requested": adjacent,
                "adjacent_available": index == 0 and adjacent,
                "adjacent_unavailable_reason": (
                    None
                    if index == 0
                    else "case-excluded-before-adjacent-position-selection"
                    if adjacent
                    else None
                ),
                "execution_status": "completed" if index == 0 else "position-unavailable",
                "exclusion_reason": (
                    None if index == 0 else "no-position-with-clean-token-below-edited-top1"
                ),
                "position_exclusion_evidence": evidence,
                "record_sha256": _canonical_sha256(record) if index == 0 else None,
            }
        )
    summary = {
        "schema_version": "one-token-prefix-replacement-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "setting": {
            "model": model,
            "benchmark": benchmark,
            "cohort": cohort,
            "position_controls": controls,
        },
        "protocol": PROTOCOL,
        "counts": {
            "source_pairs": target_count,
            "candidate_eligible": target_count,
            "selected_full": target_count,
            "selected_for_execution": target_count,
            "executed": 1,
            "records": 1,
            "arms": len(record["arms"]),  # type: ignore[arg-type]
            "execution_status": {
                "completed": 1,
                "position-unavailable": target_count - 1,
            },
        },
        "metrics": aggregate_one_token_events(
            [record["events"]],  # type: ignore[list-item]
            adjacent_requested=adjacent,
        ),
        "comparability": comparability,
    }
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    summary_path = run_dir / "one_token_summary.json"
    _write_jsonl(records_path, records)
    _write_jsonl(statuses_path, statuses)
    _write_json(summary_path, summary)
    outputs = {
        path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
        for path in (records_path, statuses_path, summary_path)
    }
    manifest = {
        "schema_version": "one-token-prefix-replacement-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "status": "completed",
        "arguments": {
            "model": model,
            "benchmark": benchmark,
            "cohort": cohort,
            "position_controls": controls,
            "limit": None,
        },
        "protocol": PROTOCOL,
        "protocol_sha256": _canonical_sha256(PROTOCOL),
        "comparability": comparability,
        "plan": plan,
        "runtime": {
            "effective_eos_token_ids": [99],
            "implementation_code_identity": {
                "algorithm": "one-token-executable-code-bundle-sha256/v1",
                "python_file_count": 12,
                "sha256": code_sha,
            },
        },
        "checkpoints": {},
        "counts": {
            "source_pairs": target_count,
            "selected_full": target_count,
            "selected_for_execution": target_count,
            "records": 1,
        },
        "failures": [],
        "outputs": outputs,
    }
    _write_json(run_dir / "run.json", manifest)
    return run_dir


def _rehash_output(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = run_dir / name
    manifest["outputs"][name] = {
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }
    _write_json(manifest_path, manifest)


def test_recursive_discovery_uses_verified_manifest_identity_not_directory_name(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path / "arbitrary" / "nested" / "name")

    runs = discover_and_validate_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].run_dir == run_dir.resolve()
    assert runs[0].setting_id == "gemma1b_gsm8k"
    assert runs[0].position_controls == ("distant", "adjacent")
    assert len(runs[0].records) == 1


def test_empty_runs_root_is_rejected_instead_of_publishing_an_empty_grid(
    tmp_path: Path,
) -> None:
    with pytest.raises(OneTokenTablesInputError, match="no.*run.json"):
        discover_and_validate_runs(tmp_path)


def test_output_checksum_tampering_is_rejected_before_aggregation(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    with (run_dir / "one_token_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(OneTokenTablesInputError, match="SHA-256|byte count"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_event_tampering_is_rejected_by_arm_reconstruction(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["events"]["table10"]["selected_correct_to_incorrect"] = False
    _write_jsonl(records_path, [record])
    _rehash_output(run_dir, records_path.name)

    with pytest.raises(OneTokenTablesInputError, match="events.*reconstruct"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_record_source_identity_must_match_the_frozen_plan(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    records = _read_jsonl(records_path)
    records[0]["source"]["source_record_sha256"] = "f" * 64
    _write_jsonl(records_path, records)
    statuses = _read_jsonl(statuses_path)
    statuses[0]["record_sha256"] = _canonical_sha256(records[0])
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, records_path.name)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="record source.*frozen plan"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_generation_semantics_are_revalidated_from_text_and_gold(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    selected = next(arm for arm in record["arms"] if arm["name"] == "selected_from_selected")
    selected["generation"]["text"] = "The answer is 2."
    _write_jsonl(records_path, [record])
    statuses = _read_jsonl(statuses_path)
    statuses[0]["record_sha256"] = _canonical_sha256(record)
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, records_path.name)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="extraction|correctness"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_position_selection_is_reconstructed_from_the_profile(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["positions"]["selected"] = 2
    _write_jsonl(records_path, [record])
    statuses = _read_jsonl(statuses_path)
    statuses[0]["record_sha256"] = _canonical_sha256(record)
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, records_path.name)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="position|selection"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_adjacent_availability_reason_is_reconstructed(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    records = _read_jsonl(records_path)
    records[0]["adjacent_unavailable_reason"] = "fabricated-reason"
    _write_jsonl(records_path, records)
    statuses = _read_jsonl(statuses_path)
    statuses[0]["adjacent_unavailable_reason"] = "fabricated-reason"
    statuses[0]["record_sha256"] = _canonical_sha256(records[0])
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, records_path.name)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="adjacent availability"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_arm_plan_is_reconstructed_from_input_and_profile(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path / "run")
    records_path = run_dir / "one_token_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    for arm in record["arms"]:
        if arm["name"] in {"selected_from_selected", "distant_from_selected"}:
            arm["forced_token_id"] = 99
            arm["input_ids"][-1] = 99
    _write_jsonl(records_path, [record])
    statuses = _read_jsonl(statuses_path)
    statuses[0]["record_sha256"] = _canonical_sha256(record)
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, records_path.name)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="arm.*reconstruct"):
        discover_and_validate_runs(tmp_path)


def test_rehashed_position_exclusion_requires_reconstructable_evidence(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path / "run")
    statuses_path = run_dir / "pair_status_records.jsonl"
    statuses = _read_jsonl(statuses_path)
    assert statuses[1]["execution_status"] == "position-unavailable"
    statuses[1]["position_exclusion_evidence"] = None
    _write_jsonl(statuses_path, statuses)
    _rehash_output(run_dir, statuses_path.name)

    with pytest.raises(OneTokenTablesInputError, match="exclusion.*evidence"):
        discover_and_validate_runs(tmp_path)


def test_duplicate_setting_manifests_are_rejected_deterministically(tmp_path: Path) -> None:
    _write_run(tmp_path / "first")
    _write_run(tmp_path / "second")

    with pytest.raises(OneTokenTablesInputError, match="duplicate.*gemma1b_gsm8k"):
        discover_and_validate_runs(tmp_path)


def test_mixed_producer_code_identities_are_rejected(tmp_path: Path) -> None:
    _write_run(tmp_path / "first", code_sha="a" * 64)
    _write_run(
        tmp_path / "second",
        model="google/gemma-3-1b-it",
        benchmark="mmlu",
        adjacent=False,
        code_sha="b" * 64,
    )

    with pytest.raises(OneTokenTablesInputError, match="code identity"):
        discover_and_validate_runs(tmp_path)


def test_partial_smoke_manifest_is_not_promoted_to_a_paper_input(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparability"]["status"] = "partial-smoke-run"
    _write_json(manifest_path, manifest)

    with pytest.raises(OneTokenTablesInputError, match="fresh-paper-protocol-run"):
        discover_and_validate_runs(tmp_path)


def test_fully_rehashed_short_extension_plan_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run")
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest["plan"]
    plan["cases"].pop()
    plan["selected_full"].pop()
    plan["selected_for_execution"].pop()
    plan["cases_sha256"] = _canonical_sha256(plan["cases"])
    plan["selected_full_sha256"] = _canonical_sha256(plan["selected_full"])
    plan["selected_for_execution_sha256"] = _canonical_sha256(plan["selected_for_execution"])
    plan["source_case_count"] = 149
    plan["eligible_case_count"] = 149
    manifest["counts"].update(
        source_pairs=149,
        selected_full=149,
        selected_for_execution=149,
    )
    requirements = manifest["comparability"]["requirements"]
    requirements["expected_selected_target_count"] = 149
    requirements["selected_exact_boundary_valid_count"] = 149
    summary_path = run_dir / "one_token_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["comparability"] = manifest["comparability"]
    summary["counts"].update(
        source_pairs=149,
        candidate_eligible=149,
        selected_full=149,
        selected_for_execution=149,
    )
    _write_json(summary_path, summary)
    manifest["outputs"][summary_path.name] = {
        "sha256": _file_sha256(summary_path),
        "bytes": summary_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(OneTokenTablesInputError, match="plan.*150|150.*plan"):
        discover_and_validate_runs(tmp_path)


def test_missing_prespecified_adjacent_control_is_rejected(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", adjacent=False)

    with pytest.raises(OneTokenTablesInputError, match="adjacent"):
        discover_and_validate_runs(tmp_path)


def test_symlinks_below_runs_root_are_rejected(tmp_path: Path) -> None:
    _write_run(tmp_path / "run")
    (tmp_path / "alias").symlink_to(tmp_path / "run", target_is_directory=True)

    with pytest.raises(OneTokenTablesInputError, match="symlink"):
        discover_and_validate_runs(tmp_path)


def test_unrelated_run_manifest_is_not_silently_ignored(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _write_json(
        foreign / "run.json",
        {"operation": "clean-prefix-scan", "status": "completed"},
    )

    with pytest.raises(OneTokenTablesInputError, match="unexpected operation"):
        discover_and_validate_runs(tmp_path)

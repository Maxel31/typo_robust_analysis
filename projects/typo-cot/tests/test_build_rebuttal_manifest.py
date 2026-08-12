"""End-to-end contracts for the six-setting rebuttal pair manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typo_cot.cli import main
from typo_cot.experiments.build_rebuttal_manifest import runner as manifest_runner
from typo_cot.experiments.build_rebuttal_manifest import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    BuildRebuttalManifestConfig,
    load_rebuttal_pair_manifest,
    run_build_rebuttal_manifest,
)
from typo_cot.experiments.build_rebuttal_manifest.records import normalize_prepared_pair
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.fixed_window_answer_patching import (
    FIXED_WINDOW_ANSWER_PATCHING_PROTOCOL,
)


_FIXTURE_DATASET_RECORDS = 150


def test_evaluation_benchmark_normalizes_mmlu_pro() -> None:
    assert manifest_runner._evaluation_benchmark("mmlu-pro") == "mmlu_pro"
    assert manifest_runner._evaluation_benchmark("mmlu") == "mmlu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = _canonical_json(row)
            handle.write(line + "\n")
            fingerprints[str(row["sample_id"])] = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return fingerprints


def _answer_values(task: str) -> tuple[str, str]:
    return ("2", "3") if task == "gsm8k" else ("A", "B")


def _pair_record(
    *,
    model: str,
    task: str,
    target_rule: str,
    sample_id: str,
    typo_correct: bool,
    aligned: bool = True,
) -> dict[str, object]:
    gold, wrong = _answer_values(task)
    typo_value = gold if typo_correct else wrong
    clean_prompt = f"Question {sample_id}: alpha omega."
    typo_prompt = f"Question {sample_id}: alpah omega." if aligned else clean_prompt
    clean_start = clean_prompt.index("alpha")
    typo_start = typo_prompt.index("alpah") if aligned else clean_start
    clean_continuation = f"The answer is {gold}."
    typo_continuation = f"The answer is {typo_value}."
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": task,
        "targeting": target_rule,
        "seed": 42,
        "num_edits_requested": 4,
        "num_candidates": 8,
        "num_target_attempts": int(aligned),
        "num_aligned_words": int(aligned),
        "gold_answer": gold,
        "subset": "fixture",
        "clean": {
            "question": clean_prompt,
            "choices": None,
            "editable_text": clean_prompt,
            "editable_prompt_span": {"start": 0, "end": len(clean_prompt)},
            "prompt": clean_prompt,
            "prompt_token_count": 10,
            "continuation": clean_continuation,
            "continuation_token_count": 5,
            "termination": "eos",
            "answer": {
                "value": gold,
                "is_extracted": True,
                "is_correct": True,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "edited": {
            "editable_text": typo_prompt,
            "editable_prompt_span": {"start": 0, "end": len(typo_prompt)},
            "prompt": typo_prompt,
            "prompt_token_count": 10,
            "continuation": typo_continuation,
            "continuation_token_count": 5,
            "termination": "eos",
            "answer": {
                "value": typo_value,
                "is_extracted": True,
                "is_correct": typo_correct,
                "method": "fixture",
                "confidence": 1.0,
            },
        },
        "answer_changed": not typo_correct,
        "excluded_attribution_tokens": [],
        "target_attempts": [],
        "aligned_words": [
            {
                "word_index": 2,
                "clean_text": "alpha",
                "edited_text": "alpah",
                "clean_editable_span": {"start": clean_start, "end": clean_start + 5},
                "edited_editable_span": {"start": typo_start, "end": typo_start + 5},
                "clean_prompt_span": {"start": clean_start, "end": clean_start + 5},
                "edited_prompt_span": {"start": typo_start, "end": typo_start + 5},
                "target_ranks": [1],
                "target_token_indices": [2],
                "clean_token_indices": [2],
                "edited_token_indices": [3],
                "clean_final_token": 2,
                "edited_final_token": 3,
            }
        ]
        if aligned
        else [],
    }


def _write_prepared_source(
    *,
    root: Path,
    model: str,
    task: str,
    target_rule: str,
    wrong_pairs: int,
    total_pairs: int,
) -> dict[str, object]:
    model_slug = model.rsplit("/", 1)[-1]
    source_dir = root / model_slug / task / target_rule
    pairs_path = source_dir / "pairs.jsonl"
    records = [
        _pair_record(
            model=model,
            task=task,
            target_rule=target_rule,
            sample_id=f"sample-{index:04d}",
            typo_correct=index >= wrong_pairs,
            aligned=index != total_pairs - 1,
        )
        for index in range(total_pairs)
    ]
    fingerprints = _write_jsonl(pairs_path, records)
    revision = hashlib.sha256(model.encode("utf-8")).hexdigest()
    dataset_sha = hashlib.sha256(f"{task}\0fixture".encode()).hexdigest()
    run = {
        "schema_version": "prepare-edited-pairs-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "prepare-edited-pairs",
        "status": "completed",
        "arguments": {
            "model": model,
            "benchmark": task,
            "targeting": target_rule,
            "num_edits": 4,
            "seed": 42,
            "max_new_tokens": 512,
            "gpu_id": "0",
            "limit": None,
            "output_dir": str(source_dir.resolve()),
        },
        "decoding": {
            "strategy": "greedy",
            "dtype": "bfloat16",
            "padding_side": "left",
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "use_cache": True,
            "return_dict_in_generate": False,
            "output_scores": False,
        },
        "counts": {"discovered": total_pairs, "written": total_pairs, "failed": 0},
        "failures": [],
        "provenance": {
            "model": model,
            "model_revision": revision,
            "dataset_cohort_rule": "paper-model-benchmark-cohort/v1",
            "dataset_sample_count": total_pairs,
            "dataset_records_sha256": dataset_sha,
            "random_seed_algorithm": "sha256-first-64-bits/v1",
            "generation_protocol": "explicit-greedy-generation/v1",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
        },
        "outputs": {
            "pairs": {
                "path": "pairs.jsonl",
                "sha256": _sha256(pairs_path),
                "records": total_pairs,
            }
        },
    }
    run_path = source_dir / "run.json"
    _write_json(run_path, run)
    return {
        "path": pairs_path,
        "pairs_sha256": _sha256(pairs_path),
        "run_sha256": _sha256(run_path),
        "record_count": total_pairs,
        "fingerprints": fingerprints,
        "records": records,
        "revision": revision,
        "dataset_sha256": dataset_sha,
    }


def _write_fixed_run(
    *,
    root: Path,
    model: str,
    task: str,
    denominator: int,
    successes: int,
    selected_anchors: int,
    sources: Mapping[str, Mapping[str, object]],
) -> Path:
    model_slug = model.rsplit("/", 1)[-1]
    run_dir = root / model_slug / task
    wrong_by_target = {
        "attribution-4": selected_anchors // 2,
        "random-4": selected_anchors - selected_anchors // 2,
    }
    identities = [
        (target_rule, f"sample-{index:04d}")
        for target_rule in ("attribution-4", "random-4")
        for index in range(wrong_by_target[target_rule])
    ]
    included_identities = set(identities[:denominator])
    successful = set(identities[:successes])

    def generation(value: str, *, correct: bool, token: int) -> dict[str, object]:
        return {
            "token_ids": [token, 2],
            "text": f"The answer is {value}.",
            "termination": "eos",
            "value": value,
            "is_extracted": True,
            "is_correct": correct,
            "method": "primary:pattern_1",
            "primary_method": "pattern_1",
        }

    gold, wrong = _answer_values(task)

    window_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for target_rule, sample_id in identities:
        source = sources[target_rule]
        fingerprint = source["fingerprints"][sample_id]  # type: ignore[index]
        included = (target_rule, sample_id) in included_identities
        event = (target_rule, sample_id) in successful
        if included:
            clean = generation(gold, correct=True, token=10)
            edited = generation(wrong, correct=False, token=11)
            patched = generation(
                gold if event else wrong,
                correct=event,
                token=12 if event else 13,
            )
            window_rows.append(
                {
                    "schema_version": "fixed-window-answer-patching-window/v1",
                    "paper_sha256": PAPER_SHA256,
                    "model": model,
                    "benchmark": task,
                    "targeting": target_rule,
                    "sample_id": sample_id,
                    "source_record_sha256": fingerprint,
                    "direction": "clean-to-edited",
                    "window": "0:6",
                    "window_start": 0,
                    "window_stop": 6,
                    "num_layers": 12,
                    "aligned_positions": 1,
                    "event": event,
                    "recipient_generation_identical": False,
                    "baseline": {"clean": clean, "edited": edited},
                    "patched_answer": patched,
                }
            )
        status_rows.append(
            {
                "schema_version": "fixed-window-answer-patching-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": model,
                "benchmark": task,
                "targeting": target_rule,
                "sample_id": sample_id,
                "source_record_sha256": fingerprint,
                "aligned_positions": 1,
                "direction_status": {
                    "clean-to-edited": {
                        "included": included,
                        "exclusion_reason": (None if included else "regenerated-edited-not-wrong"),
                    }
                },
            }
        )

    records_path = run_dir / "fixed_window_records.jsonl"
    statuses_path = run_dir / "pair_status_records.jsonl"
    summary_path = run_dir / "setting_summary.json"
    _write_jsonl(records_path, window_rows)
    _write_jsonl(statuses_path, status_rows)
    _write_json(
        summary_path,
        {
            "schema_version": "fixed-window-answer-patching-setting-summary/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "fixed-window-answer-patching",
            "setting": {
                "model": model,
                "benchmark": task,
                "num_layers": 12,
                "layer_windows": ["0:6"],
            },
            "population": {
                "selected_anchors": selected_anchors,
                "direction_denominators": {"clean-to-edited": denominator},
            },
            "directions": {
                "clean-to-edited": {
                    "included_pairs": denominator,
                    "windows": {
                        "0:6": {
                            "successes": successes,
                            "total": denominator,
                        }
                    },
                }
            },
        },
    )
    revision = str(sources["attribution-4"]["revision"])
    run = {
        "schema_version": "fixed-window-answer-patching-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "fixed-window-answer-patching",
        "status": "completed",
        "protocol": json.loads(_canonical_json(FIXED_WINDOW_ANSWER_PATCHING_PROTOCOL)),
        "arguments": {
            "model": model,
            "benchmark": task,
            "pairs": [
                str(sources[target_rule]["path"]) for target_rule in ("attribution-4", "random-4")
            ],
            "layers": ["0:6"],
            "directions": ["clean-to-edited"],
            "output_dir": str(run_dir.resolve()),
            "gpu_id": "0",
            "limit": None,
        },
        "input": {
            "source_model_revision": revision,
            "dataset_records_sha256": sources["attribution-4"]["dataset_sha256"],
            "dataset_sample_count": sources["attribution-4"]["record_count"],
            "targeting_sources": {
                target_rule: {
                    "pairs_path": str(sources[target_rule]["path"]),
                    "pairs_sha256": sources[target_rule]["pairs_sha256"],
                    "source_run_sha256": sources[target_rule]["run_sha256"],
                    "source_record_count": sources[target_rule]["record_count"],
                    "source_schema": "prepare-edited-pairs/v1",
                }
                for target_rule in ("attribution-4", "random-4")
            },
        },
        "runtime": {
            "requested_revision": revision,
            "model_revision": revision,
            "tokenizer_revision": revision,
            "num_decoder_layers": 12,
            "effective_eos_token_ids": [2],
            "effective_eos_token_ids_source": "fixture",
        },
        "counts": {
            "selected_anchors": selected_anchors,
            "failed_pairs": 0,
            "included_direction_pairs": denominator,
            "window_records": denominator,
        },
        "failures": [],
        "outputs": {
            records_path.name: {
                "sha256": _sha256(records_path),
                "records": len(window_rows),
            },
            statuses_path.name: {
                "sha256": _sha256(statuses_path),
                "records": len(status_rows),
            },
            summary_path.name: {"sha256": _sha256(summary_path), "records": 1},
        },
    }
    _write_json(run_dir / "run.json", run)
    return run_dir


def _six_setting_fixture(tmp_path: Path) -> tuple[Path, Path]:
    prepared_root = tmp_path / "prepared"
    fixed_root = tmp_path / "fixed"
    for setting in REBUTTAL_SETTINGS:
        selected_anchors = setting.paper_denominator + 2
        wrong_by_target = {
            "attribution-4": selected_anchors // 2,
            "random-4": selected_anchors - selected_anchors // 2,
        }
        total_per_target = _FIXTURE_DATASET_RECORDS
        sources = {
            target_rule: _write_prepared_source(
                root=prepared_root,
                model=setting.model,
                task=setting.task,
                target_rule=target_rule,
                wrong_pairs=wrong_by_target[target_rule],
                total_pairs=total_per_target,
            )
            for target_rule in ("attribution-4", "random-4")
        }
        _write_fixed_run(
            root=fixed_root,
            model=setting.model,
            task=setting.task,
            denominator=setting.paper_denominator,
            successes=setting.paper_successes,
            selected_anchors=selected_anchors,
            sources=sources,
        )
    return prepared_root, fixed_root


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_rebuttal_setting_contract_matches_the_submitted_paper() -> None:
    assert REBUTTAL_MANIFEST_PROTOCOL.schema_version == "rebuttal-manifest-protocol/v1"
    assert REBUTTAL_MANIFEST_PROTOCOL.target_rules == (
        "attribution-4",
        "random-4",
    )
    assert REBUTTAL_MANIFEST_PROTOCOL.source_seed == 42
    assert REBUTTAL_MANIFEST_PROTOCOL.requested_edits == 4
    assert REBUTTAL_MANIFEST_PROTOCOL.source_max_new_tokens == 512
    assert REBUTTAL_MANIFEST_PROTOCOL.fixed_direction == "clean-to-edited"
    assert REBUTTAL_MANIFEST_PROTOCOL.fixed_window == "0:6"
    assert REBUTTAL_MANIFEST_PROTOCOL.offset_tokens == 2
    assert REBUTTAL_MANIFEST_PROTOCOL.window_split_seed == 42
    assert len(REBUTTAL_SETTINGS) == 6
    assert REBUTTAL_MANIFEST_PROTOCOL.restoration_pairs == 1_241
    assert REBUTTAL_MANIFEST_PROTOCOL.fixed_window_successes == 800
    assert {(setting.task, setting.model) for setting in REBUTTAL_SETTINGS} == {
        (task, model)
        for task in ("gsm8k", "mmlu")
        for model in (
            "google/gemma-3-4b-it",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        )
    }


def test_normalized_pair_preserves_length_cap_and_disables_positional_fallback() -> None:
    pair = _pair_record(
        model="google/gemma-3-4b-it",
        task="gsm8k",
        target_rule="attribution-4",
        sample_id="length-cap",
        typo_correct=False,
    )
    clean = pair["clean"]
    assert isinstance(clean, dict)
    clean["continuation"] = "The computation is unfinished and currently reaches 2 dollars."
    clean["termination"] = "length-cap"
    clean["answer"] = {
        "value": "",
        "is_extracted": False,
        "is_correct": False,
        "method": "unextractable",
        "confidence": 0.0,
    }

    normalized = normalize_prepared_pair(
        pair,
        source_record_sha256="a" * 64,
        prepared_pairs_path="fixture/pairs.jsonl",
        prepared_pairs_sha256="b" * 64,
        prepared_run_sha256="c" * 64,
        model_revision="d" * 64,
    )

    assert normalized["clean_termination"] == "length-cap"
    assert normalized["clean_answer"] == ""
    assert normalized["clean_correct"] is False


def test_build_rebuttal_manifest_is_deterministic_complete_and_hash_bound(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    first_output = tmp_path / "manifest-a"
    second_output = tmp_path / "manifest-b"

    first = run_build_rebuttal_manifest(
        BuildRebuttalManifestConfig(prepared_root, fixed_root, first_output)
    )
    second = run_build_rebuttal_manifest(
        BuildRebuttalManifestConfig(prepared_root, fixed_root, second_output)
    )

    assert first.pair_records == second.pair_records
    assert first.restoration_pairs == second.restoration_pairs == 1_241
    assert first.fixed_window_successes == second.fixed_window_successes == 800
    assert first.harm_pairs == second.harm_pairs > 0
    assert first.fixed_selected_anchors == second.fixed_selected_anchors == 1_253
    assert first.fixed_excluded_anchors == second.fixed_excluded_anchors == 12
    assert first.prepared_wrong_outside_restoration == 12
    for name in ("pair_manifest.jsonl", "cohort_ids.json", "source_audit.json"):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()

    records = load_rebuttal_pair_manifest(first.pair_manifest_path)
    assert len(records) == first.pair_records
    assert len({record["pair_id"] for record in records}) == len(records)
    restoration = [record for record in records if record["cohorts"]["restoration"]]
    harm = [record for record in records if record["cohorts"]["harm"]]
    fixed_excluded = [record for record in records if record["cohorts"]["fixed_excluded_anchor"]]
    prepared_wrong_outside = [
        record for record in records if record["cohorts"]["prepared_typo_wrong_outside_restoration"]
    ]
    assert len(restoration) == 1_241
    assert len(harm) == first.harm_pairs
    assert len(fixed_excluded) == 12
    assert len(prepared_wrong_outside) == 12
    assert all(record["clean_correct"] is True for record in (*restoration, *harm))
    assert all(record["typo_correct"] is False for record in restoration)
    assert all(record["typo_correct"] is True for record in harm)
    assert all(record["controls"]["offset_2"]["valid"] is True for record in restoration)
    unedited = [record for record in records if not record["edits"]]
    assert len(unedited) == 12
    assert all(record["cohorts"]["full_clean_correct"] is True for record in unedited)
    assert all(record["cohorts"]["patch_eligible_clean_correct"] is False for record in unedited)
    assert all(
        record["cohorts"]["alignment_ineligible_clean_correct"] is True for record in unedited
    )
    assert all(record["controls"]["offset_2"]["valid"] is False for record in unedited)
    assert {record["manifest_protocol_sha256"] for record in records} == {
        REBUTTAL_MANIFEST_PROTOCOL.sha256()
    }

    by_pair_id = {record["pair_id"]: record for record in records}
    for record in restoration:
        donor_id = record["controls"]["cross_item"]["donor_pair_id"]
        assert donor_id != record["pair_id"]
        donor = by_pair_id[donor_id]
        assert (
            donor["task"],
            donor["model"],
            donor["target_rule"],
            donor["number_of_aligned_words"],
        ) == (
            record["task"],
            record["model"],
            record["target_rule"],
            record["number_of_aligned_words"],
        )

    cohorts = _load_json(first.cohort_ids_path)
    cohort_lists = cohorts["cohorts"]
    selection = set(cohort_lists["window_selection"])
    evaluation = set(cohort_lists["window_evaluation"])
    restoration_ids = set(cohort_lists["restoration"])
    assert selection.isdisjoint(evaluation)
    assert selection | evaluation == restoration_ids
    assert len(cohort_lists["harm"]) == first.harm_pairs
    assert len(cohort_lists["fixed_selected_anchor"]) == 1_241 + 12
    assert len(cohort_lists["fixed_excluded_anchor"]) == 12
    assert set(cohort_lists["repair_harm_composite"]) == restoration_ids | set(cohort_lists["harm"])
    assert set(cohort_lists["patch_eligible_clean_correct"]) == (
        set(cohort_lists["prepared_typo_wrong"]) | set(cohort_lists["harm"])
    )
    assert set(cohort_lists["full_clean_correct"]) == (
        set(cohort_lists["patch_eligible_clean_correct"])
        | set(cohort_lists["alignment_ineligible_clean_correct"])
    )

    audit = _load_json(first.source_audit_path)
    assert audit["paper_totals"] == {
        "fixed_window_successes": 800,
        "restoration_pairs": 1_241,
    }
    assert len(audit["settings"]) == 6
    assert all(item["offset_valid"] == item["restoration_pairs"] for item in audit["settings"])
    assert all(item["cross_item_valid"] == item["restoration_pairs"] for item in audit["settings"])
    assert all(
        item["fixed_selected_anchors"] == item["restoration_pairs"] + 2
        for item in audit["settings"]
    )
    assert all(item["fixed_excluded_anchors"] == 2 for item in audit["settings"])
    assert all(
        source["model_revision"] == source["tokenizer_revision"]
        for source in audit["sources"]["fixed_window"]
    )
    assert all(
        source["length_cap_generations"] == 0
        for source in audit["sources"]["fixed_window"]
    )

    run = _load_json(first.run_path)
    assert run["status"] == "completed"
    assert run["operation"] == "build-rebuttal-manifest"
    assert run["protocol"]["manifest_contract"] == REBUTTAL_MANIFEST_PROTOCOL.as_dict()
    assert run["protocol"]["fixed_window_reference"] == {
        "event_validation": "explicit-termination-cap-aware-reextraction/v1",
        "producer_protocol": FIXED_WINDOW_ANSWER_PATCHING_PROTOCOL,
    }
    assert run["runtime"]["gpu_required"] is False
    for name, metadata in run["outputs"].items():
        assert metadata["sha256"] == _sha256(first_output / name)

    cli_output = tmp_path / "manifest-cli"
    assert (
        main(
            [
                "build-rebuttal-manifest",
                "--prepared-pairs-root",
                str(prepared_root),
                "--fixed-window-root",
                str(fixed_root),
                "--output-dir",
                str(cli_output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "1,241 restoration pair(s)" in stdout
    assert "800 fixed-window restoration(s)" in stdout

    with pytest.raises(FileExistsError, match="not empty"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(prepared_root, fixed_root, first_output)
        )

    with pytest.raises(ValueError, match="must not overlap"):
        BuildRebuttalManifestConfig(
            prepared_root,
            fixed_root,
            prepared_root / "nested-output",
        )

    tampered_dir = tmp_path / "tampered-artifact-set"
    tampered_dir.mkdir()
    for name in ("pair_manifest.jsonl", "cohort_ids.json", "source_audit.json", "run.json"):
        (tampered_dir / name).write_bytes((first_output / name).read_bytes())
    tampered_manifest = tampered_dir / "pair_manifest.jsonl"
    tampered_rows = [
        json.loads(line)
        for line in first.pair_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    tampered_rows[0]["controls"]["common_valid"] = not tampered_rows[0]["controls"]["common_valid"]
    _write_jsonl(tampered_manifest, tampered_rows)
    tampered_cohorts = _load_json(tampered_dir / "cohort_ids.json")
    tampered_cohorts["pair_manifest_sha256"] = _sha256(tampered_manifest)
    _write_json(tampered_dir / "cohort_ids.json", tampered_cohorts)
    tampered_run = _load_json(tampered_dir / "run.json")
    tampered_run["outputs"]["pair_manifest.jsonl"]["sha256"] = _sha256(tampered_manifest)
    tampered_run["outputs"]["cohort_ids.json"]["sha256"] = _sha256(tampered_dir / "cohort_ids.json")
    _write_json(tampered_dir / "run.json", tampered_run)
    with pytest.raises(ValueError, match="common-valid membership differs"):
        load_rebuttal_pair_manifest(tampered_manifest)

    tampered_audit_dir = tmp_path / "tampered-source-audit"
    tampered_audit_dir.mkdir()
    for name in ("pair_manifest.jsonl", "cohort_ids.json", "source_audit.json", "run.json"):
        (tampered_audit_dir / name).write_bytes((first_output / name).read_bytes())
    tampered_audit = _load_json(tampered_audit_dir / "source_audit.json")
    tampered_audit["settings"][0]["harm_pairs"] += 1
    _write_json(tampered_audit_dir / "source_audit.json", tampered_audit)
    tampered_audit_run = _load_json(tampered_audit_dir / "run.json")
    tampered_audit_run["outputs"]["source_audit.json"]["sha256"] = _sha256(
        tampered_audit_dir / "source_audit.json"
    )
    _write_json(tampered_audit_dir / "run.json", tampered_audit_run)
    with pytest.raises(ValueError, match="source audit setting counts differ"):
        load_rebuttal_pair_manifest(tampered_audit_dir / "pair_manifest.jsonl")

    fixed_run_path = next(fixed_root.rglob("run.json"))
    fixed_run = _load_json(fixed_run_path)
    fixed_run["schema_version"] = "unknown-fixed-window-run/v999"
    _write_json(fixed_run_path, fixed_run)
    with pytest.raises(ValueError, match="unknown fixed-window run schema"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-wrong-fixed-schema",
            )
        )
    fixed_run["schema_version"] = "fixed-window-answer-patching-run/v1"
    _write_json(fixed_run_path, fixed_run)

    source_to_tamper = next(prepared_root.rglob("pairs.jsonl"))
    source_to_tamper.write_text(
        source_to_tamper.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-tampered",
            )
        )


def _prepared_run_path(
    root: Path,
    *,
    model: str,
    task: str,
    target_rule: str,
) -> Path:
    return root / model.rsplit("/", 1)[-1] / task / target_rule / "run.json"


def test_rejects_model_revision_drift_between_tasks(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    model = "google/gemma-3-4b-it"
    for target_rule in ("attribution-4", "random-4"):
        run_path = _prepared_run_path(
            prepared_root,
            model=model,
            task="mmlu",
            target_rule=target_rule,
        )
        run = _load_json(run_path)
        run["provenance"]["model_revision"] = "f" * 64
        _write_json(run_path, run)

    with pytest.raises(ValueError, match="model revisions differ across tasks"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-model-revision-drift",
            )
        )


def test_rejects_dataset_cohort_drift_between_models(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    model = "google/gemma-3-4b-it"
    for target_rule in ("attribution-4", "random-4"):
        run_path = _prepared_run_path(
            prepared_root,
            model=model,
            task="gsm8k",
            target_rule=target_rule,
        )
        run = _load_json(run_path)
        run["provenance"]["dataset_records_sha256"] = "e" * 64
        _write_json(run_path, run)

    with pytest.raises(ValueError, match="dataset SHA-256 differs across models"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-dataset-digest-drift",
            )
        )


def test_rejects_dataset_count_drift_between_models(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    model = "google/gemma-3-4b-it"
    for target_rule in ("attribution-4", "random-4"):
        run_path = _prepared_run_path(
            prepared_root,
            model=model,
            task="gsm8k",
            target_rule=target_rule,
        )
        pairs_path = run_path.parent / "pairs.jsonl"
        rows = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
        ]
        rows.append(
            _pair_record(
                model=model,
                task="gsm8k",
                target_rule=target_rule,
                sample_id=f"sample-{_FIXTURE_DATASET_RECORDS:04d}",
                typo_correct=True,
            )
        )
        _write_jsonl(pairs_path, rows)
        run = _load_json(run_path)
        run["counts"] = {
            "discovered": _FIXTURE_DATASET_RECORDS + 1,
            "written": _FIXTURE_DATASET_RECORDS + 1,
            "failed": 0,
        }
        run["provenance"]["dataset_sample_count"] = _FIXTURE_DATASET_RECORDS + 1
        run["outputs"]["pairs"] = {
            "path": "pairs.jsonl",
            "sha256": _sha256(pairs_path),
            "records": _FIXTURE_DATASET_RECORDS + 1,
        }
        _write_json(run_path, run)

    with pytest.raises(ValueError, match="dataset counts differ across models"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-dataset-count-drift",
            )
        )


def test_rejects_explicit_sample_id_prepared_cohort(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    run_path = next(prepared_root.rglob("run.json"))
    run = _load_json(run_path)
    run["arguments"]["sample_ids"] = "/tmp/rebuttal-cohort.json"
    run["provenance"]["dataset_cohort_rule"] = "explicit-sample-id-cohort/v1"
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="sample-ID cohort"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-explicit-cohort",
            )
        )


def test_rejects_nested_fixed_window_producer_protocol_drift(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    run_path = next(fixed_root.rglob("run.json"))
    run = _load_json(run_path)
    run["protocol"]["generation"]["max_new_tokens"] = 256
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="fixed-window producer protocol differs"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-fixed-protocol-drift",
            )
        )


def _rewrite_fixed_records_metadata(run_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    records_path = run_path.parent / "fixed_window_records.jsonl"
    _write_jsonl(records_path, rows)
    run = _load_json(run_path)
    run["outputs"][records_path.name]["sha256"] = _sha256(records_path)
    _write_json(run_path, run)


def test_rejects_fixed_event_that_disagrees_with_recorded_generations(
    tmp_path: Path,
) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    run_path = next(fixed_root.rglob("run.json"))
    records_path = run_path.parent / "fixed_window_records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["event"] = not rows[0]["event"]
    _rewrite_fixed_records_metadata(run_path, rows)

    with pytest.raises(ValueError, match="event differs from re-extracted generations"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-inconsistent-event",
            )
        )


def test_rejects_positional_only_answer_from_capped_fixed_generation(
    tmp_path: Path,
) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    run_path = next(
        path
        for path in fixed_root.rglob("run.json")
        if _load_json(path)["arguments"]["benchmark"] == "gsm8k"
    )
    records_path = run_path.parent / "fixed_window_records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    patched = rows[0]["patched_answer"]
    patched.update(
        {
            "token_ids": [7] * 512,
            "text": "The unfinished calculation currently reaches 2 dollars.",
            "termination": "length-cap",
            "value": "2",
            "is_extracted": True,
            "is_correct": True,
            "method": "fallback:N5_tail_number",
            "primary_method": "no_match",
        }
    )
    rows[0]["event"] = True
    _rewrite_fixed_records_metadata(run_path, rows)

    with pytest.raises(ValueError, match="extraction differs under cap-aware validation"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-capped-positional-answer",
            )
        )


def test_rejects_fixed_termination_inconsistent_with_token_count(tmp_path: Path) -> None:
    prepared_root, fixed_root = _six_setting_fixture(tmp_path)
    run_path = next(fixed_root.rglob("run.json"))
    records_path = run_path.parent / "fixed_window_records.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["patched_answer"]["termination"] = "length-cap"
    _rewrite_fixed_records_metadata(run_path, rows)

    with pytest.raises(ValueError, match="length-cap must contain exactly 512 tokens"):
        run_build_rebuttal_manifest(
            BuildRebuttalManifestConfig(
                prepared_root,
                fixed_root,
                tmp_path / "manifest-invalid-termination",
            )
        )


def test_build_rebuttal_manifest_catalog_contract_is_implemented() -> None:
    spec = get_experiment("build-rebuttal-manifest")

    assert spec.status == "implemented"
    assert spec.compute == "cpu"
    assert spec.required_arguments == (
        "--prepared-pairs-root",
        "--fixed-window-root",
        "--output-dir",
    )
    assert spec.outputs == (
        "pair_manifest.jsonl",
        "cohort_ids.json",
        "source_audit.json",
        "run.json",
    )

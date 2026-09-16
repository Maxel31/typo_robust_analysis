"""Independent R0 acceptance fixtures: provenance is not an outcome quota."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2.source import run_intake


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs/rebuttal_v2/protocol.json"
NAMESPACE = "submitted-archive-fixture"


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(key: str = "p1", **changes) -> dict:
    return {
        "schema_version": "rebuttal-source-pair/v2",
        "source_record_id": key,
        "source_pair_key": key,
        "historical_pair_id": key,
        "cohort_ids": ["R3"],
        "dataset_id": "gsm8k",
        "split": "test",
        "original_problem_id": key,
        "task": "gsm8k",
        "model_id": "google/gemma-3-4b-it",
        "target_rule": "max-logit",
        "perturbation_id": "edit-1",
        "clean_text": "How many apples?",
        "typo_text": "How many aplpes?",
        **changes,
    }


def _generation(key: str = "p1", **changes) -> dict:
    return {
        "schema_version": "rebuttal-source-generation/v2",
        "source_record_id": f"g-{key}",
        "source_generation_key": f"g-{key}",
        "source_pair_key": key,
        "arm": "correct",
        "window": [0, 6],
        "raw_text": "The answer is 42.",
        **changes,
    }


def _source(path: Path | None, *, role="pairs", adapter=None, key="pairs") -> dict:
    return {
        "source_id": key,
        "source_kind": "submitted-archive",
        "role": role,
        "path": None if path is None else path.name,
        "expected_sha256": None if path is None else _sha(path),
        "format": "jsonl",
        "adapter_version": adapter
        or f"rebuttal-source-{'pair' if role == 'pairs' else 'generation'}/v2",
        "source_commit": None,
        "model_revision": None,
        "tokenizer_revision": None,
        "prompt_template_revision": None,
        "generation_config": None,
        "availability": "unknown" if path is None else "available",
        "reason_codes": ["archive_provenance_unknown"],
    }


def _fixture(
    root: Path,
    *,
    pairs: list[dict] | None = None,
    generations: list[dict] | None = None,
    expected: list[str] | None = None,
    historical_counts: dict | None = None,
) -> Path:
    pair_path = _jsonl(root / "pairs.jsonl", pairs if pairs is not None else [_pair()])
    sources = [_source(pair_path)]
    if generations is not None:
        source = _jsonl(root / "generations.jsonl", generations)
        sources.append(_source(source, role="generations", key="generations"))
    expected_ref = None
    if expected is not None:
        ids = _json(
            root / "ids.json",
            {
                "schema_version": "rebuttal-cohort-ids/v2",
                "cohort_id": "R3",
                "source_namespace": NAMESPACE,
                "source_pair_keys": expected,
            },
        )
        expected_ref = {"path": ids.name, "sha256": _sha(ids)}
    cohort = {
        "cohort_id": "R3",
        "setting": "R3",
        "historical_n_reference": 172,
        "expected_ids_ref": expected_ref,
        "id_namespace": NAMESPACE,
        "membership_provenance_ref": None,
    }
    if historical_counts is not None:
        cohort["historical_counts"] = historical_counts
    return _json(
        root / "archive_index.json",
        {
            "schema_version": "rebuttal-archive-index/v2",
            "created_at": "2026-09-16T00:00:00Z",
            "source_namespace": NAMESPACE,
            "sources": sources,
            "cohorts": [cohort],
            "identity_aliases_ref": None,
            "notes": "Independent test data, not actual paper results.",
        },
    )


def _comparison(output: Path) -> list[dict]:
    with (output / "historical_count_comparison.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_refs(value: object, containing_file: Path) -> None:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            target = containing_file.parent / value["path"]
            assert target.is_file(), target
            assert _sha(target) == value["sha256"]
        for nested in value.values():
            _validate_refs(nested, containing_file)
    elif isinstance(value, list):
        for nested in value:
            _validate_refs(nested, containing_file)


def test_missing_original_pair_is_reported_not_replaced(tmp_path: Path) -> None:
    index = _fixture(tmp_path, expected=["p1", "missing-original"])
    output = tmp_path / "intake"
    result = run_intake(index, PROTOCOL, output)
    assert result["run_status"] == "complete"
    manifest = _rows(output / "pair_manifest.jsonl")
    assert [row["source_pair_key"] for row in manifest] == ["p1"]
    comparison = _comparison(output)[0]
    assert comparison["coverage_status"] == "partial"
    assert int(comparison["missing_count"]) == 1
    assert "missing-original" in (output / "missing_inputs.csv").read_text()
    assert _rows(output / "archive_generation_records.jsonl") == []


def test_unknown_revisions_and_exact_text_survive_intake(tmp_path: Path) -> None:
    text = "  Café\r\n同じ語 e\u0301\n"
    index = _fixture(tmp_path, pairs=[_pair(clean_text=text)], expected=["p1"])
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    (row,) = _rows(output / "pair_manifest.jsonl")
    assert row["clean_text"] == text
    assert row["clean_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert row["model_revision"] is None
    assert row["tokenizer_revision"] is None
    assert row["clean_prompt"] is None
    assert row["reason_codes"]


def test_different_target_rules_share_original_problem_group(tmp_path: Path) -> None:
    pairs = [
        _pair("first", original_problem_id="same-question", target_rule="max-logit"),
        _pair("second", original_problem_id="same-question", target_rule="min-logit"),
    ]
    index = _fixture(tmp_path, pairs=pairs, expected=["first", "second"])
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    rows = _rows(output / "pair_manifest.jsonl")
    assert len({row["pair_id"] for row in rows}) == 2
    groups = {row["original_problem_group_id"] for row in rows}
    payload = {"dataset_id": "gsm8k", "split": "test", "original_problem_id": "same-question"}
    expected = hashlib.sha256(
        (
            "rebuttal-original-problem/v2\n"
            + json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        ).encode()
    ).hexdigest()
    assert groups == {expected}


def test_conflicting_text_cannot_hide_behind_another_record_id(tmp_path: Path) -> None:
    index = _fixture(
        tmp_path, pairs=[_pair(), _pair(source_record_id="another", typo_text="different")]
    )
    with pytest.raises(ValueError, match="(?i)conflict|duplicate"):
        run_intake(index, PROTOCOL, tmp_path / "intake")


@pytest.mark.parametrize(
    "changes",
    [
        {"gold": "different gold"},
        {"model_revision": "another-archived-revision"},
        {"valid_labels": ["A", "B"]},
        {"intended_edits": []},
    ],
)
def test_duplicate_identity_cannot_select_first_conflicting_metadata(
    tmp_path: Path,
    changes: dict,
) -> None:
    index = _fixture(
        tmp_path,
        pairs=[
            _pair(),
            _pair(source_record_id="second-acquisition", **changes),
        ],
    )
    with pytest.raises(ValueError, match="(?i)conflict|duplicate"):
        run_intake(index, PROTOCOL, tmp_path / "intake")


def test_original_group_unknown_does_not_block_cpu_source_audit(tmp_path: Path) -> None:
    index = _fixture(tmp_path, pairs=[_pair(original_problem_id=None)], expected=["p1"])
    output = tmp_path / "intake"
    assert run_intake(index, PROTOCOL, output)["run_status"] == "complete"
    (row,) = _rows(output / "pair_manifest.jsonl")
    assert row["original_problem_group_id"] is None
    assert "original_problem_identity_unknown" in row["reason_codes"]


def test_extra_pair_is_retained_but_not_admitted_to_known_cohort(tmp_path: Path) -> None:
    index = _fixture(tmp_path, pairs=[_pair(), _pair("extra")], expected=["p1"])
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    rows = {row["source_pair_key"]: row for row in _rows(output / "pair_manifest.jsonl")}
    assert set(rows) == {"p1", "extra"}
    assert rows["extra"]["cohort_membership"][0]["membership"] == "extra"
    comparison = _comparison(output)[0]
    assert int(comparison["extra_count"]) == 1
    assert int(comparison["recovered_count"]) == 1


def test_unsupported_adapter_is_inventoried_not_guessed(tmp_path: Path) -> None:
    index = _fixture(tmp_path, expected=["p1"])
    data = _read(index)
    data["sources"][0]["adapter_version"] = "unreviewed/v99"
    _json(index, data)
    output = tmp_path / "intake"
    assert run_intake(index, PROTOCOL, output)["run_status"] == "complete"
    assert _rows(output / "pair_manifest.jsonl") == []
    (inventory,) = _rows(output / "archive_inventory.jsonl")
    assert inventory["observed_sha256"] == _sha(tmp_path / "pairs.jsonl")
    assert inventory["adapter_status"] == "unsupported"
    assert "unsupported_adapter" in (output / "missing_inputs.csv").read_text()


def test_modified_source_hash_is_rejected(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    with (tmp_path / "pairs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="(?i)sha|hash"):
        run_intake(index, PROTOCOL, tmp_path / "intake")
    assert not (tmp_path / "intake/run.json").exists()


def test_same_count_without_ids_is_unknown_not_exact_recovery(tmp_path: Path) -> None:
    index = _fixture(tmp_path, pairs=[_pair(str(i)) for i in range(172)])
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    comparison = _comparison(output)[0]
    assert int(comparison["recovered_count"]) == 172
    assert comparison["coverage_status"] == "unknown"


def test_zero_recovered_sources_still_bind_protocol_and_coverage(tmp_path: Path) -> None:
    index = _fixture(tmp_path, expected=["p1"])
    payload = _read(index)
    payload["sources"] = [_source(None)]
    _json(index, payload)
    output = tmp_path / "intake"
    result = run_intake(index, PROTOCOL, output)
    assert result["run_status"] == "complete"
    assert _rows(output / "pair_manifest.jsonl") == []
    metadata = _read(output / "pair_manifest.meta.json")
    assert (
        metadata["protocol_sha256"]
        == "d13794e9d29f1c1750c66f996475ba05f7932e239768963ef1014b562868e70b"
    )
    assert "run.json" not in json.dumps(metadata)
    _validate_refs(metadata, output / "pair_manifest.meta.json")
    _validate_refs(_read(output / "run.json"), output / "run.json")
    assert int(_comparison(output)[0]["missing_count"]) == 1


def test_aggregates_do_not_create_raw_generation_rows(tmp_path: Path) -> None:
    index = _fixture(tmp_path, historical_counts={"correct": 129, "offset": 44, "cross": 42})
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    assert _rows(output / "archive_generation_records.jsonl") == []
    assert json.loads(_comparison(output)[0]["historical_counts_json"])["correct"] == 129


def test_128_successes_are_valid_observations_against_129_reference(tmp_path: Path) -> None:
    keys = [f"p-{i:03}" for i in range(172)]
    index = _fixture(
        tmp_path,
        pairs=[_pair(key) for key in keys],
        generations=[
            _generation(
                key,
                archived_scoring={
                    "parser_id": "archive-test",
                    "parser_version": "1",
                    "gold_match": i < 128,
                },
            )
            for i, key in enumerate(keys)
        ],
        expected=keys,
        historical_counts={"correct": 129},
    )
    output = tmp_path / "intake"
    assert run_intake(index, PROTOCOL, output)["run_status"] == "complete"
    assert len(_rows(output / "pair_manifest.jsonl")) == 172
    assert len(_rows(output / "archive_generation_records.jsonl")) == 172
    comparison = _comparison(output)[0]
    assert comparison["coverage_status"] == "complete"
    assert json.loads(comparison["historical_counts_json"])["correct"] == 129
    (observed,) = json.loads(comparison["observed_scoring_counts_json"])
    assert observed["correct_n"] == 128
    assert observed["incorrect_n"] == 44
    assert observed["scored_n"] == 172


def test_real_raw_generations_and_all_refs_are_preserved(tmp_path: Path) -> None:
    raw = "Answer: 42.\r\n"
    index = _fixture(tmp_path, generations=[_generation(raw_text=raw)], expected=["p1"])
    output = tmp_path / "intake"
    run_intake(index, PROTOCOL, output)
    (generation,) = _rows(output / "archive_generation_records.jsonl")
    assert generation["raw_text"] == raw
    assert generation["termination"] == "unknown"
    (pair,) = _rows(output / "pair_manifest.jsonl")
    assert generation["pair_id"] == pair["pair_id"]
    assert pair["archived_generations"][0]["generation_id"] == generation["generation_id"]
    for path in output.glob("*.json"):
        _validate_refs(_read(path), path)
    for path in output.glob("*.jsonl"):
        for row in _rows(path):
            _validate_refs(row, path)


def test_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    output = tmp_path / "intake"
    output.mkdir()
    sentinel = output / "user_data.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        run_intake(index, PROTOCOL, output)
    assert sentinel.read_text() == "keep me"


def test_cli_help_is_cpu_only_and_unimplemented_operations_are_rejected() -> None:
    script = """
import sys
from typo_cot.cli import main
try:
    main(['rebuttal-v2', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "intake" in result.stdout
    unsupported = subprocess.run(
        [sys.executable, "-c", "from typo_cot.cli import main; main(['rebuttal-v2', 'run'])"],
        capture_output=True,
        text=True,
    )
    assert unsupported.returncode == 2


def test_cli_dispatches_intake_and_reports_invalid_input(tmp_path: Path, capsys) -> None:
    from typo_cot.cli import main

    index = _fixture(tmp_path, expected=["p1"])
    output = tmp_path / "intake"
    args = [
        "rebuttal-v2",
        "intake",
        "--archive-index",
        str(index),
        "--protocol",
        str(PROTOCOL),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["run_status"] == "complete"
    assert main(args) == 1
    assert "error:" in capsys.readouterr().err

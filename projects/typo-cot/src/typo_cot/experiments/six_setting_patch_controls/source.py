"""Hash-bound fixed-window reference answers for the control comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.build_rebuttal_manifest.records import (
    iter_jsonl_objects,
    load_json_object,
    sha256_file,
)
from typo_cot.experiments.catalog import PAPER_SHA256


@dataclass(frozen=True, slots=True)
class FixedReferenceAnswer:
    """Correct-arm event and its paired regenerated clean answer."""

    clean_answer: str
    correct_event: bool


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _safe_run_dir(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("fixed-window run path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("fixed-window run path must remain relative to its root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("fixed-window run path escaped its configured root")
    return resolved


def _output_paths(run_dir: Path, run: Mapping[str, object]) -> dict[str, Path]:
    expected = {
        "fixed_window_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    }
    outputs = _mapping(run.get("outputs"), field="fixed-window outputs")
    if set(outputs) != expected:
        raise ValueError("fixed-window output inventory differs")
    paths: dict[str, Path] = {}
    for name in sorted(expected):
        metadata = _mapping(outputs.get(name), field=f"fixed-window outputs.{name}")
        digest = metadata.get("sha256")
        records = metadata.get("records")
        path = run_dir / name
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"fixed-window output SHA-256 differs: {path}")
        if not isinstance(records, int) or isinstance(records, bool) or records <= 0:
            raise ValueError(f"fixed-window output record count is invalid: {path}")
        paths[name] = path
    return paths


def _generation_answer(value: object, *, field: str) -> tuple[str, bool, bool]:
    generation = _mapping(value, field=field)
    answer = generation.get("value")
    extracted = generation.get("is_extracted")
    correct = generation.get("is_correct")
    if not isinstance(answer, str):
        raise ValueError(f"{field}.value must be a string")
    if not isinstance(extracted, bool) or extracted is not bool(answer):
        raise ValueError(f"{field}.is_extracted differs from its value")
    if not isinstance(correct, bool):
        raise ValueError(f"{field}.is_correct must be boolean")
    return answer, extracted, correct


def load_fixed_references(
    records: Sequence[Mapping[str, object]],
    fixed_window_root: Path,
) -> dict[str, FixedReferenceAnswer]:
    """Validate every referenced fixed run and return the paired clean answers."""

    root = Path(fixed_window_root)
    if not root.is_dir():
        raise ValueError(f"fixed-window root is not a directory: {root}")
    restoration = [record for record in records if record["cohorts"]["restoration"]]
    by_reference: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in restoration:
        fixed = _mapping(record.get("fixed_window"), field="manifest fixed_window")
        key = str(fixed.get("run_path")), str(fixed.get("run_sha256"))
        by_reference.setdefault(key, []).append(record)

    references: dict[str, FixedReferenceAnswer] = {}
    for (relative_run, recorded_run_sha), reference_records in sorted(by_reference.items()):
        run_dir = _safe_run_dir(root, relative_run)
        run_path = run_dir / "run.json"
        if not run_path.is_file() or sha256_file(run_path) != recorded_run_sha:
            raise ValueError(f"fixed-window run SHA-256 differs: {run_path}")
        run = load_json_object(run_path)
        if (
            run.get("schema_version") != "fixed-window-answer-patching-run/v1"
            or run.get("paper_sha256") != PAPER_SHA256
            or run.get("operation") != "fixed-window-answer-patching"
            or run.get("status") != "completed"
            or run.get("failures") not in ([], None)
        ):
            raise ValueError(f"fixed-window run contract differs: {run_path}")
        model = str(reference_records[0]["model"])
        task = str(reference_records[0]["task"])
        arguments = _mapping(run.get("arguments"), field="fixed-window arguments")
        if arguments.get("model") != model or arguments.get("benchmark") != task:
            raise ValueError(f"fixed-window run setting differs: {run_path}")
        if (
            not isinstance(arguments.get("layers"), list)
            or "0:6" not in arguments["layers"]
            or not isinstance(arguments.get("directions"), list)
            or "clean-to-edited" not in arguments["directions"]
        ):
            raise ValueError(f"fixed-window run lacks the required arm: {run_path}")
        producer_protocol = _mapping(run.get("protocol"), field="fixed-window producer protocol")
        if producer_protocol.get("schema_version") != "fixed-window-answer-patching-protocol/v1":
            raise ValueError(f"fixed-window producer protocol differs: {run_path}")
        runtime = _mapping(run.get("runtime"), field="fixed-window runtime")
        revisions = {str(record["source"]["model_revision"]) for record in reference_records}
        if len(revisions) != 1:
            raise ValueError("manifest model revisions differ within a fixed-window run")
        revision = next(iter(revisions))
        for field in ("requested_revision", "model_revision", "tokenizer_revision"):
            if runtime.get(field) != revision:
                raise ValueError(f"fixed-window runtime {field} differs from the manifest")
        paths = _output_paths(run_dir, run)

        expected_by_key = {
            (str(record["target_rule"]), str(record["sample_id"])): record
            for record in reference_records
        }
        if len(expected_by_key) != len(reference_records):
            raise ValueError("fixed-window reference identities are duplicated")
        clean_by_key: dict[tuple[str, str], str] = {}
        status_rows = tuple(iter_jsonl_objects(paths["pair_status_records.jsonl"]))
        status_metadata = _mapping(
            _mapping(run.get("outputs"), field="fixed-window outputs").get(
                "pair_status_records.jsonl"
            ),
            field="fixed-window pair status metadata",
        )
        if len(status_rows) != status_metadata.get("records"):
            raise ValueError("fixed-window status count differs from run metadata")
        for line_number, _line, status in status_rows:
            key = str(status.get("targeting")), str(status.get("sample_id"))
            record = expected_by_key.get(key)
            if record is None:
                continue
            context = f"{paths['pair_status_records.jsonl']}:{line_number}"
            direction = _mapping(
                _mapping(status.get("direction_status"), field=f"{context}.direction_status").get(
                    "clean-to-edited"
                ),
                field=f"{context}.direction_status.clean-to-edited",
            )
            baseline = _mapping(status.get("baseline"), field=f"{context}.baseline")
            clean_answer, clean_extracted, clean_correct = _generation_answer(
                baseline.get("clean"), field=f"{context}.baseline.clean"
            )
            _edited_answer, _edited_extracted, edited_correct = _generation_answer(
                baseline.get("edited"), field=f"{context}.baseline.edited"
            )
            if (
                status.get("schema_version") != "fixed-window-answer-patching-pair-status/v1"
                or status.get("paper_sha256") != PAPER_SHA256
                or status.get("model") != model
                or status.get("benchmark") != task
                or status.get("source_record_sha256") != record["source"]["source_record_sha256"]
                or direction.get("included") is not True
                or direction.get("exclusion_reason") is not None
                or not clean_extracted
                or not clean_correct
                or edited_correct
            ):
                raise ValueError(f"fixed-window included status differs: {context}")
            if key in clean_by_key:
                raise ValueError(f"fixed-window included status is duplicated: {context}")
            clean_by_key[key] = clean_answer
        if set(clean_by_key) != set(expected_by_key):
            raise ValueError("fixed-window statuses do not cover the manifest denominator")

        correct_by_key: dict[tuple[str, str], bool] = {}
        record_rows = tuple(iter_jsonl_objects(paths["fixed_window_records.jsonl"]))
        record_metadata = _mapping(
            _mapping(run.get("outputs"), field="fixed-window outputs").get(
                "fixed_window_records.jsonl"
            ),
            field="fixed-window record metadata",
        )
        if len(record_rows) != record_metadata.get("records"):
            raise ValueError("fixed-window record count differs from run metadata")
        for line_number, _line, row in record_rows:
            if row.get("direction") != "clean-to-edited" or row.get("window") != "0:6":
                continue
            key = str(row.get("targeting")), str(row.get("sample_id"))
            record = expected_by_key.get(key)
            if record is None:
                continue
            context = f"{paths['fixed_window_records.jsonl']}:{line_number}"
            event = row.get("event")
            patched_answer, patched_extracted, _patched_correct = _generation_answer(
                row.get("patched_answer"), field=f"{context}.patched_answer"
            )
            expected_event = bool(
                patched_extracted
                and answers_equal(patched_answer, clean_by_key[key], benchmark=task)
            )
            if (
                row.get("schema_version") != "fixed-window-answer-patching-window/v1"
                or row.get("paper_sha256") != PAPER_SHA256
                or row.get("model") != model
                or row.get("benchmark") != task
                or row.get("source_record_sha256") != record["source"]["source_record_sha256"]
                or row.get("window_start") != 0
                or row.get("window_stop") != 6
                or not isinstance(row.get("num_layers"), int)
                or isinstance(row.get("num_layers"), bool)
                or int(row["num_layers"]) < 6
                or not isinstance(event, bool)
                or event is not expected_event
                or event is not record["fixed_window"]["event"]
            ):
                raise ValueError(f"fixed-window correct record differs: {context}")
            if key in correct_by_key:
                raise ValueError(f"fixed-window correct record is duplicated: {context}")
            correct_by_key[key] = event
        if set(correct_by_key) != set(expected_by_key):
            raise ValueError("fixed-window correct records do not cover the manifest denominator")
        for key, record in expected_by_key.items():
            pair_id = str(record["pair_id"])
            references[pair_id] = FixedReferenceAnswer(
                clean_answer=clean_by_key[key],
                correct_event=correct_by_key[key],
            )
    if len(references) != len(restoration):
        raise ValueError("fixed-window references do not cover every restoration pair")
    return references


__all__ = ["FixedReferenceAnswer", "load_fixed_references"]

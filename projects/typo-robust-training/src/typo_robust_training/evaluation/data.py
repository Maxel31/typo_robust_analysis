"""Hash-bound evaluation pairs and one-use sealed-role access."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.records import TypoEdit, infer_single_word_typo_edit


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_ROLES = {
    "tune": ("tune_manifest.jsonl", "tune"),
    "pre-pr-gate": ("pre_pr_gate_manifest.jsonl", "pre_pr_gate"),
    "final-test": ("final_test_manifest.jsonl", "final_test"),
}
_SPLITS = ("same-task", "unseen-task", "unseen-content", "unseen-typo")
_SAME_TASKS = frozenset({"gsm8k", "mmlu", "arc"})
_UNSEEN_TASKS = frozenset({"mmlu_pro", "math_500", "commonsense_qa"})
_UNSEEN_CONTENT_SOURCES = frozenset({"fineweb_edu", "dolma"})
_SYNTHETIC_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "clean_text",
    "typo_text",
    "task",
    "answer",
    "metadata",
    "operation",
    "operations",
    "edit_count",
    "generator_seed",
    "generator_variant",
    "edits",
}
_NATURAL_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "clean_text",
    "typo_text",
    "task",
    "answer",
    "operation",
    "training_eligible",
    "repository",
    "repository_license",
    "clean_sha256",
    "typo_sha256",
    "metadata",
}
_EDIT_FIELDS = {
    "operation",
    "clean_word",
    "typo_word",
    "clean_char_span",
    "typo_char_span",
}
_EVALUATION_MANIFEST_FIELDS = {
    "schema_version",
    "protocol_sha256",
    "source_revisions",
    "split_roles",
    "training_operations",
    "operation_probabilities",
    "held_out_operations",
    "pre_pr_gate_consumed",
    "final_test_opened",
    "artifact_sha256",
    "data_identity_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"evaluation artifact is not a file: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"evaluation artifact must contain an object: {path}")
    return payload


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation pair {field} must be non-empty")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"evaluation pair {field} must be a non-negative integer")
    return value


def _span(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ValueError(f"evaluation pair {field} must be an increasing integer span")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class EvaluationPair:
    record_id: str
    kind: str
    source: str
    source_revision: str
    source_split: str
    source_id: str
    group_id: str
    role: str
    clean_text: str
    typo_text: str
    task: str | None
    answer: str | None
    operation: str
    edits: tuple[TypoEdit, ...]
    metadata: Mapping[str, object]
    strata: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        expected_role: str,
        held_out_operations: frozenset[str],
    ) -> EvaluationPair:
        if not isinstance(value, Mapping):
            raise ValueError("evaluation pair must contain an object")
        kind = value.get("kind")
        expected_fields = _SYNTHETIC_FIELDS if kind == "synthetic" else _NATURAL_FIELDS
        expected_schema = (
            "robustness-fixed-typo-pair/v1" if kind == "synthetic" else "robustness-natural-pair/v1"
        )
        if kind not in {"synthetic", "natural"} or set(value) != expected_fields:
            raise ValueError("evaluation pair fields differ")
        if value.get("schema_version") != expected_schema or value.get("split") != expected_role:
            raise ValueError("evaluation pair schema or role differs")
        record_id = _text(value.get("record_id"), field="record_id")
        revision = _text(value.get("source_revision"), field="source_revision")
        if _SHA64.fullmatch(record_id) is None or _SHA40.fullmatch(revision) is None:
            raise ValueError("evaluation pair identity hashes differ")
        source = _text(value.get("source"), field="source")
        clean = _text(value.get("clean_text"), field="clean_text")
        typo = _text(value.get("typo_text"), field="typo_text")
        if clean == typo:
            raise ValueError("evaluation pair must contain a typo")
        task = _optional_text(value.get("task"), field="task")
        answer = _optional_text(value.get("answer"), field="answer")
        if (task is None) != (answer is None):
            raise ValueError("evaluation pair task and answer must be jointly present")
        if task is not None and task not in _SAME_TASKS | _UNSEEN_TASKS:
            raise ValueError("evaluation pair task is outside the frozen inventory")
        operation = _text(value.get("operation"), field="operation")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("evaluation pair metadata must be an object")
        try:
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation pair metadata must be canonical JSON") from exc

        if kind == "synthetic":
            raw_edits = value.get("edits")
            operations = value.get("operations")
            if not isinstance(raw_edits, list) or not raw_edits:
                raise ValueError("synthetic evaluation pair must contain edits")
            edits: list[TypoEdit] = []
            for raw_edit in raw_edits:
                if not isinstance(raw_edit, Mapping) or set(raw_edit) != _EDIT_FIELDS:
                    raise ValueError("synthetic evaluation edit fields differ")
                edit = TypoEdit(
                    operation=_text(raw_edit.get("operation"), field="edit.operation"),
                    clean_word=_text(raw_edit.get("clean_word"), field="edit.clean_word"),
                    typo_word=_text(raw_edit.get("typo_word"), field="edit.typo_word"),
                    clean_char_span=_span(
                        raw_edit.get("clean_char_span"), field="edit.clean_char_span"
                    ),
                    typo_char_span=_span(
                        raw_edit.get("typo_char_span"), field="edit.typo_char_span"
                    ),
                )
                if (
                    clean[slice(*edit.clean_char_span)] != edit.clean_word
                    or typo[slice(*edit.typo_char_span)] != edit.typo_word
                ):
                    raise ValueError("synthetic evaluation edit text differs from its spans")
                edits.append(edit)
            if (
                not isinstance(operations, list)
                or operations != [edit.operation for edit in edits]
                or _integer(value.get("edit_count"), field="edit_count") != len(edits)
                or operation != (edits[0].operation if len(edits) == 1 else "multiple")
            ):
                raise ValueError("synthetic evaluation edit inventory differs")
            _integer(value.get("generator_seed"), field="generator_seed")
            _integer(value.get("generator_variant"), field="generator_variant")
            parsed_edits = tuple(edits)
        else:
            if type(value.get("training_eligible")) is not bool:
                raise ValueError("natural evaluation training_eligible must be boolean")
            _text(value.get("repository"), field="repository")
            _text(value.get("repository_license"), field="repository_license")
            if (
                value.get("clean_sha256") != hashlib.sha256(clean.encode()).hexdigest()
                or value.get("typo_sha256") != hashlib.sha256(typo.encode()).hexdigest()
            ):
                raise ValueError("natural evaluation pair content hash differs")
            parsed_edits = (infer_single_word_typo_edit(clean, typo, operation=operation),)

        strata: list[str] = []
        if task in _SAME_TASKS:
            strata.append("same-task")
        if task in _UNSEEN_TASKS:
            strata.append("unseen-task")
        if source in _UNSEEN_CONTENT_SOURCES:
            strata.append("unseen-content")
        if kind == "natural" or any(edit.operation in held_out_operations for edit in parsed_edits):
            strata.append("unseen-typo")
        return cls(
            record_id=record_id,
            kind=str(kind),
            source=source,
            source_revision=revision,
            source_split=_text(value.get("source_split"), field="source_split"),
            source_id=_text(value.get("source_id"), field="source_id"),
            group_id=_text(value.get("group_id"), field="group_id"),
            role=expected_role,
            clean_text=clean,
            typo_text=typo,
            task=task,
            answer=answer,
            operation=operation,
            edits=parsed_edits,
            metadata=MappingProxyType(dict(metadata)),
            strata=tuple(strata),
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataBundle:
    root: Path
    evaluation_role: str
    records: tuple[EvaluationPair, ...]
    manifest_path: Path
    manifest_sha256: str
    evaluation_manifest_sha256: str
    data_identity_sha256: str
    held_out_operations: tuple[str, ...]


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _access_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": "robustness-evaluation-access/v1", "roles": {}}
    payload = _object(path)
    if (
        set(payload) != {"schema_version", "roles"}
        or payload.get("schema_version") != "robustness-evaluation-access/v1"
    ):
        raise ValueError("evaluation access registry fields differ")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("evaluation access registry roles differ")
    return {"schema_version": payload["schema_version"], "roles": dict(roles)}


@contextmanager
def _evaluation_access_lock(root: Path) -> Iterator[None]:
    """Serialize the complete read-modify-write transaction for sealed roles."""

    import fcntl

    lock_path = Path(root).resolve() / "evaluation_access.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _claim_evaluation_role(
    root: Path,
    *,
    role: str,
    binding: str,
    output_dir: Path,
    confirm: bool,
    resume: bool,
) -> None:
    if role == "tune":
        if confirm:
            raise ValueError("tune evaluation does not accept sealed-role confirmation")
        return
    if not confirm:
        raise ValueError("sealed evaluation role requires explicit confirmation")
    with _evaluation_access_lock(root):
        access_path = root / "evaluation_access.json"
        registry = _access_registry(access_path)
        roles = registry["roles"]
        if not isinstance(roles, dict):
            raise RuntimeError("validated evaluation access roles changed type")
        key = _ROLES[role][1]
        if role == "final-test":
            gate = roles.get("pre_pr_gate")
            if (
                not isinstance(gate, Mapping)
                or gate.get("status") != "completed"
                or gate.get("gate_passed") is not True
            ):
                raise ValueError("final-test requires a completed passing pre-PR gate")
        existing = roles.get(key)
        expected_identity = {
            "access_binding_sha256": binding,
            "output_dir": str(output_dir.resolve()),
        }
        if existing is not None:
            if (
                not resume
                or not isinstance(existing, Mapping)
                or any(existing.get(field) != value for field, value in expected_identity.items())
            ):
                raise ValueError(f"sealed evaluation role {role} was already opened")
            return
        if resume:
            raise ValueError("sealed evaluation --resume has no matching access record")
        roles[key] = {
            **expected_identity,
            "status": "opened",
            "report_sha256": None,
            "gate_passed": None,
        }
        _atomic_json(access_path, registry)


def complete_evaluation_role(
    root: Path,
    *,
    evaluation_role: str,
    access_binding_sha256: str,
    report_sha256: str,
    gate_passed: bool,
) -> None:
    """Seal one role's result and expose final-test only after a passing gate."""

    if evaluation_role == "tune":
        return
    if evaluation_role not in _ROLES or any(
        _SHA64.fullmatch(value) is None for value in (access_binding_sha256, report_sha256)
    ):
        raise ValueError("evaluation completion identity differs")
    resolved_root = Path(root).resolve()
    with _evaluation_access_lock(resolved_root):
        access_path = resolved_root / "evaluation_access.json"
        registry = _access_registry(access_path)
        roles = registry["roles"]
        if not isinstance(roles, dict):
            raise RuntimeError("validated evaluation access roles changed type")
        key = _ROLES[evaluation_role][1]
        entry = roles.get(key)
        if (
            not isinstance(entry, Mapping)
            or entry.get("access_binding_sha256") != access_binding_sha256
            or entry.get("status") not in {"opened", "completed"}
        ):
            raise ValueError("evaluation completion does not match the opened role")
        roles[key] = {
            **dict(entry),
            "status": "completed",
            "report_sha256": report_sha256,
            "gate_passed": bool(gate_passed),
        }
        _atomic_json(access_path, registry)


def _declared_hash(
    run: Mapping[str, object],
    evaluation: Mapping[str, object],
    *,
    name: str,
    path: Path,
) -> str:
    outputs = run.get("outputs")
    artifacts = evaluation.get("artifact_sha256")
    if not isinstance(outputs, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("evaluation data artifact inventory differs")
    output = outputs.get(name)
    if not isinstance(output, Mapping):
        raise ValueError(f"evaluation data run does not declare {name}")
    digest = _sha256_file(path) if path.is_file() else None
    manifest_digest = digest if name == "evaluation_manifest.json" else artifacts.get(name)
    if output.get("sha256") != digest or manifest_digest != digest:
        raise ValueError(f"evaluation data {name} hash differs")
    return str(digest)


def _rows(path: Path) -> tuple[object, ...]:
    rows: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"evaluation manifest has a blank line at {line_number}")
            rows.append(strict_loads(line, context=f"{path}:{line_number}"))
    if not rows:
        raise ValueError("evaluation role manifest is empty")
    return tuple(rows)


def load_evaluation_bundle(
    root: Path,
    *,
    evaluation_role: str,
    splits: Sequence[str],
    model: str,
    model_revision: str,
    access_binding_sha256: str,
    output_dir: Path,
    confirm_sealed_role: bool,
    resume: bool,
    expected_data_identity_sha256: str | None = None,
) -> EvaluationDataBundle:
    """Validate one role and return only the requested overlapping strata."""

    if evaluation_role not in _ROLES:
        raise ValueError("evaluation role is unsupported")
    requested = tuple(splits)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(split not in _SPLITS for split in requested)
    ):
        raise ValueError("evaluation splits must be unique members of the frozen inventory")
    if _SHA40.fullmatch(model_revision) is None or _SHA64.fullmatch(access_binding_sha256) is None:
        raise ValueError("evaluation model or access identity differs")
    resolved = Path(root).resolve()
    run = _object(resolved / "run.json")
    if (
        run.get("schema_version") != "build-robustness-training-data-run/v1"
        or run.get("status") != "completed"
    ):
        raise ValueError("evaluation data build is not completed")
    data_protocol = run.get("protocol")
    if (
        not isinstance(data_protocol, Mapping)
        or data_protocol.get("model") != model
        or data_protocol.get("model_revision") != model_revision
    ):
        raise ValueError("evaluation data model identity differs")
    evaluation_path = resolved / "evaluation_manifest.json"
    evaluation = _object(evaluation_path)
    if (
        set(evaluation) != _EVALUATION_MANIFEST_FIELDS
        or evaluation.get("schema_version") != "robustness-evaluation-manifest/v1"
    ):
        raise ValueError("evaluation manifest fields or schema differ")
    held_out = evaluation.get("held_out_operations")
    if not isinstance(held_out, list) or held_out != ["adjacent-transposition"]:
        raise ValueError("evaluation held-out typo inventory differs")
    filename, expected_role = _ROLES[evaluation_role]
    manifest_path = resolved / filename
    manifest_sha = _declared_hash(run, evaluation, name=filename, path=manifest_path)
    evaluation_sha = _declared_hash(
        run, evaluation, name="evaluation_manifest.json", path=evaluation_path
    )
    identity = evaluation.get("data_identity_sha256")
    if not isinstance(identity, str) or _SHA64.fullmatch(identity) is None:
        raise ValueError("evaluation data identity differs")
    if expected_data_identity_sha256 is not None and identity != expected_data_identity_sha256:
        raise ValueError("evaluation data identity differs from the trained adapters")
    _claim_evaluation_role(
        resolved,
        role=evaluation_role,
        binding=access_binding_sha256,
        output_dir=Path(output_dir),
        confirm=confirm_sealed_role,
        resume=resume,
    )
    parsed = tuple(
        EvaluationPair.from_dict(
            row,
            expected_role=expected_role,
            held_out_operations=frozenset(held_out),
        )
        for row in _rows(manifest_path)
    )
    if len({row.record_id for row in parsed}) != len(parsed):
        raise ValueError("evaluation role contains duplicate record IDs")
    selected = tuple(
        sorted(
            (row for row in parsed if set(row.strata) & set(requested)),
            key=lambda row: row.record_id,
        )
    )
    if not selected:
        raise ValueError("evaluation requested splits select no records")
    return EvaluationDataBundle(
        root=resolved,
        evaluation_role=evaluation_role,
        records=selected,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        evaluation_manifest_sha256=evaluation_sha,
        data_identity_sha256=identity,
        held_out_operations=tuple(held_out),
    )


__all__ = [
    "EvaluationDataBundle",
    "EvaluationPair",
    "complete_evaluation_role",
    "load_evaluation_bundle",
]

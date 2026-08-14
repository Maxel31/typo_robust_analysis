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
from typo_robust_training.evaluation.perturb import FROZEN_EVALUATION_TYPO_VERSION
from typo_robust_training.integrity import sha256_file as _sha256_file


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_SOURCE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_ROLES = {
    "tune": ("tune_manifest.jsonl", "tune"),
    "pre-pr-gate": ("pre_pr_gate_manifest.jsonl", "pre_pr_gate"),
    "final-test": ("final_test_manifest.jsonl", "final_test"),
}
_SPLITS = ("same-task", "unseen-task", "unseen-content", "unseen-typo")
_SAME_TASKS = frozenset({"gsm8k", "mmlu", "arc"})
_UNSEEN_TASKS = frozenset({"mmlu_pro", "math_500", "commonsense_qa"})
_UNSEEN_CONTENT_SOURCES = frozenset({"fineweb_edu", "dolma"})
_EVALUATION_CONDITIONS = frozenset(
    {
        "random-1",
        "random-2",
        "random-4",
        "transposition-2",
        "natural-injection",
        "natural-lm-pair",
    }
)
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
_CLEAN_CORPUS_FIELDS = {
    "schema_version",
    "kind",
    "record_id",
    "source",
    "source_revision",
    "source_split",
    "source_id",
    "group_id",
    "split",
    "text",
    "content_sha256",
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
_FROZEN_REGISTRY_FIELDS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "source_config_sha256",
    "exclusion_data_protocol_sha256",
    "source_revisions",
    "exclusion_artifact_sha256",
    "artifact_sha256",
    "data_identity_sha256",
    "roles",
    "opening_order",
    "primary_condition",
    "generator_seed",
    "generator",
    "task_capacity_census",
    "natural_evaluation_axes",
    "corpus_exact_text_policy",
}
_EXCLUSION_ARTIFACTS = {
    "training_sources.jsonl",
    "diagnostic_manifest.jsonl",
    "tune_manifest.jsonl",
    "run.json",
}


def _valid_task_capacity_census(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"tune", "sealed"}:
        return False
    expected_tasks = {
        "tune": {"gsm8k", "mmlu", "arc"},
        "sealed": {"gsm8k", "mmlu", "arc", "mmlu_pro", "math_500", "commonsense_qa"},
    }
    fields = {
        "source_split_records",
        "after_exclusions",
        "typo_grid_eligible",
        "transposition_eligible",
        "required",
    }
    for role, tasks in expected_tasks.items():
        rows = value.get(role)
        if not isinstance(rows, Mapping) or set(rows) != tasks:
            return False
        for row in rows.values():
            if (
                not isinstance(row, Mapping)
                or set(row) != fields
                or any(type(row[field]) is not int or row[field] < 0 for field in fields)
                or not row["source_split_records"]
                >= row["after_exclusions"]
                >= row["typo_grid_eligible"]
                >= row["required"]
                or row["transposition_eligible"] > row["typo_grid_eligible"]
            ):
                return False
    return True


def _object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"evaluation artifact is not a file: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"evaluation artifact must contain an object: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class FrozenEvaluationProvenance:
    """Data identities that a frozen evaluation population excluded."""

    root: Path
    data_identity_sha256: str
    exclusion_artifact_sha256: Mapping[str, str]


def _frozen_registry(
    root: Path,
    *,
    study_protocol_sha256: str,
) -> tuple[Mapping[str, object], Path]:
    if _SHA64.fullmatch(study_protocol_sha256) is None:
        raise ValueError("frozen evaluation study protocol hash differs")
    registry_path = root / "registry.json"
    registry = _object(registry_path)
    natural_axes = registry.get("natural_evaluation_axes")
    exclusion_artifacts = registry.get("exclusion_artifact_sha256")
    corrected_words = (
        natural_axes.get("corrected_word_split") if isinstance(natural_axes, Mapping) else None
    )
    if (
        set(registry) != _FROZEN_REGISTRY_FIELDS
        or registry.get("schema_version") != "robustness-evaluation-registry/v1"
        or registry.get("opening_order") != ["pre_pr_gate", "final_test"]
        or registry.get("primary_condition") != "random-2"
        or registry.get("generator_seed") != 42
        or registry.get("generator") != FROZEN_EVALUATION_TYPO_VERSION
        or registry.get("corpus_exact_text_policy")
        != "deduplicate-exact-utf8-text-across-sources-and-roles/v1"
        or not _valid_task_capacity_census(registry.get("task_capacity_census"))
        or not isinstance(exclusion_artifacts, Mapping)
        or set(exclusion_artifacts) != _EXCLUSION_ARTIFACTS
        or any(
            not isinstance(digest, str) or _SHA64.fullmatch(digest) is None
            for digest in exclusion_artifacts.values()
        )
        or registry.get("source_config_sha256") != registry.get("exclusion_data_protocol_sha256")
        or registry.get("protocol_sha256") != study_protocol_sha256
        or not isinstance(natural_axes, Mapping)
        or natural_axes.get("language_model_pairs") != "repository-disjoint/v1"
        or natural_axes.get("task_injection") != "corrected-word-disjoint/v1"
        or not isinstance(corrected_words, Mapping)
        or set(corrected_words) != {"train", "tune", "pre_pr_gate", "final_test"}
    ):
        raise ValueError("frozen evaluation registry fields or protocol differ")
    run = _object(root / "run.json")
    if (
        run.get("schema_version") != "freeze-robustness-evaluation-run/v1"
        or run.get("status") != "completed"
        or run.get("protocol_sha256") != study_protocol_sha256
        or run.get("source_config_sha256") != registry.get("source_config_sha256")
        or run.get("task_capacity_census") != registry.get("task_capacity_census")
        or run.get("registry_sha256") != _sha256_file(registry_path)
    ):
        raise ValueError("frozen evaluation data build is not completed")
    return registry, registry_path


def load_frozen_evaluation_provenance(
    root: Path,
    *,
    study_protocol_sha256: str,
) -> FrozenEvaluationProvenance:
    """Return the exclusion identity before any frozen role is claimed."""

    resolved = Path(root).resolve()
    registry, _registry_path = _frozen_registry(
        resolved,
        study_protocol_sha256=study_protocol_sha256,
    )
    identity = registry.get("data_identity_sha256")
    exclusions = registry.get("exclusion_artifact_sha256")
    if (
        not isinstance(identity, str)
        or _SHA64.fullmatch(identity) is None
        or not isinstance(exclusions, Mapping)
    ):
        raise ValueError("frozen evaluation provenance differs")
    return FrozenEvaluationProvenance(
        root=resolved,
        data_identity_sha256=identity,
        exclusion_artifact_sha256=MappingProxyType(
            {str(name): str(digest) for name, digest in exclusions.items()}
        ),
    )


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
    mechanistic_audit: bool
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
        if _SHA64.fullmatch(record_id) is None or _SOURCE_REVISION.fullmatch(revision) is None:
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
        evaluation_condition = metadata.get("evaluation_condition")
        if evaluation_condition not in _EVALUATION_CONDITIONS:
            raise ValueError("evaluation pair typo condition is unsupported")
        mechanistic_audit = metadata.get("mechanistic_audit", False)
        if type(mechanistic_audit) is not bool:
            raise ValueError("evaluation pair mechanistic_audit must be boolean")
        if mechanistic_audit and (
            kind != "synthetic" or evaluation_condition != "random-2" or task is None
        ):
            raise ValueError("mechanistic audit must select a synthetic random-2 task pair")
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
        if (
            kind == "natural"
            or evaluation_condition in {"natural-injection", "transposition-2"}
            or any(edit.operation in held_out_operations for edit in parsed_edits)
        ):
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
            mechanistic_audit=mechanistic_audit,
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


@dataclass(frozen=True, slots=True)
class EvaluationCorpusRecord:
    record_id: str
    kind: str
    source: str
    source_revision: str
    source_split: str
    source_id: str
    group_id: str
    role: str
    clean_text: str
    typo_text: str | None
    edits: tuple[TypoEdit, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EvaluationCorpusBundle:
    root: Path
    evaluation_role: str
    records: tuple[EvaluationCorpusRecord, ...]
    manifest_path: Path
    manifest_sha256: str


def _corpus_record(value: object, *, expected_role: str) -> EvaluationCorpusRecord:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation corpus row must be an object")
    kind = value.get("kind")
    expected_fields = _CLEAN_CORPUS_FIELDS if kind == "clean-corpus" else _NATURAL_FIELDS
    expected_schema = (
        "robustness-evaluation-corpus-record/v1"
        if kind == "clean-corpus"
        else "robustness-natural-pair/v1"
    )
    if kind not in {"clean-corpus", "natural"} or set(value) != expected_fields:
        raise ValueError("evaluation corpus row fields differ")
    if value.get("schema_version") != expected_schema or value.get("split") != expected_role:
        raise ValueError("evaluation corpus row schema or role differs")
    record_id = _text(value.get("record_id"), field="record_id")
    revision = _text(value.get("source_revision"), field="source_revision")
    if _SHA64.fullmatch(record_id) is None or _SOURCE_REVISION.fullmatch(revision) is None:
        raise ValueError("evaluation corpus row identity differs")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("evaluation corpus metadata must be an object")
    try:
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation corpus metadata must be canonical JSON") from exc
    if kind == "clean-corpus":
        clean_text = _text(value.get("text"), field="text")
        if value.get("content_sha256") != hashlib.sha256(clean_text.encode()).hexdigest():
            raise ValueError("evaluation corpus content hash differs")
        typo_text = None
        edits: tuple[TypoEdit, ...] = ()
    else:
        clean_text = _text(value.get("clean_text"), field="clean_text")
        typo_text = _text(value.get("typo_text"), field="typo_text")
        if (
            value.get("clean_sha256") != hashlib.sha256(clean_text.encode()).hexdigest()
            or value.get("typo_sha256") != hashlib.sha256(typo_text.encode()).hexdigest()
        ):
            raise ValueError("evaluation natural corpus content hash differs")
        edits = (
            infer_single_word_typo_edit(
                clean_text,
                typo_text,
                operation=_text(value.get("operation"), field="operation"),
            ),
        )
    return EvaluationCorpusRecord(
        record_id=record_id,
        kind=str(kind),
        source=_text(value.get("source"), field="source"),
        source_revision=revision,
        source_split=_text(value.get("source_split"), field="source_split"),
        source_id=_text(value.get("source_id"), field="source_id"),
        group_id=_text(value.get("group_id"), field="group_id"),
        role=expected_role,
        clean_text=clean_text,
        typo_text=typo_text,
        edits=edits,
        metadata=MappingProxyType(dict(metadata)),
    )


def load_evaluation_corpus_bundle(
    root: Path,
    *,
    evaluation_role: str,
    study_protocol_sha256: str,
    access_binding_sha256: str,
    experiment_binding_sha256: str,
    output_dir: Path,
    confirm_sealed_role: bool,
    resume: bool,
) -> EvaluationCorpusBundle:
    """Claim one role and load its hash-bound corpus artifact."""

    if evaluation_role not in _ROLES:
        raise ValueError("evaluation corpus role is unsupported")
    if (
        _SHA64.fullmatch(access_binding_sha256) is None
        or _SHA64.fullmatch(experiment_binding_sha256) is None
    ):
        raise ValueError("evaluation corpus access identity differs")
    resolved = Path(root).resolve()
    registry, _registry_path = _frozen_registry(
        resolved,
        study_protocol_sha256=study_protocol_sha256,
    )
    roles = registry.get("roles")
    expected_role = _ROLES[evaluation_role][1]
    role = roles.get(expected_role) if isinstance(roles, Mapping) else None
    if not isinstance(role, Mapping):
        raise ValueError("evaluation corpus role is missing")
    expected_filename = f"{expected_role}_corpus_manifest.jsonl"
    if role.get("corpus_artifact") != expected_filename:
        raise ValueError("evaluation corpus artifact name differs")
    path = resolved / expected_filename
    digest = _sha256_file(path) if path.is_file() else None
    artifacts = registry.get("artifact_sha256")
    if (
        digest != role.get("corpus_sha256")
        or not isinstance(artifacts, Mapping)
        or digest != artifacts.get(expected_filename)
    ):
        raise ValueError("evaluation corpus artifact hash differs")
    _claim_evaluation_role(
        resolved,
        role=evaluation_role,
        artifact_kind="corpus",
        binding=access_binding_sha256,
        experiment_binding=experiment_binding_sha256,
        output_dir=Path(output_dir),
        confirm=confirm_sealed_role,
        resume=resume,
    )
    records = tuple(_corpus_record(row, expected_role=expected_role) for row in _rows(path))
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("evaluation corpus role contains duplicate record IDs")
    return EvaluationCorpusBundle(
        root=resolved,
        evaluation_role=evaluation_role,
        records=records,
        manifest_path=path,
        manifest_sha256=str(digest),
    )


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
    experiment_binding: str,
    output_dir: Path,
    confirm: bool,
    resume: bool,
    artifact_kind: str = "task",
) -> None:
    if artifact_kind not in {"task", "corpus"}:
        raise ValueError("evaluation artifact kind is unsupported")
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
            if gate.get("experiment_binding_sha256") != experiment_binding:
                raise ValueError("final-test candidate differs from the passing pre-PR gate")
        existing = roles.get(key)
        expected_identity = {
            "access_binding_sha256": binding,
            "experiment_binding_sha256": experiment_binding,
            "output_dir": str(output_dir.resolve()),
        }
        if existing is not None:
            if not isinstance(existing, Mapping) or any(
                existing.get(field) != value for field, value in expected_identity.items()
            ):
                raise ValueError(f"sealed evaluation role {role} was already opened")
            if existing.get("status") != "opened":
                raise ValueError(f"sealed evaluation role {role} was already completed")
            raw_claimed = existing.get("claimed_artifacts", ["task"])
            if (
                not isinstance(raw_claimed, list)
                or any(item not in {"task", "corpus"} for item in raw_claimed)
                or len(set(raw_claimed)) != len(raw_claimed)
            ):
                raise ValueError("sealed evaluation artifact claims differ")
            if artifact_kind in raw_claimed:
                if not resume:
                    raise ValueError(f"sealed evaluation role {role} was already opened")
                return
            roles[key] = {
                **dict(existing),
                "claimed_artifacts": sorted((*raw_claimed, artifact_kind)),
            }
            _atomic_json(access_path, registry)
            return
        if resume:
            raise ValueError("sealed evaluation --resume has no matching access record")
        roles[key] = {
            **expected_identity,
            "status": "opened",
            "report_sha256": None,
            "gate_passed": None,
            "claimed_artifacts": [artifact_kind],
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
            or entry.get("status") != "opened"
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
    digest = _sha256_file(path)
    manifest_digest = digest if name == "evaluation_manifest.json" else artifacts.get(name)
    if output.get("sha256") != digest or manifest_digest != digest:
        raise ValueError(f"evaluation data {name} hash differs")
    return digest


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


def _load_frozen_evaluation_bundle(
    root: Path,
    *,
    evaluation_role: str,
    requested_splits: tuple[str, ...],
    access_binding_sha256: str,
    experiment_binding_sha256: str,
    output_dir: Path,
    confirm_sealed_role: bool,
    resume: bool,
    study_protocol_sha256: str,
    expected_data_identity_sha256: str | None,
) -> EvaluationDataBundle:
    registry, registry_path = _frozen_registry(
        root,
        study_protocol_sha256=study_protocol_sha256,
    )
    roles = registry.get("roles")
    expected_role = _ROLES[evaluation_role][1]
    role = roles.get(expected_role) if isinstance(roles, Mapping) else None
    if not isinstance(role, Mapping):
        raise ValueError("frozen evaluation role is missing")
    filename = role.get("task_artifact")
    declared_sha = role.get("task_sha256")
    if not isinstance(filename, str) or filename != _ROLES[evaluation_role][0]:
        raise ValueError("frozen evaluation task artifact name differs")
    manifest_path = root / filename
    manifest_sha = _sha256_file(manifest_path) if manifest_path.is_file() else None
    artifacts = registry.get("artifact_sha256")
    if (
        not isinstance(artifacts, Mapping)
        or declared_sha != manifest_sha
        or artifacts.get(filename) != manifest_sha
    ):
        raise ValueError("frozen evaluation task artifact hash differs")
    identity = registry.get("data_identity_sha256")
    if not isinstance(identity, str) or _SHA64.fullmatch(identity) is None:
        raise ValueError("frozen evaluation data identity differs")
    if expected_data_identity_sha256 is not None and identity != expected_data_identity_sha256:
        raise ValueError("evaluation data identity differs from the trained adapters")
    _claim_evaluation_role(
        root,
        role=evaluation_role,
        artifact_kind="task",
        binding=access_binding_sha256,
        experiment_binding=experiment_binding_sha256,
        output_dir=output_dir,
        confirm=confirm_sealed_role,
        resume=resume,
    )
    held_out = ("adjacent-transposition",)
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
    if evaluation_role != "tune":
        observed_strata = {stratum for row in parsed for stratum in row.strata}
        expected_task_strata = set(_SPLITS) - {"unseen-content"}
        if any(not row.strata for row in parsed) or observed_strata != expected_task_strata:
            raise ValueError("sealed evaluation manifest lacks the complete frozen strata")
    selected = tuple(
        sorted(
            (row for row in parsed if set(row.strata) & set(requested_splits)),
            key=lambda row: row.record_id,
        )
    )
    if not selected:
        raise ValueError("evaluation requested splits select no records")
    return EvaluationDataBundle(
        root=root,
        evaluation_role=evaluation_role,
        records=selected,
        manifest_path=manifest_path,
        manifest_sha256=str(manifest_sha),
        evaluation_manifest_sha256=_sha256_file(registry_path),
        data_identity_sha256=identity,
        held_out_operations=held_out,
    )


def load_evaluation_bundle(
    root: Path,
    *,
    evaluation_role: str,
    splits: Sequence[str],
    model: str,
    model_revision: str,
    access_binding_sha256: str,
    experiment_binding_sha256: str,
    output_dir: Path,
    confirm_sealed_role: bool,
    resume: bool,
    expected_data_identity_sha256: str | None = None,
    study_protocol_sha256: str | None = None,
) -> EvaluationDataBundle:
    """Validate one role and return only the requested overlapping strata.

    Frozen text is intentionally model-independent: ``model`` and
    ``model_revision`` retain the legacy-call signature and the revision format
    check, while the evaluation runner binds the concrete model/config in its
    experiment and access hashes. The frozen registry instead binds source
    revisions, the study protocol, and every realized text artifact.
    """

    if evaluation_role not in _ROLES:
        raise ValueError("evaluation role is unsupported")
    requested = tuple(splits)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(split not in _SPLITS for split in requested)
    ):
        raise ValueError("evaluation splits must be unique members of the frozen inventory")
    if (
        not isinstance(model, str)
        or not model
        or _SHA40.fullmatch(model_revision) is None
        or _SHA64.fullmatch(access_binding_sha256) is None
        or _SHA64.fullmatch(experiment_binding_sha256) is None
    ):
        raise ValueError("evaluation model or access identity differs")
    if evaluation_role != "tune" and requested != _SPLITS:
        raise ValueError("sealed evaluation roles require the complete frozen split inventory")
    resolved = Path(root).resolve()
    if (resolved / "registry.json").is_file():
        if study_protocol_sha256 is None:
            raise ValueError("frozen evaluation requires the study protocol hash")
        return _load_frozen_evaluation_bundle(
            resolved,
            evaluation_role=evaluation_role,
            requested_splits=requested,
            access_binding_sha256=access_binding_sha256,
            experiment_binding_sha256=experiment_binding_sha256,
            output_dir=Path(output_dir),
            confirm_sealed_role=confirm_sealed_role,
            resume=resume,
            study_protocol_sha256=study_protocol_sha256,
            expected_data_identity_sha256=expected_data_identity_sha256,
        )
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
        artifact_kind="task",
        binding=access_binding_sha256,
        experiment_binding=experiment_binding_sha256,
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
    if evaluation_role != "tune":
        observed_strata = {stratum for row in parsed for stratum in row.strata}
        if any(not row.strata for row in parsed) or observed_strata != set(_SPLITS):
            raise ValueError("sealed evaluation manifest lacks the complete frozen strata")
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
    "EvaluationCorpusBundle",
    "EvaluationCorpusRecord",
    "EvaluationDataBundle",
    "EvaluationPair",
    "FrozenEvaluationProvenance",
    "complete_evaluation_role",
    "load_evaluation_corpus_bundle",
    "load_evaluation_bundle",
    "load_frozen_evaluation_provenance",
]

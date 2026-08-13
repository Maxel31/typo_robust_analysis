"""Materialize model-independent evaluation text before training iteration."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typo_robust_training.data.builder import (
    DataSourceProvider,
    collect_sources,
    source_collection_limit,
)
from typo_robust_training.data.config import (
    TrainingDataProtocol,
    load_training_data_config,
    strict_loads,
)
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.natural_words import natural_dictionary_role_for_word
from typo_robust_training.data.records import (
    CleanRecord,
    NaturalTypoRecord,
    TypoEdit,
    infer_single_word_typo_edit,
)
from typo_robust_training.data.splits import NearDuplicateTextIndex, assign_balanced_group_roles
from typo_robust_training.data.task_splits import (
    FROZEN_TASK_EVALUATION_SPLITS,
    REASONING_TRAINING_SPLITS,
)
from typo_robust_training.evaluation.perturb import (
    FrozenEvaluationTypo,
    NoNaturalInjectionTargetError,
    evaluation_eligible_word_spans,
    freeze_natural_injection_dictionary,
    generate_evaluation_typo,
    generate_natural_injection,
)
from typo_robust_training.evaluation.study import (
    EvaluationStudyProtocol,
    load_evaluation_study_protocol,
)
from typo_robust_training.integrity import sha256_file as _sha256_file


_TASK_TUNE_SPLITS = REASONING_TRAINING_SPLITS
_TASK_EVALUATION_SPLITS = FROZEN_TASK_EVALUATION_SPLITS
_ARTIFACTS = (
    "tune_manifest.jsonl",
    "pre_pr_gate_manifest.jsonl",
    "final_test_manifest.jsonl",
    "tune_corpus_manifest.jsonl",
    "pre_pr_gate_corpus_manifest.jsonl",
    "final_test_corpus_manifest.jsonl",
)


@dataclass(frozen=True, slots=True)
class FreezeEvaluationRunConfig:
    protocol_path: Path
    source_config_path: Path
    exclude_data_dir: Path
    output_dir: Path

    def __post_init__(self) -> None:
        for field in (
            "protocol_path",
            "source_config_path",
            "exclude_data_dir",
            "output_dir",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class FreezeEvaluationResult:
    registry_path: Path
    run_path: Path


@dataclass(frozen=True, slots=True)
class _Exclusions:
    hard_source_ids: frozenset[str]
    hard_groups: frozenset[tuple[str, str]]
    prior_tune_source_ids: frozenset[str]
    prior_tune_groups: frozenset[tuple[str, str]]
    hard_near_duplicates: NearDuplicateTextIndex
    prior_tune_near_duplicates: NearDuplicateTextIndex
    training_repositories: frozenset[str]
    tune_repositories: frozenset[str]
    artifact_sha256: Mapping[str, str]
    data_protocol_sha256: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


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


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in read_lf_jsonl_lines(
        path,
        context="evaluation exclusion artifact",
    ):
        row = strict_loads(line, context=f"{path}:{line_number}")
        if not isinstance(row, Mapping):
            raise ValueError(f"evaluation exclusion row must be an object: {path}:{line_number}")
        rows.append(row)
    return tuple(rows)


def _exclusions(root: Path) -> _Exclusions:
    resolved = root.resolve()
    training_path = resolved / "training_sources.jsonl"
    diagnostic_path = resolved / "diagnostic_manifest.jsonl"
    tune_path = resolved / "tune_manifest.jsonl"
    training = _jsonl(training_path)
    diagnostic = _jsonl(diagnostic_path)
    tune = _jsonl(tune_path)
    data_run_path = resolved / "run.json"
    data_run = strict_loads(
        data_run_path.read_text(encoding="utf-8"),
        context=str(data_run_path),
    )
    if not isinstance(data_run, Mapping) or data_run.get("status") != "completed":
        raise ValueError("evaluation exclusions require a completed data-build run")
    data_protocol_sha256 = data_run.get("protocol_sha256")
    if not isinstance(data_protocol_sha256, str) or len(data_protocol_sha256) != 64:
        raise ValueError("evaluation exclusion run has no protocol SHA-256")

    def source_id(row: Mapping[str, object]) -> str:
        value = row.get("source_id")
        if not isinstance(value, str) or not value:
            raise ValueError("evaluation exclusion row has no source_id")
        return value

    def group(row: Mapping[str, object]) -> tuple[str, str]:
        source, group_id = row.get("source"), row.get("group_id")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(group_id, str)
            or not group_id
        ):
            raise ValueError("evaluation exclusion row has no source/group identity")
        return source, group_id

    def repositories(rows: Sequence[Mapping[str, object]]) -> frozenset[str]:
        values = {
            str(row["repository"])
            for row in rows
            if isinstance(row.get("repository"), str) and row["repository"]
        }
        return frozenset(values)

    def clean_texts(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
        texts: list[str] = []
        for row in rows:
            value = row.get("text")
            if not isinstance(value, str) or not value:
                value = row.get("clean_text")
            if not isinstance(value, str) or not value:
                raise ValueError("evaluation exclusion row has no clean text")
            texts.append(value)
        return tuple(texts)

    return _Exclusions(
        hard_source_ids=frozenset(source_id(row) for row in (*training, *diagnostic)),
        hard_groups=frozenset(group(row) for row in (*training, *diagnostic)),
        prior_tune_source_ids=frozenset(source_id(row) for row in tune),
        prior_tune_groups=frozenset(group(row) for row in tune),
        hard_near_duplicates=NearDuplicateTextIndex(clean_texts((*training, *diagnostic))),
        prior_tune_near_duplicates=NearDuplicateTextIndex(clean_texts(tune)),
        training_repositories=repositories(training),
        tune_repositories=repositories(tune),
        artifact_sha256={
            path.name: _sha256_file(path)
            for path in (training_path, diagnostic_path, tune_path, data_run_path)
        },
        data_protocol_sha256=data_protocol_sha256,
    )


def _order_key(record: CleanRecord | NaturalTypoRecord, *, namespace: str, seed: int) -> str:
    return hashlib.sha256(f"{namespace}\0{seed}\0{record.record_id}".encode("utf-8")).hexdigest()


def _retain_smallest(
    heap: list[tuple[int, str, CleanRecord]],
    record: CleanRecord,
    *,
    key: int,
    limit: int,
) -> None:
    entry = (-key, record.record_id, record)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _could_retain_smallest(
    heap: Sequence[tuple[int, str, CleanRecord]],
    *,
    record_id: str,
    key: int,
    limit: int,
) -> bool:
    """Return whether an item can enter a bounded bottom-k heap."""

    if limit < 0:
        raise ValueError("bottom-k limit must be non-negative")
    if limit == 0:
        return False
    if len(heap) < limit:
        return True
    return (-key, record_id) > heap[0][:2]


def _collect_evaluation_sources(
    sources: TrainingDataProtocol,
    source_provider: DataSourceProvider,
    *,
    protocol: EvaluationStudyProtocol,
    exclusions: _Exclusions,
) -> dict[str, tuple[CleanRecord | NaturalTypoRecord, ...]]:
    """Scan the source universe while retaining only bounded FineWeb candidates."""

    name = "fineweb_edu"
    source = sources.sources[name]
    tune_limit = protocol.corpus_counts["tune"][name]
    sealed_count = (
        protocol.corpus_counts["pre_pr_gate"][name] + protocol.corpus_counts["final_test"][name]
    )
    sealed_limit = sealed_count + tune_limit
    tune_heap: list[tuple[int, str, CleanRecord]] = []
    sealed_heap: list[tuple[int, str, CleanRecord]] = []
    split_counts: dict[str, int] = defaultdict(int)
    per_split_limit = source_collection_limit(name, sources)
    fineweb_ids: set[str] = set()
    iterator = iter(source_provider.iter_records(name, source))
    try:
        for raw_record in iterator:
            if not isinstance(raw_record, CleanRecord):
                raise TypeError("FineWeb evaluation source emitted a non-clean record")
            if raw_record.source != name or raw_record.source_revision != source.revision:
                raise ValueError("FineWeb evaluation source provenance differs")
            if raw_record.source_split not in source.splits:
                raise ValueError("FineWeb evaluation source emitted an undeclared split")
            if raw_record.record_id in fineweb_ids:
                raise ValueError("FineWeb evaluation source duplicated record identity")
            if split_counts[raw_record.source_split] >= per_split_limit:
                continue
            fineweb_ids.add(raw_record.record_id)
            split_counts[raw_record.source_split] += 1
            tune_digest = int(
                _order_key(
                    raw_record,
                    namespace="evaluation-corpus-fineweb_edu-tune/v1",
                    seed=protocol.seed,
                ),
                16,
            )
            tune_priority = 0 if raw_record.source_id in exclusions.prior_tune_source_ids else 1
            tune_key = (tune_priority << 256) | tune_digest
            if _could_retain_smallest(
                tune_heap,
                record_id=raw_record.record_id,
                key=tune_key,
                limit=tune_limit,
            ) and _eligible_clean(raw_record, exclusions=exclusions, sealed=False):
                _retain_smallest(
                    tune_heap,
                    raw_record,
                    key=tune_key,
                    limit=tune_limit,
                )
            sealed_key = int(
                _order_key(
                    raw_record,
                    namespace="evaluation-corpus-fineweb_edu-sealed/v1",
                    seed=protocol.seed,
                ),
                16,
            )
            if _could_retain_smallest(
                sealed_heap,
                record_id=raw_record.record_id,
                key=sealed_key,
                limit=sealed_limit,
            ) and _eligible_clean(raw_record, exclusions=exclusions, sealed=True):
                _retain_smallest(
                    sealed_heap,
                    raw_record,
                    key=sealed_key,
                    limit=sealed_limit,
                )
            if all(split_counts[split] >= per_split_limit for split in source.splits):
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if len(tune_heap) != tune_limit or len(sealed_heap) < sealed_count:
        raise ValueError("FineWeb evaluation source has insufficient disjoint candidates")
    retained = {record.record_id: record for _key, _record_id, record in (*tune_heap, *sealed_heap)}
    collected = collect_sources(
        sources,
        source_provider,
        source_names=tuple(source_name for source_name in sources.sources if source_name != name),
    )
    if any(record.record_id in fineweb_ids for records in collected.values() for record in records):
        raise ValueError("evaluation source identities cross configured sources")
    collected[name] = tuple(retained[record_id] for record_id in sorted(retained))
    return collected


def _task_stratum(record: CleanRecord) -> str:
    if record.source == "mmlu":
        return f"subject:{record.metadata.get('subject')}"
    if record.source == "mmlu_pro":
        return f"category:{record.metadata.get('category')}"
    if record.source == "math_500":
        return f"subject:{record.metadata.get('subject')}|level:{record.metadata.get('level')}"
    return "all"


def _stratified_order(
    records: Sequence[CleanRecord],
    *,
    namespace: str,
    seed: int,
) -> tuple[CleanRecord, ...]:
    strata: dict[str, list[CleanRecord]] = defaultdict(list)
    for record in records:
        strata[_task_stratum(record)].append(record)
    for rows in strata.values():
        rows.sort(key=lambda item: _order_key(item, namespace=namespace, seed=seed))
    ordered: list[CleanRecord] = []
    depth = 0
    while True:
        added = False
        for stratum in sorted(strata):
            if depth < len(strata[stratum]):
                ordered.append(strata[stratum][depth])
                added = True
        if not added:
            return tuple(ordered)
        depth += 1


def _eligible_clean(
    record: CleanRecord,
    *,
    exclusions: _Exclusions,
    sealed: bool,
) -> bool:
    identity = (record.source, record.group_id)
    if (
        record.source_id in exclusions.hard_source_ids
        or identity in exclusions.hard_groups
        or exclusions.hard_near_duplicates.contains_near_duplicate(record.text)
    ):
        return False
    if sealed and (
        record.source_id in exclusions.prior_tune_source_ids
        or identity in exclusions.prior_tune_groups
        or exclusions.prior_tune_near_duplicates.contains_near_duplicate(record.text)
    ):
        return False
    return True


def _supports_frozen_typo_grid(
    record: CleanRecord,
    *,
    protocol: EvaluationStudyProtocol,
) -> bool:
    """Require enough valid question words for every frozen text-level condition."""

    try:
        spans = evaluation_eligible_word_spans(
            record,
            minimum_word_letters=protocol.minimum_word_letters,
        )
    except (TypeError, ValueError):
        return False
    required = max((protocol.primary_edit_count, *protocol.severity_edit_counts))
    transposable = sum(
        any(left != right for left, right in zip(word, word[1:]))
        for start, stop in spans
        if (word := record.text[start:stop])
    )
    return len(spans) >= required and transposable >= 2


def _select_task_items(
    collected: Mapping[str, Sequence[CleanRecord | NaturalTypoRecord]],
    *,
    protocol: EvaluationStudyProtocol,
    exclusions: _Exclusions,
) -> dict[str, dict[str, tuple[CleanRecord, ...]]]:
    selected: dict[str, dict[str, tuple[CleanRecord, ...]]] = {
        "tune": {},
        "pre_pr_gate": {},
        "final_test": {},
    }
    tune_tasks = tuple(sorted(_TASK_TUNE_SPLITS))
    tune_base, tune_remainder = divmod(protocol.tune_task_records_total, len(tune_tasks))
    for task_index, source in enumerate(tune_tasks):
        splits = _TASK_TUNE_SPLITS[source]
        candidates = [
            record
            for record in collected[source]
            if isinstance(record, CleanRecord)
            and record.source_split in splits
            and _eligible_clean(record, exclusions=exclusions, sealed=False)
            and _supports_frozen_typo_grid(record, protocol=protocol)
        ]
        candidates.sort(
            key=lambda record: (
                record.source_id not in exclusions.prior_tune_source_ids,
                _order_key(
                    record,
                    namespace=f"evaluation-tune-task-{source}/v1",
                    seed=protocol.seed,
                ),
            )
        )
        count = tune_base + (1 if task_index < tune_remainder else 0)
        rows = tuple(candidates[:count])
        if len(rows) != count:
            raise ValueError(
                f"evaluation tune task {source} has {len(rows)} records but requires {count}"
            )
        selected["tune"][source] = rows

    sealed_by_task: dict[str, tuple[CleanRecord, ...]] = {}
    for task, splits in _TASK_EVALUATION_SPLITS.items():
        candidates = tuple(
            record
            for record in collected[task]
            if isinstance(record, CleanRecord)
            and record.source_split in splits
            and _eligible_clean(record, exclusions=exclusions, sealed=True)
            and _supports_frozen_typo_grid(record, protocol=protocol)
        )
        sealed_by_task[task] = _stratified_order(
            candidates,
            namespace=f"evaluation-sealed-task-{task}/v1",
            seed=protocol.seed,
        )

    for task in protocol.role_tasks["pre_pr_gate"]:
        count = protocol.records_per_task["pre_pr_gate"][task]
        rows = sealed_by_task[task][:count]
        if len(rows) != count:
            raise ValueError(f"evaluation pre-PR task {task} has fewer than {count} records")
        selected["pre_pr_gate"][task] = rows
    for task in protocol.role_tasks["final_test"]:
        count = protocol.records_per_task["final_test"][task]
        offset = (
            protocol.records_per_task["pre_pr_gate"][task]
            if task in protocol.role_tasks["pre_pr_gate"]
            else 0
        )
        rows = sealed_by_task[task][offset : offset + count]
        if len(rows) != count:
            raise ValueError(f"evaluation final task {task} has fewer than {count} records")
        selected["final_test"][task] = rows
    return selected


def _pair_payload(
    record: CleanRecord,
    typo: FrozenEvaluationTypo,
    *,
    role: str,
    primary: bool,
    mechanistic_audit: bool = False,
) -> dict[str, object]:
    edits = typo.edits
    if not edits:
        raise ValueError("frozen evaluation pair cannot serialize a no-op")
    metadata = {
        **dict(record.metadata),
        **dict(typo.metadata),
        "evaluation_primary": primary,
        "mechanistic_audit": mechanistic_audit,
    }
    return {
        "schema_version": "robustness-fixed-typo-pair/v1",
        "kind": "synthetic",
        "record_id": typo.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": role,
        "clean_text": record.text,
        "typo_text": typo.typo_text,
        "task": record.task,
        "answer": record.answer,
        "metadata": metadata,
        "operation": edits[0].operation if len(edits) == 1 else "multiple",
        "operations": [edit.operation for edit in edits],
        "edit_count": len(edits),
        "generator_seed": typo.seed,
        "generator_variant": typo.variant,
        "edits": [
            {
                "operation": edit.operation,
                "clean_word": edit.clean_word,
                "typo_word": edit.typo_word,
                "clean_char_span": list(edit.clean_char_span),
                "typo_char_span": list(edit.typo_char_span),
            }
            for edit in edits
        ],
    }


def _flat_task_items(
    rows_by_task: Mapping[str, Sequence[CleanRecord]],
) -> tuple[CleanRecord, ...]:
    return tuple(record for task in sorted(rows_by_task) for record in rows_by_task[task])


def _balanced_subset(
    rows_by_task: Mapping[str, Sequence[CleanRecord]],
    *,
    total: int,
    namespace: str,
    seed: int,
) -> tuple[CleanRecord, ...]:
    tasks = tuple(sorted(rows_by_task))
    base, remainder = divmod(total, len(tasks))
    selected: list[CleanRecord] = []
    for index, task in enumerate(tasks):
        count = base + (index < remainder)
        ordered = sorted(
            rows_by_task[task],
            key=lambda record: _order_key(record, namespace=namespace, seed=seed),
        )
        if len(ordered) < count:
            raise ValueError(f"evaluation secondary subset {task} has insufficient records")
        selected.extend(ordered[:count])
    return tuple(selected)


def _natural_edit(
    record: NaturalTypoRecord,
    *,
    minimum_word_letters: int,
) -> TypoEdit | None:
    try:
        edit = infer_single_word_typo_edit(
            record.clean_text,
            record.typo_text,
            operation=record.operation,
        )
    except ValueError:
        return None
    if (
        len(edit.clean_word) < minimum_word_letters
        or not edit.clean_word.isascii()
        or not edit.clean_word.isalpha()
        or not edit.typo_word.isascii()
        or not edit.typo_word.isalpha()
        or edit.clean_word.casefold() == edit.typo_word.casefold()
    ):
        return None
    return edit


def _select_natural(
    records: Sequence[CleanRecord | NaturalTypoRecord],
    *,
    protocol: EvaluationStudyProtocol,
    exclusions: _Exclusions,
    dictionary_word_split: Mapping[str, float] | None = None,
    dictionary_word_seed: int = 42,
) -> tuple[
    dict[str, tuple[NaturalTypoRecord, ...]],
    dict[str, dict[str, tuple[str, ...]]],
]:
    natural = tuple(record for record in records if isinstance(record, NaturalTypoRecord))
    valid = tuple(
        (record, edit)
        for record in natural
        if (
            edit := _natural_edit(
                record,
                minimum_word_letters=protocol.minimum_word_letters,
            )
        )
        is not None
    )
    edits_by_record_id = {record.record_id: edit for record, edit in valid}
    dictionary_roles_by_record_id = (
        {
            record.record_id: natural_dictionary_role_for_word(
                edit.clean_word.casefold(),
                seed=dictionary_word_seed,
                weights=dictionary_word_split,
            )
            for record, edit in valid
        }
        if dictionary_word_split is not None
        else {}
    )
    training_words = (
        {
            edit.clean_word.casefold()
            for record, edit in valid
            if record.repository in exclusions.training_repositories
        }
        if dictionary_word_split is None
        else set()
    )
    tune = tuple(
        sorted(
            (
                record
                for record, _ in valid
                if record.repository in exclusions.tune_repositories
                and record.repository not in exclusions.training_repositories
            ),
            key=lambda record: (
                record.source_id not in exclusions.prior_tune_source_ids,
                _order_key(record, namespace="evaluation-natural-tune/v1", seed=protocol.seed),
            ),
        )[: protocol.corpus_counts["tune"]["natural_pairs"]]
    )
    if len(tune) != protocol.corpus_counts["tune"]["natural_pairs"]:
        raise ValueError("evaluation natural tune pool is too small")
    tune_words = (
        {
            edit.clean_word.casefold()
            for record, edit in valid
            if record.repository in exclusions.tune_repositories
            and edit.clean_word.casefold() not in training_words
        }
        if dictionary_word_split is None
        else set()
    )
    candidates = tuple(
        record
        for record, _ in valid
        if record.repository not in exclusions.training_repositories
        and record.repository not in exclusions.tune_repositories
        and record.source_id not in exclusions.prior_tune_source_ids
    )
    by_repository: dict[str, list[NaturalTypoRecord]] = defaultdict(list)
    for record in candidates:
        by_repository[record.repository].append(record)
    pre_count = protocol.corpus_counts["pre_pr_gate"]["natural_pairs"]
    final_count = protocol.corpus_counts["final_test"]["natural_pairs"]
    total = pre_count + final_count
    assignments = assign_balanced_group_roles(
        {repository: len(rows) for repository, rows in by_repository.items()},
        seed=protocol.seed,
        namespace="evaluation-natural-sealed-repositories/v1",
        weights={"pre_pr_gate": pre_count / total, "final_test": final_count / total},
    )
    selected: dict[str, tuple[NaturalTypoRecord, ...]] = {"tune": tune}
    dictionary_records: dict[str, tuple[NaturalTypoRecord, ...]] = {
        "tune": tuple(
            record
            for record, edit in valid
            if (
                (
                    record.repository in exclusions.tune_repositories
                    and edit.clean_word.casefold() not in training_words
                )
                if dictionary_word_split is None
                else dictionary_roles_by_record_id[record.record_id] == "tune"
            )
        )
    }
    for role, count in (("pre_pr_gate", pre_count), ("final_test", final_count)):
        ordered = sorted(
            (record for record in candidates if assignments.get(record.repository) == role),
            key=lambda record: _order_key(
                record, namespace=f"evaluation-natural-{role}/v1", seed=protocol.seed
            ),
        )
        rows = tuple(ordered[:count])
        if len(rows) != count:
            raise ValueError(
                f"evaluation natural {role} has {len(rows)} records but requires {count}"
            )
        selected[role] = rows
        ordered_ids = {record.record_id for record in ordered}
        dictionary_records[role] = tuple(
            record
            for record, edit in valid
            if (
                (
                    record.record_id in ordered_ids
                    and edit.clean_word.casefold() not in training_words
                    and edit.clean_word.casefold() not in tune_words
                )
                if dictionary_word_split is None
                else dictionary_roles_by_record_id[record.record_id] == role
            )
        )

    dictionaries: dict[str, dict[str, tuple[str, ...]]] = {}
    for role, rows in dictionary_records.items():
        variants: dict[str, set[str]] = defaultdict(set)
        for record in rows:
            edit = edits_by_record_id[record.record_id]
            variants[edit.clean_word.casefold()].add(edit.typo_word.casefold())
        dictionaries[role] = {
            word: tuple(sorted(typos)) for word, typos in sorted(variants.items())
        }
    dictionary_roles = tuple(dictionaries)
    for index, left in enumerate(dictionary_roles):
        for right in dictionary_roles[index + 1 :]:
            if set(dictionaries[left]) & set(dictionaries[right]):
                raise ValueError("natural injection word crosses evaluation roles")
    return selected, dictionaries


def _natural_payload(record: NaturalTypoRecord, *, role: str) -> dict[str, object]:
    return {
        "schema_version": "robustness-natural-pair/v1",
        "kind": "natural",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": role,
        "clean_text": record.clean_text,
        "typo_text": record.typo_text,
        "task": None,
        "answer": None,
        "operation": record.operation,
        "training_eligible": record.training_eligible,
        "repository": record.repository,
        "repository_license": record.repository_license,
        "clean_sha256": hashlib.sha256(record.clean_text.encode()).hexdigest(),
        "typo_sha256": hashlib.sha256(record.typo_text.encode()).hexdigest(),
        "metadata": {**dict(record.metadata), "evaluation_condition": "natural-lm-pair"},
    }


def _corpus_payload(record: CleanRecord, *, role: str) -> dict[str, object]:
    return {
        "schema_version": "robustness-evaluation-corpus-record/v1",
        "kind": "clean-corpus",
        "record_id": record.record_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_split": record.source_split,
        "source_id": record.source_id,
        "group_id": record.group_id,
        "split": role,
        "text": record.text,
        "content_sha256": hashlib.sha256(record.text.encode()).hexdigest(),
        "metadata": dict(record.metadata),
    }


def _select_corpus(
    collected: Mapping[str, Sequence[CleanRecord | NaturalTypoRecord]],
    *,
    protocol: EvaluationStudyProtocol,
    exclusions: _Exclusions,
) -> dict[str, dict[str, tuple[CleanRecord, ...]]]:
    output: dict[str, dict[str, tuple[CleanRecord, ...]]] = {
        "tune": {},
        "pre_pr_gate": {},
        "final_test": {},
    }
    for source in ("fineweb_edu", "dolma"):
        candidates = tuple(
            record
            for record in collected[source]
            if isinstance(record, CleanRecord)
            and _eligible_clean(record, exclusions=exclusions, sealed=False)
        )
        tune_count = protocol.corpus_counts["tune"][source]
        tune_ordered = sorted(
            candidates,
            key=lambda record: (
                record.source_id not in exclusions.prior_tune_source_ids,
                _order_key(
                    record,
                    namespace=f"evaluation-corpus-{source}-tune/v1",
                    seed=protocol.seed,
                ),
            ),
        )
        tune_rows = tuple(tune_ordered[:tune_count])
        if len(tune_rows) != tune_count:
            raise ValueError(f"evaluation corpus {source}/tune is too small")
        output["tune"][source] = tune_rows
        selected_tune_ids = {record.source_id for record in tune_rows}
        selected_tune_groups = {(record.source, record.group_id) for record in tune_rows}
        sealed = sorted(
            (
                record
                for record in candidates
                if _eligible_clean(record, exclusions=exclusions, sealed=True)
                and record.source_id not in exclusions.prior_tune_source_ids
                and (record.source, record.group_id) not in exclusions.prior_tune_groups
                and record.source_id not in selected_tune_ids
                and (record.source, record.group_id) not in selected_tune_groups
            ),
            key=lambda record: _order_key(
                record, namespace=f"evaluation-corpus-{source}-sealed/v1", seed=protocol.seed
            ),
        )
        offset = 0
        for role in ("pre_pr_gate", "final_test"):
            count = protocol.corpus_counts[role][source]
            rows = tuple(sealed[offset : offset + count])
            if len(rows) != count:
                raise ValueError(f"evaluation corpus {source}/{role} is too small")
            output[role][source] = rows
            offset += count
    return output


def _task_rows(
    selected: Mapping[str, Mapping[str, Sequence[CleanRecord]]],
    natural_dictionaries: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    protocol: EvaluationStudyProtocol,
) -> dict[str, list[dict[str, object]]]:
    outputs: dict[str, list[dict[str, object]]] = {}
    for role in ("tune", "pre_pr_gate", "final_test"):
        by_task = selected[role]
        primary_items = _flat_task_items(by_task)
        natural_dictionary = freeze_natural_injection_dictionary(
            natural_dictionaries[role],
            minimum_word_letters=protocol.minimum_word_letters,
        )
        audit_ids = {
            record.record_id
            for record in _balanced_subset(
                by_task,
                total=protocol.audit_records[role],
                namespace=f"evaluation-{role}-mechanistic-audit/v1",
                seed=protocol.seed,
            )
        }
        rows: list[dict[str, object]] = []
        for variant, record in enumerate(primary_items):
            typo = generate_evaluation_typo(
                record,
                condition=protocol.primary_typo_condition,
                edit_count=protocol.primary_edit_count,
                operations=protocol.primary_operations,
                seed=protocol.seed,
                role=role,
                variant=variant,
                minimum_word_letters=protocol.minimum_word_letters,
            )
            rows.append(
                _pair_payload(
                    record,
                    typo,
                    role=role,
                    primary=True,
                    mechanistic_audit=record.record_id in audit_ids,
                )
            )
        if role != "tune":
            severity_items = _balanced_subset(
                by_task,
                total=protocol.severity_records_total,
                namespace=f"evaluation-{role}-severity/v1",
                seed=protocol.seed,
            )
            for edit_count in protocol.severity_edit_counts:
                for variant, record in enumerate(severity_items):
                    typo = generate_evaluation_typo(
                        record,
                        condition=f"random-{edit_count}",
                        edit_count=edit_count,
                        operations=protocol.primary_operations,
                        seed=protocol.seed,
                        role=role,
                        variant=variant,
                        minimum_word_letters=protocol.minimum_word_letters,
                    )
                    rows.append(_pair_payload(record, typo, role=role, primary=False))
            transposition_items = _balanced_subset(
                by_task,
                total=protocol.held_out_records_total,
                namespace=f"evaluation-{role}-transposition/v1",
                seed=protocol.seed,
            )
            for variant, record in enumerate(transposition_items):
                typo = generate_evaluation_typo(
                    record,
                    condition=protocol.held_out_condition,
                    edit_count=2,
                    operations=("adjacent-transposition",),
                    seed=protocol.seed,
                    role=role,
                    variant=variant,
                    minimum_word_letters=protocol.minimum_word_letters,
                )
                rows.append(_pair_payload(record, typo, role=role, primary=False))
        natural_required = (
            protocol.tune_natural_injection_records
            if role == "tune"
            else protocol.natural_injection_records_total
        )
        natural_rows: list[dict[str, object]] = []
        for variant, record in enumerate(
            sorted(
                primary_items,
                key=lambda item: _order_key(
                    item, namespace=f"evaluation-{role}-natural-injection/v1", seed=protocol.seed
                ),
            )
        ):
            try:
                typo = generate_natural_injection(
                    record,
                    replacements=natural_dictionary,
                    seed=protocol.seed,
                    role=role,
                    variant=variant,
                    minimum_word_letters=protocol.minimum_word_letters,
                )
            except NoNaturalInjectionTargetError:
                continue
            natural_rows.append(_pair_payload(record, typo, role=role, primary=False))
            if len(natural_rows) == natural_required:
                break
        if len(natural_rows) != natural_required:
            raise ValueError(
                f"evaluation {role} natural injection coverage is "
                f"{len(natural_rows)}/{natural_required}"
            )
        rows.extend(natural_rows)
        rows.sort(
            key=lambda row: (
                str(row["metadata"]["evaluation_condition"]),
                str(row["source"]),
                str(row["record_id"]),
            )
        )
        outputs[role] = rows
    return outputs


def _assert_role_disjointness(
    task_items: Mapping[str, Mapping[str, Sequence[CleanRecord]]],
    corpus_items: Mapping[str, Mapping[str, Sequence[CleanRecord]]],
    natural_items: Mapping[str, Sequence[NaturalTypoRecord]],
) -> None:
    groups: dict[str, set[tuple[str, str]]] = {}
    for role in ("tune", "pre_pr_gate", "final_test"):
        groups[role] = {
            (record.source, record.group_id)
            for rows in task_items[role].values()
            for record in rows
        }
        groups[role].update(
            (record.source, record.group_id)
            for rows in corpus_items[role].values()
            for record in rows
        )
        groups[role].update((record.source, record.group_id) for record in natural_items[role])
    roles = tuple(groups)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if groups[left] & groups[right]:
                raise ValueError(f"evaluation roles {left} and {right} share source groups")


def run_freeze_robustness_evaluation(
    config: FreezeEvaluationRunConfig,
    *,
    source_provider: DataSourceProvider | None = None,
) -> FreezeEvaluationResult:
    """Freeze item identities and typo text without running any model."""

    if not isinstance(config, FreezeEvaluationRunConfig):
        raise TypeError("freeze evaluation config has the wrong type")
    study = load_evaluation_study_protocol(config.protocol_path)
    sources = load_training_data_config(config.source_config_path)
    if sources.natural_dictionary_word_split is None:
        raise ValueError("evaluation freeze requires a v3 source config with corrected-word splits")
    exclusions = _exclusions(config.exclude_data_dir)
    if exclusions.data_protocol_sha256 != sources.config_sha256:
        raise ValueError("evaluation source config differs from the exclusion data build")
    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation freeze output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if source_provider is None:
        from typo_robust_training.data.sources import HuggingFaceDataSourceProvider

        source_provider = HuggingFaceDataSourceProvider(protocol=sources)
    started = _now()
    try:
        collected = _collect_evaluation_sources(
            sources,
            source_provider,
            protocol=study,
            exclusions=exclusions,
        )
        task_items = _select_task_items(
            collected,
            protocol=study,
            exclusions=exclusions,
        )
        natural_items, natural_dictionaries = _select_natural(
            collected["github_typo_corpus"],
            protocol=study,
            exclusions=exclusions,
            dictionary_word_split=sources.natural_dictionary_word_split,
            dictionary_word_seed=sources.seed,
        )
        corpus_items = _select_corpus(
            collected,
            protocol=study,
            exclusions=exclusions,
        )
        _assert_role_disjointness(task_items, corpus_items, natural_items)
        task_rows = _task_rows(task_items, natural_dictionaries, protocol=study)
        corpus_rows: dict[str, list[dict[str, object]]] = {}
        for role in ("tune", "pre_pr_gate", "final_test"):
            rows = [
                _corpus_payload(record, role=role)
                for source in ("fineweb_edu", "dolma")
                for record in corpus_items[role][source]
            ]
            rows.extend(_natural_payload(record, role=role) for record in natural_items[role])
            rows.sort(key=lambda row: (str(row["source"]), str(row["record_id"])))
            corpus_rows[role] = rows
        artifacts = {
            "tune_manifest.jsonl": task_rows["tune"],
            "pre_pr_gate_manifest.jsonl": task_rows["pre_pr_gate"],
            "final_test_manifest.jsonl": task_rows["final_test"],
            "tune_corpus_manifest.jsonl": corpus_rows["tune"],
            "pre_pr_gate_corpus_manifest.jsonl": corpus_rows["pre_pr_gate"],
            "final_test_corpus_manifest.jsonl": corpus_rows["final_test"],
        }
        for name, rows in artifacts.items():
            _atomic_jsonl(output / name, rows)
        artifact_sha = {name: _sha256_file(output / name) for name in _ARTIFACTS}
        role_registry: dict[str, object] = {}
        for role in ("tune", "pre_pr_gate", "final_test"):
            task_name = f"{role}_manifest.jsonl"
            corpus_name = f"{role}_corpus_manifest.jsonl"
            primary = sum(
                row["metadata"]["evaluation_condition"] == study.primary_typo_condition
                for row in task_rows[role]
            )
            role_registry[role] = {
                "task_artifact": task_name,
                "task_sha256": artifact_sha[task_name],
                "task_records": len(task_rows[role]),
                "task_primary_records": primary,
                "corpus_artifact": corpus_name,
                "corpus_sha256": artifact_sha[corpus_name],
                "corpus_records": len(corpus_rows[role]),
                "maximum_openings": 0 if role == "tune" else study.maximum_openings[role],
            }
        registry = {
            "schema_version": "robustness-evaluation-registry/v1",
            "protocol_id": study.protocol_id,
            "protocol_sha256": study.config_sha256,
            "source_config_sha256": sources.config_sha256,
            "exclusion_data_protocol_sha256": exclusions.data_protocol_sha256,
            "source_revisions": {
                name: source.revision for name, source in sorted(sources.sources.items())
            },
            "exclusion_artifact_sha256": dict(exclusions.artifact_sha256),
            "artifact_sha256": artifact_sha,
            "data_identity_sha256": _canonical_sha256(
                {name: [str(row["record_id"]) for row in rows] for name, rows in artifacts.items()}
            ),
            "roles": role_registry,
            "opening_order": ["pre_pr_gate", "final_test"],
            "primary_condition": study.primary_typo_condition,
            "generator_seed": study.seed,
            "generator": "frozen-evaluation-typo/v1",
            "natural_evaluation_axes": {
                "language_model_pairs": "repository-disjoint/v1",
                "task_injection": "corrected-word-disjoint/v1",
                "corrected_word_split": (
                    dict(sources.natural_dictionary_word_split)
                    if sources.natural_dictionary_word_split is not None
                    else None
                ),
            },
        }
        _atomic_json(output / "registry.json", registry)
        _atomic_json(
            output / "run.json",
            {
                "schema_version": "freeze-robustness-evaluation-run/v1",
                "status": "completed",
                "started_at": started,
                "completed_at": _now(),
                "protocol_sha256": study.config_sha256,
                "source_config_sha256": sources.config_sha256,
                "source_provider": dict(source_provider.provenance()),
                "registry_sha256": _sha256_file(output / "registry.json"),
                "artifact_sha256": artifact_sha,
            },
        )
    except BaseException as exc:
        for name in (*_ARTIFACTS, "registry.json"):
            (output / name).unlink(missing_ok=True)
        _atomic_json(
            output / "run.json",
            {
                "schema_version": "freeze-robustness-evaluation-run/v1",
                "status": "failed",
                "started_at": started,
                "failed_at": _now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    return FreezeEvaluationResult(
        registry_path=output / "registry.json",
        run_path=output / "run.json",
    )


__all__ = [
    "FreezeEvaluationResult",
    "FreezeEvaluationRunConfig",
    "run_freeze_robustness_evaluation",
]

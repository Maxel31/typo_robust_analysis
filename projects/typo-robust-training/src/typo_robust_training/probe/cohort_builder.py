"""Build model-output-free, leakage-resistant linear-probe cohorts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.perturb import (
    apply_typo_operation_to_word,
    eligible_word_spans,
)
from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe.config import load_probe_producer_config
from typo_robust_training.probe.producer import _load_protected_registry
from typo_robust_training.probe.runtime import _inflation_bucket
from typo_robust_training.training.json_io import write_json_atomic
from typo_robust_training.training.pairs import TrainingSource


_REVISION = re.compile(r"[0-9a-f]{40}")
_OPERATIONS = (
    "keyboard-neighbor-substitution",
    "deletion",
    "duplication",
)
_ALLOWED_BUCKETS = (
    "minus-two-or-more",
    "minus-one",
    "same",
    "plus-one",
    "plus-two-or-more",
)
_FEASIBILITY_PARTITION_MODULUS = 5
_FEASIBILITY_PARTITION_REMAINDER = 0
_TEMPLATE_TOP = {
    "schema_version",
    "model",
    "source",
    "cohorts",
    "perturbations",
    "probe",
    "selection",
}
_MODEL_FIELDS = {"id", "revision", "decoder_layers", "hidden_size", "dtype"}
_SOURCE_FIELDS = {"id", "dataset", "revision", "subset", "split", "manifest_schema"}
_COHORT_FIELDS = {
    "class_count",
    "fit_records_per_class",
    "paired_records_per_class",
    "min_word_letters",
    "max_word_letters",
    "max_text_characters",
}
_PERTURBATION_FIELDS = {"seed", "operations", "stratum_counts"}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _namespaced_sha256(namespace: str, *values: str) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for value in values:
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def probe_source_group_sha256(source: TrainingSource) -> str:
    """Return the public identity used to compare a source group with protected tiers."""

    return _namespaced_sha256(
        "typo-probe-source-group/v1",
        source.source,
        source.source_revision,
        source.group_id,
    )


def probe_parent_source_sha256(source: TrainingSource) -> str:
    """Return the public identity used to compare a parent document across roles."""

    return _namespaced_sha256(
        "typo-probe-parent-source/v1",
        source.source,
        source.source_revision,
        source.source_id,
    )


def _positive_integer(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be one regular file")
    return supplied.resolve()


@dataclass(frozen=True, slots=True)
class ProbeCohortTemplate:
    """A closed-world, output-independent recipe for one model's probe data."""

    model: str
    model_revision: str
    decoder_layers: int
    hidden_size: int
    source_id: str
    source_dataset: str
    source_revision: str
    source_subset: str
    source_split: str
    class_count: int
    fit_records_per_class: int
    paired_records_per_class: int
    min_word_letters: int
    max_word_letters: int
    max_text_characters: int
    perturbation_seed: int
    operations: tuple[str, ...]
    stratum_counts: Mapping[str, int]
    probe: Mapping[str, object]
    selection: Mapping[str, object]
    template_sha256: str

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(sorted(self.stratum_counts))

    @property
    def token_inflation_buckets(self) -> tuple[str, ...]:
        return tuple(
            bucket
            for bucket in _ALLOWED_BUCKETS
            if any(stratum.endswith(f"|{bucket}") for stratum in self.strata)
        )


def load_probe_cohort_template(path: Path) -> ProbeCohortTemplate:
    """Load the tracked Mistral cohort recipe without resolving data or a model."""

    resolved = _regular_file(path, label="probe cohort template")
    raw = resolved.read_bytes()
    try:
        value = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("probe cohort template must be UTF-8") from exc
    if not isinstance(value, dict) or set(value) != _TEMPLATE_TOP:
        raise ValueError("probe cohort template fields differ")
    if value["schema_version"] != "typo-probe-transition-data-template/v1":
        raise ValueError("probe cohort template schema differs")
    model = value["model"]
    source = value["source"]
    cohorts = value["cohorts"]
    perturbations = value["perturbations"]
    probe = value["probe"]
    selection = value["selection"]
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise ValueError("probe cohort template model fields differ")
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise ValueError("probe cohort template source fields differ")
    if not isinstance(cohorts, dict) or set(cohorts) != _COHORT_FIELDS:
        raise ValueError("probe cohort template cohort fields differ")
    if not isinstance(perturbations, dict) or set(perturbations) != _PERTURBATION_FIELDS:
        raise ValueError("probe cohort template perturbation fields differ")
    if not isinstance(probe, dict) or not isinstance(selection, dict):
        raise ValueError("probe cohort template producer protocol must contain objects")
    model_revision = _string(model["revision"], field="probe model revision")
    source_revision = _string(source["revision"], field="probe source revision")
    if _REVISION.fullmatch(model_revision) is None or _REVISION.fullmatch(source_revision) is None:
        raise ValueError("probe model and source revisions must be pinned commits")
    if model["dtype"] != "bfloat16":
        raise ValueError("probe cohort template dtype differs")
    if source["manifest_schema"] != "robustness-clean-record-jsonl/v1":
        raise ValueError("probe cohort source manifest schema differs")
    operations = perturbations["operations"]
    raw_stratum_counts = perturbations["stratum_counts"]
    if operations != list(_OPERATIONS):
        raise ValueError("probe cohort typo operations must be the frozen three-operation order")
    if not isinstance(raw_stratum_counts, dict) or not raw_stratum_counts:
        raise ValueError("probe cohort stratum counts must be one non-empty object")
    stratum_counts: dict[str, int] = {}
    represented_operations: set[str] = set()
    for stratum, raw_count in raw_stratum_counts.items():
        if not isinstance(stratum, str):
            raise ValueError("probe cohort stratum names must be strings")
        parts = stratum.split("|")
        if (
            len(parts) != 3
            or parts[0] not in _OPERATIONS
            or parts[1] != "1"
            or parts[2] not in _ALLOWED_BUCKETS
        ):
            raise ValueError("probe cohort stratum name differs from the producer contract")
        stratum_counts[stratum] = _positive_integer(
            raw_count,
            field=f"probe stratum count {stratum}",
        )
        represented_operations.add(parts[0])
    if represented_operations != set(_OPERATIONS):
        raise ValueError("probe cohort strata must represent all three typo operations")
    fit_per_class = _positive_integer(
        cohorts["fit_records_per_class"], field="fit records per class", minimum=2
    )
    if fit_per_class % 2:
        raise ValueError("fit records per class must be even for disjoint probe fits")
    min_letters = _positive_integer(cohorts["min_word_letters"], field="minimum word letters")
    max_letters = _positive_integer(cohorts["max_word_letters"], field="maximum word letters")
    if max_letters < min_letters:
        raise ValueError("maximum word letters must not be below the minimum")
    max_characters = _positive_integer(
        cohorts["max_text_characters"], field="maximum text characters", minimum=64
    )
    class_count = _positive_integer(cohorts["class_count"], field="probe class count", minimum=2)
    paired_per_class = _positive_integer(
        cohorts["paired_records_per_class"], field="paired records per class"
    )
    if sum(stratum_counts.values()) != class_count * paired_per_class:
        raise ValueError(
            "probe cohort global stratum counts must equal class_count times "
            "paired_records_per_class"
        )
    return ProbeCohortTemplate(
        model=_string(model["id"], field="probe model id"),
        model_revision=model_revision,
        decoder_layers=_positive_integer(
            model["decoder_layers"], field="probe decoder layers", minimum=2
        ),
        hidden_size=_positive_integer(model["hidden_size"], field="probe hidden size"),
        source_id=_string(source["id"], field="probe source id"),
        source_dataset=_string(source["dataset"], field="probe source dataset"),
        source_revision=source_revision,
        source_subset=_string(source["subset"], field="probe source subset"),
        source_split=_string(source["split"], field="probe source split"),
        class_count=class_count,
        fit_records_per_class=fit_per_class,
        paired_records_per_class=paired_per_class,
        min_word_letters=min_letters,
        max_word_letters=max_letters,
        max_text_characters=max_characters,
        perturbation_seed=_positive_integer(
            perturbations["seed"], field="probe perturbation seed", minimum=0
        ),
        operations=tuple(operations),
        stratum_counts=stratum_counts,
        probe=dict(probe),
        selection=dict(selection),
        template_sha256=hashlib.sha256(raw).hexdigest(),
    )


class TokenInflationCounter(Protocol):
    """Tokenizer-only boundary; cohort selection never accesses a model output."""

    def bucket(self, *, clean_text: str, typo_text: str) -> str: ...

    def provenance(self) -> Mapping[str, object]: ...


class _AttestedMistralTokenCounter:
    def __init__(
        self,
        *,
        model: str,
        revision: str,
        attestation_path: Path,
    ) -> None:
        from typo_cot.models.tokenizer_attestation import (
            TOKENIZER_ATTESTATION_MANIFEST_ENV,
            load_attested_tokenizer,
            load_tokenizer_attestation_manifest,
            preflight_frozen_tokenizer_attestation,
        )

        manifest_path = _regular_file(attestation_path, label="tokenizer attestation")
        frozen = load_tokenizer_attestation_manifest(manifest_path)
        previous = os.environ.get(TOKENIZER_ATTESTATION_MANIFEST_ENV)
        if previous is not None and Path(previous).resolve() != manifest_path:
            raise ValueError("tokenizer attestation environment conflicts with the explicit path")
        os.environ[TOKENIZER_ATTESTATION_MANIFEST_ENV] = str(manifest_path)
        try:
            preflight = preflight_frozen_tokenizer_attestation(
                expected_model=model,
                expected_revision=revision,
            )
            tokenizer, loaded = load_attested_tokenizer(model, revision)
        finally:
            if previous is None:
                os.environ.pop(TOKENIZER_ATTESTATION_MANIFEST_ENV, None)
            else:
                os.environ[TOKENIZER_ATTESTATION_MANIFEST_ENV] = previous
        if preflight.provenance_dict() != frozen.provenance_dict() or (
            loaded.provenance_dict() != frozen.provenance_dict()
        ):
            raise ValueError("loaded tokenizer differs from the explicit frozen attestation")
        if getattr(tokenizer, "is_fast", False) is not True:
            raise ValueError("probe cohort construction requires the exact fast tokenizer")
        self._tokenizer = tokenizer
        self._attestation = frozen.provenance_dict()
        self._count_cache: dict[str, int] = {}

    def _count(self, text: str) -> int:
        cached = self._count_cache.get(text)
        if cached is not None:
            return cached
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("probe tokenizer must return one mapping")
        ids = encoded.get("input_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise ValueError("probe tokenizer returned an invalid token inventory")
        self._count_cache[text] = len(ids)
        return len(ids)

    def bucket(self, *, clean_text: str, typo_text: str) -> str:
        return _inflation_bucket(self._count(typo_text) - self._count(clean_text))

    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "attested-tokenizer-only-inflation-counter/v1",
            "tokenizer_snapshot_attestation": self._attestation,
            "model_outputs_observed": False,
        }


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    source: TrainingSource
    clean_text: str
    clean_span: tuple[int, int]
    label: str
    source_group_sha256: str
    parent_source_sha256: str
    normalized_clean_sha256: str
    variants: Mapping[str, tuple[str, tuple[int, int], str, str]]

    @property
    def base_identities(self) -> frozenset[str]:
        return frozenset(
            (
                self.source_group_sha256,
                self.parent_source_sha256,
                self.normalized_clean_sha256,
            )
        )

    @property
    def all_identities(self) -> frozenset[str]:
        return self.base_identities | frozenset(row[3] for row in self.variants.values())


def _bounded_text(text: str, *, maximum: int) -> str:
    clean = text.strip()
    if len(clean) <= maximum:
        return clean
    prefix = clean[:maximum]
    boundary = prefix.rfind(" ")
    if boundary >= maximum // 2:
        prefix = prefix[:boundary]
    return prefix.strip()


def _load_sources(path: Path, *, protocol: ProbeCohortTemplate) -> tuple[TrainingSource, ...]:
    resolved = _regular_file(path, label="probe clean source manifest")
    rows: list[TrainingSource] = []
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for line_number, line in read_lf_jsonl_lines(resolved, context="probe clean source manifest"):
        value = strict_loads(line, context=f"{resolved}:{line_number}")
        source = TrainingSource.from_dict(value)
        if (
            source.kind != "clean"
            or source.source != protocol.source_id
            or source.source_revision != protocol.source_revision
            or source.source_split != protocol.source_split
            or source.task is not None
            or source.answer is not None
        ):
            raise ValueError("probe source row is not clean pinned FineWeb-Edu train text")
        if source.clean_record().record_id != source.record_id:
            raise ValueError("probe source record id differs from its pinned source identity")
        if source.record_id in seen_record_ids or source.source_id in seen_source_ids:
            raise ValueError("probe clean source manifest contains duplicate identities")
        seen_record_ids.add(source.record_id)
        seen_source_ids.add(source.source_id)
        rows.append(source)
    if not rows:
        raise ValueError("probe clean source manifest contains no records")
    return tuple(sorted(rows, key=lambda item: item.record_id))


def _candidate_for_source(
    source: TrainingSource,
    *,
    protocol: ProbeCohortTemplate,
) -> _SourceCandidate | None:
    text = _bounded_text(source.clean_text, maximum=protocol.max_text_characters)
    if not text:
        return None
    eligible: list[tuple[str, tuple[int, int]]] = []
    for span in eligible_word_spans(text, minimum_letters=protocol.min_word_letters):
        word = text[slice(*span)]
        if (
            not word.isascii()
            or not word.isalpha()
            or word != word.lower()
            or len(word) > protocol.max_word_letters
        ):
            continue
        eligible.append((word, span))
    if not eligible:
        return None
    label, span = min(
        eligible,
        key=lambda row: (
            _namespaced_sha256(
                "typo-probe-class-designation/v1",
                source.record_id,
                row[0],
                str(row[1][0]),
                str(row[1][1]),
            ),
            row,
        ),
    )
    return _SourceCandidate(
        source=source,
        clean_text=text,
        clean_span=span,
        label=label,
        source_group_sha256=probe_source_group_sha256(source),
        parent_source_sha256=probe_parent_source_sha256(source),
        normalized_clean_sha256=normalized_content_sha256(text),
        variants={},
    )


def _attach_variants(
    candidate: _SourceCandidate,
    *,
    protocol: ProbeCohortTemplate,
    counter: TokenInflationCounter,
) -> _SourceCandidate:
    variants: dict[str, tuple[str, tuple[int, int], str, str]] = {}
    for operation in protocol.operations:
        seed_material = _namespaced_sha256(
            "typo-probe-fixed-operation/v1",
            str(protocol.perturbation_seed),
            candidate.source.record_id,
            candidate.label,
            str(candidate.clean_span[0]),
            operation,
        )
        typo_word = apply_typo_operation_to_word(
            candidate.label,
            operation,
            random.Random(int(seed_material, 16)),
        )
        typo_text = (
            candidate.clean_text[: candidate.clean_span[0]]
            + typo_word
            + candidate.clean_text[candidate.clean_span[1] :]
        )
        typo_span = (
            candidate.clean_span[0],
            candidate.clean_span[0] + len(typo_word),
        )
        bucket = counter.bucket(clean_text=candidate.clean_text, typo_text=typo_text)
        if bucket not in _ALLOWED_BUCKETS:
            raise ValueError("token inflation counter returned an unknown bucket")
        variants[operation] = (
            typo_text,
            typo_span,
            bucket,
            normalized_content_sha256(typo_text),
        )
    return _SourceCandidate(
        source=candidate.source,
        clean_text=candidate.clean_text,
        clean_span=candidate.clean_span,
        label=candidate.label,
        source_group_sha256=candidate.source_group_sha256,
        parent_source_sha256=candidate.parent_source_sha256,
        normalized_clean_sha256=candidate.normalized_clean_sha256,
        variants=variants,
    )


def _is_feasibility_candidate(candidate: _SourceCandidate) -> bool:
    value = int(
        _namespaced_sha256(
            "typo-probe-feasibility-partition/v1",
            candidate.source.record_id,
        ),
        16,
    )
    return value % _FEASIBILITY_PARTITION_MODULUS == _FEASIBILITY_PARTITION_REMAINDER


def _order_key(*values: str) -> tuple[str, ...]:
    return (_namespaced_sha256("typo-probe-cohort-order/v1", *values), *values)


def _cohort_record_id(*, role: str, candidate: _SourceCandidate) -> str:
    return _namespaced_sha256(
        "typo-probe-cohort-record/v1",
        role,
        candidate.source.record_id,
        candidate.label,
        str(candidate.clean_span[0]),
        str(candidate.clean_span[1]),
    )


def _fit_row(candidate: _SourceCandidate, *, class_id: int) -> dict[str, object]:
    return {
        "record_id": _cohort_record_id(role="fit", candidate=candidate),
        "source_group_sha256": candidate.source_group_sha256,
        "parent_source_sha256": candidate.parent_source_sha256,
        "normalized_clean_sha256": candidate.normalized_clean_sha256,
        "class_id": class_id,
        "clean_text": candidate.clean_text,
        "clean_word_char_span": list(candidate.clean_span),
    }


def _paired_row(
    candidate: _SourceCandidate,
    *,
    role: str,
    class_id: int,
    operation: str,
) -> dict[str, object]:
    typo_text, typo_span, bucket, noisy_hash = candidate.variants[operation]
    record_id = _cohort_record_id(role=role, candidate=candidate)
    return {
        "record_id": record_id,
        "source_group_sha256": candidate.source_group_sha256,
        "parent_source_sha256": candidate.parent_source_sha256,
        "normalized_clean_sha256": candidate.normalized_clean_sha256,
        "class_id": class_id,
        "clean_text": candidate.clean_text,
        "clean_word_char_span": list(candidate.clean_span),
        "pair_id": _namespaced_sha256("typo-probe-pair/v1", role, record_id, operation, bucket),
        "normalized_noisy_sha256": noisy_hash,
        "edit_type": operation,
        "edit_count": 1,
        "token_inflation_bucket": bucket,
        "typo_text": typo_text,
        "typo_word_char_span": list(typo_span),
    }


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    initial_capacity: int


class _DeterministicFlow:
    """Small deterministic Dinic implementation for exact cohort quotas."""

    def __init__(self, nodes: int) -> None:
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(nodes)]

    def add_edge(self, source: int, target: int, capacity: int) -> int:
        forward_index = len(self.graph[source])
        reverse_index = len(self.graph[target])
        self.graph[source].append(_FlowEdge(target, reverse_index, capacity, capacity))
        self.graph[target].append(_FlowEdge(source, forward_index, 0, 0))
        return forward_index

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue: deque[int] = deque((source,))
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and levels[edge.to] < 0:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if levels[sink] < 0:
                return total
            offsets = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while offsets[node] < len(self.graph[node]):
                    edge = self.graph[node][offsets[node]]
                    if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                        sent = send(edge.to, min(amount, edge.capacity))
                        if sent:
                            edge.capacity -= sent
                            reverse = self.graph[edge.to][edge.reverse]
                            reverse.capacity += sent
                            return sent
                    offsets[node] += 1
                return 0

            while sent := send(source, 1 << 60):
                total += sent


@dataclass(frozen=True, slots=True)
class _RoleAllocation:
    rows: tuple[tuple[_SourceCandidate, str], ...]
    identities: frozenset[str]


def _allocate_paired_role(
    candidates_by_label: Mapping[str, Sequence[_SourceCandidate]],
    *,
    labels: Sequence[str],
    role: str,
    protocol: ProbeCohortTemplate,
    forbidden: set[str],
) -> _RoleAllocation | None:
    candidates = [
        candidate
        for label in labels
        for candidate in sorted(
            candidates_by_label[label],
            key=lambda item: _order_key(role, label, item.source.record_id),
        )
        if not candidate.all_identities & forbidden
    ]
    source_node = 0
    class_offset = 1
    candidate_offset = class_offset + len(labels)
    stratum_offset = candidate_offset + len(candidates)
    strata = protocol.strata
    sink_node = stratum_offset + len(strata)
    flow = _DeterministicFlow(sink_node + 1)
    class_ids = {label: index for index, label in enumerate(labels)}
    for label in labels:
        flow.add_edge(
            source_node,
            class_offset + class_ids[label],
            protocol.paired_records_per_class,
        )
    chosen_edges: list[tuple[_SourceCandidate, str, int, int]] = []
    stratum_nodes = {stratum: stratum_offset + index for index, stratum in enumerate(strata)}
    for index, candidate in enumerate(candidates):
        candidate_node = candidate_offset + index
        flow.add_edge(
            class_offset + class_ids[candidate.label],
            candidate_node,
            1,
        )
        for operation in protocol.operations:
            bucket = candidate.variants[operation][2]
            stratum = f"{operation}|1|{bucket}"
            if stratum not in stratum_nodes:
                continue
            edge_index = flow.add_edge(candidate_node, stratum_nodes[stratum], 1)
            chosen_edges.append((candidate, operation, candidate_node, edge_index))
    for stratum in strata:
        flow.add_edge(
            stratum_nodes[stratum],
            sink_node,
            protocol.stratum_counts[stratum],
        )
    required = protocol.class_count * protocol.paired_records_per_class
    if flow.maximum_flow(source_node, sink_node) != required:
        return None
    rows = tuple(
        (candidate, operation)
        for candidate, operation, node, edge_index in chosen_edges
        if flow.graph[node][edge_index].initial_capacity == 1
        and flow.graph[node][edge_index].capacity == 0
    )
    if len(rows) != required:
        raise RuntimeError("probe cohort flow extraction differs from the solved flow")
    identities = frozenset(
        identity
        for candidate, operation in rows
        for identity in (
            *candidate.base_identities,
            candidate.variants[operation][3],
        )
    )
    return _RoleAllocation(rows=rows, identities=identities)


def _allocate_fit(
    candidates_by_label: Mapping[str, Sequence[_SourceCandidate]],
    *,
    labels: Sequence[str],
    protocol: ProbeCohortTemplate,
    forbidden: set[str],
) -> tuple[tuple[_SourceCandidate, ...], frozenset[str]] | None:
    rows: list[_SourceCandidate] = []
    identities: set[str] = set()
    for label in labels:
        accepted = 0
        for candidate in sorted(
            candidates_by_label[label],
            key=lambda item: _order_key("fit", label, item.source.record_id),
        ):
            if candidate.all_identities & forbidden:
                continue
            rows.append(candidate)
            forbidden.update(candidate.base_identities)
            identities.update(candidate.base_identities)
            accepted += 1
            if accepted == protocol.fit_records_per_class:
                break
        if accepted != protocol.fit_records_per_class:
            return None
    return tuple(rows), frozenset(identities)


def _allocate_complete_cohort(
    candidates_by_label: Mapping[str, Sequence[_SourceCandidate]],
    *,
    labels: Sequence[str],
    paired_roles: Sequence[str],
    protocol: ProbeCohortTemplate,
    forbidden: set[str],
) -> (
    tuple[
        Mapping[str, tuple[tuple[_SourceCandidate, str], ...]],
        tuple[_SourceCandidate, ...],
        frozenset[str],
    ]
    | None
):
    used = set(forbidden)
    paired: dict[str, tuple[tuple[_SourceCandidate, str], ...]] = {}
    allocated_identities: set[str] = set()
    for role in paired_roles:
        allocation = _allocate_paired_role(
            candidates_by_label,
            labels=labels,
            role=role,
            protocol=protocol,
            forbidden=used,
        )
        if allocation is None:
            return None
        paired[role] = allocation.rows
        used.update(allocation.identities)
        allocated_identities.update(allocation.identities)
    fit = _allocate_fit(
        candidates_by_label,
        labels=labels,
        protocol=protocol,
        forbidden=used,
    )
    if fit is None:
        return None
    fit_rows, fit_identities = fit
    allocated_identities.update(fit_identities)
    return paired, fit_rows, frozenset(allocated_identities)


def _deduplicate_full_candidates(
    candidates: Sequence[_SourceCandidate],
    *,
    protected: set[str],
) -> tuple[_SourceCandidate, ...]:
    """Keep one deterministic representative of every transitive identity component."""

    observed = set(protected)
    accepted: list[_SourceCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: _order_key("full-identity", item.source.record_id),
    ):
        if candidate.all_identities & observed:
            continue
        accepted.append(candidate)
        observed.update(candidate.all_identities)
    return tuple(accepted)


def _group_candidates(
    candidates: Sequence[_SourceCandidate],
    *,
    labels: Sequence[str],
) -> Mapping[str, tuple[_SourceCandidate, ...]]:
    grouped: dict[str, list[_SourceCandidate]] = {label: [] for label in labels}
    for candidate in candidates:
        if candidate.label in grouped:
            grouped[candidate.label].append(candidate)
    return {label: tuple(rows) for label, rows in grouped.items()}


def _select_labels_from_feasibility(
    candidates: Sequence[_SourceCandidate],
    *,
    protocol: ProbeCohortTemplate,
) -> tuple[str, ...]:
    counts = Counter(candidate.label for candidate in candidates)
    minimum = protocol.fit_records_per_class + protocol.paired_records_per_class
    ranked = sorted(
        (label for label, count in counts.items() if count >= minimum),
        key=lambda label: (
            -counts[label],
            _namespaced_sha256("typo-probe-class-rank-tie/v1", label),
            label,
        ),
    )
    if len(ranked) < protocol.class_count:
        raise ValueError(
            "insufficient disjoint feasibility capacity for the frozen class count: "
            f"required_classes={protocol.class_count}, eligible_classes={len(ranked)}"
        )
    return tuple(ranked[: protocol.class_count])


def _attest_clean_code_revision() -> str:
    module = Path(__file__).resolve()
    root_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=module.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise ValueError("probe cohort builder checkout is not a git worktree")
    root = Path(root_result.stdout.strip()).resolve()
    revision_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    if revision_result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        raise ValueError("probe cohort builder code revision is unavailable")
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "projects/typo-robust-training/src/typo_robust_training",
            "projects/typo-cot/src/typo_cot",
        ),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("probe cohort builder runtime source trees are not clean")
    return revision


@dataclass(frozen=True, slots=True)
class ProbeTransitionDataBuildConfig:
    template_path: Path
    source_manifest_path: Path
    protected_registry_path: Path
    tokenizer_attestation_path: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ProbeTransitionDataBuildResult:
    class_inventory_path: Path
    fit_manifest_path: Path
    selection_manifest_path: Path
    validation_manifest_path: Path
    protected_registry_path: Path
    feasibility_report_path: Path
    producer_config_path: Path
    run_path: Path
    classes: int
    records: int


CounterFactory = Callable[[ProbeCohortTemplate, Path], TokenInflationCounter]


def _default_counter_factory(
    protocol: ProbeCohortTemplate,
    attestation_path: Path,
) -> TokenInflationCounter:
    return _AttestedMistralTokenCounter(
        model=protocol.model,
        revision=protocol.model_revision,
        attestation_path=attestation_path,
    )


def _manifest(role: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "typo-probe-cohort/v2",
        "role": role,
        "records": [dict(row) for row in rows],
    }


def run_build_probe_transition_data(
    config: ProbeTransitionDataBuildConfig,
    *,
    counter: TokenInflationCounter | None = None,
    counter_factory: CounterFactory = _default_counter_factory,
    code_revision: str | None = None,
) -> ProbeTransitionDataBuildResult:
    """Materialize every hash-bound input required by ``select-probe-transition``."""

    if not isinstance(config, ProbeTransitionDataBuildConfig):
        raise TypeError("probe transition data build config has the wrong type")
    target = Path(config.output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"probe transition data output must not exist: {target}")
    protocol = load_probe_cohort_template(config.template_path)
    source_path = _regular_file(config.source_manifest_path, label="probe clean source manifest")
    protected_path = _regular_file(config.protected_registry_path, label="protected registry")
    attestation_path = _regular_file(
        config.tokenizer_attestation_path, label="tokenizer attestation"
    )
    protected_sets = _load_protected_registry(protected_path)
    protected_union = set().union(*protected_sets)
    sources = _load_sources(source_path, protocol=protocol)
    observed_revision = code_revision or _attest_clean_code_revision()
    if _REVISION.fullmatch(observed_revision) is None:
        raise ValueError("probe cohort builder code revision must be one pinned commit")

    base_candidates: list[_SourceCandidate] = []
    source_identities_seen: set[str] = set()
    for source in sources:
        candidate = _candidate_for_source(source, protocol=protocol)
        if candidate is None:
            continue
        if candidate.base_identities & protected_union:
            continue
        # Keep one deterministic representative from every transitive source,
        # parent, or exact-content component before assigning scientific roles.
        if candidate.base_identities & source_identities_seen:
            continue
        source_identities_seen.update(candidate.base_identities)
        base_candidates.append(candidate)

    feasibility_base = tuple(
        candidate for candidate in base_candidates if _is_feasibility_candidate(candidate)
    )
    labels = _select_labels_from_feasibility(feasibility_base, protocol=protocol)
    token_counter = counter or counter_factory(protocol, attestation_path)
    provenance = dict(token_counter.provenance())
    if provenance.get("model_outputs_observed") is not False:
        raise ValueError(
            "probe cohort token counter must attest that no model outputs were observed"
        )
    with_variants = tuple(
        _attach_variants(candidate, protocol=protocol, counter=token_counter)
        for candidate in base_candidates
        if candidate.label in labels
    )
    unique_candidates = _deduplicate_full_candidates(
        with_variants,
        protected=protected_union,
    )
    feasibility_candidates = tuple(
        candidate for candidate in unique_candidates if _is_feasibility_candidate(candidate)
    )
    material_candidates = tuple(
        candidate for candidate in unique_candidates if not _is_feasibility_candidate(candidate)
    )
    feasibility_by_label = _group_candidates(feasibility_candidates, labels=labels)
    material_by_label = _group_candidates(material_candidates, labels=labels)
    feasibility = _allocate_complete_cohort(
        feasibility_by_label,
        labels=labels,
        paired_roles=("feasibility",),
        protocol=protocol,
        forbidden=set(protected_union),
    )
    if feasibility is None:
        raise ValueError(
            "insufficient disjoint FineWeb-Edu feasibility capacity for the frozen "
            "global class/stratum quotas"
        )
    feasibility_paired, feasibility_fit, feasibility_identities = feasibility
    allocation = _allocate_complete_cohort(
        material_by_label,
        labels=labels,
        paired_roles=("selection", "validation"),
        protocol=protocol,
        forbidden=set(protected_union) | set(feasibility_identities),
    )
    if allocation is None:
        raise ValueError(
            "insufficient disjoint FineWeb-Edu capacity for the frozen global class/stratum quotas"
        )
    paired, fit_rows, _ = allocation

    class_inventory = {
        "schema_version": "typo-word-identity-classes/v1",
        "classes": [
            {"class_id": class_id, "label": label} for class_id, label in enumerate(labels)
        ],
    }
    class_ids = {label: class_id for class_id, label in enumerate(labels)}
    role_rows: dict[str, list[dict[str, object]]] = {
        "fit": [_fit_row(candidate, class_id=class_ids[candidate.label]) for candidate in fit_rows],
        "selection": [
            _paired_row(
                candidate,
                role="selection",
                class_id=class_ids[candidate.label],
                operation=operation,
            )
            for candidate, operation in paired["selection"]
        ],
        "validation": [
            _paired_row(
                candidate,
                role="validation",
                class_id=class_ids[candidate.label],
                operation=operation,
            )
            for candidate, operation in paired["validation"]
        ],
    }
    observed_strata = {
        split: Counter(
            f"{operation}|1|{candidate.variants[operation][2]}"
            for candidate in candidates
            for operation in protocol.operations
        )
        for split, candidates in (
            ("feasibility", feasibility_candidates),
            ("materialization", material_candidates),
        )
    }
    feasibility_report = {
        "schema_version": "typo-probe-cohort-feasibility/v1",
        "model_outputs_observed": False,
        "partition": {
            "rule": "source-record-id-sha256-modulo/v1",
            "namespace": "typo-probe-feasibility-partition/v1",
            "modulus": _FEASIBILITY_PARTITION_MODULUS,
            "feasibility_remainder": _FEASIBILITY_PARTITION_REMAINDER,
        },
        "class_selection": {
            "rule": "feasibility-frequency-descending-then-namespaced-sha256/v1",
            "labels": list(labels),
            "minimum_candidates_per_class": (
                protocol.fit_records_per_class + protocol.paired_records_per_class
            ),
        },
        "candidate_counts": {
            "feasibility": len(feasibility_candidates),
            "materialization": len(material_candidates),
        },
        "available_stratum_counts": {
            split: dict(sorted(counts.items())) for split, counts in observed_strata.items()
        },
        "simulated_cohort": {
            "fit_records": len(feasibility_fit),
            "paired_records": len(feasibility_paired["feasibility"]),
            "stratum_counts": dict(protocol.stratum_counts),
        },
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        paths = {
            "class_inventory": temporary / "class_inventory.json",
            "fit_manifest": temporary / "fit_manifest.json",
            "selection_manifest": temporary / "selection_manifest.json",
            "validation_manifest": temporary / "validation_manifest.json",
            "protected_registry": temporary / "protected_split_registry.json",
            "feasibility_report": temporary / "probe_cohort_feasibility.json",
            "producer_config": temporary / "probe_producer_config.json",
            "run": temporary / "build_probe_transition_data_run.json",
        }
        write_json_atomic(paths["class_inventory"], class_inventory)
        for role in ("fit", "selection", "validation"):
            write_json_atomic(paths[f"{role}_manifest"], _manifest(role, role_rows[role]))
        paths["protected_registry"].write_bytes(protected_path.read_bytes())
        write_json_atomic(paths["feasibility_report"], feasibility_report)
        producer_config = {
            "schema_version": "typo-linear-probe-producer-config/v4",
            "model": {
                "id": protocol.model,
                "revision": protocol.model_revision,
                "code_revision": observed_revision,
                "decoder_layers": protocol.decoder_layers,
                "hidden_size": protocol.hidden_size,
                "dtype": "bfloat16",
            },
            "inputs": {
                "class_inventory_sha256": sha256_file(paths["class_inventory"]),
                "fit_manifest_sha256": sha256_file(paths["fit_manifest"]),
                "selection_manifest_sha256": sha256_file(paths["selection_manifest"]),
                "validation_manifest_sha256": sha256_file(paths["validation_manifest"]),
                "protected_registry_sha256": sha256_file(paths["protected_registry"]),
            },
            "cohorts": {
                "records_per_class": {
                    "fit": protocol.fit_records_per_class,
                    "selection": protocol.paired_records_per_class,
                    "validation": protocol.paired_records_per_class,
                },
                "min_source_groups_per_class": {
                    "fit": protocol.fit_records_per_class,
                    "selection": protocol.paired_records_per_class,
                    "validation": protocol.paired_records_per_class,
                },
                "stratum_counts": {
                    "selection": dict(protocol.stratum_counts),
                    "validation": dict(protocol.stratum_counts),
                },
            },
            "probe": dict(protocol.probe),
            "selection": dict(protocol.selection),
        }
        write_json_atomic(paths["producer_config"], producer_config)
        # The consumer's existing closed-world loader is the final schema oracle.
        load_probe_producer_config(paths["producer_config"])
        artifacts = {
            name: {
                "relative_path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
            if name != "run"
        }
        run_payload = {
            "schema_version": "build-probe-transition-data-run/v1",
            "status": "completed",
            "code_revision": observed_revision,
            "model_outputs_observed": False,
            "template": {
                "sha256": protocol.template_sha256,
                "schema_version": "typo-probe-transition-data-template/v1",
            },
            "source": {
                "id": protocol.source_id,
                "dataset": protocol.source_dataset,
                "revision": protocol.source_revision,
                "subset": protocol.source_subset,
                "split": protocol.source_split,
                "eligibility_contract": (
                    "unused-record-group-parent-content-disjoint-from-all-protected-tiers/v1"
                ),
                "manifest_sha256": sha256_file(source_path),
                "records": len(sources),
            },
            "token_counter": provenance,
            "tokenizer_attestation_manifest_sha256": sha256_file(attestation_path),
            "identity_rules": {
                "source_group": "typo-probe-source-group/v1",
                "parent_source": "typo-probe-parent-source/v1",
                "cohort_record": "typo-probe-cohort-record/v1",
                "pair": "typo-probe-pair/v1",
                "class_designation": "typo-probe-class-designation/v1",
                "class_ranking": "typo-probe-class-rank-tie/v1",
                "selection_order": "typo-probe-cohort-order/v1",
                "feasibility_partition": "typo-probe-feasibility-partition/v1",
            },
            "cohorts": {
                "classes": protocol.class_count,
                "class_labels_sha256": _canonical_sha256(list(labels)),
                "fit_records": len(role_rows["fit"]),
                "selection_records": len(role_rows["selection"]),
                "validation_records": len(role_rows["validation"]),
                "operations": list(protocol.operations),
                "token_inflation_buckets": list(protocol.token_inflation_buckets),
                "global_stratum_counts": dict(protocol.stratum_counts),
            },
            "artifacts": artifacts,
        }
        run_payload["self_hash"] = {
            "algorithm": "sha256",
            "canonicalization": "canonical-json-without-self-hash/v1",
            "sha256": _canonical_sha256(run_payload),
        }
        write_json_atomic(paths["run"], run_payload)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    total_records = sum(len(rows) for rows in role_rows.values())
    return ProbeTransitionDataBuildResult(
        class_inventory_path=target / "class_inventory.json",
        fit_manifest_path=target / "fit_manifest.json",
        selection_manifest_path=target / "selection_manifest.json",
        validation_manifest_path=target / "validation_manifest.json",
        protected_registry_path=target / "protected_split_registry.json",
        feasibility_report_path=target / "probe_cohort_feasibility.json",
        producer_config_path=target / "probe_producer_config.json",
        run_path=target / "build_probe_transition_data_run.json",
        classes=protocol.class_count,
        records=total_records,
    )


__all__ = [
    "ProbeCohortTemplate",
    "ProbeTransitionDataBuildConfig",
    "ProbeTransitionDataBuildResult",
    "TokenInflationCounter",
    "load_probe_cohort_template",
    "probe_parent_source_sha256",
    "probe_source_group_sha256",
    "run_build_probe_transition_data",
]

"""Build a disjoint clean FineWeb-Edu supplement for WP-1 SAE training."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.data.sources import HuggingFaceTokenCounter
from typo_robust_training.sae.config import SaeProtocol, load_sae_protocol
from typo_robust_training.sae.data import (
    prepare_sae_sources,
    record_id_sha256,
    sha256_file,
    write_json_atomic,
)
from typo_robust_training.sae.duplicates import CharacterNgramDuplicateGuard
from typo_robust_training.sae.registry import load_sae_preregistration
from typo_robust_training.training.pairs import TrainingSource


_REQUIRED_EXCLUSION_ROLES = frozenset(
    {
        "tune",
        "pre_pr_gate",
        "final_test",
        "localization-selection",
        "localization-validation",
    }
)


@dataclass(frozen=True, slots=True)
class SaeCorpusBuildConfig:
    config_path: Path
    registry_path: Path
    existing_data_paths: tuple[Path, ...]
    exclusion_paths: tuple[Path, ...]
    training_budget: str
    output_dir: Path

    def __post_init__(self) -> None:
        for field in ("config_path", "registry_path", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        object.__setattr__(
            self,
            "existing_data_paths",
            tuple(Path(path) for path in self.existing_data_paths),
        )
        object.__setattr__(
            self,
            "exclusion_paths",
            tuple(Path(path) for path in self.exclusion_paths),
        )
        if self.training_budget not in {"minimum", "preferred"}:
            raise ValueError("SAE corpus training budget must be minimum or preferred")


@dataclass(frozen=True, slots=True)
class SaeCorpusBuildResult:
    supplement_path: Path
    registry_path: Path
    run_path: Path
    supplement_records: int
    supplement_tokens: int
    total_eligible_tokens: int


class SaeCleanSourceProvider(Protocol):
    def iter_sources(self) -> Iterable[TrainingSource]: ...

    def provenance(self) -> Mapping[str, object]: ...


def _segment_document(record: CleanRecord, *, character_limit: int) -> CleanRecord:
    text = record.text
    if len(text) <= character_limit:
        start, end, segment = 0, len(text), text
    else:
        starts = len(text) - character_limit + 1
        start = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big") % starts
        if start > 0 and not text[start - 1].isspace():
            while start < len(text) and not text[start].isspace():
                start += 1
        while start < len(text) and text[start].isspace():
            start += 1
        end = min(start + character_limit, len(text))
        if end < len(text) and not text[end].isspace():
            while end > start and not text[end - 1].isspace():
                end -= 1
        while end > start and text[end - 1].isspace():
            end -= 1
        segment = text[start:end]
        if not segment:
            raise ValueError("FineWeb-Edu supplement window contains no complete word")
    return replace(
        record,
        text=segment,
        metadata={
            **dict(record.metadata),
            "original_character_count": len(text),
            "document_window": [start, end],
            "document_character_limit": character_limit,
        },
    )


def _source_payload(source: TrainingSource) -> dict[str, object]:
    text = source.clean_text
    return {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": source.record_id,
        "source": source.source,
        "source_revision": source.source_revision,
        "source_split": source.source_split,
        "source_id": source.source_id,
        "group_id": source.group_id,
        "split": "train",
        "text": text,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_content_sha256": hashlib.sha256(
            " ".join(text.casefold().split()).encode("utf-8")
        ).hexdigest(),
        "metadata": dict(source.metadata),
        "token_count": source.token_count,
    }


class HuggingFaceSaeCleanSourceProvider:
    """Pinned, unshuffled FineWeb-Edu stream with exact tokenizer accounting."""

    def __init__(self, *, protocol: SaeProtocol, tokenizer: object | None = None) -> None:
        self.protocol = protocol
        self.counter = HuggingFaceTokenCounter(
            model=protocol.model,
            revision=protocol.model_revision,
            max_sequence_length=protocol.max_sequence_length,
            typo_token_headroom=0,
            tokenizer=tokenizer,
        )

    def iter_sources(self) -> Iterable[TrainingSource]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.protocol.source_dataset,
            name=self.protocol.source_subset,
            split=self.protocol.source_split,
            revision=self.protocol.source_revision,
            streaming=True,
        )
        for index, row in enumerate(dataset):
            if not isinstance(row, Mapping):
                raise ValueError("FineWeb-Edu supplement emitted a non-object row")
            raw_text = row.get("text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            upstream_id = str(row.get("id") or f"train-{index}")
            group = str(row.get("id") or row.get("url") or upstream_id)
            metadata: dict[str, object] = {"dataset_row_index": index}
            metadata.update(
                {
                    key: row[key]
                    for key in ("url", "dump", "date", "language", "score", "token_count")
                    if key in row and isinstance(row[key], (str, int, float, bool, type(None)))
                }
            )
            record = CleanRecord(
                source="fineweb_edu",
                source_revision=self.protocol.source_revision,
                source_split="train",
                source_id=f"fineweb_edu:{upstream_id}",
                group_id=f"fineweb_edu:{group}",
                text=raw_text,
                task=None,
                answer=None,
                metadata=metadata,
            )
            prepared = self.counter.prepare_record(
                _segment_document(
                    record,
                    character_limit=self.protocol.document_character_limit,
                )
            )
            payload = {
                "schema_version": "robustness-clean-record/v1",
                "kind": "clean",
                "record_id": prepared.record_id,
                "source": prepared.source,
                "source_revision": prepared.source_revision,
                "source_split": prepared.source_split,
                "source_id": prepared.source_id,
                "group_id": prepared.group_id,
                "split": "train",
                "text": prepared.text,
                "task": None,
                "answer": None,
                "content_sha256": hashlib.sha256(prepared.text.encode()).hexdigest(),
                "normalized_content_sha256": hashlib.sha256(
                    " ".join(prepared.text.casefold().split()).encode()
                ).hexdigest(),
                "metadata": dict(prepared.metadata),
                "token_count": self.counter(prepared.text),
            }
            yield TrainingSource.from_dict(payload)

    def provenance(self) -> Mapping[str, object]:
        import datasets
        import transformers

        return {
            "provider": "pinned-unshuffled-fineweb-edu-stream/v1",
            "dataset": self.protocol.source_dataset,
            "subset": self.protocol.source_subset,
            "revision": self.protocol.source_revision,
            "split": self.protocol.source_split,
            "datasets_version": datasets.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer": self.protocol.model,
            "tokenizer_revision": self.protocol.model_revision,
        }


def _jsonl_files(paths: Sequence[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise ValueError(f"SAE exclusion path does not exist: {path}")
        candidates = (
            (path,)
            if path.is_file()
            else tuple(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and (candidate.suffix == ".jsonl" or candidate.name.endswith(".jsonl.gz"))
                )
            )
        )
        if not candidates:
            raise ValueError(f"SAE exclusion path has no JSONL artifacts: {path}")
        files.extend(candidates)
    resolved = tuple(sorted(files))
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("SAE exclusion files are empty or duplicated")
    return resolved


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    opener: Any = gzip.open if path.name.endswith(".jsonl.gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, Mapping):
                raise ValueError("SAE exclusion row must be an object")
            yield payload


def _seed_exclusions(
    *,
    guard: CharacterNgramDuplicateGuard,
    existing: Sequence[TrainingSource],
    exclusion_files: Sequence[Path],
) -> tuple[set[str], set[str], set[str], frozenset[str], int]:
    record_ids = {source.record_id for source in existing}
    source_ids = {source.source_id for source in existing}
    group_ids = {source.group_id for source in existing}
    roles: set[str] = set()
    texts = 0

    def register(identity: str, text: str) -> None:
        nonlocal texts
        guard.add(identity, text)
        texts += 1

    for source in existing:
        register(f"existing:{source.record_id}", source.clean_text)
    for file_index, path in enumerate(exclusion_files):
        for line_number, row in enumerate(_iter_jsonl(path), 1):
            role = row.get("split")
            if isinstance(role, str):
                roles.add(role)
            for field, inventory in (
                ("record_id", record_ids),
                ("source_id", source_ids),
                ("group_id", group_ids),
            ):
                value = row.get(field)
                if isinstance(value, str) and value:
                    inventory.add(value)
            for field in ("text", "clean_text", "typo_text"):
                value = row.get(field)
                if isinstance(value, str) and value:
                    register(f"excluded:{file_index}:{line_number}:{field}", value)
    missing = _REQUIRED_EXCLUSION_ROLES - roles
    if missing:
        raise ValueError(f"SAE exclusions omit frozen data roles: {sorted(missing)}")
    return record_ids, source_ids, group_ids, frozenset(roles), texts


def _target_source_tokens(protocol: SaeProtocol, *, training_budget: str) -> int:
    training = (
        protocol.minimum_training_tokens
        if training_budget == "minimum"
        else protocol.preferred_training_tokens
    )
    return (
        training
        + protocol.statistics_tokens
        + protocol.splice_documents * protocol.max_sequence_length
    )


def run_build_sae_clean_corpus(
    config: SaeCorpusBuildConfig,
    *,
    source_provider: SaeCleanSourceProvider | None = None,
) -> SaeCorpusBuildResult:
    """Extend the eligible stream to its frozen token budget without role leakage."""

    if not isinstance(config, SaeCorpusBuildConfig):
        raise TypeError("SAE corpus config must be SaeCorpusBuildConfig")
    protocol = load_sae_protocol(config.config_path)
    preregistration = load_sae_preregistration(config.registry_path, protocol=protocol)
    output = config.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"SAE corpus output is not an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_sae_sources(
        config.existing_data_paths,
        protected_manifest_sha256=preregistration.source_manifest_sha256,
        reserved_seed=protocol.reserved_order_seed,
        reserved_epoch=protocol.reserved_order_epoch,
        reserved_records=protocol.reserved_prefix_records,
    )
    existing = (*prepared.reserved, *prepared.sources)
    if any(
        source.source_revision != protocol.source_revision
        or source.metadata.get("tokenizer_model") != protocol.model
        or source.metadata.get("tokenizer_revision") != protocol.model_revision
        for source in existing
    ):
        raise ValueError("SAE existing source provenance differs from the frozen protocol")
    exclusion_files = _jsonl_files(config.exclusion_paths)
    guard = CharacterNgramDuplicateGuard(
        shingle_size=protocol.near_duplicate_shingle_size,
        threshold=protocol.near_duplicate_jaccard_threshold,
    )
    record_ids, source_ids, group_ids, roles, exclusion_texts = _seed_exclusions(
        guard=guard,
        existing=existing,
        exclusion_files=exclusion_files,
    )
    target = _target_source_tokens(protocol, training_budget=config.training_budget)
    needed = target - prepared.source_tokens
    if needed <= 0:
        raise ValueError("SAE existing clean stream already satisfies the requested budget")
    provider = source_provider or HuggingFaceSaeCleanSourceProvider(protocol=protocol)
    supplement_path = output / "sae_clean_supplement.jsonl"
    temporary = supplement_path.with_name(f".{supplement_path.name}.{os.getpid()}.tmp")
    selected: list[TrainingSource] = []
    skipped_identity = 0
    skipped_near_duplicate = 0
    supplement_tokens = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for source in provider.iter_sources():
                if (
                    not isinstance(source, TrainingSource)
                    or source.kind != "clean"
                    or source.source != protocol.source
                    or source.source_revision != protocol.source_revision
                    or source.source_split != protocol.source_split
                    or source.task is not None
                    or source.answer is not None
                    or source.metadata.get("tokenizer_model") != protocol.model
                    or source.metadata.get("tokenizer_revision") != protocol.model_revision
                ):
                    raise ValueError("SAE supplement provider emitted an invalid source")
                if (
                    source.record_id in record_ids
                    or source.source_id in source_ids
                    or source.group_id in group_ids
                ):
                    skipped_identity += 1
                    continue
                duplicate = guard.add(f"selected:{source.record_id}", source.clean_text)
                if duplicate is not None:
                    skipped_near_duplicate += 1
                    continue
                record_ids.add(source.record_id)
                source_ids.add(source.source_id)
                group_ids.add(source.group_id)
                selected.append(source)
                supplement_tokens += source.token_count
                handle.write(
                    json.dumps(
                        _source_payload(source),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                if supplement_tokens >= needed:
                    break
        if supplement_tokens < needed:
            raise ValueError("FineWeb-Edu stream was exhausted before the SAE source budget")
        os.replace(temporary, supplement_path)
    finally:
        temporary.unlink(missing_ok=True)
    registry_path = output / "source_registry.json"
    write_json_atomic(
        registry_path,
        {
            "schema_version": "robustness-sae-supplement-registry/v1",
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "training_budget": config.training_budget,
            "target_total_eligible_source_tokens": target,
            "prior_eligible_source_tokens": prepared.source_tokens,
            "supplement_records": len(selected),
            "supplement_source_tokens": supplement_tokens,
            "total_eligible_source_tokens": prepared.source_tokens + supplement_tokens,
            "supplement_record_ids_sha256": record_id_sha256(selected),
            "supplement_sha256": sha256_file(supplement_path),
            "exclusion_roles": sorted(roles),
            "exclusion_texts": exclusion_texts,
            "exclusion_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in exclusion_files
            ],
            "skipped_exact_identity": skipped_identity,
            "skipped_exact_or_near_duplicate": skipped_near_duplicate,
            "near_duplicate_method": (
                f"character-{protocol.near_duplicate_shingle_size}gram-"
                "minhash32-lsh8x4-exact-jaccard-"
                f"{protocol.near_duplicate_jaccard_threshold:g}/v1"
            ),
        },
    )
    run_path = output / "run.json"
    write_json_atomic(
        run_path,
        {
            "schema_version": "robustness-sae-clean-corpus-build-run/v1",
            "operation": "build-sae-clean-corpus",
            "status": "completed",
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "provider": dict(provider.provenance()),
            "outputs": {
                supplement_path.name: sha256_file(supplement_path),
                registry_path.name: sha256_file(registry_path),
            },
        },
    )
    return SaeCorpusBuildResult(
        supplement_path=supplement_path,
        registry_path=registry_path,
        run_path=run_path,
        supplement_records=len(selected),
        supplement_tokens=supplement_tokens,
        total_eligible_tokens=prepared.source_tokens + supplement_tokens,
    )


__all__ = [
    "HuggingFaceSaeCleanSourceProvider",
    "SaeCleanSourceProvider",
    "SaeCorpusBuildConfig",
    "SaeCorpusBuildResult",
    "run_build_sae_clean_corpus",
]

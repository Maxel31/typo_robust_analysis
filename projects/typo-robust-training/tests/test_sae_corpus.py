"""Leak-resistant construction of the clean SAE source supplement."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.data.records import CleanRecord
from typo_robust_training.sae.corpus import (
    HuggingFaceSaeCleanSourceProvider,
    SaeCorpusBuildConfig,
    _seed_exclusions,
    _segment_document,
    run_build_sae_clean_corpus,
)
from typo_robust_training.sae.data import prepare_sae_sources, sha256_file
from typo_robust_training.sae.duplicates import CharacterNgramDuplicateGuard
from typo_robust_training.training.pairs import TrainingSource


def _source(index: int, *, text: str | None = None, tokens: int = 2) -> TrainingSource:
    value = text or f"distinct educational content number {index} with a long suffix"
    source_id = f"fineweb_edu:{index}"
    payload = {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": hashlib.sha256(
            json.dumps(
                {
                    "source": "fineweb_edu",
                    "source_id": source_id,
                    "source_revision": "f" * 40,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "source": "fineweb_edu",
        "source_revision": "f" * 40,
        "source_split": "train",
        "source_id": source_id,
        "group_id": source_id,
        "split": "train",
        "text": value,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(value.encode()).hexdigest(),
        "normalized_content_sha256": hashlib.sha256(
            " ".join(value.casefold().split()).encode()
        ).hexdigest(),
        "metadata": {"tokenizer_model": "model", "tokenizer_revision": "a" * 40},
        "token_count": tokens,
    }
    return TrainingSource.from_dict(payload)


def _write_sources(path: Path, rows: tuple[TrainingSource, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "robustness-clean-record/v1",
                    "kind": "clean",
                    "record_id": row.record_id,
                    "source": row.source,
                    "source_revision": row.source_revision,
                    "source_split": row.source_split,
                    "source_id": row.source_id,
                    "group_id": row.group_id,
                    "split": "train",
                    "text": row.clean_text,
                    "task": None,
                    "answer": None,
                    "content_sha256": hashlib.sha256(row.clean_text.encode()).hexdigest(),
                    "normalized_content_sha256": hashlib.sha256(
                        " ".join(row.clean_text.casefold().split()).encode()
                    ).hexdigest(),
                    "metadata": dict(row.metadata),
                    "token_count": row.token_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _preregistration_for(path: Path, protocol: SimpleNamespace) -> SimpleNamespace:
    prepared = prepare_sae_sources(
        [path],
        protected_manifest_sha256=sha256_file(path),
        reserved_seed=protocol.reserved_order_seed,
        reserved_epoch=protocol.reserved_order_epoch,
        reserved_records=protocol.reserved_prefix_records,
    )
    return SimpleNamespace(
        source_manifest_sha256=sha256_file(path),
        sha256="d" * 64,
        initial_eligible_records=prepared.protected_eligible_records,
        initial_eligible_source_tokens=prepared.protected_eligible_source_tokens,
        initial_eligible_record_ids_sha256=prepared.protected_eligible_record_ids_sha256,
        eligible_records_removed=prepared.protected_normalized_duplicates_removed,
    )


def test_character_ngram_guard_rejects_normalized_and_near_duplicates() -> None:
    guard = CharacterNgramDuplicateGuard(shingle_size=3, threshold=0.80)
    assert guard.add("first", "The airport is located in Chicago.") is None
    assert guard.add("exact", "  the AIRPORT is located in Chicago. ") == "first"
    assert guard.add("near", "The airport is located in Chicagoo.") == "first"
    assert guard.add("other", "Sparse autoencoders expose residual features.") is None
    assert guard.records == 2


def test_exclusion_seeding_indexes_each_nontransitive_near_duplicate(tmp_path: Path) -> None:
    generator = random.Random(42)
    clean = "".join(generator.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(1500))

    def mutate(text: str, index: int) -> str:
        replacement = "z" if text[index] != "z" else "y"
        return f"{text[:index]}{replacement}{text[index + 1 :]}"

    typo = mutate(clean, 500)
    candidate = mutate(typo, 1000)
    exclusions = tmp_path / "frozen.jsonl"
    roles = (
        "tune",
        "pre_pr_gate",
        "final_test",
        "localization-selection",
        "localization-validation",
    )
    rows = []
    for index, role in enumerate(roles):
        rows.append(
            {
                "split": role,
                "text": typo if role == "pre_pr_gate" else f"unrelated frozen text {index}",
            }
        )
    exclusions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    guard = CharacterNgramDuplicateGuard(shingle_size=5, threshold=0.99)
    _seed_exclusions(
        guard=guard,
        existing=(_source(1, text=clean),),
        exclusion_files=(exclusions,),
    )

    assert guard.records == 1 + len(roles)
    assert guard.find_duplicate(candidate) is not None


def test_default_sae_source_provider_accepts_the_production_counter_contract() -> None:
    protocol = SimpleNamespace(
        model="google/gemma-3-4b-it",
        model_revision="a" * 40,
        max_sequence_length=512,
    )
    tokenizer = object()
    provider = HuggingFaceSaeCleanSourceProvider(protocol=protocol, tokenizer=tokenizer)
    assert provider.counter.tokenizer is tokenizer
    assert provider.counter.max_sequence_length == 512


def test_sae_source_provider_replays_fineweb_identity_convention(monkeypatch) -> None:
    class Tokenizer:
        def __call__(self, text, *, return_offsets_mapping=False, **_kwargs):
            result = {"input_ids": list(range(len(text)))}
            if return_offsets_mapping:
                result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
            return result

    row_id = "<urn:uuid:replayed>"
    dataset = ({"id": row_id, "text": "Clean educational text."},)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: dataset),
    )
    protocol = SimpleNamespace(
        model="google/gemma-3-4b-it",
        model_revision="a" * 40,
        max_sequence_length=512,
        source_dataset="HuggingFaceFW/fineweb-edu",
        source_subset="sample-10BT",
        source_split="train",
        source_revision="f" * 40,
        document_character_limit=8192,
    )
    provider = HuggingFaceSaeCleanSourceProvider(protocol=protocol, tokenizer=Tokenizer())
    (source,) = tuple(provider.iter_sources())
    assert source.source_id == f"fineweb_edu:{row_id}"
    assert source.group_id == f"fineweb_edu:{row_id}"


def test_sae_source_provider_skips_only_unpreparable_unspaced_rows(monkeypatch) -> None:
    class Tokenizer:
        def __call__(self, text, *, return_offsets_mapping=False, **_kwargs):
            retained = min(len(text), 10 if not any(char.isspace() for char in text) else len(text))
            result = {"input_ids": list(range(retained))}
            if return_offsets_mapping:
                result["offset_mapping"] = [(index, index + 1) for index in range(retained)]
            return result

    dataset = (
        {"id": "unspaced", "text": "文" * 20_000},
        {"id": "usable", "text": "Clean educational text."},
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: dataset),
    )
    protocol = SimpleNamespace(
        model="google/gemma-3-4b-it",
        model_revision="a" * 40,
        max_sequence_length=512,
        source_dataset="HuggingFaceFW/fineweb-edu",
        source_subset="sample-10BT",
        source_split="train",
        source_revision="f" * 40,
        document_character_limit=8192,
    )
    provider = HuggingFaceSaeCleanSourceProvider(protocol=protocol, tokenizer=Tokenizer())
    rows = tuple(provider.iter_sources())
    assert [row.source_id for row in rows] == ["fineweb_edu:usable"]
    assert provider.skipped_unsegmentable == 1


@pytest.mark.parametrize("text", ("A" * 20_000, "文" * 20_000))
def test_sae_segmentation_keeps_long_documents_without_whitespace(text: str) -> None:
    record = CleanRecord(
        source="fineweb_edu",
        source_revision="f" * 40,
        source_split="train",
        source_id="fineweb_edu:unspaced",
        group_id="fineweb_edu:unspaced",
        text=text,
        task=None,
        answer=None,
        metadata={},
    )
    segmented = _segment_document(record, character_limit=8192)
    assert len(segmented.text) == 8192
    assert segmented.metadata["document_window_strategy"] == (
        "fixed-character-hash-window-no-whitespace"
    )


def test_sae_corpus_builder_requires_all_roles_and_reaches_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_path = tmp_path / "existing.jsonl"
    existing = tuple(_source(index) for index in range(3))
    _write_sources(existing_path, existing)
    exclusion_path = tmp_path / "excluded.jsonl"
    roles = (
        "tune",
        "pre_pr_gate",
        "final_test",
        "localization-selection",
        "localization-validation",
    )
    exclusion_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": f"excluded-{index}",
                    "source_id": f"excluded-source-{index}",
                    "group_id": f"excluded-group-{index}",
                    "split": role,
                    "text": f"held out document role {role} unique {index}",
                }
            )
            + "\n"
            for index, role in enumerate(roles)
        ),
        encoding="utf-8",
    )
    protocol = SimpleNamespace(
        config_sha256="c" * 64,
        reserved_order_seed=42,
        reserved_order_epoch=0,
        reserved_prefix_records=1,
        source_revision="f" * 40,
        source="fineweb_edu",
        source_split="train",
        model="model",
        model_revision="a" * 40,
        near_duplicate_shingle_size=3,
        near_duplicate_jaccard_threshold=0.99,
        minimum_training_tokens=10,
        preferred_training_tokens=20,
        statistics_tokens=2,
        splice_documents=1,
        max_sequence_length=2,
    )
    preregistration = _preregistration_for(existing_path, protocol)
    monkeypatch.setattr(
        "typo_robust_training.sae.corpus.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.corpus.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )

    class Provider:
        def iter_sources(self):
            yield existing[0]
            for index in range(10, 20):
                yield _source(index)

        def provenance(self):
            return {"provider": "fake-clean-stream/v1"}

    output = tmp_path / "output"
    result = run_build_sae_clean_corpus(
        SaeCorpusBuildConfig(
            config_path=tmp_path / "config.json",
            registry_path=tmp_path / "registry.json",
            existing_data_paths=(existing_path,),
            exclusion_paths=(exclusion_path,),
            training_budget="minimum",
            output_dir=output,
        ),
        source_provider=Provider(),
    )
    assert result.supplement_records == 5
    assert result.supplement_tokens == 10
    assert result.total_eligible_tokens == 14
    assert result.supplement_path.is_file()
    registry = json.loads(result.registry_path.read_text(encoding="utf-8"))
    assert registry["exclusion_roles"] == sorted(roles)
    assert registry["skipped_exact_identity"] == 1


def test_sae_corpus_builder_blocks_deduplicated_identity_with_new_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_path = tmp_path / "existing.jsonl"
    duplicate_text = "same normalized protected document with sufficient educational content"
    existing = (
        _source(0, text=duplicate_text),
        _source(1, text=duplicate_text),
        _source(2),
        _source(3),
    )
    _write_sources(existing_path, existing)
    exclusion_path = tmp_path / "excluded.jsonl"
    roles = (
        "tune",
        "pre_pr_gate",
        "final_test",
        "localization-selection",
        "localization-validation",
    )
    exclusion_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": f"excluded-{index}",
                    "source_id": f"excluded-source-{index}",
                    "group_id": f"excluded-group-{index}",
                    "split": role,
                    "text": f"held out document role {role} unique {index}",
                }
            )
            + "\n"
            for index, role in enumerate(roles)
        ),
        encoding="utf-8",
    )
    protocol = SimpleNamespace(
        config_sha256="c" * 64,
        reserved_order_seed=42,
        reserved_order_epoch=0,
        reserved_prefix_records=1,
        source_revision="f" * 40,
        source="fineweb_edu",
        source_split="train",
        model="model",
        model_revision="a" * 40,
        near_duplicate_shingle_size=3,
        near_duplicate_jaccard_threshold=0.99,
        minimum_training_tokens=10,
        preferred_training_tokens=20,
        statistics_tokens=2,
        splice_documents=1,
        max_sequence_length=2,
    )
    preregistration = _preregistration_for(existing_path, protocol)
    prepared = prepare_sae_sources(
        [existing_path],
        protected_manifest_sha256=preregistration.source_manifest_sha256,
        reserved_seed=42,
        reserved_epoch=0,
        reserved_records=1,
    )
    retained_ids = {row.record_id for row in (*prepared.reserved, *prepared.sources)}
    (dropped,) = tuple(row for row in existing if row.record_id not in retained_ids)
    replayed = replace(
        dropped,
        clean_text="same source identity replayed from a different document segment",
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.corpus.load_sae_protocol",
        lambda _path: protocol,
    )
    monkeypatch.setattr(
        "typo_robust_training.sae.corpus.load_sae_preregistration",
        lambda _path, protocol: preregistration,
    )

    class Provider:
        def iter_sources(self):
            yield replayed
            for index in range(10, 20):
                yield _source(index)

        def provenance(self):
            return {"provider": "fake-clean-stream/v1"}

    output = tmp_path / "output"
    result = run_build_sae_clean_corpus(
        SaeCorpusBuildConfig(
            config_path=tmp_path / "config.json",
            registry_path=tmp_path / "registry.json",
            existing_data_paths=(existing_path,),
            exclusion_paths=(exclusion_path,),
            training_budget="minimum",
            output_dir=output,
        ),
        source_provider=Provider(),
    )

    rows = [json.loads(line) for line in result.supplement_path.read_text().splitlines()]
    assert dropped.record_id not in {row["record_id"] for row in rows}
    registry = json.loads(result.registry_path.read_text(encoding="utf-8"))
    assert registry["skipped_exact_identity"] == 1
    assert registry["protected_normalized_duplicates_removed"] == 1

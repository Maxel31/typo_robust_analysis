"""Leak-resistant construction of the clean SAE source supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from typo_robust_training.sae.corpus import (
    SaeCorpusBuildConfig,
    run_build_sae_clean_corpus,
)
from typo_robust_training.sae.data import sha256_file
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


def test_character_ngram_guard_rejects_normalized_and_near_duplicates() -> None:
    guard = CharacterNgramDuplicateGuard(shingle_size=3, threshold=0.80)
    assert guard.add("first", "The airport is located in Chicago.") is None
    assert guard.add("exact", "  the AIRPORT is located in Chicago. ") == "first"
    assert guard.add("near", "The airport is located in Chicagoo.") == "first"
    assert guard.add("other", "Sparse autoencoders expose residual features.") is None
    assert guard.records == 2


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
    preregistration = SimpleNamespace(
        source_manifest_sha256=sha256_file(existing_path),
        sha256="d" * 64,
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

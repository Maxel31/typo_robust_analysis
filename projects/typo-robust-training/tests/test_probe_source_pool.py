from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from typo_robust_training.cli import register_commands
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.integrity import sha256_file
from typo_robust_training.probe import source_pool
from typo_robust_training.probe.cohort_builder import (
    probe_parent_source_sha256,
    probe_source_group_sha256,
)
from typo_robust_training.probe.source_pool import (
    ProbeSourcePoolFreezeConfig,
    freeze_probe_source_pool,
    load_probe_source_pool_bundle,
)
from typo_robust_training.training.pairs import TrainingSource


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class _Checkout:
    revision: str

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "typo_robust_training_tree": "1" * 40,
            "typo_cot_tree": "2" * 40,
            "typo_cot_runtime_sources": ["projects/typo-cot/src/typo_cot/__init__.py"],
        }


def _row(identifier: str, text: str, *, tokens: int = 12) -> dict[str, object]:
    return {
        "text": text,
        "id": identifier,
        "dump": "CC-MAIN-2026-01",
        "url": f"https://example.test/{identifier}",
        "file_path": f"crawl/{identifier}.warc.gz",
        "language": "en",
        "language_score": 0.99,
        "token_count": tokens,
        "score": 4.2,
        "int_score": 4,
    }


def _source(identifier: str, text: str) -> TrainingSource:
    record = CleanRecord(
        source="fineweb_edu",
        source_revision=source_pool._REVISION,
        source_split="train",
        source_id=f"fineweb_edu:{identifier}",
        group_id=f"fineweb_edu:{identifier}",
        text=text,
        task=None,
        answer=None,
    )
    return TrainingSource.from_dict(
        {
            "schema_version": "robustness-clean-record/v1",
            "kind": "clean",
            "record_id": record.record_id,
            "source": record.source,
            "source_revision": record.source_revision,
            "source_split": record.source_split,
            "source_id": record.source_id,
            "group_id": record.group_id,
            "split": "train",
            "text": text,
            "task": None,
            "answer": None,
            "content_sha256": _sha(text),
            "normalized_content_sha256": _sha(text.casefold().strip()),
            "metadata": {},
            "token_count": 1,
        }
    )


def _registry(path: Path, *, protected: TrainingSource | None = None) -> Path:
    tiers = ("training", "localization", "tune", "pre-pr", "sealed")
    rows: list[dict[str, object]] = []
    for tier in tiers:
        groups = [_sha(f"group-{tier}")]
        parents = [_sha(f"parent-{tier}")]
        contents = [_sha(f"content-{tier}")]
        if tier == "training" and protected is not None:
            groups.append(probe_source_group_sha256(protected))
            parents.append(probe_parent_source_sha256(protected))
            contents.append(_sha(protected.clean_text.casefold().strip()))
        rows.append(
            {
                "tier": tier,
                "source_group_sha256": groups,
                "parent_source_sha256": parents,
                "normalized_content_sha256": contents,
            }
        )
    path.write_text(
        json.dumps(
            {"schema_version": "typo-protected-split-registry/v1", "registries": rows},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _config(
    tmp_path: Path,
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ProbeSourcePoolFreezeConfig:
    parquet = tmp_path / "013_00000.parquet"
    parquet.write_bytes(b"pinned-test-parquet")
    monkeypatch.setattr(source_pool, "_SHARD_SHA256", sha256_file(parquet))
    return ProbeSourcePoolFreezeConfig(
        parquet_path=parquet,
        parquet_sha256=sha256_file(parquet),
        protected_registry_path=registry,
        protected_registry_sha256=sha256_file(registry),
        code_revision="a" * 40,
        output_dir=tmp_path / "pool",
    )


def _patch_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_pool,
        "attest_runtime_checkout",
        lambda revision: _Checkout(revision),
    )


def test_freezer_excludes_protected_and_normalized_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checkout(monkeypatch)
    protected = _source("protected", "this protected document must never be reused")
    registry = _registry(tmp_path / "protected.json", protected=protected)
    config = _config(tmp_path, registry, monkeypatch)
    rows = (
        _row("eligible", "A clean eligible document with a stable word identity."),
        _row("duplicate", "  a CLEAN eligible document with a stable word identity.  "),
        _row("protected", protected.clean_text),
    )

    result = freeze_probe_source_pool(config, row_provider=lambda _path: iter(rows))

    assert result.records == 1
    assert result.protected_records_removed == 1
    assert result.duplicate_records_removed == 1
    assert result.run_path.is_file()
    frozen = load_probe_source_pool_bundle(
        result.run_path,
        expected_run_sha256=result.run_sha256,
        expected_code_revision="a" * 40,
    )
    assert frozen.records == 1
    source_text = result.source_manifest_path.read_text(encoding="utf-8")
    assert "eligible" in source_text
    assert "protected document" not in source_text
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["model_outputs_observed"] is False
    assert run["token_count"]["used_for_probe_selection"] is False


def test_freezer_rejects_wrong_fixed_shard_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)
    config = ProbeSourcePoolFreezeConfig(
        parquet_path=config.parquet_path,
        parquet_sha256="f" * 64,
        protected_registry_path=config.protected_registry_path,
        protected_registry_sha256=config.protected_registry_sha256,
        code_revision=config.code_revision,
        output_dir=config.output_dir,
    )
    with pytest.raises(ValueError, match="preregistered final"):
        freeze_probe_source_pool(config, row_provider=lambda _path: iter(()))


def test_freezer_never_clobbers_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)
    config.output_dir.mkdir()
    marker = config.output_dir / "owned"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_probe_source_pool(
            config,
            row_provider=lambda _path: iter((_row("a", "enough source text for a row"),)),
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_loader_rejects_rehashed_adjacent_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)
    result = freeze_probe_source_pool(
        config,
        row_provider=lambda _path: iter((_row("a", "enough source text for a row"),)),
    )
    result.source_manifest_path.write_text(
        result.source_manifest_path.read_text(encoding="utf-8").replace("enough", "altered"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact differs"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256=result.run_sha256,
            expected_code_revision="a" * 40,
        )


def test_loader_requires_external_run_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)
    result = freeze_probe_source_pool(
        config,
        row_provider=lambda _path: iter((_row("a", "enough source text for a row"),)),
    )
    with pytest.raises(ValueError, match="externally pinned"):
        load_probe_source_pool_bundle(
            result.run_path,
            expected_run_sha256="0" * 64,
            expected_code_revision="a" * 40,
        )


def test_freezer_rejects_symlinked_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)
    link = tmp_path / "link.parquet"
    link.symlink_to(config.parquet_path)
    config = ProbeSourcePoolFreezeConfig(
        parquet_path=link,
        parquet_sha256=config.parquet_sha256,
        protected_registry_path=config.protected_registry_path,
        protected_registry_sha256=config.protected_registry_sha256,
        code_revision=config.code_revision,
        output_dir=config.output_dir,
    )
    with pytest.raises(ValueError, match="regular file"):
        freeze_probe_source_pool(config, row_provider=lambda _path: iter(()))


def test_cli_requires_every_external_identity() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    register_commands(commands)
    args = parser.parse_args(
        [
            "freeze-probe-source-pool",
            "--source-parquet",
            "shard.parquet",
            "--source-parquet-sha256",
            "a" * 64,
            "--protected-registry",
            "protected.json",
            "--protected-registry-sha256",
            "b" * 64,
            "--code-revision",
            "c" * 40,
            "--output-dir",
            "output",
        ]
    )
    assert args.source_parquet == Path("shard.parquet")
    assert args.protected_registry == Path("protected.json")
    assert args.code_revision == "c" * 40


def test_freezer_rejects_parquet_toctou(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_checkout(monkeypatch)
    registry = _registry(tmp_path / "protected.json")
    config = _config(tmp_path, registry, monkeypatch)

    def mutate(path: Path):
        yield _row("a", "enough source text for a row")
        path.write_bytes(b"changed-after-pinned-read")

    with pytest.raises(ValueError, match="changed while"):
        freeze_probe_source_pool(config, row_provider=mutate)
    assert not config.output_dir.exists()

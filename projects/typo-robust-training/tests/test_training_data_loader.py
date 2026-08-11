"""Training refuses data artifacts whose hashes or frozen typo protocol drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typo_robust_training.data.splits import normalized_content_sha256
from typo_robust_training.training.config import load_adapter_training_config
from typo_robust_training.training.data import load_training_data_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/gemma4b-targeted-lora.yaml"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    text = "Educational airport context remains useful."
    source = {
        "schema_version": "robustness-clean-record/v1",
        "kind": "clean",
        "record_id": "a" * 64,
        "source": "fineweb_edu",
        "source_revision": "b" * 40,
        "source_split": "train",
        "source_id": "source-1",
        "group_id": "group-1",
        "split": "train",
        "text": text,
        "task": None,
        "answer": None,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_content_sha256": normalized_content_sha256(text),
        "metadata": {},
        "token_count": 7,
    }
    training = root / "training_sources.jsonl"
    training.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    statistics = {
        "schema_version": "natural-typo-statistics/v1",
        "source_role": "natural-training-repositories-only",
        "input_records": 1,
        "contributing_records": 1,
        "contributing_record_ids_sha256": "c" * 64,
        "held_out_operations": ["adjacent-transposition"],
        "operation_counts": {},
        "substitutions": {character: {"z" if character != "z" else "x": 1} for character in "abcdefghijklmnopqrstuvwxyz"},
    }
    stats = root / "typo_statistics.json"
    _write_json(stats, statistics)
    hashes = {
        training.name: hashlib.sha256(training.read_bytes()).hexdigest(),
        stats.name: hashlib.sha256(stats.read_bytes()).hexdigest(),
    }
    evaluation = {
        "schema_version": "robustness-evaluation-manifest/v1",
        "artifact_sha256": hashes,
        "data_identity_sha256": "d" * 64,
    }
    evaluation_path = root / "evaluation_manifest.json"
    _write_json(evaluation_path, evaluation)
    hashes[evaluation_path.name] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    _write_json(
        root / "run.json",
        {
            "schema_version": "build-robustness-training-data-run/v1",
            "status": "completed",
            "protocol": {
                "model": "google/gemma-3-4b-it",
                "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
                "training": {
                    "max_sequence_length": 512,
                    "explicit_clean_pair_probability": 0.10,
                },
                "typos": {
                    "edit_count_probabilities": {"1": 0.50, "2": 0.30, "3-4": 0.20},
                    "operation_probabilities": {
                        "keyboard-neighbor-substitution": 0.20,
                        "deletion": 0.20,
                        "insertion": 0.20,
                        "duplication": 0.20,
                        "natural-statistics-substitution": 0.20,
                    },
                    "held_out_operations": ["adjacent-transposition"],
                    "minimum_word_letters": 2,
                },
            },
            "outputs": {name: {"sha256": digest} for name, digest in hashes.items()},
        },
    )
    return root


def test_training_data_bundle_revalidates_hashes_and_builds_seeded_generator(
    tmp_path: Path,
) -> None:
    bundle = load_training_data_bundle(
        _fixture(tmp_path),
        protocol=load_adapter_training_config(CONFIG),
        seed=43,
    )
    assert len(bundle.sources) == 1
    assert bundle.data_identity_sha256 == "d" * 64
    assert len(bundle.training_data_sha256) == 64
    assert bundle.generator.seed == 43
    assert bundle.generator.explicit_clean_pair_probability == pytest.approx(0.10)
    assert "adjacent-transposition" not in bundle.generator.operation_weights


def test_training_data_bundle_rejects_artifact_tampering(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    with (root / "training_sources.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        load_training_data_bundle(
            root,
            protocol=load_adapter_training_config(CONFIG),
            seed=42,
        )

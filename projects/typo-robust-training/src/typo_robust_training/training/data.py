"""Load and revalidate immutable builder artifacts before adapter training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.typo_stats import substitutions_from_statistics
from typo_robust_training.training.config import AdapterTrainingProtocol
from typo_robust_training.training.pairs import TrainingSource


@dataclass(frozen=True, slots=True)
class TrainingDataBundle:
    root: Path
    sources: tuple[TrainingSource, ...]
    generator: TypoGenerator
    data_identity_sha256: str
    training_data_sha256: str
    artifact_sha256: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"training data artifact is missing: {path}")
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"training data artifact must contain an object: {path}")
    return payload


def _declared_hash(run: Mapping[str, object], *, name: str, path: Path) -> str:
    outputs = run.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(name), Mapping):
        raise ValueError(f"training data run has no hash for {name}")
    expected = outputs[name].get("sha256")  # type: ignore[union-attr]
    actual = _sha256_file(path)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"training data artifact hash differs: {name}")
    return actual


def _probabilities(value: object, *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    result: dict[str, float] = {}
    for name, weight in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError(f"{field} contains an invalid probability")
        result[name] = float(weight)
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise ValueError(f"{field} probabilities must sum to one")
    return result


def _sources(path: Path) -> tuple[TrainingSource, ...]:
    rows: list[TrainingSource] = []
    for line_number, line in read_lf_jsonl_lines(path, context="training sources"):
        payload = strict_loads(line, context=f"{path}:{line_number}")
        rows.append(TrainingSource.from_dict(payload))
    if not rows:
        raise ValueError("training_sources.jsonl is empty")
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("training source record IDs are duplicated")
    return tuple(rows)


def load_training_data_bundle(
    root: Path,
    *,
    protocol: AdapterTrainingProtocol,
    seed: int,
) -> TrainingDataBundle:
    """Revalidate builder hashes and reconstruct the frozen typo generator."""

    if not isinstance(protocol, AdapterTrainingProtocol):
        raise TypeError("training data requires AdapterTrainingProtocol")
    if seed not in protocol.seed_inventory:
        raise ValueError("training seed is outside the frozen seed inventory")
    resolved = Path(root).resolve()
    run_path = resolved / "run.json"
    source_path = resolved / "training_sources.jsonl"
    statistics_path = resolved / "typo_statistics.json"
    evaluation_path = resolved / "evaluation_manifest.json"
    run = _object(run_path)
    if (
        run.get("schema_version") != "build-robustness-training-data-run/v1"
        or run.get("status") != "completed"
    ):
        raise ValueError("training data build is not completed")
    data_protocol = run.get("protocol")
    if not isinstance(data_protocol, Mapping):
        raise ValueError("training data run has no protocol")
    if (
        data_protocol.get("model") != protocol.model
        or data_protocol.get("model_revision") != protocol.model_revision
    ):
        raise ValueError("training data model identity differs from adapter config")
    training = data_protocol.get("training")
    typos = data_protocol.get("typos")
    if not isinstance(training, Mapping) or not isinstance(typos, Mapping):
        raise ValueError("training data protocol sections are missing")
    if training.get("max_sequence_length") != protocol.max_sequence_length:
        raise ValueError("training data max sequence length differs from adapter config")
    clean_probability = training.get("explicit_clean_pair_probability")
    if (
        isinstance(clean_probability, bool)
        or not isinstance(clean_probability, (int, float))
        or not 0.0 <= float(clean_probability) <= 1.0
    ):
        raise ValueError("training data explicit-clean probability differs")
    operations = _probabilities(
        typos.get("operation_probabilities"), field="typos.operation_probabilities"
    )
    edit_counts = _probabilities(
        typos.get("edit_count_probabilities"), field="typos.edit_count_probabilities"
    )
    held_out = typos.get("held_out_operations")
    if held_out != ["adjacent-transposition"] or "adjacent-transposition" in operations:
        raise ValueError("held-out typo operation reached the training generator")
    minimum_letters = typos.get("minimum_word_letters")
    if (
        isinstance(minimum_letters, bool)
        or not isinstance(minimum_letters, int)
        or minimum_letters < 2
    ):
        raise ValueError("training minimum word length differs")

    artifacts = {
        "training_sources.jsonl": _declared_hash(
            run, name="training_sources.jsonl", path=source_path
        ),
        "typo_statistics.json": _declared_hash(
            run, name="typo_statistics.json", path=statistics_path
        ),
        "evaluation_manifest.json": _declared_hash(
            run, name="evaluation_manifest.json", path=evaluation_path
        ),
    }
    evaluation = _object(evaluation_path)
    if evaluation.get("schema_version") != "robustness-evaluation-manifest/v1":
        raise ValueError("training evaluation manifest schema differs")
    evaluation_hashes = evaluation.get("artifact_sha256")
    if not isinstance(evaluation_hashes, Mapping) or any(
        evaluation_hashes.get(name) != artifacts[name]
        for name in ("training_sources.jsonl", "typo_statistics.json")
    ):
        raise ValueError("evaluation manifest training artifact hash differs")
    identity = evaluation.get("data_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("evaluation manifest data identity differs")
    statistics = _object(statistics_path)
    substitutions = substitutions_from_statistics(statistics)
    if statistics.get("held_out_operations") != ["adjacent-transposition"]:
        raise ValueError("natural typo statistics held-out inventory differs")
    sources = _sources(source_path)
    binding = hashlib.sha256(
        json.dumps(
            {
                "artifacts": artifacts,
                "data_identity_sha256": identity,
                "protocol": data_protocol,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return TrainingDataBundle(
        root=resolved,
        sources=sources,
        generator=TypoGenerator(
            seed=seed,
            operation_weights=operations,
            edit_count_weights=edit_counts,
            natural_substitutions=substitutions,
            explicit_clean_pair_probability=float(clean_probability),
            minimum_word_letters=minimum_letters,
        ),
        data_identity_sha256=identity,
        training_data_sha256=binding,
        artifact_sha256=artifacts,
    )


__all__ = ["TrainingDataBundle", "load_training_data_bundle"]

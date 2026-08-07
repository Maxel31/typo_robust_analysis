"""Resumable writer for versioned clean/edited pair records."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.experiments.catalog import PAPER_SHA256

PUBLIC_BENCHMARKS = ("gsm8k", "math-500", "mmlu", "mmlu-pro", "arc", "csqa")
TARGETING_CONDITIONS = ("attribution-4", "random-4")


class PairPreparationRunError(RuntimeError):
    """Raised after a run records one or more per-item failures."""


@dataclass(frozen=True, slots=True)
class PrepareEditedPairsConfig:
    """Frozen arguments for one model/benchmark/targeting preparation run."""

    model: str
    benchmark: str
    targeting: Literal["attribution-4", "random-4"]
    num_edits: int
    output_dir: Path
    seed: int = 42
    max_new_tokens: int = 512
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in PUBLIC_BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if self.targeting not in TARGETING_CONDITIONS:
            raise ValueError(f"unsupported targeting condition: {self.targeting!r}")
        if not 1 <= self.num_edits <= 4:
            raise ValueError("num_edits must be between 1 and 4")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")

    def public_arguments(self) -> dict[str, object]:
        """Return stable manifest arguments, excluding the transport-only resume flag."""
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class PrepareEditedPairsResult:
    """Paths and counts returned after a completed run."""

    pairs_path: Path
    run_path: Path
    written: int


class PairPreparationRuntime(Protocol):
    """Runtime seam used by the GPU implementation and CPU integration tests."""

    def load_samples(self, config: PrepareEditedPairsConfig) -> Sequence[Any]: ...

    def prepare_pair(
        self, sample: Any, config: PrepareEditedPairsConfig
    ) -> Mapping[str, object]: ...

    def provenance(self) -> Mapping[str, object]: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sample_id(sample: Any) -> str:
    if isinstance(sample, Mapping):
        value = sample.get("sample_id")
    else:
        value = getattr(sample, "sample_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("every sample must expose a non-empty sample_id")
    return value


def _record_path(records_dir: Path, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return records_dir / f"{digest}.json"


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _manifest(
    *,
    config: PrepareEditedPairsConfig,
    status: str,
    started_at: str,
    discovered: int,
    written: int,
    failures: list[dict[str, str]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "prepare-edited-pairs-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "prepare-edited-pairs",
        "status": status,
        "arguments": config.public_arguments(),
        "decoding": {
            "strategy": "greedy",
            "dtype": "bfloat16",
            "padding_side": "left",
            "max_new_tokens": config.max_new_tokens,
        },
        "counts": {
            "discovered": discovered,
            "written": written,
            "failed": len(failures),
        },
        "failures": failures,
        "provenance": dict(provenance),
        "started_at": started_at,
        "updated_at": _now(),
    }


def _validate_resume(
    run_path: Path, config: PrepareEditedPairsConfig
) -> tuple[str, dict[str, object]]:
    previous = _load_json(run_path)
    if previous.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise ValueError(f"cannot resume an unknown run schema: {run_path}")
    if previous.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    started_at = previous.get("started_at")
    return (started_at if isinstance(started_at, str) else _now()), previous


_COMPLETED_RESUME_PROTOCOL_FIELDS = (
    "model",
    "benchmark_dataset_loader",
    "dataset_cohort_rule",
    "dataset_samples_per_subset",
    "random_seed_algorithm",
    "target_position",
    "alignment",
    "historical_compatibility_notes",
)


def _validate_preload_resume_provenance(
    previous_manifest: Mapping[str, object],
    current_provenance: Mapping[str, object],
    *,
    protocol_only: bool = False,
) -> None:
    previous_provenance = previous_manifest.get("provenance")
    fields = (
        [key for key in _COMPLETED_RESUME_PROTOCOL_FIELDS if key in current_provenance]
        if protocol_only
        else list(current_provenance)
    )
    # Injected CPU test runtimes may expose only one synthetic provenance key.
    # Preserve their completed-resume contract while production manifests use
    # the explicit protocol field set above.
    if not fields:
        fields = list(current_provenance)
    if not isinstance(previous_provenance, Mapping) or any(
        previous_provenance.get(key) != current_provenance.get(key) for key in fields
    ):
        raise ValueError(
            "resume provenance does not match the current environment or protocol "
            "before model loading"
        )


def run_prepare_edited_pairs(
    config: PrepareEditedPairsConfig,
    *,
    runtime: PairPreparationRuntime | None = None,
) -> PrepareEditedPairsResult:
    """Prepare every selected item, resume safely, and finalize the two public outputs."""
    output_dir = config.output_dir
    run_path = output_dir / "run.json"
    pairs_path = output_dir / "pairs.jsonl"
    work_dir = output_dir / ".prepare-edited-pairs-work"
    records_dir = work_dir / "records"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.exists():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest: dict[str, object] | None = None
    if config.resume:
        started_at, previous_manifest = _validate_resume(run_path, config)
        previous = previous_manifest
        if previous.get("status") == "completed":
            if not pairs_path.exists():
                raise ValueError(
                    "completed run is missing pairs.jsonl; restore the published file or use "
                    "a new output directory"
                )
            if runtime is None:
                from typo_cot.experiments.prepare_edited_pairs.runtime import (
                    preload_provenance,
                )

                current_preload_provenance = preload_provenance(config)
            else:
                current_preload_provenance = runtime.provenance()
            _validate_preload_resume_provenance(
                previous,
                current_preload_provenance,
                protocol_only=True,
            )
            counts = previous.get("counts")
            written = int(counts.get("written", 0)) if isinstance(counts, Mapping) else 0
            return PrepareEditedPairsResult(pairs_path, run_path, written)
    else:
        started_at = _now()

    # Validate paths and completed resumes before loading a multi-billion
    # parameter model. This also makes a mistaken output directory fail fast.
    if runtime is None:
        from typo_cot.experiments.prepare_edited_pairs.runtime import (
            HuggingFacePairPreparationRuntime,
            preload_provenance,
        )

        if previous_manifest is not None:
            current_preload_provenance = preload_provenance(config)
            _validate_preload_resume_provenance(
                previous_manifest,
                current_preload_provenance,
            )
        runtime = HuggingFacePairPreparationRuntime(config)

    samples = sorted(runtime.load_samples(config), key=_sample_id)
    if config.limit is not None:
        samples = samples[: config.limit]
    if not samples:
        raise ValueError("no benchmark samples were selected")
    sample_ids = [_sample_id(sample) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("dataset returned duplicate sample IDs")

    provenance = dict(runtime.provenance())
    if previous_manifest is not None:
        previous_provenance = previous_manifest.get("provenance")
        if not isinstance(previous_provenance, Mapping) or dict(previous_provenance) != provenance:
            raise ValueError(
                "resume provenance does not match the original model, dataset, environment, "
                "or protocol"
            )

    records_dir.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    for path in records_dir.glob("*.json"):
        record = _load_json(path)
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            existing.add(sample_id)

    _write_json_atomic(
        run_path,
        _manifest(
            config=config,
            status="running",
            started_at=started_at,
            discovered=len(samples),
            written=len(existing & set(sample_ids)),
            failures=[],
            provenance=provenance,
        ),
    )

    failures: list[dict[str, str]] = []
    by_id = {_sample_id(sample): sample for sample in samples}
    selected_ids = set(sample_ids)
    completed_count = len(existing & selected_ids)
    pending_ids = [sample_id for sample_id in sample_ids if sample_id not in existing]
    for sample_id in tqdm(
        pending_ids,
        desc="prepare-edited-pairs",
        unit="pair",
        initial=completed_count,
        total=len(sample_ids),
        disable=None,
    ):
        try:
            record = dict(runtime.prepare_pair(by_id[sample_id], config))
            if record.get("schema_version") != "prepare-edited-pairs/v1":
                raise ValueError("runtime returned an unknown pair schema")
            if record.get("sample_id") != sample_id:
                raise ValueError("runtime record sample_id does not match the input sample")
            _write_json_atomic(_record_path(records_dir, sample_id), record)
            existing.add(sample_id)
        except Exception as exc:  # noqa: BLE001 - record item failure for resumability
            failures.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                status="running",
                started_at=started_at,
                discovered=len(samples),
                written=len(existing & selected_ids),
                failures=failures,
                provenance=provenance,
            ),
        )

    if failures:
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                status="failed",
                started_at=started_at,
                discovered=len(samples),
                written=len(existing & selected_ids),
                failures=failures,
                provenance=provenance,
            ),
        )
        raise PairPreparationRunError(
            f"{len(failures)} item(s) failed; inspect run.json and rerun with --resume"
        )

    temporary_pairs = work_dir / "pairs.jsonl.tmp"
    with temporary_pairs.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            record = _load_json(_record_path(records_dir, sample_id))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_pairs, pairs_path)

    _write_json_atomic(
        run_path,
        _manifest(
            config=config,
            status="completed",
            started_at=started_at,
            discovered=len(samples),
            written=len(samples),
            failures=[],
            provenance=provenance,
        ),
    )
    shutil.rmtree(work_dir)
    return PrepareEditedPairsResult(pairs_path, run_path, len(samples))

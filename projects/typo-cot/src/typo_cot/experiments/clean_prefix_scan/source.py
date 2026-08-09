"""Strict read-only source adapters for the clean-prefix scan.

The primary cohort is the exact clean-to-edited denominator of a completed
fixed-window ``[0, 6)`` run.  Extension cohorts are built from the two
completed prepared-pair targeting arms.  This module deliberately delegates
JSON and semantic validation to the producer loaders; it only projects their
validated objects into one common representation and freezes every source-file
fingerprint for later mutation checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from typo_cot.experiments.fixed_window_answer_patching import LayerWindow
from typo_cot.experiments.fixed_window_answer_patching import runner as fixed_runner
from typo_cot.experiments.patch_coordinate_controls import runner as coordinate_runner


SourceCohort = Literal["primary", "extension"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TARGETING_ORDER = {"attribution-4": 0, "random-4": 1}
_PRIMARY_WINDOW = LayerWindow(0, 6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """One byte-exact upstream file captured after semantic validation."""

    role: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("source file role must be a non-empty string")
        object.__setattr__(self, "path", Path(self.path).resolve())
        _require_sha256(self.sha256, field=f"source file {self.role} sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceCase:
    """One validated prepared pair in a primary or extension source cohort."""

    cohort: SourceCohort
    targeting: str
    sample_id: str
    record: Mapping[str, object]
    source_record_sha256: str
    prepared_clean_correct: bool
    prepared_edited_wrong: bool
    aligned_word_count: int
    reference_event: bool | None = None
    reference_recipient_generation_identical: bool | None = None

    def __post_init__(self) -> None:
        if self.cohort not in {"primary", "extension"}:
            raise ValueError("source case cohort must be primary or extension")
        if self.targeting not in _TARGETING_ORDER:
            raise ValueError("source case targeting must be attribution-4 or random-4")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("source case sample_id must be a non-empty string")
        if not isinstance(self.record, Mapping):
            raise TypeError("source case record must be a JSON object")
        _require_sha256(
            self.source_record_sha256,
            field="source case source_record_sha256",
        )
        if type(self.prepared_clean_correct) is not bool:
            raise TypeError("prepared_clean_correct must be boolean")
        if type(self.prepared_edited_wrong) is not bool:
            raise TypeError("prepared_edited_wrong must be boolean")
        if (
            not isinstance(self.aligned_word_count, int)
            or isinstance(self.aligned_word_count, bool)
            or self.aligned_word_count < 0
        ):
            raise ValueError("aligned_word_count must be a non-negative integer")
        for field in (
            "reference_event",
            "reference_recipient_generation_identical",
        ):
            value = getattr(self, field)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{field} must be boolean or null")
        if self.cohort == "primary" and self.reference_event is None:
            raise ValueError("primary source cases must retain their fixed-window event")
        if self.cohort == "extension" and (
            self.reference_event is not None
            or self.reference_recipient_generation_identical is not None
        ):
            raise ValueError("extension source cases must not claim fixed-window results")

    @property
    def key(self) -> tuple[str, str]:
        return self.targeting, self.sample_id

    def identity_dict(self) -> dict[str, object]:
        return {
            "cohort": self.cohort,
            "targeting": self.targeting,
            "sample_id": self.sample_id,
            "source_record_sha256": self.source_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Immutable source identity shared by both clean-prefix cohort modes."""

    cohort: SourceCohort
    model: str
    benchmark: str
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int
    cases: tuple[SourceCase, ...]
    source_files: tuple[SourceFileSnapshot, ...]
    cohort_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "source_files", tuple(self.source_files))
        if self.cohort not in {"primary", "extension"}:
            raise ValueError("source bundle cohort must be primary or extension")
        for field in ("model", "benchmark", "model_revision"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"source bundle {field} must be a non-empty string")
        _require_sha256(
            self.dataset_records_sha256,
            field="source bundle dataset_records_sha256",
        )
        _require_sha256(self.cohort_sha256, field="source bundle cohort_sha256")
        if (
            not isinstance(self.dataset_sample_count, int)
            or isinstance(self.dataset_sample_count, bool)
            or self.dataset_sample_count <= 0
        ):
            raise ValueError("dataset_sample_count must be a positive integer")
        if not self.cases:
            raise ValueError("source bundle must contain at least one case")
        if any(case.cohort != self.cohort for case in self.cases):
            raise ValueError("source bundle cases have a different cohort")
        keys = tuple(case.key for case in self.cases)
        if len(keys) != len(set(keys)):
            raise ValueError("source bundle contains duplicate targeting/sample identities")
        if not self.source_files:
            raise ValueError("source bundle must retain source files")
        paths = tuple(snapshot.path for snapshot in self.source_files)
        if len(paths) != len(set(paths)):
            raise ValueError("source bundle contains duplicate source files")

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort": self.cohort,
            "model": self.model,
            "benchmark": self.benchmark,
            "model_revision": self.model_revision,
            "dataset_records_sha256": self.dataset_records_sha256,
            "dataset_sample_count": self.dataset_sample_count,
            "cohort_sha256": self.cohort_sha256,
            "case_count": len(self.cases),
            "source_files": [snapshot.to_dict() for snapshot in self.source_files],
        }


def validate_source_snapshot(source: SourceBundle) -> None:
    """Rehash every upstream file and reject any post-load byte mutation."""

    for snapshot in source.source_files:
        try:
            actual = _sha256_file(snapshot.path)
        except OSError as exc:
            raise ValueError(
                f"source snapshot changed: {snapshot.role} is unavailable: {snapshot.path}"
            ) from exc
        if actual != snapshot.sha256:
            raise ValueError(
                f"source snapshot changed: {snapshot.role} SHA-256 mismatch: {snapshot.path}"
            )


def _snapshot(role: str, path: Path, sha256: str) -> SourceFileSnapshot:
    return SourceFileSnapshot(role=role, path=path, sha256=sha256)


def _primary_files(reference: Any) -> tuple[SourceFileSnapshot, ...]:
    run_dir = reference.path
    files: list[SourceFileSnapshot] = [
        _snapshot(
            "fixed-window-run-manifest",
            run_dir / "run.json",
            reference.run_sha256,
        )
    ]
    for name, role in (
        ("fixed_window_records.jsonl", "fixed-window-records"),
        ("pair_status_records.jsonl", "fixed-window-pair-statuses"),
        ("setting_summary.json", "fixed-window-summary"),
    ):
        metadata = reference.output_metadata[name]
        sha256 = metadata.get("sha256")
        if not isinstance(sha256, str):
            raise ValueError(f"fixed-window output {name} has no SHA-256")
        files.append(_snapshot(role, run_dir / name, sha256))
    for targeting, source in sorted(
        reference.sources.by_targeting.items(),
        key=lambda item: _TARGETING_ORDER[item[0]],
    ):
        files.extend(
            (
                _snapshot(f"prepared-pairs:{targeting}", source.path, source.pairs_sha256),
                _snapshot(
                    f"prepared-pairs-run-manifest:{targeting}",
                    source.path.parent / "run.json",
                    source.run_sha256,
                ),
            )
        )
    return tuple(files)


def load_primary_source(
    fixed_window_run: Path,
    *,
    model: str,
    benchmark: str,
) -> SourceBundle:
    """Load the exact unlimited ``[0,6)`` clean-to-edited denominator."""

    if benchmark != "gsm8k":
        raise ValueError("the primary clean-prefix cohort requires gsm8k")
    run_dir = Path(fixed_window_run)
    request = coordinate_runner.CoordinateControlConfig(
        model=model,
        benchmark="gsm8k",
        fixed_window_run=run_dir,
        layers=(_PRIMARY_WINDOW,),
        controls=("correct", "offset-2"),
        output_dir=run_dir.parent / f".{run_dir.name}-clean-prefix-source-validation",
        gpu_id="0",
        limit=None,
    )
    reference = coordinate_runner._load_reference(request)
    arguments = coordinate_runner._mapping(
        reference.manifest.get("arguments"),
        field="fixed-window arguments",
    )
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("primary fixed-window source must be unlimited")

    dataset_sha256 = _require_sha256(
        reference.sources.dataset_records_sha256,
        field="primary source dataset_records_sha256",
    )
    cases = tuple(
        SourceCase(
            cohort="primary",
            targeting=pair.source.targeting,
            sample_id=pair.source.sample_id,
            record=pair.source.record,
            source_record_sha256=pair.source.fingerprint,
            prepared_clean_correct=pair.source.prepared_clean_correct,
            prepared_edited_wrong=pair.source.prepared_edited_wrong,
            aligned_word_count=pair.source.aligned_count,
            reference_event=pair.correct.event,
            reference_recipient_generation_identical=(pair.correct.recipient_generation_identical),
        )
        for pair in reference.pairs
    )
    source = SourceBundle(
        cohort="primary",
        model=model,
        benchmark=benchmark,
        model_revision=reference.sources.model_revision,
        dataset_records_sha256=dataset_sha256,
        dataset_sample_count=reference.sources.dataset_sample_count,
        cases=cases,
        source_files=_primary_files(reference),
        cohort_sha256=reference.denominator_sha256,
    )
    # Close the load-time TOCTOU window: every byte used by the delegated
    # validators must still match before the bundle can escape this function.
    validate_source_snapshot(source)
    return source


@dataclass(frozen=True, slots=True)
class _PreparedSourceRequest:
    """Duck-typed request accepted by the strict prepared-pair loader."""

    model: str
    benchmark: str


def load_extension_source(
    pairs: Sequence[Path],
    *,
    model: str,
    benchmark: str,
) -> SourceBundle:
    """Load completed unlimited Attribution-4 and Random-4 pair sources."""

    paths = tuple(Path(path) for path in pairs)
    if len(paths) != 2 or len({path.resolve() for path in paths}) != 2:
        raise ValueError("extension source requires two different prepared-pair files")
    request = _PreparedSourceRequest(model=model, benchmark=benchmark)
    detected = tuple((fixed_runner._source_targeting(path), path) for path in paths)
    targetings = tuple(targeting for targeting, _ in detected)
    if set(targetings) != set(_TARGETING_ORDER) or len(set(targetings)) != 2:
        raise ValueError("extension source requires one attribution-4 and one random-4 arm")
    # Bind the delegated parser to the exact path bytes observed before it
    # starts.  Without this first snapshot, an atomic path replacement between
    # parsing and the producer loader's later hash pass could pair old parsed
    # records with a new file fingerprint.
    before_parse = {
        targeting: {
            "pairs": _sha256_file(path),
            "run": _sha256_file(path.parent / "run.json"),
        }
        for targeting, path in detected
    }
    loaded = {
        targeting: fixed_runner._load_source(
            path,
            targeting=targeting,
            config=request,  # type: ignore[arg-type]
        )
        for targeting, path in detected
    }
    for targeting, source in loaded.items():
        if (
            source.pairs_sha256 != before_parse[targeting]["pairs"]
            or source.run_sha256 != before_parse[targeting]["run"]
        ):
            raise ValueError(
                f"extension source {targeting} changed before parsing; "
                "the source hash must bind the exact parsed bytes"
            )
    revisions = {source.model_revision for source in loaded.values()}
    dataset_hashes = {source.dataset_records_sha256 for source in loaded.values()}
    dataset_counts = {source.dataset_sample_count for source in loaded.values()}
    if len(revisions) != 1:
        raise ValueError("extension source model revisions differ between targeting arms")
    if len(dataset_hashes) != 1 or len(dataset_counts) != 1:
        raise ValueError("extension source dataset fingerprints differ between targeting arms")
    dataset_sha256 = _require_sha256(
        next(iter(dataset_hashes)),
        field="extension source dataset_records_sha256",
    )

    cases = tuple(
        SourceCase(
            cohort="extension",
            targeting=pair.targeting,
            sample_id=pair.sample_id,
            record=pair.record,
            source_record_sha256=pair.fingerprint,
            prepared_clean_correct=pair.prepared_clean_correct,
            prepared_edited_wrong=pair.prepared_edited_wrong,
            aligned_word_count=pair.aligned_count,
        )
        for targeting in _TARGETING_ORDER
        for pair in loaded[targeting].pairs
    )
    source_files = tuple(
        snapshot
        for targeting in _TARGETING_ORDER
        for snapshot in (
            _snapshot(
                f"prepared-pairs:{targeting}",
                loaded[targeting].path,
                loaded[targeting].pairs_sha256,
            ),
            _snapshot(
                f"prepared-pairs-run-manifest:{targeting}",
                loaded[targeting].path.parent / "run.json",
                loaded[targeting].run_sha256,
            ),
        )
    )
    source = SourceBundle(
        cohort="extension",
        model=model,
        benchmark=benchmark,
        model_revision=next(iter(revisions)),
        dataset_records_sha256=dataset_sha256,
        dataset_sample_count=next(iter(dataset_counts)),
        cases=cases,
        source_files=source_files,
        cohort_sha256=_canonical_sha256([case.identity_dict() for case in cases]),
    )
    validate_source_snapshot(source)
    return source


def load_source_bundle(
    *,
    cohort: SourceCohort,
    model: str,
    benchmark: str,
    fixed_window_run: Path | None = None,
    pairs: Sequence[Path] = (),
) -> SourceBundle:
    """Dispatch the two explicit source modes without accepting mixed inputs."""

    if cohort == "primary":
        if fixed_window_run is None or tuple(pairs):
            raise ValueError("primary source requires fixed_window_run and forbids pairs")
        return load_primary_source(
            fixed_window_run,
            model=model,
            benchmark=benchmark,
        )
    if cohort == "extension":
        if fixed_window_run is not None:
            raise ValueError("extension source forbids fixed_window_run")
        return load_extension_source(pairs, model=model, benchmark=benchmark)
    raise ValueError("source cohort must be primary or extension")


__all__ = [
    "SourceBundle",
    "SourceCase",
    "SourceCohort",
    "SourceFileSnapshot",
    "load_extension_source",
    "load_primary_source",
    "load_source_bundle",
    "validate_source_snapshot",
]

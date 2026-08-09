"""Strict readers for versioned sample-ID cohort protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from typo_cot.experiments.catalog import PAPER_SHA256

_SHA256 = re.compile(r"[0-9a-f]{64}")
MODEL_SCALE_COHORT_ID = "model-scale-mmlu-first500"
MODEL_SCALE_COHORT_SELECTION = "first-500-seed-42-loader-order-with-100-per-subject/v1"
MODEL_SCALE_COHORT_SAMPLE_IDS_SHA256 = (
    "7663efab7085892e60ba7a68c6b3c857101468aef2a2cff5acda998d7b6c637d"
)
MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET = {
    "google/gemma-3-1b-it": 50,
    "google/gemma-3-4b-it": 50,
    "google/gemma-3-12b-it": 100,
    "google/gemma-3-27b-it": 100,
    "meta-llama/Llama-3.2-1B-Instruct": 50,
    "meta-llama/Llama-3.2-3B-Instruct": 50,
    "meta-llama/Llama-3.1-70B-Instruct": 100,
    "mistralai/Mistral-7B-Instruct-v0.3": 50,
    "Qwen/Qwen2.5-72B-Instruct": 100,
}
MODEL_SCALE_COHORT_SELECTED_SAMPLE_COUNTS = {
    model: 500 if count == 100 else 250
    for model, count in MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET.items()
}
_COHORT_KEYS = {
    "schema_version",
    "paper_sha256",
    "cohort_id",
    "benchmark",
    "selection",
    "provenance",
    "sample_count",
    "sample_ids_sha256",
    "sample_ids",
    "model_samples_per_subset",
    "model_selected_sample_counts",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_ids_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(sample_ids),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample-ID cohort {field} must be a non-empty string")
    return value


def _model_mapping(
    value: object,
    *,
    field: str,
    allow_null: bool,
) -> Mapping[str, int | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"sample-ID cohort {field} must be a JSON object")
    result: dict[str, int | None] = {}
    for raw_model, raw_count in value.items():
        model = _nonempty_string(raw_model, field=f"{field} model")
        if raw_count is None and allow_null:
            result[model] = None
            continue
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count <= 0:
            label = "a positive integer or null" if allow_null else "a positive integer"
            raise ValueError(f"sample-ID cohort {field}.{model} must be {label}")
        result[model] = raw_count
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class SampleIdCohort:
    """Validated identity and model-specific coverage of one cohort artifact."""

    path: Path
    schema_version: str
    paper_sha256: str
    cohort_id: str
    benchmark: str
    selection: str
    provenance: str
    sample_ids: tuple[str, ...]
    sample_ids_sha256: str
    artifact_sha256: str
    model_samples_per_subset: Mapping[str, int | None]
    model_selected_sample_counts: Mapping[str, int | None]

    def expected_count_for(self, model: str) -> int | None:
        """Return a frozen selected-count expectation when the artifact defines one."""
        return self.model_selected_sample_counts.get(model)

    def provenance_for(self, model: str) -> dict[str, object]:
        """Return the compact identity embedded in producer manifests."""
        return {
            "schema_version": self.schema_version,
            "cohort_id": self.cohort_id,
            "benchmark": self.benchmark,
            "selection": self.selection,
            "provenance": self.provenance,
            "sample_count": len(self.sample_ids),
            "sample_ids_sha256": self.sample_ids_sha256,
            "artifact_sha256": self.artifact_sha256,
            "model_samples_per_subset": self.model_samples_per_subset.get(model),
            "selected_sample_count": self.expected_count_for(model),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete versioned protocol payload."""
        return {
            "schema_version": self.schema_version,
            "paper_sha256": self.paper_sha256,
            "cohort_id": self.cohort_id,
            "benchmark": self.benchmark,
            "selection": self.selection,
            "provenance": self.provenance,
            "sample_count": len(self.sample_ids),
            "sample_ids_sha256": self.sample_ids_sha256,
            "sample_ids": list(self.sample_ids),
            "model_samples_per_subset": dict(self.model_samples_per_subset),
            "model_selected_sample_counts": dict(self.model_selected_sample_counts),
        }


def load_sample_id_cohort(path: Path) -> SampleIdCohort:
    """Load a cohort fail-closed, including duplicate-key and digest checks."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"sample-ID cohort is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid sample-ID cohort JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("sample-ID cohort must be a JSON object")
    if set(payload) != _COHORT_KEYS:
        missing = sorted(_COHORT_KEYS - set(payload))
        unexpected = sorted(set(payload) - _COHORT_KEYS)
        raise ValueError(
            "sample-ID cohort fields do not match sample-id-cohort/v1: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if payload.get("schema_version") != "sample-id-cohort/v1":
        raise ValueError("sample-ID cohort has an unknown schema_version")
    if payload.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("sample-ID cohort paper SHA-256 does not match the final paper")
    cohort_id = _nonempty_string(payload.get("cohort_id"), field="cohort_id")
    benchmark = _nonempty_string(payload.get("benchmark"), field="benchmark")
    selection = _nonempty_string(payload.get("selection"), field="selection")
    provenance = _nonempty_string(payload.get("provenance"), field="provenance")
    raw_ids = payload.get("sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("sample-ID cohort sample_ids must be a non-empty list")
    sample_ids = tuple(
        _nonempty_string(value, field=f"sample_ids[{index}]") for index, value in enumerate(raw_ids)
    )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample-ID cohort sample_ids must be unique")
    sample_count = payload.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count != len(sample_ids)
    ):
        raise ValueError("sample-ID cohort sample_count must match sample_ids")
    sample_ids_sha256 = payload.get("sample_ids_sha256")
    if (
        not isinstance(sample_ids_sha256, str)
        or _SHA256.fullmatch(sample_ids_sha256) is None
        or sample_ids_sha256 != _canonical_ids_sha256(sample_ids)
    ):
        raise ValueError("sample-ID cohort sample_ids SHA-256 does not match its ordered IDs")
    samples_per_subset = _model_mapping(
        payload.get("model_samples_per_subset"),
        field="model_samples_per_subset",
        allow_null=True,
    )
    selected_counts = _model_mapping(
        payload.get("model_selected_sample_counts"),
        field="model_selected_sample_counts",
        allow_null=False,
    )
    if set(samples_per_subset) != set(selected_counts):
        raise ValueError(
            "sample-ID cohort model coverage keys must match between cap and count maps"
        )
    if any(count is not None and count > len(sample_ids) for count in selected_counts.values()):
        raise ValueError("sample-ID cohort selected counts cannot exceed sample_count")
    if cohort_id == MODEL_SCALE_COHORT_ID:
        if sample_ids_sha256 != MODEL_SCALE_COHORT_SAMPLE_IDS_SHA256:
            raise ValueError(
                "frozen cohort model-scale-mmlu-first500 ordered ID SHA-256 does not match"
            )
        frozen_scalars = {
            "benchmark": (benchmark, "mmlu"),
            "selection": (selection, MODEL_SCALE_COHORT_SELECTION),
            "sample_count": (len(sample_ids), 500),
        }
        for field, (actual, expected) in frozen_scalars.items():
            if actual != expected:
                raise ValueError(
                    f"frozen cohort model-scale-mmlu-first500 {field} must be {expected!r}"
                )
        if dict(samples_per_subset) != MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET:
            raise ValueError("frozen cohort model samples-per-subset map does not match")
        if dict(selected_counts) != MODEL_SCALE_COHORT_SELECTED_SAMPLE_COUNTS:
            raise ValueError("frozen cohort model selected-count map does not match")
    return SampleIdCohort(
        path=resolved,
        schema_version="sample-id-cohort/v1",
        paper_sha256=PAPER_SHA256,
        cohort_id=cohort_id,
        benchmark=benchmark,
        selection=selection,
        provenance=provenance,
        sample_ids=sample_ids,
        sample_ids_sha256=sample_ids_sha256,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        model_samples_per_subset=samples_per_subset,
        model_selected_sample_counts=selected_counts,
    )


def sample_id_of(sample: Any) -> str:
    """Read one dataset item's stable ID from a mapping or object."""
    value = (
        sample.get("sample_id")
        if isinstance(sample, Mapping)
        else getattr(sample, "sample_id", None)
    )
    if not isinstance(value, str) or not value:
        raise ValueError("every sample must expose a non-empty sample_id")
    return value


def select_cohort_samples(
    samples: Sequence[Any],
    cohort: SampleIdCohort,
    *,
    model: str,
) -> list[Any]:
    """Intersect dataset samples with a cohort and enforce frozen model coverage."""
    selected_ids = frozenset(cohort.sample_ids)
    selected = [sample for sample in samples if sample_id_of(sample) in selected_ids]
    observed = [sample_id_of(sample) for sample in selected]
    if len(observed) != len(set(observed)):
        raise ValueError("dataset returned duplicate sample IDs in the selected cohort")
    expected = cohort.expected_count_for(model)
    if expected is not None and len(selected) != expected:
        raise ValueError(
            f"sample-ID cohort expected {expected} selected sample(s) for {model}, "
            f"found {len(selected)}"
        )
    return selected


__all__ = [
    "MODEL_SCALE_COHORT_ID",
    "MODEL_SCALE_COHORT_SAMPLES_PER_SUBSET",
    "MODEL_SCALE_COHORT_SELECTED_SAMPLE_COUNTS",
    "MODEL_SCALE_COHORT_SELECTION",
    "MODEL_SCALE_COHORT_SAMPLE_IDS_SHA256",
    "SampleIdCohort",
    "load_sample_id_cohort",
    "sample_id_of",
    "select_cohort_samples",
]

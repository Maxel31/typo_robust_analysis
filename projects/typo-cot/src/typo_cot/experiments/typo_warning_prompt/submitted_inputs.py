"""Strict, output-free input fixture for the submitted Appendix E audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ROOT_KEYS = {
    "schema_version",
    "paper_sha256",
    "artifact_role",
    "canonicalization",
    "provenance",
    "dataset_revisions",
    "model_revisions",
    "aggregate_hash_semantics",
    "patch_semantics",
    "benchmarks",
    "settings",
}
_PROVENANCE_KEYS = {
    "contains_generated_outputs",
    "selection",
    "legacy_input_source_fingerprints",
    "prompt_template_file_sha256",
    "warning_text_sha256",
}
_LEGACY_FINGERPRINT_KEYS = {
    "submitted_attribution_full_input_sha256",
    "submitted_selected_input_sha256",
    "submitted_selector_manifest_sha256",
}
_BENCHMARK_KEYS = {
    "sample_count",
    "sample_ids",
    "ordered_sample_ids_sha256",
    "dataset_records_sha256",
    "clean_editable_records_sha256",
    "clean_prompt_records_sha256",
}
_SETTING_KEYS = {
    "model",
    "benchmark",
    "sample_count",
    "historical_target_attempts",
    "net_edit_count_distribution",
    "records",
    "records_sha256",
    "edited_editable_records_sha256",
    "without_warning_prompt_records_sha256",
    "with_warning_prompt_records_sha256",
}
_CANONICALIZATION = "json-utf8-sort-keys-compact-no-nan/v1"
_AGGREGATE_HASH_SEMANTICS = {
    "dataset_records": "sha256(canonical-json(ordered dataset identity records))/v1",
    "text_records": "sha256(canonical-json(ordered [{sample_id,text},...]))/v1",
}
_PATCH_SEMANTICS = {
    "kind": "sequential-replace-python-unicode-code-point/v1",
    "derivation": "python-difflib-sequence-matcher-autojunk-false/v1",
    "record_shape": "[start,expected,replacement]",
    "application": (
        "left-to-right; start indexes current editable immediately before patch; "
        "require exact expected slice"
    ),
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _string(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase SHA-256")
    return text


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if extra:
            details.append(f"unexpected keys={extra}")
        if missing:
            details.append(f"missing keys={missing}")
        raise ValueError(f"{field} has an invalid schema ({'; '.join(details)})")


def _subset_from_id(sample_id: str, *, benchmark: str) -> str:
    if benchmark == "gsm8k":
        if not re.fullmatch(r"gsm8k_[0-9]{5}", sample_id):
            raise ValueError(f"invalid GSM8K sample ID: {sample_id!r}")
        return "default"
    if benchmark == "mmlu":
        prefix = "mmlu_"
        if not sample_id.startswith(prefix):
            raise ValueError(f"invalid MMLU sample ID: {sample_id!r}")
        body = sample_id[len(prefix) :]
        try:
            subset, index = body.rsplit("_", 1)
        except ValueError as exc:
            raise ValueError(f"invalid MMLU sample ID: {sample_id!r}") from exc
        if not subset or re.fullmatch(r"[0-9]{4}", index) is None:
            raise ValueError(f"invalid MMLU sample ID: {sample_id!r}")
        return subset
    raise ValueError(f"unsupported submitted benchmark: {benchmark!r}")


@dataclass(frozen=True, slots=True)
class SubmittedInputPatch:
    """One fail-closed replacement against the current editable string."""

    start: int
    expected: str
    replacement: str

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError("submitted patch start must be a non-negative integer")
        if not isinstance(self.expected, str) or not isinstance(self.replacement, str):
            raise TypeError("submitted patch expected and replacement values must be strings")
        if self.expected == self.replacement:
            raise ValueError("submitted patch must change the input")

    def to_list(self) -> list[object]:
        return [self.start, self.expected, self.replacement]


def apply_submitted_patches(
    clean_text: str,
    patches: Sequence[SubmittedInputPatch],
) -> str:
    """Apply current-coordinate patches from left to right, checking every slice."""

    if not isinstance(clean_text, str) or not clean_text:
        raise ValueError("submitted clean text must be a non-empty string")
    edited = clean_text
    for index, patch in enumerate(patches):
        if not isinstance(patch, SubmittedInputPatch):
            raise TypeError(f"submitted patch {index} has the wrong type")
        end = patch.start + len(patch.expected)
        if patch.start > len(edited) or end > len(edited):
            raise ValueError("submitted patch lies outside the current input")
        if edited[patch.start : end] != patch.expected:
            raise ValueError("submitted patch expected text differs from the current input")
        edited = edited[: patch.start] + patch.replacement + edited[end:]
    return edited


@dataclass(frozen=True, slots=True)
class SubmittedInputRecord:
    sample_id: str
    subset: str
    patches: tuple[SubmittedInputPatch, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "subset": self.subset,
            "patches": [patch.to_list() for patch in self.patches],
        }


@dataclass(frozen=True, slots=True)
class SubmittedBenchmark:
    name: str
    sample_count: int
    sample_ids: tuple[str, ...]
    ordered_sample_ids_sha256: str
    dataset_records_sha256: str
    clean_editable_records_sha256: str
    clean_prompt_records_sha256: str


@dataclass(frozen=True, slots=True)
class SubmittedInputSetting:
    model: str
    benchmark: str
    sample_count: int
    historical_target_attempts: int
    net_edit_count_distribution: Mapping[str, int]
    records_sha256: str
    edited_editable_records_sha256: str
    without_warning_prompt_records_sha256: str
    with_warning_prompt_records_sha256: str
    records: tuple[SubmittedInputRecord, ...]


@dataclass(frozen=True, slots=True)
class SubmittedInputFixture:
    path: Path
    sha256: str
    canonical_sha256: str
    dataset_revisions: Mapping[str, str]
    model_revisions: Mapping[str, Mapping[str, str]]
    provenance: Mapping[str, object]
    benchmarks: Mapping[str, SubmittedBenchmark]
    settings: tuple[SubmittedInputSetting, ...]

    def setting(self, *, model: str, benchmark: str) -> SubmittedInputSetting:
        matches = [
            setting
            for setting in self.settings
            if setting.model == model and setting.benchmark == benchmark
        ]
        if len(matches) != 1:
            raise ValueError(
                "submitted input fixture must contain exactly one setting for "
                f"model={model!r}, benchmark={benchmark!r}"
            )
        return matches[0]


def _parse_benchmark(name: str, value: object) -> SubmittedBenchmark:
    field = f"benchmarks.{name}"
    payload = _mapping(value, field=field)
    _exact_keys(payload, _BENCHMARK_KEYS, field=field)
    count = _positive_int(payload.get("sample_count"), field=f"{field}.sample_count")
    raw_ids = _list(payload.get("sample_ids"), field=f"{field}.sample_ids")
    sample_ids = tuple(
        _string(value, field=f"{field}.sample_ids[{index}]") for index, value in enumerate(raw_ids)
    )
    if len(sample_ids) != count:
        raise ValueError(f"{field}.sample_count must match sample_ids")
    if list(sample_ids) != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{field}.sample_ids must be strictly sorted and unique")
    for sample_id in sample_ids:
        _subset_from_id(sample_id, benchmark=name)
    ordered_hash = _sha256(
        payload.get("ordered_sample_ids_sha256"),
        field=f"{field}.ordered_sample_ids_sha256",
    )
    if ordered_hash != _canonical_sha256(list(sample_ids)):
        raise ValueError(f"{field} ordered sample IDs SHA-256 differs")
    return SubmittedBenchmark(
        name=name,
        sample_count=count,
        sample_ids=sample_ids,
        ordered_sample_ids_sha256=ordered_hash,
        dataset_records_sha256=_sha256(
            payload.get("dataset_records_sha256"),
            field=f"{field}.dataset_records_sha256",
        ),
        clean_editable_records_sha256=_sha256(
            payload.get("clean_editable_records_sha256"),
            field=f"{field}.clean_editable_records_sha256",
        ),
        clean_prompt_records_sha256=_sha256(
            payload.get("clean_prompt_records_sha256"),
            field=f"{field}.clean_prompt_records_sha256",
        ),
    )


def _parse_setting(
    value: object,
    *,
    index: int,
    benchmarks: Mapping[str, SubmittedBenchmark],
) -> SubmittedInputSetting:
    field = f"settings[{index}]"
    payload = _mapping(value, field=field)
    _exact_keys(payload, _SETTING_KEYS, field=field)
    model = _string(payload.get("model"), field=f"{field}.model")
    benchmark = _string(payload.get("benchmark"), field=f"{field}.benchmark")
    if benchmark not in benchmarks:
        raise ValueError(f"{field}.benchmark is absent from benchmark metadata")
    benchmark_metadata = benchmarks[benchmark]
    count = _positive_int(payload.get("sample_count"), field=f"{field}.sample_count")
    if count != benchmark_metadata.sample_count:
        raise ValueError(f"{field}.sample_count differs from benchmark metadata")
    if payload.get("historical_target_attempts") != 4:
        raise ValueError(f"{field}.historical_target_attempts must be 4")

    raw_distribution = _mapping(
        payload.get("net_edit_count_distribution"),
        field=f"{field}.net_edit_count_distribution",
    )
    if set(raw_distribution) != {"2", "3", "4"}:
        raise ValueError(f"{field}.net_edit_count_distribution must contain 2, 3, and 4")
    distribution: dict[str, int] = {}
    for distance in ("2", "3", "4"):
        value = raw_distribution[distance]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field}.net_edit_count_distribution.{distance} is invalid")
        distribution[distance] = value
    if sum(distribution.values()) != count:
        raise ValueError(f"{field}.net_edit_count_distribution must sum to sample_count")

    raw_records = _list(payload.get("records"), field=f"{field}.records")
    if len(raw_records) != count:
        raise ValueError(f"{field}.sample_count must match records")
    records: list[SubmittedInputRecord] = []
    canonical_records: list[list[list[object]]] = []
    for record_index, (sample_id, raw_record) in enumerate(
        zip(benchmark_metadata.sample_ids, raw_records, strict=True)
    ):
        record_field = f"{field}.records[{record_index}]"
        raw_patches = _list(raw_record, field=record_field)
        if not raw_patches:
            raise ValueError(f"{record_field} must contain at least one patch")
        patches: list[SubmittedInputPatch] = []
        for patch_index, raw_patch in enumerate(raw_patches):
            patch_field = f"{record_field}[{patch_index}]"
            values = _list(raw_patch, field=patch_field)
            if len(values) != 3:
                raise ValueError(f"{patch_field} must have [start, expected, replacement]")
            start, expected, replacement = values
            if not isinstance(start, int) or isinstance(start, bool):
                raise ValueError(f"{patch_field} start must be an integer")
            if not isinstance(expected, str) or not isinstance(replacement, str):
                raise ValueError(f"{patch_field} text values must be strings")
            patches.append(SubmittedInputPatch(start, expected, replacement))
        record = SubmittedInputRecord(
            sample_id=sample_id,
            subset=_subset_from_id(sample_id, benchmark=benchmark),
            patches=tuple(patches),
        )
        records.append(record)
        canonical_records.append([patch.to_list() for patch in patches])
    records_hash = _sha256(payload.get("records_sha256"), field=f"{field}.records_sha256")
    if records_hash != _canonical_sha256(canonical_records):
        raise ValueError(f"{field} records SHA-256 differs")
    return SubmittedInputSetting(
        model=model,
        benchmark=benchmark,
        sample_count=count,
        historical_target_attempts=4,
        net_edit_count_distribution=distribution,
        records_sha256=records_hash,
        edited_editable_records_sha256=_sha256(
            payload.get("edited_editable_records_sha256"),
            field=f"{field}.edited_editable_records_sha256",
        ),
        without_warning_prompt_records_sha256=_sha256(
            payload.get("without_warning_prompt_records_sha256"),
            field=f"{field}.without_warning_prompt_records_sha256",
        ),
        with_warning_prompt_records_sha256=_sha256(
            payload.get("with_warning_prompt_records_sha256"),
            field=f"{field}.with_warning_prompt_records_sha256",
        ),
        records=tuple(records),
    )


def load_submitted_input_fixture(path: Path) -> SubmittedInputFixture:
    """Load and validate the compact input-only fixture without benchmark data."""

    path = Path(path).resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid submitted input fixture {path}: {exc}") from exc
    root = _mapping(payload, field="submitted input fixture")
    _exact_keys(root, _ROOT_KEYS, field="submitted input fixture")
    if root.get("schema_version") != "typo-warning-submitted-input-patches/v2":
        raise ValueError("submitted input fixture has an unknown schema")
    if root.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("submitted input fixture paper SHA-256 differs")
    if root.get("artifact_role") != "input-only-no-generated-outputs":
        raise ValueError("submitted fixture artifact role must prohibit generated outputs")
    if root.get("canonicalization") != _CANONICALIZATION:
        raise ValueError("submitted input fixture canonicalization differs")
    if root.get("aggregate_hash_semantics") != _AGGREGATE_HASH_SEMANTICS:
        raise ValueError("submitted input fixture aggregate hash semantics differ")
    if root.get("patch_semantics") != _PATCH_SEMANTICS:
        raise ValueError("submitted input fixture patch semantics differ")

    provenance = _mapping(root.get("provenance"), field="fixture provenance")
    _exact_keys(provenance, _PROVENANCE_KEYS, field="fixture provenance")
    if provenance.get("contains_generated_outputs") is not False:
        raise ValueError("submitted fixture must declare that it contains no generated outputs")
    if provenance.get("selection") != "submitted-attribution-random-intersection-seed-42/v1":
        raise ValueError("submitted fixture selection provenance differs")
    _sha256(
        provenance.get("prompt_template_file_sha256"),
        field="fixture provenance prompt_template_file_sha256",
    )
    _sha256(
        provenance.get("warning_text_sha256"),
        field="fixture provenance warning_text_sha256",
    )

    raw_revisions = _mapping(root.get("dataset_revisions"), field="dataset_revisions")
    if set(raw_revisions) != {"gsm8k", "mmlu"}:
        raise ValueError("dataset_revisions must contain gsm8k and mmlu")
    dataset_revisions: dict[str, str] = {}
    for benchmark, revision in raw_revisions.items():
        text = _string(revision, field=f"dataset_revisions.{benchmark}")
        if _REVISION.fullmatch(text) is None:
            raise ValueError(f"dataset_revisions.{benchmark} must be a 40-character revision")
        dataset_revisions[benchmark] = text

    raw_benchmarks = _mapping(root.get("benchmarks"), field="fixture benchmarks")
    if set(raw_benchmarks) != {"gsm8k", "mmlu"}:
        raise ValueError("fixture benchmarks must contain gsm8k and mmlu")
    benchmarks = {name: _parse_benchmark(name, raw_benchmarks[name]) for name in ("gsm8k", "mmlu")}

    raw_settings = _list(root.get("settings"), field="fixture settings")
    settings = tuple(
        _parse_setting(value, index=index, benchmarks=benchmarks)
        for index, value in enumerate(raw_settings)
    )
    keys = [(setting.model, setting.benchmark) for setting in settings]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("submitted fixture settings must be strictly sorted and unique")

    raw_model_revisions = _mapping(root.get("model_revisions"), field="fixture model_revisions")
    models = {setting.model for setting in settings}
    if set(raw_model_revisions) != models:
        raise ValueError("fixture model revisions must match setting models")
    model_revisions: dict[str, Mapping[str, str]] = {}
    for model, raw_revision in raw_model_revisions.items():
        revision_payload = _mapping(raw_revision, field=f"model_revisions.{model}")
        if set(revision_payload) != {"revision", "evidence"}:
            raise ValueError(f"model_revisions.{model} has an invalid schema")
        revision = _string(
            revision_payload.get("revision"), field=f"model_revisions.{model}.revision"
        )
        if _REVISION.fullmatch(revision) is None:
            raise ValueError(f"model_revisions.{model}.revision must be 40 hexadecimal chars")
        evidence = _string(
            revision_payload.get("evidence"), field=f"model_revisions.{model}.evidence"
        )
        model_revisions[model] = {"revision": revision, "evidence": evidence}

    raw_fingerprints = _mapping(
        provenance.get("legacy_input_source_fingerprints"),
        field="fixture legacy_input_source_fingerprints",
    )
    expected_fingerprint_keys = {f"{model}|{benchmark}" for model, benchmark in keys}
    if set(raw_fingerprints) != expected_fingerprint_keys:
        raise ValueError("legacy input fingerprints must match fixture settings")
    for key, raw_fingerprint in raw_fingerprints.items():
        fingerprint = _mapping(raw_fingerprint, field=f"legacy fingerprint {key}")
        _exact_keys(fingerprint, _LEGACY_FINGERPRINT_KEYS, field=f"legacy fingerprint {key}")
        for name in _LEGACY_FINGERPRINT_KEYS:
            _sha256(fingerprint[name], field=f"legacy fingerprint {key}.{name}")

    return SubmittedInputFixture(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=_canonical_sha256(root),
        dataset_revisions=dataset_revisions,
        model_revisions=model_revisions,
        provenance=dict(provenance),
        benchmarks=benchmarks,
        settings=settings,
    )


__all__ = [
    "SubmittedBenchmark",
    "SubmittedInputFixture",
    "SubmittedInputPatch",
    "SubmittedInputRecord",
    "SubmittedInputSetting",
    "apply_submitted_patches",
    "load_submitted_input_fixture",
]

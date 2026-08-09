"""Fail-closed reconstruction of the submitted Appendix E edited inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from typo_cot.experiments.typo_warning_prompt.protocol import PROMPT_MARKERS, WARNING_TEXT
from typo_cot.experiments.typo_warning_prompt.planning import inject_typo_warning
from typo_cot.experiments.typo_warning_prompt.submitted_inputs import (
    SubmittedInputRecord,
    apply_submitted_patches,
    load_submitted_input_fixture,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
DEFAULT_SUBMITTED_INPUTS = (
    Path(__file__).with_name("data") / "submitted_input_edits.json"
).resolve()
DEFAULT_SUBMITTED_INPUTS_SHA256 = "d838d8d2f6b0eda212ec70a53b5d66c9b22c351b526b1fcec71af21232b755ae"
DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256 = (
    "1833ad0b0324e55cef79c0f3cf273d73f0ebdd07c975ca77b276c2755dedae73"
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def strict_loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {context}: {exc}") from exc


def load_json(path: Path) -> dict[str, object]:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_records_sha256(sample_ids: Sequence[str], texts: Sequence[str]) -> str:
    if len(sample_ids) != len(texts):
        raise ValueError("submitted text aggregate lengths differ")
    return canonical_sha256(
        [
            {"sample_id": sample_id, "text": text}
            for sample_id, text in zip(sample_ids, texts, strict=True)
        ]
    )


def _banded_levenshtein(clean: str, edited: str, *, maximum: int = 4) -> int:
    if abs(len(clean) - len(edited)) > maximum:
        return maximum + 1
    infinity = maximum + 1
    previous = {index: index for index in range(min(len(edited), maximum) + 1)}
    for clean_index in range(1, len(clean) + 1):
        current: dict[int, int] = {}
        lower = max(0, clean_index - maximum)
        upper = min(len(edited), clean_index + maximum)
        for edited_index in range(lower, upper + 1):
            deletion = previous.get(edited_index, infinity) + 1
            insertion = current.get(edited_index - 1, infinity) + 1
            substitution = (
                previous.get(edited_index - 1, infinity)
                + (clean[clean_index - 1] != edited[edited_index - 1])
                if edited_index > 0
                else infinity
            )
            current[edited_index] = min(deletion, insertion, substitution)
        previous = current
    return previous.get(len(edited), infinity)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    role: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("source file role must not be empty")
        object.__setattr__(self, "path", Path(self.path).resolve())
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source file sha256 must be a full lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _DatasetSample:
    sample_id: str
    question: str
    choices: tuple[str, ...] | None
    correct_answer: str
    subset: str

    @property
    def identity(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "choices": list(self.choices) if self.choices is not None else None,
            "correct_answer": self.correct_answer,
            "subset": self.subset,
        }


@dataclass(frozen=True, slots=True)
class WarningSourceCase:
    sample_id: str
    submitted_record: dict[str, object]
    submitted_record_sha256: str
    selector_record_sha256: str

    @property
    def identity(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "submitted_record_sha256": self.submitted_record_sha256,
            "selector_record_sha256": self.selector_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class WarningSourceBundle:
    model: str
    benchmark: str
    model_revision: str
    model_revision_evidence: str
    dataset_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int
    shared_sample_count: int
    cases: tuple[WarningSourceCase, ...]
    source_files: tuple[SourceFileSnapshot, ...]
    source_sha256: str
    fixture_sha256: str
    fixture_canonical_sha256: str
    fixture_is_bundled_submitted_input: bool
    setting_records_sha256: str
    ordered_sample_ids_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "input_kind": "submitted-attribution-4-edit-manifest/v1",
            "model": self.model,
            "benchmark": self.benchmark,
            "model_revision": self.model_revision,
            "model_revision_evidence": self.model_revision_evidence,
            "dataset_revision": self.dataset_revision,
            "dataset_records_sha256": self.dataset_records_sha256,
            "dataset_sample_count": self.dataset_sample_count,
            "shared_sample_count": self.shared_sample_count,
            "setting_records_sha256": self.setting_records_sha256,
            "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256,
            "fixture_sha256": self.fixture_sha256,
            "fixture_canonical_sha256": self.fixture_canonical_sha256,
            "fixture_is_bundled_submitted_input": self.fixture_is_bundled_submitted_input,
            "contains_generated_outputs": False,
            "source_sha256": self.source_sha256,
            "source_files": [snapshot.to_dict() for snapshot in self.source_files],
        }


def _normalize_injected_sample(value: object) -> _DatasetSample:
    def get(field: str) -> object:
        return value.get(field) if isinstance(value, Mapping) else getattr(value, field, None)

    sample_id = _string(get("sample_id"), field="dataset sample_id")
    question = _string(get("question"), field=f"dataset sample {sample_id}.question")
    correct_answer = _string(
        get("correct_answer"), field=f"dataset sample {sample_id}.correct_answer"
    )
    raw_choices = get("choices")
    choices: tuple[str, ...] | None
    if raw_choices is None:
        choices = None
    elif (
        isinstance(raw_choices, (list, tuple))
        and raw_choices
        and all(isinstance(choice, str) for choice in raw_choices)
    ):
        choices = tuple(raw_choices)
    else:
        raise ValueError(f"dataset sample {sample_id}.choices must be non-empty strings or null")
    raw_subset = get("subset")
    subset = raw_subset if isinstance(raw_subset, str) and raw_subset else "default"
    return _DatasetSample(sample_id, question, choices, correct_answer, subset)


def _gsm8k_samples(
    sample_specs: Sequence[tuple[str, str]], *, revision: str
) -> list[_DatasetSample]:
    from datasets import load_dataset

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=revision,
    )
    samples: list[_DatasetSample] = []
    for sample_id, _subset in sample_specs:
        prefix = "gsm8k_"
        if not sample_id.startswith(prefix):
            raise ValueError(f"invalid GSM8K sample ID in submitted fixture: {sample_id}")
        try:
            index = int(sample_id[len(prefix) :])
            item = dataset[index]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"submitted GSM8K sample ID is outside the pinned dataset: {sample_id}"
            ) from exc
        samples.append(
            _DatasetSample(
                sample_id=f"gsm8k_{index:05d}",
                question=str(item["question"]),
                choices=None,
                correct_answer=str(item["answer"]).split("####")[-1].strip(),
                subset="default",
            )
        )
    return samples


def _mmlu_samples(
    sample_specs: Sequence[tuple[str, str]], *, revision: str
) -> list[_DatasetSample]:
    from datasets import load_dataset

    by_subset: dict[str, list[str]] = {}
    for sample_id, subset in sample_specs:
        by_subset.setdefault(subset, []).append(sample_id)
    samples: list[_DatasetSample] = []
    for subset, subset_records in sorted(by_subset.items()):
        dataset = load_dataset("cais/mmlu", subset, split="test", revision=revision)
        prefix = f"mmlu_{subset}_"
        for sample_id in subset_records:
            if not sample_id.startswith(prefix):
                raise ValueError(
                    f"submitted MMLU sample ID and subset disagree: {sample_id}, {subset}"
                )
            try:
                index = int(sample_id[len(prefix) :])
                item = dataset[index]
                answer_index = int(item["answer"])
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"submitted MMLU sample ID is outside the pinned dataset: {sample_id}"
                ) from exc
            samples.append(
                _DatasetSample(
                    sample_id=f"mmlu_{subset}_{index:04d}",
                    question=str(item["question"]),
                    choices=tuple(str(choice) for choice in item["choices"]),
                    correct_answer=chr(ord("A") + answer_index),
                    subset=subset,
                )
            )
    return samples


@lru_cache(maxsize=4)
def _load_pinned_samples_cached(
    sample_specs: tuple[tuple[str, str], ...],
    *,
    benchmark: str,
    revision: str,
) -> tuple[_DatasetSample, ...]:
    if benchmark == "gsm8k":
        samples = _gsm8k_samples(sample_specs, revision=revision)
    elif benchmark == "mmlu":
        samples = _mmlu_samples(sample_specs, revision=revision)
    else:
        raise ValueError(f"unsupported typo-warning benchmark: {benchmark!r}")
    return tuple(samples)


def _load_pinned_samples(
    records: Sequence[SubmittedInputRecord],
    *,
    benchmark: str,
    revision: str,
) -> list[_DatasetSample]:
    sample_specs = tuple((record.sample_id, record.subset) for record in records)
    return list(
        _load_pinned_samples_cached(
            sample_specs,
            benchmark=benchmark,
            revision=revision,
        )
    )


def _clean_editable(sample: _DatasetSample, *, benchmark: str) -> str:
    if benchmark == "gsm8k":
        if sample.choices is not None:
            raise ValueError(f"GSM8K sample {sample.sample_id} must not contain choices")
        return sample.question
    if benchmark == "mmlu":
        if sample.choices is None or len(sample.choices) != 4:
            raise ValueError(f"MMLU sample {sample.sample_id} must contain four choices")
        choices = " ".join(
            f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(sample.choices)
        )
        return f"{sample.question}\n{choices}"
    raise ValueError(f"unsupported typo-warning benchmark: {benchmark!r}")


def _full_prompt(template: Any, *, benchmark: str, text: str, subset: str) -> str:
    if benchmark == "mmlu":
        return template.generate(question=text, choices=None, subject=subset).get_full_prompt()
    return template.generate(question=text).get_full_prompt()


def validate_source_snapshot(source: WarningSourceBundle) -> None:
    for snapshot in source.source_files:
        try:
            actual = sha256_file(snapshot.path)
        except OSError as exc:
            raise ValueError(
                f"source snapshot changed: {snapshot.role} is unavailable: {snapshot.path}"
            ) from exc
        if actual != snapshot.sha256:
            raise ValueError(
                f"source snapshot changed: {snapshot.role} SHA-256 mismatch: {snapshot.path}"
            )


def load_warning_sources(
    input_fixture: Path | None = None,
    *,
    model: str,
    benchmark: str,
    samples: Sequence[object] | None = None,
) -> WarningSourceBundle:
    """Reconstruct one exact submitted setting from pinned public data and patches."""

    if benchmark not in PROMPT_MARKERS:
        raise ValueError("benchmark must be gsm8k or mmlu")
    fixture_path = DEFAULT_SUBMITTED_INPUTS if input_fixture is None else Path(input_fixture)
    fixture = load_submitted_input_fixture(fixture_path)
    setting = fixture.setting(model=model, benchmark=benchmark)
    try:
        dataset_revision = fixture.dataset_revisions[benchmark]
        revision_payload = fixture.model_revisions[model]
        model_revision = revision_payload["revision"]
        model_revision_evidence = revision_payload["evidence"]
    except KeyError as exc:
        raise ValueError(f"submitted fixture is missing revision metadata: {exc}") from exc
    benchmark_metadata = fixture.benchmarks[benchmark]
    loaded_samples = (
        [_normalize_injected_sample(sample) for sample in samples]
        if samples is not None
        else _load_pinned_samples(
            setting.records,
            benchmark=benchmark,
            revision=dataset_revision,
        )
    )
    sample_by_id = {sample.sample_id: sample for sample in loaded_samples}
    if len(sample_by_id) != len(loaded_samples):
        raise ValueError("pinned dataset loader returned duplicate sample IDs")
    expected_ids = [record.sample_id for record in setting.records]
    if set(sample_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(sample_by_id))
        extra = sorted(set(sample_by_id) - set(expected_ids))
        raise ValueError(
            "pinned dataset samples differ from the submitted setting: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    import typo_cot.models.prompts as prompt_templates

    prompt_file = Path(prompt_templates.__file__).resolve()
    if sha256_file(prompt_file) != fixture.provenance["prompt_template_file_sha256"]:
        raise ValueError("prompt template implementation SHA-256 differs from the input fixture")
    if _text_sha256(WARNING_TEXT) != fixture.provenance["warning_text_sha256"]:
        raise ValueError("warning text SHA-256 differs from the input fixture")

    template = prompt_templates.create_prompt_template(benchmark)
    cases: list[WarningSourceCase] = []
    dataset_identities: list[dict[str, object]] = []
    clean_editables: list[str] = []
    edited_editables: list[str] = []
    clean_prompts: list[str] = []
    edited_prompts: list[str] = []
    warning_prompts: list[str] = []
    edit_distances = {"2": 0, "3": 0, "4": 0}
    for record in setting.records:
        sample = sample_by_id[record.sample_id]
        if sample.subset != record.subset:
            raise ValueError(f"submitted sample subset differs for {record.sample_id}")
        dataset_identity = sample.identity
        clean_editable = _clean_editable(sample, benchmark=benchmark)
        edited_editable = apply_submitted_patches(clean_editable, record.patches)
        clean_prompt = _full_prompt(
            template,
            benchmark=benchmark,
            text=clean_editable,
            subset=sample.subset,
        )
        edited_prompt = _full_prompt(
            template,
            benchmark=benchmark,
            text=edited_editable,
            subset=sample.subset,
        )
        warning_prompt = inject_typo_warning(
            edited_prompt,
            benchmark=benchmark,
            enabled=True,
        )
        distance = _banded_levenshtein(clean_editable, edited_editable)
        if str(distance) not in edit_distances:
            raise ValueError(
                f"reconstructed net edit distance is outside 2-4 for {record.sample_id}"
            )
        edit_distances[str(distance)] += 1
        clean_prompt_sha256 = _text_sha256(clean_prompt)
        clean_editable_sha256 = _text_sha256(clean_editable)
        edited_prompt_sha256 = _text_sha256(edited_prompt)
        edited_editable_sha256 = _text_sha256(edited_editable)
        submitted_record = {
            "schema_version": "typo-warning-submitted-input/v1",
            "sample_id": sample.sample_id,
            "model": model,
            "benchmark": benchmark,
            "subset": sample.subset,
            "gold_answer": sample.correct_answer,
            "clean": {
                "prompt_sha256": clean_prompt_sha256,
                "editable_text_sha256": clean_editable_sha256,
            },
            "edited": {
                "prompt": edited_prompt,
                "prompt_sha256": edited_prompt_sha256,
                "editable_text_sha256": edited_editable_sha256,
            },
            "fixture_record_sha256": canonical_sha256(record.to_dict()),
        }
        submitted_record_sha256 = canonical_sha256(
            {key: value for key, value in submitted_record.items() if key != "edited"}
            | {
                "edited": {
                    "prompt_sha256": edited_prompt_sha256,
                    "editable_text_sha256": edited_editable_sha256,
                }
            }
        )
        selector_record_sha256 = canonical_sha256(
            {
                "setting_records_sha256": setting.records_sha256,
                "ordered_sample_ids_sha256": benchmark_metadata.ordered_sample_ids_sha256,
                "sample_id": sample.sample_id,
            }
        )
        cases.append(
            WarningSourceCase(
                sample_id=sample.sample_id,
                submitted_record=submitted_record,
                submitted_record_sha256=submitted_record_sha256,
                selector_record_sha256=selector_record_sha256,
            )
        )
        dataset_identities.append(dataset_identity)
        clean_editables.append(clean_editable)
        edited_editables.append(edited_editable)
        clean_prompts.append(clean_prompt)
        edited_prompts.append(edited_prompt)
        warning_prompts.append(warning_prompt)

    sample_ids = [record.sample_id for record in setting.records]
    aggregate_checks = {
        "dataset records": (
            canonical_sha256(dataset_identities),
            benchmark_metadata.dataset_records_sha256,
        ),
        "clean editable records": (
            _text_records_sha256(sample_ids, clean_editables),
            benchmark_metadata.clean_editable_records_sha256,
        ),
        "clean prompt records": (
            _text_records_sha256(sample_ids, clean_prompts),
            benchmark_metadata.clean_prompt_records_sha256,
        ),
        "edited editable records": (
            _text_records_sha256(sample_ids, edited_editables),
            setting.edited_editable_records_sha256,
        ),
        "without-warning prompt records": (
            _text_records_sha256(sample_ids, edited_prompts),
            setting.without_warning_prompt_records_sha256,
        ),
        "with-warning prompt records": (
            _text_records_sha256(sample_ids, warning_prompts),
            setting.with_warning_prompt_records_sha256,
        ),
    }
    for label, (observed, expected) in aggregate_checks.items():
        if observed != expected:
            raise ValueError(f"submitted {label} SHA-256 differs")
    if edit_distances != dict(setting.net_edit_count_distribution):
        raise ValueError("submitted net edit-count distribution differs")

    fixture_digest = sha256_file(fixture.path)
    if fixture_digest != fixture.sha256:
        raise ValueError("submitted fixture changed while its setting was reconstructed")
    if fixture.path == DEFAULT_SUBMITTED_INPUTS and (
        fixture_digest != DEFAULT_SUBMITTED_INPUTS_SHA256
        or fixture.canonical_sha256 != DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256
    ):
        raise ValueError("bundled submitted fixture identity differs from the public contract")
    source_files = (
        SourceFileSnapshot(
            role="submitted-input-edit-manifest",
            path=fixture.path,
            sha256=fixture_digest,
        ),
    )
    source = WarningSourceBundle(
        model=model,
        benchmark=benchmark,
        model_revision=model_revision,
        model_revision_evidence=model_revision_evidence,
        dataset_revision=dataset_revision,
        dataset_records_sha256=benchmark_metadata.dataset_records_sha256,
        dataset_sample_count=len(dataset_identities),
        shared_sample_count=len(cases),
        cases=tuple(cases),
        source_files=source_files,
        source_sha256=canonical_sha256([case.identity for case in cases]),
        fixture_sha256=fixture_digest,
        fixture_canonical_sha256=fixture.canonical_sha256,
        fixture_is_bundled_submitted_input=(
            fixture_digest == DEFAULT_SUBMITTED_INPUTS_SHA256
            and fixture.canonical_sha256 == DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256
        ),
        setting_records_sha256=setting.records_sha256,
        ordered_sample_ids_sha256=benchmark_metadata.ordered_sample_ids_sha256,
    )
    validate_source_snapshot(source)
    return source


def select_warning_cases(source: WarningSourceBundle) -> tuple[WarningSourceCase, ...]:
    """Return the already-selected and sorted submitted cohort."""

    return source.cases


__all__ = [
    "DEFAULT_SUBMITTED_INPUTS",
    "DEFAULT_SUBMITTED_INPUTS_CANONICAL_SHA256",
    "DEFAULT_SUBMITTED_INPUTS_SHA256",
    "SourceFileSnapshot",
    "WarningSourceBundle",
    "WarningSourceCase",
    "canonical_sha256",
    "load_json",
    "load_warning_sources",
    "select_warning_cases",
    "sha256_file",
    "strict_loads",
    "validate_source_snapshot",
]

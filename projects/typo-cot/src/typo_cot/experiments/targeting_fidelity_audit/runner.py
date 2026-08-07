"""Validate prepared pairs and aggregate the final paper's input-quality audit."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PUBLIC_BENCHMARKS,
    TARGETING_CONDITIONS,
)
from typo_cot.experiments.prepare_edited_pairs.protocol import seeded_character_edit

_OPERATIONS = ("substitution", "duplication", "deletion")
_WORD = re.compile(r"\S+")
_MULTIPLE_CHOICE_BENCHMARKS = frozenset({"mmlu", "mmlu-pro", "arc", "csqa"})
_FULL_GRID_MODELS = frozenset(
    {
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "llama-3.2-1b-instruct",
        "llama-3.2-3b-instruct",
        "mistral-7b-instruct-v0.3",
        "qwen2.5-7b-instruct",
    }
)
_SCALE_EXTENSION_MODELS = frozenset({"gemma-3-12b-it", "gemma-3-27b-it"})
_FULL_GRID_BENCHMARKS = frozenset(PUBLIC_BENCHMARKS)
_SCALE_EXTENSION_BENCHMARKS = frozenset({"gsm8k", "math-500", "mmlu"})
_BENCHMARK_DATASET_LOADERS = {
    "gsm8k": "gsm8k",
    "math-500": "math",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
_PAPER_BENCHMARK_ITEM_COUNTS = {
    "arc": 1172,
    "csqa": 1221,
    "gsm8k": 1319,
    "math-500": 500,
    "mmlu": 2850,
    "mmlu-pro": 1400,
}
_EXPECTED_PAPER_SETTINGS = frozenset(
    {(model, benchmark) for model in _FULL_GRID_MODELS for benchmark in _FULL_GRID_BENCHMARKS}
    | {
        (model, benchmark)
        for model in _SCALE_EXTENSION_MODELS
        for benchmark in _SCALE_EXTENSION_BENCHMARKS
    }
)
_EXPECTED_PAPER_CELLS = frozenset(
    (model, benchmark, targeting)
    for model, benchmark in _EXPECTED_PAPER_SETTINGS
    for targeting in TARGETING_CONDITIONS
)
_DOUBLE_MMLU_MODELS = frozenset({"qwen2.5-7b-instruct"}) | _SCALE_EXTENSION_MODELS
_DATASET_COHORT_RULE = "paper-model-benchmark-cohort/v1"
_RANDOM_SEED_ALGORITHM = "sha256-first-64-bits/v1"
_TARGET_POSITION = "maximum-logit-after-first-cot-token"
_ALIGNMENT = "actual-edited-word-final-token"
_HISTORICAL_COMPATIBILITY_NOTES = (
    "stable-sha256-seeds-replace-process-random-python-hash",
    "mistral-attnlrp-rules-target-mistral-classes",
    "actual-word-final-alignment-replaces-token-substring-coordinates",
    "parenthesized-choice-markers-use-recorded-choice-boundary",
    "arc-numeric-answer-keys-normalized-to-prompt-letters",
    "model-specific-mmlu-cohort-matches-final-paper-denominators",
)
_ENVIRONMENT_VERSION_FIELDS = (
    "python",
    "torch",
    "transformers",
    "accelerate",
    "lxt",
    "datasets",
)
_ARCHIVAL_SOURCE_PROVENANCE_SHA256 = (
    "e74361590a00021fdc3871605fc9ee22772d8d0b493bfc564ab1daa57f8246c1"
)
_ARCHIVAL_TARGETING_REFERENCE: dict[str, object] = {
    "artifact_id": "final-paper-targeting-fidelity-reanalysis/v1",
    "availability": "author-local; not distributed with the public repository",
    "scope": "exact numerators, denominators, rank 2-4 breakdowns, and cohort counts",
    "artifacts": {
        "source_provenance.csv": (_ARCHIVAL_SOURCE_PROVENANCE_SHA256),
        "task2_aggregates.json": (
            "c9cffac8188efe6dceca4ee1c6493a006af0ce2a2da40dd7d9481ede79ce02d6"
        ),
        "task2_misalign_by_rank.csv": (
            "ccf286f3c1e06dc29137ae235276909d9ae41ffe02bb8a3fffeadc391dc0d380"
        ),
        "task2_method_numbers.py": (
            "cf4c780a6992a74c085c8e944d705b950be8ff4f01a83863c13a8225f04668f2"
        ),
    },
}
_EXPECTED_PAPER_ITEM_COUNTS = {
    (model, benchmark): (
        5700
        if benchmark == "mmlu" and model in _DOUBLE_MMLU_MODELS
        else _PAPER_BENCHMARK_ITEM_COUNTS[benchmark]
    )
    for model, benchmark in _EXPECTED_PAPER_SETTINGS
}


def _expected_samples_per_subset(model: str, benchmark: str) -> int | None:
    """Return the final-paper subset cap recorded by pair preparation."""
    model_basename = model.rsplit("/", 1)[-1].lower()
    if benchmark == "mmlu":
        return 100 if model_basename in _DOUBLE_MMLU_MODELS else 50
    if benchmark == "mmlu-pro":
        return 100
    return None


_CSV_FIELDS = (
    "row_type",
    "model",
    "benchmark",
    "targeting",
    "seed",
    "items",
    "zero_attempt_items",
    "zero_aligned_word_items",
    "attempted_but_zero_aligned_word_items",
    "four_distinct_word_items",
    "four_distinct_word_rate",
    "target_attempts",
    "faithful_target_attempts",
    "misplaced_target_attempts",
    "targeting_fidelity_rate",
    "targeting_miss_rate",
    "all_attempts_faithful_items",
    "selection_rank_1_attempts",
    "selection_rank_1_faithful_attempts",
    "selection_rank_1_fidelity_rate",
    "selection_rank_2_attempts",
    "selection_rank_2_faithful_attempts",
    "selection_rank_2_fidelity_rate",
    "selection_rank_3_attempts",
    "selection_rank_3_faithful_attempts",
    "selection_rank_3_fidelity_rate",
    "selection_rank_4_attempts",
    "selection_rank_4_faithful_attempts",
    "selection_rank_4_fidelity_rate",
    "prepared_multiple_choice_items",
    "prepared_gold_option_edited_items",
    "prepared_pair_gold_option_edit_rate",
    "substitution_count",
    "duplication_count",
    "deletion_count",
)

PAPER_REFERENCE_VALUES: dict[str, object] = {
    "sources": {
        "final_pdf": "rounded published rates and 0/68,650 rank-1 result",
        "archival_reanalysis": _ARCHIVAL_TARGETING_REFERENCE,
    },
    "source_by_metric": {
        "four_distinct_words": {
            "rounded_rates": "final_pdf",
            "exact_counts": "archival_reanalysis",
        },
        "top_selected_attribution_attempt": {
            "rank_1_result": "final_pdf",
        },
        "attribution_apply_rank_miss": {
            "rank_2_to_4_exact_counts": "archival_reanalysis",
        },
        "all_evaluable_target_miss_rate": {
            "rounded_rate": "final_pdf",
            "exact_counts": "archival_reanalysis",
        },
        "conditional_gold_option_edit_rate": {
            "rounded_rate": "final_pdf",
            "exact_counts": "archival_reanalysis",
        },
    },
    "four_distinct_words": {
        "settings": 42,
        "attribution-4": {"items_with_four": 56141, "items": 68660, "rate": 0.818},
        "random-4": {"items_with_four": 65702, "items": 68660, "rate": 0.957},
    },
    "top_selected_attribution_attempt": {"misplaced": 0, "attempts": 68650},
    "attribution_apply_rank_miss": {
        "rank_1": {"misplaced": 0, "decidable": 68650, "undecidable": 0},
        "rank_2": {"misplaced": 22049, "decidable": 68148, "undecidable": 486},
        "rank_3": {"misplaced": 27053, "decidable": 67717, "undecidable": 876},
        "rank_4": {"misplaced": 31652, "decidable": 67402, "undecidable": 1148},
    },
    "all_evaluable_target_miss_rate": {
        "rate": 0.302,
        "misplaced": 163043,
        "attempts": 540724,
        "legacy_unevaluable_attempts": 7589,
        "targeting_conditions": ["attribution-4", "random-4"],
    },
    "conditional_gold_option_edit_rate": {
        "rate": 0.215,
        "numerator": 3501,
        "denominator": 16316,
        "settings": 20,
        "cohort": "Attribution-4 CoT-swap included multiple-choice items",
        "computable_from_prepared_pairs_alone": False,
    },
}


class TargetingFidelityAuditError(ValueError):
    """Raised when an input cannot support the paper's stated denominators."""


@dataclass(frozen=True, slots=True)
class TargetingFidelityAuditConfig:
    """Arguments for one deterministic, CPU-only audit."""

    pairs_root: Path
    output_dir: Path
    expected_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs_root", Path(self.pairs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if isinstance(self.expected_seed, bool) or not isinstance(self.expected_seed, int):
            raise ValueError("expected_seed must be an integer")


@dataclass(frozen=True, slots=True)
class TargetingFidelityAuditResult:
    """Published paths and headline input counts."""

    output_dir: Path
    records_path: Path
    summary_path: Path
    operation_counts_path: Path
    run_path: Path
    items: int
    settings: int
    input_cells: int


@dataclass(frozen=True, order=True, slots=True)
class _Cell:
    model: str
    benchmark: str
    targeting: str
    seed: int


@dataclass(frozen=True, slots=True)
class _ManifestAudit:
    cell: _Cell
    record_count: int
    max_new_tokens: int
    dataset_sample_count: int
    dataset_records_sha256: str
    model_revision: str | None
    dataset_loader: str
    dataset_cohort_rule: str
    dataset_samples_per_subset: int | None
    random_seed_algorithm: str
    target_position: str
    alignment: str
    historical_compatibility_notes: tuple[str, ...]
    environment_versions: tuple[tuple[str, str], ...]
    cuda: str | None
    gpu_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InputAudit:
    manifest: _ManifestAudit
    sample_ids_sha256: str


@dataclass(frozen=True, slots=True)
class _AuditedItem:
    cell: _Cell
    sample_id: str
    four_distinct_words: bool
    num_distinct_edited_words: int
    attempts: tuple[dict[str, object], ...]
    gold_option_applicable: bool
    gold_option_edited: bool | None
    payload: dict[str, object]


@dataclass(slots=True)
class _Totals:
    items: int = 0
    zero_attempt_items: int = 0
    zero_aligned_word_items: int = 0
    attempted_but_zero_aligned_word_items: int = 0
    four_distinct_word_items: int = 0
    target_attempts: int = 0
    faithful_target_attempts: int = 0
    all_attempts_faithful_items: int = 0
    multiple_choice_items: int = 0
    gold_option_edited_items: int = 0
    operations: Counter[str] = field(default_factory=Counter)
    rank_attempts: Counter[int] = field(default_factory=Counter)
    rank_faithful: Counter[int] = field(default_factory=Counter)

    def add(self, item: _AuditedItem) -> None:
        self.items += 1
        self.zero_attempt_items += int(not item.attempts)
        self.zero_aligned_word_items += int(item.num_distinct_edited_words == 0)
        self.attempted_but_zero_aligned_word_items += int(
            bool(item.attempts) and item.num_distinct_edited_words == 0
        )
        self.four_distinct_word_items += int(item.four_distinct_words)
        if item.gold_option_applicable:
            self.multiple_choice_items += 1
            self.gold_option_edited_items += int(item.gold_option_edited is True)
        self.all_attempts_faithful_items += int(
            bool(item.attempts)
            and all(bool(attempt["landed_on_intended_token"]) for attempt in item.attempts)
        )
        for attempt in item.attempts:
            faithful = bool(attempt["landed_on_intended_token"])
            operation = str(attempt["operation"])
            self.target_attempts += 1
            self.faithful_target_attempts += int(faithful)
            self.operations[operation] += 1
            if item.cell.targeting == "attribution-4":
                rank = int(attempt["selection_rank"])
                if 1 <= rank <= 4:
                    self.rank_attempts[rank] += 1
                    self.rank_faithful[rank] += int(faithful)


@dataclass(slots=True)
class _Aggregates:
    by_cell: dict[_Cell, _Totals] = field(default_factory=dict)
    by_targeting: dict[str, _Totals] = field(default_factory=dict)
    overall: _Totals = field(default_factory=_Totals)

    def add(self, item: _AuditedItem) -> None:
        self.by_cell.setdefault(item.cell, _Totals()).add(item)
        self.by_targeting.setdefault(item.cell.targeting, _Totals()).add(item)
        self.overall.add(item)


def _error(path: Path, message: str) -> TargetingFidelityAuditError:
    return TargetingFidelityAuditError(f"{path}: {message}")


def _mapping(value: object, *, path: Path, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, f"{field_name} must be an object")
    return value


def _list(value: object, *, path: Path, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(path, f"{field_name} must be a list")
    return value


def _string(value: object, *, path: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, f"{field_name} must be a non-empty string")
    return value


def _integer(value: object, *, path: Path, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, f"{field_name} must be an integer")
    return value


def _boolean(value: object, *, path: Path, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, f"{field_name} must be a boolean")
    return value


def _finite_number(value: object, *, path: Path, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, f"{field_name} must be a finite relevance number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, f"{field_name} must be a finite relevance number")
    return result


def _optional_integer(value: object, *, path: Path, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path=path, field_name=field_name)


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
    except TargetingFidelityAuditError as exc:
        raise _error(path, str(exc)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(path, f"cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _error(path, "top-level JSON value must be an object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise TargetingFidelityAuditError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> object:
    raise TargetingFidelityAuditError(f"non-finite JSON constant {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise _error(path, f"input is outside pairs root {root.resolve()}") from exc


def _validate_manifest(
    path: Path,
    *,
    expected_seed: int,
) -> _ManifestAudit:
    manifest = _load_json(path)
    if manifest.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise _error(path, "unknown prepare-edited-pairs run schema")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise _error(path, "paper SHA-256 does not match the canonical final PDF")
    if manifest.get("operation") != "prepare-edited-pairs":
        raise _error(path, "manifest operation is not prepare-edited-pairs")
    if manifest.get("status") != "completed":
        raise _error(path, "input run is not completed")
    if manifest.get("failures") != []:
        raise _error(path, "completed input run still records failures")

    arguments = _mapping(manifest.get("arguments"), path=path, field_name="arguments")
    model = _string(arguments.get("model"), path=path, field_name="arguments.model")
    benchmark = _string(arguments.get("benchmark"), path=path, field_name="arguments.benchmark")
    if benchmark not in PUBLIC_BENCHMARKS:
        raise _error(path, f"unsupported benchmark {benchmark!r}")
    targeting = _string(arguments.get("targeting"), path=path, field_name="arguments.targeting")
    if targeting not in TARGETING_CONDITIONS:
        raise _error(path, f"unsupported targeting condition {targeting!r}")
    if _integer(arguments.get("num_edits"), path=path, field_name="arguments.num_edits") != 4:
        raise _error(path, "Appendix A audit requires the requested four edits")
    if arguments.get("limit") is not None:
        raise _error(path, "partial --limit inputs cannot enter the Appendix A denominator")
    seed = _integer(arguments.get("seed"), path=path, field_name="arguments.seed")
    if seed != expected_seed:
        raise _error(path, f"input uses seed {seed}; expected seed {expected_seed}")
    max_new_tokens = _integer(
        arguments.get("max_new_tokens"),
        path=path,
        field_name="arguments.max_new_tokens",
    )
    if max_new_tokens <= 0:
        raise _error(path, "arguments.max_new_tokens must be positive")

    decoding = _mapping(manifest.get("decoding"), path=path, field_name="decoding")
    if decoding.get("strategy") != "greedy":
        raise _error(path, "decoding strategy must be greedy")
    if decoding.get("dtype") != "bfloat16":
        raise _error(path, "decoding dtype must be bfloat16")
    if decoding.get("padding_side") != "left":
        raise _error(path, "decoding padding_side must be left")
    if decoding.get("max_new_tokens") != max_new_tokens:
        raise _error(path, "decoding max_new_tokens does not match arguments")

    provenance = _mapping(manifest.get("provenance"), path=path, field_name="provenance")
    expected_protocol = {
        "random_seed_algorithm": _RANDOM_SEED_ALGORITHM,
        "target_position": _TARGET_POSITION,
        "alignment": _ALIGNMENT,
    }
    for field_name, expected_value in expected_protocol.items():
        if provenance.get(field_name) != expected_value:
            raise _error(
                path,
                f"provenance.{field_name} must be {expected_value!r}",
            )
    if provenance.get("model") != model:
        raise _error(path, "provenance.model does not match arguments.model")
    dataset_cohort_rule = _string(
        provenance.get("dataset_cohort_rule"),
        path=path,
        field_name="provenance.dataset_cohort_rule",
    )
    if dataset_cohort_rule != _DATASET_COHORT_RULE:
        raise _error(
            path,
            f"provenance.dataset_cohort_rule must be {_DATASET_COHORT_RULE!r}",
        )
    dataset_samples_per_subset = _optional_integer(
        provenance.get("dataset_samples_per_subset"),
        path=path,
        field_name="provenance.dataset_samples_per_subset",
    )
    expected_samples_per_subset = _expected_samples_per_subset(model, benchmark)
    if dataset_samples_per_subset != expected_samples_per_subset:
        raise _error(
            path,
            "provenance.dataset_samples_per_subset does not match the final-paper "
            f"model/benchmark cohort (expected {expected_samples_per_subset!r})",
        )
    dataset_loader = _string(
        provenance.get("benchmark_dataset_loader"),
        path=path,
        field_name="provenance.benchmark_dataset_loader",
    )
    if dataset_loader != _BENCHMARK_DATASET_LOADERS[benchmark]:
        raise _error(path, "provenance benchmark dataset loader does not match benchmark")
    dataset_sample_count = _integer(
        provenance.get("dataset_sample_count"),
        path=path,
        field_name="provenance.dataset_sample_count",
    )
    dataset_records_sha256 = _string(
        provenance.get("dataset_records_sha256"),
        path=path,
        field_name="provenance.dataset_records_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{64}", dataset_records_sha256) is None:
        raise _error(path, "provenance.dataset_records_sha256 must be lowercase SHA-256")
    if "model_revision" not in provenance:
        raise _error(path, "provenance.model_revision is required")
    raw_model_revision = provenance.get("model_revision")
    if raw_model_revision is not None and (
        not isinstance(raw_model_revision, str) or not raw_model_revision
    ):
        raise _error(path, "provenance.model_revision must be null or a non-empty string")
    model_revision = raw_model_revision if isinstance(raw_model_revision, str) else None
    historical_compatibility_notes = tuple(
        _string(
            note,
            path=path,
            field_name=f"provenance.historical_compatibility_notes[{index}]",
        )
        for index, note in enumerate(
            _list(
                provenance.get("historical_compatibility_notes"),
                path=path,
                field_name="provenance.historical_compatibility_notes",
            )
        )
    )
    if historical_compatibility_notes != _HISTORICAL_COMPATIBILITY_NOTES:
        raise _error(
            path,
            "provenance.historical_compatibility_notes does not match the public-v1 protocol",
        )
    environment_versions = tuple(
        (
            field_name,
            _string(
                provenance.get(field_name),
                path=path,
                field_name=f"provenance.{field_name}",
            ),
        )
        for field_name in _ENVIRONMENT_VERSION_FIELDS
    )
    raw_cuda = provenance.get("cuda")
    if raw_cuda is not None and (not isinstance(raw_cuda, str) or not raw_cuda):
        raise _error(path, "provenance.cuda must be null or a non-empty string")
    cuda = raw_cuda if isinstance(raw_cuda, str) else None
    gpu_names = tuple(
        _string(
            gpu_name,
            path=path,
            field_name=f"provenance.gpu_names[{index}]",
        )
        for index, gpu_name in enumerate(
            _list(
                provenance.get("gpu_names"),
                path=path,
                field_name="provenance.gpu_names",
            )
        )
    )

    counts = _mapping(manifest.get("counts"), path=path, field_name="counts")
    discovered = _integer(counts.get("discovered"), path=path, field_name="counts.discovered")
    written = _integer(counts.get("written"), path=path, field_name="counts.written")
    failed = _integer(counts.get("failed"), path=path, field_name="counts.failed")
    if written <= 0:
        raise _error(path, "completed input contains no pair records")
    if discovered != written or failed != 0:
        raise _error(path, "completed input has partial discovered/written/failed counts")
    if dataset_sample_count != written:
        raise _error(path, "dataset sample count does not match completed record count")
    return _ManifestAudit(
        cell=_Cell(model, benchmark, targeting, seed),
        record_count=written,
        max_new_tokens=max_new_tokens,
        dataset_sample_count=dataset_sample_count,
        dataset_records_sha256=dataset_records_sha256,
        model_revision=model_revision,
        dataset_loader=dataset_loader,
        dataset_cohort_rule=dataset_cohort_rule,
        dataset_samples_per_subset=dataset_samples_per_subset,
        random_seed_algorithm=_RANDOM_SEED_ALGORITHM,
        target_position=_TARGET_POSITION,
        alignment=_ALIGNMENT,
        historical_compatibility_notes=historical_compatibility_notes,
        environment_versions=environment_versions,
        cuda=cuda,
        gpu_names=gpu_names,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise _error(path, f"blank JSONL row at line {line_number}")
                try:
                    payload = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_nonfinite_constant,
                    )
                except (json.JSONDecodeError, TargetingFidelityAuditError) as exc:
                    raise _error(path, f"invalid JSON at line {line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise _error(path, f"line {line_number} is not a JSON object")
                payload["__source_line__"] = line_number
                rows.append(payload)
    except OSError as exc:
        raise _error(path, f"cannot read JSONL: {exc}") from exc
    return rows


def _dataset_records_sha256(rows: Sequence[Mapping[str, object]], *, path: Path) -> str:
    fingerprint_rows: list[dict[str, object]] = []
    for row in rows:
        line = row.get("__source_line__")
        prefix = f"line {line}"
        clean = _mapping(row.get("clean"), path=path, field_name=f"{prefix}.clean")
        raw_choices = clean.get("choices")
        if raw_choices is None:
            choices: list[str] | None = None
        else:
            choices = [
                _string(
                    choice,
                    path=path,
                    field_name=f"{prefix}.clean.choices[{index}]",
                )
                for index, choice in enumerate(
                    _list(raw_choices, path=path, field_name=f"{prefix}.clean.choices")
                )
            ]
        subset = row.get("subset")
        if subset is not None and not isinstance(subset, str):
            raise _error(path, f"{prefix}.subset must be null or a string")
        fingerprint_rows.append(
            {
                "sample_id": _string(
                    row.get("sample_id"), path=path, field_name=f"{prefix}.sample_id"
                ),
                "question": _string(
                    clean.get("question"),
                    path=path,
                    field_name=f"{prefix}.clean.question",
                ),
                "choices": choices,
                "correct_answer": _string(
                    row.get("gold_answer"),
                    path=path,
                    field_name=f"{prefix}.gold_answer",
                ),
                "subset": subset,
            }
        )
    serialized = json.dumps(
        fingerprint_rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _span(
    value: object,
    *,
    path: Path,
    field_name: str,
    text_length: int,
) -> tuple[int, int]:
    payload = _mapping(value, path=path, field_name=field_name)
    start = _integer(payload.get("start"), path=path, field_name=f"{field_name}.start")
    end = _integer(payload.get("end"), path=path, field_name=f"{field_name}.end")
    if not 0 <= start < end <= text_length:
        raise _error(path, f"{field_name} is outside its editable text")
    return start, end


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _WORD.finditer(text)]


def _word_index_at(
    text: str,
    position: int,
    *,
    path: Path,
    field_name: str,
) -> int:
    for index, (start, end) in enumerate(_word_spans(text)):
        if start <= position < end:
            return index
    raise _error(path, f"{field_name} does not belong to a whitespace-delimited word")


def _first_ascii_letter(
    text: str,
    start: int,
    end: int,
    *,
    path: Path,
    field_name: str,
) -> int:
    for position in range(start, end):
        character = text[position]
        if character.isascii() and character.isalpha():
            return position
    raise _error(path, f"{field_name} contains no eligible ASCII letter")


def _validate_attribution_target(
    record: Mapping[str, object],
    *,
    path: Path,
    clean_prompt_token_count: int,
) -> None:
    target = _mapping(
        record.get("attribution_target"),
        path=path,
        field_name="attribution_target",
    )
    definition = _string(
        target.get("definition"),
        path=path,
        field_name="attribution_target.definition",
    )
    if definition != _TARGET_POSITION:
        raise _error(path, f"attribution_target.definition must be {_TARGET_POSITION!r}")
    context = _string(
        target.get("context"),
        path=path,
        field_name="attribution_target.context",
    )
    if context != "complete-clean-generation":
        raise _error(
            path,
            "attribution_target.context must be 'complete-clean-generation'",
        )
    position = _integer(
        target.get("position"),
        path=path,
        field_name="attribution_target.position",
    )
    if position != clean_prompt_token_count:
        raise _error(
            path,
            "attribution_target.position must equal the clean prompt token count",
        )
    first_cot_token_id = _integer(
        target.get("first_cot_token_id"),
        path=path,
        field_name="attribution_target.first_cot_token_id",
    )
    if first_cot_token_id < 0:
        raise _error(path, "attribution_target.first_cot_token_id must be non-negative")
    _string(
        target.get("first_cot_token_text"),
        path=path,
        field_name="attribution_target.first_cot_token_text",
    )


def _token_index_sequence(
    value: object,
    *,
    path: Path,
    field_name: str,
    prompt_token_count: int,
) -> tuple[int, ...]:
    raw_indices = _list(value, path=path, field_name=field_name)
    if not raw_indices:
        raise _error(path, f"{field_name} must be non-empty")
    indices = tuple(
        _integer(
            token_index,
            path=path,
            field_name=f"{field_name}[{index}]",
        )
        for index, token_index in enumerate(raw_indices)
    )
    if any(token_index < 0 or token_index >= prompt_token_count for token_index in indices):
        raise _error(path, f"{field_name} must lie inside its prompt token count")
    if any(previous >= current for previous, current in zip(indices, indices[1:], strict=False)):
        raise _error(path, f"{field_name} must be strictly increasing")
    return indices


def _gold_option(
    record: Mapping[str, object],
    *,
    path: Path,
    benchmark: str,
    aligned_words: Sequence[Mapping[str, object]],
) -> tuple[bool, bool | None, dict[str, int] | None]:
    clean = _mapping(record.get("clean"), path=path, field_name="clean")
    question = _string(clean.get("question"), path=path, field_name="clean.question")
    editable = _string(clean.get("editable_text"), path=path, field_name="clean.editable_text")
    choices_value = clean.get("choices")
    expects_choices = benchmark in _MULTIPLE_CHOICE_BENCHMARKS
    if not expects_choices:
        if choices_value is not None:
            raise _error(path, f"{benchmark} must not contribute a gold-option denominator")
        if editable != question:
            raise _error(path, "non-multiple-choice editable text differs from the frozen question")
        return False, None, None

    raw_choices = _list(choices_value, path=path, field_name="clean.choices")
    if not 1 <= len(raw_choices) <= 10:
        raise _error(path, "clean.choices must contain between one and ten options")
    choices = [
        _string(choice, path=path, field_name=f"clean.choices[{index}]")
        for index, choice in enumerate(raw_choices)
    ]
    gold_answer = _string(record.get("gold_answer"), path=path, field_name="gold_answer")
    if len(gold_answer) != 1 or not "A" <= gold_answer <= "J":
        raise _error(path, "gold_answer must be an uppercase option letter")
    gold_index = ord(gold_answer) - ord("A")
    if gold_index >= len(choices):
        raise _error(path, "gold_answer is outside clean.choices")

    option_segments = [
        f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(choices)
    ]
    expected_editable = f"{question}\n{' '.join(option_segments)}"
    if editable != expected_editable:
        raise _error(path, "clean editable text does not match the frozen formatted choices")
    gold_start = len(question) + 1
    if gold_index:
        gold_start += sum(len(segment) for segment in option_segments[:gold_index]) + gold_index
    gold_start += len(f"({gold_answer}) ")
    gold_end = gold_start + len(choices[gold_index])
    edited = any(
        int(word["clean_span_start"]) < gold_end and gold_start < int(word["clean_span_end"])
        for word in aligned_words
    )
    return True, edited, {"start": gold_start, "end": gold_end}


def _validate_excluded_tokens(
    record: Mapping[str, object],
    *,
    path: Path,
    cell: _Cell,
    editable_prompt_span: tuple[int, int],
    clean_prompt_token_count: int,
    num_candidates: int,
) -> set[int]:
    raw_excluded = _list(
        record.get("excluded_attribution_tokens"),
        path=path,
        field_name="excluded_attribution_tokens",
    )
    if cell.targeting == "attribution-4":
        if raw_excluded:
            raise _error(path, "Attribution-4 must not declare excluded attribution tokens")
        return set()
    expected_excluded = min(4, num_candidates)
    if len(raw_excluded) != expected_excluded:
        raise _error(
            path,
            "Random-4 must exclude the attribution top four, or every candidate when fewer "
            f"than four exist (expected {expected_excluded})",
        )

    editable_start, editable_end = editable_prompt_span
    token_indices: set[int] = set()
    for expected_rank, value in enumerate(raw_excluded, 1):
        prefix = f"excluded_attribution_tokens[{expected_rank - 1}]"
        token = _mapping(value, path=path, field_name=prefix)
        rank = _integer(
            token.get("attribution_rank"),
            path=path,
            field_name=f"{prefix}.attribution_rank",
        )
        if rank != expected_rank:
            raise _error(path, "Random-4 excluded tokens must be attribution ranks 1-4")
        token_index = _integer(
            token.get("token_index"),
            path=path,
            field_name=f"{prefix}.token_index",
        )
        if (
            token_index < 0
            or token_index >= clean_prompt_token_count
            or token_index in token_indices
        ):
            raise _error(
                path,
                "Random-4 excluded token indices must be distinct and inside the clean "
                "prompt token count",
            )
        token_indices.add(token_index)
        _string(token.get("text"), path=path, field_name=f"{prefix}.text")
        _finite_number(token.get("relevance"), path=path, field_name=f"{prefix}.relevance")
        prompt_start = _integer(
            token.get("prompt_start"), path=path, field_name=f"{prefix}.prompt_start"
        )
        prompt_end = _integer(token.get("prompt_end"), path=path, field_name=f"{prefix}.prompt_end")
        if not editable_start <= prompt_start < prompt_end <= editable_end:
            raise _error(path, f"{prefix} lies outside the clean editable prompt span")
    return token_indices


def _validate_pair(
    record: dict[str, object],
    *,
    path: Path,
    relative_path: str,
    cell: _Cell,
) -> _AuditedItem:
    line_number = _integer(record.pop("__source_line__", None), path=path, field_name="source line")
    item_path = Path(f"{path}:{line_number}")
    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise _error(item_path, "unknown pair schema")
    sample_id = _string(record.get("sample_id"), path=item_path, field_name="sample_id")
    expected = {
        "model": cell.model,
        "benchmark": cell.benchmark,
        "targeting": cell.targeting,
        "seed": cell.seed,
    }
    for field_name, expected_value in expected.items():
        if record.get(field_name) != expected_value:
            raise _error(
                item_path,
                f"pair {field_name} {record.get(field_name)!r} does not match manifest "
                f"{expected_value!r}",
            )
    if (
        _integer(
            record.get("num_edits_requested"),
            path=item_path,
            field_name="num_edits_requested",
        )
        != 4
    ):
        raise _error(item_path, "pair did not request four edits")

    clean = _mapping(record.get("clean"), path=item_path, field_name="clean")
    editable = _string(clean.get("editable_text"), path=item_path, field_name="clean.editable_text")
    clean_prompt = _string(clean.get("prompt"), path=item_path, field_name="clean.prompt")
    clean_prompt_token_count = _integer(
        clean.get("prompt_token_count"),
        path=item_path,
        field_name="clean.prompt_token_count",
    )
    if clean_prompt_token_count <= 0:
        raise _error(item_path, "clean prompt token count must be positive")
    _validate_attribution_target(
        record,
        path=item_path,
        clean_prompt_token_count=clean_prompt_token_count,
    )
    editable_prompt_start, editable_prompt_end = _span(
        clean.get("editable_prompt_span"),
        path=item_path,
        field_name="clean.editable_prompt_span",
        text_length=len(clean_prompt),
    )
    if clean_prompt[editable_prompt_start:editable_prompt_end] != editable:
        raise _error(item_path, "clean editable prompt span does not match editable_text")
    num_candidates = _integer(
        record.get("num_candidates"), path=item_path, field_name="num_candidates"
    )
    if num_candidates < 0:
        raise _error(item_path, "num_candidates must be non-negative")
    excluded_token_indices = _validate_excluded_tokens(
        record,
        path=item_path,
        cell=cell,
        editable_prompt_span=(editable_prompt_start, editable_prompt_end),
        clean_prompt_token_count=clean_prompt_token_count,
        num_candidates=num_candidates,
    )
    raw_attempts = _list(
        record.get("target_attempts"), path=item_path, field_name="target_attempts"
    )
    declared_attempts = _integer(
        record.get("num_target_attempts"),
        path=item_path,
        field_name="num_target_attempts",
    )
    if declared_attempts != len(raw_attempts):
        raise _error(item_path, "num_target_attempts does not match target_attempts")
    if len(raw_attempts) > 4:
        raise _error(item_path, "target_attempts must contain at most four edits")
    if num_candidates < len(raw_attempts) + len(excluded_token_indices):
        raise _error(item_path, "num_candidates is smaller than recorded target provenance")

    current_text = editable
    origins: list[int | None] = list(range(len(editable)))
    cumulative_shift = 0
    seen_target_token_indices: set[int] = set()
    seen_attribution_ranks: set[int] = set()
    attempts: list[dict[str, object]] = []
    for index, value in enumerate(raw_attempts, 1):
        field_prefix = f"target_attempts[{index - 1}]"
        attempt = _mapping(value, path=item_path, field_name=field_prefix)
        selection_rank = _integer(
            attempt.get("selection_rank"),
            path=item_path,
            field_name=f"{field_prefix}.selection_rank",
        )
        if selection_rank != index:
            raise _error(item_path, "target-attempt selection ranks must be contiguous from one")
        attribution_rank = _integer(
            attempt.get("attribution_rank"),
            path=item_path,
            field_name=f"{field_prefix}.attribution_rank",
        )
        if attribution_rank <= 0:
            raise _error(item_path, "attribution ranks must be positive")
        if attribution_rank > num_candidates:
            raise _error(
                item_path,
                f"{field_prefix}.attribution_rank must not exceed num_candidates",
            )
        if attribution_rank in seen_attribution_ranks:
            raise _error(item_path, "target attribution ranks must be distinct")
        if cell.targeting == "random-4" and attribution_rank <= 4:
            raise _error(item_path, "Random-4 targets must exclude the attribution top four")
        if (
            cell.targeting == "attribution-4"
            and seen_attribution_ranks
            and attribution_rank <= max(seen_attribution_ranks)
        ):
            raise _error(item_path, "Attribution-4 target ranks must remain increasing")
        seen_attribution_ranks.add(attribution_rank)
        target_token_index = _integer(
            attempt.get("target_token_index"),
            path=item_path,
            field_name=f"{field_prefix}.target_token_index",
        )
        if (
            target_token_index < 0
            or target_token_index >= clean_prompt_token_count
            or target_token_index in seen_target_token_indices
            or target_token_index in excluded_token_indices
        ):
            raise _error(
                item_path,
                "target token indices must be distinct, inside the clean prompt token count, "
                "and not excluded",
            )
        seen_target_token_indices.add(target_token_index)
        target_token_text = _string(
            attempt.get("target_token_text"),
            path=item_path,
            field_name=f"{field_prefix}.target_token_text",
        )
        _finite_number(
            attempt.get("relevance"),
            path=item_path,
            field_name=f"{field_prefix}.relevance",
        )

        intended_start, intended_end = _span(
            attempt.get("intended_editable_span"),
            path=item_path,
            field_name=f"{field_prefix}.intended_editable_span",
            text_length=len(editable),
        )
        intended_prompt_start, intended_prompt_end = _span(
            attempt.get("intended_prompt_span"),
            path=item_path,
            field_name=f"{field_prefix}.intended_prompt_span",
            text_length=len(clean_prompt),
        )
        if (intended_prompt_start, intended_prompt_end) != (
            editable_prompt_start + intended_start,
            editable_prompt_start + intended_end,
        ):
            raise _error(
                item_path,
                f"{field_prefix}.intended_prompt_span does not match clean editable coordinates",
            )
        landed_start, landed_end = _span(
            attempt.get("landed_editable_span_before"),
            path=item_path,
            field_name=f"{field_prefix}.landed_editable_span_before",
            text_length=len(current_text),
        )
        if (landed_start, landed_end) != (
            intended_start + cumulative_shift,
            intended_end + cumulative_shift,
        ):
            raise _error(
                item_path,
                f"{field_prefix} violates the producer cumulative-shift landing invariant",
            )
        landed_text = _string(
            attempt.get("landed_text_before"),
            path=item_path,
            field_name=f"{field_prefix}.landed_text_before",
        )
        if current_text[landed_start:landed_end] != landed_text:
            raise _error(item_path, f"{field_prefix}.landed_text_before does not match replay")

        character_index = _integer(
            attempt.get("character_index"),
            path=item_path,
            field_name=f"{field_prefix}.character_index",
        )
        if not 0 <= character_index < len(landed_text):
            raise _error(item_path, f"{field_prefix}.character_index is outside landed text")
        original_character = _string(
            attempt.get("original_character"),
            path=item_path,
            field_name=f"{field_prefix}.original_character",
        )
        if len(original_character) != 1:
            raise _error(item_path, f"{field_prefix}.original_character must be one character")
        if landed_text[character_index] != original_character:
            raise _error(item_path, f"{field_prefix}.original_character does not match replay")
        if not (original_character.isascii() and original_character.isalpha()):
            raise _error(item_path, f"{field_prefix} must edit an ASCII letter")

        reported_faithful = _boolean(
            attempt.get("landed_on_intended_token"),
            path=item_path,
            field_name=f"{field_prefix}.landed_on_intended_token",
        )
        operation = _string(
            attempt.get("operation"),
            path=item_path,
            field_name=f"{field_prefix}.operation",
        )
        if operation not in _OPERATIONS:
            raise _error(item_path, f"unknown typo operation {operation!r}")
        new_character = attempt.get("new_character")
        if operation == "substitution":
            if not isinstance(new_character, str) or len(new_character) != 1:
                raise _error(item_path, f"{field_prefix}.new_character must be one character")
            if new_character == original_character:
                raise _error(item_path, "substitution must change the selected character")
            expected_edited = (
                landed_text[:character_index] + new_character + landed_text[character_index + 1 :]
            )
            replacement_origins = origins[landed_start:landed_end]
        elif operation == "duplication":
            if new_character != original_character:
                raise _error(item_path, "duplication new_character must equal original_character")
            expected_edited = (
                landed_text[: character_index + 1]
                + original_character
                + landed_text[character_index + 1 :]
            )
            local_origins = origins[landed_start:landed_end]
            replacement_origins = (
                local_origins[: character_index + 1] + [None] + local_origins[character_index + 1 :]
            )
        else:
            if new_character is not None:
                raise _error(item_path, "deletion new_character must be null")
            ascii_letters = sum(
                character.isascii() and character.isalpha() for character in landed_text
            )
            if ascii_letters <= 1:
                raise _error(item_path, "deletion requires more than one ASCII letter")
            expected_edited = landed_text[:character_index] + landed_text[character_index + 1 :]
            local_origins = origins[landed_start:landed_end]
            replacement_origins = (
                local_origins[:character_index] + local_origins[character_index + 1 :]
            )
        edited_token_text = _string(
            attempt.get("edited_token_text"),
            path=item_path,
            field_name=f"{field_prefix}.edited_token_text",
        )
        seeded_edit = seeded_character_edit(
            landed_text,
            _stable_seed(
                cell.seed,
                sample_id,
                target_token_index,
                target_token_text,
            ),
        )
        if seeded_edit is None or (
            seeded_edit.operation,
            seeded_edit.character_index,
            seeded_edit.original_character,
            seeded_edit.new_character,
            seeded_edit.edited,
        ) != (
            operation,
            character_index,
            original_character,
            new_character,
            edited_token_text,
        ):
            raise _error(
                item_path,
                f"{field_prefix} does not match the SHA-seeded Table 4 edit",
            )
        if edited_token_text != expected_edited:
            raise _error(item_path, f"{field_prefix}.edited_token_text does not match replay")

        affected_position = landed_start + character_index
        replayed_origin = origins[affected_position]
        reported_origin = _optional_integer(
            attempt.get("landed_origin_index"),
            path=item_path,
            field_name=f"{field_prefix}.landed_origin_index",
        )
        if reported_origin != replayed_origin:
            raise _error(item_path, f"{field_prefix}.landed_origin_index does not match replay")
        replayed_faithful = (
            replayed_origin is not None and intended_start <= replayed_origin < intended_end
        )
        if reported_faithful != replayed_faithful:
            raise _error(item_path, f"{field_prefix} landing flag does not match origin replay")

        intended_letter = _first_ascii_letter(
            editable,
            intended_start,
            intended_end,
            path=item_path,
            field_name=f"{field_prefix}.intended_editable_span",
        )
        replayed_intended_word = _word_index_at(
            editable,
            intended_letter,
            path=item_path,
            field_name=f"{field_prefix}.intended_editable_span",
        )
        replayed_landed_word = _word_index_at(
            current_text,
            affected_position,
            path=item_path,
            field_name=f"{field_prefix}.landed_editable_span_before",
        )
        intended_word = _integer(
            attempt.get("intended_word_index"),
            path=item_path,
            field_name=f"{field_prefix}.intended_word_index",
        )
        landed_word = _integer(
            attempt.get("landed_word_index"),
            path=item_path,
            field_name=f"{field_prefix}.landed_word_index",
        )
        if intended_word != replayed_intended_word:
            raise _error(item_path, f"{field_prefix}.intended_word_index does not match replay")
        if landed_word != replayed_landed_word:
            raise _error(item_path, f"{field_prefix}.landed_word_index does not match replay")

        attempts.append(
            {
                "selection_rank": selection_rank,
                "attribution_rank": attribution_rank,
                "target_token_index": target_token_index,
                "landed_on_intended_token": replayed_faithful,
                "intended_word_index": intended_word,
                "landed_word_index": landed_word,
                "operation": operation,
            }
        )
        origins = origins[:landed_start] + replacement_origins + origins[landed_end:]
        current_text = current_text[:landed_start] + edited_token_text + current_text[landed_end:]
        cumulative_shift += len(edited_token_text) - len(landed_text)

    edited = _mapping(record.get("edited"), path=item_path, field_name="edited")
    recorded_edited = _string(
        edited.get("editable_text"), path=item_path, field_name="edited.editable_text"
    )
    if recorded_edited != current_text:
        raise _error(item_path, "replayed edited text does not match edited.editable_text")
    edited_prompt = _string(edited.get("prompt"), path=item_path, field_name="edited.prompt")
    edited_prompt_token_count = _integer(
        edited.get("prompt_token_count"),
        path=item_path,
        field_name="edited.prompt_token_count",
    )
    if edited_prompt_token_count <= 0:
        raise _error(item_path, "edited prompt token count must be positive")
    edited_prompt_start, edited_prompt_end = _span(
        edited.get("editable_prompt_span"),
        path=item_path,
        field_name="edited.editable_prompt_span",
        text_length=len(edited_prompt),
    )
    expected_edited_prompt = (
        clean_prompt[:editable_prompt_start] + recorded_edited + clean_prompt[editable_prompt_end:]
    )
    if (
        (edited_prompt_start, edited_prompt_end)
        != (editable_prompt_start, editable_prompt_start + len(recorded_edited))
        or edited_prompt[edited_prompt_start:edited_prompt_end] != recorded_edited
        or edited_prompt != expected_edited_prompt
    ):
        raise _error(item_path, "edited prompt reconstruction does not match the clean prompt")

    clean_word_spans = _word_spans(editable)
    edited_word_spans = _word_spans(recorded_edited)
    if len(clean_word_spans) != len(edited_word_spans):
        raise _error(item_path, "character edits changed the whitespace-delimited word count")
    changed_words = [
        (word_index, clean_span, edited_span)
        for word_index, (clean_span, edited_span) in enumerate(
            zip(clean_word_spans, edited_word_spans, strict=True)
        )
        if editable[slice(*clean_span)] != recorded_edited[slice(*edited_span)]
    ]
    raw_words = _list(record.get("aligned_words"), path=item_path, field_name="aligned_words")
    declared_words = _integer(
        record.get("num_aligned_words"), path=item_path, field_name="num_aligned_words"
    )
    if declared_words != len(raw_words):
        raise _error(item_path, "num_aligned_words does not match aligned_words")
    if len(raw_words) != len(changed_words):
        raise _error(item_path, "aligned_words does not match the replayed changed words")
    if len(raw_words) > 4:
        raise _error(item_path, "aligned_words must contain at most four words")
    aligned_words: list[dict[str, object]] = []
    for index, (value, changed) in enumerate(zip(raw_words, changed_words, strict=True)):
        word = _mapping(value, path=item_path, field_name=f"aligned_words[{index}]")
        word_index = _integer(
            word.get("word_index"), path=item_path, field_name=f"aligned_words[{index}].word_index"
        )
        replayed_word_index, replayed_clean_span, replayed_edited_span = changed
        if word_index != replayed_word_index:
            raise _error(item_path, "aligned word indices do not match replayed word order")
        start, end = _span(
            word.get("clean_editable_span"),
            path=item_path,
            field_name=f"aligned_words[{index}].clean_editable_span",
            text_length=len(editable),
        )
        edited_start, edited_end = _span(
            word.get("edited_editable_span"),
            path=item_path,
            field_name=f"aligned_words[{index}].edited_editable_span",
            text_length=len(recorded_edited),
        )
        if (start, end) != replayed_clean_span or (
            edited_start,
            edited_end,
        ) != replayed_edited_span:
            raise _error(item_path, "aligned editable spans do not match replayed words")
        clean_prompt_word_start, clean_prompt_word_end = _span(
            word.get("clean_prompt_span"),
            path=item_path,
            field_name=f"aligned_words[{index}].clean_prompt_span",
            text_length=len(clean_prompt),
        )
        edited_prompt_word_start, edited_prompt_word_end = _span(
            word.get("edited_prompt_span"),
            path=item_path,
            field_name=f"aligned_words[{index}].edited_prompt_span",
            text_length=len(edited_prompt),
        )
        if (clean_prompt_word_start, clean_prompt_word_end) != (
            editable_prompt_start + start,
            editable_prompt_start + end,
        ) or (edited_prompt_word_start, edited_prompt_word_end) != (
            edited_prompt_start + edited_start,
            edited_prompt_start + edited_end,
        ):
            raise _error(item_path, "aligned prompt spans do not match editable word spans")
        clean_text = _string(
            word.get("clean_text"), path=item_path, field_name=f"aligned_words[{index}].clean_text"
        )
        if editable[start:end] != clean_text:
            raise _error(item_path, "aligned clean span does not match clean_text")
        edited_text = _string(
            word.get("edited_text"),
            path=item_path,
            field_name=f"aligned_words[{index}].edited_text",
        )
        if recorded_edited[edited_start:edited_end] != edited_text:
            raise _error(item_path, "aligned edited span does not match edited_text")
        matching_attempts = [
            attempt for attempt in attempts if int(attempt["landed_word_index"]) == word_index
        ]
        replayed_ranks = [int(attempt["selection_rank"]) for attempt in matching_attempts]
        target_ranks = [
            _integer(
                rank,
                path=item_path,
                field_name=f"aligned_words[{index}].target_ranks",
            )
            for rank in _list(
                word.get("target_ranks"),
                path=item_path,
                field_name=f"aligned_words[{index}].target_ranks",
            )
        ]
        if target_ranks != replayed_ranks:
            raise _error(item_path, "aligned target_ranks do not match target-attempt replay")
        replayed_token_indices = [
            int(attempt["target_token_index"]) for attempt in matching_attempts
        ]
        target_token_indices = [
            _integer(
                token_index,
                path=item_path,
                field_name=f"aligned_words[{index}].target_token_indices",
            )
            for token_index in _list(
                word.get("target_token_indices"),
                path=item_path,
                field_name=f"aligned_words[{index}].target_token_indices",
            )
        ]
        if target_token_indices != replayed_token_indices:
            raise _error(
                item_path,
                "aligned target_token_indices do not match target-attempt replay",
            )
        clean_token_indices = _token_index_sequence(
            word.get("clean_token_indices"),
            path=item_path,
            field_name=f"aligned_words[{index}].clean_token_indices",
            prompt_token_count=clean_prompt_token_count,
        )
        edited_token_indices = _token_index_sequence(
            word.get("edited_token_indices"),
            path=item_path,
            field_name=f"aligned_words[{index}].edited_token_indices",
            prompt_token_count=edited_prompt_token_count,
        )
        clean_final_token = _integer(
            word.get("clean_final_token"),
            path=item_path,
            field_name=f"aligned_words[{index}].clean_final_token",
        )
        if clean_final_token != clean_token_indices[-1]:
            raise _error(
                item_path,
                f"aligned_words[{index}].clean_final_token must equal the final "
                "clean_token_indices entry",
            )
        edited_final_token = _integer(
            word.get("edited_final_token"),
            path=item_path,
            field_name=f"aligned_words[{index}].edited_final_token",
        )
        if edited_final_token != edited_token_indices[-1]:
            raise _error(
                item_path,
                f"aligned_words[{index}].edited_final_token must equal the final "
                "edited_token_indices entry",
            )
        aligned_words.append(
            {
                "word_index": word_index,
                "clean_text": clean_text,
                "clean_span_start": start,
                "clean_span_end": end,
            }
        )

    applicable, gold_edited, gold_span = _gold_option(
        record,
        path=item_path,
        benchmark=cell.benchmark,
        aligned_words=aligned_words,
    )
    operation_counts = Counter(str(attempt["operation"]) for attempt in attempts)
    faithful_count = sum(bool(attempt["landed_on_intended_token"]) for attempt in attempts)
    payload: dict[str, object] = {
        "schema_version": "targeting-fidelity-record/v1",
        "model": cell.model,
        "benchmark": cell.benchmark,
        "targeting": cell.targeting,
        "seed": cell.seed,
        "sample_id": sample_id,
        "source_pairs_file": relative_path,
        "source_line": line_number,
        "num_target_attempts": len(attempts),
        "faithful_target_attempts": faithful_count,
        "misplaced_target_attempts": len(attempts) - faithful_count,
        "all_attempts_faithful": bool(attempts) and faithful_count == len(attempts),
        "num_distinct_edited_words": len(aligned_words),
        "four_distinct_words": len(aligned_words) == 4,
        "gold_option_applicable": applicable,
        "gold_option_edited": gold_edited,
        "gold_option_clean_span": gold_span,
        "operation_counts": {operation: operation_counts[operation] for operation in _OPERATIONS},
        "target_attempts": attempts,
    }
    return _AuditedItem(
        cell=cell,
        sample_id=sample_id,
        four_distinct_words=len(aligned_words) == 4,
        num_distinct_edited_words=len(aligned_words),
        attempts=tuple(attempts),
        gold_option_applicable=applicable,
        gold_option_edited=gold_edited,
        payload=payload,
    )


def _rate(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else format(numerator / denominator, ".12g")


def _summary_row(
    *,
    row_type: str,
    model: str,
    benchmark: str,
    targeting: str,
    seed: str,
    totals: _Totals,
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_type": row_type,
        "model": model,
        "benchmark": benchmark,
        "targeting": targeting,
        "seed": seed,
        "items": totals.items,
        "zero_attempt_items": totals.zero_attempt_items,
        "zero_aligned_word_items": totals.zero_aligned_word_items,
        "attempted_but_zero_aligned_word_items": totals.attempted_but_zero_aligned_word_items,
        "four_distinct_word_items": totals.four_distinct_word_items,
        "four_distinct_word_rate": _rate(totals.four_distinct_word_items, totals.items),
        "target_attempts": totals.target_attempts,
        "faithful_target_attempts": totals.faithful_target_attempts,
        "misplaced_target_attempts": totals.target_attempts - totals.faithful_target_attempts,
        "targeting_fidelity_rate": _rate(totals.faithful_target_attempts, totals.target_attempts),
        "targeting_miss_rate": _rate(
            totals.target_attempts - totals.faithful_target_attempts,
            totals.target_attempts,
        ),
        "all_attempts_faithful_items": totals.all_attempts_faithful_items,
        "prepared_multiple_choice_items": totals.multiple_choice_items,
        "prepared_gold_option_edited_items": totals.gold_option_edited_items,
        "prepared_pair_gold_option_edit_rate": _rate(
            totals.gold_option_edited_items, totals.multiple_choice_items
        ),
        "substitution_count": totals.operations["substitution"],
        "duplication_count": totals.operations["duplication"],
        "deletion_count": totals.operations["deletion"],
    }
    for rank in range(1, 5):
        row[f"selection_rank_{rank}_attempts"] = totals.rank_attempts[rank]
        row[f"selection_rank_{rank}_faithful_attempts"] = totals.rank_faithful[rank]
        row[f"selection_rank_{rank}_fidelity_rate"] = _rate(
            totals.rank_faithful[rank], totals.rank_attempts[rank]
        )
    return row


def _render_aggregates(
    aggregates: _Aggregates,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    summary_rows = [
        _summary_row(
            row_type="setting",
            model=cell.model,
            benchmark=cell.benchmark,
            targeting=cell.targeting,
            seed=str(cell.seed),
            totals=totals,
        )
        for cell, totals in sorted(aggregates.by_cell.items())
    ]
    summary_rows.extend(
        _summary_row(
            row_type="targeting",
            model="",
            benchmark="",
            targeting=targeting,
            seed="",
            totals=aggregates.by_targeting[targeting],
        )
        for targeting in TARGETING_CONDITIONS
        if targeting in aggregates.by_targeting
    )
    summary_rows.append(
        _summary_row(
            row_type="overall",
            model="",
            benchmark="",
            targeting="",
            seed="",
            totals=aggregates.overall,
        )
    )

    def operation_payload(totals: _Totals) -> dict[str, object]:
        return {
            "unit": "successful target attempt",
            "target_attempts": totals.target_attempts,
            "counts": {operation: totals.operations[operation] for operation in _OPERATIONS},
        }

    operations: dict[str, object] = {
        "schema_version": "targeting-fidelity-operation-counts/v1",
        "paper_sha256": PAPER_SHA256,
        "by_setting": [
            {
                "model": cell.model,
                "benchmark": cell.benchmark,
                "targeting": cell.targeting,
                "seed": cell.seed,
                **operation_payload(totals),
            }
            for cell, totals in sorted(aggregates.by_cell.items())
        ],
        "by_targeting": [
            {
                "targeting": targeting,
                **operation_payload(aggregates.by_targeting[targeting]),
            }
            for targeting in TARGETING_CONDITIONS
            if targeting in aggregates.by_targeting
        ],
        "overall": operation_payload(aggregates.overall),
    }
    return summary_rows, operations


def _normalized_setting(cell: _Cell) -> tuple[str, str]:
    return cell.model.rsplit("/", 1)[-1].lower(), cell.benchmark


def _validate_paired_sources(inputs: Sequence[_InputAudit]) -> None:
    by_setting: dict[tuple[str, str], dict[str, _InputAudit]] = {}
    for source in inputs:
        cell = source.manifest.cell
        by_setting.setdefault(_normalized_setting(cell), {})[cell.targeting] = source
    for (model, benchmark), arms in by_setting.items():
        if set(arms) != set(TARGETING_CONDITIONS):
            continue
        attribution = arms["attribution-4"]
        random_control = arms["random-4"]
        left = attribution.manifest
        right = random_control.manifest
        if (
            left.cell.model != right.cell.model
            or attribution.sample_ids_sha256 != random_control.sample_ids_sha256
            or left.record_count != right.record_count
            or left.dataset_sample_count != right.dataset_sample_count
            or left.dataset_records_sha256 != right.dataset_records_sha256
            or left.model_revision != right.model_revision
            or left.dataset_loader != right.dataset_loader
            or left.max_new_tokens != right.max_new_tokens
            or left.dataset_cohort_rule != right.dataset_cohort_rule
            or left.dataset_samples_per_subset != right.dataset_samples_per_subset
            or left.random_seed_algorithm != right.random_seed_algorithm
            or left.target_position != right.target_position
            or left.alignment != right.alignment
            or left.historical_compatibility_notes != right.historical_compatibility_notes
            or left.environment_versions != right.environment_versions
            or left.cuda != right.cuda
            or left.gpu_names != right.gpu_names
        ):
            raise TargetingFidelityAuditError(
                "paired targeting provenance or cohort mismatch for "
                f"model={model!r}, benchmark={benchmark!r}"
            )


def _paper_comparison(
    inputs: Sequence[_InputAudit],
    *,
    expected_seed: int,
) -> dict[str, object]:
    input_cells = {source.manifest.cell for source in inputs}
    observed_settings = {_normalized_setting(cell) for cell in input_cells}
    observed_cells = {(*_normalized_setting(cell), cell.targeting) for cell in input_cells}
    missing_settings = sorted(_EXPECTED_PAPER_SETTINGS - observed_settings)
    unexpected_settings = sorted(observed_settings - _EXPECTED_PAPER_SETTINGS)
    missing_cells = sorted(_EXPECTED_PAPER_CELLS - observed_cells)
    unexpected_cells = sorted(observed_cells - _EXPECTED_PAPER_CELLS)
    item_counts: Counter[str] = Counter()
    cell_count_mismatches: list[dict[str, object]] = []
    for source in inputs:
        manifest = source.manifest
        cell = manifest.cell
        item_counts[cell.targeting] += manifest.record_count
        normalized = _normalized_setting(cell)
        expected_count = _EXPECTED_PAPER_ITEM_COUNTS.get(normalized)
        if expected_count != manifest.record_count:
            cell_count_mismatches.append(
                {
                    "model": normalized[0],
                    "benchmark": normalized[1],
                    "targeting": cell.targeting,
                    "expected_items": expected_count,
                    "observed_items": manifest.record_count,
                }
            )
    targeting_by_setting: dict[tuple[str, str], set[str]] = {}
    for cell in input_cells:
        targeting_by_setting.setdefault(_normalized_setting(cell), set()).add(cell.targeting)
    checks = {
        "paper_seed_42": expected_seed == 42,
        "exact_42_setting_84_cell_grid": not missing_cells and not unexpected_cells,
        "exact_per_cell_item_counts": not cell_count_mismatches,
        "paired_targeting_arms": all(
            arms == set(TARGETING_CONDITIONS) for arms in targeting_by_setting.values()
        ),
        "paper_generation_cap_512": all(source.manifest.max_new_tokens == 512 for source in inputs),
        "paper_dataset_cohort_rule": all(
            source.manifest.dataset_cohort_rule == _DATASET_COHORT_RULE for source in inputs
        ),
        "paper_subset_caps": all(
            source.manifest.dataset_samples_per_subset
            == _expected_samples_per_subset(
                source.manifest.cell.model,
                source.manifest.cell.benchmark,
            )
            for source in inputs
        ),
        "pinned_model_revisions": all(
            source.manifest.model_revision is not None for source in inputs
        ),
        "exact_arm_item_totals": item_counts["attribution-4"] == 68660
        and item_counts["random-4"] == 68660,
    }
    full_coverage = all(checks.values())
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "status": "descriptive_only" if full_coverage else "not_comparable",
        "reason": (
            "The complete paper grid and item denominators are present, but the public-v1 "
            "origin replay is a descriptive successor to the legacy decidability metric."
            if full_coverage
            else "Failed paper-comparison preconditions: " + ", ".join(failed_checks)
        ),
        "checks": checks,
        "expected_cell_counts_source": {
            "artifact": "source_provenance.csv",
            "artifact_id": _ARCHIVAL_TARGETING_REFERENCE["artifact_id"],
            "availability": _ARCHIVAL_TARGETING_REFERENCE["availability"],
            "sha256": _ARCHIVAL_SOURCE_PROVENANCE_SHA256,
        },
        "expected_model_benchmark_settings": len(_EXPECTED_PAPER_SETTINGS),
        "observed_model_benchmark_settings": len(observed_settings),
        "expected_input_cells": len(_EXPECTED_PAPER_CELLS),
        "observed_input_cells": len(observed_cells),
        "expected_items_by_targeting": {
            "attribution-4": 68660,
            "random-4": 68660,
        },
        "observed_items_by_targeting": {
            targeting: item_counts[targeting] for targeting in TARGETING_CONDITIONS
        },
        "missing_model_benchmark_settings": [
            {"model": model, "benchmark": benchmark} for model, benchmark in missing_settings
        ],
        "unexpected_model_benchmark_settings": [
            {"model": model, "benchmark": benchmark} for model, benchmark in unexpected_settings
        ],
        "missing_input_cells": [
            {"model": model, "benchmark": benchmark, "targeting": targeting}
            for model, benchmark, targeting in missing_cells
        ],
        "unexpected_input_cells": [
            {"model": model, "benchmark": benchmark, "targeting": targeting}
            for model, benchmark, targeting in unexpected_cells
        ],
        "cell_count_mismatches": sorted(
            cell_count_mismatches,
            key=lambda row: (
                str(row["model"]),
                str(row["benchmark"]),
                str(row["targeting"]),
            ),
        ),
        "metric_compatibility": {
            "four_distinct_words": "same definition after exact-grid validation",
            "selection_rank": (
                "public selection_rank corresponds to the paper's successful apply_rank"
            ),
            "landing": (
                "not numerically identical: public-v1 classifies every attempt by origin; "
                "the paper excluded legacy-undecidable attempts"
            ),
            "gold_option": (
                "not directly comparable: prepared-pair rate lacks the paper's downstream "
                "Attribution-4 CoT-swap inclusion cohort"
            ),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _concatenate_jsonl(path: Path, sources: Iterable[Path]) -> None:
    with path.open("wb") as output_stream:
        for source in sources:
            with source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish atomically, with Linux no-replace and a portable pre-check fallback."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            status = renameat2(
                -100,  # AT_FDCWD
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,  # RENAME_NOREPLACE
            )
            if status == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(
                    error_number,
                    os.strerror(error_number),
                    destination,
                )
            if error_number not in {errno.ENOSYS, errno.EINVAL}:
                raise OSError(
                    error_number,
                    os.strerror(error_number),
                    destination,
                )

    # Windows refuses an existing rename destination. The pre-check is also a
    # best-effort fallback for platforms without Linux's renameat2 primitive.
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
    os.rename(source, destination)


def _sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(
            json.dumps(
                sample_id,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_payloads(
    rows: Iterable[Mapping[str, object]],
    *,
    path: Path,
    relative_path: str,
    cell: _Cell,
    aggregates: _Aggregates,
) -> Iterable[Mapping[str, object]]:
    for row in rows:
        item = _validate_pair(
            row,
            path=path,
            relative_path=relative_path,
            cell=cell,
        )
        aggregates.add(item)
        yield item.payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_targeting_fidelity_audit(
    config: TargetingFidelityAuditConfig,
) -> TargetingFidelityAuditResult:
    """Validate every completed input cell and atomically publish Appendix A summaries."""
    pairs_root = config.pairs_root
    output_dir = config.output_dir
    if os.path.lexists(output_dir):
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not pairs_root.exists():
        raise TargetingFidelityAuditError(f"pairs root does not exist: {pairs_root}")
    if not pairs_root.is_dir():
        raise TargetingFidelityAuditError(f"pairs root is not a directory: {pairs_root}")

    pairs_paths = sorted(
        pairs_root.rglob("pairs.jsonl"), key=lambda path: _relative(path, pairs_root)
    )
    if not pairs_paths:
        raise TargetingFidelityAuditError(f"no pairs.jsonl files found under {pairs_root}")
    resolved_sources = [path.resolve() for path in pairs_paths]
    if len(resolved_sources) != len(set(resolved_sources)):
        raise TargetingFidelityAuditError("the same pairs.jsonl was discovered more than once")

    with tempfile.TemporaryDirectory(prefix="typo-cot-targeting-fidelity-") as spool_dir:
        return _run_targeting_fidelity_audit_with_spool(
            config,
            pairs_paths=pairs_paths,
            spool_dir=Path(spool_dir),
        )


def _run_targeting_fidelity_audit_with_spool(
    config: TargetingFidelityAuditConfig,
    *,
    pairs_paths: Sequence[Path],
    spool_dir: Path,
) -> TargetingFidelityAuditResult:
    pairs_root = config.pairs_root
    output_dir = config.output_dir

    input_cells: set[_Cell] = set()
    normalized_input_cells: set[tuple[str, str, str]] = set()
    input_audits: list[_InputAudit] = []
    aggregates = _Aggregates()
    record_spools: list[tuple[_Cell, Path]] = []
    input_provenance: list[dict[str, object]] = []
    for source_index, pairs_path in enumerate(pairs_paths):
        manifest_path = pairs_path.with_name("run.json")
        if not manifest_path.is_file():
            raise TargetingFidelityAuditError(f"{pairs_path}: missing sibling completed run.json")
        manifest_audit = _validate_manifest(manifest_path, expected_seed=config.expected_seed)
        cell = manifest_audit.cell
        expected_rows = manifest_audit.record_count
        if cell in input_cells:
            raise TargetingFidelityAuditError(
                "duplicate input cell for "
                f"model={cell.model!r}, benchmark={cell.benchmark!r}, "
                f"targeting={cell.targeting!r}, seed={cell.seed}"
            )
        normalized_cell = (*_normalized_setting(cell), cell.targeting)
        if normalized_cell in normalized_input_cells:
            raise TargetingFidelityAuditError(
                "duplicate normalized input cell for "
                f"model={normalized_cell[0]!r}, benchmark={cell.benchmark!r}, "
                f"targeting={cell.targeting!r}"
            )
        input_cells.add(cell)
        normalized_input_cells.add(normalized_cell)
        relative_pairs = _relative(pairs_path, pairs_root)
        raw_rows = _load_jsonl(pairs_path)
        if len(raw_rows) != expected_rows:
            raise TargetingFidelityAuditError(
                f"{pairs_path}: pairs.jsonl has {len(raw_rows)} rows but manifest records "
                f"{expected_rows}"
            )
        source_sample_ids = [
            _string(
                raw_row.get("sample_id"),
                path=pairs_path,
                field_name=f"line {raw_row.get('__source_line__')} sample_id",
            )
            for raw_row in raw_rows
        ]
        if any(
            previous >= current
            for previous, current in zip(
                source_sample_ids,
                source_sample_ids[1:],
                strict=False,
            )
        ):
            raise TargetingFidelityAuditError(
                f"{pairs_path}: sample_id rows must remain strictly sorted"
            )
        if (
            _dataset_records_sha256(raw_rows, path=pairs_path)
            != manifest_audit.dataset_records_sha256
        ):
            raise TargetingFidelityAuditError(
                f"{pairs_path}: dataset records SHA-256 does not match reconstructed pairs"
            )
        spool_path = spool_dir / f"{source_index:04d}.jsonl"

        _write_jsonl(
            spool_path,
            _validated_payloads(
                raw_rows,
                path=pairs_path,
                relative_path=relative_pairs,
                cell=cell,
                aggregates=aggregates,
            ),
        )
        record_spools.append((cell, spool_path))
        sample_ids_sha256 = _sample_ids_sha256(source_sample_ids)
        input_audits.append(
            _InputAudit(
                manifest=manifest_audit,
                sample_ids_sha256=sample_ids_sha256,
            )
        )
        input_provenance.append(
            {
                "model": cell.model,
                "benchmark": cell.benchmark,
                "targeting": cell.targeting,
                "seed": cell.seed,
                "record_count": expected_rows,
                "sample_ids_sha256": sample_ids_sha256,
                "dataset_records_sha256": manifest_audit.dataset_records_sha256,
                "model_revision": manifest_audit.model_revision,
                "max_new_tokens": manifest_audit.max_new_tokens,
                "dataset_loader": manifest_audit.dataset_loader,
                "dataset_cohort_rule": manifest_audit.dataset_cohort_rule,
                "dataset_samples_per_subset": manifest_audit.dataset_samples_per_subset,
                "random_seed_algorithm": manifest_audit.random_seed_algorithm,
                "target_position": manifest_audit.target_position,
                "alignment": manifest_audit.alignment,
                "historical_compatibility_notes": list(
                    manifest_audit.historical_compatibility_notes
                ),
                "environment_versions": dict(manifest_audit.environment_versions),
                "cuda": manifest_audit.cuda,
                "gpu_names": list(manifest_audit.gpu_names),
                "pairs_path": relative_pairs,
                "pairs_sha256": _sha256(pairs_path),
                "manifest_path": _relative(manifest_path, pairs_root),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
        del raw_rows, source_sample_ids

    _validate_paired_sources(input_audits)
    summary_rows, operation_counts = _render_aggregates(aggregates)
    settings = len({(cell.model, cell.benchmark) for cell in input_cells})
    paper_comparison = _paper_comparison(
        input_audits,
        expected_seed=config.expected_seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        records_name = "targeting_fidelity_records.jsonl"
        summary_name = "targeting_fidelity.csv"
        operation_name = "operation_counts.json"
        run_name = "run.json"
        _concatenate_jsonl(
            temporary_dir / records_name,
            (spool_path for _, spool_path in sorted(record_spools)),
        )
        _write_csv(temporary_dir / summary_name, summary_rows)
        _write_json(temporary_dir / operation_name, operation_counts)
        output_provenance = [
            {"path": name, "sha256": _sha256(temporary_dir / name)}
            for name in (records_name, summary_name, operation_name)
        ]
        _write_json(
            temporary_dir / run_name,
            {
                "schema_version": "targeting-fidelity-audit-run/v1",
                "paper_sha256": PAPER_SHA256,
                "operation": "targeting-fidelity-audit",
                "status": "completed",
                "arguments": {
                    "pairs_root": str(pairs_root.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "expected_seed": config.expected_seed,
                },
                "counts": {
                    "input_files": len(pairs_paths),
                    "input_cells": len(input_cells),
                    "settings": settings,
                    "items": aggregates.overall.items,
                    "zero_attempt_items": aggregates.overall.zero_attempt_items,
                    "zero_aligned_word_items": aggregates.overall.zero_aligned_word_items,
                    "attempted_but_zero_aligned_word_items": (
                        aggregates.overall.attempted_but_zero_aligned_word_items
                    ),
                },
                "inputs": input_provenance,
                "outputs": output_provenance,
                "paper_reference_values": PAPER_REFERENCE_VALUES,
                "paper_comparison": paper_comparison,
                "metric_protocol": {
                    "rank_metric": "successful Attribution-4 application selection_rank",
                    "landing_metric": "public-v1 origin-based Boolean for every target attempt",
                    "prepared_gold_option_metric": (
                        "all prepared multiple-choice pairs whose final changed-word span "
                        "overlaps the formatted gold-option text"
                    ),
                },
                "completed_at": _now(),
            },
        )
        if os.path.lexists(output_dir):
            raise FileExistsError(f"output directory already exists: {output_dir}")
        _rename_directory_noreplace(temporary_dir, output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    return TargetingFidelityAuditResult(
        output_dir=output_dir,
        records_path=output_dir / records_name,
        summary_path=output_dir / summary_name,
        operation_counts_path=output_dir / operation_name,
        run_path=output_dir / run_name,
        items=aggregates.overall.items,
        settings=settings,
        input_cells=len(input_cells),
    )

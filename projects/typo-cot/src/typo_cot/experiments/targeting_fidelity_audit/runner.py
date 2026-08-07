"""Validate prepared pairs and aggregate the final paper's input-quality audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PUBLIC_BENCHMARKS,
    TARGETING_CONDITIONS,
)

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
_CSV_FIELDS = (
    "row_type",
    "model",
    "benchmark",
    "targeting",
    "seed",
    "items",
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
    "source": "Final PDF Appendix A, Targeting fidelity and Perturbation strata",
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
class _AuditedItem:
    cell: _Cell
    sample_id: str
    four_distinct_words: bool
    attempts: tuple[dict[str, object], ...]
    gold_option_applicable: bool
    gold_option_edited: bool | None
    payload: dict[str, object]


@dataclass(slots=True)
class _Totals:
    items: int = 0
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
        self.four_distinct_word_items += int(item.four_distinct_words)
        if item.gold_option_applicable:
            self.multiple_choice_items += 1
            self.gold_option_edited_items += int(item.gold_option_edited is True)
        self.all_attempts_faithful_items += int(
            all(bool(attempt["landed_on_intended_token"]) for attempt in item.attempts)
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
    except (OSError, json.JSONDecodeError, TargetingFidelityAuditError) as exc:
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


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise _error(path, f"input is outside pairs root {root.resolve()}") from exc


def _validate_manifest(
    path: Path,
    *,
    expected_seed: int,
) -> tuple[_Cell, int]:
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
        "random_seed_algorithm": "sha256-first-64-bits/v1",
        "target_position": "maximum-logit-after-first-cot-token",
        "alignment": "actual-edited-word-final-token",
    }
    for field_name, expected_value in expected_protocol.items():
        if provenance.get(field_name) != expected_value:
            raise _error(
                path,
                f"provenance.{field_name} must be {expected_value!r}",
            )

    counts = _mapping(manifest.get("counts"), path=path, field_name="counts")
    discovered = _integer(counts.get("discovered"), path=path, field_name="counts.discovered")
    written = _integer(counts.get("written"), path=path, field_name="counts.written")
    failed = _integer(counts.get("failed"), path=path, field_name="counts.failed")
    if written <= 0:
        raise _error(path, "completed input contains no pair records")
    if discovered != written or failed != 0:
        raise _error(path, "completed input has partial discovered/written/failed counts")
    return _Cell(model, benchmark, targeting, seed), written


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
    if not raw_attempts or len(raw_attempts) > 4:
        raise _error(item_path, "target_attempts must contain between one and four edits")

    current_text = editable
    origins: list[int | None] = list(range(len(editable)))
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
        target_token_index = _integer(
            attempt.get("target_token_index"),
            path=item_path,
            field_name=f"{field_prefix}.target_token_index",
        )
        if target_token_index < 0:
            raise _error(item_path, "target token indices must be non-negative")
        _string(
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
        landed_start, landed_end = _span(
            attempt.get("landed_editable_span_before"),
            path=item_path,
            field_name=f"{field_prefix}.landed_editable_span_before",
            text_length=len(current_text),
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

    edited = _mapping(record.get("edited"), path=item_path, field_name="edited")
    recorded_edited = _string(
        edited.get("editable_text"), path=item_path, field_name="edited.editable_text"
    )
    if recorded_edited != current_text:
        raise _error(item_path, "replayed edited text does not match edited.editable_text")

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
    if not raw_words or len(raw_words) > 4:
        raise _error(item_path, "aligned_words must contain between one and four words")
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
        "all_attempts_faithful": faithful_count == len(attempts),
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


def _group(items: Sequence[_AuditedItem]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_cell: dict[_Cell, _Totals] = {}
    by_targeting: dict[str, _Totals] = {}
    overall = _Totals()
    for item in items:
        by_cell.setdefault(item.cell, _Totals()).add(item)
        by_targeting.setdefault(item.cell.targeting, _Totals()).add(item)
        overall.add(item)

    summary_rows = [
        _summary_row(
            row_type="setting",
            model=cell.model,
            benchmark=cell.benchmark,
            targeting=cell.targeting,
            seed=str(cell.seed),
            totals=totals,
        )
        for cell, totals in sorted(by_cell.items())
    ]
    summary_rows.extend(
        _summary_row(
            row_type="targeting",
            model="",
            benchmark="",
            targeting=targeting,
            seed="",
            totals=by_targeting[targeting],
        )
        for targeting in TARGETING_CONDITIONS
        if targeting in by_targeting
    )
    summary_rows.append(
        _summary_row(
            row_type="overall",
            model="",
            benchmark="",
            targeting="",
            seed="",
            totals=overall,
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
            for cell, totals in sorted(by_cell.items())
        ],
        "by_targeting": [
            {
                "targeting": targeting,
                **operation_payload(by_targeting[targeting]),
            }
            for targeting in TARGETING_CONDITIONS
            if targeting in by_targeting
        ],
        "overall": operation_payload(overall),
    }
    return summary_rows, operations


def _normalized_setting(cell: _Cell) -> tuple[str, str]:
    return cell.model.rsplit("/", 1)[-1].lower(), cell.benchmark


def _paper_comparison(
    input_cells: set[_Cell],
    audited_items: Sequence[_AuditedItem],
) -> dict[str, object]:
    observed_settings = {_normalized_setting(cell) for cell in input_cells}
    observed_cells = {(*_normalized_setting(cell), cell.targeting) for cell in input_cells}
    missing_settings = sorted(_EXPECTED_PAPER_SETTINGS - observed_settings)
    unexpected_settings = sorted(observed_settings - _EXPECTED_PAPER_SETTINGS)
    missing_cells = sorted(_EXPECTED_PAPER_CELLS - observed_cells)
    unexpected_cells = sorted(observed_cells - _EXPECTED_PAPER_CELLS)
    item_counts = Counter(item.cell.targeting for item in audited_items)
    full_coverage = (
        not missing_cells
        and not unexpected_cells
        and item_counts["attribution-4"] == 68660
        and item_counts["random-4"] == 68660
    )
    return {
        "status": "descriptive_only" if full_coverage else "not_comparable",
        "reason": (
            "The complete paper grid and item denominators are present, but the public-v1 "
            "origin replay is a descriptive successor to the legacy decidability metric."
            if full_coverage
            else "Paper values require the complete 42-setting, two-condition grid and exact "
            "68,660-item denominator in each condition."
        ),
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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

    input_cells: set[_Cell] = set()
    pair_identities: set[tuple[_Cell, str]] = set()
    audited_items: list[_AuditedItem] = []
    input_provenance: list[dict[str, object]] = []
    for pairs_path in pairs_paths:
        manifest_path = pairs_path.with_name("run.json")
        if not manifest_path.is_file():
            raise TargetingFidelityAuditError(f"{pairs_path}: missing sibling completed run.json")
        cell, expected_rows = _validate_manifest(manifest_path, expected_seed=config.expected_seed)
        if cell in input_cells:
            raise TargetingFidelityAuditError(
                "duplicate input cell for "
                f"model={cell.model!r}, benchmark={cell.benchmark!r}, "
                f"targeting={cell.targeting!r}, seed={cell.seed}"
            )
        input_cells.add(cell)
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
        for raw_row in raw_rows:
            item = _validate_pair(
                raw_row,
                path=pairs_path,
                relative_path=relative_pairs,
                cell=cell,
            )
            identity = (cell, item.sample_id)
            if identity in pair_identities:
                raise TargetingFidelityAuditError(
                    f"{pairs_path}: duplicate pair identity for sample {item.sample_id!r}"
                )
            pair_identities.add(identity)
            audited_items.append(item)
        input_provenance.append(
            {
                "model": cell.model,
                "benchmark": cell.benchmark,
                "targeting": cell.targeting,
                "seed": cell.seed,
                "record_count": expected_rows,
                "pairs_path": relative_pairs,
                "pairs_sha256": _sha256(pairs_path),
                "manifest_path": _relative(manifest_path, pairs_root),
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    audited_items.sort(
        key=lambda item: (
            item.cell.model,
            item.cell.benchmark,
            item.cell.targeting,
            item.cell.seed,
            item.sample_id,
        )
    )
    summary_rows, operation_counts = _group(audited_items)
    settings = len({(item.cell.model, item.cell.benchmark) for item in audited_items})
    paper_comparison = _paper_comparison(input_cells, audited_items)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        records_name = "targeting_fidelity_records.jsonl"
        summary_name = "targeting_fidelity.csv"
        operation_name = "operation_counts.json"
        run_name = "run.json"
        _write_jsonl(temporary_dir / records_name, [item.payload for item in audited_items])
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
                    "items": len(audited_items),
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
        os.rename(temporary_dir, output_dir)
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
        items=len(audited_items),
        settings=settings,
        input_cells=len(input_cells),
    )

"""Checkpointed setting runner for the one-token clean-prefix diagnostic."""

# Malformed persisted JSON/JSONL is reported uniformly as a schema-value error.
# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.clean_prefix_scan.planning import select_extension_sample_ids
from typo_cot.experiments.one_token_prefix_replacement.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.one_token_prefix_replacement.metrics import (
    aggregate_one_token_events,
    classify_one_token_events,
)
from typo_cot.experiments.one_token_prefix_replacement.planning import (
    DistantPositionSelection,
    OneTokenArmSpec,
    OneTokenInputPlan,
    OneTokenProfile,
    build_arm_specs,
    choose_adjacent_position,
    choose_distant_positions,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    ADJACENT_SETTINGS,
    ANSWER_DECODING,
    ANSWER_EXTRACTION,
    BENCHMARK_DATASET_NAMES,
    EXTENSION_SETTINGS,
    GENERATION,
    GENERATION_INPUT,
    HISTORICAL_REFERENCE,
    IMPLEMENTATION,
    LEGACY_SETTING_IDS,
    LEGACY_TARGETING_CODES,
    POSITION_CONTROLS,
    PRE_ANSWER_BOUNDARY,
    PRIMARY_SETTING,
    PROFILE_INPUT,
    PROTOCOL,
    TOKEN_ADMISSIBILITY,
)
from typo_cot.experiments.one_token_prefix_replacement.source import (
    SourceBundle,
    SourceCase,
    load_source_bundle,
    validate_source_snapshot,
)

_TARGETING_ORDER = ("attribution-4", "random-4")
_OUTPUT_NAMES = (
    "one_token_records.jsonl",
    "pair_status_records.jsonl",
    "one_token_summary.json",
)
_CHECKPOINT_SCHEMA = "one-token-prefix-replacement-checkpoint/v1"
_RUN_SCHEMA = "one-token-prefix-replacement-run/v1"
_RECORD_SCHEMA = "one-token-prefix-replacement-record/v1"
_STATUS_SCHEMA = "one-token-prefix-replacement-pair-status/v1"
_SUMMARY_SCHEMA = "one-token-prefix-replacement-summary/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GPU_ID_PATTERN = re.compile(r"0|[1-9][0-9]*")


class OneTokenPrefixReplacementRunError(RuntimeError):
    """A run failed after its reproducible state was written."""


def _path(value: object | None) -> Path | None:
    return None if value is None else Path(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class OneTokenPrefixReplacementConfig:
    """Public setting-level command configuration."""

    model: str
    benchmark: str
    cohort: str
    fixed_window_run: Path | None
    pairs: tuple[Path, ...]
    max_pairs: int | None
    position_controls: tuple[str, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_window_run", _path(self.fixed_window_run))
        object.__setattr__(self, "pairs", tuple(Path(path) for path in self.pairs))
        object.__setattr__(self, "position_controls", tuple(self.position_controls))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty Hugging Face identifier")
        if self.benchmark not in BENCHMARK_DATASET_NAMES:
            raise ValueError("benchmark must be gsm8k, mmlu, or arc")
        if self.cohort not in {"primary", "extension"}:
            raise ValueError("cohort must be primary or extension")
        if self.position_controls not in {("distant",), ("distant", "adjacent")}:
            raise ValueError("position controls must be 'distant' or ordered 'distant adjacent'")
        if any(value not in POSITION_CONTROLS for value in self.position_controls):
            raise ValueError("unknown position control")
        if "adjacent" in self.position_controls and (
            self.cohort != "extension" or (self.model, self.benchmark) not in ADJACENT_SETTINGS
        ):
            raise ValueError("adjacent is restricted to the three prespecified extension settings")
        if self.cohort == "primary":
            if self.fixed_window_run is None or self.pairs or self.max_pairs is not None:
                raise ValueError("primary requires fixed_window_run and forbids pairs/max_pairs")
            if (self.model, self.benchmark) != PRIMARY_SETTING:
                raise ValueError("the primary cohort is exactly Gemma-3-4B/GSM8K")
        else:
            if self.fixed_window_run is not None:
                raise ValueError("extension forbids fixed_window_run")
            if len(self.pairs) != 2 or len({path.resolve() for path in self.pairs}) != 2:
                raise ValueError("extension requires two distinct prepared-pair files")
            if (
                not isinstance(self.max_pairs, int)
                or isinstance(self.max_pairs, bool)
                or self.max_pairs <= 0
            ):
                raise ValueError("extension max_pairs must be a positive integer")
            if (self.model, self.benchmark) not in EXTENSION_SETTINGS:
                raise ValueError("extension setting is outside the fourteen final-PDF cells")
            if self.max_pairs != 150:
                raise ValueError("final-PDF extension runs require max_pairs=150")
        if not isinstance(self.gpu_id, str) or _GPU_ID_PATTERN.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer or null")
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")
        source_paths = (
            () if self.fixed_window_run is None else (self.fixed_window_run,)
        ) + self.pairs
        output = self.output_dir.resolve()
        for source_path in source_paths:
            source = source_path.resolve()
            source_root = source if source_path.suffix == "" else source.parent
            if (
                output == source_root
                or output in source_root.parents
                or source_root in output.parents
            ):
                raise ValueError("output_dir must be separate from every source run")

    @property
    def adjacent_requested(self) -> bool:
        return "adjacent" in self.position_controls

    @property
    def setting_id(self) -> str:
        try:
            return LEGACY_SETTING_IDS[(self.model, self.benchmark)]
        except KeyError as exc:
            raise ValueError("setting has no submitted-producer identity") from exc

    def adjacent_tie_key(self, *, targeting: str, sample_id: str) -> str:
        """Return the exact submitted-producer identity encoding."""

        try:
            targeting_code = LEGACY_TARGETING_CODES[targeting]
        except KeyError as exc:
            raise ValueError(f"unknown adjacent targeting condition {targeting!r}") from exc
        return f"{self.setting_id}|{targeting_code}|{sample_id}"

    def public_arguments(self) -> dict[str, object]:
        """Stable argument representation, excluding transport-only resume."""

        payload = asdict(self)
        payload["fixed_window_run"] = (
            None if self.fixed_window_run is None else str(self.fixed_window_run.resolve())
        )
        payload["pairs"] = [str(path.resolve()) for path in self.pairs]
        payload["position_controls"] = list(self.position_controls)
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class OneTokenGeneration:
    """One generated continuation and its canonical gold-answer readout."""

    token_ids: tuple[int, ...]
    text: str
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str
    stop_reason: Literal["eos_token", "max_new_tokens"]
    stop_token_id: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("generation token_ids must be non-negative integers")
        for field in ("text", "value", "method", "primary_method"):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"generation {field} must be a string")
        for field in ("is_extracted", "is_correct"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"generation {field} must be boolean")
        if self.is_extracted is not bool(self.value):
            raise ValueError("generation is_extracted must agree with its value")
        if self.stop_reason not in {"eos_token", "max_new_tokens"}:
            raise ValueError("unknown generation stop reason")
        max_new_tokens = int(GENERATION["max_new_tokens"])
        if len(self.token_ids) > max_new_tokens:
            raise ValueError(f"generation cannot exceed {max_new_tokens} new tokens")
        if self.stop_reason == "eos_token":
            if (
                not isinstance(self.stop_token_id, int)
                or isinstance(self.stop_token_id, bool)
                or self.stop_token_id < 0
                or not self.token_ids
                or self.token_ids[-1] != self.stop_token_id
            ):
                raise ValueError("EOS generation must end in its stop token")
        elif self.stop_token_id is not None:
            raise ValueError("max-token generation must not claim an EOS stop token")
        elif len(self.token_ids) != max_new_tokens:
            raise ValueError(f"max_new_tokens termination requires exactly {max_new_tokens} tokens")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "text": self.text,
            "value": self.value,
            "is_extracted": self.is_extracted,
            "is_correct": self.is_correct,
            "method": self.method,
            "primary_method": self.primary_method,
            "stop_reason": self.stop_reason,
            "stop_token_id": self.stop_token_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OneTokenGeneration:
        expected = {
            "token_ids",
            "text",
            "value",
            "is_extracted",
            "is_correct",
            "method",
            "primary_method",
            "stop_reason",
            "stop_token_id",
        }
        if set(value) != expected:
            raise ValueError("generation has an unexpected shape")
        return cls(
            token_ids=tuple(value["token_ids"]),  # type: ignore[arg-type]
            text=value["text"],  # type: ignore[arg-type]
            value=value["value"],  # type: ignore[arg-type]
            is_extracted=value["is_extracted"],  # type: ignore[arg-type]
            is_correct=value["is_correct"],  # type: ignore[arg-type]
            method=value["method"],  # type: ignore[arg-type]
            primary_method=value["primary_method"],  # type: ignore[arg-type]
            stop_reason=value["stop_reason"],  # type: ignore[arg-type]
            stop_token_id=value["stop_token_id"],  # type: ignore[arg-type]
        )


class OneTokenPrefixReplacementRuntime(Protocol):
    """Injectable runtime boundary used by CPU tests and the GPU implementation."""

    def provenance(self) -> Mapping[str, object]: ...

    def prepare_pair(self, pair: Mapping[str, object]) -> OneTokenInputPlan: ...

    def profile_pair(self, plan: OneTokenInputPlan) -> OneTokenProfile: ...

    def generate_arm(
        self,
        plan: OneTokenInputPlan,
        *,
        position: int,
        forced_token_id: int,
        gold_answer: str,
    ) -> OneTokenGeneration: ...


@dataclass(frozen=True, slots=True)
class OneTokenPrefixReplacementResult:
    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    source_pairs: int
    selected_pairs: int
    executed_pairs: int
    records: int


@dataclass(frozen=True, slots=True)
class _PlannedCase:
    source: SourceCase
    input_plan: OneTokenInputPlan | None
    cot_token_count: int | None
    candidate_eligible: bool
    boundary_valid: bool | None
    exclusion_reason: str | None

    @property
    def key(self) -> tuple[str, str]:
        return self.source.key


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _serialized(payload: object) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_serialized(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            try:
                value = json.loads(
                    raw,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                )
            except ValueError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _checkpoint_identity(case: SourceCase) -> str:
    return hashlib.sha256(
        f"one-token-prefix-replacement\0{case.targeting}\0{case.sample_id}".encode()
    ).hexdigest()


def _checkpoint_path(directory: Path, case: SourceCase) -> Path:
    return directory / f"{_checkpoint_identity(case)}.json"


def _source_exclusion(case: SourceCase) -> str | None:
    if not case.prepared_clean_correct:
        return "clean-answer-not-correct-after-reextraction"
    if not case.prepared_edited_wrong:
        return "edited-answer-not-wrong-after-reextraction"
    if case.aligned_word_count <= 0:
        return "no-aligned-edited-word"
    return None


def _prepare_cases(
    source: SourceBundle,
    runtime: OneTokenPrefixReplacementRuntime,
) -> tuple[_PlannedCase, ...]:
    from typo_cot.experiments.one_token_prefix_replacement.runtime import (
        OneTokenBoundaryInvalid,
    )

    planned: list[_PlannedCase] = []
    for case in source.cases:
        exclusion = _source_exclusion(case)
        if exclusion is not None:
            planned.append(_PlannedCase(case, None, None, False, None, exclusion))
            continue
        try:
            plan = runtime.prepare_pair(case.record)
            if not isinstance(plan, OneTokenInputPlan):
                raise TypeError("runtime.prepare_pair did not return OneTokenInputPlan")
        except OneTokenBoundaryInvalid as exc:
            planned.append(
                _PlannedCase(
                    case,
                    None,
                    exc.cot_token_count,
                    exc.eligible_length,
                    False,
                    (
                        "prompt-boundary-invalid"
                        if exc.eligible_length
                        else "clean-pre-answer-length-outside-8-through-512"
                    ),
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            planned.append(_PlannedCase(case, None, None, False, None, f"token-plan-invalid:{exc}"))
            continue
        eligible = 8 <= plan.cot_token_count <= 512
        planned.append(
            _PlannedCase(
                case,
                plan,
                plan.cot_token_count,
                eligible,
                True,
                None if eligible else "clean-pre-answer-length-outside-8-through-512",
            )
        )
    return tuple(planned)


def _select_cases(
    planned: Sequence[_PlannedCase],
    config: OneTokenPrefixReplacementConfig,
) -> tuple[tuple[_PlannedCase, ...], tuple[_PlannedCase, ...]]:
    by_key = {case.key: case for case in planned}
    if config.cohort == "primary":
        full = tuple(planned)
    else:
        eligible = {targeting: [] for targeting in _TARGETING_ORDER}
        for case in planned:
            if case.candidate_eligible:
                eligible[case.source.targeting].append(case.source.sample_id)
        keys = select_extension_sample_ids(
            eligible,
            max_pairs=int(config.max_pairs),
            cap_per_targeting=400,
        )
        full = tuple(by_key[key] for key in keys)
    if not full:
        raise ValueError("one-token source contains no selectable targets")
    selected = full if config.limit is None else full[: config.limit]
    if not selected:
        raise ValueError("one-token execution selection is empty")
    return full, selected


def _plan_row(case: _PlannedCase) -> dict[str, object]:
    plan = None if case.input_plan is None else case.input_plan.to_dict()
    return {
        **case.source.identity_dict(),
        "candidate_eligible": case.candidate_eligible,
        "boundary_valid": case.boundary_valid,
        "cot_token_count": case.cot_token_count,
        "exclusion_reason": case.exclusion_reason,
        "input_plan_sha256": None if plan is None else _canonical_sha256(plan),
    }


def _plan_payload(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
) -> dict[str, object]:
    rows = [_plan_row(case) for case in planned]
    full = [list(case.key) for case in selected_full]
    execution = [list(case.key) for case in selected]
    return {
        "algorithm": "shared-clean-prefix-cohort-selection-before-limit/v1",
        "cases": rows,
        "cases_sha256": _canonical_sha256(rows),
        "source_case_count": len(planned),
        "eligible_case_count": sum(case.candidate_eligible for case in planned),
        "selected_full": full,
        "selected_full_sha256": _canonical_sha256(full),
        "selected_for_execution": execution,
        "selected_for_execution_sha256": _canonical_sha256(execution),
    }


def _validate_runtime_provenance(
    value: Mapping[str, object],
    *,
    source: SourceBundle,
    gpu_id: str,
) -> dict[str, object]:
    runtime = dict(value)
    expected_fields = {
        "runtime",
        "python",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "requested_revision",
        "model_revision",
        "tokenizer_revision",
        "dtype",
        "device",
        "cuda",
        "gpu_name",
        "cuda_visible_devices",
        "generation",
        "effective_eos_token_ids",
        "effective_eos_token_ids_source",
        "answer_decoding",
        "answer_extraction",
        "pre_answer_boundary",
        "profile_input",
        "generation_input",
        "token_admissibility",
        "implementation_code_identity",
    }
    if set(runtime) != expected_fields or runtime.get("runtime") != IMPLEMENTATION:
        raise ValueError("runtime provenance has an unexpected shape or implementation")
    for field in (
        "python",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "cuda",
        "gpu_name",
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise ValueError(f"runtime {field} must be a non-empty string")
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if runtime.get(field) != source.model_revision:
            raise ValueError(f"runtime {field} does not match the source revision")
    if runtime.get("dtype") != "bfloat16" or runtime.get("device") != "cuda:0":
        raise ValueError("runtime must use bfloat16 on logical cuda:0")
    if runtime.get("cuda_visible_devices") != gpu_id:
        raise ValueError("runtime CUDA visibility does not match the requested physical GPU")
    generation = _mapping(runtime.get("generation"), field="runtime generation")
    if set(generation) != set(GENERATION) or any(
        generation.get(field) != expected or type(generation.get(field)) is not type(expected)
        for field, expected in GENERATION.items()
    ):
        raise ValueError("runtime generation protocol does not match")
    for field, expected in {
        "answer_decoding": ANSWER_DECODING,
        "answer_extraction": ANSWER_EXTRACTION,
        "pre_answer_boundary": PRE_ANSWER_BOUNDARY,
        "profile_input": PROFILE_INPUT,
        "generation_input": GENERATION_INPUT,
    }.items():
        if runtime.get(field) != expected:
            raise ValueError(f"runtime {field} does not match the protocol")
    eos = runtime.get("effective_eos_token_ids")
    if (
        not isinstance(eos, list)
        or not eos
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos)
        or eos != sorted(set(eos))
    ):
        raise ValueError("runtime effective_eos_token_ids are invalid")
    if runtime.get("effective_eos_token_ids_source") not in {
        "model-generation-config",
        "tokenizer-fallback",
    }:
        raise ValueError("runtime effective EOS source is invalid")
    token_pool = _mapping(runtime.get("token_admissibility"), field="runtime token admissibility")
    expected_pool_fields = {
        "implementation",
        "model_logit_size",
        "tokenizer_vocab_entries",
        "n_real_tokenizer_ids_in_logits",
        "n_special_ids",
        "n_marker_ids",
        "n_admissible_ids",
        "admissible_token_ids_sha256_algorithm",
        "admissible_token_ids_sha256",
        "marker_regex",
    }
    if (
        set(token_pool) != expected_pool_fields
        or token_pool.get("implementation") != TOKEN_ADMISSIBILITY["implementation"]
        or token_pool.get("marker_regex") != TOKEN_ADMISSIBILITY["marker_regex"]
        or token_pool.get("admissible_token_ids_sha256_algorithm")
        != TOKEN_ADMISSIBILITY["admissible_token_ids_sha256_algorithm"]
        or not isinstance(token_pool.get("admissible_token_ids_sha256"), str)
        or _SHA256_PATTERN.fullmatch(str(token_pool["admissible_token_ids_sha256"])) is None
    ):
        raise ValueError("runtime token admissibility protocol does not match")
    count_fields = expected_pool_fields - {
        "implementation",
        "marker_regex",
        "admissible_token_ids_sha256_algorithm",
        "admissible_token_ids_sha256",
    }
    if any(
        not isinstance(token_pool.get(field), int)
        or isinstance(token_pool[field], bool)
        or int(token_pool[field]) < 0
        for field in count_fields
    ):
        raise ValueError("runtime token admissibility counts are invalid")
    if (
        int(token_pool["model_logit_size"]) <= 0
        or int(token_pool["tokenizer_vocab_entries"]) <= 0
        or int(token_pool["n_real_tokenizer_ids_in_logits"])
        > int(token_pool["tokenizer_vocab_entries"])
        or int(token_pool["n_admissible_ids"]) > int(token_pool["n_real_tokenizer_ids_in_logits"])
    ):
        raise ValueError("runtime token admissibility counts are inconsistent")
    if runtime.get("implementation_code_identity") != implementation_code_identity():
        raise ValueError(
            "runtime executable-code identity does not match the installed source tree"
        )
    return runtime


def _profile_from_dict(value: Mapping[str, object]) -> OneTokenProfile:
    expected = {
        "clean_to_edited_kl",
        "clean_token_rank_under_clean",
        "clean_token_rank_under_edited",
        "edited_top1_ids",
        "edited_top1_is_admissible",
    }
    if set(value) != expected:
        raise ValueError("one-token profile has an unexpected shape")
    return OneTokenProfile(
        clean_to_edited_kl=tuple(value["clean_to_edited_kl"]),  # type: ignore[arg-type]
        clean_token_rank_under_clean=tuple(value["clean_token_rank_under_clean"]),  # type: ignore[arg-type]
        clean_token_rank_under_edited=tuple(value["clean_token_rank_under_edited"]),  # type: ignore[arg-type]
        edited_top1_ids=tuple(value["edited_top1_ids"]),  # type: ignore[arg-type]
        edited_top1_is_admissible=tuple(value["edited_top1_is_admissible"]),  # type: ignore[arg-type]
    )


def _input_plan_from_dict(value: Mapping[str, object]) -> OneTokenInputPlan:
    expected = {
        "clean_prompt_ids",
        "edited_prompt_ids",
        "clean_full_ids",
        "edited_full_ids",
        "clean_cot_ids",
    }
    if set(value) != expected:
        raise ValueError("one-token input plan has an unexpected shape")
    return OneTokenInputPlan(
        clean_prompt_ids=tuple(value["clean_prompt_ids"]),  # type: ignore[arg-type]
        edited_prompt_ids=tuple(value["edited_prompt_ids"]),  # type: ignore[arg-type]
        clean_full_ids=tuple(value["clean_full_ids"]),  # type: ignore[arg-type]
        edited_full_ids=tuple(value["edited_full_ids"]),  # type: ignore[arg-type]
        clean_cot_ids=tuple(value["clean_cot_ids"]),  # type: ignore[arg-type]
    )


def _positions_payload(
    selection: DistantPositionSelection,
    *,
    adjacent_position: int | None,
    adjacent_requested: bool,
) -> dict[str, object]:
    return {
        "selected": selection.selected_position,
        "distant": selection.distant_position,
        "adjacent": adjacent_position if adjacent_requested else None,
    }


def _arm_row(
    spec: OneTokenArmSpec,
    generation: OneTokenGeneration,
    plan: OneTokenInputPlan,
) -> dict[str, object]:
    return {
        **spec.to_dict(),
        "input_ids": list(plan.generation_input_ids(spec.position, spec.forced_token_id)),
        "generation": generation.to_dict(),
    }


def _validate_generation(
    generation: OneTokenGeneration,
    *,
    benchmark: str,
    gold_answer: str,
    eos_ids: Sequence[int],
) -> None:
    ended = generation.stop_reason == "eos_token"
    if ended and generation.stop_token_id not in eos_ids:
        raise ValueError("generation EOS stop token is outside runtime provenance")
    effective_eos = frozenset(eos_ids)
    if ended and any(token in effective_eos for token in generation.token_ids[:-1]):
        raise ValueError("generation continued after an effective EOS token")
    if not ended and any(token in effective_eos for token in generation.token_ids):
        raise ValueError("max-token generation contains an effective EOS token")
    extraction = extract_with_fallback(
        generation.text,
        benchmark=BENCHMARK_DATASET_NAMES[benchmark],
        correct_answer=gold_answer,
        allow_positional=ended,
    )
    if (
        generation.value != extraction.value
        or generation.is_extracted is not extraction.is_extracted
        or generation.is_correct is not extraction.is_correct
        or generation.method != extraction.method
        or generation.primary_method != extraction.primary_method
    ):
        raise ValueError("generation extraction or correctness is inconsistent")


def _checkpoint_base(
    case: _PlannedCase,
    *,
    profile: OneTokenProfile,
    selection: DistantPositionSelection,
    adjacent_position: int | None,
    adjacent_requested: bool,
    arm_specs: Sequence[OneTokenArmSpec],
) -> dict[str, object]:
    if case.input_plan is None:
        raise ValueError("checkpoint case has no input plan")
    plan = case.input_plan.to_dict()
    profile_row = profile.to_dict()
    specs = [spec.to_dict() for spec in arm_specs]
    return {
        "schema_version": _CHECKPOINT_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "source": case.source.identity_dict(),
        "input_plan": plan,
        "input_plan_sha256": _canonical_sha256(plan),
        "profile": profile_row,
        "profile_sha256": _canonical_sha256(profile_row),
        "selection": selection.to_dict(),
        "positions": _positions_payload(
            selection,
            adjacent_position=adjacent_position,
            adjacent_requested=adjacent_requested,
        ),
        "adjacent_requested": adjacent_requested,
        "adjacent_unavailable_reason": (
            "no-nearest-strictly-lower-kl-position"
            if adjacent_requested and adjacent_position is None
            else None
        ),
        "arm_specs": specs,
        "arm_specs_sha256": _canonical_sha256(specs),
        "arms": [],
    }


def _position_exclusion_checkpoint(
    case: _PlannedCase,
    *,
    profile: OneTokenProfile,
    config: OneTokenPrefixReplacementConfig,
) -> dict[str, object]:
    """Build a self-validating checkpoint for a paper-defined position exclusion."""

    if case.input_plan is None:
        raise ValueError("position-exclusion case has no input plan")
    selection: DistantPositionSelection | None = None
    try:
        selection = choose_distant_positions(profile)
    except ValueError as exc:
        reason = str(exc)
    else:
        if selection.distant_position is not None:
            raise ValueError("profile is executable and cannot be position-excluded")
        reason = "no-distant-lower-median-control"
    plan = case.input_plan.to_dict()
    profile_row = profile.to_dict()
    positions = {
        "selected": None if selection is None else selection.selected_position,
        "distant": None,
        "adjacent": None,
    }
    return {
        "schema_version": _CHECKPOINT_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "source": case.source.identity_dict(),
        "input_plan": plan,
        "input_plan_sha256": _canonical_sha256(plan),
        "profile": profile_row,
        "profile_sha256": _canonical_sha256(profile_row),
        "selection": None if selection is None else selection.to_dict(),
        "positions": positions,
        "adjacent_requested": config.adjacent_requested,
        "adjacent_unavailable_reason": (
            "case-excluded-before-adjacent-position-selection"
            if config.adjacent_requested
            else None
        ),
        "position_exclusion_reason": reason,
        "arm_specs": [],
        "arm_specs_sha256": _canonical_sha256([]),
        "arms": [],
    }


def _validate_position_exclusion_checkpoint(
    value: Mapping[str, object],
    *,
    case: _PlannedCase,
    config: OneTokenPrefixReplacementConfig,
) -> None:
    """Reconstruct an exclusion from its bound input plan and profile."""

    if case.input_plan is None:
        raise ValueError("position-exclusion source case has no input plan")
    profile_row = _mapping(value.get("profile"), field="exclusion checkpoint profile")
    if value.get("profile_sha256") != _canonical_sha256(profile_row):
        raise ValueError("exclusion checkpoint profile fingerprint does not match")
    profile = _profile_from_dict(profile_row)
    if len(profile.clean_to_edited_kl) != case.input_plan.cot_token_count:
        raise ValueError("exclusion checkpoint profile length does not match")
    expected = _position_exclusion_checkpoint(case, profile=profile, config=config)
    if dict(value) != expected:
        raise ValueError("position-exclusion checkpoint does not reconstruct")


def _validate_checkpoint(
    value: Mapping[str, object],
    *,
    case: _PlannedCase,
    config: OneTokenPrefixReplacementConfig,
    eos_ids: Sequence[int],
) -> tuple[OneTokenProfile, DistantPositionSelection, int | None, tuple[OneTokenArmSpec, ...]]:
    if case.input_plan is None:
        raise ValueError("checkpoint source case has no input plan")
    expected_fields = {
        "schema_version",
        "paper_sha256",
        "operation",
        "source",
        "input_plan",
        "input_plan_sha256",
        "profile",
        "profile_sha256",
        "selection",
        "positions",
        "adjacent_requested",
        "adjacent_unavailable_reason",
        "arm_specs",
        "arm_specs_sha256",
        "arms",
    }
    if set(value) != expected_fields:
        raise ValueError("checkpoint has an unexpected shape")
    if (
        value.get("schema_version") != _CHECKPOINT_SCHEMA
        or value.get("paper_sha256") != PAPER_SHA256
        or value.get("operation") != "one-token-prefix-replacement"
        or value.get("source") != case.source.identity_dict()
    ):
        raise ValueError("checkpoint identity does not match")
    plan = case.input_plan.to_dict()
    if value.get("input_plan") != plan or value.get("input_plan_sha256") != _canonical_sha256(plan):
        raise ValueError("checkpoint input plan does not match")
    profile_row = _mapping(value.get("profile"), field="checkpoint profile")
    if value.get("profile_sha256") != _canonical_sha256(profile_row):
        raise ValueError("checkpoint profile fingerprint does not match")
    profile = _profile_from_dict(profile_row)
    if len(profile.clean_to_edited_kl) != case.input_plan.cot_token_count:
        raise ValueError("checkpoint profile length does not match the input plan")
    selection = choose_distant_positions(profile)
    if selection.distant_position is None:
        raise ValueError("checkpoint claims an executable case without a distant position")
    if value.get("selection") != selection.to_dict():
        raise ValueError("checkpoint position selection does not reconstruct")
    adjacent_position = None
    if config.adjacent_requested:
        adjacent_position = choose_adjacent_position(
            profile.clean_to_edited_kl,
            selected_position=selection.selected_position,
            tie_key=config.adjacent_tie_key(
                targeting=case.source.targeting,
                sample_id=case.source.sample_id,
            ),
        )
    if value.get("positions") != _positions_payload(
        selection,
        adjacent_position=adjacent_position,
        adjacent_requested=config.adjacent_requested,
    ):
        raise ValueError("checkpoint positions do not reconstruct")
    expected_adjacent_reason = (
        "no-nearest-strictly-lower-kl-position"
        if config.adjacent_requested and adjacent_position is None
        else None
    )
    if (
        value.get("adjacent_requested") is not config.adjacent_requested
        or value.get("adjacent_unavailable_reason") != expected_adjacent_reason
    ):
        raise ValueError("checkpoint adjacent availability does not reconstruct")
    specs = build_arm_specs(
        case.input_plan,
        profile,
        selection,
        adjacent_position=adjacent_position,
    )
    spec_rows = [spec.to_dict() for spec in specs]
    if value.get("arm_specs") != spec_rows or value.get("arm_specs_sha256") != _canonical_sha256(
        spec_rows
    ):
        raise ValueError("checkpoint arm plan does not reconstruct")
    raw_arms = value.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) > len(specs):
        raise ValueError("checkpoint arms are invalid")
    gold = case.source.record.get("gold_answer")
    if not isinstance(gold, str) or not gold:
        raise ValueError("source case has no gold answer")
    for row, spec in zip(raw_arms, specs, strict=False):
        if not isinstance(row, Mapping):
            raise ValueError("checkpoint arm is not an object")
        generation = OneTokenGeneration.from_dict(
            _mapping(row.get("generation"), field="checkpoint arm generation")
        )
        expected = _arm_row(spec, generation, case.input_plan)
        if dict(row) != expected:
            raise ValueError("checkpoint arm semantics do not reconstruct")
        _validate_generation(
            generation,
            benchmark=config.benchmark,
            gold_answer=gold,
            eos_ids=eos_ids,
        )
    return profile, selection, adjacent_position, specs


def _record_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    case: _PlannedCase,
    source: SourceBundle,
    config: OneTokenPrefixReplacementConfig,
) -> dict[str, object]:
    arms = checkpoint.get("arms")
    specs = checkpoint.get("arm_specs")
    if not isinstance(arms, list) or not isinstance(specs, list) or len(arms) != len(specs):
        raise ValueError("completed checkpoint does not contain every arm")
    by_name = {str(_mapping(row, field="completed arm").get("name")): row for row in arms}
    positions = _mapping(checkpoint.get("positions"), field="checkpoint positions")
    adjacent_available = config.adjacent_requested and positions.get("adjacent") is not None
    events = classify_one_token_events(
        by_name,
        selected_before_control=int(positions["selected"]) < int(positions["distant"]),
        adjacent_requested=adjacent_available,
    )
    return {
        "schema_version": _RECORD_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "model": config.model,
        "benchmark": config.benchmark,
        "cohort": config.cohort,
        "targeting": case.source.targeting,
        "sample_id": case.source.sample_id,
        "gold_answer": case.source.record.get("gold_answer"),
        "source": {
            "cohort_sha256": source.cohort_sha256,
            "source_record_sha256": case.source.source_record_sha256,
        },
        "input_plan": checkpoint["input_plan"],
        "input_plan_sha256": checkpoint["input_plan_sha256"],
        "profile": checkpoint["profile"],
        "profile_sha256": checkpoint["profile_sha256"],
        "selection": checkpoint["selection"],
        "positions": dict(positions),
        "adjacent_unavailable_reason": checkpoint["adjacent_unavailable_reason"],
        "arms": arms,
        "events": events,
    }


def _status_rows(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    checkpoints: Mapping[tuple[str, str], Mapping[str, object]],
    records: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    config: OneTokenPrefixReplacementConfig,
) -> list[dict[str, object]]:
    full_keys = {case.key for case in selected_full}
    selected_keys = {case.key for case in selected}
    rows: list[dict[str, object]] = []
    for case in planned:
        checkpoint = checkpoints.get(case.key)
        record = records.get(case.key)
        positions = None if checkpoint is None else checkpoint.get("positions")
        if not case.candidate_eligible:
            execution_status = "not-candidate"
        elif case.key not in full_keys:
            execution_status = "not-selected"
        elif case.key not in selected_keys:
            execution_status = "not-executed-limit"
        elif case.boundary_valid is False:
            execution_status = "invalid-boundary"
        elif record is not None:
            execution_status = "completed"
        elif checkpoint is not None and checkpoint.get("position_exclusion_reason"):
            execution_status = "position-unavailable"
        else:
            execution_status = "failed"
        rows.append(
            {
                "schema_version": _STATUS_SCHEMA,
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "cohort": config.cohort,
                "targeting": case.source.targeting,
                "sample_id": case.source.sample_id,
                "source_record_sha256": case.source.source_record_sha256,
                "candidate_eligible": case.candidate_eligible,
                "boundary_valid": case.boundary_valid,
                "selected_full": case.key in full_keys,
                "selected_for_execution": case.key in selected_keys,
                "positions": positions,
                "adjacent_requested": config.adjacent_requested,
                "adjacent_available": (
                    isinstance(positions, Mapping) and positions.get("adjacent") is not None
                ),
                "adjacent_unavailable_reason": (
                    checkpoint.get("adjacent_unavailable_reason")
                    if checkpoint is not None
                    else None
                ),
                "execution_status": execution_status,
                "exclusion_reason": (
                    case.exclusion_reason
                    if case.exclusion_reason is not None
                    else (
                        checkpoint.get("position_exclusion_reason")
                        if checkpoint is not None
                        else None
                    )
                ),
                "position_exclusion_evidence": (
                    dict(checkpoint)
                    if checkpoint is not None and checkpoint.get("position_exclusion_reason")
                    else None
                ),
                "record_sha256": (None if record is None else _canonical_sha256(record)),
            }
        )
    return rows


def _comparability(
    config: OneTokenPrefixReplacementConfig,
    *,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Label complete paper-shaped runs separately from partial executions."""

    raw_selected = plan.get("selected_full")
    raw_cases = plan.get("cases")
    if not isinstance(raw_selected, list) or not isinstance(raw_cases, list):
        raise ValueError("comparability requires a complete selection plan")
    selected_keys: set[tuple[str, str]] = set()
    for value in raw_selected:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(part, str) or not part for part in value)
        ):
            raise ValueError("comparability selected identity is invalid")
        selected_keys.add((value[0], value[1]))
    selected_boundary_valid_count = 0
    for raw_case in raw_cases:
        case = _mapping(raw_case, field="comparability plan case")
        key = (case.get("targeting"), case.get("sample_id"))
        if key in selected_keys and case.get("boundary_valid") is True:
            selected_boundary_valid_count += 1
    selected_full_count = len(raw_selected)
    expected_target_count = 172 if config.cohort == "primary" else 150
    expected_controls = (
        ("distant", "adjacent")
        if (config.model, config.benchmark) in ADJACENT_SETTINGS
        else ("distant",)
    )
    limitations: list[str] = []
    if selected_full_count != expected_target_count:
        limitations.append("selected-target-count-differs-from-paper")
    if config.position_controls != expected_controls:
        limitations.append("prespecified-adjacent-control-not-requested")
    if selected_boundary_valid_count != selected_full_count:
        limitations.append("selected-target-has-invalid-exact-boundary")
    if config.limit is not None:
        limitations.append("limit-is-smoke-only")
    if config.limit is not None:
        status = "partial-smoke-run"
    elif limitations:
        status = "partial-paper-protocol"
    else:
        status = "fresh-paper-protocol-run"
    return {
        "status": status,
        "requirements": {
            "paper_setting": True,
            "paper_source_protocol": True,
            "paper_source_cohort_identity": False,
            "expected_selected_target_count": expected_target_count,
            "selected_target_count_matches": selected_full_count == expected_target_count,
            "selected_exact_boundary_valid_count": selected_boundary_valid_count,
            "selected_exact_boundaries_all_valid": (
                selected_boundary_valid_count == selected_full_count
            ),
            "prespecified_position_controls": config.position_controls == expected_controls,
            "unlimited": config.limit is None,
        },
        "limitations": limitations,
        "primary_in_fourteen_extension_aggregate": False,
        "fresh_public_preparation_is_historical_identity_proof": False,
        "single_setting_runner_computes_cross_setting_interval": False,
    }


def _build_summary(
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
    plan: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    statuses: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    event_rows = [_mapping(record.get("events"), field="record events") for record in records]
    metrics = aggregate_one_token_events(
        event_rows,
        adjacent_requested=config.adjacent_requested,
    )
    stop_reasons: Counter[str] = Counter()
    unextractable = 0
    arm_count = 0
    for record in records:
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise ValueError("record arms must be a list")
        for raw_arm in arms:
            arm = _mapping(raw_arm, field="summary arm")
            generation = _mapping(arm.get("generation"), field="summary generation")
            stop_reasons[str(generation.get("stop_reason"))] += 1
            unextractable += generation.get("is_extracted") is False
            arm_count += 1
    status_counts = Counter(str(row.get("execution_status")) for row in statuses)
    return {
        "schema_version": _SUMMARY_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "cohort": config.cohort,
            "position_controls": list(config.position_controls),
        },
        "protocol": PROTOCOL,
        "source": source.to_dict(),
        "counts": {
            "source_pairs": len(statuses),
            "candidate_eligible": sum(row.get("candidate_eligible") is True for row in statuses),
            "selected_full": len(plan["selected_full"]),  # type: ignore[arg-type]
            "selected_for_execution": len(plan["selected_for_execution"]),  # type: ignore[arg-type]
            "executed": len(records),
            "records": len(records),
            "arms": arm_count,
            "execution_status": dict(sorted(status_counts.items())),
        },
        "metrics": metrics,
        "generation_diagnostics": {
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "unextractable": unextractable,
        },
        "historical_reference": HISTORICAL_REFERENCE,
        "comparability": _comparability(
            config,
            plan=plan,
        ),
    }


def _manifest(
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
    plan: Mapping[str, object],
    status: str,
    started_at: str,
    runtime: Mapping[str, object] | None,
    checkpoints: Mapping[str, object],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": _RUN_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": PROTOCOL,
        "protocol_sha256": _canonical_sha256(PROTOCOL),
        "comparability": _comparability(
            config,
            plan=plan,
        ),
        "source": source.to_dict(),
        "plan": dict(plan),
        "runtime": None if runtime is None else dict(runtime),
        "checkpoints": dict(checkpoints),
        "counts": dict(counts),
        "failures": list(failures),
        "outputs": None if outputs is None else dict(outputs),
        "started_at": started_at,
        "updated_at": _now(),
    }


def _validate_resume_core(
    previous: Mapping[str, object],
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
) -> None:
    if (
        previous.get("schema_version") != _RUN_SCHEMA
        or previous.get("paper_sha256") != PAPER_SHA256
        or previous.get("operation") != "one-token-prefix-replacement"
    ):
        raise ValueError("cannot resume an unknown one-token run")
    if previous.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run")
    if previous.get("protocol") != PROTOCOL or previous.get("protocol_sha256") != _canonical_sha256(
        PROTOCOL
    ):
        raise ValueError("resume protocol fingerprint does not match")
    previous_runtime = _mapping(previous.get("runtime"), field="resume runtime")
    if previous_runtime.get("implementation_code_identity") != implementation_code_identity():
        raise ValueError("resume executable-code identity does not match the installed source tree")
    plan = _mapping(previous.get("plan"), field="resume plan")
    if previous.get("comparability") != _comparability(config, plan=plan):
        raise ValueError("resume comparability label does not match")
    if previous.get("source") != source.to_dict():
        raise ValueError("resume source identity does not match")


def _plan_key(value: object, *, field: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(part, str) or not part for part in value)
    ):
        raise ValueError(f"{field} must be a targeting/sample-id pair")
    return value[0], value[1]


def _validate_completed_plan(
    value: Mapping[str, object],
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Validate all manifest plan hashes and deterministic selection relations."""

    expected_fields = {
        "algorithm",
        "cases",
        "cases_sha256",
        "source_case_count",
        "eligible_case_count",
        "selected_full",
        "selected_full_sha256",
        "selected_for_execution",
        "selected_for_execution_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("completed plan has an unexpected shape")
    raw_rows = value.get("cases")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(source.cases):
        raise ValueError("completed plan does not cover the source cohort")
    if value.get("cases_sha256") != _canonical_sha256(raw_rows):
        raise ValueError("completed plan fingerprint does not match its cases")
    if value.get("source_case_count") != len(source.cases):
        raise ValueError("completed plan source count does not match")

    row_fields = {
        "cohort",
        "targeting",
        "sample_id",
        "source_record_sha256",
        "candidate_eligible",
        "boundary_valid",
        "cot_token_count",
        "exclusion_reason",
        "input_plan_sha256",
    }
    rows: dict[tuple[str, str], Mapping[str, object]] = {}
    eligible_by_targeting = {targeting: [] for targeting in _TARGETING_ORDER}
    for raw_row, case in zip(raw_rows, source.cases, strict=True):
        row = _mapping(raw_row, field="completed plan case")
        if set(row) != row_fields:
            raise ValueError("completed plan case has an unexpected shape")
        if any(row.get(field) != expected for field, expected in case.identity_dict().items()):
            raise ValueError("completed plan case source identity does not match")
        key = case.key
        if key in rows:
            raise ValueError("completed plan contains a duplicate case identity")
        rows[key] = row
        candidate = row.get("candidate_eligible")
        if type(candidate) is not bool:
            raise ValueError("completed plan candidate flag must be boolean")
        source_reason = _source_exclusion(case)
        if source_reason is not None:
            expected_state = (False, None, None, source_reason, None)
        elif candidate:
            count = row.get("cot_token_count")
            plan_sha = row.get("input_plan_sha256")
            if not isinstance(count, int) or isinstance(count, bool) or not 8 <= count <= 512:
                raise ValueError("completed eligible plan case is internally inconsistent")
            if row.get("boundary_valid") is True:
                if (
                    not isinstance(plan_sha, str)
                    or _SHA256_PATTERN.fullmatch(plan_sha) is None
                    or row.get("exclusion_reason") is not None
                ):
                    raise ValueError("completed executable plan case is inconsistent")
                expected_state = (True, True, count, None, plan_sha)
            elif row.get("boundary_valid") is False:
                if plan_sha is not None or row.get("exclusion_reason") != "prompt-boundary-invalid":
                    raise ValueError("completed boundary-invalid selected case is inconsistent")
                expected_state = (True, False, count, "prompt-boundary-invalid", None)
            else:
                raise ValueError("completed eligible plan case has no boundary decision")
            eligible_by_targeting[case.targeting].append(case.sample_id)
        elif row.get("boundary_valid") is None:
            reason = row.get("exclusion_reason")
            if not isinstance(reason, str) or not reason.startswith("token-plan-invalid:"):
                raise ValueError("completed token-plan-invalid case has no reason")
            expected_state = (False, None, None, reason, None)
        else:
            count = row.get("cot_token_count")
            plan_sha = row.get("input_plan_sha256")
            if not isinstance(count, int) or isinstance(count, bool) or 8 <= count <= 512:
                raise ValueError("completed length-excluded plan case is inconsistent")
            if row.get("boundary_valid") is True:
                if not isinstance(plan_sha, str) or _SHA256_PATTERN.fullmatch(plan_sha) is None:
                    raise ValueError("completed valid-boundary length exclusion has no plan")
            elif row.get("boundary_valid") is False:
                if plan_sha is not None:
                    raise ValueError("completed invalid-boundary length exclusion claims a plan")
            else:
                raise ValueError("completed length exclusion has no boundary decision")
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or row.get("exclusion_reason") != "clean-pre-answer-length-outside-8-through-512"
            ):
                raise ValueError("completed length-excluded plan case is inconsistent")
            expected_state = (
                False,
                row.get("boundary_valid"),
                count,
                "clean-pre-answer-length-outside-8-through-512",
                plan_sha,
            )
        actual_state = (
            candidate,
            row.get("boundary_valid"),
            row.get("cot_token_count"),
            row.get("exclusion_reason"),
            row.get("input_plan_sha256"),
        )
        if actual_state != expected_state:
            raise ValueError("completed plan case eligibility state does not match")

    if value.get("eligible_case_count") != sum(
        row.get("candidate_eligible") is True for row in rows.values()
    ):
        raise ValueError("completed plan eligible count does not match")
    if value.get("algorithm") != "shared-clean-prefix-cohort-selection-before-limit/v1":
        raise ValueError("completed plan selection algorithm does not match")

    raw_full = value.get("selected_full")
    raw_execution = value.get("selected_for_execution")
    if not isinstance(raw_full, list) or not isinstance(raw_execution, list):
        raise ValueError("completed plan selections must be lists")
    if value.get("selected_full_sha256") != _canonical_sha256(raw_full) or value.get(
        "selected_for_execution_sha256"
    ) != _canonical_sha256(raw_execution):
        raise ValueError("completed plan fingerprint does not match its selection")
    full = tuple(_plan_key(item, field="completed selected_full") for item in raw_full)
    execution = tuple(
        _plan_key(item, field="completed selected_for_execution") for item in raw_execution
    )
    if len(full) != len(set(full)) or len(execution) != len(set(execution)):
        raise ValueError("completed plan selection contains duplicate identities")
    if config.cohort == "primary":
        expected_full = tuple(case.key for case in source.cases)
    else:
        expected_full = select_extension_sample_ids(
            eligible_by_targeting,
            max_pairs=int(config.max_pairs),
            cap_per_targeting=400,
        )
    expected_execution = expected_full if config.limit is None else expected_full[: config.limit]
    if full != expected_full or execution != expected_execution:
        raise ValueError("completed plan selection does not reconstruct")
    return rows, full, execution


def _validate_completed_record(
    record: Mapping[str, object],
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
    eos_ids: Sequence[int],
    manifest_plan_row: Mapping[str, object],
) -> None:
    key = (record.get("targeting"), record.get("sample_id"))
    source_cases = {case.key: case for case in source.cases}
    if key not in source_cases:
        raise ValueError("completed record is outside the source cohort")
    case = source_cases[key]  # type: ignore[index]
    expected_fields = {
        "schema_version",
        "paper_sha256",
        "operation",
        "model",
        "benchmark",
        "cohort",
        "targeting",
        "sample_id",
        "gold_answer",
        "source",
        "input_plan",
        "input_plan_sha256",
        "profile",
        "profile_sha256",
        "selection",
        "positions",
        "adjacent_unavailable_reason",
        "arms",
        "events",
    }
    if set(record) != expected_fields:
        raise ValueError("completed record has an unexpected shape")
    if (
        record.get("schema_version") != _RECORD_SCHEMA
        or record.get("paper_sha256") != PAPER_SHA256
        or record.get("operation") != "one-token-prefix-replacement"
        or record.get("model") != config.model
        or record.get("benchmark") != config.benchmark
        or record.get("cohort") != config.cohort
        or record.get("gold_answer") != case.record.get("gold_answer")
    ):
        raise ValueError("completed record setting or source identity is inconsistent")
    if record.get("source") != {
        "cohort_sha256": source.cohort_sha256,
        "source_record_sha256": case.source_record_sha256,
    }:
        raise ValueError("completed record source identity does not match")
    if (
        manifest_plan_row.get("candidate_eligible") is not True
        or manifest_plan_row.get("boundary_valid") is not True
        or manifest_plan_row.get("exclusion_reason") is not None
        or not isinstance(manifest_plan_row.get("input_plan_sha256"), str)
        or _SHA256_PATTERN.fullmatch(str(manifest_plan_row["input_plan_sha256"])) is None
    ):
        raise ValueError("completed record is not bound to an executable manifest-plan case")
    input_plan_row = _mapping(record.get("input_plan"), field="completed input plan")
    if record.get("input_plan_sha256") != _canonical_sha256(input_plan_row):
        raise ValueError("completed input plan fingerprint does not match")
    if record.get("input_plan_sha256") != manifest_plan_row.get("input_plan_sha256"):
        raise ValueError("completed input plan does not match the manifest plan")
    plan = _input_plan_from_dict(input_plan_row)
    if plan.cot_token_count != manifest_plan_row.get("cot_token_count"):
        raise ValueError("completed input plan length does not match the manifest plan")
    profile_row = _mapping(record.get("profile"), field="completed profile")
    if record.get("profile_sha256") != _canonical_sha256(profile_row):
        raise ValueError("completed profile fingerprint does not match")
    profile = _profile_from_dict(profile_row)
    if len(profile.clean_to_edited_kl) != plan.cot_token_count:
        raise ValueError("completed profile length does not match the input plan")
    selection = choose_distant_positions(profile)
    if record.get("selection") != selection.to_dict():
        raise ValueError("completed selection does not reconstruct")
    positions = _mapping(record.get("positions"), field="completed positions")
    if set(positions) != {"selected", "distant", "adjacent"}:
        raise ValueError("completed positions have an unexpected shape")
    if (
        positions.get("selected") != selection.selected_position
        or positions.get("distant") != selection.distant_position
    ):
        raise ValueError("completed positions do not reconstruct")
    adjacent = None
    if config.adjacent_requested:
        adjacent = choose_adjacent_position(
            profile.clean_to_edited_kl,
            selected_position=selection.selected_position,
            tie_key=config.adjacent_tie_key(
                targeting=case.targeting,
                sample_id=case.sample_id,
            ),
        )
    if positions.get("adjacent") != adjacent:
        raise ValueError("completed adjacent position does not reconstruct")
    expected_adjacent_reason = (
        "no-nearest-strictly-lower-kl-position"
        if config.adjacent_requested and adjacent is None
        else None
    )
    if record.get("adjacent_unavailable_reason") != expected_adjacent_reason:
        raise ValueError("completed adjacent availability does not reconstruct")
    specs = build_arm_specs(plan, profile, selection, adjacent_position=adjacent)
    raw_arms = record.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != len(specs):
        raise ValueError("completed record arm count does not match")
    arms: dict[str, object] = {}
    gold = case.record.get("gold_answer")
    if not isinstance(gold, str) or not gold:
        raise ValueError("completed source has no gold answer")
    for raw, spec in zip(raw_arms, specs, strict=True):
        arm = _mapping(raw, field="completed arm")
        generation = OneTokenGeneration.from_dict(
            _mapping(arm.get("generation"), field="completed generation")
        )
        if dict(arm) != _arm_row(spec, generation, plan):
            raise ValueError("completed arm semantics do not reconstruct")
        _validate_generation(
            generation,
            benchmark=config.benchmark,
            gold_answer=gold,
            eos_ids=eos_ids,
        )
        arms[spec.name] = arm
    events = classify_one_token_events(
        arms,
        selected_before_control=selection.selected_position < int(selection.distant_position),
        adjacent_requested=adjacent is not None,
    )
    if record.get("events") != events:
        raise ValueError("completed event flags do not reconstruct")


def _validate_completed_statuses(
    statuses: Sequence[Mapping[str, object]],
    *,
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
    plan_rows: Mapping[tuple[str, str], Mapping[str, object]],
    selected_full: Sequence[tuple[str, str]],
    selected_execution: Sequence[tuple[str, str]],
    records: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    """Reconstruct every public status row without loading model weights."""

    if len(statuses) != len(source.cases):
        raise ValueError("completed statuses do not cover the source cohort")
    full_keys = set(selected_full)
    execution_keys = set(selected_execution)
    status_fields = {
        "schema_version",
        "paper_sha256",
        "model",
        "benchmark",
        "cohort",
        "targeting",
        "sample_id",
        "source_record_sha256",
        "candidate_eligible",
        "boundary_valid",
        "selected_full",
        "selected_for_execution",
        "positions",
        "adjacent_requested",
        "adjacent_available",
        "adjacent_unavailable_reason",
        "execution_status",
        "exclusion_reason",
        "position_exclusion_evidence",
        "record_sha256",
    }
    position_reasons = {
        "no-position-with-clean-token-below-edited-top1",
        "no-distant-lower-median-control",
    }
    for status, case in zip(statuses, source.cases, strict=True):
        if set(status) != status_fields:
            raise ValueError("completed status semantics have an unexpected shape")
        key = case.key
        row = plan_rows[key]
        record = records.get(key)
        if row.get("candidate_eligible") is not True:
            execution_status = "not-candidate"
            positions = None
            exclusion = row.get("exclusion_reason")
        elif key not in full_keys:
            execution_status = "not-selected"
            positions = None
            exclusion = row.get("exclusion_reason")
        elif key not in execution_keys:
            execution_status = "not-executed-limit"
            positions = None
            exclusion = row.get("exclusion_reason")
        elif row.get("boundary_valid") is False:
            execution_status = "invalid-boundary"
            positions = None
            exclusion = "prompt-boundary-invalid"
        elif record is not None:
            execution_status = "completed"
            positions = record.get("positions")
            exclusion = None
        else:
            execution_status = "position-unavailable"
            raw_evidence = status.get("position_exclusion_evidence")
            if not isinstance(raw_evidence, Mapping):
                raise ValueError("completed status has no position-exclusion evidence")
            input_plan_row = _mapping(
                raw_evidence.get("input_plan"),
                field="completed position-exclusion evidence input plan",
            )
            if raw_evidence.get("input_plan_sha256") != _canonical_sha256(
                input_plan_row
            ) or raw_evidence.get("input_plan_sha256") != row.get("input_plan_sha256"):
                raise ValueError("completed position-exclusion evidence input plan does not match")
            input_plan = _input_plan_from_dict(input_plan_row)
            if (
                row.get("candidate_eligible") is not True
                or row.get("boundary_valid") is not True
                or row.get("exclusion_reason") is not None
                or input_plan.cot_token_count != row.get("cot_token_count")
            ):
                raise ValueError("completed position-exclusion evidence is outside the plan")
            planned_case = _PlannedCase(
                source=case,
                input_plan=input_plan,
                cot_token_count=input_plan.cot_token_count,
                candidate_eligible=True,
                boundary_valid=True,
                exclusion_reason=None,
            )
            _validate_position_exclusion_checkpoint(
                raw_evidence,
                case=planned_case,
                config=config,
            )
            positions = raw_evidence.get("positions")
            exclusion = raw_evidence.get("position_exclusion_reason")
            if exclusion not in position_reasons:
                raise ValueError("completed status semantics have an invalid position reason")
            if not isinstance(positions, Mapping) or set(positions) != {
                "selected",
                "distant",
                "adjacent",
            }:
                raise ValueError("completed status semantics have invalid excluded positions")
            if positions.get("distant") is not None or positions.get("adjacent") is not None:
                raise ValueError("completed status semantics claim an excluded control position")
            selected_position = positions.get("selected")
            if exclusion == "no-position-with-clean-token-below-edited-top1":
                if selected_position is not None:
                    raise ValueError(
                        "completed status semantics claim an excluded selected position"
                    )
            elif (
                not isinstance(selected_position, int)
                or isinstance(selected_position, bool)
                or selected_position < 0
            ):
                raise ValueError("completed status semantics have no selected position")
        adjacent_available = (
            isinstance(positions, Mapping) and positions.get("adjacent") is not None
        )
        if record is not None:
            adjacent_unavailable_reason = record.get("adjacent_unavailable_reason")
        elif execution_status == "position-unavailable":
            adjacent_unavailable_reason = (
                "case-excluded-before-adjacent-position-selection"
                if config.adjacent_requested
                else None
            )
            if status.get("adjacent_unavailable_reason") != adjacent_unavailable_reason:
                raise ValueError("completed status adjacent availability does not reconstruct")
        else:
            adjacent_unavailable_reason = None
        expected = {
            "schema_version": _STATUS_SCHEMA,
            "paper_sha256": PAPER_SHA256,
            "model": config.model,
            "benchmark": config.benchmark,
            "cohort": config.cohort,
            "targeting": case.targeting,
            "sample_id": case.sample_id,
            "source_record_sha256": case.source_record_sha256,
            "candidate_eligible": row.get("candidate_eligible"),
            "boundary_valid": row.get("boundary_valid"),
            "selected_full": key in full_keys,
            "selected_for_execution": key in execution_keys,
            "positions": positions,
            "adjacent_requested": config.adjacent_requested,
            "adjacent_available": adjacent_available,
            "adjacent_unavailable_reason": adjacent_unavailable_reason,
            "execution_status": execution_status,
            "exclusion_reason": exclusion,
            "position_exclusion_evidence": (
                dict(raw_evidence) if execution_status == "position-unavailable" else None
            ),
            "record_sha256": (None if record is None else _canonical_sha256(record)),
        }
        if dict(status) != expected:
            raise ValueError("completed status semantics do not reconstruct")


def _completed_resume(
    *,
    previous: Mapping[str, object],
    config: OneTokenPrefixReplacementConfig,
    source: SourceBundle,
    paths: tuple[Path, Path, Path, Path],
) -> OneTokenPrefixReplacementResult:
    records_path, statuses_path, summary_path, run_path = paths
    outputs = _mapping(previous.get("outputs"), field="completed outputs")
    if set(outputs) != set(_OUTPUT_NAMES):
        raise ValueError("completed outputs do not name the three public data files")
    for path in (records_path, statuses_path, summary_path):
        metadata = _mapping(outputs.get(path.name), field=f"output {path.name}")
        if set(metadata) != {"sha256", "bytes"}:
            raise ValueError(f"completed output metadata has an invalid shape: {path.name}")
        expected_sha = metadata.get("sha256")
        expected_bytes = metadata.get("bytes")
        if (
            not isinstance(expected_sha, str)
            or _SHA256_PATTERN.fullmatch(expected_sha) is None
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise ValueError(f"completed output metadata has invalid values: {path.name}")
        if not path.is_file():
            raise ValueError(f"completed output is missing or is not a file: {path.name}")
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"completed output SHA-256 mismatch: {path.name}")
        if metadata.get("bytes") != path.stat().st_size:
            raise ValueError(f"completed output byte count mismatch: {path.name}")
    runtime = _validate_runtime_provenance(
        _mapping(previous.get("runtime"), field="completed runtime"),
        source=source,
        gpu_id=config.gpu_id,
    )
    eos = runtime.get("effective_eos_token_ids")
    if not isinstance(eos, list):
        raise ValueError("completed runtime has no EOS IDs")
    plan = _mapping(previous.get("plan"), field="completed plan")
    plan_rows, selected_full, selected_execution = _validate_completed_plan(
        plan,
        config=config,
        source=source,
    )
    records = _load_jsonl(records_path)
    statuses = _load_jsonl(statuses_path)
    summary = _load_json(summary_path)
    records_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for record in records:
        targeting = record.get("targeting")
        sample_id = record.get("sample_id")
        if not isinstance(targeting, str) or not isinstance(sample_id, str):
            raise ValueError("completed record identity is invalid")
        key = (targeting, sample_id)
        if key in records_by_key:
            raise ValueError("completed records contain a duplicate identity")
        if key not in selected_execution or plan_rows[key].get("candidate_eligible") is not True:
            raise ValueError("completed record is outside the executable plan")
        _validate_completed_record(
            record,
            config=config,
            source=source,
            eos_ids=eos,
            manifest_plan_row=plan_rows[key],
        )
        records_by_key[key] = record
    _validate_completed_statuses(
        statuses,
        config=config,
        source=source,
        plan_rows=plan_rows,
        selected_full=selected_full,
        selected_execution=selected_execution,
        records=records_by_key,
    )
    rebuilt = _build_summary(
        config=config,
        source=source,
        plan=plan,
        records=records,
        statuses=statuses,
    )
    if summary != rebuilt:
        raise ValueError("completed one-token summary does not reconstruct")
    counts = _mapping(previous.get("counts"), field="completed counts")
    expected_counts = {
        "source_pairs": len(source.cases),
        "selected_full": len(selected_full),
        "selected_for_execution": len(selected_execution),
        "records": len(records),
    }
    if dict(counts) != expected_counts or previous.get("checkpoints") != {}:
        raise ValueError("completed manifest counts or checkpoint state is inconsistent")
    validate_source_snapshot(source)
    residual_work_dir = config.output_dir / ".one-token-prefix-replacement-work"
    if residual_work_dir.exists():
        with suppress(OSError):
            shutil.rmtree(residual_work_dir)
    return OneTokenPrefixReplacementResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        source_pairs=len(source.cases),
        selected_pairs=len(selected_full),
        executed_pairs=len(selected_execution),
        records=len(records),
    )


def run_one_token_prefix_replacement(
    config: OneTokenPrefixReplacementConfig,
    *,
    runtime: OneTokenPrefixReplacementRuntime | None = None,
) -> OneTokenPrefixReplacementResult:
    """Validate, profile, generate, checkpoint, and publish one setting."""

    if not isinstance(config, OneTokenPrefixReplacementConfig):
        raise TypeError("config must be OneTokenPrefixReplacementConfig")
    output_dir = config.output_dir
    records_path = output_dir / _OUTPUT_NAMES[0]
    statuses_path = output_dir / _OUTPUT_NAMES[1]
    summary_path = output_dir / _OUTPUT_NAMES[2]
    run_path = output_dir / "run.json"
    paths = (records_path, statuses_path, summary_path, run_path)
    work_dir = output_dir / ".one-token-prefix-replacement-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError("cannot resume without the original run.json")
    if output_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    source = load_source_bundle(
        cohort=config.cohort,  # type: ignore[arg-type]
        model=config.model,
        benchmark=config.benchmark,
        fixed_window_run=config.fixed_window_run,
        pairs=config.pairs,
    )
    validate_source_snapshot(source)
    previous = _load_json(run_path) if config.resume else None
    if previous is not None:
        _validate_resume_core(previous, config=config, source=source)
        if previous.get("status") == "completed":
            return _completed_resume(
                previous=previous,
                config=config,
                source=source,
                paths=paths,
            )
        if previous.get("status") not in {"running", "failed"}:
            raise ValueError("resume status is not recognized")

    # Incomplete public data files can only be crash remnants. They are never
    # inputs to resume; the hash-bound private checkpoints are the sole source
    # of recoverable progress.
    for path in (records_path, statuses_path, summary_path):
        path.unlink(missing_ok=True)

    if runtime is None:
        from typo_cot.experiments.one_token_prefix_replacement.runtime import (
            HuggingFaceOneTokenPrefixReplacementRuntime,
        )

        runtime = HuggingFaceOneTokenPrefixReplacementRuntime(
            config,
            revision=source.model_revision,
        )
    runtime_provenance = _validate_runtime_provenance(
        runtime.provenance(),
        source=source,
        gpu_id=config.gpu_id,
    )
    runtime_provenance_sha256 = _canonical_sha256(runtime_provenance)
    eos_ids = runtime_provenance["effective_eos_token_ids"]
    planned = _prepare_cases(source, runtime)
    selected_full, selected = _select_cases(planned, config)
    plan = _plan_payload(planned, selected_full, selected)
    if previous is not None and previous.get("plan") != plan:
        raise ValueError("resume full source or selection plan does not match")
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    started_at = str(previous.get("started_at")) if previous is not None else _now()
    checkpoint_registry: dict[str, object] = {}
    checkpoints_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    failures: list[dict[str, object]] = []

    def update_registry(case: _PlannedCase, path: Path, value: Mapping[str, object]) -> None:
        metadata = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "arms_completed": len(value.get("arms", [])),  # type: ignore[arg-type]
            "runtime_provenance_sha256": runtime_provenance_sha256,
        }
        checkpoint_registry[_checkpoint_identity(case.source)] = metadata
        checkpoints_by_key[case.key] = value

    def write_progress(status: str) -> None:
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                source=source,
                plan=plan,
                status=status,
                started_at=started_at,
                runtime=runtime_provenance,
                checkpoints=checkpoint_registry,
                counts={
                    "source_pairs": len(source.cases),
                    "selected_full": len(selected_full),
                    "selected_for_execution": len(selected),
                    "checkpoints": len(checkpoint_registry),
                    "records": sum(
                        isinstance(value.get("arms"), list)
                        and len(value["arms"]) == len(value.get("arm_specs", []))
                        and value.get("position_exclusion_reason") is None
                        for value in checkpoints_by_key.values()
                    ),
                },
                failures=failures,
            ),
        )

    try:
        # Adopt only byte-exact checkpoints registered by the prior manifest.
        # A checkpoint can contain generated outcomes, so semantic validation
        # alone is not enough to authorize unregistered or reserialized bytes.
        for temporary in checkpoints_dir.glob(".*.tmp"):
            temporary.unlink(missing_ok=True)
        prior_registry: Mapping[str, object] = {}
        if previous is not None:
            prior_registry = _mapping(
                previous.get("checkpoints"), field="resume checkpoint registry"
            )
            selected_by_identity = {_checkpoint_identity(case.source): case for case in selected}
            disk_paths = {path.resolve() for path in checkpoints_dir.glob("*.json")}
            registered_paths: set[Path] = set()
            for identity, raw_metadata in prior_registry.items():
                if identity not in selected_by_identity:
                    raise ValueError("resume checkpoint is outside the execution plan")
                case = selected_by_identity[identity]
                path = _checkpoint_path(checkpoints_dir, case.source)
                metadata = _mapping(raw_metadata, field=f"resume checkpoint {identity}")
                if set(metadata) != {
                    "path",
                    "sha256",
                    "arms_completed",
                    "runtime_provenance_sha256",
                }:
                    raise ValueError("resume checkpoint metadata has an unexpected shape")
                if metadata.get("path") != str(path.resolve()):
                    raise ValueError("resume checkpoint path does not match the run manifest")
                if metadata.get("runtime_provenance_sha256") != runtime_provenance_sha256:
                    raise ValueError("resume checkpoint runtime provenance does not match")
                if not path.is_file():
                    raise ValueError("resume checkpoint registered by the manifest is missing")
                if metadata.get("sha256") != _sha256(path):
                    raise ValueError("resume checkpoint SHA-256 does not match the run manifest")
                registered_paths.add(path.resolve())
            if disk_paths != registered_paths:
                raise ValueError("unregistered resume checkpoint file is present")
        for case in selected:
            if not case.candidate_eligible or case.input_plan is None:
                continue
            path = _checkpoint_path(checkpoints_dir, case.source)
            if path.is_file():
                value = _load_json(path)
                metadata = _mapping(
                    prior_registry.get(_checkpoint_identity(case.source)),
                    field="resume checkpoint metadata",
                )
                arms = value.get("arms")
                if not isinstance(arms, list) or metadata.get("arms_completed") != len(arms):
                    raise ValueError("resume checkpoint arm count does not match the manifest")
                if value.get("position_exclusion_reason"):
                    _validate_position_exclusion_checkpoint(
                        value,
                        case=case,
                        config=config,
                    )
                else:
                    _validate_checkpoint(
                        value,
                        case=case,
                        config=config,
                        eos_ids=eos_ids,  # type: ignore[arg-type]
                    )
                update_registry(case, path, value)
        write_progress("running")

        executable = [
            case for case in selected if case.candidate_eligible and case.input_plan is not None
        ]
        for case in tqdm(executable, desc="one-token-prefix-replacement", unit="pair"):
            if case.input_plan is None:
                raise AssertionError("executable case lost its input plan")
            checkpoint_path = _checkpoint_path(checkpoints_dir, case.source)
            checkpoint = (
                dict(checkpoints_by_key[case.key]) if case.key in checkpoints_by_key else None
            )
            if checkpoint is None:
                profile = runtime.profile_pair(case.input_plan)
                if not isinstance(profile, OneTokenProfile):
                    raise TypeError("runtime.profile_pair did not return OneTokenProfile")
                if len(profile.clean_to_edited_kl) != case.input_plan.cot_token_count:
                    raise ValueError("runtime profile length does not match the clean CoT")
                try:
                    selection = choose_distant_positions(profile)
                except ValueError:
                    checkpoint = _position_exclusion_checkpoint(
                        case,
                        profile=profile,
                        config=config,
                    )
                    _write_json_atomic(checkpoint_path, checkpoint)
                    update_registry(case, checkpoint_path, checkpoint)
                    write_progress("running")
                    continue
                if selection.distant_position is None:
                    checkpoint = _position_exclusion_checkpoint(
                        case,
                        profile=profile,
                        config=config,
                    )
                    _write_json_atomic(checkpoint_path, checkpoint)
                    update_registry(case, checkpoint_path, checkpoint)
                    write_progress("running")
                    continue
                adjacent = None
                if config.adjacent_requested:
                    adjacent = choose_adjacent_position(
                        profile.clean_to_edited_kl,
                        selected_position=selection.selected_position,
                        tie_key=config.adjacent_tie_key(
                            targeting=case.source.targeting,
                            sample_id=case.source.sample_id,
                        ),
                    )
                specs = build_arm_specs(
                    case.input_plan,
                    profile,
                    selection,
                    adjacent_position=adjacent,
                )
                checkpoint = _checkpoint_base(
                    case,
                    profile=profile,
                    selection=selection,
                    adjacent_position=adjacent,
                    adjacent_requested=config.adjacent_requested,
                    arm_specs=specs,
                )
                _write_json_atomic(checkpoint_path, checkpoint)
                update_registry(case, checkpoint_path, checkpoint)
                write_progress("running")
            elif checkpoint.get("position_exclusion_reason"):
                _validate_position_exclusion_checkpoint(
                    checkpoint,
                    case=case,
                    config=config,
                )
                continue
            else:
                _, _, _, specs = _validate_checkpoint(
                    checkpoint,
                    case=case,
                    config=config,
                    eos_ids=eos_ids,  # type: ignore[arg-type]
                )

            raw_arms = checkpoint.get("arms")
            if not isinstance(raw_arms, list):
                raise ValueError("checkpoint arms must be a list")
            gold = case.source.record.get("gold_answer")
            if not isinstance(gold, str) or not gold:
                raise ValueError("source record has no gold answer")
            for spec in specs[len(raw_arms) :]:
                generation = runtime.generate_arm(
                    case.input_plan,
                    position=spec.position,
                    forced_token_id=spec.forced_token_id,
                    gold_answer=gold,
                )
                if not isinstance(generation, OneTokenGeneration):
                    raise TypeError("runtime.generate_arm did not return OneTokenGeneration")
                _validate_generation(
                    generation,
                    benchmark=config.benchmark,
                    gold_answer=gold,
                    eos_ids=eos_ids,  # type: ignore[arg-type]
                )
                raw_arms.append(_arm_row(spec, generation, case.input_plan))
                checkpoint["arms"] = raw_arms
                _write_json_atomic(checkpoint_path, checkpoint)
                update_registry(case, checkpoint_path, checkpoint)
                write_progress("running")

        # Revalidate every complete checkpoint before public serialization.
        records_list: list[dict[str, object]] = []
        records_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
        for case in selected:
            checkpoint = checkpoints_by_key.get(case.key)
            if checkpoint is None or checkpoint.get("position_exclusion_reason"):
                continue
            _validate_checkpoint(
                checkpoint,
                case=case,
                config=config,
                eos_ids=eos_ids,  # type: ignore[arg-type]
            )
            if len(checkpoint["arms"]) != len(checkpoint["arm_specs"]):  # type: ignore[arg-type]
                raise ValueError("checkpoint is not complete")
            record = _record_from_checkpoint(
                checkpoint,
                case=case,
                source=source,
                config=config,
            )
            records_list.append(record)
            records_by_key[case.key] = record
        statuses = _status_rows(
            planned,
            selected_full,
            selected,
            checkpoints_by_key,
            records_by_key,
            config=config,
        )
        summary = _build_summary(
            config=config,
            source=source,
            plan=plan,
            records=records_list,
            statuses=statuses,
        )
        validate_source_snapshot(source)
        _write_jsonl_atomic(records_path, records_list)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        validate_source_snapshot(source)
        outputs = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (records_path, statuses_path, summary_path)
        }
        final_counts = {
            "source_pairs": len(source.cases),
            "selected_full": len(selected_full),
            "selected_for_execution": len(selected),
            "records": len(records_list),
        }
        validate_source_snapshot(source)
        _write_json_atomic(
            run_path,
            _manifest(
                config=config,
                source=source,
                plan=plan,
                status="completed",
                started_at=started_at,
                runtime=runtime_provenance,
                checkpoints={},
                counts=final_counts,
                failures=[],
                outputs=outputs,
            ),
        )
        # The completed manifest is the commit point. Checkpoints remain
        # recoverable if that atomic write fails, and are removed only after it
        # succeeds.
        if work_dir.exists():
            with suppress(OSError):
                shutil.rmtree(work_dir)
    except Exception as exc:
        failures.append({"type": type(exc).__name__, "message": str(exc)})
        for path in (records_path, statuses_path, summary_path):
            path.unlink(missing_ok=True)
        with suppress(OSError, TypeError, ValueError):
            write_progress("failed")
        raise OneTokenPrefixReplacementRunError(str(exc)) from exc

    return OneTokenPrefixReplacementResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        source_pairs=len(source.cases),
        selected_pairs=len(selected_full),
        executed_pairs=len(selected),
        records=len(records_list),
    )


__all__ = [
    "OneTokenGeneration",
    "OneTokenPrefixReplacementConfig",
    "OneTokenPrefixReplacementResult",
    "OneTokenPrefixReplacementRunError",
    "OneTokenPrefixReplacementRuntime",
    "classify_one_token_events",
    "run_one_token_prefix_replacement",
]

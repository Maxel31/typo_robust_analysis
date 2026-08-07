"""Validated, point-resumable runner for the clean pre-answer prefix scan."""

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
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.clean_prefix_scan.metrics import summarize_prefix_correctness
from typo_cot.experiments.clean_prefix_scan.planning import PrefixInputPlan
from typo_cot.experiments.clean_prefix_scan.planning import (
    PrefixBudget,
    build_budget_grid,
    select_extension_sample_ids,
)
from typo_cot.experiments.clean_prefix_scan.protocol import (
    ANSWER_DECODING,
    ANSWER_EXTRACTION,
    GENERATION,
    IMPLEMENTATION,
    INPUT_ASSEMBLY,
    PRE_ANSWER_BOUNDARY,
    SCAN_BOUNDARY,
    SELECTION_ALIGNMENT,
)
from typo_cot.experiments.clean_prefix_scan.source import (
    SourceBundle,
    SourceCase,
    load_source_bundle,
    validate_source_snapshot,
)

_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_BENCHMARKS = frozenset({"gsm8k", "mmlu", "arc"})
_TARGETING_ORDER = {"attribution-4": 0, "random-4": 1}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLIC_OUTPUT_NAMES = (
    "prefix_scan_records.jsonl",
    "pair_status_records.jsonl",
    "prefix_scan_summary.json",
)
_PROTOCOL = {
    "schema_version": "clean-prefix-scan-protocol/v1",
    "paper_defined": {
        "intervention": "edited-question-plus-first-k-clean-pre-answer-token-ids",
        "relative_budget": "k=round(r*L_C)",
        "generation": "independent-greedy-bfloat16-left-padding-max-new-tokens-512",
        "readout": "gold-answer-correctness-unextractable-is-wrong",
        "denominator": "complete-valid-scan-and-fresh-k0-wrong",
        "stable": "earliest-union-grid-k-with-all-later-points-correct",
        "short": "k-star-divided-by-L_C-less-than-or-equal-to-0.2",
        "non_monotone": "at-least-two-adjacent-correctness-transitions",
        "extension_pool": "arm-proportional-targets-after-per-arm-cap",
    },
    "legacy_backed": {
        "pre_answer_boundary": PRE_ANSWER_BOUNDARY,
        "selection_alignment": SELECTION_ALIGNMENT,
        "scan_boundary": SCAN_BOUNDARY,
        "input_assembly": INPUT_ASSEMBLY,
        "rounding": "python-round-ties-to-even/v1",
        "grid": "caller-supplied-frozen-relative-and-absolute-union/v1",
        "targeting_cap": 400,
        "quota": "python-round-then-largest-arm-adjustment/v1",
        "within_arm": "sample-id-sort-systematic-floor-index/v1",
        "batch_size": 1,
    },
    "runtime": IMPLEMENTATION,
    "generation": dict(GENERATION),
    "answer_decoding": ANSWER_DECODING,
    "answer_extraction": ANSWER_EXTRACTION,
    "checkpoint": "one-pair-growing-point-grid-atomic/v1",
}


class CleanPrefixScanRunError(RuntimeError):
    """Raised after preserving valid checkpoints from a failed prefix scan."""


@dataclass(frozen=True, slots=True)
class CleanPrefixScanConfig:
    """Frozen public arguments for one model/task clean-prefix setting."""

    model: str
    benchmark: Literal["gsm8k", "mmlu", "arc"]
    cohort: Literal["primary", "extension"]
    relative_budgets: tuple[float, ...]
    absolute_budgets: tuple[int, ...]
    output_dir: Path
    fixed_window_run: Path | None = None
    pairs: tuple[Path, ...] = ()
    max_pairs: int | None = None
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "fixed_window_run",
            None if self.fixed_window_run is None else Path(self.fixed_window_run),
        )
        object.__setattr__(self, "pairs", tuple(Path(path) for path in self.pairs))
        object.__setattr__(self, "relative_budgets", tuple(self.relative_budgets))
        object.__setattr__(self, "absolute_budgets", tuple(self.absolute_budgets))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in _BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if self.cohort == "primary":
            if self.fixed_window_run is None or self.pairs or self.max_pairs is not None:
                raise ValueError(
                    "primary cohort requires exactly --fixed-window-run and forbids "
                    "extension pairs/max-pairs"
                )
        elif self.cohort == "extension":
            if self.fixed_window_run is not None or len(self.pairs) != 2 or self.max_pairs is None:
                raise ValueError(
                    "extension cohort requires exactly two --pairs and --max-pairs, "
                    "and forbids --fixed-window-run"
                )
        else:
            raise ValueError("cohort must be primary or extension")
        if len({path.resolve() for path in self.pairs}) != len(self.pairs):
            raise ValueError("extension pairs must identify two different files")
        if self.max_pairs is not None and (
            not isinstance(self.max_pairs, int)
            or isinstance(self.max_pairs, bool)
            or self.max_pairs <= 0
        ):
            raise ValueError("max_pairs must be a positive integer")
        if not isinstance(self.gpu_id, str) or _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")
        # Validate the complete argument-level grid here. Pair-specific clipping
        # happens later once L_C is known.
        from typo_cot.experiments.clean_prefix_scan.planning import build_budget_grid

        build_budget_grid(
            1,
            relative_budgets=self.relative_budgets,
            absolute_budgets=self.absolute_budgets,
        )
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

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""

        payload = asdict(self)
        payload["fixed_window_run"] = (
            None if self.fixed_window_run is None else str(self.fixed_window_run.resolve())
        )
        payload["pairs"] = [str(path.resolve()) for path in self.pairs]
        payload["relative_budgets"] = list(self.relative_budgets)
        payload["absolute_budgets"] = list(self.absolute_budgets)
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class CleanPrefixGeneration:
    """One independently generated continuation and extracted gold readout."""

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
            raise ValueError("token_ids must contain non-negative integers")
        for field in ("text", "value", "method", "primary_method"):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"{field} must be a string")
        if not isinstance(self.is_extracted, bool) or not isinstance(self.is_correct, bool):
            raise TypeError("generation extraction and correctness flags must be boolean")
        if self.is_extracted is not bool(self.value):
            raise ValueError("is_extracted must agree with the extracted value")
        if self.stop_reason not in {"eos_token", "max_new_tokens"}:
            raise ValueError("stop_reason must be eos_token or max_new_tokens")
        if self.stop_token_id is not None and (
            not isinstance(self.stop_token_id, int)
            or isinstance(self.stop_token_id, bool)
            or self.stop_token_id < 0
        ):
            raise ValueError("stop_token_id must be null or a non-negative integer")
        max_new_tokens = int(GENERATION["max_new_tokens"])
        if len(self.token_ids) > max_new_tokens:
            raise ValueError(f"generation cannot exceed {max_new_tokens} new tokens")
        if self.stop_reason == "eos_token":
            if not self.token_ids or self.stop_token_id != self.token_ids[-1]:
                raise ValueError("EOS stop_token_id must equal the last generated token")
        elif self.stop_token_id is not None:
            raise ValueError("max_new_tokens termination must not retain a stop token")
        elif len(self.token_ids) != max_new_tokens:
            raise ValueError(
                f"max_new_tokens termination requires exactly {max_new_tokens} generated tokens"
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        return payload


@dataclass(frozen=True, slots=True)
class CleanPrefixInputUse:
    """Exact token IDs supplied at one clean-prefix budget."""

    k: int
    input_ids: tuple[int, ...]
    prompt_token_count: int
    prefix_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        for field in ("k", "prompt_token_count", "prefix_token_count"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.k != self.prefix_token_count:
            raise ValueError("k must equal prefix_token_count")
        if self.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if len(self.input_ids) != self.prompt_token_count + self.prefix_token_count:
            raise ValueError("input token counts are inconsistent")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.input_ids
        ):
            raise ValueError("input_ids must contain non-negative integers")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_ids"] = list(self.input_ids)
        return payload


@dataclass(frozen=True, slots=True)
class CleanPrefixPointScan:
    """One completed budget point."""

    k: int
    input_use: CleanPrefixInputUse
    generation: CleanPrefixGeneration

    def __post_init__(self) -> None:
        if self.k != self.input_use.k:
            raise ValueError("point k does not match its input use")

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "input_use": self.input_use.to_dict(),
            "generation": self.generation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CleanPrefixPairScan:
    """All distinct prefix-budget points for one selected pair."""

    sample_id: str
    points: tuple[CleanPrefixPointScan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        ks = tuple(point.k for point in self.points)
        if not ks or ks != tuple(sorted(set(ks))):
            raise ValueError("pair scan points must be a non-empty, distinct increasing grid")


class CleanPrefixScanRuntime(Protocol):
    """Point-level runtime seam used by GPU execution and CPU fixtures."""

    def provenance(self) -> Mapping[str, object]: ...

    def prepare_pair(self, pair: Mapping[str, object]) -> PrefixInputPlan: ...

    def generate_point(
        self,
        plan: PrefixInputPlan,
        k: int,
        *,
        gold_answer: str,
    ) -> CleanPrefixPointScan: ...


@dataclass(frozen=True, slots=True)
class CleanPrefixScanResult:
    """Published paths and record counts for one completed setting."""

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
    plan: PrefixInputPlan | None
    cot_token_count: int | None
    candidate_eligible: bool
    boundary_valid: bool | None
    exclusion_reason: str | None

    @property
    def key(self) -> tuple[str, str]:
        return self.source.key

    @property
    def identity(self) -> str:
        return f"{self.source.targeting}\0{self.source.sample_id}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_json(payload: object) -> str:
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
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_serialized_json(payload))
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON at {context}: {exc}") from exc


def _load_json(path: Path) -> dict[str, object]:
    payload = _loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = _loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _checkpoint_identity(case: SourceCase) -> str:
    return hashlib.sha256(
        f"clean-prefix-scan\0{case.targeting}\0{case.sample_id}".encode()
    ).hexdigest()


def _checkpoint_path(directory: Path, case: SourceCase) -> Path:
    return directory / f"{_checkpoint_identity(case)}.json"


def _plan_row(item: _PlannedCase) -> dict[str, object]:
    payload: dict[str, object] = {
        **item.source.identity_dict(),
        "candidate_eligible": item.candidate_eligible,
        "boundary_valid": item.boundary_valid,
        "cot_token_count": item.cot_token_count,
        "exclusion_reason": item.exclusion_reason,
    }
    if item.plan is not None:
        payload["token_plan_sha256"] = _canonical_sha256(item.plan.to_dict())
    else:
        payload["token_plan_sha256"] = None
    return payload


def _plan_payload(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
) -> dict[str, object]:
    rows = [_plan_row(item) for item in planned]
    full_keys = [list(item.key) for item in selected_full]
    executed_keys = [list(item.key) for item in selected]
    return {
        "algorithm": "full-candidate-plan-before-limit/v1",
        "cases": rows,
        "cases_sha256": _canonical_sha256(rows),
        "source_case_count": len(planned),
        "eligible_case_count": sum(item.candidate_eligible for item in planned),
        "selected_full": full_keys,
        "selected_full_sha256": _canonical_sha256(full_keys),
        "selected_for_execution": executed_keys,
        "selected_for_execution_sha256": _canonical_sha256(executed_keys),
    }


def _source_eligibility(case: SourceCase) -> str | None:
    if not case.prepared_clean_correct:
        return "clean-answer-not-correct-after-reextraction"
    if not case.prepared_edited_wrong:
        return "edited-answer-not-wrong-after-reextraction"
    if case.aligned_word_count <= 0:
        return "no-aligned-edited-word"
    return None


def _prepare_cases(
    source: SourceBundle,
    runtime: CleanPrefixScanRuntime,
) -> tuple[_PlannedCase, ...]:
    from typo_cot.experiments.clean_prefix_scan.runtime import (
        CleanPrefixBoundaryInvalid,
    )

    planned: list[_PlannedCase] = []
    for case in source.cases:
        exclusion = _source_eligibility(case)
        if exclusion is not None:
            planned.append(
                _PlannedCase(
                    source=case,
                    plan=None,
                    cot_token_count=None,
                    candidate_eligible=False,
                    boundary_valid=None,
                    exclusion_reason=exclusion,
                )
            )
            continue
        try:
            token_plan = runtime.prepare_pair(case.record)
        except CleanPrefixBoundaryInvalid as exc:
            eligible = exc.alignment.eligible_length
            planned.append(
                _PlannedCase(
                    source=case,
                    plan=None,
                    cot_token_count=exc.alignment.cot_token_count,
                    candidate_eligible=eligible,
                    boundary_valid=False,
                    exclusion_reason=(
                        "edited-prompt-boundary-invalid"
                        if eligible
                        else "clean-pre-answer-length-outside-8-through-512"
                    ),
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            planned.append(
                _PlannedCase(
                    source=case,
                    plan=None,
                    cot_token_count=None,
                    candidate_eligible=False,
                    boundary_valid=None,
                    exclusion_reason=f"token-plan-invalid:{exc}",
                )
            )
            continue
        if not isinstance(token_plan, PrefixInputPlan):
            raise TypeError("runtime.prepare_pair must return PrefixInputPlan")
        eligible = token_plan.cot_token_count >= 8 and token_plan.cot_token_count <= 512
        planned.append(
            _PlannedCase(
                source=case,
                plan=token_plan,
                cot_token_count=token_plan.cot_token_count,
                candidate_eligible=eligible,
                boundary_valid=True,
                exclusion_reason=(
                    None if eligible else "clean-pre-answer-length-outside-8-through-512"
                ),
            )
        )
    return tuple(planned)


def _select_cases(
    planned: Sequence[_PlannedCase],
    config: CleanPrefixScanConfig,
) -> tuple[tuple[_PlannedCase, ...], tuple[_PlannedCase, ...]]:
    by_key = {item.key: item for item in planned}
    if config.cohort == "primary":
        # The upstream fixed-window adapter has already frozen the exact
        # clean-to-edited denominator. Keep every member so any boundary issue
        # remains an explicit invalid scan instead of silently changing n=172.
        full = tuple(planned)
    else:
        eligible_by_targeting: dict[str, list[str]] = {
            targeting: [] for targeting in _TARGETING_ORDER
        }
        for item in planned:
            if item.candidate_eligible:
                eligible_by_targeting[item.source.targeting].append(item.source.sample_id)
        selected_keys = select_extension_sample_ids(
            eligible_by_targeting,
            max_pairs=int(config.max_pairs),
            cap_per_targeting=400,
        )
        full = tuple(by_key[key] for key in selected_keys)
    if not full:
        raise ValueError("clean-prefix source contains no selectable targets")
    selected = full if config.limit is None else full[: config.limit]
    if not selected:
        raise ValueError("clean-prefix execution selection is empty")
    return full, selected


def _validate_runtime_provenance(
    value: Mapping[str, object],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
) -> dict[str, object]:
    runtime = dict(value)
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if runtime.get(field) != source.model_revision:
            raise ValueError(f"runtime {field} does not match the prepared source revision")
    if runtime.get("dtype") != "bfloat16":
        raise ValueError("runtime dtype must be bfloat16")
    device = runtime.get("device")
    if device != "cuda:0":
        raise ValueError("runtime device must be logical cuda:0 on the requested visible GPU")
    generation = _mapping(runtime.get("generation"), field="runtime generation")
    if dict(generation) != GENERATION:
        raise ValueError("runtime generation protocol must match the frozen greedy protocol")
    required_protocol = {
        "answer_decoding": ANSWER_DECODING,
        "answer_extraction": ANSWER_EXTRACTION,
        "pre_answer_boundary": PRE_ANSWER_BOUNDARY,
        "selection_alignment": SELECTION_ALIGNMENT,
        "scan_boundary": SCAN_BOUNDARY,
        "input_assembly": INPUT_ASSEMBLY,
    }
    for field, expected in required_protocol.items():
        if runtime.get(field) != expected:
            raise ValueError(f"runtime {field} does not match the frozen protocol")
    runtime_name = runtime.get("runtime")
    if not isinstance(runtime_name, str) or not runtime_name:
        raise ValueError("runtime implementation identity must be a non-empty string")
    raw_eos = runtime.get("effective_eos_token_ids")
    if not isinstance(raw_eos, list) or not raw_eos:
        raise ValueError("runtime effective_eos_token_ids must be a non-empty list")
    eos_ids = tuple(raw_eos)
    if (
        any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in eos_ids)
        or len(eos_ids) != len(set(eos_ids))
        or tuple(sorted(eos_ids)) != eos_ids
    ):
        raise ValueError("runtime effective EOS token IDs must be distinct increasing integers")
    eos_source = runtime.get("effective_eos_token_ids_source")
    if not isinstance(eos_source, str) or not eos_source:
        raise ValueError("runtime effective EOS source must be a non-empty string")
    visible = runtime.get("cuda_visible_devices")
    if visible != config.gpu_id:
        raise ValueError("runtime CUDA_VISIBLE_DEVICES does not match --gpu-id")
    return runtime


HuggingFaceCleanPrefixScanRuntime: Any = None


def _runtime_class() -> Any:
    if HuggingFaceCleanPrefixScanRuntime is not None:
        return HuggingFaceCleanPrefixScanRuntime
    from typo_cot.experiments.clean_prefix_scan.runtime import (
        HuggingFaceCleanPrefixScanRuntime as runtime_class,
    )

    return runtime_class


def _validate_generation(
    generation: CleanPrefixGeneration,
    *,
    benchmark: str,
    gold_answer: str,
    effective_eos_token_ids: Sequence[int],
) -> None:
    if not isinstance(generation, CleanPrefixGeneration):
        raise TypeError("runtime point generation must be CleanPrefixGeneration")
    eos_ids = frozenset(effective_eos_token_ids)
    if not eos_ids:
        raise ValueError("runtime provenance contains no effective EOS token IDs")
    if generation.stop_reason == "eos_token":
        if generation.stop_token_id not in eos_ids:
            raise ValueError("generation stop token is outside the runtime effective EOS set")
        if any(token in eos_ids for token in generation.token_ids[:-1]):
            raise ValueError("generation continued after an effective EOS token")
    elif any(token in eos_ids for token in generation.token_ids):
        raise ValueError("max-token generation contains an effective EOS token")
    extraction = extract_with_fallback(
        generation.text,
        benchmark=benchmark,
        correct_answer=gold_answer,
        allow_positional=generation.stop_reason == "eos_token",
    )
    observed = (
        generation.value,
        generation.is_extracted,
        generation.is_correct,
        generation.method,
        generation.primary_method,
    )
    expected = (
        extraction.value,
        extraction.is_extracted,
        extraction.is_correct,
        extraction.method,
        extraction.primary_method,
    )
    if observed != expected:
        raise ValueError("runtime extraction does not match the generated answer text")


def _validate_point(
    point: CleanPrefixPointScan,
    *,
    item: _PlannedCase,
    k: int,
    benchmark: str,
    effective_eos_token_ids: Sequence[int],
) -> None:
    if not isinstance(point, CleanPrefixPointScan):
        raise TypeError("runtime.generate_point must return CleanPrefixPointScan")
    if item.plan is None:
        raise ValueError("cannot validate a point without a boundary-valid token plan")
    if point.k != k or point.input_use.k != k:
        raise ValueError("runtime point uses the wrong prefix budget")
    expected_ids = item.plan.input_ids(k)
    if point.input_use.input_ids != expected_ids:
        raise ValueError("runtime point input IDs do not match the token-exact plan")
    if point.input_use.prompt_token_count != len(item.plan.edited_prompt_ids):
        raise ValueError("runtime point prompt token count does not match the plan")
    if point.input_use.prefix_token_count != k:
        raise ValueError("runtime point prefix token count does not match k")
    gold = item.source.record.get("gold_answer")
    if not isinstance(gold, str) or not gold:
        raise ValueError("source pair gold_answer must be a non-empty string")
    _validate_generation(
        point.generation,
        benchmark=benchmark,
        gold_answer=gold,
        effective_eos_token_ids=effective_eos_token_ids,
    )


def _point_from_dict(value: object) -> CleanPrefixPointScan:
    row = _mapping(value, field="checkpoint point")
    if set(row) != {"k", "input_use", "generation"}:
        raise ValueError("checkpoint point has an unexpected shape")
    input_row = dict(_mapping(row.get("input_use"), field="checkpoint input_use"))
    generation_row = dict(_mapping(row.get("generation"), field="checkpoint generation"))
    if set(input_row) != {"k", "input_ids", "prompt_token_count", "prefix_token_count"}:
        raise ValueError("checkpoint input_use has an unexpected shape")
    if set(generation_row) != {
        "token_ids",
        "text",
        "value",
        "is_extracted",
        "is_correct",
        "method",
        "primary_method",
        "stop_reason",
        "stop_token_id",
    }:
        raise ValueError("checkpoint generation has an unexpected shape")
    input_ids = input_row.get("input_ids")
    token_ids = generation_row.get("token_ids")
    if not isinstance(input_ids, list) or not isinstance(token_ids, list):
        raise ValueError("checkpoint token IDs must be JSON lists")
    input_row["input_ids"] = tuple(input_ids)
    generation_row["token_ids"] = tuple(token_ids)
    return CleanPrefixPointScan(
        k=row.get("k"),  # type: ignore[arg-type]
        input_use=CleanPrefixInputUse(**input_row),  # type: ignore[arg-type]
        generation=CleanPrefixGeneration(**generation_row),  # type: ignore[arg-type]
    )


def _checkpoint_payload(
    item: _PlannedCase,
    points: Sequence[CleanPrefixPointScan],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    plan_sha256: str,
    runtime_sha256: str,
    grid: Sequence[PrefixBudget],
) -> dict[str, object]:
    return {
        "schema_version": "clean-prefix-scan-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "cohort": config.cohort,
        "targeting": item.source.targeting,
        "sample_id": item.source.sample_id,
        "source_cohort_sha256": source.cohort_sha256,
        "source_record_sha256": item.source.source_record_sha256,
        "plan_sha256": plan_sha256,
        "token_plan_sha256": _canonical_sha256(item.plan.to_dict()) if item.plan else None,
        "runtime_sha256": runtime_sha256,
        "budget_grid": [budget.to_dict() for budget in grid],
        "points": [point.to_dict() for point in points],
    }


def _load_checkpoint(
    path: Path,
    *,
    item: _PlannedCase,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    plan_sha256: str,
    runtime_sha256: str,
    grid: Sequence[PrefixBudget],
    effective_eos_token_ids: Sequence[int],
) -> tuple[CleanPrefixPointScan, ...]:
    payload = _load_json(path)
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("checkpoint points must be a list")
    points = tuple(_point_from_dict(value) for value in raw_points)
    expected_k = [budget.k for budget in grid]
    observed_k = [point.k for point in points]
    if observed_k != expected_k[: len(observed_k)]:
        raise ValueError("checkpoint points are not an ordered prefix of the budget grid")
    for point in points:
        _validate_point(
            point,
            item=item,
            k=point.k,
            benchmark=config.benchmark,
            effective_eos_token_ids=effective_eos_token_ids,
        )
    expected = _checkpoint_payload(
        item,
        points,
        config=config,
        source=source,
        plan_sha256=plan_sha256,
        runtime_sha256=runtime_sha256,
        grid=grid,
    )
    if _serialized_json(payload) != _serialized_json(expected):
        raise ValueError("checkpoint payload does not match reconstructed semantics")
    return points


def _checkpoint_entry(path: Path, item: _PlannedCase) -> dict[str, object]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "targeting": item.source.targeting,
        "sample_id": item.source.sample_id,
    }


def _checkpoint_registry_key(item: _PlannedCase) -> str:
    return _checkpoint_identity(item.source)


def _load_checkpoints(
    registry: Mapping[str, object],
    *,
    checkpoints_dir: Path,
    selected: Sequence[_PlannedCase],
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    plan_sha256: str,
    runtime_sha256: str,
    grids: Mapping[tuple[str, str], tuple[PrefixBudget, ...]],
    effective_eos_token_ids: Sequence[int],
) -> tuple[dict[tuple[str, str], tuple[CleanPrefixPointScan, ...]], dict[str, object]]:
    valid_items = {
        item.key: item for item in selected if item.candidate_eligible and item.plan is not None
    }
    points_by_key: dict[tuple[str, str], tuple[CleanPrefixPointScan, ...]] = {}
    normalized: dict[str, object] = {}
    expected_files: set[str] = set()
    for raw_identity, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"checkpoint registry {raw_identity}")
        if set(metadata) != {"file", "sha256", "targeting", "sample_id"}:
            raise ValueError("checkpoint registry has an invalid shape")
        key = (metadata.get("targeting"), metadata.get("sample_id"))
        if key not in valid_items:
            raise ValueError("checkpoint registry references an unknown executable pair")
        item = valid_items[key]  # type: ignore[index]
        identity = _checkpoint_registry_key(item)
        if raw_identity != identity:
            raise ValueError("checkpoint registry identity does not match its pair")
        path = _checkpoint_path(checkpoints_dir, item.source)
        expected_files.add(path.name)
        if (
            metadata.get("file") != path.name
            or not path.is_file()
            or metadata.get("sha256") != _sha256(path)
        ):
            raise ValueError("checkpoint file or SHA-256 does not match its registry")
        points_by_key[item.key] = _load_checkpoint(
            path,
            item=item,
            config=config,
            source=source,
            plan_sha256=plan_sha256,
            runtime_sha256=runtime_sha256,
            grid=grids[item.key],
            effective_eos_token_ids=effective_eos_token_ids,
        )
        normalized[identity] = dict(metadata)

    # Adopt a crash-left file only after full semantic validation.
    for item in valid_items.values():
        identity = _checkpoint_registry_key(item)
        if identity in normalized:
            continue
        path = _checkpoint_path(checkpoints_dir, item.source)
        if not path.is_file():
            continue
        expected_files.add(path.name)
        points_by_key[item.key] = _load_checkpoint(
            path,
            item=item,
            config=config,
            source=source,
            plan_sha256=plan_sha256,
            runtime_sha256=runtime_sha256,
            grid=grids[item.key],
            effective_eos_token_ids=effective_eos_token_ids,
        )
        normalized[identity] = _checkpoint_entry(path, item)
    actual_files = {path.name for path in checkpoints_dir.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("checkpoint directory contains unknown or missing files")
    return points_by_key, normalized


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _record(
    item: _PlannedCase,
    points: Sequence[CleanPrefixPointScan],
    grid: Sequence[PrefixBudget],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
) -> dict[str, object]:
    if item.plan is None or item.cot_token_count is None:
        raise ValueError("cannot publish a scan without a valid token plan")
    if [point.k for point in points] != [budget.k for budget in grid]:
        raise ValueError("cannot publish an incomplete prefix trajectory")
    trajectory = summarize_prefix_correctness(
        tuple((point.k, point.generation.is_correct) for point in points),
        cot_token_count=item.cot_token_count,
    )
    point_rows: list[dict[str, object]] = []
    for budget, point in zip(grid, points, strict=True):
        row = budget.to_dict()
        row["input_use"] = point.input_use.to_dict()
        row["generation"] = point.generation.to_dict()
        row["stable_through_later"] = trajectory.stable_at_k[point.k]
        point_rows.append(row)
    gold = item.source.record.get("gold_answer")
    return {
        "schema_version": "clean-prefix-scan-record/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "clean-prefix-scan",
        "model": config.model,
        "benchmark": config.benchmark,
        "cohort": config.cohort,
        "targeting": item.source.targeting,
        "sample_id": item.source.sample_id,
        "gold_answer": gold,
        "source": {
            "cohort_sha256": source.cohort_sha256,
            "source_record_sha256": item.source.source_record_sha256,
            "model_revision": source.model_revision,
        },
        "token_plan": {
            "sha256": _canonical_sha256(item.plan.to_dict()),
            "edited_prompt_token_count": len(item.plan.edited_prompt_ids),
            "clean_cot_token_count": item.cot_token_count,
            "boundary_valid": True,
            "input_assembly": INPUT_ASSEMBLY,
        },
        "points": point_rows,
        "trajectory": trajectory.to_dict(),
        "headline_denominator": not trajectory.k0_correct,
    }


def _status_rows(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    points_by_key: Mapping[tuple[str, str], Sequence[CleanPrefixPointScan]],
    grids: Mapping[tuple[str, str], tuple[PrefixBudget, ...]],
    *,
    config: CleanPrefixScanConfig,
) -> list[dict[str, object]]:
    full_keys = {item.key for item in selected_full}
    selected_keys = {item.key for item in selected}
    rows: list[dict[str, object]] = []
    for item in planned:
        points = tuple(points_by_key.get(item.key, ()))
        complete = bool(
            item.plan is not None and item.key in grids and len(points) == len(grids[item.key])
        )
        fresh_k0_correct = points[0].generation.is_correct if complete else None
        if not item.candidate_eligible:
            execution_status = "not-candidate"
        elif item.key not in full_keys:
            execution_status = "not-selected"
        elif item.key not in selected_keys:
            execution_status = "limit-truncated"
        elif item.boundary_valid is False:
            execution_status = "invalid-boundary"
        elif complete:
            execution_status = "completed"
        else:
            execution_status = "failed"
        rows.append(
            {
                "schema_version": "clean-prefix-scan-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "cohort": config.cohort,
                "targeting": item.source.targeting,
                "sample_id": item.source.sample_id,
                "source_record_sha256": item.source.source_record_sha256,
                "prepared_clean_correct": item.source.prepared_clean_correct,
                "prepared_edited_wrong": item.source.prepared_edited_wrong,
                "aligned_word_count": item.source.aligned_word_count,
                "candidate_eligible": item.candidate_eligible,
                "candidate_exclusion_reason": item.exclusion_reason,
                "cot_token_count": item.cot_token_count,
                "selected_by_full_plan": item.key in full_keys,
                "selected_for_execution": item.key in selected_keys,
                "boundary_valid": item.boundary_valid,
                "complete_scan": complete,
                "fresh_k0_correct": fresh_k0_correct,
                "headline_denominator": complete and fresh_k0_correct is False,
                "execution_status": execution_status,
            }
        )
    return rows


_TABLE10_REFERENCES: dict[tuple[str, str], tuple[int, int, int, int]] = {
    ("gemma-3-1b-it", "gsm8k"): (140, 136, 52, 50),
    ("gemma-3-1b-it", "mmlu"): (126, 77, 17, 61),
    ("gemma-3-1b-it", "arc"): (128, 60, 24, 51),
    ("gemma-3-4b-it", "gsm8k"): (172, 168, 85, 56),
    ("gemma-3-4b-it", "mmlu"): (127, 105, 38, 53),
    ("gemma-3-4b-it", "arc"): (132, 81, 26, 59),
    ("Llama-3.2-1B-Instruct", "gsm8k"): (140, 134, 41, 47),
    ("Llama-3.2-1B-Instruct", "mmlu"): (126, 87, 35, 43),
    ("Llama-3.2-1B-Instruct", "arc"): (136, 79, 46, 33),
    ("Llama-3.2-3B-Instruct", "gsm8k"): (133, 125, 48, 48),
    ("Llama-3.2-3B-Instruct", "mmlu"): (133, 108, 38, 48),
    ("Llama-3.2-3B-Instruct", "arc"): (137, 108, 61, 48),
    ("Mistral-7B-Instruct-v0.3", "gsm8k"): (126, 120, 41, 38),
    ("Mistral-7B-Instruct-v0.3", "mmlu"): (133, 107, 31, 42),
    ("Mistral-7B-Instruct-v0.3", "arc"): (141, 98, 39, 34),
}


def _historical_reference(config: CleanPrefixScanConfig) -> dict[str, object]:
    key = (config.model.rsplit("/", 1)[-1], config.benchmark)
    values = _TABLE10_REFERENCES.get(key)
    setting = None
    if values is not None:
        denominator, full, short, nonmono = values
        setting = {
            "fresh_k0_wrong": denominator,
            "full_correct": full,
            "stable_by_20_percent": short,
            "non_monotone": nonmono,
        }
    return {
        "table_10_setting": setting,
        "appendix_d_primary_absolute_point_correct_percent": {
            "denominator": 172,
            # Preserve the percentages exactly as printed.  In particular,
            # 8.7% identifies the submitted exact-clean-answer k=1 readout,
            # whereas the PDF's canonical-gold definition gives 16/172.
            "pdf_printed_percent_by_k": {
                "0": 0.0,
                "1": 8.7,
                "2": 23.3,
                "4": 24.4,
                "8": 36.6,
                "16": 44.8,
                "32": 59.9,
                "64": 75.9,
            },
        },
        "appendix_d_grid_sensitivity": {
            "denominator": 1858,
            "stable_by_20_percent": {
                "full": 537,
                "relative_only": 544,
                "absolute_only": 541,
                "half_relative_plus_absolute": 545,
                "sparse_relative_plus_absolute": 561,
            },
            "includes_primary": False,
        },
        "fourteen_extension_aggregate": {
            "selected": 2100,
            "valid_scans": 2094,
            "fresh_k0_wrong": 1858,
            "full_correct": 1425,
            "stable_by_20_percent": 537,
            "non_monotone": 655,
            "includes_primary": False,
        },
        "primary_k1_inconsistency": {
            "pdf_printed_legacy_exact_clean_answer": {"numerator": 15, "denominator": 172},
            "pdf_defined_canonical_gold_correctness": {"numerator": 16, "denominator": 172},
            "affected_sample": "lxt4/gsm8k_00272",
            "affects_table_10_full_short_nonmono": False,
        },
        "use_as_acceptance_target": False,
    }


def _summary(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    records: Sequence[Mapping[str, object]],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
) -> dict[str, object]:
    headline = [row for row in records if row.get("headline_denominator") is True]
    denominator = len(headline)
    trajectories = [
        _mapping(row.get("trajectory"), field="prefix record trajectory") for row in headline
    ]
    full = sum(trajectory.get("full_correct") is True for trajectory in trajectories)
    short = sum(trajectory.get("stable_le_0_2") is True for trajectory in trajectories)
    nonmono = sum(trajectory.get("non_monotone") is True for trajectory in trajectories)

    relative_metrics: list[dict[str, object]] = []
    for relative in config.relative_budgets:
        point_success = 0
        stable_success = 0
        for row, trajectory in zip(headline, trajectories, strict=True):
            token_plan = _mapping(row.get("token_plan"), field="prefix record token_plan")
            cot_count = int(token_plan["clean_cot_token_count"])
            k = round(relative * cot_count)
            points = row.get("points")
            if not isinstance(points, list):
                raise ValueError("prefix record points must be a list")
            point = next(raw for raw in points if isinstance(raw, Mapping) and raw.get("k") == k)
            generation = _mapping(point.get("generation"), field="prefix point generation")
            point_success += generation.get("is_correct") is True
            r_star = trajectory.get("r_star")
            stable_success += isinstance(r_star, (int, float)) and float(r_star) <= relative
        relative_metrics.append(
            {
                "relative_budget": relative,
                "point_correct": _rate(point_success, denominator),
                "stable_through_later": _rate(stable_success, denominator),
            }
        )

    absolute_metrics: list[dict[str, object]] = []
    for absolute in config.absolute_budgets:
        applicable: list[Mapping[str, object]] = []
        for row in headline:
            token_plan = _mapping(row.get("token_plan"), field="prefix record token_plan")
            if int(token_plan["clean_cot_token_count"]) >= absolute:
                applicable.append(row)
        point_success = 0
        for row in applicable:
            points = row.get("points")
            if not isinstance(points, list):
                raise ValueError("prefix record points must be a list")
            point = next(
                raw for raw in points if isinstance(raw, Mapping) and raw.get("k") == absolute
            )
            generation = _mapping(point.get("generation"), field="prefix point generation")
            point_success += generation.get("is_correct") is True
        # Exact point correctness is undefined when L_C < absolute.  Stable
        # recovery is not: k* <= absolute remains meaningful for every row in
        # the common fresh-k=0-wrong cohort, including shorter clean CoTs.
        stable_success = 0
        for trajectory in trajectories:
            k_star = trajectory.get("k_star")
            stable_success += isinstance(k_star, int) and k_star <= absolute
        absolute_metrics.append(
            {
                "absolute_budget": absolute,
                "point_correct": _rate(point_success, len(applicable)),
                "stable_through_later": _rate(stable_success, denominator),
                "exact_point_applicable": len(applicable),
                "shorter_than_budget": denominator - len(applicable),
            }
        )

    selected_counts = Counter(item.source.targeting for item in selected)
    eligible_counts = Counter(item.source.targeting for item in planned if item.candidate_eligible)
    stop_reasons: Counter[str] = Counter()
    unextractable = 0
    for row in records:
        points = row.get("points")
        if not isinstance(points, list):
            continue
        for point in points:
            generation = _mapping(
                _mapping(point, field="prefix point").get("generation"),
                field="prefix point generation",
            )
            stop_reasons[str(generation.get("stop_reason"))] += 1
            unextractable += generation.get("is_extracted") is False
    return {
        "schema_version": "clean-prefix-scan-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "clean-prefix-scan",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "cohort": config.cohort,
        },
        "protocol": _PROTOCOL,
        "counts": {
            "source_pairs": len(planned),
            "candidate_eligible": sum(item.candidate_eligible for item in planned),
            "selected_targets_before_limit": len(selected_full),
            "selected_targets": len(selected),
            "valid_scans": len(records),
            "fresh_k0_correct": len(records) - denominator,
            "fresh_k0_wrong": denominator,
        },
        "counts_by_targeting": {
            targeting: {
                "eligible": eligible_counts[targeting],
                "selected": selected_counts[targeting],
            }
            for targeting in _TARGETING_ORDER
        },
        "metrics": {
            "full_correct": _rate(full, denominator),
            "stable_by_20_percent": _rate(short, denominator),
            "non_monotone": _rate(nonmono, denominator),
            "relative_budgets": relative_metrics,
            "absolute_budgets": absolute_metrics,
        },
        "diagnostics": {
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "unextractable_points": unextractable,
        },
        "source": source.to_dict(),
        "historical_reference": _historical_reference(config),
        "comparability": _comparability(config, source, selected_full),
    }


def _comparability(
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    selected_full: Sequence[_PlannedCase],
) -> dict[str, object]:
    basename = config.model.rsplit("/", 1)[-1]
    paper_primary = bool(
        config.cohort == "primary"
        and basename == "gemma-3-4b-it"
        and config.benchmark == "gsm8k"
        and len(selected_full) == 172
    )
    paper_extension = bool(
        config.cohort == "extension"
        and basename
        in {
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
            "Mistral-7B-Instruct-v0.3",
        }
        and not (basename == "gemma-3-4b-it" and config.benchmark == "gsm8k")
        and len(selected_full) == 150
    )
    full_grid = config.relative_budgets == (
        0.0,
        0.02,
        0.05,
        0.08,
        0.12,
        0.16,
        0.2,
        0.25,
        0.325,
        0.4,
        0.5,
        0.65,
        0.8,
        1.0,
    ) and config.absolute_budgets == (1, 2, 4, 8, 16, 32, 64)
    limitations: list[str] = []
    if not full_grid:
        limitations.append("custom-prefix-budget-grid")
    if config.limit is not None:
        limitations.append("limit-truncated-smoke-run")
    if not (paper_primary or paper_extension):
        limitations.append("setting-or-denominator-differs-from-paper")
    return {
        "status": (
            "fresh-paper-protocol-run"
            if full_grid and config.limit is None and (paper_primary or paper_extension)
            else "partial-paper-protocol"
        ),
        "requirements": {
            "paper_primary_setting": paper_primary,
            "paper_extension_setting": paper_extension,
            "full_legacy_backed_grid": full_grid,
            "unlimited": config.limit is None,
            "source_cohort": source.cohort,
            "historical_cohort_identity": False,
        },
        "limitations": limitations,
        "primary_excluded_from_fourteen_setting_aggregate": True,
        "historical_cohort_identity": False,
    }


def _output_metadata(path: Path, *, records: int) -> dict[str, object]:
    return {"sha256": _sha256(path), "records": records}


def _counts(
    planned: Sequence[_PlannedCase],
    selected_full: Sequence[_PlannedCase],
    selected: Sequence[_PlannedCase],
    registry: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    return {
        "source_pairs": len(planned),
        "candidate_eligible": sum(item.candidate_eligible for item in planned),
        "selected_targets_before_limit": len(selected_full),
        "selected_targets": len(selected),
        "boundary_valid_selected": sum(item.plan is not None for item in selected),
        "checkpointed_pairs": len(registry),
        "failed_points": len(failures),
    }


def _manifest(
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    plan: Mapping[str, object],
    status: str,
    started_at: str,
    runtime: Mapping[str, object] | None,
    registry: Mapping[str, object],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "clean-prefix-scan-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "clean-prefix-scan",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "protocol_sha256": _canonical_sha256(_PROTOCOL),
        "source": source.to_dict(),
        "plan": dict(plan),
        "runtime": dict(runtime) if runtime is not None else None,
        "checkpoints": dict(registry),
        "counts": dict(counts),
        "failures": list(failures),
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_core(
    previous: Mapping[str, object],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
) -> str:
    if previous.get("schema_version") != "clean-prefix-scan-run/v1":
        raise ValueError("cannot resume an unknown clean-prefix run schema")
    if previous.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("resume paper SHA-256 does not match")
    if previous.get("operation") != "clean-prefix-scan":
        raise ValueError("resume operation does not match clean-prefix-scan")
    if previous.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("resume status is not recognized")
    if previous.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run")
    if previous.get("protocol") != _PROTOCOL or previous.get(
        "protocol_sha256"
    ) != _canonical_sha256(_PROTOCOL):
        raise ValueError("resume protocol fingerprint does not match")
    if previous.get("source") != source.to_dict():
        raise ValueError("resume source fingerprint does not match")
    started_at = previous.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _validate_output_hashes(
    manifest: Mapping[str, object],
    output_dir: Path,
) -> dict[str, Mapping[str, object]]:
    outputs = _mapping(manifest.get("outputs"), field="completed outputs")
    if set(outputs) != set(_PUBLIC_OUTPUT_NAMES):
        raise ValueError("completed output hash registry is incomplete")
    validated: dict[str, Mapping[str, object]] = {}
    for name in _PUBLIC_OUTPUT_NAMES:
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"completed output hash mismatch: {name}")
        records = metadata.get("records")
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            raise ValueError(f"completed output record count is invalid: {name}")
        validated[name] = metadata
    return validated


def _validate_completed_record(
    row: Mapping[str, object],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    effective_eos_token_ids: Sequence[int],
) -> tuple[_PlannedCase, tuple[CleanPrefixPointScan, ...], tuple[PrefixBudget, ...]]:
    key = (row.get("targeting"), row.get("sample_id"))
    by_key = {case.key: case for case in source.cases}
    if key not in by_key:
        raise ValueError("completed record is outside the source cohort")
    case = by_key[key]  # type: ignore[index]
    if (
        row.get("schema_version") != "clean-prefix-scan-record/v1"
        or row.get("paper_sha256") != PAPER_SHA256
        or row.get("operation") != "clean-prefix-scan"
        or row.get("model") != config.model
        or row.get("benchmark") != config.benchmark
        or row.get("cohort") != config.cohort
        or row.get("gold_answer") != case.record.get("gold_answer")
    ):
        raise ValueError("completed record setting or source identity is inconsistent")
    source_row = _mapping(row.get("source"), field="completed record source")
    if (
        source_row.get("cohort_sha256") != source.cohort_sha256
        or source_row.get("source_record_sha256") != case.source_record_sha256
        or source_row.get("model_revision") != source.model_revision
    ):
        raise ValueError("completed record source fingerprint is inconsistent")
    token_plan = _mapping(row.get("token_plan"), field="completed token_plan")
    prompt_count = token_plan.get("edited_prompt_token_count")
    cot_count = token_plan.get("clean_cot_token_count")
    if (
        not isinstance(prompt_count, int)
        or isinstance(prompt_count, bool)
        or prompt_count <= 0
        or not isinstance(cot_count, int)
        or isinstance(cot_count, bool)
        or not 8 <= cot_count <= 512
        or token_plan.get("boundary_valid") is not True
        or token_plan.get("input_assembly") != INPUT_ASSEMBLY
    ):
        raise ValueError("completed token plan metadata is invalid")
    raw_points = row.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("completed record points must be a list")
    grid = build_budget_grid(
        cot_count,
        relative_budgets=config.relative_budgets,
        absolute_budgets=config.absolute_budgets,
    )
    if len(raw_points) != len(grid):
        raise ValueError("completed record does not contain the full budget grid")
    points: list[CleanPrefixPointScan] = []
    prompt_ids: tuple[int, ...] | None = None
    full_prefix_ids: tuple[int, ...] | None = None
    for budget, raw_point in zip(grid, raw_points, strict=True):
        point_row = _mapping(raw_point, field="completed point")
        budget_fields = budget.to_dict()
        for field, expected in budget_fields.items():
            if point_row.get(field) != expected:
                raise ValueError("completed point budget origin is inconsistent")
        point = _point_from_dict(
            {
                "k": point_row.get("k"),
                "input_use": point_row.get("input_use"),
                "generation": point_row.get("generation"),
            }
        )
        gold = case.record.get("gold_answer")
        if not isinstance(gold, str) or not gold:
            raise ValueError("completed source gold answer is invalid")
        _validate_generation(
            point.generation,
            benchmark=config.benchmark,
            gold_answer=gold,
            effective_eos_token_ids=effective_eos_token_ids,
        )
        if point.input_use.prompt_token_count != prompt_count:
            raise ValueError("completed point prompt length changed")
        candidate_prompt = point.input_use.input_ids[:prompt_count]
        if prompt_ids is None:
            prompt_ids = candidate_prompt
        elif candidate_prompt != prompt_ids:
            raise ValueError("completed point edited prompt IDs changed across budgets")
        if point.k == cot_count:
            full_prefix_ids = point.input_use.input_ids[prompt_count:]
        points.append(point)
    if prompt_ids is None or full_prefix_ids is None or len(full_prefix_ids) != cot_count:
        raise ValueError("completed record lacks no-prefix or complete-prefix IDs")
    reconstructed_plan = PrefixInputPlan(prompt_ids, full_prefix_ids)
    if token_plan.get("sha256") != _canonical_sha256(reconstructed_plan.to_dict()):
        raise ValueError("completed token plan SHA-256 is inconsistent")
    for point in points:
        if point.input_use.input_ids != reconstructed_plan.input_ids(point.k):
            raise ValueError("completed point is not an exact clean-token ID prefix")
    trajectory = summarize_prefix_correctness(
        tuple((point.k, point.generation.is_correct) for point in points),
        cot_token_count=cot_count,
    )
    normalized_trajectory = json.loads(
        json.dumps(trajectory.to_dict(), ensure_ascii=False, allow_nan=False)
    )
    if row.get("trajectory") != normalized_trajectory:
        raise ValueError("completed trajectory metrics are inconsistent")
    if row.get("headline_denominator") is not (not trajectory.k0_correct):
        raise ValueError("completed fresh-k0 denominator flag is inconsistent")
    item = _PlannedCase(
        source=case,
        plan=reconstructed_plan,
        cot_token_count=cot_count,
        candidate_eligible=True,
        boundary_valid=True,
        exclusion_reason=None,
    )
    expected = _loads(
        json.dumps(
            _record(item, points, grid, config=config, source=source),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        context="reconstructed completed prefix record",
    )
    if row != expected:
        raise ValueError("completed record semantics contain missing or unexpected fields")
    return item, tuple(points), grid


def _completed_plan_keys(
    value: object,
    *,
    field: str,
    source_keys: set[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"completed plan {field} must be a list")
    keys: list[tuple[str, str]] = []
    for raw_key in value:
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 2
            or any(not isinstance(part, str) or not part for part in raw_key)
        ):
            raise ValueError(f"completed plan {field} contains an invalid pair key")
        key = (raw_key[0], raw_key[1])
        if key not in source_keys:
            raise ValueError(f"completed plan {field} references a pair outside the source")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"completed plan {field} contains duplicate pair keys")
    return tuple(keys)


def _completed_planned_cases(
    statuses: Sequence[Mapping[str, object]],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
    records_by_key: Mapping[
        tuple[str, str],
        tuple[_PlannedCase, tuple[CleanPrefixPointScan, ...], tuple[PrefixBudget, ...]],
    ],
) -> tuple[_PlannedCase, ...]:
    expected_fields = {
        "schema_version",
        "paper_sha256",
        "model",
        "benchmark",
        "cohort",
        "targeting",
        "sample_id",
        "source_record_sha256",
        "prepared_clean_correct",
        "prepared_edited_wrong",
        "aligned_word_count",
        "candidate_eligible",
        "candidate_exclusion_reason",
        "cot_token_count",
        "selected_by_full_plan",
        "selected_for_execution",
        "boundary_valid",
        "complete_scan",
        "fresh_k0_correct",
        "headline_denominator",
        "execution_status",
    }
    if len(statuses) != len(source.cases):
        raise ValueError("completed pair statuses do not cover the source cohort")
    planned: list[_PlannedCase] = []
    for row, case in zip(statuses, source.cases, strict=True):
        if set(row) != expected_fields:
            raise ValueError("completed pair status has an unexpected shape")
        if (
            row.get("schema_version") != "clean-prefix-scan-pair-status/v1"
            or row.get("paper_sha256") != PAPER_SHA256
            or row.get("model") != config.model
            or row.get("benchmark") != config.benchmark
            or row.get("cohort") != config.cohort
            or (row.get("targeting"), row.get("sample_id")) != case.key
            or row.get("source_record_sha256") != case.source_record_sha256
            or row.get("prepared_clean_correct") is not case.prepared_clean_correct
            or row.get("prepared_edited_wrong") is not case.prepared_edited_wrong
            or row.get("aligned_word_count") != case.aligned_word_count
        ):
            raise ValueError("completed pair status source identity is inconsistent")

        candidate = row.get("candidate_eligible")
        boundary = row.get("boundary_valid")
        cot_count = row.get("cot_token_count")
        exclusion = row.get("candidate_exclusion_reason")
        if type(candidate) is not bool or boundary not in {None, False, True}:
            raise ValueError("completed pair status candidate flags are invalid")
        source_exclusion = _source_eligibility(case)
        if source_exclusion is not None:
            if not (
                candidate is False
                and boundary is None
                and cot_count is None
                and exclusion == source_exclusion
            ):
                raise ValueError("completed pair status contradicts source eligibility")
        elif boundary is None:
            if not (
                candidate is False
                and cot_count is None
                and isinstance(exclusion, str)
                and exclusion.startswith("token-plan-invalid:")
            ):
                raise ValueError("completed token-plan exclusion is inconsistent")
        else:
            if not isinstance(cot_count, int) or isinstance(cot_count, bool) or cot_count < 0:
                raise ValueError("completed pair status clean-CoT length is invalid")
            expected_candidate = 8 <= cot_count <= 512
            expected_exclusion = (
                None
                if expected_candidate and boundary is True
                else (
                    "edited-prompt-boundary-invalid"
                    if expected_candidate
                    else "clean-pre-answer-length-outside-8-through-512"
                )
            )
            if candidate is not expected_candidate or exclusion != expected_exclusion:
                raise ValueError("completed pair status token boundary is inconsistent")

        completed = records_by_key.get(case.key)
        plan = None if completed is None else completed[0].plan
        if completed is not None and (
            boundary is not True
            or candidate is not True
            or cot_count != completed[0].cot_token_count
        ):
            raise ValueError("completed record contradicts its pair status")
        planned.append(
            _PlannedCase(
                source=case,
                plan=plan,
                cot_token_count=cot_count if isinstance(cot_count, int) else None,
                candidate_eligible=candidate,
                boundary_valid=boundary,  # type: ignore[arg-type]
                exclusion_reason=exclusion if isinstance(exclusion, str) else None,
            )
        )
    return tuple(planned)


def _completed_result(
    manifest: Mapping[str, object],
    *,
    config: CleanPrefixScanConfig,
    source: SourceBundle,
) -> CleanPrefixScanResult:
    try:
        metadata = _validate_output_hashes(manifest, config.output_dir)
        runtime = _validate_runtime_provenance(
            _mapping(manifest.get("runtime"), field="completed runtime"),
            config=config,
            source=source,
        )
        effective_eos_token_ids = tuple(runtime["effective_eos_token_ids"])
        records = _load_jsonl(config.output_dir / _PUBLIC_OUTPUT_NAMES[0])
        statuses = _load_jsonl(config.output_dir / _PUBLIC_OUTPUT_NAMES[1])
        summary = _load_json(config.output_dir / _PUBLIC_OUTPUT_NAMES[2])
        if len(records) != metadata[_PUBLIC_OUTPUT_NAMES[0]].get("records"):
            raise ValueError("completed prefix record count does not match its registry")
        if len(statuses) != metadata[_PUBLIC_OUTPUT_NAMES[1]].get("records"):
            raise ValueError("completed status record count does not match its registry")
        if metadata[_PUBLIC_OUTPUT_NAMES[2]].get("records") != 1:
            raise ValueError("completed summary registry must contain one document")

        completed: dict[
            tuple[str, str],
            tuple[_PlannedCase, tuple[CleanPrefixPointScan, ...], tuple[PrefixBudget, ...]],
        ] = {}
        for row in records:
            validated = _validate_completed_record(
                row,
                config=config,
                source=source,
                effective_eos_token_ids=effective_eos_token_ids,  # type: ignore[arg-type]
            )
            if validated[0].key in completed:
                raise ValueError("completed records contain a duplicate pair identity")
            completed[validated[0].key] = validated

        planned = _completed_planned_cases(
            statuses,
            config=config,
            source=source,
            records_by_key=completed,
        )
        source_keys = {item.key for item in planned}
        plan = _mapping(manifest.get("plan"), field="completed plan")
        if set(plan) != {
            "algorithm",
            "cases",
            "cases_sha256",
            "source_case_count",
            "eligible_case_count",
            "selected_full",
            "selected_full_sha256",
            "selected_for_execution",
            "selected_for_execution_sha256",
        }:
            raise ValueError("completed plan has an unexpected shape")
        selected_full_keys = _completed_plan_keys(
            plan.get("selected_full"),
            field="selected_full",
            source_keys=source_keys,
        )
        selected_keys = _completed_plan_keys(
            plan.get("selected_for_execution"),
            field="selected_for_execution",
            source_keys=source_keys,
        )
        raw_case_rows = plan.get("cases")
        if not isinstance(raw_case_rows, list) or len(raw_case_rows) != len(planned):
            raise ValueError("completed frozen plan rows do not cover the source cohort")
        frozen_plan_fields = {
            "cohort",
            "targeting",
            "sample_id",
            "source_record_sha256",
            "candidate_eligible",
            "boundary_valid",
            "cot_token_count",
            "exclusion_reason",
            "token_plan_sha256",
        }
        for raw_row, item in zip(raw_case_rows, planned, strict=True):
            frozen = _mapping(raw_row, field="completed frozen plan row")
            expected_metadata = {
                **item.source.identity_dict(),
                "candidate_eligible": item.candidate_eligible,
                "boundary_valid": item.boundary_valid,
                "cot_token_count": item.cot_token_count,
                "exclusion_reason": item.exclusion_reason,
            }
            if set(frozen) != frozen_plan_fields or any(
                frozen.get(field) != expected for field, expected in expected_metadata.items()
            ):
                raise ValueError("completed frozen plan row contradicts source or status")
            token_plan_sha256 = frozen.get("token_plan_sha256")
            if item.boundary_valid is True:
                completed_item = completed.get(item.key)
                expected_token_sha256 = (
                    None
                    if completed_item is None or completed_item[0].plan is None
                    else _canonical_sha256(completed_item[0].plan.to_dict())
                )
                if expected_token_sha256 is not None:
                    if token_plan_sha256 != expected_token_sha256:
                        raise ValueError("completed record does not match its frozen plan")
                elif (
                    not isinstance(token_plan_sha256, str)
                    or _SHA256.fullmatch(token_plan_sha256) is None
                ):
                    raise ValueError("completed frozen plan token fingerprint is invalid")
            elif token_plan_sha256 is not None:
                raise ValueError("boundary-invalid frozen plan must not claim token IDs")
        if config.cohort == "primary":
            expected_full_keys = tuple(item.key for item in planned)
        else:
            eligible_by_targeting = {
                targeting: [
                    item.source.sample_id
                    for item in planned
                    if item.candidate_eligible and item.source.targeting == targeting
                ]
                for targeting in _TARGETING_ORDER
            }
            expected_full_keys = select_extension_sample_ids(
                eligible_by_targeting,
                max_pairs=int(config.max_pairs),
                cap_per_targeting=400,
            )
        expected_selected_keys = (
            expected_full_keys if config.limit is None else expected_full_keys[: config.limit]
        )
        if (
            plan.get("algorithm") != "full-candidate-plan-before-limit/v1"
            or plan.get("source_case_count") != len(planned)
            or plan.get("eligible_case_count") != sum(item.candidate_eligible for item in planned)
            or plan.get("cases_sha256") != _canonical_sha256(raw_case_rows)
            or selected_full_keys != expected_full_keys
            or selected_keys != expected_selected_keys
            or plan.get("selected_full_sha256")
            != _canonical_sha256([list(key) for key in expected_full_keys])
            or plan.get("selected_for_execution_sha256")
            != _canonical_sha256([list(key) for key in expected_selected_keys])
        ):
            raise ValueError("completed frozen plan is semantically inconsistent")

        by_key = {item.key: item for item in planned}
        selected_full = tuple(by_key[key] for key in expected_full_keys)
        selected = tuple(by_key[key] for key in expected_selected_keys)
        expected_record_keys = tuple(
            item.key for item in selected if item.candidate_eligible and item.boundary_valid is True
        )
        if tuple(completed) != expected_record_keys:
            raise ValueError("completed records do not match the executable selected plan")
        points_by_key = {key: value[1] for key, value in completed.items()}
        grids = {key: value[2] for key, value in completed.items()}
        expected_statuses = _status_rows(
            planned,
            selected_full,
            selected,
            points_by_key,
            grids,
            config=config,
        )
        if statuses != expected_statuses:
            raise ValueError("completed pair statuses are semantically inconsistent")
        expected_summary = _loads(
            json.dumps(
                _summary(
                    planned,
                    selected_full,
                    selected,
                    records,
                    config=config,
                    source=source,
                ),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            context="reconstructed completed summary",
        )
        if summary != expected_summary:
            raise ValueError("completed summary is semantically inconsistent")

        registry = _mapping(manifest.get("checkpoints"), field="completed checkpoints")
        if registry:
            raise ValueError("completed run must not reference deleted work checkpoints")
        counts = _mapping(manifest.get("counts"), field="completed counts")
        if counts != _counts(planned, selected_full, selected, registry, []):
            raise ValueError("completed run counts are semantically inconsistent")
        expected_outputs = {
            config.output_dir.joinpath(name).name: _output_metadata(
                config.output_dir / name,
                records=(len(records) if name == _PUBLIC_OUTPUT_NAMES[0] else len(statuses)),
            )
            for name in _PUBLIC_OUTPUT_NAMES[:2]
        }
        expected_outputs[_PUBLIC_OUTPUT_NAMES[2]] = _output_metadata(
            config.output_dir / _PUBLIC_OUTPUT_NAMES[2],
            records=1,
        )
        if dict(metadata) != expected_outputs:
            raise ValueError("completed output registry metadata is semantically inconsistent")
        if manifest.get("failures") != []:
            raise ValueError("completed run retains failures")
    except (OSError, TypeError, ValueError) as exc:
        raise CleanPrefixScanRunError(
            f"completed run hash or semantic validation failed: {exc}"
        ) from exc
    return CleanPrefixScanResult(
        records_path=config.output_dir / _PUBLIC_OUTPUT_NAMES[0],
        pair_status_records_path=config.output_dir / _PUBLIC_OUTPUT_NAMES[1],
        summary_path=config.output_dir / _PUBLIC_OUTPUT_NAMES[2],
        run_path=config.output_dir / "run.json",
        source_pairs=len(source.cases),
        selected_pairs=len(selected),
        executed_pairs=len(records),
        records=len(records),
    )


def run_clean_prefix_scan(
    config: CleanPrefixScanConfig,
    *,
    runtime: CleanPrefixScanRuntime | None = None,
) -> CleanPrefixScanResult:
    """Validate, execute, checkpoint, and publish one clean-prefix setting."""

    output_dir = config.output_dir
    run_path = output_dir / "run.json"
    records_path = output_dir / _PUBLIC_OUTPUT_NAMES[0]
    statuses_path = output_dir / _PUBLIC_OUTPUT_NAMES[1]
    summary_path = output_dir / _PUBLIC_OUTPUT_NAMES[2]
    public_paths = (records_path, statuses_path, summary_path)
    work_dir = output_dir / ".clean-prefix-scan-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    source = load_source_bundle(
        cohort=config.cohort,
        model=config.model,
        benchmark=config.benchmark,
        fixed_window_run=config.fixed_window_run,
        pairs=config.pairs,
    )
    previous: dict[str, object] | None = None
    if config.resume:
        previous = _load_json(run_path)
        started_at = _validate_resume_core(previous, config=config, source=source)
        if previous.get("status") == "completed":
            return _completed_result(previous, config=config, source=source)
        for path in public_paths:
            path.unlink(missing_ok=True)
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
            for temporary in output_dir.glob(f".{path.name}.*.tmp"):
                temporary.unlink(missing_ok=True)
    else:
        started_at = _now()

    if runtime is None:
        runtime = _runtime_class()(config, revision=source.model_revision)
    active_runtime = _validate_runtime_provenance(
        runtime.provenance(),
        config=config,
        source=source,
    )
    if previous is not None:
        previous_runtime = _validate_runtime_provenance(
            _mapping(previous.get("runtime"), field="resume runtime"),
            config=config,
            source=source,
        )
        if active_runtime != previous_runtime:
            raise ValueError("resume runtime provenance differs from the original run")
        runtime_provenance = previous_runtime
    else:
        runtime_provenance = active_runtime
    runtime_sha256 = _canonical_sha256(runtime_provenance)
    effective_eos_token_ids = tuple(runtime_provenance["effective_eos_token_ids"])

    planned = _prepare_cases(source, runtime)
    selected_full, selected = _select_cases(planned, config)
    plan = _plan_payload(planned, selected_full, selected)
    plan_sha256 = _canonical_sha256(plan)
    if previous is not None and previous.get("plan") != plan:
        raise ValueError("resume deterministic plan fingerprint does not match")
    validate_source_snapshot(source)

    grids = {
        item.key: build_budget_grid(
            int(item.cot_token_count),
            relative_budgets=config.relative_budgets,
            absolute_budgets=config.absolute_budgets,
        )
        for item in selected
        if item.candidate_eligible and item.plan is not None and item.cot_token_count is not None
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previous_registry = (
        _mapping(previous.get("checkpoints"), field="resume checkpoints")
        if previous is not None
        else {}
    )
    points_by_key, registry = _load_checkpoints(
        previous_registry,
        checkpoints_dir=checkpoints_dir,
        selected=selected,
        config=config,
        source=source,
        plan_sha256=plan_sha256,
        runtime_sha256=runtime_sha256,
        grids=grids,
        effective_eos_token_ids=effective_eos_token_ids,  # type: ignore[arg-type]
    )
    failures: list[dict[str, object]] = []

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
                registry=registry,
                counts=_counts(planned, selected_full, selected, registry, failures),
                failures=failures,
            ),
        )

    write_progress("running")
    executable = tuple(
        item for item in selected if item.candidate_eligible and item.plan is not None
    )
    for item in tqdm(
        executable,
        desc="clean-prefix-scan",
        unit="pair",
        disable=None,
    ):
        grid = grids[item.key]
        points = list(points_by_key.get(item.key, ()))
        for budget in grid[len(points) :]:
            try:
                if item.plan is None:
                    raise AssertionError("executable prefix item has no token plan")
                gold_answer = item.source.record.get("gold_answer")
                if not isinstance(gold_answer, str) or not gold_answer:
                    raise ValueError("source pair gold_answer must be a non-empty string")
                point = runtime.generate_point(
                    item.plan,
                    budget.k,
                    gold_answer=gold_answer,
                )
                _validate_point(
                    point,
                    item=item,
                    k=budget.k,
                    benchmark=config.benchmark,
                    effective_eos_token_ids=effective_eos_token_ids,  # type: ignore[arg-type]
                )
                points.append(point)
                checkpoint_path = _checkpoint_path(checkpoints_dir, item.source)
                checkpoint = _checkpoint_payload(
                    item,
                    points,
                    config=config,
                    source=source,
                    plan_sha256=plan_sha256,
                    runtime_sha256=runtime_sha256,
                    grid=grid,
                )
                _write_json_atomic(checkpoint_path, checkpoint)
                registry[_checkpoint_registry_key(item)] = _checkpoint_entry(
                    checkpoint_path,
                    item,
                )
                points_by_key[item.key] = tuple(points)
                write_progress("running")
            except Exception as exc:
                failures.append(
                    {
                        "stage": "point",
                        "targeting": item.source.targeting,
                        "sample_id": item.source.sample_id,
                        "k": budget.k,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                write_progress("failed")
                raise CleanPrefixScanRunError(
                    "clean-prefix point failed for "
                    f"{item.source.targeting}/{item.source.sample_id} at k={budget.k}: {exc}; "
                    "checkpoints were retained for --resume"
                ) from exc

    for item in executable:
        if len(points_by_key.get(item.key, ())) != len(grids[item.key]):
            failures.append(
                {
                    "stage": "completion",
                    "targeting": item.source.targeting,
                    "sample_id": item.source.sample_id,
                    "error_type": "IncompleteTrajectory",
                    "message": "not every requested budget has a checkpoint",
                }
            )
    if failures:
        write_progress("failed")
        raise CleanPrefixScanRunError("not every selected valid scan completed")

    records = [
        _record(
            item,
            points_by_key[item.key],
            grids[item.key],
            config=config,
            source=source,
        )
        for item in selected
        if item.candidate_eligible and item.plan is not None
    ]
    statuses = _status_rows(
        planned,
        selected_full,
        selected,
        points_by_key,
        grids,
        config=config,
    )
    summary = _summary(
        planned,
        selected_full,
        selected,
        records,
        config=config,
        source=source,
    )

    try:
        validate_source_snapshot(source)
        for path in public_paths:
            path.unlink(missing_ok=True)
        _write_jsonl_atomic(records_path, records)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        outputs = {
            records_path.name: _output_metadata(records_path, records=len(records)),
            statuses_path.name: _output_metadata(statuses_path, records=len(statuses)),
            summary_path.name: _output_metadata(summary_path, records=1),
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
                registry={},
                counts=_counts(planned, selected_full, selected, {}, []),
                failures=[],
                outputs=outputs,
            ),
        )
    except BaseException as exc:
        for path in public_paths:
            path.unlink(missing_ok=True)
        failures.append(
            {
                "stage": "publication",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        try:
            write_progress("failed")
        except BaseException:
            pass
        if not isinstance(exc, Exception):
            raise
        raise CleanPrefixScanRunError(
            f"clean-prefix publication failed: {type(exc).__name__}: {exc}; "
            "checkpoints were retained for --resume"
        ) from exc

    with suppress(OSError):
        shutil.rmtree(work_dir)
    return CleanPrefixScanResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        source_pairs=len(planned),
        selected_pairs=len(selected),
        executed_pairs=len(records),
        records=len(records),
    )

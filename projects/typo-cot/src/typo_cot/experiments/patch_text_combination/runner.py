"""Strict, resumable writer for the final paper's Table 2 comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.fixed_window_answer_patching import LayerWindow
from typo_cot.experiments.fixed_window_answer_patching import runner as fixed_runner
from typo_cot.experiments.patch_coordinate_controls import runner as coordinate_runner
from typo_cot.experiments.patch_text_combination.planning import (
    PRE_ANSWER_BOUNDARY_METHOD,
    CompletePreAnswer,
    locate_complete_pre_answer,
)
from typo_cot.experiments.patch_text_combination.runtime import (
    HuggingFacePatchTextCombinationRuntime,
)

CELL_ORDER = (
    "patch-absent__text-none",
    "patch-present__text-none",
    "patch-absent__text-complete",
    "patch-present__text-complete",
)
_CELL_DEFINITIONS = {
    CELL_ORDER[0]: (False, "none", "fixed-window-reference"),
    CELL_ORDER[1]: (True, "none", "fixed-window-reference"),
    CELL_ORDER[2]: (False, "complete", "patch-text-runtime"),
    CELL_ORDER[3]: (True, "complete", "patch-text-runtime"),
}
_PAPER_WINDOW = LayerWindow(0, 6)
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLIC_OUTPUT_NAMES = (
    "patch_text_records.jsonl",
    "pair_status_records.jsonl",
    "patch_text_summary.json",
)
_PROTOCOL = {
    "schema_version": "patch-text-combination-protocol/v1",
    "source_operation": "fixed-window-answer-patching",
    "source_direction": "clean-to-edited",
    "source_window": "0:6",
    "denominator": "same-clean-correct-edited-wrong-pairs-in-all-four-cells",
    "cell_order": list(CELL_ORDER),
    "complete_text_source": "prepared-clean-continuation",
    "complete_text_boundary": PRE_ANSWER_BOUNDARY_METHOD,
    "no_trigger_policy": "retain-entire-continuation",
    "boundary_policy": "concatenated-text-tokenization-must-preserve-edited-prompt-prefix",
    "patch_site": "aligned-edited-word-final-token",
    "patch_value": "clean-question-only-complete-decoder-block-residual-output",
    "patch_application": "all-layers-in-0:6-during-prefill",
    "generation": "greedy-bfloat16-left-padding-max-new-tokens-512",
    "answer_extraction": "primary-then-empty-only-fallback/v1",
    "readout": "gold-answer-correctness-unextractable-is-failure",
    "aggregation": "four-descriptive-counts-and-rates",
    "checkpoint_policy": "all-baselines-before-pair-atomic-complete-text-cells",
    "design_status": "descriptive-same-pair-comparison",
}
_EXPECTED_TEXT_INTERVENTION = {
    "source": "prepared-clean-continuation",
    "boundary": PRE_ANSWER_BOUNDARY_METHOD,
    "recipient": "edited-prompt-plus-complete-clean-pre-answer-text",
    "donor": "clean-question-only-prompt",
    "tokenization": "single-concatenated-text-call-with-prompt-prefix-check",
}
_HISTORICAL_REFERENCE = {
    "total": 172,
    "cells": [
        {"cell": CELL_ORDER[0], "successes": 0, "total": 172, "rate": 0.0},
        {"cell": CELL_ORDER[1], "successes": 129, "total": 172, "rate": 0.75},
        {"cell": CELL_ORDER[2], "successes": 168, "total": 172, "rate": 168 / 172},
        {"cell": CELL_ORDER[3], "successes": 171, "total": 172, "rate": 171 / 172},
    ],
    "comparison": "descriptive-only-historical-cohort",
    "historical_cohort_identity": False,
    "note": (
        "Published values are metadata only; fresh corrected-alignment runs do not "
        "claim the unpublished historical sample identities."
    ),
}


class PatchTextCombinationRunError(RuntimeError):
    """Raised after preserving valid work from a failed patch/text run."""


@dataclass(frozen=True, slots=True)
class PatchTextCombinationConfig:
    """Frozen public arguments for the Table 2 comparison."""

    model: str
    benchmark: Literal["gsm8k"]
    fixed_window_run: Path
    layers: tuple[LayerWindow, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_window_run", Path(self.fixed_window_run))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark != "gsm8k":
            raise ValueError("benchmark must be gsm8k for the paper patch/text comparison")
        if len(self.layers) != 1:
            raise ValueError("layers must contain exactly one layer window")
        if not isinstance(self.layers[0], LayerWindow):
            raise TypeError("layers must contain only LayerWindow values")
        if self.layers[0] != _PAPER_WINDOW:
            raise ValueError("patch/text comparison requires the paper window 0:6")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""

        payload = asdict(self)
        payload["fixed_window_run"] = str(self.fixed_window_run.resolve())
        payload["layers"] = [window.label for window in self.layers]
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class CompleteTextInputUse:
    """Exact concatenated input and aligned coordinates used by the runtime."""

    pre_answer_text_sha256: str
    pre_answer_char_count: int
    pre_answer_token_count: int
    edited_prompt_token_count: int
    full_input_token_count: int
    full_input_ids_sha256: str
    clean_positions: tuple[int, ...]
    edited_positions: tuple[int, ...]
    boundary_stable: bool

    def __post_init__(self) -> None:
        for field, value in (
            ("pre_answer_text_sha256", self.pre_answer_text_sha256),
            ("full_input_ids_sha256", self.full_input_ids_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field} must be a full lowercase SHA-256")
        for field, value, minimum in (
            ("pre_answer_char_count", self.pre_answer_char_count, 0),
            ("pre_answer_token_count", self.pre_answer_token_count, 0),
            ("edited_prompt_token_count", self.edited_prompt_token_count, 1),
            ("full_input_token_count", self.full_input_token_count, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if (
            self.full_input_token_count
            != self.edited_prompt_token_count + self.pre_answer_token_count
        ):
            raise ValueError("complete-text token counts are inconsistent")
        for field, positions in (
            ("clean_positions", self.clean_positions),
            ("edited_positions", self.edited_positions),
        ):
            if (
                not isinstance(positions, tuple)
                or not positions
                or len(set(positions)) != len(positions)
                or any(
                    not isinstance(position, int) or isinstance(position, bool) or position < 0
                    for position in positions
                )
            ):
                raise ValueError(f"{field} must contain unique non-negative integers")
        if len(self.clean_positions) != len(self.edited_positions):
            raise ValueError("clean and edited coordinate cardinalities differ")
        if self.boundary_stable is not True:
            raise ValueError("complete-text input must preserve the edited prompt-token prefix")

    def to_dict(self) -> dict[str, object]:
        """Return the stable public input-use record."""

        payload = asdict(self)
        payload["clean_positions"] = list(self.clean_positions)
        payload["edited_positions"] = list(self.edited_positions)
        return payload


@dataclass(frozen=True, slots=True)
class CompleteTextScan:
    """Both freshly generated complete-text cells for one pair."""

    sample_id: str
    input_use: CompleteTextInputUse
    no_patch: fixed_runner.AnswerGeneration
    fixed_window_patch: fixed_runner.AnswerGeneration


class PatchTextCombinationRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def regenerate_baseline(self, pair: dict[str, object]) -> fixed_runner.BaselineScan: ...

    def scan_complete_text(
        self,
        pair: dict[str, object],
        pre_answer_text: str,
        window: LayerWindow,
    ) -> CompleteTextScan: ...


@dataclass(frozen=True, slots=True)
class PatchTextCombinationResult:
    """Published paths and record counts for one completed run."""

    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    records: int


@dataclass(frozen=True, slots=True)
class _PairPlan:
    reference: Any
    complete_text: CompletePreAnswer
    clean_positions: tuple[int, ...]
    edited_positions: tuple[int, ...]
    edited_prompt_token_count: int

    @property
    def key(self) -> tuple[str, str]:
        return self.reference.key

    @property
    def identity(self) -> str:
        return f"{self.key[0]}\0{self.key[1]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


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


def _serialized_json_sha256(payload: object) -> str:
    return hashlib.sha256(_serialized_json(payload).encode("utf-8")).hexdigest()


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
        if temporary.exists():
            temporary.unlink()


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
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, object]:
    return fixed_runner._load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            value = fixed_runner._strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _generation_record(generation: fixed_runner.AnswerGeneration) -> dict[str, object]:
    return fixed_runner._generation_record(generation)


def _generation_from_record(
    value: object,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
) -> fixed_runner.AnswerGeneration:
    return coordinate_runner._generation_from_record(
        value,
        field=field,
        benchmark=benchmark,
        gold_answer=gold_answer,
    )


def _baseline_record(baseline: fixed_runner.BaselineScan) -> dict[str, object]:
    return {
        "clean": _generation_record(baseline.clean),
        "edited": _generation_record(baseline.edited),
    }


def _validate_baseline(
    baseline: fixed_runner.BaselineScan,
    *,
    plan: _PairPlan,
    config: PatchTextCombinationConfig,
) -> None:
    if not isinstance(baseline, fixed_runner.BaselineScan):
        raise TypeError("runtime baseline result must be BaselineScan")
    if baseline.sample_id != plan.key[1]:
        raise ValueError("runtime baseline sample_id does not match its source pair")
    gold = _nonempty_string(plan.reference.source.record.get("gold_answer"), field="gold_answer")
    fixed_runner._validate_generation(
        baseline.clean,
        field="baseline.clean",
        benchmark=config.benchmark,
        gold_answer=gold,
    )
    fixed_runner._validate_generation(
        baseline.edited,
        field="baseline.edited",
        benchmark=config.benchmark,
        gold_answer=gold,
    )


def _pair_positions(source: Any) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    record = source.record
    aligned = record.get("aligned_words")
    if not isinstance(aligned, list) or len(aligned) != source.aligned_count or not aligned:
        raise ValueError(f"source aligned-word grid is invalid: {source.key}")
    clean_positions: list[int] = []
    edited_positions: list[int] = []
    for index, raw_word in enumerate(aligned):
        word = _mapping(raw_word, field=f"source aligned_words[{index}]")
        clean = word.get("clean_final_token")
        edited = word.get("edited_final_token")
        if (
            not isinstance(clean, int)
            or isinstance(clean, bool)
            or clean < 0
            or not isinstance(edited, int)
            or isinstance(edited, bool)
            or edited < 0
        ):
            raise ValueError(f"source aligned endpoint is invalid: {source.key}")
        clean_positions.append(clean)
        edited_positions.append(edited)
    if len(set(clean_positions)) != len(clean_positions) or len(set(edited_positions)) != len(
        edited_positions
    ):
        raise ValueError(f"source aligned endpoints are duplicated: {source.key}")
    edited_payload = _mapping(record.get("edited"), field="source edited")
    prompt_token_count = edited_payload.get("prompt_token_count")
    if (
        not isinstance(prompt_token_count, int)
        or isinstance(prompt_token_count, bool)
        or prompt_token_count <= 0
        or max(edited_positions) >= prompt_token_count
    ):
        raise ValueError(f"source edited prompt token count is invalid: {source.key}")
    return tuple(clean_positions), tuple(edited_positions), prompt_token_count


def _build_plans(reference: Any) -> tuple[_PairPlan, ...]:
    plans: list[_PairPlan] = []
    for pair in reference.pairs:
        if pair.correct.event is not pair.correct.generation.is_correct:
            raise ValueError(
                f"fixed-window restoration event and gold correctness disagree for {pair.key}"
            )
        clean_payload = _mapping(pair.source.record.get("clean"), field="source clean")
        continuation = clean_payload.get("continuation")
        if not isinstance(continuation, str):
            raise TypeError(f"source clean continuation is not a string: {pair.key}")
        complete_text = locate_complete_pre_answer(continuation)
        clean_positions, edited_positions, edited_prompt_token_count = _pair_positions(pair.source)
        plans.append(
            _PairPlan(
                reference=pair,
                complete_text=complete_text,
                clean_positions=clean_positions,
                edited_positions=edited_positions,
                edited_prompt_token_count=edited_prompt_token_count,
            )
        )
    if not plans:
        raise ValueError("fixed-window reference has an empty clean-to-edited denominator")
    return tuple(plans)


def _plan_rows(plans: Sequence[_PairPlan]) -> list[dict[str, object]]:
    return [
        {
            "targeting": plan.key[0],
            "sample_id": plan.key[1],
            "source_record_sha256": plan.reference.source.fingerprint,
            "pre_answer_text_sha256": plan.complete_text.sha256,
            "clean_positions": list(plan.clean_positions),
            "edited_positions": list(plan.edited_positions),
        }
        for plan in plans
    ]


def _manifest_input(config: PatchTextCombinationConfig, reference: Any) -> dict[str, object]:
    return {
        "fixed_window_run": str(reference.path),
        "fixed_window_run_sha256": reference.run_sha256,
        "fixed_window_denominator_sha256": reference.denominator_sha256,
        "fixed_window_outputs": {
            name: dict(metadata) for name, metadata in sorted(reference.output_metadata.items())
        },
        "model_revision": reference.sources.model_revision,
        "dataset_records_sha256": reference.sources.dataset_records_sha256,
        "requested_window": config.layers[0].label,
        "requested_direction": "clean-to-edited",
    }


def _manifest_plan(
    all_plans: Sequence[_PairPlan], selected: Sequence[_PairPlan]
) -> dict[str, object]:
    return {
        "reference_pairs": len(all_plans),
        "executed_pairs": len(selected),
        "full_fingerprint": _canonical_sha256(_plan_rows(all_plans)),
        "executed_fingerprint": _canonical_sha256(_plan_rows(selected)),
        "cell_order": list(CELL_ORDER),
    }


def _boundary_diagnostics(plans: Sequence[_PairPlan]) -> dict[str, object]:
    no_trigger = sum(not plan.complete_text.trigger_found for plan in plans)
    multiple_triggers = sum(plan.complete_text.trigger_count > 1 for plan in plans)
    empty_text = sum(not plan.complete_text.text for plan in plans)
    residual_fragment = sum(plan.complete_text.residual_fragment for plan in plans)
    anomalous = sum(
        not plan.complete_text.trigger_found
        or plan.complete_text.trigger_count > 1
        or not plan.complete_text.text
        or plan.complete_text.residual_fragment
        for plan in plans
    )
    return {
        "total_pairs": len(plans),
        "no_trigger_pairs": no_trigger,
        "multiple_trigger_pairs": multiple_triggers,
        "empty_text_pairs": empty_text,
        "residual_fragment_pairs": residual_fragment,
        "anomalous_pairs": anomalous,
    }


def _prepared_text_agreement(plans: Sequence[_PairPlan]) -> dict[str, int]:
    matching = sum(
        _mapping(plan.reference.source.record.get("clean"), field="source clean").get(
            "continuation"
        )
        == plan.reference.baseline.clean.text
        for plan in plans
    )
    return {"matching_pairs": matching, "total": len(plans)}


def _comparability(
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
) -> dict[str, object]:
    targetings = Counter(plan.key[0] for plan in all_plans)
    source_comparability = _mapping(
        reference.manifest.get("comparability", {}),
        field="fixed-window comparability",
    )
    source_status = source_comparability.get("status")
    source_arguments = _mapping(
        reference.manifest.get("arguments"),
        field="fixed-window arguments",
    )
    source_unlimited = source_arguments.get("limit") is None
    both_targeting_arms = set(targetings) == {"attribution-4", "random-4"}
    paper_pair_count = len(all_plans) == 172
    source_is_full_paper_protocol = source_status == "fresh-paper-protocol-run"
    boundary_diagnostics = _boundary_diagnostics(all_plans)
    boundary_is_unambiguous = boundary_diagnostics["anomalous_pairs"] == 0
    prepared_text_agreement = _prepared_text_agreement(all_plans)
    prepared_text_matches = (
        prepared_text_agreement["matching_pairs"] == prepared_text_agreement["total"]
    )
    limitations: list[str] = []
    if config.model != "google/gemma-3-4b-it":
        limitations.append("model-is-not-the-table-2-paper-model")
    if not both_targeting_arms:
        limitations.append("reference-denominator-does-not-contain-both-targeting-arms")
    if not paper_pair_count:
        limitations.append("reference-denominator-is-not-172-pairs")
    if not source_unlimited:
        limitations.append("fixed-window-reference-is-limit-truncated")
    if not source_is_full_paper_protocol:
        limitations.append("fixed-window-reference-is-not-a-full-paper-protocol-run")
    if not boundary_is_unambiguous:
        limitations.append("legacy-backed-complete-text-boundary-has-diagnostic-anomalies")
    if not prepared_text_matches:
        limitations.append("prepared-clean-continuation-differs-from-reference-baseline")
    if config.limit is not None:
        limitations.append("current-run-limit-is-smoke-only")
    if config.limit is not None:
        status = "partial-smoke-run"
    elif config.model != "google/gemma-3-4b-it":
        status = "non-paper-setting"
    elif limitations:
        status = "partial-paper-protocol"
    else:
        status = "fresh-paper-protocol-run"
    return {
        "status": status,
        "requirements": {
            "paper_model": config.model == "google/gemma-3-4b-it",
            "benchmark_gsm8k": config.benchmark == "gsm8k",
            "window_0:6": config.layers == (_PAPER_WINDOW,),
            "both_targeting_arms": both_targeting_arms,
            "table_2_pair_count_172": paper_pair_count,
            "fixed_window_reference_unlimited": source_unlimited,
            "fixed_window_reference_full_paper_protocol": source_is_full_paper_protocol,
            "current_run_unlimited": config.limit is None,
            "complete_text_boundary_unambiguous": boundary_is_unambiguous,
            "prepared_clean_continuations_match_reference_baselines": prepared_text_matches,
            "historical_cohort_identity": False,
        },
        "limitations": limitations,
        "reference_status": source_status,
        "denominator_by_targeting": dict(sorted(targetings.items())),
        "boundary_diagnostics": boundary_diagnostics,
        "prepared_text_reference_agreement": prepared_text_agreement,
        "historical_cohort_identity": False,
        "note": (
            "This run uses the exact verified fixed-window denominator but does not "
            "claim the unpublished historical 172 sample identities."
        ),
    }


def _paired_runtime_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in provenance.items()
        if key not in {"operation", "runtime", "text_intervention"}
    }


def _validate_runtime_provenance(
    provenance_value: object,
    *,
    reference: Any,
    config: PatchTextCombinationConfig,
) -> dict[str, object]:
    provenance = dict(_mapping(provenance_value, field="runtime provenance"))
    num_layers = provenance.get("num_decoder_layers")
    if (
        not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or num_layers != reference.num_layers
    ):
        raise ValueError("runtime provenance decoder depth does not match the reference")
    if config.layers[0].stop > num_layers:
        raise ValueError("patch/text window is outside the runtime decoder")
    revision = reference.sources.model_revision
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if provenance.get(field) != revision:
            raise ValueError(f"runtime {field} does not match pair preparation")
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in (
        ("do_sample", False),
        ("max_new_tokens", 512),
        ("use_cache", True),
        ("padding_side", "left"),
    ):
        if generation.get(field) != expected or type(generation.get(field)) is not type(expected):
            raise ValueError(f"runtime generation.{field} must be {expected!r}")
    text_intervention = _mapping(
        provenance.get("text_intervention"),
        field="runtime text_intervention",
    )
    if dict(text_intervention) != _EXPECTED_TEXT_INTERVENTION:
        raise ValueError("runtime complete-text provenance does not match the public protocol")
    if _paired_runtime_provenance(provenance) != _paired_runtime_provenance(reference.runtime):
        raise ValueError(
            "patch/text runtime provenance differs from the fixed-window reference; "
            "all four cells must use one execution environment"
        )
    return provenance


def _validate_runtime(
    runtime: PatchTextCombinationRuntime,
    *,
    reference: Any,
    config: PatchTextCombinationConfig,
) -> tuple[dict[str, object], str]:
    if (
        not isinstance(runtime.num_layers, int)
        or isinstance(runtime.num_layers, bool)
        or runtime.num_layers <= 0
    ):
        raise TypeError("runtime num_layers must be a positive integer")
    if runtime.num_layers != reference.num_layers:
        raise ValueError(
            "patch/text runtime decoder depth does not match the fixed-window reference"
        )
    if config.layers[0].stop > runtime.num_layers:
        raise ValueError("patch/text window is outside the runtime decoder")
    provenance = dict(runtime.provenance())
    if provenance.get("num_decoder_layers") != runtime.num_layers:
        raise ValueError("runtime provenance num_decoder_layers does not match runtime")
    provenance = _validate_runtime_provenance(
        provenance,
        reference=reference,
        config=config,
    )
    return provenance, _canonical_sha256(provenance)


def _checkpoint_path(directory: Path, plan: _PairPlan) -> Path:
    digest = hashlib.sha256(plan.identity.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _registry_entry(path: Path, plan: _PairPlan) -> dict[str, str]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
    }


def _baseline_checkpoint(
    plan: _PairPlan,
    baseline: fixed_runner.BaselineScan,
    *,
    runtime_fingerprint: str,
    reference: Any,
) -> dict[str, object]:
    return {
        "schema_version": "patch-text-combination-baseline-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "fixed_window_run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "baseline": _baseline_record(baseline),
    }


def _input_use_from_record(value: object) -> CompleteTextInputUse:
    row = _mapping(value, field="checkpoint complete_text_input")
    clean_positions = row.get("clean_positions")
    edited_positions = row.get("edited_positions")
    if not isinstance(clean_positions, list) or not isinstance(edited_positions, list):
        raise TypeError("checkpoint complete-text coordinates must be lists")
    return CompleteTextInputUse(
        pre_answer_text_sha256=row.get("pre_answer_text_sha256"),  # type: ignore[arg-type]
        pre_answer_char_count=row.get("pre_answer_char_count"),  # type: ignore[arg-type]
        pre_answer_token_count=row.get("pre_answer_token_count"),  # type: ignore[arg-type]
        edited_prompt_token_count=row.get("edited_prompt_token_count"),  # type: ignore[arg-type]
        full_input_token_count=row.get("full_input_token_count"),  # type: ignore[arg-type]
        full_input_ids_sha256=row.get("full_input_ids_sha256"),  # type: ignore[arg-type]
        clean_positions=tuple(clean_positions),
        edited_positions=tuple(edited_positions),
        boundary_stable=row.get("boundary_stable"),  # type: ignore[arg-type]
    )


def _validate_input_use(input_use: CompleteTextInputUse, plan: _PairPlan) -> None:
    if not isinstance(input_use, CompleteTextInputUse):
        raise TypeError("runtime complete-text input use must be CompleteTextInputUse")
    if input_use.pre_answer_text_sha256 != plan.complete_text.sha256:
        raise ValueError("runtime complete-text fingerprint does not match the pure text plan")
    if input_use.pre_answer_char_count != len(plan.complete_text.text):
        raise ValueError("runtime complete-text character count does not match the pure plan")
    if (
        input_use.clean_positions != plan.clean_positions
        or input_use.edited_positions != plan.edited_positions
    ):
        raise ValueError("runtime complete-text coordinates do not match the source plan")
    if input_use.edited_prompt_token_count != plan.edited_prompt_token_count:
        raise ValueError("runtime edited prompt-token count does not match the source plan")


def _complete_checkpoint(
    plan: _PairPlan,
    scan: CompleteTextScan,
    *,
    config: PatchTextCombinationConfig,
    runtime_fingerprint: str,
    reference: Any,
) -> dict[str, object]:
    if not isinstance(scan, CompleteTextScan):
        raise TypeError("runtime complete-text result must be CompleteTextScan")
    if scan.sample_id != plan.key[1]:
        raise ValueError("runtime complete-text sample_id does not match its source pair")
    _validate_input_use(scan.input_use, plan)
    gold = _nonempty_string(plan.reference.source.record.get("gold_answer"), field="gold_answer")
    for field, generation in (
        ("no_patch", scan.no_patch),
        ("fixed_window_patch", scan.fixed_window_patch),
    ):
        fixed_runner._validate_generation(
            generation,
            field=f"complete_text.{field}",
            benchmark=config.benchmark,
            gold_answer=gold,
        )
    return {
        "schema_version": "patch-text-combination-complete-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "fixed_window_run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "window": config.layers[0].label,
        "complete_text": plan.complete_text.to_dict(),
        "input_use": scan.input_use.to_dict(),
        "no_patch": _generation_record(scan.no_patch),
        "fixed_window_patch": _generation_record(scan.fixed_window_patch),
    }


def _load_baseline_checkpoint(
    path: Path,
    *,
    plan: _PairPlan,
    config: PatchTextCombinationConfig,
    runtime_fingerprint: str,
    reference: Any,
) -> fixed_runner.BaselineScan:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "patch-text-combination-baseline-checkpoint/v1"
        or checkpoint.get("paper_sha256") != PAPER_SHA256
        or checkpoint.get("targeting") != plan.key[0]
        or checkpoint.get("sample_id") != plan.key[1]
        or checkpoint.get("source_record_sha256") != plan.reference.source.fingerprint
        or checkpoint.get("fixed_window_run_sha256") != reference.run_sha256
        or checkpoint.get("denominator_sha256") != reference.denominator_sha256
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
    ):
        raise ValueError(f"baseline checkpoint provenance does not match: {path}")
    baseline = coordinate_runner._baseline_from_record(
        checkpoint.get("baseline"),
        sample_id=plan.key[1],
        benchmark=config.benchmark,
        gold_answer=_nonempty_string(
            plan.reference.source.record.get("gold_answer"),
            field="gold_answer",
        ),
    )
    _validate_baseline(baseline, plan=plan, config=config)
    if baseline != plan.reference.baseline:
        raise ValueError(f"baseline checkpoint does not match the fixed-window reference: {path}")
    return baseline


def _load_complete_checkpoint(
    path: Path,
    *,
    plan: _PairPlan,
    config: PatchTextCombinationConfig,
    runtime_fingerprint: str,
    reference: Any,
) -> CompleteTextScan:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "patch-text-combination-complete-checkpoint/v1"
        or checkpoint.get("paper_sha256") != PAPER_SHA256
        or checkpoint.get("targeting") != plan.key[0]
        or checkpoint.get("sample_id") != plan.key[1]
        or checkpoint.get("source_record_sha256") != plan.reference.source.fingerprint
        or checkpoint.get("fixed_window_run_sha256") != reference.run_sha256
        or checkpoint.get("denominator_sha256") != reference.denominator_sha256
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("window") != config.layers[0].label
        or checkpoint.get("complete_text") != plan.complete_text.to_dict()
    ):
        raise ValueError(f"complete-text checkpoint provenance does not match: {path}")
    input_use = _input_use_from_record(checkpoint.get("input_use"))
    gold = _nonempty_string(plan.reference.source.record.get("gold_answer"), field="gold_answer")
    scan = CompleteTextScan(
        sample_id=plan.key[1],
        input_use=input_use,
        no_patch=_generation_from_record(
            checkpoint.get("no_patch"),
            field="checkpoint no_patch",
            benchmark=config.benchmark,
            gold_answer=gold,
        ),
        fixed_window_patch=_generation_from_record(
            checkpoint.get("fixed_window_patch"),
            field="checkpoint fixed_window_patch",
            benchmark=config.benchmark,
            gold_answer=gold,
        ),
    )
    _validate_input_use(scan.input_use, plan)
    return scan


def _empty_registries() -> dict[str, dict[str, dict[str, str]]]:
    return {"baselines": {}, "complete_text": {}}


def _load_registries(
    manifest: Mapping[str, object],
    *,
    selected: Sequence[_PairPlan],
    baseline_dir: Path,
    complete_dir: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    raw = _mapping(manifest.get("checkpoints"), field="resume checkpoints")
    if set(raw) != {"baselines", "complete_text"}:
        raise ValueError("resume checkpoint registry has unexpected groups")
    plans_by_identity = {plan.identity: plan for plan in selected}
    registries = _empty_registries()
    for group, directory in (("baselines", baseline_dir), ("complete_text", complete_dir)):
        entries = _mapping(raw.get(group), field=f"resume checkpoints.{group}")
        for identity, raw_metadata in entries.items():
            if not isinstance(identity, str) or identity not in plans_by_identity:
                raise ValueError(f"resume {group} registry contains an unknown pair")
            metadata = _mapping(raw_metadata, field=f"resume checkpoint {identity}")
            if set(metadata) != {"file", "sha256", "targeting", "sample_id"}:
                raise ValueError(f"resume {group} checkpoint metadata is malformed")
            plan = plans_by_identity[identity]
            expected_path = _checkpoint_path(directory, plan)
            if (
                metadata.get("file") != expected_path.name
                or metadata.get("targeting") != plan.key[0]
                or metadata.get("sample_id") != plan.key[1]
                or not expected_path.is_file()
                or metadata.get("sha256") != _sha256(expected_path)
            ):
                raise ValueError(f"registered {group} checkpoint hash does not match: {identity}")
            registries[group][identity] = dict(metadata)  # type: ignore[arg-type]
    if not set(registries["complete_text"]).issubset(registries["baselines"]):
        raise ValueError("complete-text checkpoints exist without baseline checkpoints")
    return registries


def _cleanup_orphans(
    directory: Path,
    registry: Mapping[str, Mapping[str, str]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected = {metadata["file"] for metadata in registry.values()}
    for child in directory.iterdir():
        if child.is_file() and child.name not in expected:
            child.unlink()


def _base_manifest(
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object] | None,
    registries: Mapping[str, Mapping[str, Mapping[str, str]]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "patch-text-combination-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-text-combination",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "input": _manifest_input(config, reference),
        "plan": _manifest_plan(all_plans, selected),
        "runtime": dict(runtime_provenance) if runtime_provenance is not None else None,
        "checkpoints": {
            group: {identity: dict(metadata) for identity, metadata in sorted(entries.items())}
            for group, entries in sorted(registries.items())
        },
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": _comparability(config, reference, all_plans),
        "reference": {
            "operation": "fixed-window-answer-patching",
            "run_sha256": reference.run_sha256,
            "denominator_sha256": reference.denominator_sha256,
            "historical_cohort_identity": False,
        },
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
) -> str:
    if manifest.get("schema_version") != "patch-text-combination-run/v1":
        raise ValueError("cannot resume an unknown patch/text run schema")
    if manifest.get("operation") != "patch-text-combination":
        raise ValueError("resume run has the wrong operation")
    if manifest.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("resume run has an invalid status")
    if manifest.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    if manifest.get("paper_sha256") != PAPER_SHA256 or manifest.get("protocol") != _PROTOCOL:
        raise ValueError("resume paper or protocol fingerprint does not match")
    if manifest.get("input") != _manifest_input(config, reference):
        raise ValueError("resume fixed-window input fingerprint does not match")
    if manifest.get("plan") != _manifest_plan(all_plans, selected):
        raise ValueError("resume deterministic plan fingerprint does not match")
    if manifest.get("comparability") != _comparability(config, reference, all_plans):
        raise ValueError("resume comparability label does not match")
    reference_payload = _mapping(manifest.get("reference"), field="resume reference")
    if reference_payload != {
        "operation": "fixed-window-answer-patching",
        "run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "historical_cohort_identity": False,
    }:
        raise ValueError("resume reference lineage does not match")
    return _nonempty_string(manifest.get("started_at"), field="resume started_at")


def _result(output_dir: Path, *, pairs: int, records: int) -> PatchTextCombinationResult:
    return PatchTextCombinationResult(
        records_path=output_dir / _PUBLIC_OUTPUT_NAMES[0],
        pair_status_records_path=output_dir / _PUBLIC_OUTPUT_NAMES[1],
        summary_path=output_dir / _PUBLIC_OUTPUT_NAMES[2],
        run_path=output_dir / "run.json",
        pairs=pairs,
        records=records,
    )


def _validate_completed_checkpoint_registry(
    manifest: Mapping[str, object],
    *,
    selected: Sequence[_PairPlan],
    expected_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    checkpoints = _mapping(manifest.get("checkpoints"), field="completed checkpoints")
    if set(checkpoints) != {"baselines", "complete_text"}:
        raise ValueError("completed checkpoint registry has unexpected groups")
    expected_identities = {plan.identity for plan in selected}
    plans_by_identity = {plan.identity: plan for plan in selected}
    for group in ("baselines", "complete_text"):
        group_hashes = _mapping(
            expected_hashes.get(group),
            field=f"expected completed checkpoints.{group}",
        )
        if set(group_hashes) != expected_identities:
            raise ValueError(f"expected {group} hashes do not cover the executed plan")
        entries = _mapping(checkpoints.get(group), field=f"completed checkpoints.{group}")
        if set(entries) != expected_identities:
            raise ValueError(f"completed {group} registry does not cover the executed plan")
        for identity, raw_metadata in entries.items():
            plan = plans_by_identity[identity]
            metadata = _mapping(raw_metadata, field=f"completed checkpoint {identity}")
            expected_file = _checkpoint_path(Path("checkpoint"), plan).name
            if (
                set(metadata) != {"file", "sha256", "targeting", "sample_id"}
                or metadata.get("file") != expected_file
                or not isinstance(metadata.get("sha256"), str)
                or _SHA256.fullmatch(metadata["sha256"]) is None  # type: ignore[arg-type]
                or metadata.get("sha256") != group_hashes[identity]
                or metadata.get("targeting") != plan.key[0]
                or metadata.get("sample_id") != plan.key[1]
            ):
                raise ValueError(f"completed {group} checkpoint lineage is malformed: {identity}")


def _validate_completed_public_outputs(
    *,
    output_dir: Path,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
    runtime_fingerprint: str,
) -> dict[str, dict[str, str]]:
    records = _load_jsonl(output_dir / _PUBLIC_OUTPUT_NAMES[0])
    expected_records = len(selected) * len(CELL_ORDER)
    if len(records) != expected_records:
        raise ValueError("completed cell record count does not match the executed plan")
    input_uses: dict[str, CompleteTextInputUse] = {}
    complete_generations: dict[
        str,
        dict[str, fixed_runner.AnswerGeneration],
    ] = {}
    for index, row in enumerate(records):
        plan = selected[index // len(CELL_ORDER)]
        cell = CELL_ORDER[index % len(CELL_ORDER)]
        gold = _nonempty_string(
            plan.reference.source.record.get("gold_answer"),
            field="gold_answer",
        )
        if _CELL_DEFINITIONS[cell][1] == "complete":
            generation = _generation_from_record(
                row.get("answer"),
                field=f"completed cell record {index}.answer",
                benchmark=config.benchmark,
                gold_answer=gold,
            )
            input_use = _input_use_from_record(row.get("input_use"))
            _validate_input_use(input_use, plan)
            prior = input_uses.setdefault(plan.identity, input_use)
            if prior != input_use:
                raise ValueError("the two complete-text cells use different tokenized inputs")
            complete_generations.setdefault(plan.identity, {})[cell] = generation
        else:
            generation = (
                plan.reference.baseline.edited
                if cell == CELL_ORDER[0]
                else plan.reference.correct.generation
            )
            input_use = None
        expected = _cell_record(
            config=config,
            reference=reference,
            plan=plan,
            cell=cell,
            generation=generation,
            input_use=input_use,
        )
        if row != expected:
            raise ValueError(f"completed cell record {index} is semantically inconsistent")

    statuses = _load_jsonl(output_dir / _PUBLIC_OUTPUT_NAMES[1])
    if len(statuses) != len(all_plans):
        raise ValueError("completed pair-status count does not match the reference plan")
    selected_identities = {plan.identity for plan in selected}
    for index, (row, plan) in enumerate(zip(statuses, all_plans, strict=True)):
        selected_for_execution = plan.identity in selected_identities
        expected = _pair_status_record(
            config=config,
            reference=reference,
            plan=plan,
            selected_for_execution=selected_for_execution,
            input_use=input_uses.get(plan.identity),
        )
        if row != expected:
            raise ValueError(f"completed pair-status record {index} is semantically inconsistent")

    summary = _load_json(output_dir / _PUBLIC_OUTPUT_NAMES[2])
    expected_summary = _summary_payload(
        config=config,
        reference=reference,
        all_plans=all_plans,
        selected=selected,
        records=records,
    )
    if summary != expected_summary:
        raise ValueError("completed summary is semantically inconsistent with public records")

    expected_hashes: dict[str, dict[str, str]] = {
        "baselines": {},
        "complete_text": {},
    }
    for plan in selected:
        generations = complete_generations.get(plan.identity, {})
        if set(generations) != {CELL_ORDER[2], CELL_ORDER[3]}:
            raise ValueError("completed records do not reconstruct one complete-text checkpoint")
        scan = CompleteTextScan(
            sample_id=plan.key[1],
            input_use=input_uses[plan.identity],
            no_patch=generations[CELL_ORDER[2]],
            fixed_window_patch=generations[CELL_ORDER[3]],
        )
        baseline_payload = _baseline_checkpoint(
            plan,
            plan.reference.baseline,
            runtime_fingerprint=runtime_fingerprint,
            reference=reference,
        )
        complete_payload = _complete_checkpoint(
            plan,
            scan,
            config=config,
            runtime_fingerprint=runtime_fingerprint,
            reference=reference,
        )
        expected_hashes["baselines"][plan.identity] = _serialized_json_sha256(baseline_payload)
        expected_hashes["complete_text"][plan.identity] = _serialized_json_sha256(complete_payload)
    return expected_hashes


def _completed_result(
    manifest: Mapping[str, object],
    *,
    output_dir: Path,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
) -> PatchTextCombinationResult:
    try:
        if manifest.get("status") != "completed" or manifest.get("failures") != []:
            raise ValueError("completed manifest status or failures are inconsistent")
        _nonempty_string(manifest.get("started_at"), field="completed started_at")
        _nonempty_string(manifest.get("updated_at"), field="completed updated_at")
        runtime_provenance = _validate_runtime_provenance(
            manifest.get("runtime"),
            reference=reference,
            config=config,
        )
        runtime_fingerprint = _canonical_sha256(runtime_provenance)
        outputs = _mapping(manifest.get("outputs"), field="completed outputs")
        if set(outputs) != set(_PUBLIC_OUTPUT_NAMES):
            raise ValueError("completed output registry has unexpected entries")
        expected_counts = {
            _PUBLIC_OUTPUT_NAMES[0]: len(selected) * len(CELL_ORDER),
            _PUBLIC_OUTPUT_NAMES[1]: len(all_plans),
            _PUBLIC_OUTPUT_NAMES[2]: 1,
        }
        for name, expected_count in expected_counts.items():
            metadata = _mapping(outputs.get(name), field=f"completed output {name}")
            path = output_dir / name
            if set(metadata) != {"sha256", "records"}:
                raise ValueError(f"completed output metadata is malformed: {name}")
            if not path.is_file() or metadata.get("sha256") != _sha256(path):
                raise ValueError(f"completed output hash does not match: {name}")
            if metadata.get("records") != expected_count:
                raise ValueError(f"completed output record count does not match: {name}")
        counts = _mapping(manifest.get("counts"), field="completed counts")
        expected_manifest_counts = {
            "reference_pairs": len(all_plans),
            "executed_pairs": len(selected),
            "baseline_checkpoints": len(selected),
            "complete_text_checkpoints": len(selected),
            "failed_pairs": 0,
            "records": len(selected) * len(CELL_ORDER),
        }
        if dict(counts) != expected_manifest_counts:
            raise ValueError("completed counts do not match the deterministic plan")
        expected_checkpoint_hashes = _validate_completed_public_outputs(
            output_dir=output_dir,
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            runtime_fingerprint=runtime_fingerprint,
        )
        _validate_completed_checkpoint_registry(
            manifest,
            selected=selected,
            expected_hashes=expected_checkpoint_hashes,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PatchTextCombinationRunError(f"completed resume validation failed: {exc}") from exc
    return _result(
        output_dir,
        pairs=len(selected),
        records=len(selected) * len(CELL_ORDER),
    )


def _remove_public_outputs(output_dir: Path) -> None:
    for name in _PUBLIC_OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _cell_record(
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    plan: _PairPlan,
    cell: str,
    generation: fixed_runner.AnswerGeneration,
    input_use: CompleteTextInputUse | None,
) -> dict[str, object]:
    patch_present, clean_text, result_source = _CELL_DEFINITIONS[cell]
    return {
        "schema_version": "patch-text-combination-cell/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-text-combination",
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "fixed_window_run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "denominator": "fixed-window-clean-to-edited",
        "cell": cell,
        "patch_present": patch_present,
        "clean_text": clean_text,
        "result_source": result_source,
        "direction": "clean-to-edited" if patch_present else None,
        "window": config.layers[0].label if patch_present else None,
        "complete_text": plan.complete_text.to_dict() if clean_text == "complete" else None,
        "input_use": input_use.to_dict() if input_use is not None else None,
        "event": generation.is_correct,
        "answer": _generation_record(generation),
    }


def _pair_status_record(
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    plan: _PairPlan,
    selected_for_execution: bool,
    input_use: CompleteTextInputUse | None,
) -> dict[str, object]:
    status: dict[str, object] = {
        "schema_version": "patch-text-combination-pair-status/v1",
        "paper_sha256": PAPER_SHA256,
        "model": config.model,
        "benchmark": config.benchmark,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "denominator_sha256": reference.denominator_sha256,
        "included_in_reference_denominator": True,
        "selected_for_execution": selected_for_execution,
        "complete_text": plan.complete_text.to_dict(),
        "prepared_clean_continuation_matches_reference_baseline": (
            _mapping(plan.reference.source.record.get("clean"), field="source clean").get(
                "continuation"
            )
            == plan.reference.baseline.clean.text
        ),
        "reference_baseline": _baseline_record(plan.reference.baseline),
        "reference_patch_0:6": _generation_record(plan.reference.correct.generation),
        "execution_status": "completed" if selected_for_execution else "not-selected",
        "input_use": input_use.to_dict() if input_use is not None else None,
    }
    if selected_for_execution:
        if input_use is None:
            raise ValueError("a completed pair status requires complete-text input lineage")
        status["baseline_replay"] = _baseline_record(plan.reference.baseline)
    elif input_use is not None:
        raise ValueError("an unselected pair status cannot contain complete-text input lineage")
    return status


def _summary_payload(
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    events: dict[str, list[bool]] = {cell: [] for cell in CELL_ORDER}
    for index, record in enumerate(records):
        cell = record.get("cell")
        event = record.get("event")
        if cell not in events or not isinstance(event, bool):
            raise ValueError(f"cell record {index} has an invalid cell or event")
        events[cell].append(event)
    cell_summaries: list[dict[str, object]] = []
    for cell in CELL_ORDER:
        cell_events = events[cell]
        successes = sum(cell_events)
        total = len(cell_events)
        cell_summaries.append(
            {
                "cell": cell,
                "patch_present": _CELL_DEFINITIONS[cell][0],
                "clean_text": _CELL_DEFINITIONS[cell][1],
                "successes": successes,
                "total": total,
                "rate": successes / total if total else None,
            }
        )
    targetings = Counter(plan.key[0] for plan in selected)
    return {
        "schema_version": "patch-text-combination-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-text-combination",
        "model": config.model,
        "benchmark": config.benchmark,
        "denominator_sha256": reference.denominator_sha256,
        "population": {
            "reference_pairs": len(all_plans),
            "executed_pairs": len(selected),
            "executed_by_targeting": dict(sorted(targetings.items())),
        },
        "cells": cell_summaries,
        "historical_reference": _HISTORICAL_REFERENCE,
        "analysis_status": "descriptive-four-cell-counts-only",
        "prepared_text_reference_agreement": _prepared_text_agreement(selected),
        "comparability": _comparability(config, reference, all_plans),
        "complete_text_boundary": {
            "method": PRE_ANSWER_BOUNDARY_METHOD,
            "implementation_source": "legacy-backed-detail-not-specified-by-final-pdf",
            "no_trigger_policy": "retain-entire-continuation",
            "denominator_policy": "retain-diagnostic-anomalies-in-all-four-cells",
            "reference_diagnostics": _boundary_diagnostics(all_plans),
            "executed_diagnostics": _boundary_diagnostics(selected),
        },
    }


def _compile_outputs(
    *,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
    baseline_dir: Path,
    complete_dir: Path,
    runtime_fingerprint: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    selected_identities = {plan.identity for plan in selected}
    records: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for plan in all_plans:
        selected_for_execution = plan.identity in selected_identities
        if not selected_for_execution:
            statuses.append(
                _pair_status_record(
                    config=config,
                    reference=reference,
                    plan=plan,
                    selected_for_execution=False,
                    input_use=None,
                )
            )
            continue
        _load_baseline_checkpoint(
            _checkpoint_path(baseline_dir, plan),
            plan=plan,
            config=config,
            runtime_fingerprint=runtime_fingerprint,
            reference=reference,
        )
        scan = _load_complete_checkpoint(
            _checkpoint_path(complete_dir, plan),
            plan=plan,
            config=config,
            runtime_fingerprint=runtime_fingerprint,
            reference=reference,
        )
        statuses.append(
            _pair_status_record(
                config=config,
                reference=reference,
                plan=plan,
                selected_for_execution=True,
                input_use=scan.input_use,
            )
        )
        generations = {
            CELL_ORDER[0]: plan.reference.baseline.edited,
            CELL_ORDER[1]: plan.reference.correct.generation,
            CELL_ORDER[2]: scan.no_patch,
            CELL_ORDER[3]: scan.fixed_window_patch,
        }
        for cell in CELL_ORDER:
            generation = generations[cell]
            input_use = scan.input_use if _CELL_DEFINITIONS[cell][1] == "complete" else None
            records.append(
                _cell_record(
                    config=config,
                    reference=reference,
                    plan=plan,
                    cell=cell,
                    generation=generation,
                    input_use=input_use,
                )
            )
    summary = _summary_payload(
        config=config,
        reference=reference,
        all_plans=all_plans,
        selected=selected,
        records=records,
    )
    return records, statuses, summary


def _write_failure_manifest(
    *,
    run_path: Path,
    config: PatchTextCombinationConfig,
    reference: Any,
    all_plans: Sequence[_PairPlan],
    selected: Sequence[_PairPlan],
    started_at: str,
    runtime_provenance: Mapping[str, object],
    registries: Mapping[str, Mapping[str, Mapping[str, str]]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
) -> None:
    failed_counts = dict(counts)
    failed_counts["failed_pairs"] = len(failures)
    failed_counts["records"] = 0
    _remove_public_outputs(config.output_dir)
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            status="failed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            registries=registries,
            counts=failed_counts,
            failures=failures,
        ),
    )


def run_patch_text_combination(
    config: PatchTextCombinationConfig,
    *,
    runtime: PatchTextCombinationRuntime | None = None,
) -> PatchTextCombinationResult:
    """Verify one fixed-window run and publish its four Table 2 cells."""

    reference = coordinate_runner._load_reference(config)  # Shared strict fixed-window adapter.
    output_dir = config.output_dir
    if output_dir.resolve() == config.fixed_window_run.resolve():
        raise ValueError("output directory must not overwrite the fixed-window reference")
    run_path = output_dir / "run.json"
    work_dir = output_dir / ".patch-text-combination-work"
    baseline_dir = work_dir / "baselines"
    complete_dir = work_dir / "complete-text"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    all_plans = _build_plans(reference)
    selected = all_plans[: config.limit] if config.limit is not None else all_plans
    if not selected:
        raise ValueError("fixed-window reference contains no selected pairs")

    previous: dict[str, object] | None = None
    registries = _empty_registries()
    if config.resume:
        previous = _load_json(run_path)
        started_at = _validate_resume_manifest(
            previous,
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
        )
        if previous.get("status") == "completed":
            return _completed_result(
                previous,
                output_dir=output_dir,
                config=config,
                reference=reference,
                all_plans=all_plans,
                selected=selected,
            )
        registries = _load_registries(
            previous,
            selected=selected,
            baseline_dir=baseline_dir,
            complete_dir=complete_dir,
        )
    else:
        started_at = _now()

    if runtime is None:
        runtime = HuggingFacePatchTextCombinationRuntime(
            config,
            revision=reference.sources.model_revision,
        )
    runtime_provenance, runtime_fingerprint = _validate_runtime(
        runtime,
        reference=reference,
        config=config,
    )
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match the original run")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    complete_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_orphans(baseline_dir, registries["baselines"])
    _cleanup_orphans(complete_dir, registries["complete_text"])
    _remove_public_outputs(output_dir)

    for plan in selected:
        if plan.identity in registries["baselines"]:
            _load_baseline_checkpoint(
                _checkpoint_path(baseline_dir, plan),
                plan=plan,
                config=config,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )
        if plan.identity in registries["complete_text"]:
            _load_complete_checkpoint(
                _checkpoint_path(complete_dir, plan),
                plan=plan,
                config=config,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )

    counts: dict[str, object] = {
        "reference_pairs": len(all_plans),
        "executed_pairs": len(selected),
        "baseline_checkpoints": len(registries["baselines"]),
        "complete_text_checkpoints": len(registries["complete_text"]),
        "failed_pairs": 0,
        "records": 0,
    }
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            registries=registries,
            counts=counts,
            failures=[],
        ),
    )

    baseline_mismatches: list[dict[str, str]] = []
    for plan in tqdm(
        selected,
        desc="patch-text baselines",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        if plan.identity in registries["baselines"]:
            continue
        try:
            baseline = runtime.regenerate_baseline(plan.reference.source.record)
            _validate_baseline(baseline, plan=plan, config=config)
            if baseline != plan.reference.baseline:
                baseline_mismatches.append(
                    {
                        "targeting": plan.key[0],
                        "sample_id": plan.key[1],
                        "error_type": "BaselineMismatch",
                        "message": "baseline replay differs from the fixed-window reference",
                    }
                )
                continue
            checkpoint_path = _checkpoint_path(baseline_dir, plan)
            checkpoint = _baseline_checkpoint(
                plan,
                baseline,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )
            _write_json_atomic(checkpoint_path, checkpoint)
            _load_baseline_checkpoint(
                checkpoint_path,
                plan=plan,
                config=config,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )
            registries["baselines"][plan.identity] = _registry_entry(checkpoint_path, plan)
            counts["baseline_checkpoints"] = len(registries["baselines"])
            _write_json_atomic(
                run_path,
                _base_manifest(
                    config=config,
                    reference=reference,
                    all_plans=all_plans,
                    selected=selected,
                    status="running",
                    started_at=started_at,
                    runtime_provenance=runtime_provenance,
                    registries=registries,
                    counts=counts,
                    failures=[],
                ),
            )
        except Exception as exc:
            failure = {
                "targeting": plan.key[0],
                "sample_id": plan.key[1],
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _write_failure_manifest(
                run_path=run_path,
                config=config,
                reference=reference,
                all_plans=all_plans,
                selected=selected,
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                registries=registries,
                counts=counts,
                failures=[failure],
            )
            raise PatchTextCombinationRunError(
                f"baseline pair {plan.key[1]} failed: {exc}; rerun with --resume"
            ) from exc

    if baseline_mismatches:
        _write_failure_manifest(
            run_path=run_path,
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            registries=registries,
            counts=counts,
            failures=baseline_mismatches,
        )
        raise PatchTextCombinationRunError(
            f"baseline replay mismatch for {len(baseline_mismatches)} pair(s); "
            "no complete-text cell was generated"
        )

    for plan in tqdm(
        selected,
        desc="patch-text complete cells",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        if plan.identity in registries["complete_text"]:
            continue
        try:
            scan = runtime.scan_complete_text(
                plan.reference.source.record,
                plan.complete_text.text,
                config.layers[0],
            )
            checkpoint = _complete_checkpoint(
                plan,
                scan,
                config=config,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )
            checkpoint_path = _checkpoint_path(complete_dir, plan)
            _write_json_atomic(checkpoint_path, checkpoint)
            _load_complete_checkpoint(
                checkpoint_path,
                plan=plan,
                config=config,
                runtime_fingerprint=runtime_fingerprint,
                reference=reference,
            )
            registries["complete_text"][plan.identity] = _registry_entry(checkpoint_path, plan)
            counts["complete_text_checkpoints"] = len(registries["complete_text"])
            _write_json_atomic(
                run_path,
                _base_manifest(
                    config=config,
                    reference=reference,
                    all_plans=all_plans,
                    selected=selected,
                    status="running",
                    started_at=started_at,
                    runtime_provenance=runtime_provenance,
                    registries=registries,
                    counts=counts,
                    failures=[],
                ),
            )
        except Exception as exc:
            failure = {
                "targeting": plan.key[0],
                "sample_id": plan.key[1],
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _write_failure_manifest(
                run_path=run_path,
                config=config,
                reference=reference,
                all_plans=all_plans,
                selected=selected,
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                registries=registries,
                counts=counts,
                failures=[failure],
            )
            raise PatchTextCombinationRunError(
                f"pair {plan.key[1]} failed: {exc}; inspect run.json and rerun with --resume"
            ) from exc

    try:
        records, statuses, summary = _compile_outputs(
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            baseline_dir=baseline_dir,
            complete_dir=complete_dir,
            runtime_fingerprint=runtime_fingerprint,
        )
        records_path = output_dir / _PUBLIC_OUTPUT_NAMES[0]
        statuses_path = output_dir / _PUBLIC_OUTPUT_NAMES[1]
        summary_path = output_dir / _PUBLIC_OUTPUT_NAMES[2]
        _write_jsonl_atomic(records_path, records)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        counts["records"] = len(records)
        outputs = {
            records_path.name: {"sha256": _sha256(records_path), "records": len(records)},
            statuses_path.name: {"sha256": _sha256(statuses_path), "records": len(statuses)},
            summary_path.name: {"sha256": _sha256(summary_path), "records": 1},
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                reference=reference,
                all_plans=all_plans,
                selected=selected,
                status="completed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                registries=registries,
                counts=counts,
                failures=[],
                outputs=outputs,
            ),
        )
    except Exception as exc:
        _write_failure_manifest(
            run_path=run_path,
            config=config,
            reference=reference,
            all_plans=all_plans,
            selected=selected,
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            registries=registries,
            counts=counts,
            failures=[
                {
                    "targeting": "aggregate",
                    "sample_id": "aggregate",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        )
        raise PatchTextCombinationRunError(f"failed to publish complete outputs: {exc}") from exc

    try:
        shutil.rmtree(work_dir)
    except OSError:
        # Public outputs and their completed manifest are already durable. Keeping
        # redundant private checkpoints is safer than rolling back a successful run.
        pass

    return _result(output_dir, pairs=len(selected), records=len(records))

"""Fail-closed discovery and validation of completed one-token producer runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.one_token_prefix_replacement.metrics import (
    aggregate_one_token_events,
    classify_one_token_events,
)
from typo_cot.experiments.one_token_prefix_replacement.planning import (
    OneTokenInputPlan,
    OneTokenProfile,
    build_arm_specs,
    choose_adjacent_position,
    choose_distant_positions,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    ADJACENT_SETTINGS,
    BENCHMARK_DATASET_NAMES,
    EXTENSION_SETTINGS,
    LEGACY_SETTING_IDS,
    LEGACY_TARGETING_CODES,
    PRIMARY_SETTING,
    PROTOCOL,
)


_OUTPUT_NAMES = (
    "one_token_records.jsonl",
    "pair_status_records.jsonl",
    "one_token_summary.json",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class OneTokenTablesInputError(ValueError):
    """A supplied producer artifact failed reproducibility validation."""


@dataclass(frozen=True, slots=True)
class ValidatedOneTokenRun:
    """One setting whose manifest, records, statuses, and summary agree."""

    setting_id: str
    model: str
    benchmark: str
    cohort: str
    adjacent_requested: bool
    records: tuple[Mapping[str, object], ...]
    run_dir: Path = Path(".")
    position_controls: tuple[str, ...] = ()
    manifest_sha256: str = ""
    producer_code_identity: Mapping[str, object] = field(default_factory=dict)
    input_files: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        controls = self.position_controls or (
            ("distant", "adjacent") if self.adjacent_requested else ("distant",)
        )
        object.__setattr__(self, "position_controls", tuple(controls))
        if self.adjacent_requested != ("adjacent" in self.position_controls):
            raise ValueError("adjacent_requested disagrees with position_controls")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OneTokenTablesInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise OneTokenTablesInputError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise OneTokenTablesInputError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OneTokenTablesInputError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OneTokenTablesInputError(f"cannot read JSONL at {path}: {exc}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise OneTokenTablesInputError(f"blank JSONL row at {path}:{line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except ValueError as exc:
            raise OneTokenTablesInputError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise OneTokenTablesInputError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OneTokenTablesInputError(f"{field_name} must be a JSON object")
    return value


def _identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OneTokenTablesInputError(f"{field_name} must be a non-empty string")
    return value


def _plan_key(value: object, *, field_name: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(part, str) or not part for part in value)
    ):
        raise OneTokenTablesInputError(f"{field_name} must be a targeting/sample-id pair")
    return value[0], value[1]


def _validate_frozen_plan(
    value: object,
    *,
    cohort: str,
    expected_target_count: int,
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]],
    tuple[tuple[str, str], ...],
]:
    plan = _mapping(value, field_name="producer frozen plan")
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
    if set(plan) != expected_fields:
        raise OneTokenTablesInputError("producer frozen plan has an unexpected schema")
    if plan.get("algorithm") != "shared-clean-prefix-cohort-selection-before-limit/v1":
        raise OneTokenTablesInputError("producer frozen plan algorithm does not match")

    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise OneTokenTablesInputError("producer frozen plan has no cases")
    if plan.get("cases_sha256") != _canonical_sha256(raw_cases):
        raise OneTokenTablesInputError("producer frozen plan case SHA-256 does not match")
    if plan.get("source_case_count") != len(raw_cases):
        raise OneTokenTablesInputError("producer frozen plan source count does not match")

    case_fields = {
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
    cases: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_case in raw_cases:
        case = _mapping(raw_case, field_name="producer frozen plan case")
        if set(case) != case_fields:
            raise OneTokenTablesInputError("producer frozen plan case has an unexpected schema")
        if case.get("cohort") != cohort:
            raise OneTokenTablesInputError("producer frozen plan case cohort does not match")
        key = (
            _identity(case.get("targeting"), field_name="plan case targeting"),
            _identity(case.get("sample_id"), field_name="plan case sample ID"),
        )
        if key in cases:
            raise OneTokenTablesInputError(f"duplicate producer frozen plan identity {key!r}")
        source_sha = case.get("source_record_sha256")
        if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
            raise OneTokenTablesInputError("producer frozen plan source SHA-256 is invalid")
        if type(case.get("candidate_eligible")) is not bool:
            raise OneTokenTablesInputError("producer frozen plan candidate flag is invalid")
        cases[key] = case
    if plan.get("eligible_case_count") != sum(
        case.get("candidate_eligible") is True for case in cases.values()
    ):
        raise OneTokenTablesInputError("producer frozen plan eligible count does not match")

    raw_full = plan.get("selected_full")
    raw_execution = plan.get("selected_for_execution")
    if not isinstance(raw_full, list) or not isinstance(raw_execution, list):
        raise OneTokenTablesInputError("producer frozen plan selections must be lists")
    if plan.get("selected_full_sha256") != _canonical_sha256(raw_full) or plan.get(
        "selected_for_execution_sha256"
    ) != _canonical_sha256(raw_execution):
        raise OneTokenTablesInputError("producer frozen plan selection SHA-256 does not match")
    selected_full = tuple(_plan_key(item, field_name="producer selected_full") for item in raw_full)
    selected_execution = tuple(
        _plan_key(item, field_name="producer selected_for_execution") for item in raw_execution
    )
    if len(selected_full) != expected_target_count:
        raise OneTokenTablesInputError(
            f"producer frozen plan must contain {expected_target_count} paper targets"
        )
    if len(set(selected_full)) != len(selected_full):
        raise OneTokenTablesInputError("producer frozen plan selection contains duplicates")
    if selected_execution != selected_full:
        raise OneTokenTablesInputError("unlimited producer frozen plan execution is incomplete")
    for key in selected_full:
        case = cases.get(key)
        if case is None:
            raise OneTokenTablesInputError("producer frozen plan selects an unknown case")
        cot_count = case.get("cot_token_count")
        input_sha = case.get("input_plan_sha256")
        if (
            case.get("candidate_eligible") is not True
            or case.get("boundary_valid") is not True
            or case.get("exclusion_reason") is not None
            or not isinstance(cot_count, int)
            or isinstance(cot_count, bool)
            or not 8 <= cot_count <= 512
            or not isinstance(input_sha, str)
            or _SHA256.fullmatch(input_sha) is None
        ):
            raise OneTokenTablesInputError("producer frozen plan selected target is not executable")
    return cases, selected_full


def _validate_comparability(
    value: object,
    *,
    expected_target_count: int,
) -> Mapping[str, object]:
    comparability = _mapping(value, field_name="comparability")
    expected = {
        "status": "fresh-paper-protocol-run",
        "requirements": {
            "paper_setting": True,
            "paper_source_protocol": True,
            "paper_source_cohort_identity": False,
            "expected_selected_target_count": expected_target_count,
            "selected_target_count_matches": True,
            "selected_exact_boundary_valid_count": expected_target_count,
            "selected_exact_boundaries_all_valid": True,
            "prespecified_position_controls": True,
            "unlimited": True,
        },
        "limitations": [],
        "primary_in_fourteen_extension_aggregate": False,
        "fresh_public_preparation_is_historical_identity_proof": False,
        "single_setting_runner_computes_cross_setting_interval": False,
    }
    if dict(comparability) != expected:
        raise OneTokenTablesInputError(
            "producer comparability does not prove a complete fresh-paper-protocol-run plan"
        )
    return comparability


def _validate_output_files(
    run_dir: Path,
    outputs: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if set(outputs) != set(_OUTPUT_NAMES):
        raise OneTokenTablesInputError("producer manifest must name exactly three outputs")
    validated: dict[str, dict[str, object]] = {}
    for name in _OUTPUT_NAMES:
        metadata = _mapping(outputs.get(name), field_name=f"output metadata {name}")
        if set(metadata) != {"sha256", "bytes"}:
            raise OneTokenTablesInputError(f"output metadata shape is invalid: {name}")
        expected_sha = metadata.get("sha256")
        expected_bytes = metadata.get("bytes")
        if (
            not isinstance(expected_sha, str)
            or _SHA256.fullmatch(expected_sha) is None
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise OneTokenTablesInputError(f"output metadata value is invalid: {name}")
        path = run_dir / name
        if path.is_symlink() or not path.is_file():
            raise OneTokenTablesInputError(f"producer output is missing or not regular: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = _file_sha256(path)
        if actual_bytes != expected_bytes:
            raise OneTokenTablesInputError(f"output byte count mismatch: {name}")
        if actual_sha != expected_sha:
            raise OneTokenTablesInputError(f"output SHA-256 mismatch: {name}")
        validated[name] = {"sha256": actual_sha, "bytes": actual_bytes}
    return validated


def _arm_map(record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_arms = record.get("arms")
    if not isinstance(raw_arms, list):
        raise OneTokenTablesInputError("one-token record arms must be a list")
    arms: dict[str, Mapping[str, object]] = {}
    for raw in raw_arms:
        arm = _mapping(raw, field_name="one-token arm")
        name = _identity(arm.get("name"), field_name="one-token arm name")
        if name in arms:
            raise OneTokenTablesInputError(f"duplicate one-token arm {name}")
        arms[name] = arm
    return arms


def _validate_input_plan(record: Mapping[str, object]) -> OneTokenInputPlan:
    plan = _mapping(record.get("input_plan"), field_name="one-token input plan")
    expected_fields = {
        "clean_prompt_ids",
        "edited_prompt_ids",
        "clean_full_ids",
        "edited_full_ids",
        "clean_cot_ids",
    }
    if set(plan) != expected_fields:
        raise OneTokenTablesInputError("one-token input plan has an unexpected schema")
    if record.get("input_plan_sha256") != _canonical_sha256(plan):
        raise OneTokenTablesInputError("one-token input plan SHA-256 does not match")
    try:
        rebuilt = OneTokenInputPlan(
            clean_prompt_ids=tuple(plan["clean_prompt_ids"]),  # type: ignore[arg-type]
            edited_prompt_ids=tuple(plan["edited_prompt_ids"]),  # type: ignore[arg-type]
            clean_full_ids=tuple(plan["clean_full_ids"]),  # type: ignore[arg-type]
            edited_full_ids=tuple(plan["edited_full_ids"]),  # type: ignore[arg-type]
            clean_cot_ids=tuple(plan["clean_cot_ids"]),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise OneTokenTablesInputError(f"one-token input plan is invalid: {exc}") from exc
    if rebuilt.to_dict() != dict(plan):
        raise OneTokenTablesInputError("one-token input plan does not reconstruct")
    return rebuilt


def _validate_finite_profile(record: Mapping[str, object]) -> OneTokenProfile:
    profile = _mapping(record.get("profile"), field_name="one-token profile")
    expected_fields = {
        "clean_to_edited_kl",
        "clean_token_rank_under_clean",
        "clean_token_rank_under_edited",
        "edited_top1_ids",
        "edited_top1_is_admissible",
    }
    if set(profile) != expected_fields:
        raise OneTokenTablesInputError("one-token profile has an unexpected schema")
    if record.get("profile_sha256") != _canonical_sha256(profile):
        raise OneTokenTablesInputError("one-token profile SHA-256 does not match")
    kl = profile.get("clean_to_edited_kl")
    if not isinstance(kl, list) or not kl:
        raise OneTokenTablesInputError("one-token profile has no KL sequence")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in kl
    ):
        raise OneTokenTablesInputError("one-token profile KL values must be finite non-negative")
    try:
        rebuilt = OneTokenProfile(
            clean_to_edited_kl=tuple(profile["clean_to_edited_kl"]),  # type: ignore[arg-type]
            clean_token_rank_under_clean=tuple(  # type: ignore[arg-type]
                profile["clean_token_rank_under_clean"]
            ),
            clean_token_rank_under_edited=tuple(  # type: ignore[arg-type]
                profile["clean_token_rank_under_edited"]
            ),
            edited_top1_ids=tuple(profile["edited_top1_ids"]),  # type: ignore[arg-type]
            edited_top1_is_admissible=tuple(  # type: ignore[arg-type]
                profile["edited_top1_is_admissible"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise OneTokenTablesInputError(f"one-token profile is invalid: {exc}") from exc
    if rebuilt.to_dict() != dict(profile):
        raise OneTokenTablesInputError("one-token profile does not reconstruct")
    return rebuilt


def _validate_generation(
    generation: Mapping[str, object],
    *,
    benchmark: str,
    gold_answer: str,
    eos_ids: Sequence[int],
) -> None:
    expected_fields = {
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
    if set(generation) != expected_fields:
        raise OneTokenTablesInputError("generation has an unexpected schema")
    token_ids = generation.get("token_ids")
    if not isinstance(token_ids, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids
    ):
        raise OneTokenTablesInputError("generation token IDs are invalid")
    text = generation.get("text")
    if not isinstance(text, str):
        raise OneTokenTablesInputError("generation text must be a string")
    stop_reason = generation.get("stop_reason")
    stop_token = generation.get("stop_token_id")
    if stop_reason == "eos_token":
        if stop_token not in eos_ids or not token_ids or token_ids[-1] != stop_token:
            raise OneTokenTablesInputError("generation EOS semantics are invalid")
        if any(token in eos_ids for token in token_ids[:-1]):
            raise OneTokenTablesInputError("generation continued after EOS")
        allow_positional = True
    elif stop_reason == "max_new_tokens":
        if stop_token is not None or any(token in eos_ids for token in token_ids):
            raise OneTokenTablesInputError("max-token generation EOS semantics are invalid")
        allow_positional = False
    else:
        raise OneTokenTablesInputError("generation stop reason is invalid")
    extraction = extract_with_fallback(
        text,
        benchmark=BENCHMARK_DATASET_NAMES[benchmark],
        correct_answer=gold_answer,
        allow_positional=allow_positional,
    )
    expected = {
        "value": extraction.value,
        "is_extracted": extraction.is_extracted,
        "is_correct": extraction.is_correct,
        "method": extraction.method,
        "primary_method": extraction.primary_method,
    }
    if any(generation.get(field) != value for field, value in expected.items()):
        raise OneTokenTablesInputError(
            "generation extraction or correctness does not reconstruct from text and gold"
        )


def _validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    plan_cases: Mapping[tuple[str, str], Mapping[str, object]],
    selected_plan_keys: Sequence[tuple[str, str]],
    model: str,
    benchmark: str,
    cohort: str,
    setting_id: str,
    adjacent_requested: bool,
    eos_ids: Sequence[int],
) -> dict[tuple[str, str], Mapping[str, object]]:
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    selected_keys = set(selected_plan_keys)
    for record in records:
        expected_record_fields = {
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
        if set(record) != expected_record_fields:
            raise OneTokenTablesInputError("one-token record has an unexpected schema")
        if (
            record.get("schema_version") != "one-token-prefix-replacement-record/v1"
            or record.get("paper_sha256") != PAPER_SHA256
            or record.get("operation") != "one-token-prefix-replacement"
            or record.get("model") != model
            or record.get("benchmark") != benchmark
            or record.get("cohort") != cohort
        ):
            raise OneTokenTablesInputError("one-token record identity does not match its run")
        targeting = _identity(record.get("targeting"), field_name="record targeting")
        sample_id = _identity(record.get("sample_id"), field_name="record sample_id")
        key = targeting, sample_id
        if key in by_key:
            raise OneTokenTablesInputError(f"duplicate one-token record identity {key!r}")
        plan_case = plan_cases.get(key)
        if key not in selected_keys or plan_case is None:
            raise OneTokenTablesInputError("one-token record is outside the frozen execution plan")
        source = _mapping(record.get("source"), field_name="one-token record source")
        cohort_sha = source.get("cohort_sha256")
        if (
            set(source) != {"cohort_sha256", "source_record_sha256"}
            or not isinstance(cohort_sha, str)
            or _SHA256.fullmatch(cohort_sha) is None
            or source.get("source_record_sha256") != plan_case.get("source_record_sha256")
        ):
            raise OneTokenTablesInputError("one-token record source does not match the frozen plan")
        positions = _mapping(record.get("positions"), field_name="record positions")
        selected = positions.get("selected")
        distant = positions.get("distant")
        adjacent = positions.get("adjacent")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (selected, distant)
        ):
            raise OneTokenTablesInputError("record selected/distant positions are invalid")
        if adjacent is not None and (
            not isinstance(adjacent, int) or isinstance(adjacent, bool) or adjacent < 0
        ):
            raise OneTokenTablesInputError("record adjacent position is invalid")
        if adjacent is not None and not adjacent_requested:
            raise OneTokenTablesInputError("record has an unrequested adjacent position")
        plan = _validate_input_plan(record)
        if record.get("input_plan_sha256") != plan_case.get(
            "input_plan_sha256"
        ) or plan.cot_token_count != plan_case.get("cot_token_count"):
            raise OneTokenTablesInputError("record input plan does not match the frozen plan")
        profile = _validate_finite_profile(record)
        if plan.cot_token_count != len(profile.clean_to_edited_kl):
            raise OneTokenTablesInputError("record profile length does not match the input plan")
        try:
            selection = choose_distant_positions(profile)
        except ValueError as exc:
            raise OneTokenTablesInputError(
                f"record claims completion but its position selection fails: {exc}"
            ) from exc
        if selection.distant_position is None or record.get("selection") != selection.to_dict():
            raise OneTokenTablesInputError("record selection does not reconstruct from profile")
        if selected != selection.selected_position or distant != selection.distant_position:
            raise OneTokenTablesInputError("record positions do not reconstruct from selection")
        expected_adjacent = None
        if adjacent_requested:
            try:
                targeting_code = LEGACY_TARGETING_CODES[targeting]
            except KeyError as exc:
                raise OneTokenTablesInputError(
                    f"record targeting is not a paper condition: {targeting}"
                ) from exc
            expected_adjacent = choose_adjacent_position(
                profile.clean_to_edited_kl,
                selected_position=selection.selected_position,
                tie_key=f"{setting_id}|{targeting_code}|{sample_id}",
            )
        if adjacent != expected_adjacent:
            raise OneTokenTablesInputError("record adjacent position does not reconstruct")
        expected_adjacent_reason = (
            "no-nearest-strictly-lower-kl-position"
            if adjacent_requested and expected_adjacent is None
            else None
        )
        if record.get("adjacent_unavailable_reason") != expected_adjacent_reason:
            raise OneTokenTablesInputError("record adjacent availability does not reconstruct")
        try:
            arm_specs = build_arm_specs(
                plan,
                profile,
                selection,
                adjacent_position=expected_adjacent,
            )
        except (TypeError, ValueError) as exc:
            raise OneTokenTablesInputError(
                f"record arm plan cannot be reconstructed: {exc}"
            ) from exc
        raw_arms = record.get("arms")
        if not isinstance(raw_arms, list) or len(raw_arms) != len(arm_specs):
            raise OneTokenTablesInputError("record arm count does not reconstruct")
        for raw_arm, arm_spec in zip(raw_arms, arm_specs, strict=True):
            arm = _mapping(raw_arm, field_name="one-token arm")
            generation = _mapping(
                arm.get("generation"),
                field_name=f"generation for arm {arm_spec.name}",
            )
            expected_arm = {
                **arm_spec.to_dict(),
                "input_ids": list(
                    plan.generation_input_ids(arm_spec.position, arm_spec.forced_token_id)
                ),
                "generation": dict(generation),
            }
            if dict(arm) != expected_arm:
                raise OneTokenTablesInputError("record arm semantics do not reconstruct")
        arms = _arm_map(record)
        gold_answer = record.get("gold_answer")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise OneTokenTablesInputError("one-token record has no gold answer")
        for name, arm in arms.items():
            _validate_generation(
                _mapping(
                    arm.get("generation"),
                    field_name=f"generation for arm {name}",
                ),
                benchmark=benchmark,
                gold_answer=gold_answer,
                eos_ids=eos_ids,
            )
        try:
            rebuilt = classify_one_token_events(
                arms,
                selected_before_control=int(selected) < int(distant),
                adjacent_requested=adjacent is not None,
            )
        except ValueError as exc:
            raise OneTokenTablesInputError(f"record arms are invalid: {exc}") from exc
        if record.get("events") != rebuilt:
            raise OneTokenTablesInputError("record events do not reconstruct from arms")
        by_key[key] = record
    return by_key


def _validate_position_exclusion_evidence(
    value: object,
    *,
    plan_case: Mapping[str, object],
    adjacent_requested: bool,
) -> tuple[Mapping[str, object], str, str | None]:
    evidence = _mapping(value, field_name="position-exclusion evidence")
    plan = _validate_input_plan(evidence)
    if evidence.get("input_plan_sha256") != plan_case.get(
        "input_plan_sha256"
    ) or plan.cot_token_count != plan_case.get("cot_token_count"):
        raise OneTokenTablesInputError("position-exclusion evidence input plan does not match")
    profile = _validate_finite_profile(evidence)
    if len(profile.clean_to_edited_kl) != plan.cot_token_count:
        raise OneTokenTablesInputError("position-exclusion evidence profile length does not match")
    try:
        selection = choose_distant_positions(profile)
    except ValueError as exc:
        reason = str(exc)
        selection_row = None
        selected_position = None
    else:
        if selection.distant_position is not None:
            raise OneTokenTablesInputError(
                "position-exclusion evidence reconstructs an executable distant control"
            )
        reason = "no-distant-lower-median-control"
        selection_row = selection.to_dict()
        selected_position = selection.selected_position
    if reason not in {
        "no-position-with-clean-token-below-edited-top1",
        "no-distant-lower-median-control",
    }:
        raise OneTokenTablesInputError("position-exclusion evidence has an unknown reason")
    positions = {"selected": selected_position, "distant": None, "adjacent": None}
    adjacent_reason = (
        "case-excluded-before-adjacent-position-selection" if adjacent_requested else None
    )
    source = {
        "cohort": plan_case.get("cohort"),
        "targeting": plan_case.get("targeting"),
        "sample_id": plan_case.get("sample_id"),
        "source_record_sha256": plan_case.get("source_record_sha256"),
    }
    expected = {
        "schema_version": "one-token-prefix-replacement-checkpoint/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "one-token-prefix-replacement",
        "source": source,
        "input_plan": plan.to_dict(),
        "input_plan_sha256": _canonical_sha256(plan.to_dict()),
        "profile": profile.to_dict(),
        "profile_sha256": _canonical_sha256(profile.to_dict()),
        "selection": selection_row,
        "positions": positions,
        "adjacent_requested": adjacent_requested,
        "adjacent_unavailable_reason": adjacent_reason,
        "position_exclusion_reason": reason,
        "arm_specs": [],
        "arm_specs_sha256": _canonical_sha256([]),
        "arms": [],
    }
    if dict(evidence) != expected:
        raise OneTokenTablesInputError("position-exclusion evidence does not reconstruct")
    return positions, reason, adjacent_reason


def _validate_statuses(
    statuses: Sequence[Mapping[str, object]],
    *,
    records: Mapping[tuple[str, str], Mapping[str, object]],
    plan_cases: Mapping[tuple[str, str], Mapping[str, object]],
    selected_plan_keys: Sequence[tuple[str, str]],
    adjacent_requested: bool,
    model: str,
    benchmark: str,
    cohort: str,
) -> None:
    expected_fields = {
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
    selected_keys = set(selected_plan_keys)
    seen: set[tuple[str, str]] = set()
    completed: set[tuple[str, str]] = set()
    for status in statuses:
        if set(status) != expected_fields:
            raise OneTokenTablesInputError("one-token status has an unexpected schema")
        if (
            status.get("schema_version") != "one-token-prefix-replacement-pair-status/v1"
            or status.get("paper_sha256") != PAPER_SHA256
            or status.get("model") != model
            or status.get("benchmark") != benchmark
            or status.get("cohort") != cohort
        ):
            raise OneTokenTablesInputError("one-token status identity does not match its run")
        key = (
            _identity(status.get("targeting"), field_name="status targeting"),
            _identity(status.get("sample_id"), field_name="status sample_id"),
        )
        if key in seen:
            raise OneTokenTablesInputError(f"duplicate one-token status identity {key!r}")
        if key not in plan_cases:
            raise OneTokenTablesInputError("one-token status is outside the frozen plan")
        seen.add(key)
        plan_case = plan_cases[key]
        record = records.get(key)
        evidence = None
        if plan_case.get("candidate_eligible") is not True:
            execution_status = "not-candidate"
            positions = None
            exclusion_reason = plan_case.get("exclusion_reason")
            adjacent_reason = None
        elif key not in selected_keys:
            execution_status = "not-selected"
            positions = None
            exclusion_reason = plan_case.get("exclusion_reason")
            adjacent_reason = None
        elif plan_case.get("boundary_valid") is not True:
            execution_status = "invalid-boundary"
            positions = None
            exclusion_reason = "prompt-boundary-invalid"
            adjacent_reason = None
        elif record is not None:
            execution_status = "completed"
            positions = record.get("positions")
            exclusion_reason = None
            adjacent_reason = record.get("adjacent_unavailable_reason")
            completed.add(key)
        else:
            execution_status = "position-unavailable"
            evidence = status.get("position_exclusion_evidence")
            if not isinstance(evidence, Mapping):
                raise OneTokenTablesInputError(
                    "position exclusion requires reconstructable evidence"
                )
            positions, exclusion_reason, adjacent_reason = _validate_position_exclusion_evidence(
                evidence,
                plan_case=plan_case,
                adjacent_requested=adjacent_requested,
            )
        adjacent_available = (
            isinstance(positions, Mapping) and positions.get("adjacent") is not None
        )
        expected = {
            "schema_version": "one-token-prefix-replacement-pair-status/v1",
            "paper_sha256": PAPER_SHA256,
            "model": model,
            "benchmark": benchmark,
            "cohort": cohort,
            "targeting": key[0],
            "sample_id": key[1],
            "source_record_sha256": plan_case.get("source_record_sha256"),
            "candidate_eligible": plan_case.get("candidate_eligible"),
            "boundary_valid": plan_case.get("boundary_valid"),
            "selected_full": key in selected_keys,
            "selected_for_execution": key in selected_keys,
            "positions": positions,
            "adjacent_requested": adjacent_requested,
            "adjacent_available": adjacent_available,
            "adjacent_unavailable_reason": adjacent_reason,
            "execution_status": execution_status,
            "exclusion_reason": exclusion_reason,
            "position_exclusion_evidence": None if evidence is None else dict(evidence),
            "record_sha256": None if record is None else _canonical_sha256(record),
        }
        if dict(status) != expected:
            raise OneTokenTablesInputError("one-token status semantics do not reconstruct")
    if seen != set(plan_cases):
        raise OneTokenTablesInputError("statuses do not cover the complete frozen plan")
    if completed != set(records):
        raise OneTokenTablesInputError("records and completed status rows do not correspond")


def _load_run(run_path: Path) -> ValidatedOneTokenRun:
    manifest = _load_json(run_path)
    if manifest.get("operation") != "one-token-prefix-replacement":
        raise OneTokenTablesInputError(f"unexpected operation in discovered run.json: {run_path}")
    if (
        manifest.get("schema_version") != "one-token-prefix-replacement-run/v1"
        or manifest.get("paper_sha256") != PAPER_SHA256
        or manifest.get("status") != "completed"
    ):
        raise OneTokenTablesInputError(f"producer run is not a completed canonical run: {run_path}")
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_sha256") != _canonical_sha256(
        PROTOCOL
    ):
        raise OneTokenTablesInputError("producer protocol identity does not match")
    if manifest.get("failures") != [] or manifest.get("checkpoints") != {}:
        raise OneTokenTablesInputError("completed producer has failures or checkpoints")

    arguments = _mapping(manifest.get("arguments"), field_name="producer arguments")
    if arguments.get("limit") is not None:
        raise OneTokenTablesInputError("limited one-token runs cannot build paper tables")
    model = _identity(arguments.get("model"), field_name="producer model")
    benchmark = _identity(arguments.get("benchmark"), field_name="producer benchmark")
    cohort = _identity(arguments.get("cohort"), field_name="producer cohort")
    setting = model, benchmark
    if setting not in LEGACY_SETTING_IDS:
        raise OneTokenTablesInputError(f"unexpected one-token setting {model}/{benchmark}")
    expected_cohort = "primary" if setting == PRIMARY_SETTING else "extension"
    if cohort != expected_cohort or (cohort == "extension" and setting not in EXTENSION_SETTINGS):
        raise OneTokenTablesInputError("producer cohort does not match the paper setting")
    raw_controls = arguments.get("position_controls")
    if not isinstance(raw_controls, list) or any(
        not isinstance(control, str) for control in raw_controls
    ):
        raise OneTokenTablesInputError("producer position controls are invalid")
    controls = tuple(raw_controls)
    expected_controls = ("distant", "adjacent") if setting in ADJACENT_SETTINGS else ("distant",)
    if controls != expected_controls:
        raise OneTokenTablesInputError(
            f"prespecified adjacent/distant controls do not match {LEGACY_SETTING_IDS[setting]}"
        )

    expected_target_count = 172 if cohort == "primary" else 150
    plan_cases, selected_plan_keys = _validate_frozen_plan(
        manifest.get("plan"),
        cohort=cohort,
        expected_target_count=expected_target_count,
    )
    comparability = _validate_comparability(
        manifest.get("comparability"),
        expected_target_count=expected_target_count,
    )
    manifest_counts = _mapping(manifest.get("counts"), field_name="manifest counts")
    expected_manifest_counts = {
        "source_pairs": len(plan_cases),
        "selected_full": expected_target_count,
        "selected_for_execution": expected_target_count,
    }
    if any(
        manifest_counts.get(field) != value for field, value in expected_manifest_counts.items()
    ):
        raise OneTokenTablesInputError("producer manifest counts do not match the frozen plan")

    runtime = _mapping(manifest.get("runtime"), field_name="producer runtime")
    code_identity = _mapping(
        runtime.get("implementation_code_identity"),
        field_name="producer code identity",
    )
    code_sha = code_identity.get("sha256")
    file_count = code_identity.get("python_file_count")
    if (
        not isinstance(code_identity.get("algorithm"), str)
        or not code_identity["algorithm"]
        or not isinstance(code_sha, str)
        or _SHA256.fullmatch(code_sha) is None
        or not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count <= 0
    ):
        raise OneTokenTablesInputError("producer code identity is invalid")
    raw_eos_ids = runtime.get("effective_eos_token_ids")
    if (
        not isinstance(raw_eos_ids, list)
        or not raw_eos_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in raw_eos_ids
        )
        or raw_eos_ids != sorted(set(raw_eos_ids))
    ):
        raise OneTokenTablesInputError("producer runtime EOS IDs are invalid")

    run_dir = run_path.parent
    output_files = _validate_output_files(
        run_dir,
        _mapping(manifest.get("outputs"), field_name="producer outputs"),
    )
    records = _load_jsonl(run_dir / _OUTPUT_NAMES[0])
    statuses = _load_jsonl(run_dir / _OUTPUT_NAMES[1])
    summary = _load_json(run_dir / _OUTPUT_NAMES[2])
    records_by_key = _validate_records(
        records,
        plan_cases=plan_cases,
        selected_plan_keys=selected_plan_keys,
        model=model,
        benchmark=benchmark,
        cohort=cohort,
        setting_id=LEGACY_SETTING_IDS[setting],
        adjacent_requested="adjacent" in controls,
        eos_ids=raw_eos_ids,
    )
    _validate_statuses(
        statuses,
        records=records_by_key,
        plan_cases=plan_cases,
        selected_plan_keys=selected_plan_keys,
        adjacent_requested="adjacent" in controls,
        model=model,
        benchmark=benchmark,
        cohort=cohort,
    )
    expected_setting = {
        "model": model,
        "benchmark": benchmark,
        "cohort": cohort,
        "position_controls": list(controls),
    }
    if (
        summary.get("schema_version") != "one-token-prefix-replacement-summary/v1"
        or summary.get("paper_sha256") != PAPER_SHA256
        or summary.get("operation") != "one-token-prefix-replacement"
        or summary.get("setting") != expected_setting
        or summary.get("protocol") != PROTOCOL
        or summary.get("comparability") != comparability
    ):
        raise OneTokenTablesInputError("producer summary identity does not match the manifest")
    rebuilt_metrics = aggregate_one_token_events(
        [record["events"] for record in records],
        adjacent_requested="adjacent" in controls,
    )
    if summary.get("metrics") != rebuilt_metrics:
        raise OneTokenTablesInputError("producer summary metrics do not reconstruct")
    summary_counts = _mapping(summary.get("counts"), field_name="summary counts")
    expected_summary_counts = {
        "source_pairs": len(plan_cases),
        "candidate_eligible": sum(
            case.get("candidate_eligible") is True for case in plan_cases.values()
        ),
        "selected_full": expected_target_count,
        "selected_for_execution": expected_target_count,
        "executed": len(records),
        "records": len(records),
        "arms": sum(len(record["arms"]) for record in records),  # type: ignore[arg-type]
    }
    if any(
        summary_counts.get(field) != value for field, value in expected_summary_counts.items()
    ) or manifest_counts.get("records") != len(records):
        raise OneTokenTablesInputError("producer summary or manifest counts do not match")

    return ValidatedOneTokenRun(
        setting_id=LEGACY_SETTING_IDS[setting],
        model=model,
        benchmark=benchmark,
        cohort=cohort,
        adjacent_requested="adjacent" in controls,
        records=tuple(records),
        run_dir=run_dir.resolve(),
        position_controls=controls,
        manifest_sha256=_file_sha256(run_path),
        producer_code_identity=dict(code_identity),
        input_files=output_files,
    )


def _discover_manifests(root: Path) -> list[Path]:
    manifests: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise OneTokenTablesInputError(
                    f"symlink is forbidden below runs root: {current_path / name}"
                )
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                raise OneTokenTablesInputError(f"symlink is forbidden below runs root: {path}")
            if name == "run.json":
                manifests.append(path)
    return manifests


def discover_and_validate_runs(runs_root: Path) -> tuple[ValidatedOneTokenRun, ...]:
    """Recursively discover setting runs and validate their complete hash chain."""

    root = Path(runs_root)
    if root.is_symlink() or not root.is_dir():
        raise OneTokenTablesInputError(f"runs root is not a regular directory: {root}")
    manifests = _discover_manifests(root)
    if not manifests:
        raise OneTokenTablesInputError(f"no one-token run.json found below runs root: {root}")
    runs: list[ValidatedOneTokenRun] = []
    by_setting: dict[str, Path] = {}
    code_identity: Mapping[str, object] | None = None
    for manifest_path in manifests:
        run = _load_run(manifest_path)
        if run.setting_id in by_setting:
            raise OneTokenTablesInputError(
                f"duplicate setting {run.setting_id}: {by_setting[run.setting_id]} and "
                f"{manifest_path}"
            )
        by_setting[run.setting_id] = manifest_path
        if code_identity is None:
            code_identity = run.producer_code_identity
        elif run.producer_code_identity != code_identity:
            raise OneTokenTablesInputError("producer code identity differs across input runs")
        runs.append(run)
    return tuple(sorted(runs, key=lambda run: run.setting_id))


__all__ = [
    "OneTokenTablesInputError",
    "ValidatedOneTokenRun",
    "discover_and_validate_runs",
]

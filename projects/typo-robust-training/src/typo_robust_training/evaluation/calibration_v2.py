"""Base-only severity calibration for the preregistered v2 evaluation.

This module deliberately has no adapter-facing input.  It selects the smallest
member of a frozen edit-count grid that supplies adequate Base-model headroom,
or emits a terminal stopped result.  A failed calibration cannot expand the
grid or replace a model through this API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import MappingProxyType

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.integrity import sha256_file


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_TOP = {
    "schema_version",
    "protocol_id",
    "legacy_v1",
    "model_inventory",
    "governance",
    "calibration",
    "confirmatory",
    "statistics",
    "gates",
    "freeze",
}
_MODEL_FIELDS = {"id", "revision", "role"}
_OBSERVATION_FIELDS = {
    "schema_version",
    "condition",
    "model_id",
    "model_revision",
    "adapter_checkpoint_sha256",
    "training_run_sha256",
    "task",
    "record_id",
    "source_text_sha256",
    "severity_edit_count",
    "variant",
    "realized_typo_sha256",
    "clean_correct",
    "typo_correct",
}
_TASKS = ("gsm8k", "mmlu", "arc", "mmlu_pro", "commonsense_qa")
_OPERATIONS = (
    "keyboard-neighbor-substitution",
    "deletion",
    "duplication",
)
_ARMS = (
    "base",
    "output-matching-all-layers",
    "probe-boundary-output-matching",
    "random-freeze-output-matching",
)
_SECONDARY_CONDITIONS = (
    "legacy-random-2",
    "random-1",
    "random-2",
    "random-4",
    "random-8",
    "transposition-2",
    "natural-injection",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"evaluation v2 {field} fields differ")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"evaluation v2 {field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"evaluation v2 {field} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"evaluation v2 {field} must be a finite number >= {minimum}")
    return result


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"evaluation v2 {field} must contain unique strings")
    return tuple(value)


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"evaluation v2 {field} must contain integers")
    result = tuple(_integer(item, field=field, minimum=1) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"evaluation v2 {field} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class FrozenEvaluationModel:
    model_id: str
    revision: str
    role: str


@dataclass(frozen=True, slots=True)
class EvaluationV2Protocol:
    schema_version: str
    protocol_id: str
    legacy_v1_protocol_sha256: str
    models: tuple[FrozenEvaluationModel, ...]
    tasks: tuple[str, ...]
    calibration_records_per_task: int
    calibration_variants_per_item: int
    severity_edit_counts: tuple[int, ...]
    operations: tuple[str, ...]
    minimum_model_macro_gap_points: float
    minimum_each_model_gap_points: float
    minimum_typo_to_clean_accuracy_ratio: float
    confirmatory_records_per_task: int
    confirmatory_typo_variants_per_item: int
    arms: tuple[str, ...]
    secondary_conditions: tuple[str, ...]
    training_seeds: tuple[int, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    config_sha256: str


def load_evaluation_v2_protocol(path: Path) -> EvaluationV2Protocol:
    """Load the exact preregistered v2 protocol and reject scientific drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation v2 protocol is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("evaluation v2 protocol must be UTF-8") from exc
    top = _mapping(payload, field="config", fields=_TOP)
    if (
        top["schema_version"] != "robustness-evaluation-study/v2"
        or top["protocol_id"] != "typo-robustness-evaluation-v2.0"
    ):
        raise ValueError("evaluation v2 protocol identity differs")

    legacy = _mapping(
        top["legacy_v1"],
        field="legacy_v1",
        fields={"protocol_id", "protocol_sha256", "random_2_role"},
    )
    if (
        legacy["protocol_id"] != "typo-robustness-evaluation-v1.4"
        or legacy["protocol_sha256"]
        != "2f52a300ab7ecb84288e3c67a38e61d1c37a17f48c9d01d37c6c29b701182054"
        or legacy["random_2_role"] != "secondary-continuity-only"
    ):
        raise ValueError("evaluation v2 legacy random-2 contract differs")

    model_rows = top["model_inventory"]
    if not isinstance(model_rows, list) or len(model_rows) != 2:
        raise ValueError("evaluation v2 model inventory differs")
    models: list[FrozenEvaluationModel] = []
    for index, value in enumerate(model_rows):
        row = _mapping(value, field=f"model_inventory[{index}]", fields=_MODEL_FIELDS)
        model_id, revision, role = row["id"], row["revision"], row["role"]
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(revision, str)
            or _SHA40.fullmatch(revision) is None
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError("evaluation v2 model identity is invalid")
        models.append(FrozenEvaluationModel(model_id, revision, role))
    expected_models = (
        FrozenEvaluationModel(
            "google/gemma-3-4b-it",
            "093f9f388b31de276ce2de164bdc2081324b9767",
            "development-anchor",
        ),
        FrozenEvaluationModel(
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
            "kojima-family-replication",
        ),
    )
    if tuple(models) != expected_models:
        raise ValueError("evaluation v2 frozen model inventory differs")

    governance = _mapping(
        top["governance"],
        field="governance",
        fields={
            "calibration_allowed_condition",
            "adapter_output_use",
            "replace_model_after_calibration",
            "extend_severity_grid_after_failure",
            "outcome_on_no_eligible_severity",
            "freeze_realized_text_with_sha256",
        },
    )
    if dict(governance) != {
        "calibration_allowed_condition": "base",
        "adapter_output_use": "forbidden",
        "replace_model_after_calibration": "forbidden",
        "extend_severity_grid_after_failure": "forbidden",
        "outcome_on_no_eligible_severity": "stop-and-report",
        "freeze_realized_text_with_sha256": True,
    }:
        raise ValueError("evaluation v2 anti-shopping governance differs")

    calibration = _mapping(
        top["calibration"],
        field="calibration",
        fields={
            "tasks",
            "records_per_task",
            "variants_per_item",
            "severity_edit_counts",
            "operations",
            "targeting",
            "minimum_model_macro_gap_points",
            "minimum_each_model_gap_points",
            "minimum_typo_to_clean_accuracy_ratio",
            "selection_rule",
        },
    )
    tasks = _string_tuple(calibration["tasks"], field="calibration.tasks")
    severity = _int_tuple(
        calibration["severity_edit_counts"], field="calibration.severity_edit_counts"
    )
    operations = _string_tuple(calibration["operations"], field="calibration.operations")
    records_per_task = _integer(
        calibration["records_per_task"], field="calibration.records_per_task", minimum=1
    )
    variants_per_item = _integer(
        calibration["variants_per_item"], field="calibration.variants_per_item", minimum=1
    )
    minimum_macro_gap = _number(
        calibration["minimum_model_macro_gap_points"],
        field="calibration.minimum_model_macro_gap_points",
    )
    minimum_each_gap = _number(
        calibration["minimum_each_model_gap_points"],
        field="calibration.minimum_each_model_gap_points",
    )
    minimum_ratio = _number(
        calibration["minimum_typo_to_clean_accuracy_ratio"],
        field="calibration.minimum_typo_to_clean_accuracy_ratio",
    )
    if (
        tasks != _TASKS
        or records_per_task != 200
        or variants_per_item != 3
        or severity != (2, 4, 8)
        or operations != _OPERATIONS
        or calibration["targeting"]
        != "uniform-eligible-question-words-without-replacement/v1"
        or minimum_macro_gap != 8.0
        or minimum_each_gap != 5.0
        or minimum_ratio != 0.5
        or calibration["selection_rule"] != "smallest-eligible-severity/v1"
    ):
        raise ValueError("evaluation v2 Base-only calibration rule differs")

    confirmatory = _mapping(
        top["confirmatory"],
        field="confirmatory",
        fields={
            "tasks",
            "records_per_task",
            "typo_variants_per_item",
            "primary_condition",
            "secondary_conditions",
            "arms",
            "random_freeze_control",
        },
    )
    confirmatory_tasks = _string_tuple(confirmatory["tasks"], field="confirmatory.tasks")
    confirmatory_records = _integer(
        confirmatory["records_per_task"], field="confirmatory.records_per_task", minimum=1
    )
    confirmatory_variants = _integer(
        confirmatory["typo_variants_per_item"],
        field="confirmatory.typo_variants_per_item",
        minimum=1,
    )
    secondary_conditions = _string_tuple(
        confirmatory["secondary_conditions"], field="confirmatory.secondary_conditions"
    )
    arms = _string_tuple(confirmatory["arms"], field="confirmatory.arms")
    if (
        confirmatory_tasks != _TASKS
        or confirmatory_records != 1000
        or confirmatory_variants != 2
        or confirmatory["primary_condition"] != "base-calibrated-random-k"
        or secondary_conditions != _SECONDARY_CONDITIONS
        or arms != _ARMS
        or confirmatory["random_freeze_control"]
        != "same-frozen-layer-count-hash-selected-per-training-seed/v1"
    ):
        raise ValueError("evaluation v2 confirmatory population differs")

    statistics = _mapping(
        top["statistics"],
        field="statistics",
        fields={
            "training_seeds",
            "bootstrap_replicates",
            "bootstrap_seed",
            "confidence_level",
            "cluster_unit",
            "within_item_aggregation",
            "resampling",
            "macro_weighting",
            "training_seeds_are_independent_items",
            "paired_test",
            "primary_multiplicity",
        },
    )
    seeds = _int_tuple(statistics["training_seeds"], field="statistics.training_seeds")
    if dict(statistics) != {
        "training_seeds": [42, 43, 44],
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
        "cluster_unit": "source-item",
        "within_item_aggregation": "mean-over-two-typo-variants-and-three-training-seeds/v1",
        "resampling": "paired-items-within-model-and-task/v1",
        "macro_weighting": "equal-model-equal-task/v1",
        "training_seeds_are_independent_items": False,
        "paired_test": "exact-mcnemar-per-model-seed-cell/v1",
        "primary_multiplicity": "intersection-union-no-adjustment/v1",
    }:
        raise ValueError("evaluation v2 clustered statistical protocol differs")

    gates = _mapping(
        top["gates"],
        field="gates",
        fields={
            "clean_noninferiority_margin_points",
            "minimum_task_clean_change_points",
            "minimum_proposal_vs_all_layer_typo_gain_points",
            "require_proposal_vs_all_layer_ci_lower_above_zero",
            "minimum_proposal_vs_base_typo_gain_points",
            "require_proposal_vs_base_ci_lower_above_zero",
            "require_proposal_vs_random_freeze_ci_lower_above_zero",
            "maximum_clean_ppl_ratio",
            "require_all_seed_directions_positive",
            "mechanistic_diagnostics_are_blocking",
        },
    )
    expected_gates = {
        "clean_noninferiority_margin_points": 1.0,
        "minimum_task_clean_change_points": -3.0,
        "minimum_proposal_vs_all_layer_typo_gain_points": 1.5,
        "require_proposal_vs_all_layer_ci_lower_above_zero": True,
        "minimum_proposal_vs_base_typo_gain_points": 2.0,
        "require_proposal_vs_base_ci_lower_above_zero": True,
        "require_proposal_vs_random_freeze_ci_lower_above_zero": True,
        "maximum_clean_ppl_ratio": 1.02,
        "require_all_seed_directions_positive": True,
        "mechanistic_diagnostics_are_blocking": False,
    }
    if dict(gates) != expected_gates:
        raise ValueError("evaluation v2 confirmatory gates differ")

    freeze = _mapping(
        top["freeze"],
        field="freeze",
        fields={
            "calibration_ids_disjoint_from_all_other_roles",
            "confirmatory_ids_disjoint_from_training_probe_and_calibration",
            "calibration_observations_sha256_required",
            "calibration_result_sha256_required",
            "confirmatory_item_manifest_sha256_required",
            "realized_typo_variant_manifest_sha256_required",
            "legacy_random_2_registry_sha256_required",
            "all_arms_must_share_exact_item_and_typo_hashes",
        },
    )
    if not all(type(value) is bool and value for value in freeze.values()):
        raise ValueError("evaluation v2 hash-freeze contract differs")

    return EvaluationV2Protocol(
        schema_version="robustness-evaluation-study/v2",
        protocol_id="typo-robustness-evaluation-v2.0",
        legacy_v1_protocol_sha256=str(legacy["protocol_sha256"]),
        models=tuple(models),
        tasks=tasks,
        calibration_records_per_task=records_per_task,
        calibration_variants_per_item=variants_per_item,
        severity_edit_counts=severity,
        operations=operations,
        minimum_model_macro_gap_points=minimum_macro_gap,
        minimum_each_model_gap_points=minimum_each_gap,
        minimum_typo_to_clean_accuracy_ratio=minimum_ratio,
        confirmatory_records_per_task=confirmatory_records,
        confirmatory_typo_variants_per_item=confirmatory_variants,
        arms=arms,
        secondary_conditions=secondary_conditions,
        training_seeds=seeds,
        bootstrap_replicates=10_000,
        bootstrap_seed=42,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class BaseCalibrationObservation:
    condition: str
    model_id: str
    model_revision: str
    adapter_checkpoint_sha256: str | None
    training_run_sha256: str | None
    task: str
    record_id: str
    source_text_sha256: str
    severity_edit_count: int
    variant: int
    realized_typo_sha256: str
    clean_correct: bool
    typo_correct: bool

    def __post_init__(self) -> None:
        if (
            self.condition != "base"
            or self.adapter_checkpoint_sha256 is not None
            or self.training_run_sha256 is not None
        ):
            raise ValueError("evaluation v2 calibration forbids adapter or trained-model outputs")
        if (
            not isinstance(self.model_id, str)
            or not self.model_id
            or not isinstance(self.model_revision, str)
            or _SHA40.fullmatch(self.model_revision) is None
            or not isinstance(self.task, str)
            or not self.task
        ):
            raise ValueError("evaluation v2 calibration model/task identity is invalid")
        if any(
            not isinstance(value, str) or _SHA64.fullmatch(value) is None
            for value in (self.record_id, self.source_text_sha256, self.realized_typo_sha256)
        ):
            raise ValueError("evaluation v2 calibration text identity must use SHA-256")
        if (
            isinstance(self.severity_edit_count, bool)
            or not isinstance(self.severity_edit_count, int)
            or self.severity_edit_count <= 0
            or isinstance(self.variant, bool)
            or not isinstance(self.variant, int)
            or self.variant < 0
        ):
            raise ValueError("evaluation v2 calibration severity/variant is invalid")
        if type(self.clean_correct) is not bool or type(self.typo_correct) is not bool:
            raise ValueError("evaluation v2 calibration outcomes must be boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "robustness-evaluation-v2-calibration-observation/v1",
            "condition": self.condition,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_checkpoint_sha256": self.adapter_checkpoint_sha256,
            "training_run_sha256": self.training_run_sha256,
            "task": self.task,
            "record_id": self.record_id,
            "source_text_sha256": self.source_text_sha256,
            "severity_edit_count": self.severity_edit_count,
            "variant": self.variant,
            "realized_typo_sha256": self.realized_typo_sha256,
            "clean_correct": self.clean_correct,
            "typo_correct": self.typo_correct,
        }

    @classmethod
    def from_mapping(cls, value: object) -> BaseCalibrationObservation:
        row = _mapping(value, field="calibration observation", fields=_OBSERVATION_FIELDS)
        if row["schema_version"] != "robustness-evaluation-v2-calibration-observation/v1":
            raise ValueError("evaluation v2 calibration observation schema differs")
        model_id, revision, task = row["model_id"], row["model_revision"], row["task"]
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(revision, str)
            or _SHA40.fullmatch(revision) is None
            or not isinstance(task, str)
            or not task
        ):
            raise ValueError("evaluation v2 calibration model/task identity is invalid")
        hashes = (row["record_id"], row["source_text_sha256"], row["realized_typo_sha256"])
        if any(not isinstance(value, str) or _SHA64.fullmatch(value) is None for value in hashes):
            raise ValueError("evaluation v2 calibration text identity must use SHA-256")
        clean, typo = row["clean_correct"], row["typo_correct"]
        if type(clean) is not bool or type(typo) is not bool:
            raise ValueError("evaluation v2 calibration outcomes must be boolean")
        return cls(
            condition=str(row["condition"]),
            model_id=model_id,
            model_revision=revision,
            adapter_checkpoint_sha256=row["adapter_checkpoint_sha256"],
            training_run_sha256=row["training_run_sha256"],
            task=task,
            record_id=str(row["record_id"]),
            source_text_sha256=str(row["source_text_sha256"]),
            severity_edit_count=_integer(
                row["severity_edit_count"], field="severity_edit_count", minimum=1
            ),
            variant=_integer(row["variant"], field="variant"),
            realized_typo_sha256=str(row["realized_typo_sha256"]),
            clean_correct=clean,
            typo_correct=typo,
        )


@dataclass(frozen=True, slots=True)
class SeverityCalibrationResult:
    status: str
    selected_edit_count: int | None
    summaries: Mapping[str, object]
    artifact_path: Path
    run_path: Path


def load_base_calibration_observations(path: Path) -> tuple[BaseCalibrationObservation, ...]:
    observations: list[BaseCalibrationObservation] = []
    for line_number, line in read_lf_jsonl_lines(
        Path(path), context="evaluation v2 Base-only calibration observations"
    ):
        value = strict_loads(line, context=f"{path}:{line_number}")
        observations.append(BaseCalibrationObservation.from_mapping(value))
    if not observations:
        raise ValueError("evaluation v2 calibration observations are empty")
    return tuple(observations)


def _validate_calibration_cohort(
    observations: Sequence[BaseCalibrationObservation],
    *,
    protocol: EvaluationV2Protocol,
) -> None:
    models = {(model.model_id, model.revision) for model in protocol.models}
    expected_variants = set(range(protocol.calibration_variants_per_item))
    expected_severities = set(protocol.severity_edit_counts)
    seen: set[tuple[str, str, str, int, int]] = set()
    record_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    typo_hashes: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    clean_values: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    coverage: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)

    for row in observations:
        model = (row.model_id, row.model_revision)
        if model not in models:
            raise ValueError("evaluation v2 calibration model inventory differs")
        if row.task not in protocol.tasks:
            raise ValueError("evaluation v2 calibration task inventory differs")
        if row.severity_edit_count not in expected_severities:
            raise ValueError(
                "evaluation v2 calibration severity is outside the frozen candidate grid"
            )
        if row.variant not in expected_variants:
            raise ValueError("evaluation v2 calibration variant inventory differs")
        key = (
            row.model_id,
            row.task,
            row.record_id,
            row.severity_edit_count,
            row.variant,
        )
        if key in seen:
            raise ValueError("evaluation v2 calibration contains a duplicate observation")
        seen.add(key)
        record_sets[(row.model_id, row.task)].add(row.record_id)
        source_hashes[(row.task, row.record_id)].add(row.source_text_sha256)
        typo_hashes[(row.task, row.record_id, row.severity_edit_count, row.variant)].add(
            row.realized_typo_sha256
        )
        clean_values[(row.model_id, row.task, row.record_id)].add(row.clean_correct)
        coverage[(row.model_id, row.task, row.record_id)].add(
            (row.severity_edit_count, row.variant)
        )

    reference_records: dict[str, set[str]] = {}
    for model in protocol.models:
        for task in protocol.tasks:
            records = record_sets[(model.model_id, task)]
            if len(records) != protocol.calibration_records_per_task:
                raise ValueError("evaluation v2 calibration task sample size differs")
            if task in reference_records and records != reference_records[task]:
                raise ValueError("evaluation v2 calibration models must share exact item IDs")
            reference_records.setdefault(task, records)
    expected_cells = {
        (severity, variant)
        for severity in protocol.severity_edit_counts
        for variant in range(protocol.calibration_variants_per_item)
    }
    if any(values != expected_cells for values in coverage.values()):
        raise ValueError("evaluation v2 calibration grid coverage differs")
    if any(len(values) != 1 for values in source_hashes.values()):
        raise ValueError("evaluation v2 calibration source text differs across models")
    if any(len(values) != 1 for values in typo_hashes.values()):
        raise ValueError("evaluation v2 calibration typo text differs across models")
    if any(len(values) != 1 for values in clean_values.values()):
        raise ValueError("evaluation v2 calibration clean outcome is not invariant")


def score_base_only_severity_calibration(
    observations: Sequence[BaseCalibrationObservation],
    *,
    protocol: EvaluationV2Protocol,
) -> tuple[str, int | None, Mapping[str, object]]:
    """Select the smallest eligible frozen severity, or stop without a fallback."""

    _validate_calibration_cohort(observations, protocol=protocol)
    by_model_task_record: dict[tuple[str, str, str], list[BaseCalibrationObservation]] = (
        defaultdict(list)
    )
    for row in observations:
        by_model_task_record[(row.model_id, row.task, row.record_id)].append(row)

    clean_accuracy: dict[str, float] = {}
    typo_accuracy: dict[tuple[str, int], float] = {}
    for model in protocol.models:
        clean_task_means: list[float] = []
        for task in protocol.tasks:
            clean_values = [
                float(rows[0].clean_correct)
                for (model_id, row_task, _record), rows in by_model_task_record.items()
                if model_id == model.model_id and row_task == task
            ]
            clean_task_means.append(fmean(clean_values))
        clean_accuracy[model.model_id] = fmean(clean_task_means)
        for severity in protocol.severity_edit_counts:
            typo_task_means: list[float] = []
            for task in protocol.tasks:
                record_means = [
                    fmean(
                        float(row.typo_correct)
                        for row in rows
                        if row.severity_edit_count == severity
                    )
                    for (model_id, row_task, _record), rows in by_model_task_record.items()
                    if model_id == model.model_id and row_task == task
                ]
                typo_task_means.append(fmean(record_means))
            typo_accuracy[(model.model_id, severity)] = fmean(typo_task_means)

    summaries: dict[str, object] = {}
    selected: int | None = None
    for severity in protocol.severity_edit_counts:
        model_rows: dict[str, object] = {}
        gaps: list[float] = []
        each_gap_passes = True
        ratio_passes = True
        for model in protocol.models:
            clean = clean_accuracy[model.model_id]
            typo = typo_accuracy[(model.model_id, severity)]
            gap = 100.0 * (clean - typo)
            ratio = typo / clean if clean > 0.0 else 0.0
            gaps.append(gap)
            each_gap_passes &= gap >= protocol.minimum_each_model_gap_points
            ratio_passes &= ratio >= protocol.minimum_typo_to_clean_accuracy_ratio
            model_rows[model.model_id] = {
                "clean_accuracy": clean,
                "typo_accuracy": typo,
                "gap_points": gap,
                "typo_to_clean_accuracy_ratio": ratio,
            }
        macro_gap = fmean(gaps)
        eligible = (
            macro_gap >= protocol.minimum_model_macro_gap_points
            and each_gap_passes
            and ratio_passes
        )
        summaries[str(severity)] = {
            "model_equal_macro_gap_points": macro_gap,
            "each_model_gap_passed": each_gap_passes,
            "each_model_floor_passed": ratio_passes,
            "eligible": eligible,
            "models": model_rows,
        }
        if selected is None and eligible:
            selected = severity

    if selected is None:
        return "stopped-no-eligible-severity", None, MappingProxyType(summaries)
    return "selected", selected, MappingProxyType(summaries)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_base_only_severity_calibration(
    *,
    config_path: Path,
    observations_path: Path,
    item_manifest_path: Path,
    realized_typo_manifest_path: Path,
    output_dir: Path,
) -> SeverityCalibrationResult:
    """Write a hash-bound selected or terminal-stopped calibration artifact."""

    protocol = load_evaluation_v2_protocol(config_path)
    inputs = tuple(
        Path(path).resolve()
        for path in (observations_path, item_manifest_path, realized_typo_manifest_path)
    )
    if any(not path.is_file() for path in inputs):
        raise ValueError("evaluation v2 calibration inputs must be files")
    observations = load_base_calibration_observations(inputs[0])
    status, selected, summaries = score_base_only_severity_calibration(
        observations, protocol=protocol
    )
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"evaluation v2 calibration output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "severity-calibration.json"
    run_path = destination / "run.json"
    artifact = {
        "schema_version": "robustness-evaluation-v2-severity-calibration/v1",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.config_sha256,
        "model_inventory": [
            {"id": model.model_id, "revision": model.revision} for model in protocol.models
        ],
        "candidate_edit_counts": list(protocol.severity_edit_counts),
        "status": status,
        "selected_primary_edit_count": selected,
        "stop_policy": "do-not-extend-grid-or-replace-models",
        "provenance": {
            "adapter_outputs_used": False,
            "item_manifest_sha256": sha256_file(inputs[1]),
            "realized_typo_manifest_sha256": sha256_file(inputs[2]),
            "base_observations_sha256": sha256_file(inputs[0]),
        },
        "summaries": dict(summaries),
    }
    _atomic_json(artifact_path, artifact)
    _atomic_json(
        run_path,
        {
            "schema_version": "robustness-evaluation-v2-calibration-run/v1",
            "status": "completed" if selected is not None else "stopped",
            "protocol_sha256": protocol.config_sha256,
            "artifact_sha256": sha256_file(artifact_path),
            "input_identity_sha256": _canonical_sha256(artifact["provenance"]),
        },
    )
    return SeverityCalibrationResult(
        status=status,
        selected_edit_count=selected,
        summaries=MappingProxyType(dict(summaries)),
        artifact_path=artifact_path,
        run_path=run_path,
    )


__all__ = [
    "BaseCalibrationObservation",
    "EvaluationV2Protocol",
    "FrozenEvaluationModel",
    "SeverityCalibrationResult",
    "load_base_calibration_observations",
    "load_evaluation_v2_protocol",
    "run_base_only_severity_calibration",
    "score_base_only_severity_calibration",
]

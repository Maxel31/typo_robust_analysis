"""Paired accuracy, KL, patch-reliance, and engineering-gate summaries."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, stdev

from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol
from typo_robust_training.evaluation.records import EvaluationObservation


_PATCH_GAIN_DENOMINATOR_EPSILON = 1e-6
_PRIMARY_EVALUATION_CONDITION = "random-2"


def _mean(values: Iterable[float]) -> float | None:
    rows = tuple(values)
    return fmean(rows) if rows else None


def _patch_gain(observation: EvaluationObservation) -> float | None:
    if not observation.untreated_kl_2_16 or not observation.patched_kl_2_16:
        return None
    untreated = fmean(observation.untreated_kl_2_16)
    if untreated <= _PATCH_GAIN_DENOMINATOR_EPSILON:
        return None
    return 1.0 - fmean(observation.patched_kl_2_16) / untreated


def _patch_gain_exclusion_reason(observation: EvaluationObservation) -> str | None:
    if not observation.untreated_kl_2_16:
        return "untreated-readout-unavailable"
    if not observation.patched_kl_2_16:
        return "patched-readout-unavailable"
    if fmean(observation.untreated_kl_2_16) <= _PATCH_GAIN_DENOMINATOR_EPSILON:
        return "near-zero-untreated-kl"
    return None


def _condition_summary(rows: Sequence[EvaluationObservation]) -> dict[str, object]:
    answer_rows = tuple(row for row in rows if row.clean_correct is not None)
    gains = tuple(gain for row in rows if (gain := _patch_gain(row)) is not None)
    exclusions: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = _patch_gain_exclusion_reason(row)
        if reason is not None:
            exclusions[reason] += 1
    return {
        "n_records": len(rows),
        "n_answer": len(answer_rows),
        "clean_accuracy": _mean(float(row.clean_correct) for row in answer_rows),
        "typo_accuracy": _mean(float(row.typo_correct) for row in answer_rows),
        "patched_accuracy": _mean(
            float(row.patched_correct) for row in answer_rows if row.patched_correct is not None
        ),
        "n_patch_gain": len(gains),
        "patch_gain_exclusions": dict(sorted(exclusions.items())),
        "mean_patch_gain": _mean(gains),
        "mean_clean_typo_kl": _mean(
            fmean(row.untreated_kl_2_16) for row in rows if row.untreated_kl_2_16
        ),
    }


def _breakdown(
    rows: Sequence[EvaluationObservation],
    *,
    key,
) -> dict[str, object]:
    groups: dict[str, list[EvaluationObservation]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _condition_summary(group) for name, group in sorted(groups.items())}


def _bootstrap_accuracy_difference(
    differences: Sequence[float],
    *,
    label: str,
    protocol: RobustnessEvaluationProtocol,
) -> list[float] | None:
    if not differences:
        return None
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(protocol.bootstrap_seed ^ label_seed)
    means = np.empty(protocol.bootstrap_replicates, dtype=np.float64)
    chunk_size = 512
    for start in range(0, protocol.bootstrap_replicates, chunk_size):
        stop = min(start + chunk_size, protocol.bootstrap_replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1) * 100.0
    alpha = (1.0 - protocol.confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha]).tolist()
    return [float(lower), float(upper)]


def _paired_metrics(
    base: Sequence[EvaluationObservation],
    adapter: Sequence[EvaluationObservation],
    *,
    label: str,
    protocol: RobustnessEvaluationProtocol,
    bootstrap: bool,
) -> dict[str, object]:
    base_by_id = {row.record_id: row for row in base}
    adapter_by_id = {row.record_id: row for row in adapter}
    if set(base_by_id) != set(adapter_by_id):
        raise ValueError("base and adapter paired record IDs differ")
    pairs = tuple((base_by_id[record_id], adapter_by_id[record_id]) for record_id in base_by_id)
    if any(left.evaluation_condition != right.evaluation_condition for left, right in pairs):
        raise ValueError("base and adapter typo conditions differ")
    answer_pairs = tuple(
        (left, right)
        for left, right in pairs
        if left.clean_correct is not None and right.clean_correct is not None
    )
    if any((left.clean_correct is None) != (right.clean_correct is None) for left, right in pairs):
        raise ValueError("base and adapter answer availability differs")
    base_clean = _mean(float(left.clean_correct) for left, _right in answer_pairs)
    adapter_clean = _mean(float(right.clean_correct) for _left, right in answer_pairs)
    base_typo = _mean(float(left.typo_correct) for left, _right in answer_pairs)
    adapter_typo = _mean(float(right.typo_correct) for _left, right in answer_pairs)
    differences = tuple(
        float(right.typo_correct) - float(left.typo_correct) for left, right in answer_pairs
    )
    base_gains: list[float] = []
    adapter_gains: list[float] = []
    readout_valid = 0
    exclusion_pairs: dict[str, int] = defaultdict(int)
    for left, right in pairs:
        left_reason = _patch_gain_exclusion_reason(left)
        right_reason = _patch_gain_exclusion_reason(right)
        if left_reason not in {
            "untreated-readout-unavailable",
            "patched-readout-unavailable",
        } and right_reason not in {
            "untreated-readout-unavailable",
            "patched-readout-unavailable",
        }:
            readout_valid += 1
        if left_reason is not None:
            exclusion_pairs[f"base:{left_reason}"] += 1
        if right_reason is not None:
            exclusion_pairs[f"adapter:{right_reason}"] += 1
        left_gain = None if left_reason is not None else _patch_gain(left)
        right_gain = None if right_reason is not None else _patch_gain(right)
        if left_gain is not None and right_gain is not None:
            base_gains.append(left_gain)
            adapter_gains.append(right_gain)
    base_patch_gain = _mean(base_gains)
    adapter_patch_gain = _mean(adapter_gains)
    reduction = None
    if base_patch_gain is not None and adapter_patch_gain is not None and base_patch_gain > 1e-6:
        reduction = (base_patch_gain - adapter_patch_gain) / base_patch_gain
    result: dict[str, object] = {
        "n_records": len(pairs),
        "n_answer": len(answer_pairs),
        "base_clean_accuracy": base_clean,
        "adapter_clean_accuracy": adapter_clean,
        "clean_accuracy_drop_points": (
            None
            if base_clean is None or adapter_clean is None
            else 100.0 * (base_clean - adapter_clean)
        ),
        "base_typo_accuracy": base_typo,
        "adapter_typo_accuracy": adapter_typo,
        "typo_accuracy_gain_points": (
            None
            if base_typo is None or adapter_typo is None
            else 100.0 * (adapter_typo - base_typo)
        ),
        "wrong_to_right": sum(
            left.typo_correct is False and right.typo_correct is True
            for left, right in answer_pairs
        ),
        "right_to_wrong": sum(
            left.typo_correct is True and right.typo_correct is False
            for left, right in answer_pairs
        ),
        "clean_harm": sum(
            left.clean_correct is True and right.clean_correct is False
            for left, right in answer_pairs
        ),
        "n_patch_readout_valid": readout_valid,
        "patch_readout_coverage_fraction": (readout_valid / len(pairs) if pairs else None),
        "n_paired_patch_gain": len(base_gains),
        "patch_gain_coverage_fraction": (len(base_gains) / len(pairs) if pairs else None),
        "patch_gain_exclusions": dict(sorted(exclusion_pairs.items())),
        "base_mean_patch_gain": base_patch_gain,
        "adapter_mean_patch_gain": adapter_patch_gain,
        "patch_gain_reduction_fraction": reduction,
    }
    if bootstrap:
        result["typo_accuracy_gain_ci95_points"] = _bootstrap_accuracy_difference(
            differences,
            label=label,
            protocol=protocol,
        )
    return result


def _bootstrap_method_accuracy_difference(
    differences_by_seed: Mapping[int, Sequence[float]],
    *,
    label: str,
    protocol: RobustnessEvaluationProtocol,
) -> list[float] | None:
    """Hierarchically resample training seeds, then paired items within each seed."""

    if not differences_by_seed:
        return None
    import numpy as np

    seeds = tuple(sorted(differences_by_seed))
    lengths = {len(differences_by_seed[seed]) for seed in seeds}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("method-level bootstrap requires one shared non-empty paired cohort")
    values = np.asarray([differences_by_seed[seed] for seed in seeds], dtype=np.float64)
    seed_count, item_count = values.shape
    label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(protocol.bootstrap_seed ^ label_seed)
    means = np.empty(protocol.bootstrap_replicates, dtype=np.float64)
    for replicate in range(protocol.bootstrap_replicates):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = []
        for seed_index in sampled_seeds:
            sampled_items = rng.integers(0, item_count, size=item_count)
            seed_means.append(values[seed_index, sampled_items].mean())
        means[replicate] = fmean(seed_means) * 100.0
    alpha = (1.0 - protocol.confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha]).tolist()
    return [float(lower), float(upper)]


def _method_paired_metrics(
    base: Sequence[EvaluationObservation],
    adapter_by_seed: Mapping[int, Sequence[EvaluationObservation]],
    *,
    condition: str,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object]:
    seed_metrics: dict[int, dict[str, object]] = {}
    typo_differences: dict[int, tuple[float, ...]] = {}
    clean_drop_differences: dict[int, tuple[float, ...]] = {}
    base_by_id = {row.record_id: row for row in base}
    for seed, adapter in sorted(adapter_by_seed.items()):
        seed_metrics[seed] = _paired_metrics(
            base,
            adapter,
            label=f"{condition}:seed-{seed}",
            protocol=protocol,
            bootstrap=False,
        )
        adapter_by_id = {row.record_id: row for row in adapter}
        answer_pairs = tuple(
            (base_by_id[record_id], adapter_by_id[record_id])
            for record_id in base_by_id
            if base_by_id[record_id].clean_correct is not None
        )
        typo_differences[seed] = tuple(
            float(right.typo_correct) - float(left.typo_correct) for left, right in answer_pairs
        )
        clean_drop_differences[seed] = tuple(
            float(left.clean_correct) - float(right.clean_correct) for left, right in answer_pairs
        )

    def values(name: str) -> tuple[float, ...]:
        result = tuple(
            float(metrics[name])
            for metrics in seed_metrics.values()
            if isinstance(metrics.get(name), (int, float))
        )
        if len(result) != len(seed_metrics):
            raise ValueError(f"method-level {name} is unavailable for one or more seeds")
        return result

    typo_gains = values("typo_accuracy_gain_points")
    clean_drops = values("clean_accuracy_drop_points")
    patch_reductions = tuple(
        float(metrics["patch_gain_reduction_fraction"])
        for metrics in seed_metrics.values()
        if isinstance(metrics.get("patch_gain_reduction_fraction"), (int, float))
    )
    return {
        "n_seeds": len(seed_metrics),
        "seed_inventory": sorted(seed_metrics),
        "seed_inventory_complete": set(seed_metrics) == set(protocol.seed_inventory),
        "typo_accuracy_gain_points_mean": fmean(typo_gains),
        "typo_accuracy_gain_points_sd": stdev(typo_gains) if len(typo_gains) > 1 else 0.0,
        "typo_accuracy_gain_ci95_points": _bootstrap_method_accuracy_difference(
            typo_differences,
            label=f"{condition}:typo",
            protocol=protocol,
        ),
        "clean_accuracy_drop_points_mean": fmean(clean_drops),
        "clean_accuracy_drop_points_sd": stdev(clean_drops) if len(clean_drops) > 1 else 0.0,
        "clean_accuracy_drop_ci95_points": _bootstrap_method_accuracy_difference(
            clean_drop_differences,
            label=f"{condition}:clean-drop",
            protocol=protocol,
        ),
        "patch_gain_reduction_fraction_mean": (
            fmean(patch_reductions) if len(patch_reductions) == len(seed_metrics) else None
        ),
        "per_seed": {str(seed): metrics for seed, metrics in seed_metrics.items()},
    }


def _finite_at_least(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= threshold


def _finite_at_most(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value <= threshold


def build_evaluation_report(
    observations: Sequence[EvaluationObservation],
    *,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object]:
    """Aggregate strictly paired base/adapter observations and evaluate the gate."""

    rows = tuple(observations)
    if not rows or any(not isinstance(row, EvaluationObservation) for row in rows):
        raise ValueError("evaluation report requires validated observations")
    identities = [(row.condition_id, row.record_id) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("evaluation observations duplicate a condition and record")
    grouped: dict[str, list[EvaluationObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.condition_id].append(row)
    if "base" not in grouped:
        raise ValueError("evaluation report requires the base condition")
    base = tuple(grouped["base"])
    primary_base = tuple(
        row for row in base if row.evaluation_condition == _PRIMARY_EVALUATION_CONDITION
    )
    if not primary_base:
        raise ValueError("evaluation report requires confirmatory random-2 observations")
    condition_report: dict[str, object] = {}
    for condition_id, condition_rows_list in sorted(grouped.items()):
        condition_rows = tuple(condition_rows_list)
        primary_rows = tuple(
            row
            for row in condition_rows
            if row.evaluation_condition == _PRIMARY_EVALUATION_CONDITION
        )
        condition_report[condition_id] = {
            "primary_condition": _PRIMARY_EVALUATION_CONDITION,
            "overall": _condition_summary(primary_rows),
            "all_conditions": _condition_summary(condition_rows),
            "evaluation_conditions": _breakdown(
                condition_rows, key=lambda row: row.evaluation_condition
            ),
            "strata": {
                stratum: _condition_summary(
                    tuple(row for row in primary_rows if stratum in row.strata)
                )
                for stratum in ("same-task", "unseen-task", "unseen-content", "unseen-typo")
            },
            "tasks": _breakdown(
                primary_rows, key=lambda row: row.task if row.task is not None else "none"
            ),
            "sources": _breakdown(primary_rows, key=lambda row: row.source),
            "operations": _breakdown(primary_rows, key=lambda row: row.operation),
            "edit_counts": _breakdown(primary_rows, key=lambda row: str(row.edit_count)),
            "tokenization_strata": _breakdown(
                primary_rows, key=lambda row: row.tokenization_stratum
            ),
        }

    comparisons: dict[str, object] = {}
    for condition_id, adapter_rows_list in sorted(grouped.items()):
        if condition_id == "base":
            continue
        adapter = tuple(adapter_rows_list)
        primary_adapter = tuple(
            row for row in adapter if row.evaluation_condition == _PRIMARY_EVALUATION_CONDITION
        )
        overall = _paired_metrics(
            primary_base,
            primary_adapter,
            label=condition_id,
            protocol=protocol,
            bootstrap=True,
        )
        all_conditions = _paired_metrics(
            base,
            adapter,
            label=f"{condition_id}:all-conditions",
            protocol=protocol,
            bootstrap=False,
        )
        evaluation_conditions = {}
        condition_inventory = sorted(
            {row.evaluation_condition for row in base}
            | {row.evaluation_condition for row in adapter}
        )
        for evaluation_condition in condition_inventory:
            base_subset = tuple(
                row for row in base if row.evaluation_condition == evaluation_condition
            )
            adapter_subset = tuple(
                row for row in adapter if row.evaluation_condition == evaluation_condition
            )
            evaluation_conditions[evaluation_condition] = _paired_metrics(
                base_subset,
                adapter_subset,
                label=f"{condition_id}:{evaluation_condition}",
                protocol=protocol,
                bootstrap=False,
            )
        strata = {}
        for stratum in ("same-task", "unseen-task", "unseen-content", "unseen-typo"):
            base_subset = tuple(row for row in primary_base if stratum in row.strata)
            adapter_subset = tuple(row for row in primary_adapter if stratum in row.strata)
            strata[stratum] = _paired_metrics(
                base_subset,
                adapter_subset,
                label=f"{condition_id}:{stratum}",
                protocol=protocol,
                bootstrap=False,
            )
        comparisons[condition_id] = {
            "primary_condition": _PRIMARY_EVALUATION_CONDITION,
            "overall": overall,
            "all_conditions": all_conditions,
            "evaluation_conditions": evaluation_conditions,
            "strata": strata,
        }

    method_rows: dict[str, dict[int, list[EvaluationObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.condition != "base":
            if row.seed is None:
                raise RuntimeError("validated adapter observation lost its training seed")
            method_rows[row.condition][row.seed].append(row)
    method_comparisons = {
        condition: _method_paired_metrics(
            primary_base,
            {
                seed: tuple(
                    row
                    for row in seed_rows
                    if row.evaluation_condition == _PRIMARY_EVALUATION_CONDITION
                )
                for seed, seed_rows in by_seed.items()
            },
            condition=condition,
            protocol=protocol,
        )
        for condition, by_seed in sorted(method_rows.items())
    }

    gate_seed_checks: dict[str, object] = {}
    directional_seeds: list[int] = []
    gate = protocol.gate
    for seed in protocol.seed_inventory:
        condition_id = f"localized-state-distillation:seed-{seed}"
        comparison = comparisons.get(condition_id)
        if not isinstance(comparison, dict):
            continue
        overall = comparison["overall"]
        strata = comparison["strata"]
        if not isinstance(overall, dict) or not isinstance(strata, dict):
            raise RuntimeError("validated evaluation comparison changed type")
        unseen = strata["unseen-task"]
        if not isinstance(unseen, dict):
            raise RuntimeError("validated unseen-task comparison changed type")
        checks = {
            "typo_accuracy": _finite_at_least(
                overall["typo_accuracy_gain_points"],
                float(gate["minimum_typo_accuracy_gain_points"]),
            ),
            "clean_preservation": _finite_at_most(
                overall["clean_accuracy_drop_points"],
                float(gate["maximum_clean_accuracy_drop_points"]),
            ),
            "net_repair": (
                overall["wrong_to_right"] > overall["right_to_wrong"]
                if gate["require_wrong_to_right_above_right_to_wrong"]
                else True
            ),
            "unseen_task_transfer": (
                _finite_at_least(unseen["typo_accuracy_gain_points"], 1e-12)
                if gate["require_positive_unseen_task_gain"]
                else True
            ),
        }
        passed = all(checks.values())
        gate_seed_checks[str(seed)] = {"checks": checks, "passed": passed}
        if passed:
            directional_seeds.append(seed)
    inventory_complete = set(map(int, gate_seed_checks)) == set(protocol.seed_inventory)
    gate_passed = inventory_complete and len(directional_seeds) >= int(
        gate["minimum_directional_seeds"]
    )
    return {
        "schema_version": "robustness-evaluation-report/v2",
        "conditions": condition_report,
        "comparisons": comparisons,
        "method_comparisons": method_comparisons,
        "gate": {
            "condition": "localized-state-distillation",
            "seed_inventory_complete": inventory_complete,
            "seed_checks": gate_seed_checks,
            "directional_seeds": directional_seeds,
            "minimum_directional_seeds": gate["minimum_directional_seeds"],
            "passed": gate_passed,
        },
    }


__all__ = ["build_evaluation_report"]

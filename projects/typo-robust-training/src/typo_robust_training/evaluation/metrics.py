"""Paired accuracy, KL, patch-reliance, and engineering-gate summaries."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import fmean

from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol
from typo_robust_training.evaluation.records import EvaluationObservation


def _mean(values: Iterable[float]) -> float | None:
    rows = tuple(values)
    return fmean(rows) if rows else None


def _patch_gain(observation: EvaluationObservation) -> float | None:
    if not observation.untreated_kl_2_16 or not observation.patched_kl_2_16:
        return None
    untreated = fmean(observation.untreated_kl_2_16)
    if untreated <= 1e-6:
        return None
    return 1.0 - fmean(observation.patched_kl_2_16) / untreated


def _condition_summary(rows: Sequence[EvaluationObservation]) -> dict[str, object]:
    answer_rows = tuple(row for row in rows if row.clean_correct is not None)
    gains = tuple(gain for row in rows if (gain := _patch_gain(row)) is not None)
    return {
        "n_records": len(rows),
        "n_answer": len(answer_rows),
        "clean_accuracy": _mean(float(row.clean_correct) for row in answer_rows),
        "typo_accuracy": _mean(float(row.typo_correct) for row in answer_rows),
        "patched_accuracy": _mean(
            float(row.patched_correct) for row in answer_rows if row.patched_correct is not None
        ),
        "n_patch_gain": len(gains),
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
    for left, right in pairs:
        left_gain, right_gain = _patch_gain(left), _patch_gain(right)
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
        "n_paired_patch_gain": len(base_gains),
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
    condition_report: dict[str, object] = {}
    for condition_id, condition_rows_list in sorted(grouped.items()):
        condition_rows = tuple(condition_rows_list)
        condition_report[condition_id] = {
            "overall": _condition_summary(condition_rows),
            "strata": {
                stratum: _condition_summary(
                    tuple(row for row in condition_rows if stratum in row.strata)
                )
                for stratum in ("same-task", "unseen-task", "unseen-content", "unseen-typo")
            },
            "tasks": _breakdown(
                condition_rows, key=lambda row: row.task if row.task is not None else "none"
            ),
            "sources": _breakdown(condition_rows, key=lambda row: row.source),
            "operations": _breakdown(condition_rows, key=lambda row: row.operation),
            "tokenization_strata": _breakdown(
                condition_rows, key=lambda row: row.tokenization_stratum
            ),
        }

    comparisons: dict[str, object] = {}
    for condition_id, adapter_rows_list in sorted(grouped.items()):
        if condition_id == "base":
            continue
        adapter = tuple(adapter_rows_list)
        overall = _paired_metrics(
            base,
            adapter,
            label=condition_id,
            protocol=protocol,
            bootstrap=True,
        )
        strata = {}
        for stratum in ("same-task", "unseen-task", "unseen-content", "unseen-typo"):
            base_subset = tuple(row for row in base if stratum in row.strata)
            adapter_subset = tuple(row for row in adapter if stratum in row.strata)
            strata[stratum] = _paired_metrics(
                base_subset,
                adapter_subset,
                label=f"{condition_id}:{stratum}",
                protocol=protocol,
                bootstrap=False,
            )
        comparisons[condition_id] = {"overall": overall, "strata": strata}

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
            "patch_reliance_reduction": _finite_at_least(
                overall["patch_gain_reduction_fraction"],
                float(gate["minimum_patch_gain_reduction_fraction"]),
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
        "schema_version": "robustness-evaluation-report/v1",
        "conditions": condition_report,
        "comparisons": comparisons,
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

"""Paired accuracy, KL, patch-reliance, and engineering-gate summaries."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, median, stdev

from typo_robust_training.evaluation.config import RobustnessEvaluationProtocol
from typo_robust_training.evaluation.records import (
    CorpusEvaluationObservation,
    EvaluationObservation,
)
from typo_robust_training.evaluation.study import EvaluationStudyProtocol


_PATCH_GAIN_DENOMINATOR_EPSILON = 1e-6
_PRIMARY_EVALUATION_CONDITION = "random-2"
_REPORT_STRATA = ("same-task", "unseen-task", "unseen-content", "unseen-typo")


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


def _answer_differences_by_task(
    pairs: Sequence[tuple[EvaluationObservation, EvaluationObservation]],
    *,
    field: str,
) -> dict[str, tuple[float, ...]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for base, adapter in pairs:
        if base.task is None or adapter.task != base.task:
            raise ValueError("task-macro accuracy requires matching task identities")
        left = getattr(base, field)
        right = getattr(adapter, field)
        if type(left) is not bool or type(right) is not bool:
            raise ValueError("task-macro accuracy requires boolean paired outcomes")
        grouped[base.task].append(float(right) - float(left))
    return {task: tuple(values) for task, values in sorted(grouped.items())}


def _task_macro(differences_by_task: Mapping[str, Sequence[float]]) -> float | None:
    if not differences_by_task:
        return None
    if any(not values for values in differences_by_task.values()):
        raise ValueError("task macro requires a non-empty cohort for every task")
    return 100.0 * fmean(fmean(values) for values in differences_by_task.values())


def _exact_mcnemar_pvalue(wrong_to_right: int, right_to_wrong: int) -> float:
    """Return the two-sided exact McNemar p-value for paired binary outcomes."""

    if min(wrong_to_right, right_to_wrong) < 0:
        raise ValueError("McNemar transition counts must be non-negative")
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_right, right_to_wrong)
    log_probabilities = tuple(
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail + 1)
    )
    maximum = max(log_probabilities)
    probability = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probabilities)
    return min(1.0, 2.0 * probability)


def _bootstrap_bounds(
    samples: object, *, protocol: RobustnessEvaluationProtocol
) -> dict[str, object]:
    import numpy as np

    alpha = 1.0 - protocol.confidence_level
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()
    one_sided_lower = float(np.quantile(samples, alpha))
    return {
        "ci95_points": [float(lower), float(upper)],
        "one_sided_95_lower_points": one_sided_lower,
    }


def _bootstrap_task_macro_difference(
    differences_by_task: Mapping[str, Sequence[float]],
    *,
    label: str,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object] | None:
    if not differences_by_task:
        return None
    import numpy as np

    tasks = tuple(sorted(differences_by_task))
    values = {task: np.asarray(differences_by_task[task], dtype=np.float64) for task in tasks}
    if any(array.size == 0 for array in values.values()):
        raise ValueError("task-stratified bootstrap requires non-empty task cohorts")
    label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(protocol.bootstrap_seed ^ label_seed)
    means = np.empty(protocol.bootstrap_replicates, dtype=np.float64)
    for replicate in range(protocol.bootstrap_replicates):
        task_means = []
        for task in tasks:
            array = values[task]
            indices = rng.integers(0, array.size, size=array.size)
            task_means.append(array[indices].mean())
        means[replicate] = fmean(task_means) * 100.0
    return _bootstrap_bounds(means, protocol=protocol)


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
    pairs = tuple(
        (base_by_id[record_id], adapter_by_id[record_id]) for record_id in sorted(base_by_id)
    )
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
    clean_differences_by_task = _answer_differences_by_task(
        answer_pairs,
        field="clean_correct",
    )
    typo_differences_by_task = _answer_differences_by_task(
        answer_pairs,
        field="typo_correct",
    )
    task_clean_changes = {
        task: 100.0 * fmean(values) for task, values in clean_differences_by_task.items()
    }
    task_typo_gains = {
        task: 100.0 * fmean(values) for task, values in typo_differences_by_task.items()
    }
    if any(left.mechanistic_audit != right.mechanistic_audit for left, right in pairs):
        raise ValueError("base and adapter mechanistic-audit cohorts differ")
    audit_pairs = tuple((left, right) for left, right in pairs if left.mechanistic_audit)
    base_gains: list[float] = []
    adapter_gains: list[float] = []
    readout_valid = 0
    exclusion_pairs: dict[str, int] = defaultdict(int)
    for left, right in audit_pairs:
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
    typo_wrong_to_right = sum(
        left.typo_correct is False and right.typo_correct is True for left, right in answer_pairs
    )
    typo_right_to_wrong = sum(
        left.typo_correct is True and right.typo_correct is False for left, right in answer_pairs
    )
    typo_wrong_to_wrong = sum(
        left.typo_correct is False and right.typo_correct is False for left, right in answer_pairs
    )
    typo_right_to_right = sum(
        left.typo_correct is True and right.typo_correct is True for left, right in answer_pairs
    )
    clean_wrong_to_right = sum(
        left.clean_correct is False and right.clean_correct is True for left, right in answer_pairs
    )
    clean_right_to_wrong = sum(
        left.clean_correct is True and right.clean_correct is False for left, right in answer_pairs
    )
    clean_wrong_to_wrong = sum(
        left.clean_correct is False and right.clean_correct is False for left, right in answer_pairs
    )
    clean_right_to_right = sum(
        left.clean_correct is True and right.clean_correct is True for left, right in answer_pairs
    )
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
        "clean_accuracy_change_points": (
            None
            if base_clean is None or adapter_clean is None
            else 100.0 * (adapter_clean - base_clean)
        ),
        "task_macro_clean_accuracy_change_points": _task_macro(clean_differences_by_task),
        "task_clean_accuracy_change_points": task_clean_changes,
        "base_typo_accuracy": base_typo,
        "adapter_typo_accuracy": adapter_typo,
        "typo_accuracy_gain_points": (
            None
            if base_typo is None or adapter_typo is None
            else 100.0 * (adapter_typo - base_typo)
        ),
        "task_macro_typo_accuracy_gain_points": _task_macro(typo_differences_by_task),
        "task_typo_accuracy_gain_points": task_typo_gains,
        "wrong_to_right": typo_wrong_to_right,
        "right_to_wrong": typo_right_to_wrong,
        "typo_transition": {
            "wrong_to_wrong": typo_wrong_to_wrong,
            "wrong_to_right": typo_wrong_to_right,
            "right_to_wrong": typo_right_to_wrong,
            "right_to_right": typo_right_to_right,
        },
        "typo_exact_mcnemar_pvalue": _exact_mcnemar_pvalue(
            typo_wrong_to_right,
            typo_right_to_wrong,
        ),
        "clean_harm": clean_right_to_wrong,
        "clean_transition": {
            "wrong_to_wrong": clean_wrong_to_wrong,
            "wrong_to_right": clean_wrong_to_right,
            "right_to_wrong": clean_right_to_wrong,
            "right_to_right": clean_right_to_right,
        },
        "clean_exact_mcnemar_pvalue": _exact_mcnemar_pvalue(
            clean_wrong_to_right,
            clean_right_to_wrong,
        ),
        "n_patch_audit_records": len(audit_pairs),
        "n_patch_readout_valid": readout_valid,
        "patch_readout_coverage_fraction": (
            readout_valid / len(audit_pairs) if audit_pairs else None
        ),
        "n_paired_patch_gain": len(base_gains),
        "patch_gain_coverage_fraction": (
            len(base_gains) / len(audit_pairs) if audit_pairs else None
        ),
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
        clean_bounds = _bootstrap_task_macro_difference(
            clean_differences_by_task,
            label=f"{label}:task-macro-clean",
            protocol=protocol,
        )
        typo_bounds = _bootstrap_task_macro_difference(
            typo_differences_by_task,
            label=f"{label}:task-macro-typo",
            protocol=protocol,
        )
        result["task_macro_clean_accuracy_change_ci95_points"] = (
            None if clean_bounds is None else clean_bounds["ci95_points"]
        )
        result["task_macro_clean_accuracy_change_one_sided_95_lower_points"] = (
            None if clean_bounds is None else clean_bounds["one_sided_95_lower_points"]
        )
        result["task_macro_typo_accuracy_gain_ci95_points"] = (
            None if typo_bounds is None else typo_bounds["ci95_points"]
        )
        result["task_macro_typo_accuracy_gain_one_sided_95_lower_points"] = (
            None if typo_bounds is None else typo_bounds["one_sided_95_lower_points"]
        )
    return result


def _bootstrap_method_accuracy_difference(
    differences_by_seed: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    label: str,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object] | None:
    """Hierarchically resample training seeds, then paired items within each seed."""

    if not differences_by_seed:
        return None
    import numpy as np

    seeds = tuple(sorted(differences_by_seed))
    tasks = tuple(sorted(differences_by_seed[seeds[0]]))
    if not tasks or any(tuple(sorted(differences_by_seed[seed])) != tasks for seed in seeds):
        raise ValueError("method-level bootstrap requires one shared task inventory")
    values = {
        seed: {
            task: np.asarray(differences_by_seed[seed][task], dtype=np.float64) for task in tasks
        }
        for seed in seeds
    }
    if any(array.size == 0 for by_task in values.values() for array in by_task.values()):
        raise ValueError("method-level bootstrap requires non-empty paired task cohorts")
    label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(protocol.bootstrap_seed ^ label_seed)
    means = np.empty(protocol.bootstrap_replicates, dtype=np.float64)
    for replicate in range(protocol.bootstrap_replicates):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        seed_means = []
        for seed_index in sampled_seeds:
            seed = seeds[seed_index]
            task_means = []
            for task in tasks:
                array = values[seed][task]
                sampled_items = rng.integers(0, array.size, size=array.size)
                task_means.append(array[sampled_items].mean())
            seed_means.append(fmean(task_means))
        means[replicate] = fmean(seed_means) * 100.0
    return _bootstrap_bounds(means, protocol=protocol)


def _method_paired_metrics(
    base: Sequence[EvaluationObservation],
    adapter_by_seed: Mapping[int, Sequence[EvaluationObservation]],
    *,
    condition: str,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object]:
    seed_metrics: dict[int, dict[str, object]] = {}
    typo_differences: dict[int, Mapping[str, Sequence[float]]] = {}
    clean_differences: dict[int, Mapping[str, Sequence[float]]] = {}
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
        typo_differences[seed] = _answer_differences_by_task(
            answer_pairs,
            field="typo_correct",
        )
        clean_differences[seed] = _answer_differences_by_task(
            answer_pairs,
            field="clean_correct",
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

    typo_gains = values("task_macro_typo_accuracy_gain_points")
    clean_changes = values("task_macro_clean_accuracy_change_points")
    patch_reductions = tuple(
        float(metrics["patch_gain_reduction_fraction"])
        for metrics in seed_metrics.values()
        if isinstance(metrics.get("patch_gain_reduction_fraction"), (int, float))
    )
    typo_bounds = _bootstrap_method_accuracy_difference(
        typo_differences,
        label=f"{condition}:typo",
        protocol=protocol,
    )
    clean_bounds = _bootstrap_method_accuracy_difference(
        clean_differences,
        label=f"{condition}:clean",
        protocol=protocol,
    )
    if typo_bounds is None or clean_bounds is None:
        raise RuntimeError("method-level bootstrap unexpectedly returned no interval")
    task_clean_changes = {
        task: fmean(
            100.0 * fmean(clean_differences[seed][task]) for seed in sorted(clean_differences)
        )
        for task in sorted(next(iter(clean_differences.values())))
    }
    return {
        "n_seeds": len(seed_metrics),
        "seed_inventory": sorted(seed_metrics),
        "seed_inventory_complete": set(seed_metrics) == set(protocol.seed_inventory),
        "typo_accuracy_gain_points_mean": fmean(typo_gains),
        "typo_accuracy_gain_points_sd": stdev(typo_gains) if len(typo_gains) > 1 else 0.0,
        "typo_accuracy_gain_ci95_points": typo_bounds["ci95_points"],
        "typo_accuracy_gain_one_sided_95_lower_points": typo_bounds["one_sided_95_lower_points"],
        "clean_accuracy_change_points_mean": fmean(clean_changes),
        "clean_accuracy_change_points_sd": (
            stdev(clean_changes) if len(clean_changes) > 1 else 0.0
        ),
        "clean_accuracy_change_ci95_points": clean_bounds["ci95_points"],
        "clean_accuracy_change_one_sided_95_lower_points": clean_bounds[
            "one_sided_95_lower_points"
        ],
        "task_clean_accuracy_change_points_mean": task_clean_changes,
        "patch_gain_reduction_fraction_mean": (
            fmean(patch_reductions) if len(patch_reductions) == len(seed_metrics) else None
        ),
        "per_seed": {str(seed): metrics for seed, metrics in seed_metrics.items()},
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _perplexity(
    rows: Sequence[CorpusEvaluationObservation],
    *,
    typo: bool = False,
) -> float | None:
    tokens = sum(row.typo_nll_tokens if typo else row.clean_nll_tokens for row in rows)
    if tokens < 1:
        return None
    mean_nll = sum(row.typo_nll_sum if typo else row.clean_nll_sum for row in rows) / tokens
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return None


def _corpus_condition_summary(
    rows: Sequence[CorpusEvaluationObservation],
) -> dict[str, object]:
    clean_rows = tuple(row for row in rows if row.kind == "clean-corpus")
    natural_rows = tuple(row for row in rows if row.kind == "natural")
    clean_kl = tuple(row.base_clean_kl_sum / row.base_clean_kl_tokens for row in clean_rows)
    natural_tokens = sum(row.natural_clean_typo_kl_tokens for row in natural_rows)
    return {
        "n_records": len(rows),
        "n_clean_corpus": len(clean_rows),
        "n_natural_pairs": len(natural_rows),
        "clean_perplexity": _perplexity(clean_rows),
        "natural_clean_perplexity": _perplexity(natural_rows),
        "natural_typo_perplexity": _perplexity(natural_rows, typo=True),
        "clean_base_forward_kl_median": median(clean_kl) if clean_kl else None,
        "clean_base_forward_kl_p95": _percentile(clean_kl, 0.95),
        "natural_clean_typo_kl_nats_per_token": (
            None
            if natural_tokens < 1
            else sum(row.natural_clean_typo_kl_sum for row in natural_rows) / natural_tokens
        ),
        "sources": {
            source: {
                "n_records": len(source_rows),
                "clean_perplexity": _perplexity(
                    tuple(row for row in source_rows if row.kind == "clean-corpus")
                ),
            }
            for source in sorted({row.source for row in rows})
            if (source_rows := tuple(row for row in rows if row.source == source))
        },
    }


def _corpus_paired_metrics(
    base: Sequence[CorpusEvaluationObservation],
    adapter: Sequence[CorpusEvaluationObservation],
    *,
    study: EvaluationStudyProtocol,
) -> dict[str, object]:
    base_by_id = {row.record_id: row for row in base}
    adapter_by_id = {row.record_id: row for row in adapter}
    if set(base_by_id) != set(adapter_by_id):
        raise ValueError("base and adapter corpus record IDs differ")
    pairs = tuple(
        (base_by_id[record_id], adapter_by_id[record_id]) for record_id in sorted(base_by_id)
    )
    if any(left.kind != right.kind or left.source != right.source for left, right in pairs):
        raise ValueError("base and adapter corpus identities differ")
    ppl_pairs = tuple(
        (left, right)
        for left, right in pairs
        if left.kind == "clean-corpus" and left.source in study.corpus_ppl_sources
    )
    base_ppl = _perplexity(tuple(left for left, _right in ppl_pairs))
    adapter_ppl = _perplexity(tuple(right for _left, right in ppl_pairs))
    ratio = (
        None
        if base_ppl is None or adapter_ppl is None or base_ppl <= 0.0
        else adapter_ppl / base_ppl
    )
    kl_values = tuple(
        right.base_clean_kl_sum / right.base_clean_kl_tokens
        for _left, right in pairs
        if right.kind == "clean-corpus" and right.source == study.corpus_clean_kl_source
    )
    natural_pairs = tuple((left, right) for left, right in pairs if left.kind == "natural")

    def natural_kl(items: Sequence[CorpusEvaluationObservation]) -> float | None:
        tokens = sum(row.natural_clean_typo_kl_tokens for row in items)
        return None if tokens < 1 else sum(row.natural_clean_typo_kl_sum for row in items) / tokens

    base_natural = natural_kl(tuple(left for left, _right in natural_pairs))
    adapter_natural = natural_kl(tuple(right for _left, right in natural_pairs))
    base_natural_typo_ppl = _perplexity(
        tuple(left for left, _right in natural_pairs),
        typo=True,
    )
    adapter_natural_typo_ppl = _perplexity(
        tuple(right for _left, right in natural_pairs),
        typo=True,
    )
    return {
        "n_records": len(pairs),
        "n_ppl_records": len(ppl_pairs),
        "base_clean_perplexity": base_ppl,
        "adapter_clean_perplexity": adapter_ppl,
        "clean_perplexity_ratio": ratio,
        "clean_base_forward_kl_median": median(kl_values) if kl_values else None,
        "clean_base_forward_kl_p95": _percentile(kl_values, 0.95),
        "base_natural_clean_typo_kl_nats_per_token": base_natural,
        "adapter_natural_clean_typo_kl_nats_per_token": adapter_natural,
        "base_natural_typo_perplexity": base_natural_typo_ppl,
        "adapter_natural_typo_perplexity": adapter_natural_typo_ppl,
        "natural_clean_typo_kl_change_nats_per_token": (
            None
            if base_natural is None or adapter_natural is None
            else adapter_natural - base_natural
        ),
        "ppl_sources": list(study.corpus_ppl_sources),
        "clean_kl_source": study.corpus_clean_kl_source,
    }


def _corpus_method_metrics(
    per_seed: Mapping[int, Mapping[str, object]],
    *,
    protocol: RobustnessEvaluationProtocol,
) -> dict[str, object]:
    if not per_seed:
        raise ValueError("corpus method metrics require at least one seed")

    def finite_values(field: str) -> tuple[float, ...]:
        values = tuple(
            float(metrics[field])
            for metrics in per_seed.values()
            if isinstance(metrics.get(field), (int, float)) and math.isfinite(float(metrics[field]))
        )
        if len(values) != len(per_seed):
            raise ValueError(f"corpus method metric {field} is unavailable")
        return values

    ratios = finite_values("clean_perplexity_ratio")
    clean_kl = finite_values("clean_base_forward_kl_median")
    natural_changes = finite_values("natural_clean_typo_kl_change_nats_per_token")
    return {
        "n_seeds": len(per_seed),
        "seed_inventory": sorted(per_seed),
        "seed_inventory_complete": set(per_seed) == set(protocol.seed_inventory),
        "clean_perplexity_ratio_mean": fmean(ratios),
        "clean_perplexity_ratio_max": max(ratios),
        "clean_base_forward_kl_median_mean": fmean(clean_kl),
        "clean_base_forward_kl_median_max": max(clean_kl),
        "natural_clean_typo_kl_change_nats_per_token_mean": fmean(natural_changes),
        "per_seed": {str(seed): dict(metrics) for seed, metrics in sorted(per_seed.items())},
    }


def _finite_at_least(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= threshold


def _finite_at_most(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value <= threshold


def _finite_above(value: object, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > threshold


def build_evaluation_report(
    observations: Sequence[EvaluationObservation],
    *,
    protocol: RobustnessEvaluationProtocol,
    study: EvaluationStudyProtocol,
    corpus_observations: Sequence[CorpusEvaluationObservation],
) -> dict[str, object]:
    """Aggregate strictly paired base/adapter observations and evaluate the gate."""

    rows = tuple(observations)
    if not rows or any(not isinstance(row, EvaluationObservation) for row in rows):
        raise ValueError("evaluation report requires validated observations")
    if not isinstance(study, EvaluationStudyProtocol):
        raise TypeError("evaluation report requires the frozen study protocol")
    corpus_rows = tuple(corpus_observations)
    if not corpus_rows or any(
        not isinstance(row, CorpusEvaluationObservation) for row in corpus_rows
    ):
        raise ValueError("evaluation report requires validated corpus observations")
    corpus_identities = [(row.condition_id, row.record_id) for row in corpus_rows]
    if len(set(corpus_identities)) != len(corpus_identities):
        raise ValueError("evaluation corpus observations duplicate a condition and record")
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
                    tuple(row for row in condition_rows if stratum in row.strata)
                )
                for stratum in _REPORT_STRATA
            },
            "primary_strata": {
                stratum: _condition_summary(
                    tuple(row for row in primary_rows if stratum in row.strata)
                )
                for stratum in _REPORT_STRATA
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
                bootstrap=evaluation_condition == "natural-injection",
            )
        strata = {}
        primary_strata = {}
        for stratum in _REPORT_STRATA:
            base_subset = tuple(row for row in base if stratum in row.strata)
            adapter_subset = tuple(row for row in adapter if stratum in row.strata)
            strata[stratum] = _paired_metrics(
                base_subset,
                adapter_subset,
                label=f"{condition_id}:{stratum}",
                protocol=protocol,
                bootstrap=False,
            )
            primary_base_subset = tuple(row for row in primary_base if stratum in row.strata)
            primary_adapter_subset = tuple(row for row in primary_adapter if stratum in row.strata)
            primary_strata[stratum] = _paired_metrics(
                primary_base_subset,
                primary_adapter_subset,
                label=f"{condition_id}:primary:{stratum}",
                protocol=protocol,
                bootstrap=False,
            )
        comparisons[condition_id] = {
            "primary_condition": _PRIMARY_EVALUATION_CONDITION,
            "overall": overall,
            "all_conditions": all_conditions,
            "evaluation_conditions": evaluation_conditions,
            "strata": strata,
            "primary_strata": primary_strata,
            "tasks": {
                task: _paired_metrics(
                    tuple(row for row in primary_base if row.task == task),
                    tuple(row for row in primary_adapter if row.task == task),
                    label=f"{condition_id}:task:{task}",
                    protocol=protocol,
                    bootstrap=False,
                )
                for task in sorted({row.task for row in primary_base if row.task is not None})
            },
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

    method_evaluation_conditions: dict[str, dict[str, object]] = {}
    for condition, by_seed in sorted(method_rows.items()):
        condition_inventory = sorted({row.evaluation_condition for row in base})
        method_evaluation_conditions[condition] = {
            evaluation_condition: _method_paired_metrics(
                tuple(row for row in base if row.evaluation_condition == evaluation_condition),
                {
                    seed: tuple(
                        row for row in seed_rows if row.evaluation_condition == evaluation_condition
                    )
                    for seed, seed_rows in by_seed.items()
                },
                condition=f"{condition}:{evaluation_condition}",
                protocol=protocol,
            )
            for evaluation_condition in condition_inventory
        }

    corpus_grouped: dict[str, list[CorpusEvaluationObservation]] = defaultdict(list)
    for row in corpus_rows:
        corpus_grouped[row.condition_id].append(row)
    if "base" not in corpus_grouped:
        raise ValueError("evaluation corpus report requires the base condition")
    if set(corpus_grouped) != set(grouped):
        raise ValueError("task and corpus evaluation condition inventories differ")
    corpus_base = tuple(corpus_grouped["base"])
    corpus_conditions = {
        condition_id: _corpus_condition_summary(tuple(condition_rows))
        for condition_id, condition_rows in sorted(corpus_grouped.items())
    }
    corpus_comparisons = {
        condition_id: _corpus_paired_metrics(
            corpus_base,
            tuple(condition_rows),
            study=study,
        )
        for condition_id, condition_rows in sorted(corpus_grouped.items())
        if condition_id != "base"
    }
    corpus_by_method: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for condition_id, metrics in corpus_comparisons.items():
        first = corpus_grouped[condition_id][0]
        if first.seed is None:
            raise RuntimeError("validated adapter corpus observation lost its seed")
        corpus_by_method[first.condition][first.seed] = metrics
    corpus_method_comparisons = {
        condition: _corpus_method_metrics(by_seed, protocol=protocol)
        for condition, by_seed in sorted(corpus_by_method.items())
    }

    gate_seed_checks: dict[str, object] = {}
    directional_seeds: list[int] = []
    gate = study.gates
    for seed in protocol.seed_inventory:
        condition_id = f"localized-state-distillation:seed-{seed}"
        comparison = comparisons.get(condition_id)
        if not isinstance(comparison, dict):
            gate_seed_checks[str(seed)] = {
                "checks": {"nonnegative_clean_change": False, "positive_typo_change": False},
                "passed": False,
            }
            continue
        overall = comparison["overall"]
        if not isinstance(overall, dict):
            raise RuntimeError("validated evaluation comparison changed type")
        checks = {
            "nonnegative_clean_change": _finite_at_least(
                overall["task_macro_clean_accuracy_change_points"], 0.0
            ),
            "positive_typo_change": _finite_above(
                overall["task_macro_typo_accuracy_gain_points"], 0.0
            ),
        }
        passed = all(checks.values())
        gate_seed_checks[str(seed)] = {"checks": checks, "passed": passed}
        if passed:
            directional_seeds.append(seed)
    method = method_comparisons.get("localized-state-distillation")
    natural = method_evaluation_conditions.get("localized-state-distillation", {}).get(
        "natural-injection"
    )
    corpus_method = corpus_method_comparisons.get("localized-state-distillation")
    inventory_complete = (
        isinstance(method, Mapping)
        and method.get("seed_inventory_complete") is True
        and isinstance(natural, Mapping)
        and natural.get("seed_inventory_complete") is True
        and isinstance(corpus_method, Mapping)
        and corpus_method.get("seed_inventory_complete") is True
        and set(map(int, gate_seed_checks)) == set(protocol.seed_inventory)
    )
    task_changes = (
        method.get("task_clean_accuracy_change_points_mean", {})
        if isinstance(method, Mapping)
        else {}
    )
    if not isinstance(task_changes, Mapping):
        raise RuntimeError("validated method task changes changed type")
    confirmatory_checks = {
        "clean_macro_point_noninferiority": (
            isinstance(method, Mapping)
            and _finite_at_least(
                method.get("clean_accuracy_change_points_mean"),
                -float(gate["clean_noninferiority_margin_points"]),
            )
        ),
        "clean_macro_ci_noninferiority": (
            isinstance(method, Mapping)
            and _finite_above(
                method.get("clean_accuracy_change_one_sided_95_lower_points"),
                -float(gate["clean_noninferiority_margin_points"]),
            )
        ),
        "no_task_clean_collapse": bool(task_changes)
        and all(
            _finite_above(value, float(gate["minimum_task_clean_change_points"]))
            for value in task_changes.values()
        ),
        "primary_typo_point_superiority": (
            isinstance(method, Mapping)
            and _finite_at_least(
                method.get("typo_accuracy_gain_points_mean"),
                float(gate["minimum_typo_gain_points"]),
            )
        ),
        "primary_typo_ci_superiority": (
            isinstance(method, Mapping)
            and (
                _finite_above(
                    method.get("typo_accuracy_gain_one_sided_95_lower_points"),
                    0.0,
                )
                if gate["require_typo_ci_lower_above_zero"]
                else True
            )
        ),
        "clean_perplexity": (
            isinstance(corpus_method, Mapping)
            and _finite_at_most(
                corpus_method.get("clean_perplexity_ratio_max"),
                float(gate["maximum_clean_ppl_ratio"]),
            )
        ),
        "clean_forward_kl": (
            isinstance(corpus_method, Mapping)
            and _finite_at_most(
                corpus_method.get("clean_base_forward_kl_median_max"),
                float(gate["maximum_clean_kl_nats_per_token"]),
            )
        ),
        "natural_typo_point_nondegradation": (
            isinstance(natural, Mapping)
            and _finite_at_least(
                natural.get("typo_accuracy_gain_points_mean"),
                float(gate["natural_minimum_point_change"]),
            )
        ),
        "natural_typo_ci_nondegradation": (
            isinstance(natural, Mapping)
            and _finite_above(
                natural.get("typo_accuracy_gain_one_sided_95_lower_points"),
                float(gate["natural_minimum_ci_lower"]),
            )
        ),
        "natural_corpus_kl_nondegradation": (
            isinstance(corpus_method, Mapping)
            and _finite_at_most(
                corpus_method.get("natural_clean_typo_kl_change_nats_per_token_mean"),
                0.0,
            )
        ),
        "directional_seeds": len(directional_seeds) >= int(gate["minimum_directional_seeds"]),
    }
    gate_passed = inventory_complete and all(confirmatory_checks.values())
    return {
        "schema_version": "robustness-evaluation-report/v4",
        "conditions": condition_report,
        "comparisons": comparisons,
        "method_comparisons": method_comparisons,
        "method_evaluation_conditions": method_evaluation_conditions,
        "corpus_conditions": corpus_conditions,
        "corpus_comparisons": corpus_comparisons,
        "corpus_method_comparisons": corpus_method_comparisons,
        "gate": {
            "condition": "localized-state-distillation",
            "seed_inventory_complete": inventory_complete,
            "seed_checks": gate_seed_checks,
            "directional_seeds": directional_seeds,
            "minimum_directional_seeds": gate["minimum_directional_seeds"],
            "checks": confirmatory_checks,
            "thresholds": dict(gate),
            "passed": gate_passed,
        },
    }


__all__ = ["build_evaluation_report"]

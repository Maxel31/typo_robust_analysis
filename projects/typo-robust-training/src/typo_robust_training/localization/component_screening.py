"""Rank activation/gradient screens while explicitly withholding causal status."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typo_robust_training.localization.component_config import (
    ComponentLocalizationProtocol,
)
from typo_robust_training.localization.components import ComponentRef


@dataclass(frozen=True, slots=True)
class ComponentScreenMetric:
    component: ComponentRef
    task: str
    records: int
    activation_difference: float
    gradient_attribution: float

    def __post_init__(self) -> None:
        if not isinstance(self.component, ComponentRef):
            raise TypeError("screen metric component must be ComponentRef")
        if self.task not in {"gsm8k", "mmlu", "arc"}:
            raise ValueError("screen metric task is unsupported")
        if isinstance(self.records, bool) or not isinstance(self.records, int) or self.records <= 0:
            raise ValueError("screen metric records must be a positive integer")
        if (
            not math.isfinite(float(self.activation_difference))
            or float(self.activation_difference) < 0.0
            or not math.isfinite(float(self.gradient_attribution))
        ):
            raise ValueError("screen metrics must be finite and activation difference non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "component": self.component.as_dict(),
            "task": self.task,
            "records": self.records,
            "activation_difference": self.activation_difference,
            "gradient_attribution": self.gradient_attribution,
        }

    @classmethod
    def from_dict(cls, value: object) -> ComponentScreenMetric:
        if not isinstance(value, Mapping) or set(value) != {
            "component",
            "task",
            "records",
            "activation_difference",
            "gradient_attribution",
        }:
            raise ValueError("component screen metric fields differ")
        return cls(
            component=ComponentRef.from_dict(value["component"]),
            task=value["task"],  # type: ignore[arg-type]
            records=value["records"],  # type: ignore[arg-type]
            activation_difference=value["activation_difference"],  # type: ignore[arg-type]
            gradient_attribution=value["gradient_attribution"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ScreenTaskScore:
    activation_percentile: float
    attribution_percentile: float
    combined: float
    raw_activation_difference: float
    raw_gradient_attribution: float


@dataclass(frozen=True, slots=True)
class ScreenCandidate:
    component: ComponentRef
    task_scores: Mapping[str, ScreenTaskScore]
    macro_score: float
    positive_attribution_tasks: int
    layer_shortlisted: bool
    causal_candidate: bool
    screen_only: bool = True
    causally_validated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            **self.component.as_dict(),
            "macro_score": self.macro_score,
            "positive_attribution_tasks": self.positive_attribution_tasks,
            "layer_shortlisted": self.layer_shortlisted,
            "causal_candidate": self.causal_candidate,
            "screen_only": self.screen_only,
            "causally_validated": self.causally_validated,
            "task_scores": {
                task: {
                    "activation_percentile": score.activation_percentile,
                    "attribution_percentile": score.attribution_percentile,
                    "combined": score.combined,
                    "raw_activation_difference": score.raw_activation_difference,
                    "raw_gradient_attribution": score.raw_gradient_attribution,
                }
                for task, score in sorted(self.task_scores.items())
            },
        }


@dataclass(frozen=True, slots=True)
class ComponentScreenResult:
    universe: tuple[ScreenCandidate, ...]
    causal_candidates: tuple[ScreenCandidate, ...]


def _percentiles(values: Mapping[ComponentRef, float]) -> dict[ComponentRef, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0].identifier))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    result: dict[ComponentRef, float] = {}
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        midrank = ((index + stop - 1) / 2.0) / (len(ordered) - 1)
        for position in range(index, stop):
            result[ordered[position][0]] = midrank
        index = stop
    return result


def _expected_components(
    selected_layers: tuple[int, ...], protocol: ComponentLocalizationProtocol
) -> tuple[ComponentRef, ...]:
    return tuple(
        ComponentRef(kind, layer, index)
        for layer in selected_layers
        for kind, count in (
            ("mlp-neuron", protocol.mlp_intermediate_size),
            ("attention-head", protocol.attention_heads),
        )
        for index in range(count)
    )


def rank_component_screen(
    metrics: Sequence[ComponentScreenMetric],
    *,
    selected_layers: Sequence[int],
    protocol: ComponentLocalizationProtocol,
) -> ComponentScreenResult:
    """Rank complete screen metrics and return quota-limited causal candidates."""

    layers = tuple(selected_layers)
    if not layers or len(set(layers)) != len(layers) or tuple(sorted(layers)) != layers:
        raise ValueError("selected layers must be unique and strictly increasing")
    if any(layer < 0 or layer >= protocol.decoder_layers for layer in layers):
        raise ValueError("selected layer lies outside the frozen architecture")
    expected = _expected_components(layers, protocol)
    expected_keys = {(component, task) for component in expected for task in protocol.tasks}
    by_key: dict[tuple[ComponentRef, str], ComponentScreenMetric] = {}
    for metric in metrics:
        if not isinstance(metric, ComponentScreenMetric):
            raise TypeError("screen metrics must be ComponentScreenMetric records")
        key = metric.component, metric.task
        if key in by_key:
            raise ValueError("component screen metric keys are duplicated")
        by_key[key] = metric
    if set(by_key) != expected_keys:
        raise ValueError("component screen does not cover the complete selected-layer universe")

    task_scores: dict[ComponentRef, dict[str, ScreenTaskScore]] = defaultdict(dict)
    for task in protocol.tasks:
        for layer in layers:
            for kind in ("mlp-neuron", "attention-head"):
                group = tuple(
                    component
                    for component in expected
                    if component.layer == layer and component.kind == kind
                )
                activation = {
                    component: by_key[(component, task)].activation_difference
                    for component in group
                }
                attribution = {
                    component: by_key[(component, task)].gradient_attribution for component in group
                }
                activation_ranks = _percentiles(activation)
                attribution_ranks = _percentiles(attribution)
                for component in group:
                    combined = (
                        protocol.activation_weight * activation_ranks[component]
                        + protocol.attribution_weight * attribution_ranks[component]
                    )
                    task_scores[component][task] = ScreenTaskScore(
                        activation_percentile=activation_ranks[component],
                        attribution_percentile=attribution_ranks[component],
                        combined=combined,
                        raw_activation_difference=activation[component],
                        raw_gradient_attribution=attribution[component],
                    )

    provisional: dict[ComponentRef, tuple[float, int]] = {}
    for component in expected:
        scores = task_scores[component]
        macro = sum(score.combined for score in scores.values()) / len(protocol.tasks)
        positive = sum(score.raw_gradient_attribution > 0.0 for score in scores.values())
        provisional[component] = macro, positive

    layer_shortlist: set[ComponentRef] = set()
    for layer in layers:
        for kind, limit in (
            ("mlp-neuron", protocol.mlp_shortlist_per_layer),
            ("attention-head", protocol.attention_shortlist_per_layer),
        ):
            eligible = [
                component
                for component in expected
                if component.layer == layer
                and component.kind == kind
                and provisional[component][1] >= protocol.minimum_positive_attribution_tasks
            ]
            eligible.sort(key=lambda component: (-provisional[component][0], component.identifier))
            layer_shortlist.update(eligible[:limit])

    causal_set: set[ComponentRef] = set()
    for kind, limit in sorted(protocol.causal_candidate_limits.items()):
        eligible = [component for component in layer_shortlist if component.kind == kind]
        eligible.sort(key=lambda component: (-provisional[component][0], component.identifier))
        causal_set.update(eligible[:limit])
    universe = tuple(
        ScreenCandidate(
            component=component,
            task_scores=task_scores[component],
            macro_score=provisional[component][0],
            positive_attribution_tasks=provisional[component][1],
            layer_shortlisted=component in layer_shortlist,
            causal_candidate=component in causal_set,
        )
        for component in sorted(expected, key=lambda item: item.identifier)
    )
    causal = tuple(candidate for candidate in universe if candidate.component in causal_set)
    return ComponentScreenResult(universe=universe, causal_candidates=causal)


__all__ = [
    "ComponentScreenMetric",
    "ComponentScreenResult",
    "ScreenCandidate",
    "ScreenTaskScore",
    "rank_component_screen",
]

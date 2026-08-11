"""Frozen protocol for diagnostic selection and disjoint window evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typo_cot.experiments.build_rebuttal_manifest.records import strict_loads

_CANDIDATES = ((0, 6), (6, 12), (12, 18), (18, 24), (22, 28))
_ARMS = ("selected", "runner-up")
_GENERATION = {
    "strategy": "greedy",
    "dtype": "bfloat16",
    "padding_side": "left",
    "max_new_tokens": 512,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
    "termination_protocol": "effective-eos-vs-length-cap/v1",
}


@dataclass(frozen=True, slots=True)
class HeldOutWindowProtocol:
    """Strict public configuration resolved from the versioned JSON/YAML file."""

    schema_version: str
    source_operation: str
    cohort_ids_filename: str
    selection_cohort: str
    evaluation_cohort: str
    split_algorithm: str
    split_seed: int
    require_disjoint: bool
    direction: str
    site: str
    coordinate_source: str
    diagnostic_readout: str
    candidate_windows: tuple[tuple[int, int], ...]
    window_width: int
    target_availability: str
    untreated_kl_min_exclusive: float
    selection_metric: str
    cross_setting_score: str
    ranking: str
    select_top_k: int
    evaluation_arms: tuple[str, ...]
    answer_target: str
    answer_extraction: str
    paired_test: str
    effect: str
    pair_bootstrap_replicates: int
    nested_bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    multiplicity: str
    macro: str
    reproduction_rule: str
    smoke_inference: str
    config_sha256: str

    @property
    def generation(self) -> dict[str, object]:
        return dict(_GENERATION)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "operation": self.source_operation,
                "cohort_ids_filename": self.cohort_ids_filename,
                "selection_cohort": self.selection_cohort,
                "evaluation_cohort": self.evaluation_cohort,
                "split_algorithm": self.split_algorithm,
                "split_seed": self.split_seed,
                "require_disjoint": self.require_disjoint,
            },
            "diagnostic": {
                "direction": self.direction,
                "site": self.site,
                "coordinates": self.coordinate_source,
                "readout": self.diagnostic_readout,
                "candidate_windows": [list(window) for window in self.candidate_windows],
                "window_width": self.window_width,
                "target_availability": self.target_availability,
                "untreated_kl_min_exclusive": self.untreated_kl_min_exclusive,
                "selection_metric": self.selection_metric,
                "cross_setting_score": self.cross_setting_score,
                "ranking": self.ranking,
                "select_top_k": self.select_top_k,
            },
            "evaluation": {
                "arms": list(self.evaluation_arms),
                "answer_target": self.answer_target,
                "answer_extraction": self.answer_extraction,
                "generation": self.generation,
            },
            "statistics": {
                "paired_test": self.paired_test,
                "effect": self.effect,
                "pair_bootstrap_replicates": self.pair_bootstrap_replicates,
                "nested_bootstrap_replicates": self.nested_bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "multiplicity": self.multiplicity,
                "macro": self.macro,
                "reproduction_rule": self.reproduction_rule,
                "smoke_inference": self.smoke_inference,
            },
        }


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"held-out {field} must be an object")
    if set(value) != fields:
        raise ValueError(f"held-out {field} fields differ")
    return value


def load_held_out_window_protocol(path: Path) -> HeldOutWindowProtocol:
    """Load the one accepted v1 protocol and reject every silent drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"held-out window config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"held-out window config is not valid UTF-8: {resolved}") from exc
    payload = strict_loads(text, context=str(resolved))
    root = _mapping(
        payload,
        field="config",
        fields={"schema_version", "source", "diagnostic", "evaluation", "statistics"},
    )
    if root["schema_version"] != "held-out-window-evaluation-config/v1":
        raise ValueError("held-out window schema_version differs")
    source = _mapping(
        root["source"],
        field="source",
        fields={
            "operation",
            "cohort_ids_filename",
            "selection_cohort",
            "evaluation_cohort",
            "split_algorithm",
            "split_seed",
            "require_disjoint",
        },
    )
    expected_source = {
        "operation": "build-rebuttal-manifest",
        "cohort_ids_filename": "cohort_ids.json",
        "selection_cohort": "window_selection",
        "evaluation_cohort": "window_evaluation",
        "split_algorithm": "sha256-order-sample-group-half-per-task/v2",
        "split_seed": 42,
        "require_disjoint": True,
    }
    if dict(source) != expected_source:
        raise ValueError("held-out source contract differs")
    diagnostic = _mapping(
        root["diagnostic"],
        field="diagnostic",
        fields={
            "direction",
            "site",
            "coordinates",
            "readout",
            "candidate_windows",
            "window_width",
            "target_availability",
            "untreated_kl_min_exclusive",
            "selection_metric",
            "cross_setting_score",
            "ranking",
            "select_top_k",
        },
    )
    expected_diagnostic = {
        "direction": "clean-to-typo",
        "site": "complete-decoder-block-residual-output",
        "coordinates": "manifest-edited-word-final-token/v1",
        "readout": "first-clean-continuation-token-distribution/v1",
        "candidate_windows": [list(window) for window in _CANDIDATES],
        "window_width": 6,
        "target_availability": "exact-clean-prompt-token-prefix-and-one-suffix-token/v1",
        "untreated_kl_min_exclusive": 1e-9,
        "selection_metric": "median-normalized-first-token-kl-restoration/v1",
        "cross_setting_score": "equal-model-task-target-rule-cell-macro-mean/v1",
        "ranking": "score-descending-then-start-ascending/v1",
        "select_top_k": 2,
    }
    if dict(diagnostic) != expected_diagnostic:
        raise ValueError("held-out diagnostic contract differs")
    evaluation = _mapping(
        root["evaluation"],
        field="evaluation",
        fields={"arms", "answer_target", "answer_extraction", "generation"},
    )
    expected_evaluation = {
        "arms": list(_ARMS),
        "answer_target": "manifest-stored-clean-answer/v1",
        "answer_extraction": "primary-then-empty-only-positional/v1",
        "generation": _GENERATION,
    }
    if dict(evaluation) != expected_evaluation:
        raise ValueError("held-out evaluation contract differs")
    statistics = _mapping(
        root["statistics"],
        field="statistics",
        fields={
            "paired_test",
            "effect",
            "pair_bootstrap_replicates",
            "nested_bootstrap_replicates",
            "bootstrap_seed",
            "confidence_level",
            "multiplicity",
            "macro",
            "reproduction_rule",
            "smoke_inference",
        },
    )
    expected_statistics = {
        "paired_test": "exact-mcnemar-two-sided-conditional-binomial/v1",
        "effect": "paired-risk-difference-selected-minus-runner-up/v1",
        "pair_bootstrap_replicates": 10_000,
        "nested_bootstrap_replicates": 10_000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
        "multiplicity": "holm-6-held-out-setting-contrasts/v1",
        "macro": "equal-setting-nested-bootstrap/v1",
        "reproduction_rule": "selected-is-0-6-and-held-out-macro-ci-lower-gt-zero/v1",
        "smoke_inference": "descriptive-not-in-confirmatory-family/v1",
    }
    if dict(statistics) != expected_statistics:
        raise ValueError("held-out statistics contract differs")
    return HeldOutWindowProtocol(
        schema_version="held-out-window-evaluation-config/v1",
        source_operation=str(source["operation"]),
        cohort_ids_filename=str(source["cohort_ids_filename"]),
        selection_cohort=str(source["selection_cohort"]),
        evaluation_cohort=str(source["evaluation_cohort"]),
        split_algorithm=str(source["split_algorithm"]),
        split_seed=int(source["split_seed"]),
        require_disjoint=True,
        direction=str(diagnostic["direction"]),
        site=str(diagnostic["site"]),
        coordinate_source=str(diagnostic["coordinates"]),
        diagnostic_readout=str(diagnostic["readout"]),
        candidate_windows=_CANDIDATES,
        window_width=6,
        target_availability=str(diagnostic["target_availability"]),
        untreated_kl_min_exclusive=1e-9,
        selection_metric=str(diagnostic["selection_metric"]),
        cross_setting_score=str(diagnostic["cross_setting_score"]),
        ranking=str(diagnostic["ranking"]),
        select_top_k=2,
        evaluation_arms=_ARMS,
        answer_target=str(evaluation["answer_target"]),
        answer_extraction=str(evaluation["answer_extraction"]),
        paired_test=str(statistics["paired_test"]),
        effect=str(statistics["effect"]),
        pair_bootstrap_replicates=10_000,
        nested_bootstrap_replicates=10_000,
        bootstrap_seed=42,
        confidence_level=0.95,
        multiplicity=str(statistics["multiplicity"]),
        macro=str(statistics["macro"]),
        reproduction_rule=str(statistics["reproduction_rule"]),
        smoke_inference=str(statistics["smoke_inference"]),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["HeldOutWindowProtocol", "load_held_out_window_protocol"]

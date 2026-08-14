"""Strict, model-independent typo-robustness evaluation study contract."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from typo_robust_training.data.config import strict_loads


_TOP = {
    "schema_version",
    "protocol_id",
    "seed",
    "tiers",
    "typos",
    "corpora",
    "corpus_runtime",
    "audit",
    "generation",
    "statistics",
    "gates",
    "opening",
}
_TIERS = {"monitor", "tune", "pre_pr_gate", "final_test"}
_MONITOR = {"task_accuracy_allowed"}
_TUNE = {
    "repeatable",
    "task_records_total",
    "fineweb_documents",
    "natural_pairs",
    "natural_injection_records",
}
_SEALED_TIER = {"maximum_openings", "records_per_task", "tasks"}
_TYPOS = {"primary", "secondary", "eligibility"}
_PRIMARY = {"condition", "edit_count", "operations", "targeting"}
_SECONDARY = {
    "severity_edit_counts",
    "severity_records_total",
    "held_out_condition",
    "held_out_records_total",
    "natural_injection_records_total",
    "natural_injection_edit_count",
}
_ELIGIBILITY = {
    "minimum_word_letters",
    "distinct_words",
    "question_only",
    "forbid_option_text",
    "forbid_gold_answer",
    "forbid_numbers_math_urls_email_identifiers",
}
_CORPUS_ROLES = {"tune", "pre_pr_gate", "final_test"}
_CORPUS_COUNTS = {"fineweb_edu", "dolma", "natural_pairs"}
_CORPUS_RUNTIME = {
    "max_tokens",
    "truncation",
    "ppl_sources",
    "clean_kl_source",
    "natural_alignment",
}
_AUDIT = {"tune_records", "sealed_records", "condition", "blocking", "operator"}
_GENERATION = {
    "prompt_protocol",
    "answer_extraction",
    "shots",
    "dtype",
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "temperature",
    "top_p",
    "top_k",
    "use_cache",
    "termination_protocol",
}
_STATISTICS = {
    "training_seeds",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence_level",
    "bootstrap",
    "paired_test",
    "primary_multiplicity",
}
_GATES = {
    "clean_noninferiority_margin_points",
    "minimum_task_clean_change_points",
    "minimum_typo_gain_points",
    "require_typo_ci_lower_above_zero",
    "maximum_clean_ppl_ratio",
    "maximum_clean_kl_nats_per_token",
    "minimum_directional_seeds",
    "natural_minimum_point_change",
    "natural_minimum_ci_lower",
    "patch_audit_is_blocking",
}
_OPENING = {
    "pre_pr_requires_frozen_arm_seed_config_checkpoint_hashes",
    "final_requires_passing_pre_pr",
    "final_requires_all_models_and_analyses_frozen",
    "resume_requires_exact_access_binding",
}
_TASKS = ("gsm8k", "mmlu", "arc", "mmlu_pro", "math_500", "commonsense_qa")
_PRE_PR_TASKS = ("gsm8k", "mmlu", "arc", "mmlu_pro", "commonsense_qa")
_FINAL_TASKS = _TASKS
_PRIMARY_OPERATIONS = (
    "keyboard-neighbor-substitution",
    "deletion",
    "duplication",
)


def _mapping(value: object, *, field: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"evaluation study {field} fields differ")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"evaluation study {field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"evaluation study {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"evaluation study {field} must be a finite number")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"evaluation study {field} must be boolean")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation study {field} must be non-empty text")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"evaluation study {field} must contain unique strings")
    return tuple(value)


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"evaluation study {field} must contain integers")
    values = tuple(_integer(item, field=field) for item in value)
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"evaluation study {field} must be sorted and unique")
    return values


@dataclass(frozen=True, slots=True)
class EvaluationStudyProtocol:
    """Resolved v1 study contract used by data freezing and statistics."""

    schema_version: str
    protocol_id: str
    seed: int
    training_seeds: tuple[int, ...]
    monitor_task_accuracy_allowed: bool
    tune_task_records_total: int
    tune_fineweb_documents: int
    tune_natural_pairs: int
    tune_natural_injection_records: int
    role_tasks: Mapping[str, tuple[str, ...]]
    records_per_task: Mapping[str, Mapping[str, int]]
    maximum_openings: Mapping[str, int]
    primary_typo_condition: str
    primary_edit_count: int
    primary_operations: tuple[str, ...]
    primary_targeting: str
    severity_edit_counts: tuple[int, ...]
    severity_records_total: int
    held_out_condition: str
    held_out_records_total: int
    natural_injection_records_total: int
    natural_injection_edit_count: int
    minimum_word_letters: int
    distinct_words: bool
    question_only: bool
    forbid_option_text: bool
    forbid_gold_answer: bool
    forbid_numbers_math_urls_email_identifiers: bool
    corpus_counts: Mapping[str, Mapping[str, int]]
    corpus_max_tokens: int
    corpus_truncation: str
    corpus_ppl_sources: tuple[str, ...]
    corpus_clean_kl_source: str
    corpus_natural_alignment: str
    audit_records: Mapping[str, int]
    shots: Mapping[str, int]
    max_new_tokens: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    gates: Mapping[str, object]
    opening: Mapping[str, bool]
    config_sha256: str


def load_evaluation_study_protocol(path: Path) -> EvaluationStudyProtocol:
    """Load evaluation v1 while rejecting any undeclared scientific drift."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"evaluation study config is not a file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = strict_loads(raw.decode("utf-8"), context=str(resolved))
    except UnicodeDecodeError as exc:
        raise ValueError("evaluation study config must be UTF-8") from exc
    top = _mapping(payload, field="config", fields=_TOP)
    if (
        top["schema_version"] != "robustness-evaluation-study/v1"
        or top["protocol_id"] != "typo-robustness-evaluation-v1.4"
        or top["seed"] != 42
    ):
        raise ValueError("evaluation study identity differs")

    tiers = _mapping(top["tiers"], field="tiers", fields=_TIERS)
    monitor = _mapping(tiers["monitor"], field="tiers.monitor", fields=_MONITOR)
    tune = _mapping(tiers["tune"], field="tiers.tune", fields=_TUNE)
    pre_pr = _mapping(tiers["pre_pr_gate"], field="tiers.pre_pr_gate", fields=_SEALED_TIER)
    final = _mapping(tiers["final_test"], field="tiers.final_test", fields=_SEALED_TIER)
    if (
        _boolean(monitor["task_accuracy_allowed"], field="monitor.task_accuracy_allowed")
        or not _boolean(tune["repeatable"], field="tune.repeatable")
        or tune["task_records_total"] != 500
        or tune["fineweb_documents"] != 200
        or tune["natural_pairs"] != 100
        or tune["natural_injection_records"] != 100
    ):
        raise ValueError("evaluation study tier access policy differs")
    pre_pr_tasks = _string_tuple(pre_pr["tasks"], field="pre_pr_gate.tasks")
    final_tasks = _string_tuple(final["tasks"], field="final_test.tasks")
    if pre_pr_tasks != _PRE_PR_TASKS or final_tasks != _FINAL_TASKS:
        raise ValueError("evaluation study sealed task inventory differs")
    if any(
        _integer(tier["maximum_openings"], field=f"{name}.maximum_openings", minimum=1) != 1
        for name, tier in (("pre_pr_gate", pre_pr), ("final_test", final))
    ):
        raise ValueError("evaluation study sealed opening count differs")
    pre_pr_counts_raw = _mapping(
        pre_pr["records_per_task"],
        field="pre_pr_gate.records_per_task",
        fields=set(pre_pr_tasks),
    )
    final_counts_raw = _mapping(
        final["records_per_task"],
        field="final_test.records_per_task",
        fields=set(final_tasks),
    )
    pre_pr_counts = {
        task: _integer(
            pre_pr_counts_raw[task], field=f"pre_pr_gate.records_per_task.{task}", minimum=1
        )
        for task in pre_pr_tasks
    }
    final_counts = {
        task: _integer(
            final_counts_raw[task], field=f"final_test.records_per_task.{task}", minimum=1
        )
        for task in final_tasks
    }
    expected_pre_pr_counts = {task: 500 for task in pre_pr_tasks}
    expected_final_counts = {task: (440 if task == "math_500" else 500) for task in final_tasks}
    if pre_pr_counts != expected_pre_pr_counts or final_counts != expected_final_counts:
        raise ValueError("evaluation study sealed task sample size differs")

    typos = _mapping(top["typos"], field="typos", fields=_TYPOS)
    primary = _mapping(typos["primary"], field="typos.primary", fields=_PRIMARY)
    primary_operations = _string_tuple(primary["operations"], field="primary.operations")
    if (
        primary["condition"] != "random-2"
        or primary["edit_count"] != 2
        or primary_operations != _PRIMARY_OPERATIONS
        or primary["targeting"] != "uniform-eligible-question-words-without-replacement/v1"
    ):
        raise ValueError("evaluation study primary typo protocol differs")
    secondary = _mapping(typos["secondary"], field="typos.secondary", fields=_SECONDARY)
    severity = _int_tuple(secondary["severity_edit_counts"], field="severity_edit_counts")
    expected_secondary = {
        "severity_edit_counts": (1, 4),
        "severity_records_total": 1200,
        "held_out_condition": "transposition-2",
        "held_out_records_total": 500,
        "natural_injection_records_total": 500,
        "natural_injection_edit_count": 1,
    }
    actual_secondary = {
        "severity_edit_counts": severity,
        "severity_records_total": secondary["severity_records_total"],
        "held_out_condition": secondary["held_out_condition"],
        "held_out_records_total": secondary["held_out_records_total"],
        "natural_injection_records_total": secondary["natural_injection_records_total"],
        "natural_injection_edit_count": secondary["natural_injection_edit_count"],
    }
    if actual_secondary != expected_secondary:
        raise ValueError("evaluation study secondary typo protocol differs")
    eligibility = _mapping(typos["eligibility"], field="typos.eligibility", fields=_ELIGIBILITY)
    if eligibility["minimum_word_letters"] != 3 or any(
        not _boolean(eligibility[field], field=f"eligibility.{field}")
        for field in _ELIGIBILITY - {"minimum_word_letters"}
    ):
        raise ValueError("evaluation study edit eligibility differs")

    corpora_raw = _mapping(top["corpora"], field="corpora", fields=_CORPUS_ROLES)
    corpus_counts: dict[str, Mapping[str, int]] = {}
    expected_corpus_counts = {
        "tune": {"fineweb_edu": 200, "dolma": 0, "natural_pairs": 100},
        "pre_pr_gate": {"fineweb_edu": 1000, "dolma": 500, "natural_pairs": 500},
        "final_test": {"fineweb_edu": 1000, "dolma": 1000, "natural_pairs": 1000},
    }
    for role in ("tune", "pre_pr_gate", "final_test"):
        counts = _mapping(corpora_raw[role], field=f"corpora.{role}", fields=_CORPUS_COUNTS)
        parsed = {
            source: _integer(counts[source], field=f"corpora.{role}.{source}")
            for source in _CORPUS_COUNTS
        }
        if parsed != expected_corpus_counts[role]:
            raise ValueError("evaluation study corpus sample sizes differ")
        corpus_counts[role] = MappingProxyType(parsed)

    corpus_runtime = _mapping(top["corpus_runtime"], field="corpus_runtime", fields=_CORPUS_RUNTIME)
    corpus_ppl_sources = _string_tuple(
        corpus_runtime["ppl_sources"], field="corpus_runtime.ppl_sources"
    )
    if (
        corpus_runtime["max_tokens"] != 512
        or corpus_runtime["truncation"] != "tokenizer-right-prefix/v1"
        or corpus_ppl_sources != ("fineweb_edu", "dolma")
        or corpus_runtime["clean_kl_source"] != "fineweb_edu"
        or corpus_runtime["natural_alignment"] != "exact-unchanged-token-spans/v1"
    ):
        raise ValueError("evaluation study corpus runtime differs")

    audit = _mapping(top["audit"], field="audit", fields=_AUDIT)
    if (
        audit["tune_records"] != 100
        or audit["sealed_records"] != 500
        or audit["condition"] != "random-2"
        or _boolean(audit["blocking"], field="audit.blocking")
        or audit["operator"] != "paired-clean-to-typo-frozen-window/v1"
    ):
        raise ValueError("evaluation study mechanistic audit protocol differs")

    generation = _mapping(top["generation"], field="generation", fields=_GENERATION)
    shots_raw = _mapping(generation["shots"], field="generation.shots", fields=set(_TASKS))
    shots = {task: _integer(shots_raw[task], field=f"shots.{task}") for task in _TASKS}
    expected_shots = {
        "gsm8k": 8,
        "mmlu": 5,
        "arc": 5,
        "mmlu_pro": 5,
        "math_500": 4,
        "commonsense_qa": 5,
    }
    expected_generation = {
        "prompt_protocol": "paper-cot-templates/v1",
        "answer_extraction": "paper-task-extractors/v1",
        "dtype": "bfloat16",
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
        "termination_protocol": "effective-eos-vs-length-cap/v1",
    }
    if shots != expected_shots or any(
        generation[key] != value for key, value in expected_generation.items()
    ):
        raise ValueError("evaluation study generation protocol differs")

    statistics = _mapping(top["statistics"], field="statistics", fields=_STATISTICS)
    training_seeds = _int_tuple(statistics["training_seeds"], field="training_seeds")
    confidence = _number(statistics["confidence_level"], field="confidence_level")
    if (
        training_seeds != (42, 43, 44)
        or statistics["bootstrap_replicates"] != 10_000
        or statistics["bootstrap_seed"] != 42
        or confidence != 0.95
        or statistics["bootstrap"] != "hierarchical-learning-seed-task-item/v1"
        or statistics["paired_test"] != "exact-mcnemar/v1"
        or statistics["primary_multiplicity"] != "intersection-union-no-adjustment/v1"
    ):
        raise ValueError("evaluation study statistical protocol differs")

    gates_raw = _mapping(top["gates"], field="gates", fields=_GATES)
    gates = dict(gates_raw)
    expected_gates: dict[str, object] = {
        "clean_noninferiority_margin_points": 1.0,
        "minimum_task_clean_change_points": -3.0,
        "minimum_typo_gain_points": 2.0,
        "require_typo_ci_lower_above_zero": True,
        "maximum_clean_ppl_ratio": 1.02,
        "maximum_clean_kl_nats_per_token": 0.03,
        "minimum_directional_seeds": 2,
        "natural_minimum_point_change": -1.0,
        "natural_minimum_ci_lower": -2.0,
        "patch_audit_is_blocking": False,
    }
    if gates != expected_gates:
        raise ValueError("evaluation study gate protocol differs")
    opening_raw = _mapping(top["opening"], field="opening", fields=_OPENING)
    opening = {key: _boolean(value, field=f"opening.{key}") for key, value in opening_raw.items()}
    if not all(opening.values()):
        raise ValueError("evaluation study opening protocol differs")

    return EvaluationStudyProtocol(
        schema_version=str(top["schema_version"]),
        protocol_id=str(top["protocol_id"]),
        seed=42,
        training_seeds=training_seeds,
        monitor_task_accuracy_allowed=False,
        tune_task_records_total=_integer(
            tune["task_records_total"], field="tune.task_records_total", minimum=1
        ),
        tune_fineweb_documents=_integer(
            tune["fineweb_documents"], field="tune.fineweb_documents", minimum=1
        ),
        tune_natural_pairs=_integer(tune["natural_pairs"], field="tune.natural_pairs", minimum=1),
        tune_natural_injection_records=_integer(
            tune["natural_injection_records"],
            field="tune.natural_injection_records",
            minimum=1,
        ),
        role_tasks=MappingProxyType({"pre_pr_gate": pre_pr_tasks, "final_test": final_tasks}),
        records_per_task=MappingProxyType(
            {
                "pre_pr_gate": MappingProxyType(pre_pr_counts),
                "final_test": MappingProxyType(final_counts),
            }
        ),
        maximum_openings=MappingProxyType({"pre_pr_gate": 1, "final_test": 1}),
        primary_typo_condition="random-2",
        primary_edit_count=2,
        primary_operations=primary_operations,
        primary_targeting=str(primary["targeting"]),
        severity_edit_counts=severity,
        severity_records_total=1200,
        held_out_condition="transposition-2",
        held_out_records_total=500,
        natural_injection_records_total=500,
        natural_injection_edit_count=1,
        minimum_word_letters=3,
        distinct_words=True,
        question_only=True,
        forbid_option_text=True,
        forbid_gold_answer=True,
        forbid_numbers_math_urls_email_identifiers=True,
        corpus_counts=MappingProxyType(corpus_counts),
        corpus_max_tokens=512,
        corpus_truncation="tokenizer-right-prefix/v1",
        corpus_ppl_sources=corpus_ppl_sources,
        corpus_clean_kl_source="fineweb_edu",
        corpus_natural_alignment="exact-unchanged-token-spans/v1",
        audit_records=MappingProxyType({"tune": 100, "pre_pr_gate": 500, "final_test": 500}),
        shots=MappingProxyType(shots),
        max_new_tokens=512,
        bootstrap_replicates=10_000,
        bootstrap_seed=42,
        confidence_level=0.95,
        gates=MappingProxyType(gates),
        opening=MappingProxyType(opening),
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = ["EvaluationStudyProtocol", "load_evaluation_study_protocol"]

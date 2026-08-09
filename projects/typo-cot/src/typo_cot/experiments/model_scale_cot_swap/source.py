"""Fail-closed discovery of Table 9 pair and CoT-swap producers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typo_cot.data.cohorts import SampleIdCohort, load_sample_id_cohort
from typo_cot.experiments.edit_count_sensitivity.source import (
    CotSwapEditCountRun,
    EditCountSensitivityInputError,
    PreparedEditCountRun,
    discover_cot_swap_runs,
    discover_prepared_runs,
)
from typo_cot.experiments.model_scale_cot_swap.protocol import (
    COHORT_ID,
    COHORT_SAMPLE_COUNT,
    COHORT_SAMPLE_IDS_SHA256,
    COHORT_SELECTION,
    EXPECTED_MODELS,
    EXPECTED_SETTINGS,
    MODEL_SAMPLES_PER_SUBSET,
    MODEL_SELECTED_SAMPLE_COUNTS,
    MODEL_SELECTED_SAMPLE_IDS_SHA256,
)


class ModelScaleCotSwapInputError(ValueError):
    """An input cannot support an auditable Appendix C/Table 9 analysis."""


@dataclass(frozen=True, slots=True)
class ModelScaleInputs:
    """One validated cohort plus its linked producer runs."""

    cohort: SampleIdCohort
    prepared_runs: tuple[PreparedEditCountRun, ...]
    cot_swap_runs: tuple[CotSwapEditCountRun, ...]


def _validate_table9_cohort(cohort: SampleIdCohort) -> None:
    expected_scalars = {
        "cohort_id": COHORT_ID,
        "benchmark": "mmlu",
        "selection": COHORT_SELECTION,
        "provenance": "submitted-source-recovered",
        "sample_ids_sha256": COHORT_SAMPLE_IDS_SHA256,
    }
    for field, expected in expected_scalars.items():
        if getattr(cohort, field) != expected:
            raise ModelScaleCotSwapInputError(f"Table 9 cohort {field} must be {expected!r}")
    if len(cohort.sample_ids) != COHORT_SAMPLE_COUNT:
        raise ModelScaleCotSwapInputError(
            f"Table 9 cohort must contain exactly {COHORT_SAMPLE_COUNT} sample IDs"
        )
    if dict(cohort.model_samples_per_subset) != MODEL_SAMPLES_PER_SUBSET:
        raise ModelScaleCotSwapInputError(
            "Table 9 cohort model samples-per-subset map does not match the submitted protocol"
        )
    if dict(cohort.model_selected_sample_counts) != MODEL_SELECTED_SAMPLE_COUNTS:
        raise ModelScaleCotSwapInputError(
            "Table 9 cohort model selected-count map does not match the submitted protocol"
        )
    if dict(cohort.model_selected_sample_ids_sha256) != MODEL_SELECTED_SAMPLE_IDS_SHA256:
        raise ModelScaleCotSwapInputError(
            "Table 9 cohort model selected sample-ID map does not match the submitted protocol"
        )


def discover_model_scale_inputs(
    *,
    pairs_root: Path,
    cot_swap_runs_root: Path,
    cohort_path: Path,
) -> ModelScaleInputs:
    """Validate the applicable four-edit MMLU producer grid and all links."""
    try:
        cohort = load_sample_id_cohort(cohort_path)
        _validate_table9_cohort(cohort)
        prepared = discover_prepared_runs(
            pairs_root,
            edit_counts=(4,),
            expected_settings=EXPECTED_SETTINGS,
            sample_id_cohort=cohort,
        )
        cot_swap = discover_cot_swap_runs(
            cot_swap_runs_root,
            edit_counts=(4,),
            prepared_runs=prepared,
            expected_settings=EXPECTED_SETTINGS,
        )
    except EditCountSensitivityInputError as exc:
        raise ModelScaleCotSwapInputError(str(exc)) from exc
    except ModelScaleCotSwapInputError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ModelScaleCotSwapInputError(str(exc)) from exc

    prepared_models = {run.model for run in prepared}
    cot_models = {run.model for run in cot_swap}
    if not cot_models.issubset(prepared_models):
        raise ModelScaleCotSwapInputError(
            "Table 9 CoT-swap settings are missing validated pair preparations"
        )
    if any(run.benchmark != "mmlu" or run.model not in EXPECTED_MODELS for run in cot_swap):
        raise ModelScaleCotSwapInputError("Table 9 contains an unexpected producer setting")
    return ModelScaleInputs(
        cohort=cohort,
        prepared_runs=prepared,
        cot_swap_runs=cot_swap,
    )


__all__ = [
    "ModelScaleCotSwapInputError",
    "ModelScaleInputs",
    "discover_model_scale_inputs",
]

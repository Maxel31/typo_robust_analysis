"""Machine-readable inventory of the experiments reported in the final paper.

The operation slugs in this module are the stable public names.  RQ labels are
kept only as paper cross-references because they do not describe what a command
does.  Execution implementations are intentionally added one operation at a
time; ``status`` makes that migration state explicit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal, get_args

# Contract fingerprint of the user-supplied final PDF. Public documentation
# reads it through ``typo-cot experiments source`` so this is its sole runtime
# source. Change it only when the artifact is replaced and the catalog re-audited.
PAPER_SHA256 = "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"

ComputeClass = Literal["cpu", "gpu"]
ExperimentStatus = Literal["catalogued", "implemented"]
_COMPUTE_CLASSES = frozenset(get_args(ComputeClass))
_EXPERIMENT_STATUSES = frozenset(get_args(ExperimentStatus))
_PUBLIC_SLUG_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_PAPER_QUESTION_SLUG_PATTERN = re.compile(r"(?:^|-)rq[0-9]+(?:-|$)")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Stable public contract for one paper operation."""

    slug: str
    title: str
    paper_question: str
    paper_sections: tuple[str, ...]
    summary: str
    cohort: str
    intervention: str
    readout: str
    required_arguments: tuple[str, ...]
    outputs: tuple[str, ...]
    compute: ComputeClass
    status: ExperimentStatus = "catalogued"

    def __post_init__(self) -> None:
        """Reject values outside the public catalog schema at construction time."""
        if _PUBLIC_SLUG_PATTERN.fullmatch(self.slug) is None or (
            _PAPER_QUESTION_SLUG_PATTERN.search(self.slug) is not None
        ):
            raise ValueError(
                f"slug must be descriptive kebab-case without RQ labels, got {self.slug!r}"
            )
        if self.compute not in _COMPUTE_CLASSES:
            raise ValueError(f"compute must be cpu or gpu, got {self.compute!r}")
        if self.status not in _EXPERIMENT_STATUSES:
            raise ValueError(f"status must be catalogued or implemented, got {self.status!r}")

    @property
    def target_command(self) -> str:
        """Return the repository-root runner, including before implementation.

        Each operation PR implements this exact ``typo-cot <slug>`` shape.  The
        separate ``experiments list/show`` commands only inspect this catalog.
        """
        extra = " --extra lrp" if self.compute == "gpu" else ""
        return f"uv run --project projects/typo-cot{extra} typo-cot {self.slug}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation in field order."""
        payload = asdict(self)
        payload["target_command"] = self.target_command
        return payload


PAPER_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        slug="prepare-edited-pairs",
        title="Generate aligned clean/edited pairs",
        paper_question="Setup",
        paper_sections=("§3.1", "Appendix A", "Table 4"),
        summary=(
            "Run the clean and edited prompts, select eligible edited words, and record "
            "word-final clean-to-edited token alignment."
        ),
        cohort="Items from GSM8K, MATH-500, MMLU, MMLU-Pro, ARC, or CSQA.",
        intervention=(
            "Apply up to four single-character edits using the seeded first-applicable "
            "substitution, duplication, or deletion rule."
        ),
        readout="Versioned per-item pair records with prompts, answers, edits, and alignment.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--targeting",
            "--num-edits",
            "--output-dir",
        ),
        outputs=("pairs.jsonl", "run.json"),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="targeting-fidelity-audit",
        title="Audit edit targeting and operation provenance",
        paper_question="Setup",
        paper_sections=("§3.1", "Appendix A"),
        summary=(
            "Measure whether edits landed on their selected spans and summarize gold-option "
            "edits and typo-operation counts."
        ),
        cohort="Prepared pair records for each available model-benchmark-targeting setting.",
        intervention="No model intervention; validate the frozen edit records.",
        readout=(
            "Targeting-fidelity, prepared-pair gold-option flags, and operation-count tables."
        ),
        required_arguments=("--pairs-root", "--output-dir"),
        outputs=(
            "targeting_fidelity_records.jsonl",
            "targeting_fidelity.csv",
            "operation_counts.json",
            "run.json",
        ),
        compute="cpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="layerwise-kl-patching",
        title="Patch edited-word activations and measure KL restoration",
        paper_question="RQ1",
        paper_sections=("§3.3", "§4.1", "Appendix B", "Figures 2 and 4", "Table 5"),
        summary=(
            "Copy one decoder block's residual output at aligned edited-word final tokens "
            "and scan every layer in both patch directions."
        ),
        cohort=(
            "Selected clean-correct, edited-wrong pairs with a complete finite layer grid "
            "and untreated KL denominator above 1e-9; the headline macro retains 30 of 32 "
            "settings."
        ),
        intervention="Clean-to-edited restoration and reciprocal edited-to-clean induction.",
        readout="Normalized first-CoT-token KL restoration or induction.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--targeting",
            "--directions",
            "--output-dir",
        ),
        outputs=(
            "layer_records.jsonl",
            "pair_status_records.jsonl",
            "setting_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="layerwise-answer-patching",
        title="Patch edited-word activations and regenerate answers",
        paper_question="RQ1",
        paper_sections=("§3.2", "§3.3", "§4.1", "Figure 2"),
        summary=(
            "Scan single-layer residual patches while freely regenerating the continuation "
            "and extracting the final answer."
        ),
        cohort="Eight model-task settings with a fixed denominator of at most 300 pairs.",
        intervention="One-layer clean-to-edited or edited-to-clean residual-state transfer.",
        readout="Return to the clean answer or change from the initially correct answer.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--attribution-pairs",
            "--random-pairs",
            "--directions",
            "--max-pairs",
            "--output-dir",
        ),
        outputs=(
            "answer_layer_records.jsonl",
            "pair_status_records.jsonl",
            "setting_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="fixed-window-answer-patching",
        title="Patch a fixed early layer window and regenerate answers",
        paper_question="RQ1",
        paper_sections=("§3.3", "§4.1", "Appendix B", "Tables 6 and 7"),
        summary=(
            "Apply the frozen residual-layer window [0,6) before free generation, with "
            "equal-width [6,12) available for the prespecified MMLU-Pro comparison."
        ),
        cohort=(
            "Six planned GSM8K/MMLU settings plus the two prespecified MMLU-Pro cells; "
            "the primary Gemma-3-4B/GSM8K cohort has 172 pairs."
        ),
        intervention="Multi-layer edited-word residual-state transfer over explicit windows.",
        readout="Answer restoration or reciprocal answer change.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--layers",
            "--directions",
            "--output-dir",
        ),
        outputs=(
            "fixed_window_records.jsonl",
            "pair_status_records.jsonl",
            "setting_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="build-rebuttal-manifest",
        title="Freeze the six-setting rebuttal cohorts and controls",
        paper_question="ARR addition",
        paper_sections=("Rebuttal analysis plan v1",),
        summary=(
            "Validate the prepared-pair and fixed-window producer artifacts, normalize "
            "all six settings, and freeze cohort, offset, donor, and held-out identities."
        ),
        cohort=(
            "The exact six GSM8K/MMLU Table 6 settings: 1,241 clean-correct, "
            "typo-wrong restoration pairs plus the uncapped alignment-eligible "
            "clean-correct, typo-correct harm cohort; alignment-ineligible records and "
            "selected-anchor exclusions remain explicit."
        ),
        intervention="No model intervention; hash-bound CPU validation and planning only.",
        readout=(
            "Normalized per-pair manifest, cohort IDs, deterministic control plans, and "
            "source-integrity audit."
        ),
        required_arguments=(
            "--prepared-pairs-root",
            "--fixed-window-root",
            "--output-dir",
        ),
        outputs=(
            "pair_manifest.jsonl",
            "cohort_ids.json",
            "source_audit.json",
            "run.json",
        ),
        compute="cpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="six-setting-patch-controls",
        title="Run answer-level specificity controls in all six settings",
        paper_question="ARR addition",
        paper_sections=("Rebuttal analysis plan v1: Six-setting patch controls",),
        summary=(
            "Compare the fixed [0,6) correct-coordinate patch against strict +2-token "
            "offset and deterministic cross-item donor controls in every setting."
        ),
        cohort=(
            "The exact 1,241-pair restoration manifest, with paired inference restricted "
            "to each setting's frozen common-valid subset."
        ),
        intervention=(
            "Reuse the fixed correct-coordinate outcome and generate prospective offset-2 "
            "and cross-item clean-to-typo residual-state patches."
        ),
        readout=(
            "Restoration rates, paired risk differences, exact McNemar tests, Holm-adjusted "
            "p-values, and equal-setting nested-bootstrap intervals."
        ),
        required_arguments=(
            "--config",
            "--manifest",
            "--fixed-window-root",
            "--gpu-id",
            "--output-dir",
        ),
        outputs=(
            "control_records.jsonl",
            "pair_status_records.jsonl",
            "six_setting_control_table.csv",
            "common_denominator_flow.csv",
            "multiplicity_table.csv",
            "macro_average.json",
            "risk_difference_forest.svg",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="patch-coordinate-controls",
        title="Compare edited-word patches with coordinate and donor controls",
        paper_question="RQ1",
        paper_sections=("§3.3", "§4.1", "Appendix B", "Table 7"),
        summary=(
            "Run the correct-coordinate patch beside two-token offset, matched cross-item "
            "donor, and identity self-copy controls."
        ),
        cohort=(
            "Published: same 172 primary Gemma-3-4B/GSM8K flip pairs; public runs: "
            "the exact clean-to-edited denominator of the referenced fixed-window run."
        ),
        intervention="Residual layers [0,6) with an explicitly selected coordinate control.",
        readout="Free-answer restoration and paired exact McNemar comparisons.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--fixed-window-run",
            "--layers",
            "--controls",
            "--output-dir",
        ),
        outputs=(
            "coordinate_control_records.jsonl",
            "pair_status_records.jsonl",
            "coordinate_control_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="patch-position-controls",
        title="Measure patch reachability from alternative prompt positions",
        paper_question="RQ1",
        paper_sections=("§3.3", "§4.1", "Appendix B", "Table 5"),
        summary=(
            "Repeat the layer scan at edited-word, prompt-final, and question-final positions "
            "to characterize where a patch can reach the readout."
        ),
        cohort=(
            "Gemma-3-4B/GSM8K Attribution-4 layerwise-KL complete-grid cohort; "
            "published n=109 and held common across all requested positions."
        ),
        intervention="Single-layer residual patch at each explicitly named prompt position.",
        readout="Layerwise normalized first-CoT-token KL restoration.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--layerwise-kl-run",
            "--positions",
            "--gpu-id",
            "--output-dir",
        ),
        outputs=(
            "position_control_records.jsonl",
            "pair_status_records.jsonl",
            "position_control_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="patch-text-combination",
        title="Cross fixed-window patching with complete clean text",
        paper_question="RQ1/RQ3 descriptive comparison",
        paper_sections=("§3.5", "§4.1", "Table 2", "Appendix D"),
        summary=(
            "Run the descriptive two-by-two crossing of the fixed [0,6) patch with zero or "
            "complete clean pre-answer text."
        ),
        cohort=(
            "One hash-verified Gemma-3-4B/GSM8K fixed-window denominator in all four cells; "
            "the published historical cohort had 172 pairs."
        ),
        intervention="Patch absent/present crossed with no clean text/full clean text.",
        readout="Four free-answer correctness cells; no mediation or interaction estimate.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--fixed-window-run",
            "--layers",
            "--output-dir",
        ),
        outputs=(
            "patch_text_records.jsonl",
            "pair_status_records.jsonl",
            "patch_text_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="cot-swap",
        title="Cross clean and edited questions with complete pre-answer CoTs",
        paper_question="RQ2",
        paper_sections=("§3.4", "§4.2", "Table 1", "Appendix C"),
        summary=(
            "Teacher-force the four A=(clean,clean), B=(edited,edited), C=(edited,clean), "
            "and D=(clean,edited) question/CoT cells and regenerate only the answer span."
        ),
        cohort=(
            "Applied-edit, answer-template-eligible pairs; change rates require regenerated "
            "A correctness and restoration further conditions on B changing from A."
        ),
        intervention="Supply the complete selected pre-answer CoT as fixed context.",
        readout="Both-changed, question-only, CoT-only, and B-to-C restoration rates.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--targeting",
            "--output-dir",
        ),
        outputs=(
            "cot_swap_records.jsonl",
            "pair_status_records.jsonl",
            "cot_swap_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="answer-line-deletion",
        title="Delete the final non-empty line from the supplied clean CoT",
        paper_question="RQ2 control",
        paper_sections=("§3.4", "§4.2", "Table 1"),
        summary=(
            "Repeat the edited-question/clean-CoT cell after removing its final non-empty "
            "pre-answer line to audit dependence on near-answer content and format."
        ),
        cohort=(
            "Regenerated-A-correct, B-changed cases from completed unlimited Random-4 "
            "CoT-swap runs for the three control models on GSM8K and MMLU."
        ),
        intervention="Teacher-force the clean CoT with its final answer line deleted.",
        readout="Paired complete/deleted restoration to the source clean A answer.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--cot-swap-run",
            "--max-pairs",
            "--output-dir",
        ),
        outputs=(
            "answer_line_deletion_records.jsonl",
            "pair_status_records.jsonl",
            "answer_line_deletion_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="clean-prefix-scan",
        title="Scan clean pre-answer text prefix budgets",
        paper_question="RQ3",
        paper_sections=("§3.5", "§4.3", "Figure 3", "Appendix D", "Table 10"),
        summary=(
            "Under the edited question, supply the first k clean-CoT tokens and freely "
            "regenerate over the frozen absolute and relative budget grid."
        ),
        cohort=(
            "The fixed-window clean-to-edited denominator for the primary setting, or "
            "fourteen deterministic 150-target extensions selected from completed "
            "Attribution-4 and Random-4 pair preparations; curves condition on a valid "
            "scan whose fresh k=0 rerun is wrong."
        ),
        intervention="Clean pre-answer prefix of k=round(r*L) tokens plus the absolute grid.",
        readout="Point correctness, stable-through-later correctness, and non-monotonicity.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--cohort",
            "--relative-budgets",
            "--absolute-budgets",
            "--output-dir",
        ),
        outputs=(
            "prefix_scan_records.jsonl",
            "pair_status_records.jsonl",
            "prefix_scan_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="one-token-prefix-replacement",
        title="Replace one selected clean-CoT token and regenerate",
        paper_question="Supplementary diagnostic",
        paper_sections=("§3.5", "§4.3", "Appendix D", "Tables 10 and 11"),
        summary=(
            "At the maximum clean-versus-edited next-token KL position, replace the clean "
            "token with the edited-context top-1 token and compare position controls."
        ),
        cohort=(
            "The primary fixed-window denominator or fourteen deterministic 150-target "
            "extensions selected from complete Attribution-4 and Random-4 preparations."
        ),
        intervention="One-token replacement at the selected, distant-control, or adjacent position.",
        readout="Correct-to-wrong answer changes; this is not a typo-repair estimate.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--cohort",
            "--position-controls",
            "--output-dir",
        ),
        outputs=(
            "one_token_records.jsonl",
            "pair_status_records.jsonl",
            "one_token_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="build-one-token-tables",
        title="Build the one-token paper tables",
        paper_question="Supplementary diagnostic",
        paper_sections=("§4.3", "Figure 5", "Appendix D", "Tables 10 and 11"),
        summary=(
            "Validate completed one-token producer runs, assemble the distinct paper "
            "denominators, and compute clustered Table 11 inference."
        ),
        cohort=(
            "The separate Gemma-3-4B/GSM8K primary cell and the fourteen extension "
            "model-task cells; adjacent pooling uses exactly three prespecified cells."
        ),
        intervention="No model intervention; consume verified integer producer events.",
        readout=(
            "Table 10 token columns, Table 11 position controls, and stored-field "
            "validation for the Figure 5 one-token case."
        ),
        required_arguments=("--runs-root", "--output-dir"),
        outputs=(
            "one_token_tables.json",
            "table10_one_token.csv",
            "table11_position_controls.csv",
            "one_token_tables.md",
            "one_token_tables.tex",
            "figure5_validation.json",
            "run.json",
        ),
        compute="cpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="edit-count-sensitivity",
        title="Measure accuracy and CoT-swap restoration by edit count",
        paper_question="RQ2 sensitivity",
        paper_sections=("Appendix C", "Table 8"),
        summary="Repeat accuracy and the six-setting CoT-swap pool at one, two, and four edits.",
        cohort="Complete accuracy settings and edit-count-specific CoT-swap change cohorts.",
        intervention=(
            "No additional model intervention; validate explicit-count pair preparation "
            "and CoT-swap producers, then assemble their distinct denominators."
        ),
        readout="Accuracy and conditional complete-clean-CoT restoration by edit count.",
        required_arguments=(
            "--pairs-root",
            "--cot-swap-runs-root",
            "--edit-counts",
            "--output-dir",
        ),
        outputs=(
            "edit_count_records.jsonl",
            "edit_count_summary.json",
            "table8_edit_count.csv",
            "table8_edit_count.md",
            "table8_edit_count.tex",
            "run.json",
        ),
        compute="cpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="model-scale-cot-swap",
        title="Compare MMLU CoT-swap checks across model scales",
        paper_question="RQ2 scale check",
        paper_sections=("Appendix C", "Table 9"),
        summary=(
            "Apply the CoT-swap cells to the fixed first 500 MMLU sample IDs across the "
            "reported Gemma, Llama, Mistral, and Qwen scale ladder."
        ),
        cohort=(
            "Clean-correct cases after intersecting the submitted model-specific MMLU "
            "source cohort with one versioned first-500 ID selector."
        ),
        intervention=(
            "No additional model intervention; validate the nine independent four-cell "
            "CoT-swap producers and assemble their distinct denominators."
        ),
        readout="Question/CoT change counts and conditional clean-CoT restoration by model.",
        required_arguments=(
            "--pairs-root",
            "--cot-swap-runs-root",
            "--cohort",
            "--output-dir",
        ),
        outputs=(
            "model_scale_records.jsonl",
            "model_scale_summary.json",
            "table9_model_scale.csv",
            "table9_model_scale.md",
            "table9_model_scale.tex",
            "run.json",
        ),
        compute="cpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="typo-warning-prompt",
        title="Test an explicit typo-warning instruction",
        paper_question="Input-correction audit",
        paper_sections=("§4.3", "Appendix E"),
        summary="Compare edited-input accuracy with and without the frozen typo-warning prompt.",
        cohort=(
            "Three submitted models by GSM8K and MMLU, with the exact 300 "
            "submitted Attribution-4 edited inputs per setting."
        ),
        intervention=(
            "Insert the explicit warning before the unique task-final marker while leaving "
            "the edited Attribution-4 question unchanged."
        ),
        readout="Paired free-answer accuracy difference.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--output-dir",
        ),
        outputs=("warning_prompt_records.jsonl", "warning_prompt_summary.json", "run.json"),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="input-corrector-audit",
        title="Audit edited-word restoration by input correctors",
        paper_question="Input-correction audit",
        paper_sections=("§4.3", "Appendix E", "Table 12"),
        summary=(
            "Run pyspellchecker 0.9.0, T5-large-spell, or Qwen2.5-7B-Instruct and audit "
            "edited-word restoration, intact-word changes, and prompt provenance."
        ),
        cohort="The twenty-five reported corrector settings and their paired clean prompts.",
        intervention="Correct the edited text with one explicitly selected corrector.",
        readout="Edited-word exact restoration and collateral/provenance controls.",
        required_arguments=(
            "--corrector",
            "--model",
            "--benchmark",
            "--pairs",
            "--output-dir",
        ),
        outputs=("corrector_records.jsonl", "corrector_audit_summary.json", "run.json"),
        compute="gpu",
        status="implemented",
    ),
    ExperimentSpec(
        slug="restoration-order-accuracy",
        title="Compare edited-word restoration orders",
        paper_question="Restoration-order oracle diagnostic",
        paper_sections=("§4.3", "Appendix E", "Table 13"),
        summary=(
            "Restore equal numbers of submitted edit groups in high-relevance-first, "
            "seeded-random, or low-relevance-first order and regenerate the answer."
        ),
        cohort=(
            "Fresh source-selected items from three models and two tasks; the final PDF "
            "reports a historical pooled n of 1,582 for comparison."
        ),
        intervention="Restore one, two, or three edit groups in the chosen order.",
        readout="Paired free-answer accuracy at each restoration budget.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--orders",
            "--budgets",
            "--output-dir",
        ),
        outputs=(
            "restoration_order_records.jsonl",
            "restoration_order_summary.json",
            "run.json",
        ),
        compute="gpu",
        status="implemented",
    ),
)

_BY_SLUG = {spec.slug: spec for spec in PAPER_EXPERIMENTS}


def get_experiment(slug: str) -> ExperimentSpec:
    """Look up an experiment by its public operation slug."""
    try:
        return _BY_SLUG[slug]
    except KeyError:
        raise KeyError(f"unknown experiment: {slug}") from None

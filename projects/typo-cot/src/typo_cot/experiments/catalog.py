"""Machine-readable inventory of the experiments reported in the final paper.

The operation slugs in this module are the stable public names.  RQ labels are
kept only as paper cross-references because they do not describe what a command
does.  Execution implementations are intentionally added one operation at a
time; ``status`` makes that migration state explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, get_args

# Contract fingerprint of the user-supplied final PDF. Change it only when the
# canonical paper artifact is intentionally replaced and the catalog is re-audited.
# Use ``rg '<old digest>' README.md projects/typo-cot`` to find every public copy;
# the contract tests require all documentation copies to stay synchronized.
PAPER_SHA256 = "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"

ComputeClass = Literal["cpu", "gpu"]
ExperimentStatus = Literal["catalogued", "implemented"]
_COMPUTE_CLASSES = frozenset(get_args(ComputeClass))
_EXPERIMENT_STATUSES = frozenset(get_args(ExperimentStatus))


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
        if self.compute not in _COMPUTE_CLASSES:
            raise ValueError(f"compute must be cpu or gpu, got {self.compute!r}")
        if self.status not in _EXPERIMENT_STATUSES:
            raise ValueError(f"status must be catalogued or implemented, got {self.status!r}")

    @property
    def target_command(self) -> str:
        """Return the planned top-level runner, including before implementation.

        Each operation PR implements this exact ``typo-cot <slug>`` shape.  The
        separate ``experiments list/show`` commands only inspect this catalog.
        """
        return f"uv run typo-cot {self.slug}"

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
        readout="Targeting-fidelity, gold-option, and operation-count tables.",
        required_arguments=("--pairs-root", "--output-dir"),
        outputs=("targeting_fidelity.csv", "operation_counts.json"),
        compute="cpu",
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
            "Pairs with a complete finite layer grid and an untreated KL denominator above "
            "1e-9; the headline macro summary retains 30 of 32 settings."
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
        outputs=("layer_records.jsonl", "setting_summary.json"),
        compute="gpu",
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
            "--pairs",
            "--directions",
            "--max-pairs",
            "--output-dir",
        ),
        outputs=("answer_layer_records.jsonl", "setting_summary.json"),
        compute="gpu",
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
        outputs=("fixed_window_records.jsonl", "setting_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="patch-coordinate-controls",
        title="Compare edited-word patches with coordinate and donor controls",
        paper_question="RQ1",
        paper_sections=("§3.3", "§4.1", "Appendix B", "Table 6"),
        summary=(
            "Run the correct-coordinate patch beside two-token offset, matched cross-item "
            "donor, and identity self-copy controls."
        ),
        cohort="The same primary 172 Gemma-3-4B/GSM8K flip pairs.",
        intervention="Residual layers [0,6) with an explicitly selected coordinate control.",
        readout="Free-answer restoration and paired exact McNemar comparisons.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--layers",
            "--controls",
            "--output-dir",
        ),
        outputs=("coordinate_control_records.jsonl", "coordinate_control_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="patch-position-controls",
        title="Measure patch reachability from alternative prompt positions",
        paper_question="RQ1",
        paper_sections=("§4.1", "Appendix B"),
        summary=(
            "Repeat the layer scan at edited-word, prompt-final, and question-final positions "
            "to characterize where a patch can reach the readout."
        ),
        cohort="The common 109-pair subset of the primary answer-patching cohort.",
        intervention="Single-layer residual patch at each explicitly named prompt position.",
        readout="Layerwise normalized first-CoT-token KL restoration.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--positions",
            "--output-dir",
        ),
        outputs=("position_control_records.jsonl", "position_control_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="patch-text-combination",
        title="Cross fixed-window patching with complete clean text",
        paper_question="RQ1/RQ3 control",
        paper_sections=("§3.5", "§4.1", "Table 2", "Appendix D"),
        summary=(
            "Run the descriptive two-by-two crossing of the fixed [0,6) patch with zero or "
            "complete clean pre-answer text."
        ),
        cohort="The same 172 Gemma-3-4B/GSM8K primary flip pairs.",
        intervention="Patch absent/present crossed with no clean text/full clean text.",
        readout="Four free-answer correctness cells; no mediation or interaction estimate.",
        required_arguments=("--model", "--benchmark", "--pairs", "--layers", "--output-dir"),
        outputs=("patch_text_records.jsonl", "patch_text_summary.json"),
        compute="gpu",
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
        cohort="Clean-correct pairs; restoration conditions on cases where B changes from A.",
        intervention="Supply the complete selected pre-answer CoT as fixed context.",
        readout="Both-changed, question-only, CoT-only, and B-to-C restoration rates.",
        required_arguments=("--model", "--benchmark", "--pairs", "--output-dir"),
        outputs=("cot_swap_records.jsonl", "cot_swap_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="answer-line-deletion",
        title="Delete the final answer line from the supplied clean CoT",
        paper_question="RQ2 control",
        paper_sections=("§3.4", "§4.2", "Table 1"),
        summary=(
            "Repeat the edited-question/clean-CoT cell after removing its final answer line "
            "to audit dependence on near-answer content and format."
        ),
        cohort="Gemma-3-4B, Llama-3.2-3B, and Mistral-7B on GSM8K and MMLU.",
        intervention="Teacher-force the clean CoT with its final answer line deleted.",
        readout="Answer restoration on the same eligible CoT-swap cases.",
        required_arguments=("--model", "--benchmark", "--pairs", "--output-dir"),
        outputs=("answer_line_deletion_records.jsonl", "answer_line_deletion_summary.json"),
        compute="gpu",
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
            "The frozen primary 172 pairs and fourteen deterministic 150-target extensions; "
            "curves condition on a valid scan whose fresh k=0 rerun is wrong."
        ),
        intervention="Clean pre-answer prefix of k=round(r*L) tokens plus the absolute grid.",
        readout="Point correctness, stable-through-later correctness, and non-monotonicity.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--target-set",
            "--relative-budgets",
            "--absolute-budgets",
            "--output-dir",
        ),
        outputs=("prefix_scan_records.jsonl", "prefix_scan_summary.json"),
        compute="gpu",
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
        cohort="The primary and fourteen extension cohorts after the diagnostic eligibility filters.",
        intervention="One-token replacement at the selected, distant-control, or adjacent position.",
        readout="Correct-to-wrong answer changes; this is not a typo-repair estimate.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--target-set",
            "--position-control",
            "--output-dir",
        ),
        outputs=("one_token_records.jsonl", "one_token_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="edit-count-sensitivity",
        title="Measure accuracy and CoT-swap restoration by edit count",
        paper_question="RQ2 sensitivity",
        paper_sections=("Appendix C", "Table 8"),
        summary="Repeat accuracy and the six-setting CoT-swap pool at one, two, and four edits.",
        cohort="Complete accuracy settings and edit-count-specific CoT-swap change cohorts.",
        intervention="Prepare and evaluate inputs with an explicit number of edits.",
        readout="Accuracy and conditional complete-clean-CoT restoration by edit count.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--edit-counts",
            "--pairs-root",
            "--output-dir",
        ),
        outputs=("edit_count_records.jsonl", "edit_count_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="model-scale-cot-swap",
        title="Run MMLU CoT-swap checks across model scales",
        paper_question="RQ2 scale check",
        paper_sections=("Appendix C", "Table 9"),
        summary=(
            "Apply the CoT-swap cells to the fixed first 500 MMLU sample IDs across the "
            "reported Gemma, Llama, Mistral, and Qwen scale ladder."
        ),
        cohort="Clean-correct cases among the same first 500 MMLU IDs for each model.",
        intervention="The complete four-cell CoT swap for every requested model.",
        readout="Question/CoT change rates and conditional clean-CoT restoration.",
        required_arguments=("--models", "--benchmark", "--pairs-root", "--output-dir"),
        outputs=("scale_cot_swap_records.jsonl", "scale_cot_swap_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="typo-warning-prompt",
        title="Test an explicit typo-warning instruction",
        paper_question="Input-correction audit",
        paper_sections=("§4.3", "Appendix E"),
        summary="Compare edited-input accuracy with and without the frozen typo-warning prompt.",
        cohort="One instruction and two tasks: GSM8K and MMLU.",
        intervention="Prepend the explicit warning while leaving the edited question unchanged.",
        readout="Paired free-answer accuracy difference.",
        required_arguments=("--model", "--benchmark", "--pairs", "--output-dir"),
        outputs=("warning_prompt_records.jsonl", "warning_prompt_summary.json"),
        compute="gpu",
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
        outputs=("corrector_records.jsonl", "corrector_audit_summary.json"),
        compute="gpu",
    ),
    ExperimentSpec(
        slug="restoration-order-accuracy",
        title="Restore edited words in relevance and random orders",
        paper_question="Input-correction audit",
        paper_sections=("§4.3", "Appendix E", "Table 13"),
        summary=(
            "Restore equal numbers of edited words in high-relevance-first, seeded-random, "
            "or low-relevance-first order and regenerate the answer."
        ),
        cohort="The 1,582 archived-selected items from three models and two tasks.",
        intervention="Restore one, two, or three of the four edited words in the chosen order.",
        readout="Paired free-answer accuracy at each clean-word restoration budget.",
        required_arguments=(
            "--model",
            "--benchmark",
            "--pairs",
            "--order",
            "--budgets",
            "--output-dir",
        ),
        outputs=("restoration_order_records.jsonl", "restoration_order_summary.json"),
        compute="gpu",
    ),
)

_BY_SLUG = {spec.slug: spec for spec in PAPER_EXPERIMENTS}


def get_experiment(slug: str) -> ExperimentSpec:
    """Look up an experiment by its public operation slug."""
    try:
        return _BY_SLUG[slug]
    except KeyError:
        raise KeyError(f"unknown experiment: {slug}") from None

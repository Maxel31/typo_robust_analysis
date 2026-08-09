"""Public command-line entry point for paper experiment reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from typo_cot.experiments import (
    PAPER_EXPERIMENTS,
    PAPER_SHA256,
    ExperimentSpec,
    get_experiment,
)
from typo_cot.experiments.answer_line_deletion import (
    ANSWER_LINE_DELETION_BENCHMARKS,
    AnswerLineDeletionConfig,
    AnswerLineDeletionRunError,
    run_answer_line_deletion,
)
from typo_cot.experiments.clean_prefix_scan import (
    CleanPrefixScanConfig,
    CleanPrefixScanRunError,
    run_clean_prefix_scan,
)
from typo_cot.experiments.cot_swap import (
    COT_SWAP_BENCHMARKS,
    CotSwapConfig,
    CotSwapRunError,
    run_cot_swap,
)
from typo_cot.experiments.fixed_window_answer_patching import (
    DIRECTION_NAMES as FIXED_WINDOW_DIRECTION_NAMES,
)
from typo_cot.experiments.fixed_window_answer_patching import (
    FixedWindowAnswerPatchingConfig,
    FixedWindowAnswerPatchingRunError,
    parse_layer_window,
    run_fixed_window_answer_patching,
)
from typo_cot.experiments.layerwise_answer_patching import (
    DIRECTION_NAMES as ANSWER_DIRECTION_NAMES,
)
from typo_cot.experiments.layerwise_answer_patching import (
    LayerwiseAnswerPatchingConfig,
    LayerwiseAnswerPatchingRunError,
    run_layerwise_answer_patching,
)
from typo_cot.experiments.layerwise_kl_patching import (
    DIRECTION_NAMES as KL_DIRECTION_NAMES,
)
from typo_cot.experiments.layerwise_kl_patching import (
    LayerwiseKLPatchingConfig,
    LayerwiseKLPatchingRunError,
    run_layerwise_kl_patching,
)
from typo_cot.experiments.one_token_prefix_replacement import (
    POSITION_CONTROLS as ONE_TOKEN_POSITION_CONTROLS,
)
from typo_cot.experiments.one_token_prefix_replacement import (
    OneTokenPrefixReplacementConfig,
    OneTokenPrefixReplacementRunError,
    run_one_token_prefix_replacement,
)
from typo_cot.experiments.patch_coordinate_controls import (
    CONTROL_NAMES as COORDINATE_CONTROL_NAMES,
)
from typo_cot.experiments.patch_coordinate_controls import (
    CoordinateControlConfig,
    CoordinateControlRunError,
    run_patch_coordinate_controls,
)
from typo_cot.experiments.patch_position_controls import (
    POSITION_NAMES as PATCH_POSITION_NAMES,
)
from typo_cot.experiments.patch_position_controls import (
    PositionControlConfig,
    PositionControlRunError,
    run_patch_position_controls,
)
from typo_cot.experiments.patch_text_combination import (
    PatchTextCombinationConfig,
    PatchTextCombinationRunError,
    run_patch_text_combination,
)
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PUBLIC_BENCHMARKS,
    TARGETING_CONDITIONS,
    PairPreparationRunError,
    PrepareEditedPairsConfig,
    run_prepare_edited_pairs,
)
from typo_cot.experiments.targeting_fidelity_audit import (
    TargetingFidelityAuditConfig,
    TargetingFidelityAuditError,
    run_targeting_fidelity_audit,
)


def _experiment(value: str) -> ExperimentSpec:
    try:
        return get_experiment(value)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc.args[0])) from None


def _add_format_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared human/machine-readable output selector."""
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _edit_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 4:
        raise argparse.ArgumentTypeError("must be between 1 and 4")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="typo-cot",
        description="Reproduce the operations reported in the typo activation-patching paper.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    experiments = commands.add_parser(
        "experiments", help="Inspect the paper-aligned experiment catalog."
    )
    actions = experiments.add_subparsers(dest="catalog_action", required=True)

    list_parser = actions.add_parser("list", help="List operations in reproduction order.")
    _add_format_argument(list_parser)

    source_parser = actions.add_parser(
        "source", help="Show the canonical paper artifact fingerprint."
    )
    _add_format_argument(source_parser)

    show_parser = actions.add_parser("show", help="Show one operation's public contract.")
    show_parser.add_argument("experiment", type=_experiment)
    _add_format_argument(show_parser)

    pairs = commands.add_parser(
        "prepare-edited-pairs",
        help="Generate paper-aligned clean/edited pairs and word-final token alignment.",
    )
    pairs.add_argument("--model", required=True, help="Hugging Face model identifier.")
    pairs.add_argument("--benchmark", required=True, choices=PUBLIC_BENCHMARKS)
    pairs.add_argument("--targeting", required=True, choices=TARGETING_CONDITIONS)
    pairs.add_argument("--num-edits", required=True, type=_edit_count)
    pairs.add_argument("--output-dir", required=True, type=Path)
    pairs.add_argument("--seed", type=int, default=42)
    pairs.add_argument("--max-new-tokens", type=_positive_int, default=512)
    pairs.add_argument("--gpu-id", default="0")
    pairs.add_argument("--limit", type=_positive_int)
    pairs.add_argument("--resume", action="store_true")

    cot_swap = commands.add_parser(
        "cot-swap",
        help="Cross clean/edited questions and complete pre-answer CoTs in four cells.",
    )
    cot_swap.add_argument("--model", required=True, help="Hugging Face model identifier.")
    cot_swap.add_argument("--benchmark", required=True, choices=COT_SWAP_BENCHMARKS)
    cot_swap.add_argument("--pairs", required=True, type=Path)
    cot_swap.add_argument("--targeting", required=True, choices=TARGETING_CONDITIONS)
    cot_swap.add_argument("--gpu-id", default="0")
    cot_swap.add_argument("--limit", type=_positive_int)
    cot_swap.add_argument("--output-dir", required=True, type=Path)
    cot_swap.add_argument("--resume", action="store_true")

    answer_line_deletion = commands.add_parser(
        "answer-line-deletion",
        help="Delete the final non-empty line from condition C's clean pre-answer text.",
    )
    answer_line_deletion.add_argument(
        "--model", required=True, help="Hugging Face model identifier."
    )
    answer_line_deletion.add_argument(
        "--benchmark", required=True, choices=ANSWER_LINE_DELETION_BENCHMARKS
    )
    answer_line_deletion.add_argument("--cot-swap-run", required=True, type=Path)
    answer_line_deletion.add_argument("--max-pairs", required=True, type=_positive_int)
    answer_line_deletion.add_argument("--gpu-id", default="0")
    answer_line_deletion.add_argument("--limit", type=_positive_int)
    answer_line_deletion.add_argument("--output-dir", required=True, type=Path)
    answer_line_deletion.add_argument("--resume", action="store_true")

    clean_prefix = commands.add_parser(
        "clean-prefix-scan",
        help="Supply increasing clean pre-answer token prefixes under the edited question.",
    )
    clean_prefix.add_argument("--model", required=True, help="Hugging Face model identifier.")
    clean_prefix.add_argument(
        "--benchmark",
        required=True,
        choices=("gsm8k", "mmlu", "arc"),
    )
    clean_prefix.add_argument("--cohort", required=True, choices=("primary", "extension"))
    clean_prefix.add_argument("--fixed-window-run", type=Path)
    clean_prefix.add_argument("--pairs", nargs="+", type=Path, default=[])
    clean_prefix.add_argument("--max-pairs", type=_positive_int)
    clean_prefix.add_argument(
        "--relative-budgets",
        required=True,
        nargs="+",
        type=float,
    )
    clean_prefix.add_argument(
        "--absolute-budgets",
        required=True,
        nargs="+",
        type=_positive_int,
    )
    clean_prefix.add_argument("--gpu-id", default="0")
    clean_prefix.add_argument("--limit", type=_positive_int)
    clean_prefix.add_argument("--output-dir", required=True, type=Path)
    clean_prefix.add_argument("--resume", action="store_true")

    one_token = commands.add_parser(
        "one-token-prefix-replacement",
        help="Replace one selected clean-CoT token and compare position controls.",
    )
    one_token.add_argument("--model", required=True, help="Hugging Face model identifier.")
    one_token.add_argument(
        "--benchmark",
        required=True,
        choices=("gsm8k", "mmlu", "arc"),
    )
    one_token.add_argument("--cohort", required=True, choices=("primary", "extension"))
    one_token.add_argument("--fixed-window-run", type=Path)
    one_token.add_argument("--pairs", nargs="+", type=Path, default=[])
    one_token.add_argument("--max-pairs", type=_positive_int)
    one_token.add_argument(
        "--position-controls",
        required=True,
        nargs="+",
        choices=ONE_TOKEN_POSITION_CONTROLS,
    )
    one_token.add_argument("--gpu-id", default="0")
    one_token.add_argument("--limit", type=_positive_int)
    one_token.add_argument("--output-dir", required=True, type=Path)
    one_token.add_argument("--resume", action="store_true")

    one_token_tables = commands.add_parser(
        "build-one-token-tables",
        help="Build Appendix D one-token tables from verified completed producer runs.",
    )
    one_token_tables.add_argument("--runs-root", required=True, type=Path)
    one_token_tables.add_argument("--output-dir", required=True, type=Path)

    targeting_audit = commands.add_parser(
        "targeting-fidelity-audit",
        help="Validate prepared pairs and aggregate Appendix A input-quality checks.",
    )
    targeting_audit.add_argument("--pairs-root", required=True, type=Path)
    targeting_audit.add_argument("--output-dir", required=True, type=Path)
    targeting_audit.add_argument("--expected-seed", type=int, default=42)

    layerwise_kl = commands.add_parser(
        "layerwise-kl-patching",
        help="Patch aligned edited-word block outputs and scan first-CoT-token KL.",
    )
    layerwise_kl.add_argument("--model", required=True, help="Hugging Face model identifier.")
    layerwise_kl.add_argument("--benchmark", required=True, choices=PUBLIC_BENCHMARKS)
    layerwise_kl.add_argument("--pairs", required=True, type=Path)
    layerwise_kl.add_argument("--targeting", required=True, choices=TARGETING_CONDITIONS)
    layerwise_kl.add_argument(
        "--directions",
        required=True,
        nargs="+",
        choices=KL_DIRECTION_NAMES,
    )
    layerwise_kl.add_argument("--output-dir", required=True, type=Path)
    layerwise_kl.add_argument("--gpu-id", default="0")
    layerwise_kl.add_argument("--limit", type=_positive_int)
    layerwise_kl.add_argument("--resume", action="store_true")

    layerwise_answer = commands.add_parser(
        "layerwise-answer-patching",
        help="Patch aligned edited-word block outputs and freely regenerate answers.",
    )
    layerwise_answer.add_argument("--model", required=True, help="Hugging Face model identifier.")
    layerwise_answer.add_argument("--benchmark", required=True, choices=("gsm8k", "mmlu"))
    layerwise_answer.add_argument("--attribution-pairs", required=True, type=Path)
    layerwise_answer.add_argument("--random-pairs", required=True, type=Path)
    layerwise_answer.add_argument(
        "--directions",
        required=True,
        nargs="+",
        choices=ANSWER_DIRECTION_NAMES,
    )
    layerwise_answer.add_argument("--max-pairs", required=True, type=_positive_int)
    layerwise_answer.add_argument("--output-dir", required=True, type=Path)
    layerwise_answer.add_argument("--gpu-id", default="0")
    layerwise_answer.add_argument("--resume", action="store_true")

    fixed_window = commands.add_parser(
        "fixed-window-answer-patching",
        help="Patch fixed decoder-block windows and freely regenerate answers.",
    )
    fixed_window.add_argument("--model", required=True, help="Hugging Face model identifier.")
    fixed_window.add_argument(
        "--benchmark",
        required=True,
        choices=("gsm8k", "mmlu", "mmlu-pro"),
    )
    fixed_window.add_argument("--pairs", required=True, nargs="+", type=Path)
    fixed_window.add_argument(
        "--layers",
        required=True,
        nargs="+",
        type=parse_layer_window,
        metavar="START:STOP",
    )
    fixed_window.add_argument(
        "--directions",
        required=True,
        nargs="+",
        choices=FIXED_WINDOW_DIRECTION_NAMES,
    )
    fixed_window.add_argument("--output-dir", required=True, type=Path)
    fixed_window.add_argument("--gpu-id", default="0")
    fixed_window.add_argument("--limit", type=_positive_int)
    fixed_window.add_argument("--resume", action="store_true")

    coordinate_controls = commands.add_parser(
        "patch-coordinate-controls",
        help="Compare correct edited-word patches with coordinate and donor controls.",
    )
    coordinate_controls.add_argument(
        "--model", required=True, help="Hugging Face model identifier."
    )
    coordinate_controls.add_argument("--benchmark", required=True, choices=("gsm8k",))
    coordinate_controls.add_argument("--fixed-window-run", required=True, type=Path)
    coordinate_controls.add_argument(
        "--layers",
        required=True,
        nargs="+",
        type=parse_layer_window,
        metavar="START:STOP",
    )
    coordinate_controls.add_argument(
        "--controls",
        required=True,
        nargs="+",
        choices=COORDINATE_CONTROL_NAMES,
    )
    coordinate_controls.add_argument("--output-dir", required=True, type=Path)
    coordinate_controls.add_argument("--gpu-id", default="0")
    coordinate_controls.add_argument("--limit", type=_positive_int)
    coordinate_controls.add_argument("--resume", action="store_true")

    position_controls = commands.add_parser(
        "patch-position-controls",
        help="Compare layerwise clean-state patches at three prompt positions.",
    )
    position_controls.add_argument("--model", required=True, help="Hugging Face model identifier.")
    position_controls.add_argument("--benchmark", required=True, choices=("gsm8k",))
    position_controls.add_argument("--layerwise-kl-run", required=True, type=Path)
    position_controls.add_argument(
        "--positions",
        required=True,
        nargs="+",
        choices=PATCH_POSITION_NAMES,
    )
    position_controls.add_argument("--gpu-id", default="0")
    position_controls.add_argument("--limit", type=_positive_int)
    position_controls.add_argument("--output-dir", required=True, type=Path)
    position_controls.add_argument("--resume", action="store_true")

    patch_text = commands.add_parser(
        "patch-text-combination",
        help="Cross the fixed [0,6) patch with complete clean pre-answer text.",
    )
    patch_text.add_argument("--model", required=True, help="Hugging Face model identifier.")
    patch_text.add_argument("--benchmark", required=True, choices=("gsm8k",))
    patch_text.add_argument("--fixed-window-run", required=True, type=Path)
    patch_text.add_argument(
        "--layers",
        required=True,
        nargs="+",
        type=parse_layer_window,
        metavar="START:STOP",
    )
    patch_text.add_argument("--gpu-id", default="0")
    patch_text.add_argument("--limit", type=_positive_int)
    patch_text.add_argument("--output-dir", required=True, type=Path)
    patch_text.add_argument("--resume", action="store_true")
    return parser


def _print_list(output_format: str) -> None:
    if output_format == "json":
        print(
            json.dumps([spec.to_dict() for spec in PAPER_EXPERIMENTS], indent=2, ensure_ascii=False)
        )
        return

    width = max((len(spec.slug) for spec in PAPER_EXPERIMENTS), default=0)
    status_width = max((len(spec.status) for spec in PAPER_EXPERIMENTS), default=0)
    for spec in PAPER_EXPERIMENTS:
        print(f"{spec.slug:<{width}}  {spec.status:<{status_width}}  {spec.title}")


def _print_source(output_format: str) -> None:
    if output_format == "json":
        print(json.dumps({"sha256": PAPER_SHA256}, indent=2, ensure_ascii=False))
        return
    print(f"paper sha256: {PAPER_SHA256}")


def _print_spec(spec: ExperimentSpec, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        return

    print(spec.title)
    print(f"summary: {spec.summary}")
    print(f"operation: {spec.slug}")
    print(f"paper: {spec.paper_question} ({', '.join(spec.paper_sections)})")
    print(f"status: {spec.status}")
    print(f"compute: {spec.compute}")
    print(f"target command: {spec.target_command}")
    print(f"required arguments: {' '.join(spec.required_arguments)}")
    print(f"cohort: {spec.cohort}")
    print(f"intervention: {spec.intervention}")
    print(f"readout: {spec.readout}")
    print(f"outputs: {', '.join(spec.outputs)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CLI and return a process exit code."""
    args = _parser().parse_args(argv)
    if args.command == "experiments" and args.catalog_action == "list":
        _print_list(args.format)
        return 0
    if args.command == "experiments" and args.catalog_action == "source":
        _print_source(args.format)
        return 0
    if args.command == "experiments" and args.catalog_action == "show":
        _print_spec(args.experiment, args.format)
        return 0
    if args.command == "prepare-edited-pairs":
        try:
            result = run_prepare_edited_pairs(
                PrepareEditedPairsConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    targeting=args.targeting,
                    num_edits=args.num_edits,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    max_new_tokens=args.max_new_tokens,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (FileExistsError, ValueError, PairPreparationRunError) as exc:
            print(f"prepare-edited-pairs: error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {result.written} pair(s): {result.pairs_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "cot-swap":
        try:
            result = run_cot_swap(
                CotSwapConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    pairs=args.pairs,
                    targeting=args.targeting,
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (FileExistsError, CotSwapRunError, RuntimeError, ValueError) as exc:
            print(f"cot-swap: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} CoT-swap record(s) for "
            f"{result.executed_pairs} pair(s): {result.records_path}"
        )
        print(f"setting summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "answer-line-deletion":
        try:
            result = run_answer_line_deletion(
                AnswerLineDeletionConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    cot_swap_run=args.cot_swap_run,
                    max_pairs=args.max_pairs,
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            AnswerLineDeletionRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"answer-line-deletion: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} paired control record(s) from "
            f"{result.source_cohort_pairs} source-cohort pair(s): {result.records_path}"
        )
        print(f"answer-line deletion summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "clean-prefix-scan":
        try:
            result = run_clean_prefix_scan(
                CleanPrefixScanConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    cohort=args.cohort,
                    fixed_window_run=args.fixed_window_run,
                    pairs=tuple(args.pairs),
                    max_pairs=args.max_pairs,
                    relative_budgets=tuple(args.relative_budgets),
                    absolute_budgets=tuple(args.absolute_budgets),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            CleanPrefixScanRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"clean-prefix-scan: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} clean-prefix scan record(s) for "
            f"{result.executed_pairs} pair(s): {result.records_path}"
        )
        print(f"clean-prefix summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "one-token-prefix-replacement":
        try:
            result = run_one_token_prefix_replacement(
                OneTokenPrefixReplacementConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    cohort=args.cohort,
                    fixed_window_run=args.fixed_window_run,
                    pairs=tuple(args.pairs),
                    max_pairs=args.max_pairs,
                    position_controls=tuple(args.position_controls),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            OSError,
            OneTokenPrefixReplacementRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"one-token-prefix-replacement: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} one-token record(s) for "
            f"{result.executed_pairs} selected pair(s): {result.records_path}"
        )
        print(f"pair statuses: {result.pair_status_records_path}")
        print(f"one-token summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "build-one-token-tables":
        # Keep NumPy and the analysis implementation out of the base CLI import
        # path. They are needed only when this CPU artifact builder is invoked.
        from typo_cot.experiments.build_one_token_tables import (
            BuildOneTokenTablesConfig,
            OneTokenTablesInputError,
            run_build_one_token_tables,
        )

        try:
            result = run_build_one_token_tables(
                BuildOneTokenTablesConfig(
                    runs_root=args.runs_root,
                    output_dir=args.output_dir,
                )
            )
        except (
            FileExistsError,
            OSError,
            OneTokenTablesInputError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"build-one-token-tables: error: {exc}", file=sys.stderr)
            return 1
        print(f"built one-token tables from {result.settings} setting(s): {result.tables_path}")
        print(f"Figure 5 validation: {result.figure5_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "targeting-fidelity-audit":
        try:
            result = run_targeting_fidelity_audit(
                TargetingFidelityAuditConfig(
                    pairs_root=args.pairs_root,
                    output_dir=args.output_dir,
                    expected_seed=args.expected_seed,
                )
            )
        except (FileExistsError, TargetingFidelityAuditError, ValueError) as exc:
            print(f"targeting-fidelity-audit: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"audited {result.items} pair(s) across {result.settings} setting(s): "
            f"{result.summary_path}"
        )
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "layerwise-kl-patching":
        try:
            result = run_layerwise_kl_patching(
                LayerwiseKLPatchingConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    pairs=args.pairs,
                    targeting=args.targeting,
                    directions=tuple(args.directions),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (FileExistsError, LayerwiseKLPatchingRunError, RuntimeError, ValueError) as exc:
            print(f"layerwise-kl-patching: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.layer_records} layer record(s) from "
            f"{result.included_grids} complete grid(s): {result.layer_records_path}"
        )
        print(f"setting summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "layerwise-answer-patching":
        try:
            result = run_layerwise_answer_patching(
                LayerwiseAnswerPatchingConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    attribution_pairs=args.attribution_pairs,
                    random_pairs=args.random_pairs,
                    directions=tuple(args.directions),
                    max_pairs=args.max_pairs,
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            LayerwiseAnswerPatchingRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"layerwise-answer-patching: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.layer_records} layer record(s) for "
            f"{result.fixed_pairs} fixed pair(s): {result.answer_layer_records_path}"
        )
        print(f"setting summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "fixed-window-answer-patching":
        try:
            result = run_fixed_window_answer_patching(
                FixedWindowAnswerPatchingConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    pairs=tuple(args.pairs),
                    layers=tuple(args.layers),
                    directions=tuple(args.directions),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            FixedWindowAnswerPatchingRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"fixed-window-answer-patching: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.window_records} fixed-window record(s) from "
            f"{result.included_direction_pairs} included pair-direction(s): "
            f"{result.fixed_window_records_path}"
        )
        print(f"setting summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "patch-coordinate-controls":
        try:
            result = run_patch_coordinate_controls(
                CoordinateControlConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    fixed_window_run=args.fixed_window_run,
                    layers=tuple(args.layers),
                    controls=tuple(args.controls),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            CoordinateControlRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"patch-coordinate-controls: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} coordinate-control record(s) for "
            f"{result.pairs} pair(s): {result.records_path}"
        )
        print(f"coordinate-control summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "patch-position-controls":
        try:
            result = run_patch_position_controls(
                PositionControlConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    layerwise_kl_run=args.layerwise_kl_run,
                    positions=tuple(args.positions),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            PositionControlRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"patch-position-controls: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} position-control record(s) for "
            f"{result.pairs} pair(s): {result.records_path}"
        )
        print(f"position-control summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    if args.command == "patch-text-combination":
        try:
            result = run_patch_text_combination(
                PatchTextCombinationConfig(
                    model=args.model,
                    benchmark=args.benchmark,
                    fixed_window_run=args.fixed_window_run,
                    layers=tuple(args.layers),
                    output_dir=args.output_dir,
                    gpu_id=args.gpu_id,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
        except (
            FileExistsError,
            PatchTextCombinationRunError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"patch-text-combination: error: {exc}", file=sys.stderr)
            return 1
        print(
            f"wrote {result.records} patch/text cell record(s) for "
            f"{result.pairs} pair(s): {result.records_path}"
        )
        print(f"patch/text summary: {result.summary_path}")
        print(f"run manifest: {result.run_path}")
        return 0
    raise AssertionError("argparse accepted an unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

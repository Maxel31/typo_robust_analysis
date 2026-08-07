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
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PUBLIC_BENCHMARKS,
    TARGETING_CONDITIONS,
    PairPreparationRunError,
    PrepareEditedPairsConfig,
    run_prepare_edited_pairs,
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
    raise AssertionError("argparse accepted an unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

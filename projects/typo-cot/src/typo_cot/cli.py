"""Public command-line entry point for paper experiment reproduction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from typo_cot.experiments import (
    PAPER_EXPERIMENTS,
    PAPER_SHA256,
    ExperimentSpec,
    get_experiment,
)


def _experiment(value: str) -> ExperimentSpec:
    try:
        return get_experiment(value)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc.args[0])) from None


def _add_format_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared human/machine-readable output selector."""
    parser.add_argument("--format", choices=("text", "json"), default="text")


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
    raise AssertionError("argparse accepted an unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

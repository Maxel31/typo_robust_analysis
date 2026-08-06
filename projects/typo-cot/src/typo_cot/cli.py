"""Public command-line entry point for paper experiment reproduction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from typo_cot.experiments.catalog import PAPER_EXPERIMENTS, ExperimentSpec, get_experiment


def _experiment_slug(value: str) -> str:
    try:
        get_experiment(value)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc.args[0])) from None
    return value


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
    list_parser.add_argument("--format", choices=("text", "json"), default="text")

    show_parser = actions.add_parser("show", help="Show one operation's public contract.")
    show_parser.add_argument("slug", type=_experiment_slug)
    show_parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _print_list(output_format: str) -> None:
    if output_format == "json":
        print(
            json.dumps([spec.to_dict() for spec in PAPER_EXPERIMENTS], indent=2, ensure_ascii=False)
        )
        return

    width = max(len(spec.slug) for spec in PAPER_EXPERIMENTS)
    for spec in PAPER_EXPERIMENTS:
        print(f"{spec.slug:<{width}}  {spec.status:<10}  {spec.title}")


def _print_spec(spec: ExperimentSpec, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        return

    print(spec.title)
    print(f"operation: {spec.slug}")
    print(f"paper: {spec.paper_question} ({', '.join(spec.paper_sections)})")
    print(f"status: {spec.status}")
    print(f"compute: {spec.compute}")
    print(f"command: {spec.command}")
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
    if args.command == "experiments" and args.catalog_action == "show":
        _print_spec(get_experiment(args.slug), args.format)
        return 0
    raise AssertionError("argparse accepted an unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

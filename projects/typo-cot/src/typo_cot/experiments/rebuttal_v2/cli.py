"""CPU-only command registration for implemented rebuttal v2 operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register_rebuttal_v2_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = commands.add_parser(
        "rebuttal-v2", help="Audit archived sources without historical success quotas."
    )
    operations = parser.add_subparsers(dest="rebuttal_v2_operation", required=True)
    intake = operations.add_parser(
        "intake", help="Verify source provenance and freeze an outcome-independent manifest."
    )
    intake.add_argument("--archive-index", type=Path, required=True)
    intake.add_argument("--protocol", type=Path, required=True)
    intake.add_argument("--output-dir", type=Path, required=True)
    intake.set_defaults(_rebuttal_v2_handler=_run_intake)


def _run_intake(args: argparse.Namespace) -> int:
    # Import only the selected operation; help never imports a model runtime.
    from .source import run_intake

    try:
        result = run_intake(args.archive_index, args.protocol, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rebuttal-v2 intake: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0

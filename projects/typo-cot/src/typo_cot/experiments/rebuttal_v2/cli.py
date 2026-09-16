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
    answer = operations.add_parser(
        "answer-audit", help="Compare explicit-answer and public v1 parsers on saved text."
    )
    answer.add_argument("--manifest", type=Path, required=True)
    answer.add_argument("--output-dir", type=Path, required=True)
    answer.set_defaults(_rebuttal_v2_handler=_run_answer_audit)
    inputs = operations.add_parser(
        "input-audit", help="Audit exact edits and pinned tokenizer coordinates."
    )
    inputs.add_argument("--manifest", type=Path, required=True)
    inputs.add_argument("--tokenizer-lock", type=Path, required=True)
    inputs.add_argument("--output-dir", type=Path, required=True)
    inputs.set_defaults(_rebuttal_v2_handler=_run_input_audit)
    export = operations.add_parser(
        "annotation-export", help="Export blinded Stage A or locked Stage B forms."
    )
    export.add_argument("--audit", type=Path, required=True)
    export.add_argument("--stage", choices=("a", "b"), required=True)
    export.add_argument("--stage-a-labels", type=Path)
    export.add_argument("--annotation-batch", type=Path)
    export.add_argument("--raters", nargs="+")
    export.add_argument("--single-rater-reason")
    export.add_argument("--output-dir", type=Path, required=True)
    export.set_defaults(_rebuttal_v2_handler=_run_annotation_export)
    annotation_import = operations.add_parser(
        "annotation-import", help="Validate independent ratings and finalized semantic labels."
    )
    annotation_import.add_argument("--audit", type=Path, required=True)
    annotation_import.add_argument("--labels", type=Path, required=True)
    annotation_import.add_argument("--annotation-batch", type=Path, required=True)
    annotation_import.add_argument("--output-dir", type=Path, required=True)
    annotation_import.set_defaults(_rebuttal_v2_handler=_run_annotation_import)


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


def _run_answer_audit(args: argparse.Namespace) -> int:
    from .answer_audit import run_answer_audit

    try:
        result = run_answer_audit(args.manifest, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rebuttal-v2 answer-audit: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_input_audit(args: argparse.Namespace) -> int:
    from .input_audit import run_input_audit

    try:
        result = run_input_audit(args.manifest, args.tokenizer_lock, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rebuttal-v2 input-audit: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_annotation_export(args: argparse.Namespace) -> int:
    from .annotations import export_stage_a, export_stage_b

    try:
        if args.stage == "a":
            if args.stage_a_labels is not None or args.annotation_batch is not None:
                raise ValueError(
                    "Stage A does not accept Stage A labels or a prior annotation batch"
                )
            result = export_stage_a(
                args.audit,
                args.output_dir,
                raters=args.raters or ("rater-1", "rater-2"),
                single_rater_reason=args.single_rater_reason,
            )
        else:
            if args.stage_a_labels is None or args.annotation_batch is None:
                raise ValueError("Stage B requires --stage-a-labels and --annotation-batch")
            if args.raters is not None or args.single_rater_reason is not None:
                raise ValueError("Stage B raters are frozen by the Stage A annotation batch")
            result = export_stage_b(
                args.audit, args.stage_a_labels, args.annotation_batch, args.output_dir
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rebuttal-v2 annotation-export: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_annotation_import(args: argparse.Namespace) -> int:
    from .annotations import import_adjudicated_labels

    try:
        result = import_adjudicated_labels(
            args.audit, args.labels, args.annotation_batch, args.output_dir
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rebuttal-v2 annotation-import: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0

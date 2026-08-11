"""Training command plugin for the paper-reproduction CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typo_robust_training.data.builder import BuildTrainingDataConfig, run_build_training_data


def _run_build_data(args: argparse.Namespace) -> int:
    try:
        result = run_build_training_data(
            BuildTrainingDataConfig(
                config_path=args.config,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"build-robustness-training-data: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {result.training_records} training source record(s), "
        f"{result.training_tokens} token(s): {result.training_sources_path}"
    )
    print(
        f"wrote {result.diagnostic_records} diagnostic record(s): {result.diagnostic_manifest_path}"
    )
    print(f"natural typo statistics: {result.typo_statistics_path}")
    print(f"evaluation manifest: {result.evaluation_manifest_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_select_layers(args: argparse.Namespace) -> int:
    from typo_robust_training.localization.runner import (
        LayerSelectionRunConfig,
        run_select_distillation_layers,
    )

    try:
        result = run_select_distillation_layers(
            LayerSelectionRunConfig(
                config_path=args.config,
                diagnostic_manifest_path=args.diagnostic_manifest,
                tasks=tuple(args.tasks),
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"select-distillation-layers: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"selected decoder layers [{result.selected_window[0]}, {result.selected_window[1]}) "
        f"from {result.records} diagnostic record(s): {result.selection_path}"
    )
    print(f"per-record scans: {result.scans_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def register_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register training operations only when this separate project is installed."""

    parser = commands.add_parser(
        "build-robustness-training-data",
        help="Build leakage-resistant training, diagnostic, and held-out typo manifests.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.set_defaults(_typo_cot_plugin_handler=_run_build_data)

    selection = commands.add_parser(
        "select-distillation-layers",
        help="Causally select a contiguous layer window on diagnostic typo pairs.",
    )
    selection.add_argument("--config", required=True, type=Path)
    selection.add_argument("--diagnostic-manifest", required=True, type=Path)
    selection.add_argument("--tasks", required=True, nargs="+")
    selection.add_argument("--gpu-id", required=True)
    selection.add_argument("--output-dir", required=True, type=Path)
    selection.add_argument("--resume", action="store_true")
    selection.set_defaults(_typo_cot_plugin_handler=_run_select_layers)


__all__ = ["register_commands"]

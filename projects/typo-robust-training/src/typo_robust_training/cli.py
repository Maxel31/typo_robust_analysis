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


def _run_freeze_generic_localization_pairs(args: argparse.Namespace) -> int:
    from typo_robust_training.localization.confirmatory_pairs import (
        GenericLocalizationPairFreezeConfig,
        run_freeze_generic_localization_pairs,
    )

    try:
        result = run_freeze_generic_localization_pairs(
            GenericLocalizationPairFreezeConfig(
                config_path=args.config,
                exclude_data_paths=tuple(args.exclude_data),
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-generic-localization-pairs: error: {exc}", file=sys.stderr)
        return 1
    print(f"selection pairs: {result.selection_manifest_path}")
    print(f"validation pairs: {result.validation_manifest_path}")
    print(f"registry: {result.registry_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_select_generic_joint_window(args: argparse.Namespace) -> int:
    from typo_robust_training.localization.confirmatory_runner import (
        JointWindowSelectionRunConfig,
        run_select_generic_joint_patch_window,
    )

    try:
        result = run_select_generic_joint_patch_window(
            JointWindowSelectionRunConfig(
                config_path=args.config,
                selection_manifest_path=args.selection_manifest,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"select-generic-joint-patch-window: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"selected decoder layers [{result.selected_window[0]}, {result.selected_window[1]}): "
        f"{result.selection_path}"
    )
    print(f"per-record scans: {result.scans_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_validate_generic_joint_window(args: argparse.Namespace) -> int:
    from typo_robust_training.localization.confirmatory_runner import (
        JointWindowValidationRunConfig,
        run_validate_generic_joint_patch_window,
    )

    try:
        result = run_validate_generic_joint_patch_window(
            JointWindowValidationRunConfig(
                config_path=args.config,
                validation_manifest_path=args.validation_manifest,
                window_selection_path=args.window_selection,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"validate-generic-joint-patch-window: error: {exc}", file=sys.stderr)
        return 1
    print(f"independent generic-text validation passed={result.passed}: {result.validation_path}")
    print(f"per-record scans: {result.scans_path}")
    print(f"run manifest: {result.run_path}")
    return 0 if result.passed else 1


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

    generic_pairs = commands.add_parser(
        "freeze-generic-localization-pairs",
        help="Freeze disjoint generic-text selection and validation typo pairs.",
    )
    generic_pairs.add_argument("--config", required=True, type=Path)
    generic_pairs.add_argument("--exclude-data", required=True, action="append", type=Path)
    generic_pairs.add_argument("--output-dir", required=True, type=Path)
    generic_pairs.set_defaults(_typo_cot_plugin_handler=_run_freeze_generic_localization_pairs)

    joint_selection = commands.add_parser(
        "select-generic-joint-patch-window",
        help="Select one fixed-width residual window by generic-text joint patching.",
    )
    joint_selection.add_argument("--config", required=True, type=Path)
    joint_selection.add_argument("--selection-manifest", required=True, type=Path)
    joint_selection.add_argument("--gpu-id", required=True)
    joint_selection.add_argument("--output-dir", required=True, type=Path)
    joint_selection.add_argument("--resume", action="store_true")
    joint_selection.set_defaults(_typo_cot_plugin_handler=_run_select_generic_joint_window)

    joint_validation = commands.add_parser(
        "validate-generic-joint-patch-window",
        help="Validate the frozen joint-patch window on independent generic text.",
    )
    joint_validation.add_argument("--config", required=True, type=Path)
    joint_validation.add_argument("--validation-manifest", required=True, type=Path)
    joint_validation.add_argument("--window-selection", required=True, type=Path)
    joint_validation.add_argument("--gpu-id", required=True)
    joint_validation.add_argument("--output-dir", required=True, type=Path)
    joint_validation.add_argument("--resume", action="store_true")
    joint_validation.set_defaults(_typo_cot_plugin_handler=_run_validate_generic_joint_window)

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

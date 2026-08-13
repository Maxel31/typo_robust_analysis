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


def _run_localize_components(args: argparse.Namespace) -> int:
    from typo_robust_training.localization.component_runner import (
        ComponentLocalizationRunConfig,
        run_localize_robustness_components,
    )

    try:
        result = run_localize_robustness_components(
            ComponentLocalizationRunConfig(
                config_path=args.config,
                diagnostic_manifest_path=args.diagnostic_manifest,
                layer_selection_path=args.layer_selection,
                components=tuple(args.components),
                causal_readouts=tuple(args.causal_readouts),
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"localize-robustness-components: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"selected {result.selected_components} causally validated component(s): "
        f"{result.selection_path}"
    )
    print(f"screening universe: {result.screen_path}")
    print(f"causal observations: {result.causal_records_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_adapter_training(args: argparse.Namespace) -> int:
    from typo_robust_training.training.runner import (
        AdapterTrainingRunConfig,
        run_adapter_training,
    )

    try:
        result = run_adapter_training(
            AdapterTrainingRunConfig(
                condition=args._training_condition,
                config_path=args.config,
                training_data_dir=args.training_data,
                layer_selection_path=getattr(args, "layer_selection", None),
                component_selection_path=getattr(args, "component_selection", None),
                seed=args.seed,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"{args._training_command}: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"completed {result.optimizer_steps} optimizer step(s), "
        f"{result.student_tokens} student token(s): {result.checkpoint_path}"
    )
    print(f"training log: {result.metrics_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    condition: str,
    requires_localization: bool,
) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    if requires_localization:
        parser.add_argument("--layer-selection", required=True, type=Path)
        parser.add_argument("--component-selection", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        _typo_cot_plugin_handler=_run_adapter_training,
        _training_command=command,
        _training_condition=condition,
    )


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

    components = commands.add_parser(
        "localize-robustness-components",
        help="Reproduce exploratory neuron/head causal validation inside selected layers.",
    )
    components.add_argument("--config", required=True, type=Path)
    components.add_argument("--diagnostic-manifest", required=True, type=Path)
    components.add_argument("--layer-selection", required=True, type=Path)
    components.add_argument(
        "--components",
        required=True,
        nargs="+",
        choices=("mlp-neuron", "attention-head"),
    )
    components.add_argument(
        "--causal-readouts",
        required=True,
        nargs="+",
        choices=("answer", "multitoken-kl"),
    )
    components.add_argument("--gpu-id", required=True)
    components.add_argument("--output-dir", required=True, type=Path)
    components.add_argument("--resume", action="store_true")
    components.set_defaults(_typo_cot_plugin_handler=_run_localize_components)

    for command, condition in (
        ("train-noisy-language-model", "noisy-language-model"),
        ("train-output-matching", "output-matching"),
        ("train-global-state-alignment", "global-state-alignment"),
        ("train-localized-state-distillation", "localized-state-distillation"),
    ):
        training = commands.add_parser(
            command,
            help=f"Train the frozen {condition} condition with exact resume.",
        )
        _add_training_arguments(
            training,
            command=command,
            condition=condition,
            requires_localization=condition == "localized-state-distillation",
        )


__all__ = ["register_commands"]

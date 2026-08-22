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


def _run_freeze_evaluation(args: argparse.Namespace) -> int:
    from typo_robust_training.evaluation.freeze import (
        FreezeEvaluationRunConfig,
        run_freeze_robustness_evaluation,
    )

    try:
        result = run_freeze_robustness_evaluation(
            FreezeEvaluationRunConfig(
                protocol_path=args.protocol,
                source_config_path=args.source_config,
                exclude_data_dir=args.exclude_data,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-robustness-evaluation: error: {exc}", file=sys.stderr)
        return 1
    print(f"frozen evaluation registry: {result.registry_path}")
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


def _run_select_probe_transition(args: argparse.Namespace) -> int:
    from typo_robust_training.probe.producer import (
        ProbeTransitionProducerRunConfig,
        run_select_probe_transition,
    )

    try:
        result = run_select_probe_transition(
            ProbeTransitionProducerRunConfig(
                config_path=args.config,
                class_inventory_path=args.class_inventory,
                fit_manifest_path=args.fit_manifest,
                selection_manifest_path=args.selection_manifest,
                validation_manifest_path=args.validation_manifest,
                protected_registry_path=args.protected_registry,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"select-probe-transition: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"selected probe transition layer {result.selected_transition_layer}, "
        f"validation passed={result.validation_passed}: {result.artifact_path}"
    )
    print(f"run manifest: {result.run_path}")
    return 0 if result.validation_passed else 1


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


def _run_calibrate_sparse_autoencoder_l1(args: argparse.Namespace) -> int:
    from typo_robust_training.sae.runner import (
        SaeCalibrationRunConfig,
        run_calibrate_sae_l1,
    )

    try:
        result = run_calibrate_sae_l1(
            SaeCalibrationRunConfig(
                config_path=args.config,
                registry_path=args.registry,
                training_data_paths=tuple(args.training_data),
                gpu_id=args.gpu_id,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"calibrate-sparse-autoencoder-l1: error: {exc}", file=sys.stderr)
        return 1
    print(f"selected L1 coefficients: {result.selection_path}")
    print(f"calibration report: {result.report_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_build_sae_clean_corpus(args: argparse.Namespace) -> int:
    from typo_robust_training.sae.corpus import (
        SaeCorpusBuildConfig,
        run_build_sae_clean_corpus,
    )

    try:
        result = run_build_sae_clean_corpus(
            SaeCorpusBuildConfig(
                config_path=args.config,
                registry_path=args.registry,
                existing_data_paths=tuple(args.existing_data),
                exclusion_paths=tuple(args.exclude_data),
                training_budget=args.training_budget,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"build-sae-clean-corpus: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"built {result.supplement_records} clean supplement record(s), "
        f"{result.supplement_tokens} source token(s): {result.supplement_path}"
    )
    print(f"total eligible source tokens: {result.total_eligible_tokens}")
    print(f"source registry: {result.registry_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_train_sparse_autoencoders(args: argparse.Namespace) -> int:
    from typo_robust_training.sae.runner import SaeTrainingRunConfig, run_train_saes

    try:
        result = run_train_saes(
            SaeTrainingRunConfig(
                config_path=args.config,
                registry_path=args.registry,
                training_data_paths=tuple(args.training_data),
                l1_selection_path=args.l1_selection,
                gpu_id=args.gpu_id,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"train-sparse-autoencoders: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"trained SAE(s) for {result.trained_tokens} activation token(s), "
        f"{result.optimizer_steps} optimizer step(s): {result.checkpoint_dir}"
    )
    print(f"run manifest: {result.run_path}")
    return 0


def _run_validate_sparse_autoencoders(args: argparse.Namespace) -> int:
    from typo_robust_training.sae.runner import (
        SaeValidationRunConfig,
        run_validate_saes,
    )

    try:
        result = run_validate_saes(
            SaeValidationRunConfig(
                config_path=args.config,
                registry_path=args.registry,
                validation_data_paths=tuple(args.validation_data),
                checkpoint_dir=args.checkpoint_dir,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"validate-sparse-autoencoders: error: {exc}", file=sys.stderr)
        return 1
    print(f"WP-2 acceptance passed={result.passed}: {result.acceptance_path}")
    print(f"run manifest: {result.run_path}")
    return 0 if result.passed else 2


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
                window_validation_path=getattr(args, "window_validation", None),
                component_selection_path=getattr(args, "component_selection", None),
                seed=args.seed,
                gpu_id=args.gpu_id,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                output_dir=args.output_dir,
                resume=args.resume,
                evaluation_protocol_path=args.evaluation_protocol,
                monitor_data_dir=args.monitor_data,
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


def _run_robustness_evaluation(args: argparse.Namespace) -> int:
    from typo_robust_training.evaluation.runner import (
        RobustnessEvaluationRunConfig,
        run_robustness_evaluation,
    )

    try:
        result = run_robustness_evaluation(
            RobustnessEvaluationRunConfig(
                config_path=args.config,
                study_protocol_path=args.evaluation_protocol,
                training_data_dir=args.training_data,
                evaluation_data_dir=args.evaluation_data,
                evaluation_role=args.evaluation_role,
                layer_selection_path=args.layer_selection,
                window_validation_path=args.window_validation,
                checkpoint_paths=tuple(args.checkpoints),
                splits=tuple(args.splits),
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
                confirm_sealed_role=args.confirm_sealed_role,
                resume=args.resume,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"evaluate-typo-robustness: error: {exc}", file=sys.stderr)
        return 1
    print(f"evaluated {result.records} record(s): {result.records_path}")
    print(f"evaluated {result.corpus_records} corpus record(s): {result.corpus_records_path}")
    print(f"robustness report: {result.report_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    condition: str,
    requires_layer_selection: bool,
    accepts_component_selection: bool = False,
) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    if requires_layer_selection:
        parser.add_argument("--layer-selection", required=True, type=Path)
        parser.add_argument("--window-validation", type=Path)
    if accepts_component_selection:
        parser.add_argument("--component-selection", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluation-protocol", type=Path)
    parser.add_argument("--monitor-data", type=Path)
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

    freeze = commands.add_parser(
        "freeze-robustness-evaluation",
        help="Freeze model-independent paired evaluation text and one-use roles.",
    )
    freeze.add_argument("--protocol", required=True, type=Path)
    freeze.add_argument("--source-config", required=True, type=Path)
    freeze.add_argument("--exclude-data", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.set_defaults(_typo_cot_plugin_handler=_run_freeze_evaluation)

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

    probe_transition = commands.add_parser(
        "select-probe-transition",
        help="Fit frozen word-identity probes and select a validated denoising transition.",
    )
    probe_transition.add_argument("--config", required=True, type=Path)
    probe_transition.add_argument("--class-inventory", required=True, type=Path)
    probe_transition.add_argument("--fit-manifest", required=True, type=Path)
    probe_transition.add_argument("--selection-manifest", required=True, type=Path)
    probe_transition.add_argument("--validation-manifest", required=True, type=Path)
    probe_transition.add_argument("--protected-registry", required=True, type=Path)
    probe_transition.add_argument("--gpu-id", required=True)
    probe_transition.add_argument("--output-dir", required=True, type=Path)
    probe_transition.set_defaults(_typo_cot_plugin_handler=_run_select_probe_transition)

    sae_corpus = commands.add_parser(
        "build-sae-clean-corpus",
        help="Build a role-disjoint clean FineWeb-Edu supplement for SAE training.",
    )
    sae_corpus.add_argument("--config", required=True, type=Path)
    sae_corpus.add_argument("--registry", required=True, type=Path)
    sae_corpus.add_argument("--existing-data", required=True, action="append", type=Path)
    sae_corpus.add_argument("--exclude-data", required=True, action="append", type=Path)
    sae_corpus.add_argument(
        "--training-budget",
        required=True,
        choices=("minimum", "preferred"),
    )
    sae_corpus.add_argument("--output-dir", required=True, type=Path)
    sae_corpus.set_defaults(_typo_cot_plugin_handler=_run_build_sae_clean_corpus)

    sae_calibration = commands.add_parser(
        "calibrate-sparse-autoencoder-l1",
        help="Calibrate the frozen three-value L1 grid on clean FineWeb-Edu activations.",
    )
    sae_calibration.add_argument("--config", required=True, type=Path)
    sae_calibration.add_argument("--registry", required=True, type=Path)
    sae_calibration.add_argument("--training-data", required=True, action="append", type=Path)
    sae_calibration.add_argument("--gpu-id", required=True)
    sae_calibration.add_argument("--wandb-project", required=True)
    sae_calibration.add_argument("--wandb-entity")
    sae_calibration.add_argument("--output-dir", required=True, type=Path)
    sae_calibration.add_argument("--resume", action="store_true")
    sae_calibration.set_defaults(_typo_cot_plugin_handler=_run_calibrate_sparse_autoencoder_l1)

    sae_training = commands.add_parser(
        "train-sparse-autoencoders",
        help="Train preregistered layer-5/layer-20 SAEs on clean FineWeb-Edu only.",
    )
    sae_training.add_argument("--config", required=True, type=Path)
    sae_training.add_argument("--registry", required=True, type=Path)
    sae_training.add_argument("--training-data", required=True, action="append", type=Path)
    sae_training.add_argument("--l1-selection", required=True, type=Path)
    sae_training.add_argument("--gpu-id", required=True)
    sae_training.add_argument("--wandb-project", required=True)
    sae_training.add_argument("--wandb-entity")
    sae_training.add_argument("--output-dir", required=True, type=Path)
    sae_training.add_argument("--resume", action="store_true")
    sae_training.set_defaults(_typo_cot_plugin_handler=_run_train_sparse_autoencoders)

    sae_validation = commands.add_parser(
        "validate-sparse-autoencoders",
        help="Compute held-in SAE statistics, splice KL, and frozen WP-2 gates.",
    )
    sae_validation.add_argument("--config", required=True, type=Path)
    sae_validation.add_argument("--registry", required=True, type=Path)
    sae_validation.add_argument("--validation-data", required=True, action="append", type=Path)
    sae_validation.add_argument("--checkpoint-dir", required=True, type=Path)
    sae_validation.add_argument("--gpu-id", required=True)
    sae_validation.add_argument("--output-dir", required=True, type=Path)
    sae_validation.set_defaults(_typo_cot_plugin_handler=_run_validate_sparse_autoencoders)

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
        (
            "train-random-window-state-distillation",
            "random-window-state-distillation",
        ),
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
            requires_layer_selection=condition
            in {
                "localized-state-distillation",
                "random-window-state-distillation",
            },
            accepts_component_selection=condition == "localized-state-distillation",
        )

    evaluation = commands.add_parser(
        "evaluate-typo-robustness",
        help="Evaluate base and explicit adapters on one fixed held-out role.",
    )
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--evaluation-protocol", required=True, type=Path)
    evaluation.add_argument("--training-data", required=True, type=Path)
    evaluation.add_argument("--evaluation-data", required=True, type=Path)
    evaluation.add_argument(
        "--evaluation-role",
        required=True,
        choices=("tune", "pre-pr-gate", "final-test"),
    )
    evaluation.add_argument("--layer-selection", required=True, type=Path)
    evaluation.add_argument("--window-validation", type=Path)
    evaluation.add_argument(
        "--checkpoint",
        dest="checkpoints",
        required=True,
        action="append",
        type=Path,
    )
    evaluation.add_argument(
        "--splits",
        nargs="+",
        required=True,
        choices=("same-task", "unseen-task", "unseen-content", "unseen-typo"),
    )
    evaluation.add_argument("--gpu-id", required=True)
    evaluation.add_argument("--output-dir", required=True, type=Path)
    evaluation.add_argument("--confirm-sealed-role", action="store_true")
    evaluation.add_argument("--resume", action="store_true")
    evaluation.set_defaults(_typo_cot_plugin_handler=_run_robustness_evaluation)


__all__ = ["register_commands"]

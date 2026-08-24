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


def _run_prepare_kojima_faithful_data(args: argparse.Namespace) -> int:
    from typo_robust_training.training.kojima_faithful import (
        PrepareKojimaFaithfulDataConfig,
        prepare_kojima_faithful_data,
    )

    try:
        result = prepare_kojima_faithful_data(
            PrepareKojimaFaithfulDataConfig(
                seed=args.seed,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"prepare-kojima-faithful-data: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"prepared {result.packed_examples} packed attempt(s), "
        f"targeting {result.student_tokens} usable student token(s): "
        f"{result.packed_sources_path}"
    )
    print(f"manifest: {result.manifest_path}")
    print(f"run manifest: {result.run_path}")
    return 0


def _run_prepare_mistral_factorial_data(args: argparse.Namespace) -> int:
    from typo_robust_training.training.mistral_factorial import (
        PrepareMistralFactorialDataConfig,
        prepare_mistral_factorial_data,
    )

    try:
        result = prepare_mistral_factorial_data(
            PrepareMistralFactorialDataConfig(
                seed=args.seed,
                packed_source_dir=args.packed_source_dir,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"prepare-mistral-factorial-data: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"prepared {result.usable_examples} prevalidated factorial pair(s), "
        f"exactly {result.student_tokens} student token(s): {result.pairs_path}"
    )
    print(f"skip ledger: {result.skips_path}")
    print(f"manifest: {result.manifest_path}")
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


def _run_freeze_protected_split_registry(args: argparse.Namespace) -> int:
    from typo_robust_training.data.protected_registry import (
        ProtectedSplitOverlapError,
        freeze_protected_split_registry,
    )

    try:
        result = freeze_protected_split_registry(
            inventory_path=args.inventory,
            inventory_sha256=args.inventory_sha256,
            output_dir=args.output_dir,
        )
    except ProtectedSplitOverlapError as exc:
        print(f"freeze-protected-split-registry: error: {exc}", file=sys.stderr)
        print(exc.audit_json, file=sys.stderr)
        return 1
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-protected-split-registry: error: {exc}", file=sys.stderr)
        return 1
    print(f"frozen protected split registry: {result.registry_path}")
    print(f"producer record: {result.run_path}")
    print(f"externally pin producer record SHA256: {result.producer_record_sha256}")
    return 0


def _run_verify_protected_split_registry(args: argparse.Namespace) -> int:
    from typo_robust_training.data.protected_registry import (
        load_protected_split_registry_bundle,
    )

    try:
        result = load_protected_split_registry_bundle(
            args.producer_run,
            expected_producer_record_sha256=args.producer_record_sha256,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"verify-protected-split-registry: error: {exc}", file=sys.stderr)
        return 1
    print(f"verified protected split registry: {result.registry_path}")
    print(f"verified {result.input_records} protected input record(s)")
    return 0


def _run_freeze_protected_exclusion_denylist(args: argparse.Namespace) -> int:
    from typo_robust_training.data.protected_denylist import (
        freeze_protected_exclusion_denylist,
    )

    try:
        result = freeze_protected_exclusion_denylist(
            inventory_path=args.inventory,
            inventory_sha256=args.inventory_sha256,
            output_dir=args.output_dir,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-protected-exclusion-denylist: error: {exc}", file=sys.stderr)
        return 1
    print(f"frozen protected exclusion denylist: {result.denylist_path}")
    print(f"body-free overlap audit: {result.overlap_audit_path}")
    print(f"producer record: {result.run_path}")
    print(f"externally pin producer record SHA256: {result.producer_record_sha256}")
    return 0


def _run_verify_protected_exclusion_denylist(args: argparse.Namespace) -> int:
    from typo_robust_training.data.protected_denylist import (
        load_protected_exclusion_denylist_bundle,
    )

    try:
        result = load_protected_exclusion_denylist_bundle(
            args.producer_run,
            expected_producer_record_sha256=args.producer_record_sha256,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"verify-protected-exclusion-denylist: error: {exc}", file=sys.stderr)
        return 1
    print(f"verified protected exclusion denylist: {result.denylist_path}")
    print(f"verified {result.input_records} historical protected record(s)")
    return 0


def _run_freeze_tokenizer_attestation(args: argparse.Namespace) -> int:
    from typo_robust_training.tokenizer_freeze import freeze_tokenizer_attestation

    try:
        result = freeze_tokenizer_attestation(
            model=args.model,
            revision=args.revision,
            code_revision=args.code_revision,
            output_dir=args.output_dir,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-tokenizer-attestation: error: {exc}", file=sys.stderr)
        return 1
    print(f"frozen tokenizer attestation: {result.attestation_path}")
    print(f"freeze producer manifest: {result.run_manifest_path}")
    print(f"freeze producer record SHA256: {result.run_sha256}")
    return 0


def _run_freeze_probe_source_pool(args: argparse.Namespace) -> int:
    from typo_robust_training.probe.source_pool import (
        ProbeSourcePoolFreezeConfig,
        freeze_probe_source_pool,
    )

    try:
        result = freeze_probe_source_pool(
            ProbeSourcePoolFreezeConfig(
                parquet_path=args.source_parquet,
                parquet_sha256=args.source_parquet_sha256,
                protected_registry_path=args.protected_registry,
                protected_registry_sha256=args.protected_registry_sha256,
                code_revision=args.code_revision,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"freeze-probe-source-pool: error: {exc}", file=sys.stderr)
        return 1
    print(f"frozen probe source records: {result.records}")
    print(f"source manifest: {result.source_manifest_path}")
    print(f"freeze producer manifest: {result.run_path}")
    print(f"freeze producer record SHA256: {result.run_sha256}")
    return 0


def _run_calibrate_evaluation_v2_severity(args: argparse.Namespace) -> int:
    from typo_robust_training.evaluation.calibration_v2 import (
        run_base_only_severity_calibration,
    )

    try:
        result = run_base_only_severity_calibration(
            config_path=args.protocol,
            observations_path=args.base_observations,
            item_manifest_path=args.item_manifest,
            realized_typo_manifest_path=args.realized_typo_manifest,
            output_dir=args.output_dir,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"calibrate-evaluation-v2-severity: error: {exc}", file=sys.stderr)
        return 1
    print(f"Base-only calibration status: {result.status}")
    print(f"selected primary edit count: {result.selected_edit_count}")
    print(f"calibration artifact: {result.artifact_path}")
    print(f"run manifest: {result.run_path}")
    return 0 if result.selected_edit_count is not None else 2


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
    from typo_robust_training.probe.cohort_builder import (
        load_probe_transition_data_bundle,
    )
    from typo_robust_training.probe.producer import (
        ProbeTransitionProducerRunConfig,
        run_select_probe_transition,
    )

    try:
        bundle = load_probe_transition_data_bundle(
            args.cohort_build_run,
            expected_run_sha256=args.cohort_build_run_sha256,
        )
        result = run_select_probe_transition(
            ProbeTransitionProducerRunConfig(
                config_path=bundle.producer_config_path,
                class_inventory_path=bundle.class_inventory_path,
                fit_manifest_path=bundle.fit_manifest_path,
                selection_manifest_path=bundle.selection_manifest_path,
                validation_manifest_path=bundle.validation_manifest_path,
                protected_registry_path=bundle.protected_registry_path,
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


def _run_build_probe_transition_data(args: argparse.Namespace) -> int:
    from typo_robust_training.probe.cohort_builder import (
        ProbeTransitionDataBuildConfig,
        run_build_probe_transition_data,
    )

    try:
        result = run_build_probe_transition_data(
            ProbeTransitionDataBuildConfig(
                template_path=args.template,
                template_sha256=args.template_sha256,
                source_manifest_path=args.source_manifest,
                source_manifest_sha256=args.source_manifest_sha256,
                protected_registry_path=args.protected_registry,
                protected_registry_sha256=args.protected_registry_sha256,
                tokenizer_freeze_run_path=args.tokenizer_freeze_run,
                tokenizer_freeze_run_sha256=args.tokenizer_freeze_run_sha256,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"build-probe-transition-data: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"built {result.classes} word classes and {result.records} role-disjoint record(s): "
        f"{result.producer_config_path}"
    )
    print(f"run manifest: {result.run_path}")
    print(f"externally pin this build-run SHA256: {result.run_sha256}")
    return 0


def _run_validate_probe_transition_single_layer_gate(args: argparse.Namespace) -> int:
    from typo_robust_training.state_gate.config import load_single_layer_gate_config
    from typo_robust_training.state_gate.producer import produce_single_layer_gate_artifact
    from typo_robust_training.state_gate.runtime import HuggingFaceSingleLayerGateProvider

    try:
        protocol = load_single_layer_gate_config(args.config)
        artifact = produce_single_layer_gate_artifact(
            config_path=args.config,
            parent_probe_artifact_path=args.parent_probe_artifact,
            cohort_manifest_path=args.cohort_manifest,
            protected_registry_path=args.protected_registry,
            donor_plan_path=args.donor_plan,
            runtime_manifest_path=args.runtime_manifest,
            output_dir=args.output_dir,
            provider=HuggingFaceSingleLayerGateProvider(
                protocol=protocol,
                gpu_id=args.gpu_id,
            ),
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"validate-probe-transition-single-layer-gate: error: {exc}", file=sys.stderr)
        return 1
    print(f"single-layer causal gate passed: {artifact.artifact_sha256}")
    return 0


def _run_probe_semantic_subspace_kill_test(args: argparse.Namespace) -> int:
    from typo_robust_training.probe.subspace_kill_runner import (
        SemanticSubspaceKillRunConfig,
        run_semantic_subspace_kill_test,
    )

    try:
        result = run_semantic_subspace_kill_test(
            SemanticSubspaceKillRunConfig(
                config_path=args.config,
                parent_probe_artifact_path=args.parent_probe_artifact,
                cohort_manifest_path=args.cohort_manifest,
                pca_fit_manifest_path=args.pca_fit_manifest,
                gpu_id=args.gpu_id,
                output_dir=args.output_dir,
            )
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"run-probe-semantic-subspace-kill-test: error: {exc}", file=sys.stderr)
        return 1
    print(f"semantic-subspace causal kill test passed={result.passed}: {result.artifact_path}")
    print(f"run manifest: {result.run_path}")
    return 0 if result.passed else 2


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
                probe_selection_path=getattr(args, "probe_selection", None),
                state_gate_path=getattr(args, "state_gate", None),
                seed=args.seed,
                gpu_id=args.gpu_id,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                output_dir=args.output_dir,
                resume=args.resume,
                evaluation_protocol_path=args.evaluation_protocol,
                monitor_data_dir=args.monitor_data,
                evaluation_v2_registry_bundle_path=getattr(
                    args, "evaluation_v2_registry_bundle", None
                ),
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


def _run_materialize_probe_transition_training_config(args: argparse.Namespace) -> int:
    from typo_robust_training.training.methods import (
        materialize_probe_transition_training_config,
    )

    try:
        protocol = materialize_probe_transition_training_config(
            args.template,
            evidence_path=args.probe_selection,
            output_path=args.output_config,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"materialize-probe-transition-training-config: error: {exc}", file=sys.stderr)
        return 1
    print(f"materialized training config: {args.output_config}")
    print(f"config SHA-256: {protocol.config_sha256}")
    print(f"method evidence SHA-256: {protocol.expected_method_evidence_sha256}")
    return 0


def _run_materialize_probe_output_factorial_configs(args: argparse.Namespace) -> int:
    from typo_robust_training.training.methods import (
        materialize_probe_output_factorial_configs,
    )

    try:
        protocols = materialize_probe_output_factorial_configs(
            args.template,
            evidence_path=args.probe_selection,
            output_dir=args.output_dir,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"materialize-probe-output-factorial-configs: error: {exc}", file=sys.stderr)
        return 1
    print(f"materialized {len(protocols)} factorial configs: {args.output_dir}")
    return 0


def _run_materialize_probe_transition_state_training_config(
    args: argparse.Namespace,
) -> int:
    from typo_robust_training.training.methods import (
        materialize_probe_transition_state_training_config,
    )

    try:
        protocol = materialize_probe_transition_state_training_config(
            args.template,
            evidence_path=args.state_gate,
            output_path=args.output_config,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(
            f"materialize-probe-transition-state-training-config: error: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"materialized state training config: {args.output_config}")
    print(f"method evidence SHA-256: {protocol.expected_method_evidence_sha256}")
    return 0


def _run_materialize_probe_semantic_training_config(args: argparse.Namespace) -> int:
    from typo_robust_training.training.methods import (
        materialize_probe_semantic_subspace_training_config,
    )

    try:
        protocol = materialize_probe_semantic_subspace_training_config(
            args.template,
            evidence_path=args.kill_evidence,
            output_path=args.output_config,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"materialize-probe-semantic-training-config: error: {exc}", file=sys.stderr)
        return 1
    print(f"materialized training config: {args.output_config}")
    print(f"config SHA-256: {protocol.config_sha256}")
    print(f"method evidence SHA-256: {protocol.expected_method_evidence_sha256}")
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
                evaluation_v2_registry_bundle_path=args.evaluation_v2_registry_bundle,
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
    accepts_probe_selection: bool = False,
    accepts_state_gate: bool = False,
    requires_evaluation_v2_registry: bool = False,
) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    if requires_layer_selection:
        parser.add_argument("--layer-selection", required=True, type=Path)
        parser.add_argument("--window-validation", type=Path)
    if accepts_component_selection:
        parser.add_argument("--component-selection", type=Path)
    if accepts_probe_selection:
        parser.add_argument("--probe-selection", required=True, type=Path)
    if accepts_state_gate:
        parser.add_argument("--state-gate", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluation-protocol", type=Path)
    parser.add_argument("--monitor-data", type=Path)
    if requires_evaluation_v2_registry:
        parser.add_argument(
            "--evaluation-v2-registry-bundle",
            required=True,
            type=Path,
            help=("Closed path bundle for the training-preregistered evaluation-v2 phase."),
        )
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

    tokenizer = commands.add_parser(
        "freeze-tokenizer-attestation",
        help="Freeze an exact Hub tokenizer snapshot for all scientific runtimes.",
    )
    tokenizer.add_argument("--model", required=True)
    tokenizer.add_argument("--revision", required=True)
    tokenizer.add_argument("--code-revision", required=True)
    tokenizer.add_argument("--output-dir", required=True, type=Path)
    tokenizer.set_defaults(_typo_cot_plugin_handler=_run_freeze_tokenizer_attestation)

    probe_source_pool = commands.add_parser(
        "freeze-probe-source-pool",
        help="Freeze the unused pinned FineWeb-Edu shard after five-tier exclusion.",
    )
    probe_source_pool.add_argument("--source-parquet", required=True, type=Path)
    probe_source_pool.add_argument("--source-parquet-sha256", required=True)
    probe_source_pool.add_argument("--protected-registry", required=True, type=Path)
    probe_source_pool.add_argument("--protected-registry-sha256", required=True)
    probe_source_pool.add_argument("--code-revision", required=True)
    probe_source_pool.add_argument("--output-dir", required=True, type=Path)
    probe_source_pool.set_defaults(_typo_cot_plugin_handler=_run_freeze_probe_source_pool)

    parser = commands.add_parser(
        "build-robustness-training-data",
        help="Build leakage-resistant training, diagnostic, and held-out typo manifests.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.set_defaults(_typo_cot_plugin_handler=_run_build_data)

    faithful_data = commands.add_parser(
        "prepare-kojima-faithful-data",
        help="Freeze the pinned FineWeb packing stream for one faithful Kojima seed.",
    )
    faithful_data.add_argument("--seed", required=True, type=int, choices=(1, 42, 43, 44))
    faithful_data.add_argument("--cache-dir", type=Path)
    faithful_data.add_argument("--output-dir", required=True, type=Path)
    faithful_data.set_defaults(_typo_cot_plugin_handler=_run_prepare_kojima_faithful_data)

    factorial_data = commands.add_parser(
        "prepare-mistral-factorial-data",
        help="Freeze one prevalidated 64M pair stream shared by all five Mistral arms.",
    )
    factorial_data.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    factorial_data.add_argument("--packed-source-dir", required=True, type=Path)
    factorial_data.add_argument("--output-dir", required=True, type=Path)
    factorial_data.set_defaults(_typo_cot_plugin_handler=_run_prepare_mistral_factorial_data)

    freeze = commands.add_parser(
        "freeze-robustness-evaluation",
        help="Freeze model-independent paired evaluation text and one-use roles.",
    )
    freeze.add_argument("--protocol", required=True, type=Path)
    freeze.add_argument("--source-config", required=True, type=Path)
    freeze.add_argument("--exclude-data", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.set_defaults(_typo_cot_plugin_handler=_run_freeze_evaluation)

    protected = commands.add_parser(
        "freeze-protected-split-registry",
        help="Freeze five externally pinned JSONL tiers into a closed protected registry.",
    )
    protected.add_argument("--inventory", required=True, type=Path)
    protected.add_argument("--inventory-sha256", required=True)
    protected.add_argument("--output-dir", required=True, type=Path)
    protected.set_defaults(_typo_cot_plugin_handler=_run_freeze_protected_split_registry)

    protected_verify = commands.add_parser(
        "verify-protected-split-registry",
        help="Verify a frozen registry using its externally pinned producer-record hash.",
    )
    protected_verify.add_argument("--producer-run", required=True, type=Path)
    protected_verify.add_argument("--producer-record-sha256", required=True)
    protected_verify.set_defaults(_typo_cot_plugin_handler=_run_verify_protected_split_registry)

    exclusion = commands.add_parser(
        "freeze-protected-exclusion-denylist",
        help="Freeze overlapping historical tiers into an exclusion-only identity bundle.",
    )
    exclusion.add_argument("--inventory", required=True, type=Path)
    exclusion.add_argument("--inventory-sha256", required=True)
    exclusion.add_argument("--output-dir", required=True, type=Path)
    exclusion.set_defaults(_typo_cot_plugin_handler=_run_freeze_protected_exclusion_denylist)

    exclusion_verify = commands.add_parser(
        "verify-protected-exclusion-denylist",
        help="Verify an exclusion-only bundle using its externally pinned producer hash.",
    )
    exclusion_verify.add_argument("--producer-run", required=True, type=Path)
    exclusion_verify.add_argument("--producer-record-sha256", required=True)
    exclusion_verify.set_defaults(_typo_cot_plugin_handler=_run_verify_protected_exclusion_denylist)

    calibration_v2 = commands.add_parser(
        "calibrate-evaluation-v2-severity",
        help=(
            "Select the frozen v2 typo severity from Base-only observations, "
            "or stop without extending the grid."
        ),
    )
    calibration_v2.add_argument("--protocol", required=True, type=Path)
    calibration_v2.add_argument("--base-observations", required=True, type=Path)
    calibration_v2.add_argument("--item-manifest", required=True, type=Path)
    calibration_v2.add_argument("--realized-typo-manifest", required=True, type=Path)
    calibration_v2.add_argument("--output-dir", required=True, type=Path)
    calibration_v2.set_defaults(_typo_cot_plugin_handler=_run_calibrate_evaluation_v2_severity)

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
    probe_transition.add_argument("--cohort-build-run", required=True, type=Path)
    probe_transition.add_argument("--cohort-build-run-sha256", required=True)
    probe_transition.add_argument("--gpu-id", required=True)
    probe_transition.add_argument("--output-dir", required=True, type=Path)
    probe_transition.set_defaults(_typo_cot_plugin_handler=_run_select_probe_transition)

    probe_data = commands.add_parser(
        "build-probe-transition-data",
        help="Build hash-bound Mistral word-probe cohorts without reading model outputs.",
    )
    probe_data.add_argument("--template", required=True, type=Path)
    probe_data.add_argument("--template-sha256", required=True)
    probe_data.add_argument("--source-manifest", required=True, type=Path)
    probe_data.add_argument("--source-manifest-sha256", required=True)
    probe_data.add_argument("--protected-registry", required=True, type=Path)
    probe_data.add_argument("--protected-registry-sha256", required=True)
    probe_data.add_argument("--tokenizer-freeze-run", required=True, type=Path)
    probe_data.add_argument("--tokenizer-freeze-run-sha256", required=True)
    probe_data.add_argument("--output-dir", required=True, type=Path)
    probe_data.set_defaults(_typo_cot_plugin_handler=_run_build_probe_transition_data)

    gate = commands.add_parser(
        "validate-probe-transition-single-layer-gate",
        help="Causally validate the frozen probe transition before state KD.",
    )
    gate.add_argument("--config", required=True, type=Path)
    gate.add_argument("--parent-probe-artifact", required=True, type=Path)
    gate.add_argument("--cohort-manifest", required=True, type=Path)
    gate.add_argument("--protected-registry", required=True, type=Path)
    gate.add_argument("--donor-plan", required=True, type=Path)
    gate.add_argument("--runtime-manifest", required=True, type=Path)
    gate.add_argument("--gpu-id", required=True)
    gate.add_argument("--output-dir", required=True, type=Path)
    gate.set_defaults(_typo_cot_plugin_handler=_run_validate_probe_transition_single_layer_gate)

    semantic_kill = commands.add_parser(
        "run-probe-semantic-subspace-kill-test",
        help="Causally validate both frozen rank-16 probe subspaces before training.",
    )
    semantic_kill.add_argument("--config", required=True, type=Path)
    semantic_kill.add_argument("--parent-probe-artifact", required=True, type=Path)
    semantic_kill.add_argument("--cohort-manifest", required=True, type=Path)
    semantic_kill.add_argument("--pca-fit-manifest", required=True, type=Path)
    semantic_kill.add_argument("--gpu-id", required=True)
    semantic_kill.add_argument("--output-dir", required=True, type=Path)
    semantic_kill.set_defaults(_typo_cot_plugin_handler=_run_probe_semantic_subspace_kill_test)

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

    materialize_probe_config = commands.add_parser(
        "materialize-probe-transition-training-config",
        help="Bind validated probe evidence into a non-runnable training template.",
    )
    materialize_probe_config.add_argument("--template", required=True, type=Path)
    materialize_probe_config.add_argument("--probe-selection", required=True, type=Path)
    materialize_probe_config.add_argument("--output-config", required=True, type=Path)
    materialize_probe_config.set_defaults(
        _typo_cot_plugin_handler=_run_materialize_probe_transition_training_config
    )
    materialize_factorial = commands.add_parser(
        "materialize-probe-output-factorial-configs",
        help="Bind one probe artifact into the frozen 2x2 output-KD arms and control.",
    )
    materialize_factorial.add_argument("--template", required=True, type=Path)
    materialize_factorial.add_argument("--probe-selection", required=True, type=Path)
    materialize_factorial.add_argument("--output-dir", required=True, type=Path)
    materialize_factorial.set_defaults(
        _typo_cot_plugin_handler=_run_materialize_probe_output_factorial_configs
    )
    materialize_semantic_config = commands.add_parser(
        "materialize-probe-semantic-training-config",
        help="Bind passed semantic kill evidence into a non-runnable training template.",
    )
    materialize_semantic_config.add_argument("--template", required=True, type=Path)
    materialize_semantic_config.add_argument("--kill-evidence", required=True, type=Path)
    materialize_semantic_config.add_argument("--output-config", required=True, type=Path)
    materialize_semantic_config.set_defaults(
        _typo_cot_plugin_handler=_run_materialize_probe_semantic_training_config
    )

    materialize_state_config = commands.add_parser(
        "materialize-probe-transition-state-training-config",
        help="Bind a passed transition-layer causal gate into the state-KD template.",
    )
    materialize_state_config.add_argument("--template", required=True, type=Path)
    materialize_state_config.add_argument("--state-gate", required=True, type=Path)
    materialize_state_config.add_argument("--output-config", required=True, type=Path)
    materialize_state_config.set_defaults(
        _typo_cot_plugin_handler=_run_materialize_probe_transition_state_training_config
    )

    for command, condition in (
        ("train-noisy-language-model", "noisy-language-model"),
        ("train-output-matching", "output-matching"),
        (
            "train-kojima-faithful-output-matching",
            "kojima-faithful-output-matching",
        ),
        ("train-global-state-alignment", "global-state-alignment"),
        (
            "train-random-window-state-distillation",
            "random-window-state-distillation",
        ),
        ("train-localized-state-distillation", "localized-state-distillation"),
        (
            "train-probe-transition-output-matching",
            "probe-transition-output-matching",
        ),
        (
            "train-probe-transition-single-layer-state-distillation",
            "probe-transition-single-layer-state-distillation",
        ),
        (
            "train-probe-semantic-subspace-distillation",
            "probe-semantic-subspace-distillation",
        ),
        ("train-factorial-all-layers-all-tokens", "factorial-all-layers-all-tokens"),
        (
            "train-factorial-all-layers-downstream-horizon",
            "factorial-all-layers-downstream-horizon",
        ),
        (
            "train-factorial-probe-suffix-all-tokens",
            "factorial-probe-suffix-all-tokens",
        ),
        (
            "train-factorial-probe-suffix-downstream-horizon",
            "factorial-probe-suffix-downstream-horizon",
        ),
        (
            "train-factorial-random-layers-downstream-horizon",
            "factorial-random-layers-downstream-horizon",
        ),
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
            accepts_probe_selection=condition
            in {
                "probe-transition-output-matching",
                "probe-semantic-subspace-distillation",
                "factorial-all-layers-all-tokens",
                "factorial-all-layers-downstream-horizon",
                "factorial-probe-suffix-all-tokens",
                "factorial-probe-suffix-downstream-horizon",
                "factorial-random-layers-downstream-horizon",
            },
            accepts_state_gate=(condition == "probe-transition-single-layer-state-distillation"),
            requires_evaluation_v2_registry=condition
            in {
                "kojima-faithful-output-matching",
                "factorial-all-layers-all-tokens",
                "factorial-all-layers-downstream-horizon",
                "factorial-probe-suffix-all-tokens",
                "factorial-probe-suffix-downstream-horizon",
                "factorial-random-layers-downstream-horizon",
            },
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
    evaluation.add_argument(
        "--evaluation-v2-registry-bundle",
        type=Path,
        help=(
            "Closed path bundle for an evaluation-opening-sealed v2 final opening; "
            "omit for legacy evaluation protocols."
        ),
    )
    evaluation.add_argument("--resume", action="store_true")
    evaluation.set_defaults(_typo_cot_plugin_handler=_run_robustness_evaluation)


__all__ = ["register_commands"]

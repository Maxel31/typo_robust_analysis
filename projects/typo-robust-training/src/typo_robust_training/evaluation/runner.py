"""Condition/pair checkpointed held-out robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.evaluation.checkpoints import (
    AdapterDescriptor,
    PatchWindow,
    load_adapter_descriptors,
    load_patch_window,
)
from typo_robust_training.evaluation.config import (
    RobustnessEvaluationProtocol,
    load_robustness_evaluation_config,
)
from typo_robust_training.evaluation.data import (
    EvaluationCorpusBundle,
    EvaluationCorpusRecord,
    EvaluationDataBundle,
    EvaluationPair,
    complete_evaluation_role,
    load_evaluation_corpus_bundle,
    load_evaluation_bundle,
    load_frozen_evaluation_provenance,
)
from typo_robust_training.evaluation.metrics import build_evaluation_report
from typo_robust_training.evaluation.records import (
    CorpusEvaluationObservation,
    EvaluationObservation,
)
from typo_robust_training.integrity import sha256_file as _sha256_file
from typo_robust_training.training.provenance import (
    METHOD_EVIDENCE_CONDITIONS,
    validate_condition_evidence,
)


class RobustnessEvaluationRuntime(Protocol):
    def scan_pair(self, pair: EvaluationPair) -> EvaluationObservation: ...

    def scan_corpus(
        self,
        record: EvaluationCorpusRecord,
        *,
        max_tokens: int,
    ) -> CorpusEvaluationObservation: ...

    def provenance(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[AdapterDescriptor | None], RobustnessEvaluationRuntime]


@dataclass(frozen=True, slots=True)
class RobustnessEvaluationRunConfig:
    config_path: Path
    study_protocol_path: Path
    training_data_dir: Path
    evaluation_data_dir: Path
    evaluation_role: str
    layer_selection_path: Path
    window_validation_path: Path | None
    checkpoint_paths: tuple[Path, ...]
    splits: tuple[str, ...]
    gpu_id: str
    output_dir: Path
    confirm_sealed_role: bool
    resume: bool

    def __post_init__(self) -> None:
        for field_name in (
            "config_path",
            "study_protocol_path",
            "training_data_dir",
            "evaluation_data_dir",
            "layer_selection_path",
            "output_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.window_validation_path is not None:
            object.__setattr__(
                self,
                "window_validation_path",
                Path(self.window_validation_path),
            )
        object.__setattr__(
            self,
            "checkpoint_paths",
            tuple(Path(path) for path in self.checkpoint_paths),
        )


@dataclass(frozen=True, slots=True)
class RobustnessEvaluationRunResult:
    records: int
    records_path: Path
    corpus_records: int
    corpus_records_path: Path
    report_path: Path
    run_path: Path
    gate_passed: bool


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _condition_inventory(
    descriptors: Sequence[AdapterDescriptor],
) -> tuple[AdapterDescriptor | None, ...]:
    return (None, *descriptors)


def _condition_id(descriptor: AdapterDescriptor | None) -> str:
    return "base" if descriptor is None else descriptor.condition_id


def _validate_production_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    protocol: RobustnessEvaluationProtocol,
    descriptor: AdapterDescriptor | None,
) -> None:
    """Fail closed when a production evaluator cannot attest its loaded snapshot."""

    expected = {
        "runtime": "HuggingFaceRobustnessEvaluationRuntime/v3",
        "model": protocol.model,
        "requested_revision": protocol.model_revision,
        "model_revision": protocol.model_revision,
        "tokenizer_revision": protocol.model_revision,
        "condition": "base" if descriptor is None else descriptor.condition,
        "seed": None if descriptor is None else descriptor.seed,
        "adapter_sha256": None if descriptor is None else descriptor.adapter_sha256,
        "method_evidence_sha256": (
            None if descriptor is None else descriptor.method_evidence_sha256
        ),
    }
    if any(provenance.get(field) != value for field, value in expected.items()):
        raise ValueError("evaluation runtime model/tokenizer provenance differs")
    code_revision = provenance.get("code_revision")
    source_tree_sha256 = provenance.get("source_tree_sha256")
    if (
        not isinstance(code_revision, str)
        or len(code_revision) != 40
        or any(character not in "0123456789abcdef" for character in code_revision)
        or not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_tree_sha256)
    ):
        raise ValueError("evaluation runtime source provenance differs")


def _validate_production_runtime_inventory(
    runtime: Mapping[str, object],
    *,
    protocol: RobustnessEvaluationProtocol,
    descriptors: Sequence[AdapterDescriptor],
    require_complete: bool,
) -> None:
    inventory = {
        _condition_id(descriptor): descriptor
        for descriptor in _condition_inventory(descriptors)
    }
    if not set(runtime).issubset(inventory) or (
        require_complete and set(runtime) != set(inventory)
    ):
        raise ValueError("evaluation runtime provenance inventory differs")
    for condition_id, provenance in runtime.items():
        if not isinstance(provenance, Mapping):
            raise ValueError("evaluation runtime provenance differs")
        _validate_production_runtime_provenance(
            provenance,
            protocol=protocol,
            descriptor=inventory[condition_id],
        )


def _experiment_binding(
    *,
    protocol: RobustnessEvaluationProtocol,
    study_protocol_sha256: str,
    descriptors: Sequence[AdapterDescriptor],
    patch_window: PatchWindow,
    training_sources_sha256: str | None,
) -> str:
    has_method_evidence = any(
        descriptor.method_evidence_sha256 is not None for descriptor in descriptors
    )
    adapters = []
    for descriptor in sorted(descriptors, key=lambda item: item.condition_id):
        adapter = {
            "condition_id": descriptor.condition_id,
            "adapter_sha256": descriptor.adapter_sha256,
            "config_sha256": descriptor.config_sha256,
            "training_data_sha256": descriptor.training_data_sha256,
            "data_identity_sha256": descriptor.data_identity_sha256,
            "localization_sha256": descriptor.localization_sha256,
        }
        if has_method_evidence:
            adapter["method_evidence_sha256"] = descriptor.method_evidence_sha256
        adapters.append(adapter)
    return _canonical_sha256(
        {
            "schema_version": (
                "robustness-evaluation-experiment-binding/v3"
                if has_method_evidence
                else "robustness-evaluation-experiment-binding/v2"
            ),
            "config_sha256": protocol.config_sha256,
            "study_protocol_sha256": study_protocol_sha256,
            "patch_window_sha256": patch_window.artifact_sha256,
            "patch_layers": list(patch_window.layers),
            "training_sources_sha256": training_sources_sha256,
            "adapters": adapters,
        }
    )


def _access_binding(
    *,
    experiment_binding: str,
    config: RobustnessEvaluationRunConfig,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "robustness-evaluation-access-binding/v2",
            "experiment_binding_sha256": experiment_binding,
            "evaluation_role": config.evaluation_role,
            "splits": list(config.splits),
        }
    )


def _checkpoint_path(work_dir: Path, *, condition_id: str, record_id: str) -> Path:
    safe = condition_id.replace(":", "__")
    directory = work_dir / safe
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{record_id}.json"


def _corpus_checkpoint_path(work_dir: Path, *, condition_id: str, record_id: str) -> Path:
    safe = condition_id.replace(":", "__")
    directory = work_dir / safe
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{record_id}.json"


def _write_checkpoint(
    path: Path,
    *,
    binding: str,
    condition_id: str,
    observation: EvaluationObservation,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "robustness-evaluation-checkpoint/v1",
            "access_binding_sha256": binding,
            "condition_id": condition_id,
            "record_id": observation.record_id,
            "observation": observation.as_dict(),
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    binding: str,
    condition_id: str,
    record_id: str,
) -> EvaluationObservation:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    expected = {
        "schema_version",
        "access_binding_sha256",
        "condition_id",
        "record_id",
        "observation",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version") != "robustness-evaluation-checkpoint/v1"
        or payload.get("access_binding_sha256") != binding
        or payload.get("condition_id") != condition_id
        or payload.get("record_id") != record_id
    ):
        raise ValueError(f"evaluation checkpoint binding differs: {path}")
    observation = EvaluationObservation.from_dict(payload["observation"])
    if observation.condition_id != condition_id or observation.record_id != record_id:
        raise ValueError(f"evaluation checkpoint observation identity differs: {path}")
    return observation


def _write_corpus_checkpoint(
    path: Path,
    *,
    binding: str,
    condition_id: str,
    observation: CorpusEvaluationObservation,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "robustness-evaluation-corpus-checkpoint/v1",
            "access_binding_sha256": binding,
            "condition_id": condition_id,
            "record_id": observation.record_id,
            "observation": observation.as_dict(),
        },
    )


def _load_corpus_checkpoint(
    path: Path,
    *,
    binding: str,
    condition_id: str,
    record_id: str,
) -> CorpusEvaluationObservation:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    expected = {
        "schema_version",
        "access_binding_sha256",
        "condition_id",
        "record_id",
        "observation",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version") != "robustness-evaluation-corpus-checkpoint/v1"
        or payload.get("access_binding_sha256") != binding
        or payload.get("condition_id") != condition_id
        or payload.get("record_id") != record_id
    ):
        raise ValueError(f"evaluation corpus checkpoint binding differs: {path}")
    observation = CorpusEvaluationObservation.from_dict(payload["observation"])
    if observation.condition_id != condition_id or observation.record_id != record_id:
        raise ValueError(f"evaluation corpus checkpoint observation identity differs: {path}")
    return observation


def _write_records(path: Path, rows: Sequence[EvaluationObservation]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row.as_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_corpus_records(
    path: Path,
    rows: Sequence[CorpusEvaluationObservation],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row.as_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_injected_inputs(
    config: RobustnessEvaluationRunConfig,
    *,
    protocol: RobustnessEvaluationProtocol,
    data_bundle: EvaluationDataBundle | None,
    corpus_bundle: EvaluationCorpusBundle | None,
    descriptors: Sequence[AdapterDescriptor] | None,
    patch_window: PatchWindow | None,
) -> tuple[
    tuple[AdapterDescriptor, ...],
    PatchWindow,
    EvaluationDataBundle | None,
    EvaluationCorpusBundle | None,
]:
    if descriptors is None:
        resolved_descriptors = load_adapter_descriptors(
            config.checkpoint_paths,
            protocol=protocol,
        )
    else:
        resolved_descriptors = tuple(descriptors)
        if any(not isinstance(item, AdapterDescriptor) for item in resolved_descriptors):
            raise TypeError("injected evaluation descriptors are invalid")
        if tuple(path.resolve() for path in config.checkpoint_paths) != tuple(
            item.path.resolve() for item in resolved_descriptors
        ):
            raise ValueError("injected evaluation descriptors differ from public arguments")
    identities = [item.condition_id for item in resolved_descriptors]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("evaluation adapter identities must be non-empty and unique")
    for item in resolved_descriptors:
        try:
            validate_condition_evidence(
                condition=item.condition,
                localization_sha256=item.localization_sha256,
                method_evidence_sha256=item.method_evidence_sha256,
            )
        except ValueError as exc:
            raise ValueError(f"injected evaluation {exc}") from exc
    training_hashes = {item.training_data_sha256 for item in resolved_descriptors}
    if len(training_hashes) != 1:
        raise ValueError("evaluation adapters were trained from different data identities")
    data_identities = {item.data_identity_sha256 for item in resolved_descriptors}
    if len(data_identities) != 1:
        raise ValueError("evaluation adapters use different training data identities")
    for condition in {item.condition for item in resolved_descriptors}:
        config_hashes = {
            item.config_sha256 for item in resolved_descriptors if item.condition == condition
        }
        if len(config_hashes) != 1:
            raise ValueError(f"evaluation {condition} training configuration differs across seeds")
        if condition in METHOD_EVIDENCE_CONDITIONS:
            evidence_hashes = {
                item.method_evidence_sha256
                for item in resolved_descriptors
                if item.condition == condition
            }
            if len(evidence_hashes) != 1:
                raise ValueError(f"evaluation {condition} method evidence differs across seeds")
    if patch_window is None:
        resolved_window = load_patch_window(
            config.layer_selection_path,
            validation_path=config.window_validation_path,
            protocol=protocol,
        )
    elif not isinstance(patch_window, PatchWindow):
        raise TypeError("injected evaluation patch window is invalid")
    else:
        resolved_window = patch_window
    localized = tuple(
        item for item in resolved_descriptors if item.condition == "localized-state-distillation"
    )
    if config.evaluation_role != "tune":
        expected_seeds = set(protocol.seed_inventory)
        required_complete_conditions = {
            "localized-state-distillation",
            *(
                item.condition
                for item in resolved_descriptors
                if item.condition in METHOD_EVIDENCE_CONDITIONS
            ),
        }
        for condition in sorted(required_complete_conditions):
            condition_seeds = {
                item.seed for item in resolved_descriptors if item.condition == condition
            }
            if condition_seeds != expected_seeds:
                raise ValueError(
                    f"sealed evaluation requires {condition} checkpoints "
                    "for the complete seed inventory"
                )
    if localized and (
        resolved_window.localization_sha256 is None
        or any(
            item.localization_sha256 != resolved_window.localization_sha256 for item in localized
        )
    ):
        raise ValueError("evaluation patch window differs from adapter localization evidence")
    if data_bundle is not None:
        if not isinstance(data_bundle, EvaluationDataBundle):
            raise TypeError("injected evaluation data bundle is invalid")
        if data_bundle.evaluation_role != config.evaluation_role:
            raise ValueError("injected evaluation data role differs")
    if corpus_bundle is not None:
        if not isinstance(corpus_bundle, EvaluationCorpusBundle):
            raise TypeError("injected evaluation corpus bundle is invalid")
        if corpus_bundle.evaluation_role != config.evaluation_role:
            raise ValueError("injected evaluation corpus role differs")
    return resolved_descriptors, resolved_window, data_bundle, corpus_bundle


def run_robustness_evaluation(
    config: RobustnessEvaluationRunConfig,
    *,
    runtime_factory: RuntimeFactory | None = None,
    data_bundle: EvaluationDataBundle | None = None,
    corpus_bundle: EvaluationCorpusBundle | None = None,
    descriptors: Sequence[AdapterDescriptor] | None = None,
    patch_window: PatchWindow | None = None,
) -> RobustnessEvaluationRunResult:
    """Evaluate base plus explicit adapters, checkpointing every pair."""

    if not isinstance(config, RobustnessEvaluationRunConfig):
        raise TypeError("evaluation config must be RobustnessEvaluationRunConfig")
    if not config.gpu_id or "," in config.gpu_id:
        raise ValueError("--gpu-id must name one physical GPU")
    protocol = load_robustness_evaluation_config(config.config_path)
    from typo_robust_training.evaluation.study import load_evaluation_study_protocol

    study = load_evaluation_study_protocol(config.study_protocol_path)
    if (
        study.training_seeds != protocol.seed_inventory
        or study.bootstrap_replicates != protocol.bootstrap_replicates
        or study.bootstrap_seed != protocol.bootstrap_seed
        or study.confidence_level != protocol.confidence_level
        or study.max_new_tokens != protocol.max_new_tokens
        or study.gates["patch_audit_is_blocking"] is not False
    ):
        raise ValueError("evaluation runtime and study protocols differ")
    if (data_bundle is not None or corpus_bundle is not None) and config.evaluation_role != "tune":
        raise ValueError("sealed evaluation roles cannot use injected data bundles")
    if (data_bundle is None) != (corpus_bundle is None):
        raise ValueError("injected task and corpus evaluation bundles must be provided together")
    if (
        data_bundle is None
        and not (config.evaluation_data_dir.resolve() / "registry.json").is_file()
    ):
        raise ValueError("production robustness evaluation requires a frozen evaluation registry")
    (
        resolved_descriptors,
        resolved_window,
        injected_bundle,
        injected_corpus_bundle,
    ) = _validate_injected_inputs(
        config,
        protocol=protocol,
        data_bundle=data_bundle,
        corpus_bundle=corpus_bundle,
        descriptors=descriptors,
        patch_window=patch_window,
    )
    training_sources_sha256: str | None = None
    if injected_bundle is None:
        from typo_robust_training.training.data import load_training_data_provenance

        training_bundle = load_training_data_provenance(config.training_data_dir)
        if {item.training_data_sha256 for item in resolved_descriptors} != {
            training_bundle.training_data_sha256
        } or {item.data_identity_sha256 for item in resolved_descriptors} != {
            training_bundle.data_identity_sha256
        }:
            raise ValueError("evaluation adapter training-data provenance differs")
        frozen_provenance = load_frozen_evaluation_provenance(
            config.evaluation_data_dir,
            study_protocol_sha256=study.config_sha256,
        )
        training_sources_sha256 = training_bundle.artifact_sha256["training_sources.jsonl"]
        if (
            frozen_provenance.exclusion_artifact_sha256["training_sources.jsonl"]
            != training_sources_sha256
        ):
            raise ValueError(
                "frozen evaluation exclusions were built from different training sources"
            )
    experiment_binding = _experiment_binding(
        protocol=protocol,
        study_protocol_sha256=study.config_sha256,
        descriptors=resolved_descriptors,
        patch_window=resolved_window,
        training_sources_sha256=training_sources_sha256,
    )
    binding = _access_binding(experiment_binding=experiment_binding, config=config)
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if config.resume and not run_path.is_file():
        raise ValueError("evaluation --resume requires an existing run manifest")
    prior_run: Mapping[str, object] | None = None
    if config.resume:
        loaded = strict_loads(run_path.read_text(encoding="utf-8"), context=str(run_path))
        if not isinstance(loaded, Mapping):
            raise ValueError("evaluation resume run manifest must contain an object")
        prior_run = loaded
    work_dir = output_dir / ".evaluate-work"
    work_dir.mkdir(exist_ok=True)
    corpus_work_dir = output_dir / ".evaluate-corpus-work"
    corpus_work_dir.mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"
    corpus_records_path = output_dir / "corpus_records.jsonl"
    report_path = output_dir / "report.json"
    bundle = injected_bundle or load_evaluation_bundle(
        config.evaluation_data_dir,
        evaluation_role=config.evaluation_role,
        splits=config.splits,
        model=protocol.model,
        model_revision=protocol.model_revision,
        access_binding_sha256=binding,
        experiment_binding_sha256=experiment_binding,
        output_dir=output_dir,
        confirm_sealed_role=config.confirm_sealed_role,
        resume=config.resume,
        study_protocol_sha256=study.config_sha256,
    )
    resolved_corpus_bundle = injected_corpus_bundle or load_evaluation_corpus_bundle(
        config.evaluation_data_dir,
        evaluation_role=config.evaluation_role,
        study_protocol_sha256=study.config_sha256,
        access_binding_sha256=binding,
        experiment_binding_sha256=experiment_binding,
        output_dir=output_dir,
        confirm_sealed_role=config.confirm_sealed_role,
        resume=config.resume,
    )
    if any(not set(pair.strata) & set(config.splits) for pair in bundle.records):
        raise ValueError("evaluation data bundle contains an unrequested record")
    run_base = {
        "schema_version": "robustness-evaluation-run/v1",
        "operation": "evaluate-typo-robustness",
        "evaluation_role": config.evaluation_role,
        "splits": list(config.splits),
        "config_sha256": protocol.config_sha256,
        "evaluation_data_identity_sha256": bundle.data_identity_sha256,
        "training_data_identity_sha256": next(
            iter(item.data_identity_sha256 for item in resolved_descriptors)
        ),
        "training_sources_sha256": training_sources_sha256,
        "study_protocol_sha256": study.config_sha256,
        "role_manifest_sha256": bundle.manifest_sha256,
        "corpus_manifest_sha256": resolved_corpus_bundle.manifest_sha256,
        "evaluation_manifest_sha256": bundle.evaluation_manifest_sha256,
        "patch_window_sha256": resolved_window.artifact_sha256,
        "patch_layers": list(resolved_window.layers),
        "experiment_binding_sha256": experiment_binding,
        "access_binding_sha256": binding,
        "adapters": [
            {
                "condition_id": item.condition_id,
                "adapter_sha256": item.adapter_sha256,
                "path": str(item.path),
                **(
                    {"method_evidence_sha256": item.method_evidence_sha256}
                    if item.method_evidence_sha256 is not None
                    else {}
                ),
            }
            for item in resolved_descriptors
        ],
        "gpu_id": config.gpu_id,
        "resume": config.resume,
        "python": platform.python_version(),
    }
    prior_runtime: dict[str, object] = {}
    started_at = _now()
    if prior_run is not None:
        if (
            prior_run.get("schema_version") != run_base["schema_version"]
            or prior_run.get("operation") != run_base["operation"]
            or prior_run.get("access_binding_sha256") != binding
        ):
            raise ValueError("evaluation resume run binding differs")
        previous_runtime = prior_run.get("runtime", {})
        if not isinstance(previous_runtime, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Mapping)
            for name, value in previous_runtime.items()
        ):
            raise ValueError("evaluation resume runtime provenance differs")
        prior_runtime = {name: dict(value) for name, value in previous_runtime.items()}
        if runtime_factory is None:
            _validate_production_runtime_inventory(
                prior_runtime,
                protocol=protocol,
                descriptors=resolved_descriptors,
                require_complete=False,
            )
        previous_started_at = prior_run.get("started_at")
        if isinstance(previous_started_at, str) and previous_started_at:
            started_at = previous_started_at
    _write_json(run_path, {**run_base, "status": "running", "started_at": started_at})
    observations: list[EvaluationObservation] = []
    corpus_observations: list[CorpusEvaluationObservation] = []
    runtime_provenance: dict[str, object] = prior_runtime
    owned_runtime_factory = None
    try:
        active_runtime_factory = runtime_factory
        if active_runtime_factory is None:
            from typo_robust_training.evaluation.runtime import (
                HuggingFaceRobustnessEvaluationRuntimeFactory,
            )

            owned_runtime_factory = HuggingFaceRobustnessEvaluationRuntimeFactory(
                protocol=protocol,
                gpu_id=config.gpu_id,
                patch_window=resolved_window,
            )
            active_runtime_factory = owned_runtime_factory
        for descriptor in _condition_inventory(resolved_descriptors):
            condition_id = _condition_id(descriptor)
            pending_pairs = [
                pair
                for pair in bundle.records
                if not _checkpoint_path(
                    work_dir,
                    condition_id=condition_id,
                    record_id=pair.record_id,
                ).is_file()
            ]
            pending_corpus = [
                record
                for record in resolved_corpus_bundle.records
                if not _corpus_checkpoint_path(
                    corpus_work_dir,
                    condition_id=condition_id,
                    record_id=record.record_id,
                ).is_file()
            ]
            runtime = (
                active_runtime_factory(descriptor) if pending_pairs or pending_corpus else None
            )
            try:
                if runtime is not None:
                    current_provenance = dict(runtime.provenance())
                    if owned_runtime_factory is not None:
                        _validate_production_runtime_provenance(
                            current_provenance,
                            protocol=protocol,
                            descriptor=descriptor,
                        )
                    runtime_provenance[condition_id] = current_provenance
                for pair in bundle.records:
                    checkpoint = _checkpoint_path(
                        work_dir,
                        condition_id=condition_id,
                        record_id=pair.record_id,
                    )
                    if checkpoint.is_file():
                        if not config.resume:
                            raise ValueError("evaluation found a checkpoint without --resume")
                        observation = _load_checkpoint(
                            checkpoint,
                            binding=binding,
                            condition_id=condition_id,
                            record_id=pair.record_id,
                        )
                    else:
                        if runtime is None:
                            raise RuntimeError("evaluation runtime is missing for a pending pair")
                        observation = runtime.scan_pair(pair)
                        if (
                            not isinstance(observation, EvaluationObservation)
                            or observation.condition_id != condition_id
                            or observation.record_id != pair.record_id
                        ):
                            raise ValueError(
                                "evaluation runtime returned a different pair identity"
                            )
                        _write_checkpoint(
                            checkpoint,
                            binding=binding,
                            condition_id=condition_id,
                            observation=observation,
                        )
                    observations.append(observation)
                for record in resolved_corpus_bundle.records:
                    checkpoint = _corpus_checkpoint_path(
                        corpus_work_dir,
                        condition_id=condition_id,
                        record_id=record.record_id,
                    )
                    if checkpoint.is_file():
                        if not config.resume:
                            raise ValueError(
                                "evaluation found a corpus checkpoint without --resume"
                            )
                        corpus_observation = _load_corpus_checkpoint(
                            checkpoint,
                            binding=binding,
                            condition_id=condition_id,
                            record_id=record.record_id,
                        )
                    else:
                        if runtime is None:
                            raise RuntimeError(
                                "evaluation runtime is missing for a pending corpus record"
                            )
                        corpus_observation = runtime.scan_corpus(
                            record,
                            max_tokens=study.corpus_max_tokens,
                        )
                        if (
                            not isinstance(corpus_observation, CorpusEvaluationObservation)
                            or corpus_observation.condition_id != condition_id
                            or corpus_observation.record_id != record.record_id
                        ):
                            raise ValueError(
                                "evaluation runtime returned a different corpus identity"
                            )
                        _write_corpus_checkpoint(
                            checkpoint,
                            binding=binding,
                            condition_id=condition_id,
                            observation=corpus_observation,
                        )
                    corpus_observations.append(corpus_observation)
            finally:
                if runtime is not None:
                    runtime.close()
        if owned_runtime_factory is not None:
            _validate_production_runtime_inventory(
                runtime_provenance,
                protocol=protocol,
                descriptors=resolved_descriptors,
                require_complete=True,
            )
        _write_records(records_path, observations)
        _write_corpus_records(corpus_records_path, corpus_observations)
        report = {
            **build_evaluation_report(
                observations,
                protocol=protocol,
                study=study,
                corpus_observations=corpus_observations,
            ),
            "evaluation_role": config.evaluation_role,
            "splits": list(config.splits),
            "config_sha256": protocol.config_sha256,
            "data_identity_sha256": bundle.data_identity_sha256,
            "role_manifest_sha256": bundle.manifest_sha256,
            "corpus_manifest_sha256": resolved_corpus_bundle.manifest_sha256,
            "patch_window_sha256": resolved_window.artifact_sha256,
            "experiment_binding_sha256": experiment_binding,
            "access_binding_sha256": binding,
        }
        _write_json(report_path, report)
        gate = report.get("gate")
        gate_passed = isinstance(gate, Mapping) and gate.get("passed") is True
        complete_evaluation_role(
            bundle.root,
            evaluation_role=config.evaluation_role,
            access_binding_sha256=binding,
            report_sha256=_sha256_file(report_path),
            gate_passed=gate_passed,
        )
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "started_at": started_at,
                "completed_at": _now(),
                "records": len(observations),
                "corpus_records": len(corpus_observations),
                "runtime": runtime_provenance,
                "gate_passed": gate_passed,
                "outputs": {
                    "records.jsonl": {"sha256": _sha256_file(records_path)},
                    "corpus_records.jsonl": {"sha256": _sha256_file(corpus_records_path)},
                    "report.json": {"sha256": _sha256_file(report_path)},
                },
            },
        )
    except BaseException as exc:
        _write_json(
            run_path,
            {
                **run_base,
                "status": "failed",
                "started_at": started_at,
                "failed_at": _now(),
                "completed_records": len(observations),
                "completed_corpus_records": len(corpus_observations),
                "runtime": runtime_provenance,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    finally:
        if owned_runtime_factory is not None:
            owned_runtime_factory.close()
    return RobustnessEvaluationRunResult(
        records=len(observations),
        records_path=records_path,
        corpus_records=len(corpus_observations),
        corpus_records_path=corpus_records_path,
        report_path=report_path,
        run_path=run_path,
        gate_passed=gate_passed,
    )


__all__ = [
    "RobustnessEvaluationRunConfig",
    "RobustnessEvaluationRunResult",
    "RobustnessEvaluationRuntime",
    "run_robustness_evaluation",
]

"""Hash-bound WP-1 calibration/training and WP-2 validation runners."""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typo_robust_training.data.config import strict_loads
from typo_robust_training.sae.config import SaeProtocol, load_sae_protocol
from typo_robust_training.sae.data import (
    PreparedSaeSources,
    canonical_record_registry,
    prepare_sae_sources,
    sha256_file,
    write_json_atomic,
)
from typo_robust_training.sae.metrics import SaeStatisticsAccumulator, evaluate_sae_gates
from typo_robust_training.sae.model import SparseAutoencoder
from typo_robust_training.sae.registry import (
    SaePreregistration,
    load_sae_preregistration,
    validate_sae_prepared_sources,
)
from typo_robust_training.sae.runtime import HuggingFaceSaeRuntime
from typo_robust_training.training.tracking import (
    WandbRunPresentation,
    start_wandb_training_tracker,
)


_CHECKPOINT_INTERVAL_TOKENS = 10_000_000
_CHECKPOINT_STATE_FILES = ("checkpoint.0.pt", "checkpoint.1.pt")


@dataclass(frozen=True, slots=True)
class SaeCalibrationRunConfig:
    config_path: Path
    registry_path: Path
    training_data_paths: tuple[Path, ...]
    gpu_id: str
    wandb_project: str
    wandb_entity: str | None
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True, slots=True)
class SaeTrainingRunConfig:
    config_path: Path
    registry_path: Path
    training_data_paths: tuple[Path, ...]
    l1_selection_path: Path
    gpu_id: str
    wandb_project: str
    wandb_entity: str | None
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True, slots=True)
class SaeValidationRunConfig:
    config_path: Path
    registry_path: Path
    validation_data_paths: tuple[Path, ...]
    checkpoint_dir: Path
    gpu_id: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class SaeCalibrationResult:
    selection_path: Path
    report_path: Path
    run_path: Path
    selected_by_layer: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class SaeTrainingResult:
    checkpoint_dir: Path
    run_path: Path
    trained_tokens: int
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class SaeValidationResult:
    acceptance_path: Path
    run_path: Path
    passed: bool


def _prepare_output(path: Path, *, resume: bool) -> Path:
    output = Path(path).resolve()
    if resume:
        if not output.is_dir():
            raise ValueError("SAE resume output directory does not exist")
    elif output.exists():
        if not output.is_dir():
            raise FileExistsError(f"SAE output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"fresh SAE run cannot overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load_inputs(
    *,
    protocol: SaeProtocol,
    preregistration: SaePreregistration,
    paths: Sequence[Path],
    output_dir: Path,
) -> PreparedSaeSources:
    prepared = prepare_sae_sources(
        paths,
        protected_manifest_sha256=preregistration.source_manifest_sha256,
        reserved_seed=protocol.reserved_order_seed,
        reserved_epoch=protocol.reserved_order_epoch,
        reserved_records=protocol.reserved_prefix_records,
    )
    validate_sae_prepared_sources(prepared, preregistration=preregistration)
    for source in (*prepared.reserved, *prepared.sources):
        if source.source_revision != protocol.source_revision:
            raise ValueError("SAE source revision differs from the frozen FineWeb-Edu revision")
        if (
            source.metadata.get("tokenizer_model") != protocol.model
            or source.metadata.get("tokenizer_revision") != protocol.model_revision
        ):
            raise ValueError("SAE source tokenizer provenance differs from the frozen model")
    registry = canonical_record_registry(
        input_paths=prepared.input_paths,
        input_sha256=prepared.input_sha256,
        reserved=prepared.reserved,
        eligible=prepared.sources,
        protected_normalized_duplicates_removed=(prepared.protected_normalized_duplicates_removed),
    )
    registry.update(
        {
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "protected_student_token_budget": protocol.protected_student_token_budget,
        }
    )
    write_json_atomic(output_dir / "source_registry.json", registry)
    return prepared


def _bindings(
    *,
    condition: str,
    protocol: SaeProtocol,
    preregistration: SaePreregistration,
    sources: PreparedSaeSources,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "condition": condition,
        "config_sha256": protocol.config_sha256,
        "preregistration_sha256": preregistration.sha256,
        "source_record_sha256": sources.record_id_sha256,
        "model": protocol.model,
        "model_revision": protocol.model_revision,
        "seed": 42,
        **dict(extra or {}),
    }


def _presentation(*, phase: str, protocol: SaeProtocol) -> WandbRunPresentation:
    names = {
        "calibration": "SAE L1 calibration · layers 5 and 20 · Gemma-3-4B-IT",
        "training": "SAE training · L5 seeds 42/43 + L20 seed 42 · Gemma-3-4B-IT",
    }
    return WandbRunPresentation(
        name=names[phase],
        group="SAE diagnostic track v1 · Gemma-3-4B-IT",
        job_type=f"sae-{phase}",
        tags=(
            "typo-robustness",
            "sae-diagnostic",
            f"phase:{phase}",
            f"expansion:{protocol.expansion_factor}",
            "data:clean-fineweb-edu",
        ),
        notes=(
            "Diagnostic/future-study SAE work. This run does not alter the protected "
            "10M-token adapter experiments and uses no task-accuracy evaluation."
        ),
    )


def _model_key(layer: int, seed: int) -> str:
    return f"layer-{layer}-seed-{seed}"


def _coefficient_key(value: float) -> str:
    return format(value, ".10g").replace(".", "p").replace("-", "m")


def _candidate_key(layer: int, coefficient: float) -> str:
    return f"layer-{layer}-l1-{_coefficient_key(coefficient)}"


def _instantiate_sae(*, protocol: SaeProtocol, seed: int, device: Any) -> SparseAutoencoder:
    import torch

    model = SparseAutoencoder(d_model=protocol.d_model, d_sae=protocol.d_sae, seed=seed)
    return model.to(device=device, dtype=torch.float32)


def _optimizer(model: SparseAutoencoder, *, protocol: SaeProtocol) -> Any:
    import torch

    return torch.optim.Adam(
        model.parameters(),
        lr=protocol.learning_rate,
        betas=protocol.adam_betas,
        eps=protocol.adam_epsilon,
    )


def _train_batch(
    model: SparseAutoencoder,
    optimizer: Any,
    batch: Any,
    *,
    l1_coefficient: float,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = model.loss(batch, l1_coefficient=l1_coefficient)
    total = losses["total"]
    total.backward()
    optimizer.step()
    model.renormalize_decoder_()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def select_l1_coefficients(
    metrics_by_layer: Mapping[int, Mapping[float, Mapping[str, float]]],
    *,
    median_l0_range: tuple[int, int],
) -> dict[int, float]:
    """Apply the one preregistered L0-then-FVU selection rule."""

    low, high = median_l0_range
    selected: dict[int, float] = {}
    for layer, candidates in sorted(metrics_by_layer.items()):
        eligible = [
            (float(values["fvu"]), coefficient)
            for coefficient, values in candidates.items()
            if low <= float(values["median_l0"]) <= high
        ]
        if not eligible:
            raise RuntimeError(
                f"SAE layer {layer} has no L1 candidate with median L0 in [{low}, {high}]"
            )
        selected[layer] = min(eligible, key=lambda item: (item[0], item[1]))[1]
    return selected


def _calibration_statistics(
    *,
    model: SparseAutoencoder,
    activations: Any,
    protocol: SaeProtocol,
) -> dict[str, float]:
    accumulator = SaeStatisticsAccumulator(
        d_sae=protocol.d_sae,
        dead_probability_below=protocol.dead_feature_probability_below,
    )
    import torch

    model.eval()
    with torch.inference_mode():
        for start in range(0, int(activations.shape[0]), protocol.activation_batch_size):
            batch = activations[start : start + protocol.activation_batch_size].to(
                next(model.parameters()).device,
                dtype=torch.float32,
            )
            reconstruction, features = model(batch)
            accumulator.update(batch, reconstruction, features)
    result = accumulator.finalize()
    model.train()
    return {
        "fvu": result.fvu,
        "median_l0": result.median_l0,
        "dead_feature_rate": result.dead_feature_rate,
        "reconstruction_error_median": result.reconstruction_error_median,
    }


def run_calibrate_sae_l1(
    config: SaeCalibrationRunConfig,
    *,
    runtime_factory: Callable[..., Any] = HuggingFaceSaeRuntime,
    tracker_factory: Callable[..., Any] = start_wandb_training_tracker,
) -> SaeCalibrationResult:
    protocol = load_sae_protocol(config.config_path)
    preregistration = load_sae_preregistration(config.registry_path, protocol=protocol)
    if str(config.gpu_id) != str(preregistration.sae_gpu_id):
        raise ValueError("SAE calibration must use preregistered GPU 1")
    output = _prepare_output(config.output_dir, resume=config.resume)
    selection_path = output / "l1_selection.json"
    report_path = output / "calibration_report.json"
    run_path = output / "run.json"
    if config.resume and selection_path.is_file() and report_path.is_file() and run_path.is_file():
        selection = _load_l1_selection(
            selection_path,
            protocol=protocol,
            preregistration=preregistration,
        )
        return SaeCalibrationResult(
            selection_path=selection_path,
            report_path=report_path,
            run_path=run_path,
            selected_by_layer=selection,
        )
    if config.resume:
        raise ValueError("incomplete SAE L1 calibration has no exact resume boundary")
    sources = _load_inputs(
        protocol=protocol,
        preregistration=preregistration,
        paths=config.training_data_paths,
        output_dir=output,
    )
    if sources.source_tokens < protocol.l1_calibration_tokens:
        raise ValueError("SAE calibration input has insufficient unique clean tokens")
    bindings = _bindings(
        condition="sae-l1-calibration",
        protocol=protocol,
        preregistration=preregistration,
        sources=sources,
        extra={"activation_budget": protocol.l1_calibration_tokens},
    )
    tracker = tracker_factory(
        output_dir=output,
        project=config.wandb_project,
        entity=config.wandb_entity,
        bindings=bindings,
        presentation=_presentation(phase="calibration", protocol=protocol),
        resume=False,
        resume_optimizer_step=0,
    )
    try:
        runtime = runtime_factory(protocol=protocol, gpu_id=str(config.gpu_id))
        import torch

        device = runtime.device
        candidates: dict[str, tuple[int, float, SparseAutoencoder, Any]] = {}
        for layer in protocol.probe_layers:
            for coefficient in protocol.l1_coefficients:
                key = _candidate_key(layer, coefficient)
                model = _instantiate_sae(protocol=protocol, seed=42, device=device)
                candidates[key] = (
                    layer,
                    coefficient,
                    model,
                    _optimizer(model, protocol=protocol),
                )
        buffers = tuple(
            runtime.iter_activation_buffers(
                sources.sources,
                layer_indices=protocol.probe_layers,
                target_tokens=protocol.l1_calibration_tokens,
            )
        )
        if len(buffers) != 1:
            raise RuntimeError("SAE L1 calibration must fit one frozen shuffle buffer")
        activations = buffers[0].activations_by_layer
        step = 0
        started = time.monotonic()
        for start in range(0, protocol.l1_calibration_tokens, protocol.activation_batch_size):
            step += 1
            metrics: dict[str, int | float] = {"train/optimizer_step": step}
            for key, (layer, coefficient, model, optimizer) in candidates.items():
                batch = activations[layer][start : start + protocol.activation_batch_size].to(
                    device, dtype=torch.float32
                )
                values = _train_batch(
                    model,
                    optimizer,
                    batch,
                    l1_coefficient=coefficient,
                )
                for name, value in values.items():
                    metrics[f"train/{key}/{name}"] = value
            metrics["train/activation_tokens"] = min(
                start + protocol.activation_batch_size,
                protocol.l1_calibration_tokens,
            )
            metrics["train/elapsed_seconds"] = time.monotonic() - started
            metrics["system/gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            tracker.log_optimizer_step(metrics, optimizer_step=step)

        metrics_by_layer: dict[int, dict[float, dict[str, float]]] = {
            layer: {} for layer in protocol.probe_layers
        }
        for _key, (layer, coefficient, model, _optimizer_value) in candidates.items():
            metrics_by_layer[layer][coefficient] = _calibration_statistics(
                model=model,
                activations=activations[layer],
                protocol=protocol,
            )
        selected = select_l1_coefficients(
            metrics_by_layer,
            median_l0_range=protocol.median_l0_range,
        )
        report = {
            "schema_version": "robustness-sae-l1-calibration-report/v1",
            "bindings": bindings,
            "selection_rule": protocol.l1_selection_rule,
            "candidate_metrics": {
                str(layer): {str(coefficient): values for coefficient, values in rows.items()}
                for layer, rows in metrics_by_layer.items()
            },
            "selected_by_layer": {
                str(layer): coefficient for layer, coefficient in selected.items()
            },
            "optimizer_steps": step,
            "activation_tokens": protocol.l1_calibration_tokens,
            "runtime": dict(runtime.provenance()),
        }
        write_json_atomic(report_path, report)
        selection_payload = {
            "schema_version": "robustness-sae-l1-selection/v1",
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "source_record_sha256": sources.record_id_sha256,
            "selection_rule": protocol.l1_selection_rule,
            "selected_by_layer": {
                str(layer): coefficient for layer, coefficient in selected.items()
            },
            "calibration_report_sha256": sha256_file(report_path),
        }
        write_json_atomic(selection_path, selection_payload)
        write_json_atomic(
            run_path,
            {
                "schema_version": "robustness-sae-calibration-run/v1",
                "operation": "calibrate-sparse-autoencoder-l1",
                "bindings": bindings,
                "outputs": {
                    report_path.name: sha256_file(report_path),
                    selection_path.name: sha256_file(selection_path),
                },
                "wandb": dict(tracker.provenance()),
            },
        )
        tracker.finish(
            status="completed",
            summary={
                "optimizer_steps": step,
                "activation_tokens": protocol.l1_calibration_tokens,
            },
        )
    except Exception:
        tracker.finish(status="failed", summary={"failure": 1})
        raise
    return SaeCalibrationResult(
        selection_path=selection_path,
        report_path=report_path,
        run_path=run_path,
        selected_by_layer=selected,
    )


def _load_l1_selection(
    path: Path,
    *,
    protocol: SaeProtocol,
    preregistration: SaePreregistration,
    expected_source_sha256: str | None = None,
) -> dict[int, float]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError("SAE L1 selection is not a file")
    payload = strict_loads(resolved.read_text(encoding="utf-8"), context=str(resolved))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "config_sha256",
        "preregistration_sha256",
        "source_record_sha256",
        "selection_rule",
        "selected_by_layer",
        "calibration_report_sha256",
    }:
        raise ValueError("SAE L1 selection fields differ")
    if (
        payload.get("schema_version") != "robustness-sae-l1-selection/v1"
        or payload.get("config_sha256") != protocol.config_sha256
        or payload.get("preregistration_sha256") != preregistration.sha256
        or payload.get("selection_rule") != protocol.l1_selection_rule
    ):
        raise ValueError("SAE L1 selection bindings differ")
    source_sha256 = payload.get("source_record_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or (expected_source_sha256 is not None and source_sha256 != expected_source_sha256)
    ):
        raise ValueError("SAE L1 selection source stream differs")
    report_path = resolved.with_name("calibration_report.json")
    if not report_path.is_file() or payload.get("calibration_report_sha256") != sha256_file(
        report_path
    ):
        raise ValueError("SAE L1 calibration report hash differs")
    selected = payload["selected_by_layer"]
    if not isinstance(selected, Mapping) or set(selected) != {
        str(v) for v in protocol.probe_layers
    }:
        raise ValueError("SAE L1 selected layers differ")
    result: dict[int, float] = {}
    for layer_text, value in selected.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) not in protocol.l1_coefficients
        ):
            raise ValueError("SAE selected L1 coefficient is outside the frozen grid")
        result[int(layer_text)] = float(value)
    return result


def _torch_save_atomic(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _next_training_checkpoint_state_path(output: Path) -> Path:
    """Choose the inactive checkpoint slot without invalidating the committed one."""

    metadata_path = output / "checkpoint.json"
    if not metadata_path.is_file():
        return output / _CHECKPOINT_STATE_FILES[0]
    payload = strict_loads(metadata_path.read_text(encoding="utf-8"), context=str(metadata_path))
    if not isinstance(payload, Mapping):
        raise ValueError("SAE checkpoint metadata must be an object")
    current = payload.get("state_file")
    if current == _CHECKPOINT_STATE_FILES[0]:
        return output / _CHECKPOINT_STATE_FILES[1]
    if current in {_CHECKPOINT_STATE_FILES[1], "checkpoint.pt"}:
        return output / _CHECKPOINT_STATE_FILES[0]
    raise ValueError("SAE checkpoint state file is outside the supported slots")


def _save_training_checkpoint(
    *,
    output: Path,
    bindings: Mapping[str, object],
    models: Mapping[str, SparseAutoencoder],
    optimizers: Mapping[str, Any],
    cursor: Mapping[str, int],
) -> None:
    # The metadata file is the commit point.  Write the new state to the inactive
    # slot first so a crash before the metadata swap leaves the prior generation
    # fully loadable.  Keeping two bounded slots also avoids checkpoint buildup.
    state_path = _next_training_checkpoint_state_path(output)
    _torch_save_atomic(
        state_path,
        {
            "schema_version": "robustness-sae-training-state/v1",
            "bindings": dict(bindings),
            "cursor": dict(cursor),
            "models": {key: model.state_dict() for key, model in models.items()},
            "optimizers": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
        },
    )
    write_json_atomic(
        output / "checkpoint.json",
        {
            "schema_version": "robustness-sae-training-checkpoint/v1",
            "bindings": dict(bindings),
            "cursor": dict(cursor),
            "state_file": state_path.name,
            "state_sha256": sha256_file(state_path),
        },
    )


def _load_training_checkpoint(
    *,
    output: Path,
    bindings: Mapping[str, object],
    models: Mapping[str, SparseAutoencoder],
    optimizers: Mapping[str, Any],
) -> dict[str, int]:
    import torch

    metadata_path = output / "checkpoint.json"
    if not metadata_path.is_file():
        raise ValueError("SAE resume requires checkpoint.json")
    payload = strict_loads(metadata_path.read_text(encoding="utf-8"), context=str(metadata_path))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "bindings",
        "cursor",
        "state_file",
        "state_sha256",
    }:
        raise ValueError("SAE checkpoint fields differ")
    if (
        payload.get("schema_version") != "robustness-sae-training-checkpoint/v1"
        or payload.get("bindings") != dict(bindings)
        or payload.get("state_file") not in {*_CHECKPOINT_STATE_FILES, "checkpoint.pt"}
    ):
        raise ValueError("SAE checkpoint bindings differ")
    state_path = output / str(payload["state_file"])
    if not state_path.is_file():
        raise ValueError("SAE checkpoint state file is missing")
    if payload.get("state_sha256") != sha256_file(state_path):
        raise ValueError("SAE checkpoint state hash differs")
    state = torch.load(
        state_path,
        map_location=next(iter(models.values())).encoder.weight.device,
        weights_only=True,
    )
    if not isinstance(state, Mapping) or state.get("bindings") != dict(bindings):
        raise ValueError("SAE runtime checkpoint bindings differ")
    model_states, optimizer_states = state.get("models"), state.get("optimizers")
    if not isinstance(model_states, Mapping) or set(model_states) != set(models):
        raise ValueError("SAE checkpoint model inventory differs")
    if not isinstance(optimizer_states, Mapping) or set(optimizer_states) != set(optimizers):
        raise ValueError("SAE checkpoint optimizer inventory differs")
    for key, model in models.items():
        model.load_state_dict(model_states[key])
        optimizers[key].load_state_dict(optimizer_states[key])
    cursor = state.get("cursor")
    expected = {
        "trained_tokens",
        "optimizer_steps",
        "next_source_index",
        "next_source_offset",
        "next_buffer_index",
    }
    if (
        not isinstance(cursor, Mapping)
        or set(cursor) != expected
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in cursor.values()
        )
    ):
        raise ValueError("SAE checkpoint cursor differs")
    return {str(key): int(value) for key, value in cursor.items()}


def _save_activation_subsample(
    *,
    output: Path,
    buffer: Any,
    protocol: SaeProtocol,
    already_saved_tokens: int,
) -> int:
    remaining = protocol.activation_subsample_tokens - already_saved_tokens
    if remaining <= 0:
        return already_saved_tokens
    take = min(remaining, buffer.tokens)
    for layer in protocol.activation_subsample_layers:
        path = (
            output
            / "activation_subsample"
            / f"layer-{layer}"
            / f"shard-{buffer.buffer_index:06d}.pt"
        )
        _torch_save_atomic(
            path,
            {
                "schema_version": "robustness-sae-activation-shard/v1",
                "config_sha256": protocol.config_sha256,
                "layer": layer,
                "tokens": take,
                "buffer_index": buffer.buffer_index,
                "activations": buffer.activations_by_layer[layer][:take].contiguous(),
            },
        )
    return already_saved_tokens + take


def _completed_training_result(
    *,
    output: Path,
    bindings: Mapping[str, object],
    protocol: SaeProtocol,
) -> SaeTrainingResult | None:
    run_path = output / "run.json"
    if not run_path.is_file():
        return None
    run = strict_loads(run_path.read_text(encoding="utf-8"), context=str(run_path))
    if (
        not isinstance(run, Mapping)
        or set(run)
        != {
            "schema_version",
            "operation",
            "bindings",
            "cursor",
            "l1_by_model",
            "models",
            "source_registry_sha256",
            "runtime",
            "wandb",
        }
        or run.get("schema_version") != "robustness-sae-training-run/v1"
        or run.get("operation") != "train-sparse-autoencoders"
        or run.get("bindings") != dict(bindings)
    ):
        raise ValueError("completed SAE training run bindings differ")
    source_registry = output / "source_registry.json"
    if not source_registry.is_file() or run.get("source_registry_sha256") != sha256_file(
        source_registry
    ):
        raise ValueError("completed SAE training source registry hash differs")
    cursor = run.get("cursor")
    if (
        not isinstance(cursor, Mapping)
        or cursor.get("trained_tokens") != protocol.minimum_training_tokens
        or isinstance(cursor.get("optimizer_steps"), bool)
        or not isinstance(cursor.get("optimizer_steps"), int)
        or int(cursor["optimizer_steps"]) <= 0
    ):
        raise ValueError("completed SAE training cursor differs")
    declarations = run.get("models")
    expected_names = {
        f"sae-{_model_key(layer, seed)}.pt"
        for layer, seeds in protocol.seeds_by_layer.items()
        for seed in seeds
    }
    if not isinstance(declarations, Mapping) or set(declarations) != expected_names:
        raise ValueError("completed SAE training model inventory differs")
    for name, declaration in declarations.items():
        path = output / name
        if (
            not isinstance(declaration, Mapping)
            or not path.is_file()
            or declaration.get("sha256") != sha256_file(path)
            or declaration.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"completed SAE training model hash differs: {name}")
    return SaeTrainingResult(
        checkpoint_dir=output,
        run_path=run_path,
        trained_tokens=int(cursor["trained_tokens"]),
        optimizer_steps=int(cursor["optimizer_steps"]),
    )


def run_train_saes(
    config: SaeTrainingRunConfig,
    *,
    runtime_factory: Callable[..., Any] = HuggingFaceSaeRuntime,
    tracker_factory: Callable[..., Any] = start_wandb_training_tracker,
) -> SaeTrainingResult:
    protocol = load_sae_protocol(config.config_path)
    preregistration = load_sae_preregistration(config.registry_path, protocol=protocol)
    if str(config.gpu_id) != str(preregistration.sae_gpu_id):
        raise ValueError("SAE training must use preregistered GPU 1")
    output = _prepare_output(config.output_dir, resume=config.resume)
    sources = _load_inputs(
        protocol=protocol,
        preregistration=preregistration,
        paths=config.training_data_paths,
        output_dir=output,
    )
    required = protocol.minimum_training_tokens + protocol.statistics_tokens
    if sources.source_tokens < required:
        raise ValueError(
            "SAE training manifests contain insufficient unique clean source tokens: "
            f"{sources.source_tokens} < required training+statistics budget {required}"
        )
    selected = _load_l1_selection(
        config.l1_selection_path,
        protocol=protocol,
        preregistration=preregistration,
        expected_source_sha256=sources.record_id_sha256,
    )
    l1_sha = sha256_file(config.l1_selection_path)
    bindings = _bindings(
        condition="sae-training",
        protocol=protocol,
        preregistration=preregistration,
        sources=sources,
        extra={
            "l1_selection_sha256": l1_sha,
            "activation_budget": protocol.minimum_training_tokens,
        },
    )
    if config.resume:
        completed = _completed_training_result(
            output=output,
            bindings=bindings,
            protocol=protocol,
        )
        if completed is not None:
            return completed
    runtime = runtime_factory(protocol=protocol, gpu_id=str(config.gpu_id))
    import torch

    models: dict[str, SparseAutoencoder] = {}
    optimizers: dict[str, Any] = {}
    coefficients: dict[str, float] = {}
    for layer, seeds in protocol.seeds_by_layer.items():
        for seed in seeds:
            key = _model_key(layer, seed)
            models[key] = _instantiate_sae(protocol=protocol, seed=seed, device=runtime.device)
            optimizers[key] = _optimizer(models[key], protocol=protocol)
            coefficients[key] = selected[layer]
    cursor = {
        "trained_tokens": 0,
        "optimizer_steps": 0,
        "next_source_index": 0,
        "next_source_offset": 0,
        "next_buffer_index": 0,
    }
    if config.resume:
        cursor = _load_training_checkpoint(
            output=output,
            bindings=bindings,
            models=models,
            optimizers=optimizers,
        )
    tracker = tracker_factory(
        output_dir=output,
        project=config.wandb_project,
        entity=config.wandb_entity,
        bindings=bindings,
        presentation=_presentation(phase="training", protocol=protocol),
        resume=config.resume,
        resume_optimizer_step=cursor["optimizer_steps"],
    )
    saved_subsample_tokens = min(cursor["trained_tokens"], protocol.activation_subsample_tokens)
    started = time.monotonic()
    last_checkpoint_tokens = cursor["trained_tokens"]
    try:
        remaining = protocol.minimum_training_tokens - cursor["trained_tokens"]
        if remaining < 0:
            raise ValueError("SAE checkpoint exceeds the frozen training budget")
        if remaining:
            buffers = runtime.iter_activation_buffers(
                sources.sources,
                layer_indices=protocol.activation_subsample_layers,
                target_tokens=remaining,
                start_source_index=cursor["next_source_index"],
                start_source_offset=cursor["next_source_offset"],
                start_buffer_index=cursor["next_buffer_index"],
            )
            for buffer in buffers:
                saved_subsample_tokens = _save_activation_subsample(
                    output=output,
                    buffer=buffer,
                    protocol=protocol,
                    already_saved_tokens=saved_subsample_tokens,
                )
                for start in range(0, buffer.tokens, protocol.activation_batch_size):
                    cursor["optimizer_steps"] += 1
                    step_metrics: dict[str, int | float] = {
                        "train/optimizer_step": cursor["optimizer_steps"]
                    }
                    for key, model in models.items():
                        layer = int(key.split("-")[1])
                        batch = buffer.activations_by_layer[layer][
                            start : start + protocol.activation_batch_size
                        ].to(runtime.device, dtype=torch.float32)
                        values = _train_batch(
                            model,
                            optimizers[key],
                            batch,
                            l1_coefficient=coefficients[key],
                        )
                        for name, value in values.items():
                            step_metrics[f"train/{key}/{name}"] = value
                    consumed = min(protocol.activation_batch_size, buffer.tokens - start)
                    running_tokens = cursor["trained_tokens"] + start + consumed
                    step_metrics["train/activation_tokens"] = running_tokens
                    step_metrics["train/elapsed_seconds"] = time.monotonic() - started
                    step_metrics["system/gpu_peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated() / 2**30
                    )
                    tracker.log_optimizer_step(
                        step_metrics,
                        optimizer_step=cursor["optimizer_steps"],
                    )
                cursor.update(
                    {
                        "trained_tokens": cursor["trained_tokens"] + buffer.tokens,
                        "next_source_index": buffer.next_source_index,
                        "next_source_offset": buffer.next_source_offset,
                        "next_buffer_index": buffer.buffer_index + 1,
                    }
                )
                if (
                    cursor["trained_tokens"] - last_checkpoint_tokens >= _CHECKPOINT_INTERVAL_TOKENS
                    or cursor["trained_tokens"] == protocol.minimum_training_tokens
                ):
                    _save_training_checkpoint(
                        output=output,
                        bindings=bindings,
                        models=models,
                        optimizers=optimizers,
                        cursor=cursor,
                    )
                    last_checkpoint_tokens = cursor["trained_tokens"]
        if cursor["trained_tokens"] != protocol.minimum_training_tokens:
            raise RuntimeError("SAE training stopped before the frozen token budget")
        model_outputs: dict[str, object] = {}
        for key, model in models.items():
            path = output / f"sae-{key}.pt"
            _torch_save_atomic(
                path,
                {
                    "schema_version": "robustness-sae-model/v1",
                    "bindings": bindings,
                    "key": key,
                    "l1_coefficient": coefficients[key],
                    "trained_tokens": cursor["trained_tokens"],
                    "state_dict": model.state_dict(),
                },
            )
            model_outputs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        run_path = output / "run.json"
        write_json_atomic(
            run_path,
            {
                "schema_version": "robustness-sae-training-run/v1",
                "operation": "train-sparse-autoencoders",
                "bindings": bindings,
                "cursor": cursor,
                "l1_by_model": coefficients,
                "models": model_outputs,
                "source_registry_sha256": sha256_file(output / "source_registry.json"),
                "runtime": dict(runtime.provenance()),
                "wandb": dict(tracker.provenance()),
            },
        )
        tracker.finish(
            status="completed",
            summary={
                "optimizer_steps": cursor["optimizer_steps"],
                "activation_tokens": cursor["trained_tokens"],
            },
        )
    except Exception:
        tracker.finish(
            status="failed",
            summary={
                "optimizer_steps": cursor["optimizer_steps"],
                "activation_tokens": cursor["trained_tokens"],
            },
        )
        raise
    return SaeTrainingResult(
        checkpoint_dir=output,
        run_path=run_path,
        trained_tokens=cursor["trained_tokens"],
        optimizer_steps=cursor["optimizer_steps"],
    )


def _load_final_saes(
    *,
    checkpoint_dir: Path,
    protocol: SaeProtocol,
    preregistration_sha256: str,
    device: Any,
) -> tuple[dict[str, SparseAutoencoder], Mapping[str, object]]:
    import torch

    root = Path(checkpoint_dir).resolve()
    run_path = root / "run.json"
    if not run_path.is_file():
        raise ValueError("SAE validation requires a completed training run.json")
    run = strict_loads(run_path.read_text(encoding="utf-8"), context=str(run_path))
    if (
        not isinstance(run, Mapping)
        or set(run)
        != {
            "schema_version",
            "operation",
            "bindings",
            "cursor",
            "l1_by_model",
            "models",
            "source_registry_sha256",
            "runtime",
            "wandb",
        }
        or run.get("schema_version") != "robustness-sae-training-run/v1"
        or run.get("operation") != "train-sparse-autoencoders"
    ):
        raise ValueError("SAE completed training run schema differs")
    source_registry_path = root / "source_registry.json"
    if not source_registry_path.is_file() or run.get("source_registry_sha256") != sha256_file(
        source_registry_path
    ):
        raise ValueError("SAE completed training source registry hash differs")
    bindings = run.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("config_sha256") != protocol.config_sha256
        or bindings.get("preregistration_sha256") != preregistration_sha256
    ):
        raise ValueError("SAE completed training bindings differ")
    declared_models = run.get("models")
    if not isinstance(declared_models, Mapping):
        raise ValueError("SAE completed training model inventory differs")
    models: dict[str, SparseAutoencoder] = {}
    for layer, seeds in protocol.seeds_by_layer.items():
        for seed in seeds:
            key = _model_key(layer, seed)
            path = root / f"sae-{key}.pt"
            declaration = declared_models.get(path.name)
            if (
                not path.is_file()
                or not isinstance(declaration, Mapping)
                or declaration.get("sha256") != sha256_file(path)
                or declaration.get("bytes") != path.stat().st_size
            ):
                raise ValueError(f"SAE model artifact hash differs: {key}")
            payload = torch.load(path, map_location=device, weights_only=True)
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != "robustness-sae-model/v1"
                or payload.get("bindings") != dict(bindings)
                or payload.get("key") != key
            ):
                raise ValueError(f"SAE model artifact differs: {key}")
            model = _instantiate_sae(protocol=protocol, seed=seed, device=device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            models[key] = model
    return models, run


def _wp2_attempt_ledger_path(checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir).resolve().parent / "wp2_attempts.json"


def _load_wp2_attempts(
    *,
    checkpoint_dir: Path,
    protocol: SaeProtocol,
    preregistration: SaePreregistration,
) -> tuple[Path, list[Mapping[str, object]], str]:
    checkpoint_run = Path(checkpoint_dir).resolve() / "run.json"
    if not checkpoint_run.is_file():
        raise ValueError("SAE validation requires a completed training run.json")
    checkpoint_sha256 = sha256_file(checkpoint_run)
    ledger_path = _wp2_attempt_ledger_path(checkpoint_dir)
    attempts: list[Mapping[str, object]] = []
    if ledger_path.is_file():
        payload = strict_loads(ledger_path.read_text(encoding="utf-8"), context=str(ledger_path))
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {
                "schema_version",
                "config_sha256",
                "preregistration_sha256",
                "maximum_retrains_after_failure",
                "attempts",
            }
            or payload.get("schema_version") != "robustness-sae-wp2-attempt-ledger/v1"
            or payload.get("config_sha256") != protocol.config_sha256
            or payload.get("preregistration_sha256") != preregistration.sha256
            or payload.get("maximum_retrains_after_failure") != protocol.maximum_gate_retrains
            or not isinstance(payload.get("attempts"), list)
        ):
            raise ValueError("SAE WP-2 attempt ledger differs from preregistration")
        attempts = list(payload["attempts"])
        required = {
            "attempt",
            "checkpoint_run_sha256",
            "output_dir",
            "passed",
            "acceptance_sha256",
        }
        if any(
            not isinstance(row, Mapping)
            or set(row) != required
            or isinstance(row.get("attempt"), bool)
            or not isinstance(row.get("attempt"), int)
            or row.get("attempt") != index
            or not isinstance(row.get("checkpoint_run_sha256"), str)
            or not isinstance(row.get("output_dir"), str)
            or type(row.get("passed")) is not bool
            or not isinstance(row.get("acceptance_sha256"), str)
            for index, row in enumerate(attempts, 1)
        ):
            raise ValueError("SAE WP-2 attempt ledger entries differ")
    if any(row["passed"] is True for row in attempts):
        raise ValueError("SAE WP-2 already passed; further validation attempts are prohibited")
    if any(row["checkpoint_run_sha256"] == checkpoint_sha256 for row in attempts):
        raise ValueError("SAE WP-2 checkpoint was already evaluated")
    maximum_attempts = 1 + protocol.maximum_gate_retrains
    if len(attempts) >= maximum_attempts:
        raise ValueError("SAE WP-2 preregistered retrain limit is exhausted")
    return ledger_path, attempts, checkpoint_sha256


def _record_wp2_attempt(
    *,
    ledger_path: Path,
    prior_attempts: Sequence[Mapping[str, object]],
    checkpoint_run_sha256: str,
    output_dir: Path,
    passed: bool,
    acceptance_sha256: str,
    protocol: SaeProtocol,
    preregistration: SaePreregistration,
) -> None:
    if ledger_path.is_file():
        current = strict_loads(ledger_path.read_text(encoding="utf-8"), context=str(ledger_path))
        current_attempts = current.get("attempts") if isinstance(current, Mapping) else None
        if current_attempts != list(prior_attempts):
            raise RuntimeError("SAE WP-2 attempt ledger changed during validation")
    elif prior_attempts:
        raise RuntimeError("SAE WP-2 attempt ledger disappeared during validation")
    attempts = [
        *prior_attempts,
        {
            "attempt": len(prior_attempts) + 1,
            "checkpoint_run_sha256": checkpoint_run_sha256,
            "output_dir": str(Path(output_dir).resolve()),
            "passed": passed,
            "acceptance_sha256": acceptance_sha256,
        },
    ]
    write_json_atomic(
        ledger_path,
        {
            "schema_version": "robustness-sae-wp2-attempt-ledger/v1",
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "maximum_retrains_after_failure": protocol.maximum_gate_retrains,
            "attempts": attempts,
        },
    )


def run_validate_saes(
    config: SaeValidationRunConfig,
    *,
    runtime_factory: Callable[..., Any] = HuggingFaceSaeRuntime,
) -> SaeValidationResult:
    protocol = load_sae_protocol(config.config_path)
    preregistration = load_sae_preregistration(config.registry_path, protocol=protocol)
    if str(config.gpu_id) != str(preregistration.sae_gpu_id):
        raise ValueError("SAE validation must use preregistered GPU 1")
    ledger_path, prior_attempts, checkpoint_run_sha256 = _load_wp2_attempts(
        checkpoint_dir=config.checkpoint_dir,
        protocol=protocol,
        preregistration=preregistration,
    )
    output = _prepare_output(config.output_dir, resume=False)
    sources = _load_inputs(
        protocol=protocol,
        preregistration=preregistration,
        paths=config.validation_data_paths,
        output_dir=output,
    )
    runtime = runtime_factory(protocol=protocol, gpu_id=str(config.gpu_id))
    models, training_run = _load_final_saes(
        checkpoint_dir=config.checkpoint_dir,
        protocol=protocol,
        preregistration_sha256=preregistration.sha256,
        device=runtime.device,
    )
    cursor = training_run.get("cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError("SAE training run has no validation cursor")
    start_index = cursor.get("next_source_index")
    start_offset = cursor.get("next_source_offset")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (start_index, start_offset)
    ):
        raise ValueError("SAE validation cursor differs")
    training_bindings = training_run.get("bindings")
    if (
        not isinstance(training_bindings, Mapping)
        or training_bindings.get("source_record_sha256") != sources.record_id_sha256
    ):
        raise ValueError("SAE validation data stream differs from the training stream")

    accumulators = {
        key: SaeStatisticsAccumulator(
            d_sae=protocol.d_sae,
            dead_probability_below=protocol.dead_feature_probability_below,
        )
        for key in models
    }
    import torch

    final_buffer = None
    for buffer in runtime.iter_activation_buffers(
        sources.sources,
        layer_indices=protocol.probe_layers,
        target_tokens=protocol.statistics_tokens,
        start_source_index=int(start_index),
        start_source_offset=int(start_offset),
    ):
        final_buffer = buffer
        for key, model in models.items():
            layer = int(key.split("-")[1])
            values = buffer.activations_by_layer[layer]
            with torch.inference_mode():
                for start in range(0, buffer.tokens, protocol.activation_batch_size):
                    batch = values[start : start + protocol.activation_batch_size].to(
                        runtime.device, dtype=torch.float32
                    )
                    reconstruction, features = model(batch)
                    accumulators[key].update(batch, reconstruction, features)
    if final_buffer is None:
        raise RuntimeError("SAE validation collected no statistics activations")
    splice_start = final_buffer.next_source_index + (1 if final_buffer.next_source_offset else 0)
    splice_rows = sources.sources[splice_start : splice_start + protocol.splice_documents]
    if len(splice_rows) != protocol.splice_documents:
        raise ValueError("SAE validation has insufficient unused splice documents")
    splice_by_key: dict[str, list[float]] = {key: [] for key in models}
    for source in splice_rows:
        for key, model in models.items():
            layer = int(key.split("-")[1])
            splice_by_key[key].extend(runtime.splice_kl(source.clean_text, layer=layer, sae=model))

    reports: dict[str, object] = {}
    all_passed = True
    for key, accumulator in accumulators.items():
        result = accumulator.finalize()
        splice_median = statistics.median(splice_by_key[key])
        statistics_path = output / f"statistics-{key}.json"
        statistics_payload = {
            "schema_version": "robustness-sae-statistics/v1",
            "key": key,
            "tokens": result.tokens,
            "fvu": result.fvu,
            "median_l0": result.median_l0,
            "dead_feature_rate": result.dead_feature_rate,
            "reconstruction_error_median": result.reconstruction_error_median,
            "feature_firing_probabilities": list(result.firing_probabilities),
            "splice_kl_median_nats_per_token": splice_median,
            "splice_documents": protocol.splice_documents,
        }
        write_json_atomic(statistics_path, statistics_payload)
        gate = evaluate_sae_gates(
            fvu=result.fvu,
            median_l0=result.median_l0,
            dead_feature_rate=result.dead_feature_rate,
            splice_kl_median=splice_median,
            statistics_saved=statistics_path.is_file(),
            fvu_max=protocol.fvu_max,
            median_l0_range=protocol.median_l0_range,
            dead_feature_rate_max=protocol.dead_feature_rate_max,
            splice_kl_max=protocol.splice_kl_max,
        )
        all_passed = all_passed and gate.passed
        reports[key] = {
            "passed": gate.passed,
            "conditions": dict(gate.conditions),
            "statistics_file": statistics_path.name,
            "statistics_sha256": sha256_file(statistics_path),
            "metrics": {
                "fvu": result.fvu,
                "median_l0": result.median_l0,
                "dead_feature_rate": result.dead_feature_rate,
                "splice_kl_median_nats_per_token": splice_median,
            },
        }
    acceptance_path = output / "wp2_acceptance.json"
    write_json_atomic(
        acceptance_path,
        {
            "schema_version": "robustness-sae-wp2-acceptance/v1",
            "passed": all_passed,
            "config_sha256": protocol.config_sha256,
            "preregistration_sha256": preregistration.sha256,
            "models": reports,
            "rule": "all-three-preregistered-saes-must-pass/v1",
        },
    )
    run_path = output / "run.json"
    write_json_atomic(
        run_path,
        {
            "schema_version": "robustness-sae-validation-run/v1",
            "operation": "validate-sparse-autoencoders",
            "passed": all_passed,
            "acceptance_sha256": sha256_file(acceptance_path),
            "runtime": dict(runtime.provenance()),
            "training_run_sha256": sha256_file(Path(config.checkpoint_dir) / "run.json"),
        },
    )
    _record_wp2_attempt(
        ledger_path=ledger_path,
        prior_attempts=prior_attempts,
        checkpoint_run_sha256=checkpoint_run_sha256,
        output_dir=output,
        passed=all_passed,
        acceptance_sha256=sha256_file(acceptance_path),
        protocol=protocol,
        preregistration=preregistration,
    )
    return SaeValidationResult(
        acceptance_path=acceptance_path,
        run_path=run_path,
        passed=all_passed,
    )


__all__ = [
    "SaeCalibrationResult",
    "SaeCalibrationRunConfig",
    "SaeTrainingResult",
    "SaeTrainingRunConfig",
    "SaeValidationResult",
    "SaeValidationRunConfig",
    "run_calibrate_sae_l1",
    "run_train_saes",
    "run_validate_saes",
    "select_l1_coefficients",
]

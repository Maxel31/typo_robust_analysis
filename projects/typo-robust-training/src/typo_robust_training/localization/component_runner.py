"""Hash-bound orchestration for screen-then-causal component localization."""

from __future__ import annotations

import hashlib
import platform
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.localization.component_causal import (
    ComponentCausalObservation,
    select_training_components,
)
from typo_robust_training.localization.component_config import (
    ComponentLocalizationProtocol,
    load_component_localization_config,
)
from typo_robust_training.localization.component_partition import partition_diagnostic_ids
from typo_robust_training.localization.component_screening import (
    ComponentScreenMetric,
    rank_component_screen,
)
from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.localization.records import LayerScan
from typo_robust_training.localization.runner import (
    _canonical_bytes,
    _load_manifest,
    _sha256_file,
    _write_atomic,
    _write_json,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_object(path: Path) -> Mapping[str, object]:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ComponentLocalizationRunConfig:
    config_path: Path
    diagnostic_manifest_path: Path
    layer_selection_path: Path
    components: tuple[str, ...]
    causal_readouts: tuple[str, ...]
    gpu_id: str
    output_dir: Path
    resume: bool = False


class ComponentLocalizationRuntime(Protocol):
    def screen_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        selected_layers: tuple[int, ...],
    ) -> tuple[ComponentScreenMetric, ...]: ...

    def causal_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        candidates: tuple[ComponentRef, ...],
    ) -> tuple[ComponentCausalObservation, ...]: ...

    def provenance(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class ComponentLocalizationRunResult:
    selected_components: int
    screen_path: Path
    causal_records_path: Path
    selection_path: Path
    run_path: Path


@dataclass(frozen=True, slots=True)
class _LayerEvidence:
    selection_sha256: str
    scans_sha256: str
    selected_layers: tuple[int, ...]
    scans: Mapping[str, LayerScan]


def _load_layer_evidence(
    path: Path,
    *,
    protocol: ComponentLocalizationProtocol,
    manifest_sha256: str,
) -> _LayerEvidence:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"layer selection is not a file: {resolved}")
    selection = _load_object(resolved)
    required = {
        "schema_version",
        "model",
        "model_revision",
        "records",
        "selected_window",
        "diagnostic_manifest_sha256",
        "layer_scans_sha256",
        "runtime",
    }
    if not required <= set(selection):
        raise ValueError("layer selection is missing required evidence fields")
    if selection["schema_version"] != "robustness-layer-selection/v1":
        raise ValueError("layer selection schema differs")
    if (
        selection["model"] != protocol.model
        or selection["model_revision"] != protocol.model_revision
    ):
        raise ValueError("layer selection model identity differs")
    if selection["diagnostic_manifest_sha256"] != manifest_sha256:
        raise ValueError("layer selection was computed from a different diagnostic manifest")
    runtime = selection["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("num_decoder_layers") != protocol.decoder_layers
    ):
        raise ValueError("layer selection decoder architecture differs")
    window = selection["selected_window"]
    if not isinstance(window, Mapping) or set(window) != {"start", "stop"}:
        raise ValueError("layer selection window fields differ")
    start, stop = window["start"], window["stop"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= protocol.decoder_layers
    ):
        raise ValueError("layer selection window is outside the decoder")
    scans_path = resolved.with_name("layer_scans.jsonl")
    if not scans_path.is_file() or _sha256_file(scans_path) != selection["layer_scans_sha256"]:
        raise ValueError("layer scan artifact is missing or fails its committed hash")
    scans: dict[str, LayerScan] = {}
    for line_number, line in enumerate(scans_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = strict_loads(line, context=f"{scans_path}:{line_number}")
        scan = LayerScan.from_dict(payload)
        if scan.record_id in scans:
            raise ValueError("layer scan record IDs are duplicated")
        if scan.num_layers != protocol.decoder_layers:
            raise ValueError("layer scan decoder count differs from component config")
        scans[scan.record_id] = scan
    if len(scans) != selection["records"]:
        raise ValueError("layer scan count differs from layer selection")
    return _LayerEvidence(
        selection_sha256=_sha256_file(resolved),
        scans_sha256=_sha256_file(scans_path),
        selected_layers=tuple(range(start, stop)),
        scans=scans,
    )


def _checkpoint_path(work_dir: Path, phase: str, record_id: str) -> Path:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return work_dir / phase / f"{digest}.json"


def _checkpoint_common(
    *,
    phase: str,
    record: Mapping[str, object],
    protocol: ComponentLocalizationProtocol,
    manifest_sha256: str,
    evidence: _LayerEvidence,
    runtime_sha256: str,
    candidates_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "robustness-component-checkpoint/v1",
        "phase": phase,
        "record_id": record["record_id"],
        "record_sha256": _sha256_value(record),
        "config_sha256": protocol.config_sha256,
        "diagnostic_manifest_sha256": manifest_sha256,
        "layer_selection_sha256": evidence.selection_sha256,
        "layer_scans_sha256": evidence.scans_sha256,
        "runtime_sha256": runtime_sha256,
        "candidates_sha256": candidates_sha256,
    }


def _checkpoint(
    path: Path,
    *,
    phase: str,
    record: Mapping[str, object],
    protocol: ComponentLocalizationProtocol,
    manifest_sha256: str,
    evidence: _LayerEvidence,
    runtime_sha256: str,
    candidates_sha256: str | None,
    rows: list[dict[str, object]],
) -> None:
    _write_json(
        path,
        {
            **_checkpoint_common(
                phase=phase,
                record=record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                evidence=evidence,
                runtime_sha256=runtime_sha256,
                candidates_sha256=candidates_sha256,
            ),
            "rows": rows,
        },
    )


def _load_checkpoint_rows(
    path: Path,
    *,
    phase: str,
    record: Mapping[str, object],
    protocol: ComponentLocalizationProtocol,
    manifest_sha256: str,
    evidence: _LayerEvidence,
    runtime_sha256: str,
    candidates_sha256: str | None,
) -> list[object]:
    payload = _load_object(path)
    expected = _checkpoint_common(
        phase=phase,
        record=record,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
        evidence=evidence,
        runtime_sha256=runtime_sha256,
        candidates_sha256=candidates_sha256,
    )
    if set(payload) != set(expected) | {"rows"} or any(
        payload.get(field) != value for field, value in expected.items()
    ):
        raise ValueError(f"component checkpoint binding differs: {path}")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"component checkpoint rows differ: {path}")
    return rows


def _screen_rows(
    records: tuple[dict[str, object], ...],
    *,
    record_ids: set[str],
    evidence: _LayerEvidence,
    protocol: ComponentLocalizationProtocol,
    runtime: ComponentLocalizationRuntime,
    work_dir: Path,
    resume: bool,
    manifest_sha256: str,
    runtime_sha256: str,
) -> tuple[list[dict[str, object]], tuple[ComponentScreenMetric, ...]]:
    artifact_rows: list[dict[str, object]] = []
    all_metrics: list[ComponentScreenMetric] = []
    valid_by_task: dict[str, int] = defaultdict(int)
    total_by_task: dict[str, int] = defaultdict(int)
    for record in records:
        record_id, task = str(record["record_id"]), str(record["task"])
        if record_id not in record_ids:
            continue
        total_by_task[task] += 1
        checkpoint_path = _checkpoint_path(work_dir, "screen", record_id)
        if resume and checkpoint_path.is_file():
            raw_rows = _load_checkpoint_rows(
                checkpoint_path,
                phase="screen",
                record=record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                evidence=evidence,
                runtime_sha256=runtime_sha256,
                candidates_sha256=None,
            )
            metrics = tuple(ComponentScreenMetric.from_dict(value) for value in raw_rows)
        else:
            metrics = runtime.screen_pair(
                record,
                evidence.scans[record_id],
                evidence.selected_layers,
            )
            if not isinstance(metrics, tuple) or any(
                not isinstance(metric, ComponentScreenMetric) for metric in metrics
            ):
                raise TypeError("component runtime screen_pair must return screen metrics")
            _checkpoint(
                checkpoint_path,
                phase="screen",
                record=record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                evidence=evidence,
                runtime_sha256=runtime_sha256,
                candidates_sha256=None,
                rows=[metric.as_dict() for metric in metrics],
            )
        if metrics:
            if any(metric.task != task or metric.records != 1 for metric in metrics):
                raise ValueError("per-record screen metrics have a different task or weight")
            valid_by_task[task] += 1
        all_metrics.extend(metrics)
        artifact_rows.append(
            {
                "schema_version": "robustness-component-screen-record/v1",
                "record_id": record_id,
                "task": task,
                "available": bool(metrics),
                "metrics": [metric.as_dict() for metric in metrics],
            }
        )
    for task in protocol.tasks:
        if valid_by_task[task] < protocol.minimum_kl_eligible_per_task:
            raise ValueError(f"{task} component screen has too few valid records")
        if (
            valid_by_task[task] / total_by_task[task]
            < protocol.minimum_kl_eligible_fraction_per_task
        ):
            raise ValueError(f"{task} component screen valid fraction is too small")

    aggregates: dict[tuple[ComponentRef, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for metric in all_metrics:
        values = aggregates[(metric.component, metric.task)]
        values[0] += metric.activation_difference * metric.records
        values[1] += metric.gradient_attribution * metric.records
        values[2] += metric.records
    averaged = tuple(
        ComponentScreenMetric(
            component=component,
            task=task,
            records=int(values[2]),
            activation_difference=values[0] / values[2],
            gradient_attribution=values[1] / values[2],
        )
        for (component, task), values in sorted(
            aggregates.items(), key=lambda item: (item[0][0].identifier, item[0][1])
        )
    )
    return artifact_rows, averaged


def _causal_rows(
    records: tuple[dict[str, object], ...],
    *,
    record_ids: set[str],
    candidates: tuple[ComponentRef, ...],
    evidence: _LayerEvidence,
    protocol: ComponentLocalizationProtocol,
    runtime: ComponentLocalizationRuntime,
    work_dir: Path,
    resume: bool,
    manifest_sha256: str,
    runtime_sha256: str,
) -> tuple[list[dict[str, object]], tuple[ComponentCausalObservation, ...]]:
    candidates_sha256 = _sha256_value([component.as_dict() for component in candidates])
    artifact_rows: list[dict[str, object]] = []
    observations: list[ComponentCausalObservation] = []
    for record in records:
        record_id, task = str(record["record_id"]), str(record["task"])
        if record_id not in record_ids:
            continue
        checkpoint_path = _checkpoint_path(work_dir, "causal", record_id)
        if resume and checkpoint_path.is_file():
            raw_rows = _load_checkpoint_rows(
                checkpoint_path,
                phase="causal",
                record=record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                evidence=evidence,
                runtime_sha256=runtime_sha256,
                candidates_sha256=candidates_sha256,
            )
            pair_rows = tuple(ComponentCausalObservation.from_dict(value) for value in raw_rows)
        else:
            pair_rows = runtime.causal_pair(
                record,
                evidence.scans[record_id],
                candidates,
            )
            if not isinstance(pair_rows, tuple) or any(
                not isinstance(row, ComponentCausalObservation) for row in pair_rows
            ):
                raise TypeError("component runtime causal_pair must return causal observations")
            _checkpoint(
                checkpoint_path,
                phase="causal",
                record=record,
                protocol=protocol,
                manifest_sha256=manifest_sha256,
                evidence=evidence,
                runtime_sha256=runtime_sha256,
                candidates_sha256=candidates_sha256,
                rows=[row.as_dict() for row in pair_rows],
            )
        if {row.component for row in pair_rows} != set(candidates):
            raise ValueError("causal runtime did not return every shortlisted component")
        if any(row.record_id != record_id or row.task != task for row in pair_rows):
            raise ValueError("causal runtime returned a different record identity")
        observations.extend(pair_rows)
        artifact_rows.append(
            {
                "schema_version": "robustness-component-causal-record/v1",
                "record_id": record_id,
                "task": task,
                "observations": [row.as_dict() for row in pair_rows],
            }
        )
    return artifact_rows, tuple(observations)


def run_localize_robustness_components(
    config: ComponentLocalizationRunConfig,
    *,
    runtime: ComponentLocalizationRuntime | None = None,
) -> ComponentLocalizationRunResult:
    """Screen on one diagnostic half and causally validate on the other."""

    if not isinstance(config, ComponentLocalizationRunConfig):
        raise TypeError("config must be ComponentLocalizationRunConfig")
    protocol = load_component_localization_config(config.config_path)
    if config.components != ("mlp-neuron", "attention-head"):
        raise ValueError("--components must be mlp-neuron attention-head in frozen order")
    if config.causal_readouts != protocol.readouts:
        raise ValueError("--causal-readouts must match the frozen config order")
    if not config.gpu_id or "," in config.gpu_id:
        raise ValueError("--gpu-id must name one physical GPU")
    records, manifest_sha256 = _load_manifest(
        config.diagnostic_manifest_path,
        tasks=protocol.tasks,
    )
    evidence = _load_layer_evidence(
        config.layer_selection_path,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
    )
    record_ids = {str(record["record_id"]) for record in records}
    if set(evidence.scans) != record_ids:
        raise ValueError("layer scans and diagnostic manifest contain different record IDs")
    partition = partition_diagnostic_ids(records, protocol=protocol)

    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume:
        raise FileExistsError(f"component localization output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        from typo_robust_training.localization.component_runtime import (
            HuggingFaceComponentLocalizationRuntime,
        )

        runtime = HuggingFaceComponentLocalizationRuntime(protocol=protocol, gpu_id=config.gpu_id)
    provenance = dict(runtime.provenance())
    runtime_sha256 = _sha256_value(provenance)
    run_path = output_dir / "run.json"
    run_base = {
        "schema_version": "robustness-component-localization-run/v1",
        "operation": "localize-robustness-components",
        "config_sha256": protocol.config_sha256,
        "diagnostic_manifest_sha256": manifest_sha256,
        "layer_selection_sha256": evidence.selection_sha256,
        "layer_scans_sha256": evidence.scans_sha256,
        "selected_layers": list(evidence.selected_layers),
        "gpu_id": config.gpu_id,
        "resume": config.resume,
        "python": platform.python_version(),
        "runtime": provenance,
    }
    _write_json(run_path, {**run_base, "status": "running", "started_at": _now()})
    work_dir = output_dir / ".localize-robustness-components-work"
    work_dir.mkdir(exist_ok=True)
    try:
        screen_record_rows, screen_metrics = _screen_rows(
            records,
            record_ids=set(partition.screening),
            evidence=evidence,
            protocol=protocol,
            runtime=runtime,
            work_dir=work_dir,
            resume=config.resume,
            manifest_sha256=manifest_sha256,
            runtime_sha256=runtime_sha256,
        )
        screen = rank_component_screen(
            screen_metrics,
            selected_layers=evidence.selected_layers,
            protocol=protocol,
        )
        candidates = tuple(candidate.component for candidate in screen.causal_candidates)
        screen_records_path = output_dir / "component_screen_records.jsonl"
        _write_atomic(
            screen_records_path,
            b"".join(_canonical_bytes(row) for row in screen_record_rows),
        )
        screen_path = output_dir / "component_screen.json"
        _write_json(
            screen_path,
            {
                "schema_version": "robustness-component-screen/v1",
                "selected_layers": list(evidence.selected_layers),
                "screening_ids": list(partition.screening),
                "screening_records_sha256": _sha256_file(screen_records_path),
                "universe": [candidate.as_dict() for candidate in screen.universe],
                "causal_candidates": [
                    candidate.as_dict() for candidate in screen.causal_candidates
                ],
            },
        )
        causal_record_rows, observations = _causal_rows(
            records,
            record_ids=set(partition.causal_validation),
            candidates=candidates,
            evidence=evidence,
            protocol=protocol,
            runtime=runtime,
            work_dir=work_dir,
            resume=config.resume,
            manifest_sha256=manifest_sha256,
            runtime_sha256=runtime_sha256,
        )
        causal_records_path = output_dir / "component_causal_records.jsonl"
        _write_atomic(
            causal_records_path,
            b"".join(_canonical_bytes(row) for row in causal_record_rows),
        )
        selection = select_training_components(
            observations,
            candidates=candidates,
            protocol=protocol,
        )
        selection_path = output_dir / "component_selection.json"
        partition_payload = {
            "algorithm": protocol.partition_algorithm,
            "seed": protocol.partition_seed,
            "screening_ids": list(partition.screening),
            "causal_validation_ids": list(partition.causal_validation),
        }
        _write_json(
            selection_path,
            {
                **selection.as_dict(),
                "operation": "localize-robustness-components",
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "config_sha256": protocol.config_sha256,
                "diagnostic_manifest_sha256": manifest_sha256,
                "layer_selection_sha256": evidence.selection_sha256,
                "layer_scans_sha256": evidence.scans_sha256,
                "component_screen_sha256": _sha256_file(screen_path),
                "component_causal_records_sha256": _sha256_file(causal_records_path),
                "diagnostic_partition": partition_payload,
                "selected": [
                    {**item.as_dict(), "causally_validated": True} for item in selection.selected
                ],
            },
        )
        outputs = {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                screen_records_path,
                screen_path,
                causal_records_path,
                selection_path,
            )
        }
        _write_json(
            run_path,
            {
                **run_base,
                "status": "completed",
                "completed_at": _now(),
                "selected_components": len(selection.selected),
                "outputs": outputs,
            },
        )
    except BaseException as exc:
        _write_json(
            run_path,
            {
                **run_base,
                "status": "failed",
                "failed_at": _now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    return ComponentLocalizationRunResult(
        selected_components=len(selection.selected),
        screen_path=screen_path,
        causal_records_path=causal_records_path,
        selection_path=selection_path,
        run_path=run_path,
    )


__all__ = [
    "ComponentLocalizationRunConfig",
    "ComponentLocalizationRunResult",
    "ComponentLocalizationRuntime",
    "run_localize_robustness_components",
]

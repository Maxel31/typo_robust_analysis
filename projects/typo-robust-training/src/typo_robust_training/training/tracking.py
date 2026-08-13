"""Scalar-only, hash-bound Weights & Biases training telemetry."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


_SCHEMA = "robustness-wandb-training-run/v1"
_METADATA_FIELDS = {
    "schema_version",
    "provider",
    "project",
    "entity",
    "run_id",
    "url",
    "bindings",
    "presentation",
    "last_logged_optimizer_step",
    "status",
}
_FORBIDDEN_METRIC_FRAGMENTS = ("record_id", "text", "prompt", "api_key", "secret")


@dataclass(frozen=True, slots=True)
class WandbRunPresentation:
    """Human-readable W&B identity kept outside the scientific bindings."""

    name: str
    group: str
    job_type: str
    tags: tuple[str, ...]
    notes: str


_CONFIRMATORY_PRESENTATION = {
    "output-matching": (
        "Kojima baseline",
        "Output-distribution matching",
        "kojima-output-distribution-matching",
        "baseline",
        "Kojima-style clean-teacher/noisy-student output distribution matching; "
        "no state-alignment loss.",
    ),
    "localized-state-distillation": (
        "Proposed method",
        "Causal-window localized state distillation",
        "proposed-causal-window-state-distillation",
        "proposed-method",
        "Output matching plus residual-state cosine alignment at the edited-word-final "
        "coordinates selected by Activation Patching.",
    ),
    "random-window-state-distillation": (
        "Specificity control",
        "Random-window state distillation",
        "random-window-state-control",
        "specificity-control",
        "Same-width non-overlapping random-window control for the causal layer selection.",
    ),
    "global-state-alignment": (
        "Scope control",
        "All-layer state distillation",
        "all-layer-state-control",
        "scope-control",
        "Residual-state alignment at every decoder layer.",
    ),
    "noisy-language-model": (
        "Auxiliary baseline",
        "Noisy-language-model training",
        "noisy-language-model-baseline",
        "auxiliary-baseline",
        "Ordinary causal-language-model training on noisy text.",
    ),
}
_HISTORICAL_PRESENTATION = {
    "noisy-language-model": (
        "Historical baseline",
        "Noisy-language-model training",
        "historical-noisy-language-model",
        "historical-baseline",
        "Historical Cycle 1 noisy-language-model baseline.",
    ),
    "output-matching": (
        "Historical pilot",
        "Output/answer/clean-loss training",
        "historical-output-matching-pilot",
        "historical-pilot",
        "Historical Cycle 1 output-matching pilot with answer CE and a separate clean-KL loss.",
    ),
    "global-state-alignment": (
        "Historical control",
        "Global relative-MSE state alignment",
        "historical-global-state-control",
        "historical-control",
        "Historical Cycle 1 all-layer/all-token relative-MSE state alignment.",
    ),
    "localized-state-distillation": (
        "Historical ablation",
        "Component-level relative-MSE state distillation",
        "historical-component-state-ablation",
        "historical-ablation",
        "Historical Cycle 1 neuron/head component experiment; this is not the confirmatory method.",
    ),
}


def _display_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("W&B presentation model must be non-empty")
    names = {
        "gemma": "Gemma",
        "llama": "Llama",
        "mistral": "Mistral",
        "qwen": "Qwen",
        "it": "IT",
        "instruct": "Instruct",
    }
    words = model.rsplit("/", 1)[-1].split("-")
    return "-".join(
        names.get(word.lower(), word[:-1] + "B")
        if re.fullmatch(r"\d+(?:\.\d+)?b", word.lower())
        else names.get(word.lower(), word)
        for word in words
    )


def _layer_label(layers: tuple[int, ...]) -> str | None:
    if not layers:
        return None
    if (
        any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers)
        or tuple(sorted(set(layers))) != layers
    ):
        raise ValueError("W&B presentation state layers must be unique sorted integers")
    if layers == tuple(range(layers[0], layers[-1] + 1)):
        return f"L{layers[0]}–{layers[-1]}"
    return "L{" + ",".join(map(str, layers)) + "}"


def build_wandb_run_presentation(
    *,
    condition: str,
    schema_version: str,
    model: str,
    seed: int,
    max_optimizer_steps: int,
    state_layers: tuple[int, ...],
) -> WandbRunPresentation:
    """Name a W&B series by its scientific role without opaque arm abbreviations."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("W&B presentation seed must be a non-negative integer")
    if (
        isinstance(max_optimizer_steps, bool)
        or not isinstance(max_optimizer_steps, int)
        or max_optimizer_steps <= 0
    ):
        raise ValueError("W&B presentation optimizer steps must be positive")
    cycle_match = re.fullmatch(r"robustness-adapter-training-config/v(\d+)", schema_version)
    if cycle_match is None:
        raise ValueError("W&B presentation training schema is invalid")
    cycle = int(cycle_match.group(1))
    presentation_map = _HISTORICAL_PRESENTATION if cycle == 1 else _CONFIRMATORY_PRESENTATION
    if condition not in presentation_map:
        raise ValueError(f"W&B presentation has no mapping for {condition!r}")
    role_label, operation, job_type, role_tag, notes = presentation_map[condition]
    model_name = _display_model_name(model)
    layer_label = _layer_label(state_layers)
    parts = [role_label, operation]
    if layer_label is not None:
        parts.append(layer_label)
        notes = f"{notes} State layers: {layer_label}."
    parts.extend((model_name, f"{max_optimizer_steps} steps", f"seed {seed}"))
    group_prefix = "Historical Cycle 1" if cycle == 1 else "Confirmatory comparison"
    return WandbRunPresentation(
        name=" · ".join(parts),
        group=f"{group_prefix} · {model_name} · {max_optimizer_steps} steps",
        job_type=job_type,
        tags=(
            "typo-robustness",
            f"protocol-version:{cycle}",
            f"role:{role_tag}",
            f"condition:{condition}",
            f"model:{model.rsplit('/', 1)[-1].lower()}",
            f"budget:{max_optimizer_steps}-steps",
        ),
        notes=notes,
    )


def _presentation_payload(presentation: WandbRunPresentation) -> dict[str, object]:
    fields = {
        "name": presentation.name,
        "group": presentation.group,
        "job_type": presentation.job_type,
        "tags": list(presentation.tags),
        "notes": presentation.notes,
    }
    if any(
        not isinstance(fields[name], str) or not str(fields[name]).strip()
        for name in ("name", "group", "job_type", "notes")
    ):
        raise ValueError("W&B presentation text fields must be non-empty")
    if not presentation.tags or any(
        not isinstance(tag, str) or not tag.strip() for tag in presentation.tags
    ):
        raise ValueError("W&B presentation tags must be non-empty strings")
    if len(set(presentation.tags)) != len(presentation.tags):
        raise ValueError("W&B presentation tags must be unique")
    return fields


class TrainingTracker(Protocol):
    """The runner-facing boundary for one external scalar tracker."""

    def log_optimizer_step(
        self,
        metrics: Mapping[str, int | float],
        *,
        optimizer_step: int,
    ) -> None: ...

    def finish(self, *, status: str, summary: Mapping[str, int | float]) -> None: ...

    def provenance(self) -> Mapping[str, object]: ...


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bindings(bindings: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("W&B bindings must be a non-empty object")
    try:
        encoded = json.dumps(dict(bindings), sort_keys=True, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("W&B bindings must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError("W&B bindings must be an object")
    if any(
        fragment in str(key).lower()
        for key in normalized
        for fragment in ("api_key", "credential", "secret", "token")
    ):
        raise ValueError("W&B bindings must not contain credentials")
    return normalized


def _scalar_metrics(values: Mapping[str, int | float]) -> dict[str, int | float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("W&B metrics must be a non-empty object")
    normalized: dict[str, int | float] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not name
            or any(fragment in name.lower() for fragment in _FORBIDDEN_METRIC_FRAGMENTS)
        ):
            raise ValueError("W&B metric name is empty or may expose raw training data")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("W&B metrics must contain only numeric scalars")
        if not math.isfinite(float(value)):
            raise ValueError("W&B metrics must be finite")
        normalized[name] = value
    return normalized


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError("W&B resume requires wandb_run.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("W&B run metadata is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _METADATA_FIELDS:
        raise ValueError("W&B run metadata fields differ")
    if payload.get("schema_version") != _SCHEMA or payload.get("provider") != "wandb":
        raise ValueError("W&B run metadata schema differs")
    return payload


class WandbTrainingTracker:
    """One online W&B run whose local identity is safe to resume exactly."""

    def __init__(
        self,
        *,
        run: Any,
        metadata_path: Path,
        metadata: Mapping[str, object],
        sdk_version: str | None,
    ) -> None:
        self._run = run
        self._metadata_path = metadata_path
        self._metadata = dict(metadata)
        self._sdk_version = sdk_version
        self._finished = False

    def log_optimizer_step(
        self,
        metrics: Mapping[str, int | float],
        *,
        optimizer_step: int,
    ) -> None:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step <= 0
        ):
            raise ValueError("W&B optimizer step must be positive")
        prior = self._metadata["last_logged_optimizer_step"]
        if not isinstance(prior, int) or optimizer_step != prior + 1:
            raise ValueError("W&B optimizer steps must be consecutive")
        payload = _scalar_metrics(metrics)
        if payload.get("train/optimizer_step") != optimizer_step:
            raise ValueError("W&B metrics must contain the matching optimizer step")
        self._run.log(payload, step=optimizer_step)
        self._metadata.update({"last_logged_optimizer_step": optimizer_step, "status": "running"})
        _write_json(self._metadata_path, self._metadata)

    def finish(self, *, status: str, summary: Mapping[str, int | float]) -> None:
        if self._finished:
            return
        if status not in {"completed", "failed"}:
            raise ValueError("W&B finish status must be completed or failed")
        values = _scalar_metrics(summary)
        self._run.summary.update({f"summary/{name}": value for name, value in values.items()})
        self._run.summary["run/status"] = status
        self._run.finish(exit_code=0 if status == "completed" else 1)
        self._metadata["status"] = status
        _write_json(self._metadata_path, self._metadata)
        self._finished = True

    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "wandb",
            "sdk_version": self._sdk_version,
            "project": self._metadata["project"],
            "entity": self._metadata["entity"],
            "run_id": self._metadata["run_id"],
            "url": self._metadata["url"],
            "metadata_path": str(self._metadata_path),
            "raw_training_data_uploaded": False,
        }


def start_wandb_training_tracker(
    *,
    output_dir: Path,
    project: str,
    entity: str | None,
    bindings: Mapping[str, object],
    presentation: WandbRunPresentation | None = None,
    resume: bool,
    resume_optimizer_step: int,
    environment: Mapping[str, str] | None = None,
    wandb_module: Any | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> WandbTrainingTracker:
    """Start or rewind one W&B run without copying its credential into artifacts."""

    if not isinstance(project, str) or not project.strip():
        raise ValueError("W&B project must be non-empty")
    if entity is not None and (not isinstance(entity, str) or not entity.strip()):
        raise ValueError("W&B entity must be null or non-empty")
    if (
        isinstance(resume_optimizer_step, bool)
        or not isinstance(resume_optimizer_step, int)
        or resume_optimizer_step < 0
    ):
        raise ValueError("W&B resume optimizer step must be non-negative")
    values = os.environ if environment is None else environment
    api_key = values.get("WANDB_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("W&B online tracking requires WANDB_API_KEY")
    resolved_entity = entity or values.get("WANDB_ENTITY")
    if resolved_entity is not None and not resolved_entity:
        resolved_entity = None
    frozen_bindings = _canonical_bindings(bindings)
    condition = str(frozen_bindings.get("condition", "adapter-training"))
    seed = frozen_bindings.get("seed")
    if presentation is None:
        presentation = WandbRunPresentation(
            name=f"{condition} · seed {seed}",
            group=condition,
            job_type="adapter-training",
            tags=("typo-robustness", f"condition:{condition}"),
            notes="Compatibility presentation without an explicit scientific role.",
        )
    elif not isinstance(presentation, WandbRunPresentation):
        raise TypeError("W&B presentation must be WandbRunPresentation")
    presentation_payload = _presentation_payload(presentation)
    root = Path(output_dir).resolve()
    metadata_path = root / "wandb_run.json"
    factory = run_id_factory or (lambda: secrets.token_hex(8))

    if resume:
        metadata = _load_metadata(metadata_path)
        if (
            metadata["project"] != project
            or metadata["entity"] != resolved_entity
            or metadata["bindings"] != frozen_bindings
        ):
            raise ValueError("W&B resume bindings differ")
        if metadata["presentation"] != presentation_payload:
            raise ValueError("W&B resume presentation differs")
        prior_step = metadata["last_logged_optimizer_step"]
        if not isinstance(prior_step, int) or prior_step < resume_optimizer_step:
            raise ValueError("W&B history is behind the local checkpoint")
        run_id = metadata["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("W&B run ID is invalid")
        init_resume = {"resume_from": f"{run_id}?_step={resume_optimizer_step}"}
    else:
        if metadata_path.exists():
            raise FileExistsError("fresh W&B run cannot overwrite wandb_run.json")
        run_id = factory()
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("W&B run ID factory returned an invalid ID")
        metadata = {
            "schema_version": _SCHEMA,
            "provider": "wandb",
            "project": project,
            "entity": resolved_entity,
            "run_id": run_id,
            "url": None,
            "bindings": frozen_bindings,
            "presentation": presentation_payload,
            "last_logged_optimizer_step": 0,
            "status": "initializing",
        }
        init_resume = {"id": run_id, "resume": "never"}

    module = wandb_module or importlib.import_module("wandb")
    local_dir = root / ".wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    run = module.init(
        project=project,
        entity=resolved_entity,
        name=presentation.name,
        group=presentation.group,
        job_type=presentation.job_type,
        tags=list(presentation.tags),
        notes=presentation.notes,
        config=frozen_bindings,
        dir=str(local_dir),
        mode="online",
        reinit="create_new",
        **init_resume,
    )
    if run is None or getattr(run, "id", None) != run_id:
        raise RuntimeError("W&B initialized a different or missing run ID")
    run.define_metric("train/optimizer_step")
    run.define_metric("train/*", step_metric="train/optimizer_step")
    run.define_metric("system/*", step_metric="train/optimizer_step")
    metadata.update(
        {
            "url": getattr(run, "url", None),
            "last_logged_optimizer_step": resume_optimizer_step,
            "status": "running",
        }
    )
    _write_json(metadata_path, metadata)
    sdk_version = getattr(module, "__version__", None)
    return WandbTrainingTracker(
        run=run,
        metadata_path=metadata_path,
        metadata=metadata,
        sdk_version=str(sdk_version) if sdk_version is not None else None,
    )


__all__ = [
    "TrainingTracker",
    "WandbTrainingTracker",
    "WandbRunPresentation",
    "build_wandb_run_presentation",
    "start_wandb_training_tracker",
]

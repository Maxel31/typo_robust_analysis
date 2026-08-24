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

from typo_robust_training.training.json_io import write_json_atomic, write_json_durable


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
    """Human-readable W&B identity kept separate from scientific bindings."""

    name: str
    group: str
    job_type: str
    tags: tuple[str, ...]
    notes: str


_ARM_PRESENTATION = {
    "output-matching": (
        "Kojima baseline",
        "Output-distribution matching",
        "kojima-output-matching",
        "baseline-output-matching",
        "baseline",
        "Kojima-style clean-teacher/noisy-student output distribution matching; "
        "no state-alignment loss.",
    ),
    "localized-state-distillation": (
        "Proposed method",
        "Causal-window localized state distillation",
        "causal-window-localized-state-distillation",
        "proposed-causal-window",
        "proposed",
        "Proposed arm: output matching plus residual-state cosine alignment at "
        "Activation Patching-selected edited-word-final coordinates.",
    ),
    "random-window-state-distillation": (
        "Random-window control",
        "Random-window localized state distillation",
        "random-window-state-control",
        "control-random-window",
        "control",
        "Specificity control: output matching plus residual-state alignment at a "
        "deterministic, same-width non-overlapping random window.",
    ),
    "global-state-alignment": (
        "All-layer control",
        "All-layer state distillation",
        "all-layer-state-control",
        "control-all-layers",
        "control",
        "Scope control: output matching plus residual-state alignment at every decoder layer.",
    ),
    "probe-transition-output-matching": (
        "Probe-transition baseline",
        "Probe-transition suffix output matching",
        "probe-transition-output-matching",
        "baseline-probe-transition-output",
        "baseline",
        "Matched-budget output-distribution baseline with LoRA restricted to the "
        "validated linear-probe transition suffix; no state-alignment loss.",
    ),
    "probe-transition-single-layer-state-distillation": (
        "Probe-transition state proposal",
        "Single-layer transition-state distillation",
        "probe-transition-single-layer-state-distillation",
        "proposed-probe-transition-state",
        "proposed",
        "Proposed arm: suffix LoRA output matching plus cosine residual-state alignment "
        "at the validated linear-probe transition layer and edited-word-final token.",
    ),
    "probe-semantic-subspace-distillation": (
        "Probe-subspace proposal",
        "Rank-16 semantic-subspace distillation",
        "probe-semantic-subspace-distillation",
        "proposed-probe-semantic-subspace",
        "proposed",
        "Proposed arm: output matching plus frozen-classifier forward-KL alignment "
        "in the validated rank-16 probe semantic subspace at the edited-word-final "
        "probe-transition coordinate.",
    ),
    "factorial-all-layers-all-tokens": (
        "All-layer/all-token baseline",
        "Full aligned output matching",
        "factorial-all-layers-all-tokens",
        "baseline-output-factorial",
        "baseline",
        "Kojima-style all-layer LoRA with output matching at every aligned non-edited token.",
    ),
    "factorial-all-layers-downstream-horizon": (
        "All-layer horizon control",
        "AP-horizon output matching",
        "factorial-all-layers-horizon",
        "control-output-horizon",
        "control",
        "All-layer LoRA with noisy supervision restricted to downstream offsets +2..+16.",
    ),
    "factorial-probe-suffix-all-tokens": (
        "Probe-suffix placement control",
        "Full aligned output matching",
        "factorial-probe-suffix-all-tokens",
        "control-probe-placement",
        "control",
        "Linear-probe suffix LoRA with full aligned output matching.",
    ),
    "factorial-probe-suffix-downstream-horizon": (
        "Probe-interface proposal",
        "Probe-suffix AP-horizon output matching",
        "factorial-probe-suffix-horizon",
        "proposed-probe-interface",
        "proposed",
        "Linear-probe suffix LoRA with noisy output matching only at AP-derived offsets +2..+16.",
    ),
    "factorial-random-layers-downstream-horizon": (
        "Random-freeze placement control",
        "Count-matched random-layer AP-horizon output matching",
        "factorial-random-layers-horizon",
        "control-random-freeze",
        "control",
        "Deterministic count-matched random LoRA layers with identical horizon supervision.",
    ),
}

_LEGACY_ARM_PRESENTATION = {
    "noisy-language-model": (
        "Legacy",
        "Noisy language-model training",
        "Legacy",
        "legacy-noisy-language-model",
        "legacy-baseline",
        "Cycle-1 noisy-language-model baseline with hard training targets.",
    ),
    "output-matching": (
        "Legacy",
        "Output/answer/clean-loss pilot",
        "Legacy",
        "legacy-output-matching-pilot",
        "legacy-pilot",
        "Cycle-1 output-matching pilot with answer CE and a separate clean-KL loss; "
        "this is not the cycle-2 Kojima-style output-matching baseline.",
    ),
    "global-state-alignment": (
        "Legacy",
        "Global relative-MSE state pilot",
        "Legacy",
        "legacy-global-state-pilot",
        "legacy-pilot",
        "Cycle-1 all-layer/all-token relative-MSE state-alignment pilot.",
    ),
    "localized-state-distillation": (
        "Legacy",
        "Component-level state distillation",
        "Legacy",
        "legacy-component-state-pilot",
        "legacy-pilot",
        "Cycle-1 selected-neuron/head relative-MSE state-alignment pilot; this is "
        "not the cycle-2 causal-window localized state-distillation method.",
    ),
}


def _display_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("W&B presentation model must be non-empty")
    words = model.rsplit("/", 1)[-1].split("-")
    names = {
        "gemma": "Gemma",
        "llama": "Llama",
        "mistral": "Mistral",
        "qwen": "Qwen",
        "it": "IT",
        "instruct": "Instruct",
    }
    return "-".join(
        names.get(word.lower(), word[:-1] + "B")
        if re.fullmatch(r"\d+(?:\.\d+)?b", word.lower())
        else names.get(word.lower(), word)
        for word in words
    )


def _layer_label(layers: tuple[int, ...]) -> str | None:
    if not layers:
        return None
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("W&B presentation state layers must be non-negative integers")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError("W&B presentation state layers must be unique and sorted")
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
    state_gradient_ratio: float | None,
    state_layers: tuple[int, ...],
    max_student_tokens: int | None = None,
) -> WandbRunPresentation:
    """Name one comparison series by scientific role, operation, and budget."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("W&B presentation seed must be a non-negative integer")
    if (
        isinstance(max_optimizer_steps, bool)
        or not isinstance(max_optimizer_steps, int)
        or max_optimizer_steps <= 0
    ):
        raise ValueError("W&B presentation optimizer steps must be positive")
    if max_student_tokens is not None and (
        isinstance(max_student_tokens, bool)
        or not isinstance(max_student_tokens, int)
        or max_student_tokens <= 0
    ):
        raise ValueError("W&B presentation student tokens must be positive")
    if state_gradient_ratio is not None and (
        isinstance(state_gradient_ratio, bool)
        or not isinstance(state_gradient_ratio, (int, float))
        or not math.isfinite(float(state_gradient_ratio))
        or float(state_gradient_ratio) <= 0.0
    ):
        raise ValueError("W&B presentation state-gradient ratio must be positive")
    cycle_match = re.fullmatch(r"robustness-adapter-training-config/v(\d+)", schema_version)
    if cycle_match is None:
        raise ValueError("W&B presentation training schema is invalid")
    cycle = int(cycle_match.group(1))
    presentation_map = _LEGACY_ARM_PRESENTATION if cycle == 1 else _ARM_PRESENTATION
    if condition not in presentation_map:
        raise ValueError(f"W&B presentation has no cycle-{cycle} arm mapping for {condition!r}")
    arm, operation, arm_slug, job_type, role, notes = presentation_map[condition]
    model_name = _display_model_name(model)
    layer_label = _layer_label(state_layers)
    parts = [arm, operation]
    if layer_label is not None:
        parts.append(layer_label)
    if max_student_tokens is None:
        budget_label = f"{max_optimizer_steps} steps"
        budget_tag = f"budget:{max_optimizer_steps}-steps"
    else:
        budget_label = (
            f"{max_student_tokens // 1_000_000}M tokens"
            if max_student_tokens % 1_000_000 == 0
            else f"{max_student_tokens} tokens"
        )
        budget_tag = f"budget:{max_student_tokens}-student-tokens"
    parts.extend((model_name, budget_label, f"seed {seed}"))
    tags = [
        "typo-robustness",
        f"cycle:{cycle}",
        f"arm:{arm_slug}",
        f"role:{role}",
        f"model:{model.rsplit('/', 1)[-1].lower()}",
        budget_tag,
    ]
    if cycle == 1:
        # Preserve the historical condition identity even though legacy runs
        # intentionally share one broad presentation arm.
        tags.append(f"condition:{condition}")
    if state_gradient_ratio is not None:
        tags.append(f"state-gradient-ratio:{float(state_gradient_ratio):g}")
    if layer_label is not None:
        notes = f"{notes} State layers: {layer_label}."
    return WandbRunPresentation(
        name=" · ".join(parts),
        group=f"Cycle {cycle} · {model_name} · {budget_label}",
        job_type=job_type,
        tags=tuple(tags),
        notes=notes,
    )


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
        replay_optimizer_step: int,
    ) -> None:
        self._run = run
        self._metadata_path = metadata_path
        self._metadata = dict(metadata)
        self._sdk_version = sdk_version
        self._replay_optimizer_step = replay_optimizer_step
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
        payload = _scalar_metrics(metrics)
        if payload.get("train/optimizer_step") != optimizer_step:
            raise ValueError("W&B metrics must contain the matching optimizer step")
        prior = self._metadata["last_logged_optimizer_step"]
        if not isinstance(prior, int) or optimizer_step != self._replay_optimizer_step + 1:
            raise ValueError("W&B optimizer steps must be consecutive")
        self._replay_optimizer_step = optimizer_step
        if optimizer_step <= prior:
            return
        if optimizer_step != prior + 1:
            raise ValueError("W&B optimizer steps must catch up before appending")
        self._run.log(payload, step=optimizer_step)
        self._metadata.update({"last_logged_optimizer_step": optimizer_step, "status": "running"})
        write_json_atomic(self._metadata_path, self._metadata)

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
        write_json_atomic(self._metadata_path, self._metadata)
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
            name=f"{condition}-seed-{seed}",
            group=condition,
            job_type="adapter-training",
            tags=("typo-robustness", condition),
            notes="Legacy W&B presentation without an explicit scientific arm label.",
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
        status = metadata["status"]
        if status not in {"initializing", "running", "failed", "completed"}:
            raise ValueError("W&B resume status is not recoverable")
        run_id = metadata["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("W&B run ID is invalid")
        intent_only = (
            status == "initializing"
            and prior_step == 0
            and metadata.get("url") is None
        )
        init_resume = {"id": run_id, "resume": "allow" if intent_only else "must"}
        last_logged_optimizer_step = prior_step
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
        last_logged_optimizer_step = 0

    module = wandb_module or importlib.import_module("wandb")
    local_dir = root / ".wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        # Persist the remote identity before wandb.init.  A process death after
        # the remote run is created can therefore resume the same ID with
        # ``allow`` instead of orphaning it and minting a second run.
        write_json_durable(metadata_path, metadata)
    run: Any | None = None
    valid_run = False
    try:
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
        valid_run = run is not None and getattr(run, "id", None) == run_id
        if not valid_run:
            raise RuntimeError("W&B initialized a different or missing run ID")
        metadata.update(
            {
                "url": getattr(run, "url", None),
                "last_logged_optimizer_step": last_logged_optimizer_step,
                "status": "initializing",
            }
        )
        # Persist the remote identity before fallible metric registration so
        # partial initialization remains attributable and resumable.
        write_json_atomic(metadata_path, metadata)
        run.define_metric("train/optimizer_step")
        run.define_metric("train/*", step_metric="train/optimizer_step")
        run.define_metric("system/*", step_metric="train/optimizer_step")
        metadata["status"] = "running"
        write_json_atomic(metadata_path, metadata)
    except Exception:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        if valid_run:
            metadata["status"] = "failed"
            try:
                write_json_atomic(metadata_path, metadata)
            except Exception:
                pass
        raise
    sdk_version = getattr(module, "__version__", None)
    return WandbTrainingTracker(
        run=run,
        metadata_path=metadata_path,
        metadata=metadata,
        sdk_version=str(sdk_version) if sdk_version is not None else None,
        replay_optimizer_step=resume_optimizer_step,
    )


__all__ = [
    "TrainingTracker",
    "WandbTrainingTracker",
    "WandbRunPresentation",
    "build_wandb_run_presentation",
    "start_wandb_training_tracker",
]

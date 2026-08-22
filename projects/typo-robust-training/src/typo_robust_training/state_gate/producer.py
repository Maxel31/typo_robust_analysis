"""Produce one immutable transition-layer causal-gate bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from typo_robust_training.probe import load_probe_transition_artifact
from typo_robust_training.state_gate.artifacts import (
    SingleLayerGateArtifact,
    SingleLayerGateRecord,
    deterministic_cross_item_donor_plan,
    load_gate_cohort_manifest,
    load_single_layer_gate_artifact,
)
from typo_robust_training.state_gate.config import load_single_layer_gate_config


class SingleLayerGateProvider(Protocol):
    model_id: str
    model_revision: str
    teacher_revision: str
    student_revision: str
    tokenizer_revision: str
    code_revision: str
    source_tree_sha256: str
    decoder_layers: int
    base_model_frozen: bool

    def token_inflation_bucket(self, record: SingleLayerGateRecord) -> str: ...

    def scan(
        self,
        records: Sequence[SingleLayerGateRecord],
        *,
        donor_plan: Mapping[str, str],
        transition_layer: int,
    ) -> Sequence[Mapping[str, object]]: ...

    def provenance(self) -> Mapping[str, object]: ...


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"gate producer input must be one regular file: {source}")
    shutil.copyfile(source, destination)


def _copy_parent_bundle(source_artifact: Path, destination: Path) -> Path:
    root = source_artifact.parent.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("parent probe bundle must not contain symlinks")
    shutil.copytree(root, destination)
    return destination / source_artifact.name


def produce_single_layer_gate_artifact(
    *,
    config_path: Path,
    parent_probe_artifact_path: Path,
    cohort_manifest_path: Path,
    protected_registry_path: Path,
    donor_plan_path: Path,
    runtime_manifest_path: Path,
    output_dir: Path,
    provider: SingleLayerGateProvider,
) -> SingleLayerGateArtifact:
    """Execute the preregistered gate and atomically publish a verified bundle."""

    paths = tuple(
        Path(path)
        for path in (
            config_path,
            parent_probe_artifact_path,
            cohort_manifest_path,
            protected_registry_path,
            donor_plan_path,
            runtime_manifest_path,
        )
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("single-layer gate inputs must be regular files, not symlinks")
    protocol = load_single_layer_gate_config(Path(config_path))
    parent = load_probe_transition_artifact(Path(parent_probe_artifact_path))
    if (
        (parent.model, parent.model_revision, parent.decoder_layers)
        != (protocol.model, protocol.model_revision, protocol.decoder_layers)
        or parent.artifact_sha256 != protocol.input_sha256["parent_probe_artifact"]
    ):
        raise ValueError("single-layer gate parent probe differs from preregistration")
    direct = {
        "cohort_manifest": Path(cohort_manifest_path),
        "protected_registry": Path(protected_registry_path),
        "donor_plan": Path(donor_plan_path),
        "runtime_manifest": Path(runtime_manifest_path),
    }
    if any(_digest(path) != protocol.input_sha256[name] for name, path in direct.items()):
        raise ValueError("single-layer gate input hash differs from preregistration")
    records = load_gate_cohort_manifest(Path(cohort_manifest_path), protocol=protocol)
    expected_plan = deterministic_cross_item_donor_plan(records)
    plan_value = json.loads(Path(donor_plan_path).read_text(encoding="utf-8"))
    plan_rows = plan_value.get("records") if isinstance(plan_value, dict) else None
    observed_plan = (
        {str(row["pair_id"]): str(row["donor_pair_id"]) for row in plan_rows}
        if isinstance(plan_rows, list)
        and all(isinstance(row, dict) and set(row) == {"pair_id", "donor_pair_id"} for row in plan_rows)
        else None
    )
    if observed_plan != dict(expected_plan):
        raise ValueError("single-layer gate donor plan differs from deterministic derivation")
    runtime_value = json.loads(Path(runtime_manifest_path).read_text(encoding="utf-8"))
    if (
        provider.model_id != protocol.model
        or provider.model_revision != protocol.model_revision
        or provider.teacher_revision != protocol.model_revision
        or provider.student_revision != protocol.model_revision
        or provider.tokenizer_revision != protocol.model_revision
        or provider.code_revision != protocol.code_revision
        or provider.source_tree_sha256 != runtime_value.get("source_tree_sha256")
        or provider.decoder_layers != protocol.decoder_layers
        or provider.base_model_frozen is not True
    ):
        raise ValueError(
            "single-layer gate provider identity or freeze contract differs"
        )
    if dict(provider.provenance()) != runtime_value:
        raise ValueError("single-layer gate provider differs from the bound runtime manifest")
    for record in records:
        observed_bucket = provider.token_inflation_bucket(record)
        if observed_bucket != record.token_inflation_bucket:
            raise ValueError(
                "single-layer gate token inflation bucket differs from runtime tokenizer"
            )
    raw_rows = tuple(
        provider.scan(
            records,
            donor_plan=expected_plan,
            transition_layer=parent.selected_transition_layer,
        )
    )
    if len(raw_rows) != protocol.records:
        raise ValueError("single-layer gate provider did not scan the frozen cohort")

    output = Path(output_dir).resolve()
    parent_bundle_root = Path(parent_probe_artifact_path).resolve().parent
    if output == parent_bundle_root or output.is_relative_to(parent_bundle_root):
        raise ValueError("single-layer gate output must be outside the parent probe bundle")
    if output.exists():
        raise FileExistsError(f"single-layer gate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"single-layer gate temporary output exists: {temporary}")
    temporary.mkdir()
    try:
        parent_copy = _copy_parent_bundle(
            Path(parent_probe_artifact_path).resolve(), temporary / "parent-probe-bundle"
        )
        copied = {
            "config": temporary / "gate-config.json",
            "cohort_manifest": temporary / "cohort.json",
            "protected_split_registry": temporary / "protected.json",
            "donor_plan": temporary / "donor-plan.json",
            "runtime_manifest": temporary / "runtime.json",
        }
        _copy_regular(Path(config_path), copied["config"])
        _copy_regular(Path(cohort_manifest_path), copied["cohort_manifest"])
        _copy_regular(Path(protected_registry_path), copied["protected_split_registry"])
        _copy_regular(Path(donor_plan_path), copied["donor_plan"])
        _copy_regular(Path(runtime_manifest_path), copied["runtime_manifest"])
        raw_path = temporary / "raw-kl.json"
        _write(
            raw_path,
            {
                "schema_version": "single-layer-gate-raw-kl/v1",
                "bindings": {
                    "config_sha256": protocol.config_sha256,
                    "parent_probe_artifact_sha256": parent.artifact_sha256,
                    "cohort_manifest_sha256": _digest(copied["cohort_manifest"]),
                    "protected_registry_sha256": _digest(copied["protected_split_registry"]),
                    "donor_plan_sha256": _digest(copied["donor_plan"]),
                    "runtime_manifest_sha256": _digest(copied["runtime_manifest"]),
                },
                "records": list(raw_rows),
            },
        )
        references = {
            "config": copied["config"],
            "parent_probe_artifact": parent_copy,
            "cohort_manifest": copied["cohort_manifest"],
            "protected_split_registry": copied["protected_split_registry"],
            "donor_plan": copied["donor_plan"],
            "runtime_manifest": copied["runtime_manifest"],
            "raw_kl": raw_path,
        }
        artifact_path = temporary / "single-layer-gate.json"
        _write(
            artifact_path,
            {
                "schema_version": "probe-transition-single-layer-gate/v1",
                "operation": "validate-probe-transition-single-layer-causal-gate",
                "model": protocol.model,
                "model_revision": protocol.model_revision,
                "code_revision": protocol.code_revision,
                "decoder_layers": protocol.decoder_layers,
                "selected_transition_layer": parent.selected_transition_layer,
                "hook_site": "complete-decoder-block-residual-output",
                "coordinate": "edited-word-final-token/v1",
                "readout": "teacher-forced-tokens-2-through-16-inclusive/v1",
                "controls": [
                    "offset-plus-two",
                    "cross-item-derangement",
                    "self-copy-identity",
                ],
                "references": {
                    name: {
                        "relative_path": str(path.relative_to(temporary).as_posix()),
                        "sha256": _digest(path),
                    }
                    for name, path in references.items()
                },
                # The loader recomputes this bit from raw KL before accepting it.
                "passed": True,
            },
        )
        load_single_layer_gate_artifact(artifact_path)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_single_layer_gate_artifact(output / "single-layer-gate.json")


__all__ = ["SingleLayerGateProvider", "produce_single_layer_gate_artifact"]

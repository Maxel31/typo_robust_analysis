"""The public component operation is hash-bound to its layer-selection inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typo_robust_training.localization.component_causal import ComponentCausalObservation
from typo_robust_training.localization.component_runner import (
    ComponentLocalizationRunConfig,
    run_localize_robustness_components,
)
from typo_robust_training.localization.component_screening import ComponentScreenMetric
from typo_robust_training.localization.components import ComponentRef
from typo_robust_training.localization.records import LayerScan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-component-localization.yaml"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["architecture"].update(
        {
            "decoder_layers": 4,
            "hidden_size": 6,
            "mlp_intermediate_size": 4,
            "attention_heads": 2,
            "attention_head_dim": 2,
        }
    )
    config["screening"].update(
        {
            "mlp_shortlist_per_layer": 2,
            "attention_shortlist_per_layer": 2,
            "causal_candidate_limits": {"mlp-neuron": 2, "attention-head": 1},
        }
    )
    config["causal_validation"].update(
        {
            "minimum_kl_eligible_per_task": 1,
            "minimum_kl_eligible_fraction_per_task": 0.0,
            "minimum_answer_cohort_per_task": 1,
            "bootstrap_replicates": 20,
        }
    )
    config_path = tmp_path / "components.yaml"
    _write_json(config_path, config)

    manifest_path = tmp_path / "diagnostic.jsonl"
    manifest_rows: list[dict[str, object]] = []
    scans: list[LayerScan] = []
    for task in ("gsm8k", "mmlu", "arc"):
        for index in range(4):
            record_id = f"{task}-{index}"
            manifest_rows.append(
                {
                    "schema_version": "robustness-fixed-typo-pair/v1",
                    "kind": "synthetic",
                    "record_id": record_id,
                    "source": task,
                    "source_revision": "a" * 40,
                    "source_split": "dev" if task == "mmlu" else "train",
                    "source_id": record_id,
                    "group_id": record_id,
                    "split": "diagnostic",
                    "clean_text": "The airport answer is two.",
                    "typo_text": "The arport answer is two.",
                    "task": task,
                    "answer": "2" if task == "gsm8k" else "A",
                    "metadata": {},
                    "operation": "deletion",
                    "operations": ["deletion"],
                    "edit_count": 1,
                    "generator_seed": 42,
                    "generator_variant": index,
                    "edits": [
                        {
                            "operation": "deletion",
                            "clean_word": "airport",
                            "typo_word": "arport",
                            "clean_char_span": [4, 11],
                            "typo_char_span": [4, 10],
                        }
                    ],
                }
            )
            repair = index % 2 == 0
            scans.append(
                LayerScan(
                    record_id=record_id,
                    task=task,
                    target_token_ids=tuple(range(16)),
                    untreated_kl_2_16=(2.0,) * 15,
                    patched_kl_2_16_by_layer=((1.0,) * 15,) * 4,
                    clean_correct=True,
                    typo_correct=not repair,
                    patched_correct_by_layer=(True,) * 4,
                    kl_invalid_reason=None,
                    audit={},
                )
            )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    layer_dir = tmp_path / "layers"
    layer_dir.mkdir()
    scans_path = layer_dir / "layer_scans.jsonl"
    scans_path.write_text(
        "".join(json.dumps(scan.as_dict(), sort_keys=True) + "\n" for scan in scans),
        encoding="utf-8",
    )
    selection_path = layer_dir / "layer_selection.json"
    _write_json(
        selection_path,
        {
            "schema_version": "robustness-layer-selection/v1",
            "model": "google/gemma-3-4b-it",
            "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
            "records": len(scans),
            "selected_window": {"start": 0, "stop": 2},
            "diagnostic_manifest_sha256": manifest_sha,
            "layer_scans_sha256": hashlib.sha256(scans_path.read_bytes()).hexdigest(),
            "runtime": {"num_decoder_layers": 4},
        },
    )
    return config_path, manifest_path, selection_path


class _Runtime:
    def screen_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        selected_layers: tuple[int, ...],
    ) -> tuple[ComponentScreenMetric, ...]:
        del layer_scan
        task = str(record["task"])
        metrics: list[ComponentScreenMetric] = []
        for layer in selected_layers:
            for kind, count in (("mlp-neuron", 4), ("attention-head", 2)):
                for index in range(count):
                    metrics.append(
                        ComponentScreenMetric(
                            component=ComponentRef(kind, layer, index),
                            task=task,
                            records=1,
                            activation_difference=float(count - index + layer),
                            gradient_attribution=float(count - index + layer),
                        )
                    )
        return tuple(metrics)

    def causal_pair(
        self,
        record: dict[str, object],
        layer_scan: LayerScan,
        candidates: tuple[ComponentRef, ...],
    ) -> tuple[ComponentCausalObservation, ...]:
        task = str(record["task"])
        rows: list[ComponentCausalObservation] = []
        for component in candidates:
            good = component.kind == "mlp-neuron" and component.index == 0
            one_task = component.kind == "attention-head"
            harmful = component.kind == "mlp-neuron" and component.index == 1
            rows.append(
                ComponentCausalObservation(
                    record_id=str(record["record_id"]),
                    task=task,
                    component=component,
                    untreated_mean_kl=layer_scan.untreated_mean_kl,
                    patched_mean_kl=(
                        1.0 if good or (one_task and task == "gsm8k") or harmful else 2.2
                    ),
                    clean_correct=layer_scan.clean_correct,
                    typo_correct=layer_scan.typo_correct,
                    patched_correct=(
                        False
                        if harmful and layer_scan.typo_correct
                        else good or (one_task and task == "gsm8k")
                    ),
                )
            )
        return tuple(rows)

    def provenance(self) -> dict[str, object]:
        return {"runtime": "offline-component-fixture/v1", "num_decoder_layers": 4}


def test_component_runner_binds_layers_partitions_and_causal_outputs(tmp_path: Path) -> None:
    config_path, manifest_path, selection_path = _fixture(tmp_path)
    result = run_localize_robustness_components(
        ComponentLocalizationRunConfig(
            config_path=config_path,
            diagnostic_manifest_path=manifest_path,
            layer_selection_path=selection_path,
            components=("mlp-neuron", "attention-head"),
            causal_readouts=("answer", "multitoken-kl"),
            gpu_id="3",
            output_dir=tmp_path / "output",
            resume=False,
        ),
        runtime=_Runtime(),
    )

    assert result.selected_components >= 1
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["selected"]
    assert all(item["causally_validated"] is True for item in selection["selected"])
    assert sum(item["weight"] for item in selection["selected"]) == 1.0
    partition = selection["diagnostic_partition"]
    assert set(partition["screening_ids"]).isdisjoint(partition["causal_validation_ids"])
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["layer_selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()

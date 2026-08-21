"""Hash-bound confirmatory joint-window selection and validation runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_robust_training.localization.confirmatory_records import JointWindowScan
from typo_robust_training.localization.confirmatory_runner import (
    JointWindowSelectionRunConfig,
    JointWindowValidationRunConfig,
    run_select_generic_joint_patch_window,
    run_validate_generic_joint_patch_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cycle3" / "gemma4b-generic-joint-window.yaml"


def _config(tmp_path: Path) -> Path:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["decoder_layers"] = 4
    payload["data"]["selection_records"] = 2
    payload["data"]["validation_records"] = 2
    payload["selection"]["minimum_eligible"] = 1
    payload["selection"]["minimum_eligible_fraction"] = 0.5
    payload["statistics"]["bootstrap_replicates"] = 20
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pair(record_id: str, *, role: str) -> dict[str, object]:
    return {
        "schema_version": "robustness-fixed-typo-pair/v1",
        "kind": "synthetic",
        "record_id": record_id,
        "source": "fineweb_edu",
        "source_revision": "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5",
        "source_split": "train",
        "source_id": f"fineweb_edu:{record_id}",
        "group_id": f"fineweb_edu:{record_id}",
        "split": f"localization-{role}",
        "clean_text": "The airport provides reliable transport for many communities and schools.",
        "typo_text": "The arport provides reliable transport for many communities and schools.",
        "task": None,
        "answer": None,
        "metadata": {},
        "operation": "deletion",
        "operations": ["deletion"],
        "edit_count": 1,
        "generator_seed": 42,
        "generator_variant": 0,
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


def _manifest(tmp_path: Path, *, role: str) -> Path:
    path = tmp_path / f"{role}.jsonl"
    path.write_text(
        "".join(json.dumps(_pair(f"{role}-{index}", role=role)) + "\n" for index in range(2)),
        encoding="utf-8",
    )
    return path


class _Runtime:
    def __init__(self) -> None:
        self.selection_calls: list[str] = []
        self.validation_calls: list[str] = []

    def scan_selection_pair(self, record: dict[str, object]) -> JointWindowScan:
        record_id = str(record["record_id"])
        self.selection_calls.append(record_id)
        return JointWindowScan(
            record_id=record_id,
            role="selection",
            decoder_layers=4,
            window_width=1,
            target_token_ids=tuple(range(16)),
            untreated_kl_2_16=(1.0,) * 15,
            patched_kl_2_16_by_window={
                0: (0.8,) * 15,
                1: (0.2,) * 15,
                2: (0.5,) * 15,
                3: (0.9,) * 15,
            },
            invalid_reason=None,
            audit={"runtime": "fixture"},
        )

    def scan_validation_pair(
        self, record: dict[str, object], selected_window: tuple[int, int]
    ) -> JointWindowScan:
        assert selected_window == (1, 2)
        record_id = str(record["record_id"])
        self.validation_calls.append(record_id)
        return JointWindowScan(
            record_id=record_id,
            role="validation",
            decoder_layers=4,
            window_width=1,
            target_token_ids=tuple(range(16)),
            untreated_kl_2_16=(1.0,) * 15,
            patched_kl_2_16_by_window={1: (0.3,) * 15},
            invalid_reason=None,
            audit={"runtime": "fixture"},
        )

    def provenance(self) -> dict[str, object]:
        return {
            "runtime": "offline-joint-window-fixture/v1",
            "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
            "num_decoder_layers": 4,
        }


def test_selection_and_validation_are_hash_bound_resumable_separate_operations(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runtime = _Runtime()
    selection_dir = tmp_path / "selection-output"
    selection_config = JointWindowSelectionRunConfig(
        config_path=config,
        selection_manifest_path=_manifest(tmp_path, role="selection"),
        gpu_id="5",
        output_dir=selection_dir,
    )
    selected = run_select_generic_joint_patch_window(selection_config, runtime=runtime)

    assert selected.selected_window == (1, 2)
    assert runtime.selection_calls == ["selection-0", "selection-1"]
    selection_payload = json.loads(selected.selection_path.read_text(encoding="utf-8"))
    assert selection_payload["selection_metric"] == "median-pairwise-kl-restoration/v1"
    assert selection_payload["selected_window"]["start"] == 1
    assert selection_payload["config_sha256"]
    assert selection_payload["selection_manifest_sha256"]
    assert selection_payload["joint_window_scans_sha256"]

    resumed = run_select_generic_joint_patch_window(
        JointWindowSelectionRunConfig(
            config_path=selection_config.config_path,
            selection_manifest_path=selection_config.selection_manifest_path,
            gpu_id=selection_config.gpu_id,
            output_dir=selection_config.output_dir,
            resume=True,
        ),
        runtime=_Runtime(),
    )
    assert resumed.selection_path == selected.selection_path

    validation_runtime = _Runtime()
    validated = run_validate_generic_joint_patch_window(
        JointWindowValidationRunConfig(
            config_path=config,
            validation_manifest_path=_manifest(tmp_path, role="validation"),
            window_selection_path=selected.selection_path,
            gpu_id="6",
            output_dir=tmp_path / "validation-output",
        ),
        runtime=validation_runtime,
    )
    assert validated.passed is True
    assert validation_runtime.validation_calls == ["validation-0", "validation-1"]
    validation_payload = json.loads(validated.validation_path.read_text(encoding="utf-8"))
    assert validation_payload["selected_window"] == {"start": 1, "stop": 2}
    assert validation_payload["passed"] is True
    assert validation_payload["window_selection_sha256"]


def test_validation_rejects_window_evidence_with_wrong_config_binding(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = tmp_path / "window.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "robustness-joint-window-selection/v1",
                "operation": "select-generic-joint-patch-window",
                "config_sha256": "0" * 64,
                "model": "google/gemma-3-4b-it",
                "model_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
                "selected_window": {"start": 1, "stop": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config binding"):
        run_validate_generic_joint_patch_window(
            JointWindowValidationRunConfig(
                config_path=config,
                validation_manifest_path=_manifest(tmp_path, role="validation"),
                window_selection_path=evidence,
                gpu_id="6",
                output_dir=tmp_path / "validation-output",
            ),
            runtime=_Runtime(),
        )

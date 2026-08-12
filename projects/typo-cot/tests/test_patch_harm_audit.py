"""Contracts for the correct-answer clean-to-typo patch harm audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import REBUTTAL_SETTINGS
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.patch_harm_audit import runner as harm_runner
from typo_cot.experiments.patch_harm_audit.protocol import load_patch_harm_audit_protocol
from typo_cot.experiments.patch_harm_audit.runner import (
    PatchHarmAuditConfig,
    PatchHarmAuditRunError,
    PatchHarmGeneration,
    PatchHarmScan,
    run_patch_harm_audit,
)
from typo_cot.experiments.patch_harm_audit.runtime import (
    HuggingFacePatchHarmAuditRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "patch-harm-audit.yaml"


def _write_protocol(path: Path) -> None:
    path.write_bytes(DEFAULT_CONFIG.read_bytes())


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _record(
    *,
    model: str,
    task: str,
    sample_id: str,
    harm: bool,
    restoration: bool,
    fixed_event: bool | None,
) -> dict[str, object]:
    pair_id = _digest(f"{model}\0{task}\0{sample_id}")
    return {
        "pair_id": pair_id,
        "sample_id": sample_id,
        "model": model,
        "task": task,
        "target_rule": "attribution-4",
        "gold_answer": "42",
        "clean_text": "clean prompt",
        "typo_text": "typo prompt",
        "clean_prompt_token_count": 10,
        "typo_prompt_token_count": 11,
        "clean_answer": "42",
        "typo_answer": "42" if harm else "41",
        "clean_correct": True,
        "typo_correct": harm,
        "number_of_aligned_words": 1,
        "edits": [
            {
                "clean_char_span": [0, 5],
                "typo_char_span": [0, 4],
                "clean_word_final_token": 5,
                "typo_word_final_token": 6,
            }
        ],
        "cohorts": {
            "harm": harm,
            "restoration": restoration,
            "prepared_typo_wrong_outside_restoration": False,
        },
        "controls": {
            "correct": {
                "valid": True,
                "source_positions": [5],
                "destination_positions": [6],
            }
        },
        "fixed_window": {"event": fixed_event},
        "source": {
            "source_record_sha256": _digest(pair_id),
            "model_revision": _digest(model),
        },
    }


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        for index in range(setting.paper_denominator):
            records.append(
                _record(
                    model=setting.model,
                    task=setting.task,
                    sample_id=f"repair-{index:04d}",
                    harm=False,
                    restoration=True,
                    fixed_event=index < setting.paper_successes,
                )
            )
        for index in range(3):
            records.append(
                _record(
                    model=setting.model,
                    task=setting.task,
                    sample_id=f"harm-{index:04d}",
                    harm=True,
                    restoration=False,
                    fixed_event=None,
                )
            )
    return tuple(records)


class _Runtime:
    num_layers = 12
    calls = 0

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        self.model = model
        self.task = task
        self.revision = revision
        self.gpu_id = gpu_id

    def provenance(self) -> Mapping[str, object]:
        return {
            "operation": "patch-harm-audit",
            "runtime": "patch-harm-fixture",
            "model": self.model,
            "task": self.task,
            "requested_revision": self.revision,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "cuda_visible_devices": self.gpu_id,
            "coordinate_source": "rebuttal-pair-manifest-correct/v1",
            "layer_window": [0, 6],
            "cohort": "clean-correct-typo-correct",
            "generated_arm": "correct-coordinate-clean-to-typo/v1",
            "baseline_source": "manifest-stored-deterministically-reextracted-typo-answer/v1",
            "effective_eos_token_ids": [1, 107],
            "effective_eos_token_ids_source": "fixture",
            "generation_termination_protocol": "effective-eos-vs-length-cap/v1",
            "answer_extraction": "primary-then-empty-only-positional-by-termination/v1",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 512,
                "use_cache": True,
                "return_dict_in_generate": False,
                "output_scores": False,
                "padding_side": "left",
                "patch_application": "layers-0-5-on-typo-prompt-prefill-exactly-once/v1",
            },
        }

    def scan_pair(self, pair: Mapping[str, object]) -> PatchHarmScan:
        type(self).calls += 1
        index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        if index == 0:
            value, extracted, correct = "42", True, True
        elif index == 1:
            value, extracted, correct = "99", True, False
        else:
            value, extracted, correct = "", False, False
        return PatchHarmScan(
            generation=PatchHarmGeneration(
                token_ids=(10 + index, 1),
                text=f"patched-{index}",
                termination="eos",
                value=value,
                is_extracted=extracted,
                is_correct=correct,
                method="fixture",
                primary_method="fixture-primary",
            ),
            source_positions=(5,),
            destination_positions=(6,),
        )


def _runner_config(
    tmp_path: Path,
    *,
    limit_per_setting: int | None = None,
) -> PatchHarmAuditConfig:
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("fixture\n", encoding="utf-8")
    return PatchHarmAuditConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        cohort="clean-correct-typo-correct",
        gpu_id="1",
        output_dir=tmp_path / "output",
        limit_per_setting=limit_per_setting,
    )


def test_default_protocol_catalog_and_cli_match_the_frozen_readme() -> None:
    protocol = load_patch_harm_audit_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "patch-harm-audit-config/v1"
    assert protocol.cohort == "clean-correct-typo-correct"
    assert protocol.selection == "all-prepared-clean-correct-typo-correct-aligned-uncapped/v1"
    assert protocol.window == (0, 6)
    assert protocol.direction == "clean-to-typo"
    assert protocol.max_new_tokens == 512
    assert protocol.harm_definition == "patched-incorrect-including-unextractable/v1"
    assert protocol.composite_label == "repair-harm-conditional-composite"
    assert protocol.restoration_pairs == 1_241
    assert protocol.restoration_successes == 800

    spec = get_experiment("patch-harm-audit")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--cohort",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "patch_harm_records.jsonl",
        "setting_harm_table.csv",
        "repair_harm_composite.csv",
        "patch_harm_summary.json",
        "run.json",
    )

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "patch-harm-audit" in subparsers.choices


def test_protocol_rejects_cohort_window_or_harm_definition_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["cohort"]["name"] = "other"
    changed = tmp_path / "changed.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort"):
        load_patch_harm_audit_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["intervention"]["window"]["stop"] = 7
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0,6\)"):
        load_patch_harm_audit_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["outcomes"]["harm"] = "exclude-unextractable"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outcome"):
        load_patch_harm_audit_protocol(changed)


def test_generation_and_scan_reject_inconsistent_payloads() -> None:
    with pytest.raises(ValueError, match="extraction flag"):
        PatchHarmGeneration(
            token_ids=(1,),
            text="answer",
            termination="eos",
            value="42",
            is_extracted=False,
            is_correct=True,
            method="fixture",
            primary_method="fixture",
        )
    generation = PatchHarmGeneration(
        token_ids=(1,),
        text="answer",
        termination="eos",
        value="42",
        is_extracted=True,
        is_correct=True,
        method="fixture",
        primary_method="fixture",
    )
    with pytest.raises(ValueError, match="cardinality"):
        PatchHarmScan(generation, source_positions=(5,), destination_positions=(6, 7))


def test_runtime_uses_manifest_correct_coordinates_and_layers_zero_through_five() -> None:
    pair = _record(
        model=REBUTTAL_SETTINGS[0].model,
        task=REBUTTAL_SETTINGS[0].task,
        sample_id="harm-0000",
        harm=True,
        restoration=False,
        fixed_event=None,
    )
    runtime = object.__new__(HuggingFacePatchHarmAuditRuntime)
    runtime.num_layers = 12
    runtime.layers = tuple(nn.Identity() for _ in range(12))
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)

    def tokenize(
        _self: object,
        _pair: Mapping[str, object],
        *,
        side: str,
    ) -> tuple[str, str, tuple[int, ...]]:
        positions = (5,) if side == "clean" else (6,)
        return f"{side}-ids", f"{side}-mask", positions

    captured: dict[str, object] = {}

    def capture(
        _self: object,
        *,
        input_ids: object,
        attention_mask: object,
        positions: tuple[int, ...],
    ) -> list[torch.Tensor]:
        captured["capture"] = input_ids, attention_mask, positions
        return [torch.ones((1, 2), dtype=torch.float32) for _ in range(12)]

    expected_generation = PatchHarmGeneration(
        token_ids=(1,),
        text="answer",
        termination="eos",
        value="42",
        is_extracted=True,
        is_correct=True,
        method="fixture",
        primary_method="fixture",
    )

    def generate(
        _self: object,
        *,
        input_ids: object,
        attention_mask: object,
        correct_answer: str,
        patch: object,
        field: str,
    ) -> PatchHarmGeneration:
        captured["generate"] = input_ids, attention_mask, correct_answer, field
        captured["layers"] = patch.layer_indices
        return expected_generation

    runtime._tokenize_and_validate = MethodType(tokenize, runtime)
    runtime._capture = MethodType(capture, runtime)
    runtime._generate_harm = MethodType(generate, runtime)

    scan = runtime.scan_pair(pair)

    assert scan.generation == expected_generation
    assert scan.source_positions == (5,)
    assert scan.destination_positions == (6,)
    assert captured["capture"] == ("clean-ids", "clean-mask", (5,))
    assert captured["generate"] == (
        "edited-ids",
        "edited-mask",
        "42",
        f"{pair['pair_id']}:correct-coordinate",
    )
    assert captured["layers"] == tuple(range(6))


def test_runner_compiles_harm_and_conditional_composite_then_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    monkeypatch.setattr(harm_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    _Runtime.calls = 0
    config = _runner_config(tmp_path)

    result = run_patch_harm_audit(config, runtime_factory=_Runtime)

    assert result.harm_pairs == 18
    assert result.evaluated_pairs == 18
    assert result.preserve == 6
    assert result.harm == 12
    assert result.answer_changed == 6
    assert result.unextractable == 6
    assert result.settings == 6
    assert _Runtime.calls == 18

    records_out = [
        json.loads(line) for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records_out) == 18
    assert sum(row["preserve"] for row in records_out) == 6
    assert sum(row["harm"] for row in records_out) == 12
    assert sum(row["answer_changed"] for row in records_out) == 6
    assert sum(row["unextractable"] for row in records_out) == 6
    assert all(row["baseline"]["is_correct"] is True for row in records_out)
    assert all(row["source_positions"] == [5] for row in records_out)
    assert all(row["destination_positions"] == [6] for row in records_out)

    with result.setting_table_path.open(encoding="utf-8", newline="") as handle:
        setting_rows = tuple(csv.DictReader(handle))
    assert len(setting_rows) == 6
    assert all(int(row["n_typo_correct"]) == 3 for row in setting_rows)
    assert all(int(row["n_evaluated"]) == 3 for row in setting_rows)
    assert all(int(row["preserve"]) == 1 for row in setting_rows)
    assert all(int(row["harm"]) == 2 for row in setting_rows)
    assert all(float(row["harm_rate"]) == pytest.approx(2 / 3) for row in setting_rows)

    with result.composite_path.open(encoding="utf-8", newline="") as handle:
        composite_rows = tuple(csv.DictReader(handle))
    overall = next(row for row in composite_rows if row["scope"] == "overall")
    assert overall["label"] == "repair-harm-conditional-composite"
    assert overall["complete"] == "True"
    assert int(overall["restoration_n"]) == 1_241
    assert int(overall["wrong_to_right"]) == 800
    assert int(overall["harm_n"]) == 18
    assert int(overall["right_to_wrong"]) == 12
    assert int(overall["transition_balance"]) == 788
    assert int(overall["composite_baseline_correct"]) == 18
    assert int(overall["composite_patched_correct"]) == 806
    assert float(overall["composite_accuracy_change"]) == pytest.approx(788 / 1_259)
    assert overall["population_net_accuracy"] == "False"

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["confirmatory"] is True
    assert summary["counts"]["right_to_wrong"] == 12
    assert summary["repair_harm_composite"]["population_net_accuracy"] is False
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["confirmatory"] is True

    resumed = run_patch_harm_audit(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result


def test_limit_is_nonconfirmatory_and_withholds_composite_transition_estimates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harm_runner, "load_rebuttal_pair_manifest", lambda _path: _records())
    config = _runner_config(tmp_path, limit_per_setting=1)

    result = run_patch_harm_audit(config, runtime_factory=_Runtime)

    assert result.harm_pairs == 18
    assert result.evaluated_pairs == 6
    with result.composite_path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert all(row["complete"] == "False" for row in rows)
    assert all(row["invalid_reason"] == "non-confirmatory-limit" for row in rows)
    assert all(row["transition_balance"] == "" for row in rows)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["confirmatory"] is False
    assert summary["repair_harm_composite"]["available"] is False


def test_interrupted_run_retains_verified_checkpoint_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harm_runner, "load_rebuttal_pair_manifest", lambda _path: _records())

    class FailingRuntime(_Runtime):
        attempts = 0

        def scan_pair(self, *args: object, **kwargs: object) -> PatchHarmScan:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("injected interruption")
            return super().scan_pair(*args, **kwargs)  # type: ignore[arg-type]

    config = _runner_config(tmp_path, limit_per_setting=1)
    with pytest.raises(PatchHarmAuditRunError):
        run_patch_harm_audit(config, runtime_factory=FailingRuntime)
    assert len(tuple((config.output_dir / "checkpoints").glob("*.json"))) == 1

    resumed = run_patch_harm_audit(
        replace(config, resume=True),
        runtime_factory=_Runtime,
    )
    assert resumed.evaluated_pairs == 6


def test_resume_rejects_a_tampered_pair_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harm_runner, "load_rebuttal_pair_manifest", lambda _path: _records())

    class FailAfterOne(_Runtime):
        attempts = 0

        def scan_pair(self, *args: object, **kwargs: object) -> PatchHarmScan:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("injected interruption")
            return super().scan_pair(*args, **kwargs)  # type: ignore[arg-type]

    config = _runner_config(tmp_path, limit_per_setting=1)
    with pytest.raises(PatchHarmAuditRunError):
        run_patch_harm_audit(config, runtime_factory=FailAfterOne)
    (checkpoint,) = tuple((config.output_dir / "checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["scan"]["generation"]["value"] = "tampered"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PatchHarmAuditRunError) as exc_info:
        run_patch_harm_audit(
            replace(config, resume=True),
            runtime_factory=_Runtime,
        )
    assert "checkpoint provenance differs" in str(exc_info.value.__cause__)


def test_completed_resume_rejects_tampered_public_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harm_runner, "load_rebuttal_pair_manifest", lambda _path: _records())
    config = _runner_config(tmp_path, limit_per_setting=1)
    result = run_patch_harm_audit(config, runtime_factory=_Runtime)
    result.summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output SHA-256 differs"):
        run_patch_harm_audit(
            replace(config, resume=True),
            runtime_factory=_Runtime,
        )

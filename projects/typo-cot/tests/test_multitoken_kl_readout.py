"""Contracts for the teacher-forced multi-token KL rebuttal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import typo_cot.cli as cli_module
from typo_cot.experiments.build_rebuttal_manifest import REBUTTAL_SETTINGS
from typo_cot.experiments.catalog import get_experiment
from typo_cot.experiments.multitoken_kl_readout import runner as readout_runner
from typo_cot.experiments.multitoken_kl_readout.metrics import (
    kl_trajectory_from_logits,
    restoration_score,
    summarize_restoration,
)
from typo_cot.experiments.multitoken_kl_readout.protocol import (
    load_multitoken_kl_readout_protocol,
)
from typo_cot.experiments.multitoken_kl_readout.runner import (
    MultiTokenKLReadoutConfig,
    MultiTokenKLReadoutRunError,
    MultiTokenKLScan,
    run_multitoken_kl_readout,
)
from typo_cot.experiments.multitoken_kl_readout.runtime import (
    CleanContinuationTooShort,
    CleanPromptPrefixMismatch,
    HuggingFaceMultiTokenKLReadoutRuntime,
    clean_continuation_target_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rebuttal" / "multitoken-kl-readout.yaml"


def _write_protocol(path: Path) -> None:
    path.write_bytes(DEFAULT_CONFIG.read_bytes())


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    def bootstrap(
        values: tuple[float, ...],
        **_kwargs: object,
    ) -> dict[str, float]:
        estimate = float(torch.tensor(values, dtype=torch.float64).median().item())
        return {"estimate": estimate, "lower": estimate, "upper": estimate}

    monkeypatch.setattr(readout_runner, "bootstrap_median", bootstrap)


def _records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for setting_index, setting in enumerate(REBUTTAL_SETTINGS):
        for pair_index in range(setting.paper_denominator):
            pair_id = hashlib.sha256(
                f"multitoken\0{setting_index}\0{pair_index}".encode()
            ).hexdigest()
            records.append(
                {
                    "pair_id": pair_id,
                    "sample_id": f"sample-{pair_index:04d}",
                    "task": setting.task,
                    "model": setting.model,
                    "target_rule": "attribution-4",
                    "clean_continuation": " fixture continuation with at least sixteen tokens",
                    "cohorts": {"restoration": True},
                    "source": {
                        "source_record_sha256": hashlib.sha256(pair_id.encode()).hexdigest(),
                        "model_revision": hashlib.sha256(setting.model.encode()).hexdigest(),
                    },
                }
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
            "operation": "multitoken-kl-readout",
            "runtime": "multitoken-fixture",
            "model": self.model,
            "task": self.task,
            "requested_revision": self.revision,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "num_decoder_layers": self.num_layers,
            "dtype": "bfloat16",
            "cuda_visible_devices": self.gpu_id,
            "coordinate_source": "rebuttal-pair-manifest/v1",
            "layer_window": [0, 6],
            "teacher_forced_tokens": 16,
            "target_source": "clean-continuation-token-ids/v1",
            "prompt_prefix_validation": "exact-token-id-prefix/v1",
            "model_inputs": "prompt-plus-first-15-target-token-ids/v1",
            "short_continuation_policy": "record-unavailable-before-forward/v1",
            "prompt_prefix_mismatch_policy": "record-unavailable-before-forward/v1",
            "divergence": "kl-clean-to-condition/v1",
            "negative_kl_roundoff_tolerance": 1e-12,
            "forward": {
                "use_cache": False,
                "logits_materialized_dtype": "float32",
                "kl_dtype": "float64",
            },
        }

    def scan_pair(
        self,
        pair: Mapping[str, object],
        *,
        teacher_forced_tokens: int,
        layer_window: tuple[int, int],
    ) -> MultiTokenKLScan:
        type(self).calls += 1
        assert teacher_forced_tokens == 16
        assert layer_window == (0, 6)
        pair_index = int(str(pair["sample_id"]).rsplit("-", 1)[1])
        if pair_index == 0:
            return MultiTokenKLScan(
                target_token_ids=(100, 101, 102),
                target_token_text=("token-1", "token-2", "token-3"),
                untreated_kl=(),
                patched_kl=(),
                available=False,
                invalid_reason="clean_continuation_lt_16",
            )
        untreated = tuple(float(i + 1) for i in range(16))
        patched = tuple(value * 0.25 for value in untreated)
        return MultiTokenKLScan(
            target_token_ids=tuple(range(100, 116)),
            target_token_text=tuple(f"token-{index}" for index in range(1, 17)),
            untreated_kl=untreated,
            patched_kl=patched,
        )


def test_default_protocol_catalog_and_cli_match_the_frozen_readme() -> None:
    protocol = load_multitoken_kl_readout_protocol(DEFAULT_CONFIG)

    assert protocol.schema_version == "multitoken-kl-readout-config/v1"
    assert protocol.window == (0, 6)
    assert protocol.teacher_forced_tokens == 16
    assert protocol.short_continuation_policy == "record-unavailable-before-forward/v1"
    assert protocol.prompt_prefix_mismatch_policy == "record-unavailable-before-forward/v1"
    assert protocol.primary_token_range == (2, 16)
    assert protocol.secondary_token_ranges == ((2, 4), (2, 8))
    assert protocol.denominator_epsilon == 1e-9
    assert protocol.logits_materialized_dtype == "float32"
    assert protocol.kl_dtype == "float64"
    assert protocol.negative_restoration == "retain"
    assert protocol.pair_bootstrap_replicates == 10_000
    assert protocol.bootstrap_seed == 42
    assert protocol.confidence_level == 0.95

    spec = get_experiment("multitoken-kl-readout")
    assert spec.status == "implemented"
    assert spec.compute == "gpu"
    assert spec.required_arguments == (
        "--config",
        "--manifest",
        "--teacher-forced-tokens",
        "--primary-token-range",
        "--gpu-id",
        "--output-dir",
    )
    assert spec.outputs == (
        "multitoken_kl_records.jsonl",
        "setting_metrics.csv",
        "token_position_trajectory.csv",
        "token_position_trajectory.svg",
        "multitoken_summary.json",
        "run.json",
    )

    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "multitoken-kl-readout" in subparsers.choices


def test_protocol_rejects_bootstrap_or_primary_range_drift(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["statistics"]["pair_bootstrap_replicates"] = 9_999
    changed = tmp_path / "changed.yaml"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pair_bootstrap_replicates.*10,000"):
        load_multitoken_kl_readout_protocol(changed)

    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["readout"]["primary_token_range"] = {"start": 1, "stop": 16}
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="primary token range"):
        load_multitoken_kl_readout_protocol(changed)


def test_kl_is_computed_in_float64_and_negative_restoration_is_retained() -> None:
    reference = torch.tensor([[0.0, 1.0, -2.0], [2.0, -1.0, 0.0]], dtype=torch.bfloat16)
    comparison = torch.tensor([[1.0, 0.0, -2.0], [0.0, 1.0, -1.0]], dtype=torch.bfloat16)

    actual = kl_trajectory_from_logits(reference.float(), comparison.float())
    expected = torch.sum(
        torch.log_softmax(reference.double(), dim=-1).exp()
        * (
            torch.log_softmax(reference.double(), dim=-1)
            - torch.log_softmax(comparison.double(), dim=-1)
        ),
        dim=-1,
    )

    assert actual == pytest.approx(tuple(float(value) for value in expected), abs=1e-14)
    assert restoration_score(
        untreated_kl=(1.0, 1.0),
        patched_kl=(2.0, 2.0),
        token_range=(1, 2),
        denominator_epsilon=1e-9,
    ) == -1.0
    with pytest.raises(ValueError, match="denominator_le_1e-9"):
        restoration_score(
            untreated_kl=(0.0, 0.0),
            patched_kl=(0.0, 0.0),
            token_range=(1, 2),
            denominator_epsilon=1e-9,
        )


class _Tokenizer:
    def __init__(self, values: Mapping[str, list[int]]) -> None:
        self.values = values

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": True, "return_attention_mask": False}
        return {"input_ids": self.values[text]}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"<{token}>" for token in token_ids]

    def decode(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("target labels must not independently decode token IDs")


def test_clean_continuation_targets_require_an_exact_prompt_prefix() -> None:
    prompt = "prompt"
    continuation = " continuation"
    tokenizer = _Tokenizer(
        {
            prompt: [1, 2, 3],
            prompt + continuation: [1, 2, 3, *range(10, 30)],
        }
    )

    target_ids, target_text = clean_continuation_target_ids(
        tokenizer,
        clean_prompt=prompt,
        clean_continuation=continuation,
        count=16,
    )

    assert target_ids == tuple(range(10, 26))
    assert target_text == tuple(f"<{token}>" for token in range(10, 26))

    bad = _Tokenizer({prompt: [1, 2, 3], prompt + continuation: [1, 9, 3, *range(20)]})
    with pytest.raises(CleanPromptPrefixMismatch, match="exact token-ID prefix"):
        clean_continuation_target_ids(
            bad,
            clean_prompt=prompt,
            clean_continuation=continuation,
            count=16,
        )

    short = _Tokenizer({prompt: [1, 2, 3], prompt + continuation: [1, 2, 3, 10, 11, 12]})
    with pytest.raises(CleanContinuationTooShort) as exc_info:
        clean_continuation_target_ids(
            short,
            clean_prompt=prompt,
            clean_continuation=continuation,
            count=16,
        )
    assert exc_info.value.available_token_ids == (10, 11, 12)
    assert exc_info.value.available_token_text == ("<10>", "<11>", "<12>")


def test_short_clean_continuation_is_recorded_before_alignment_or_forward() -> None:
    prompt = "prompt"
    continuation = " short"
    runtime = object.__new__(HuggingFaceMultiTokenKLReadoutRuntime)
    runtime.num_layers = 6
    runtime.tokenizer = _Tokenizer(
        {prompt: [1, 2, 3], prompt + continuation: [1, 2, 3, 10, 11, 12]}
    )
    runtime._tokenize_and_validate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("short continuations must be rejected before alignment or model use")
    )

    scan = runtime.scan_pair(
        {"clean_text": prompt, "clean_continuation": continuation},
        teacher_forced_tokens=16,
        layer_window=(0, 6),
    )

    assert scan.available is False
    assert scan.invalid_reason == "clean_continuation_lt_16"
    assert scan.target_token_ids == (10, 11, 12)
    assert scan.untreated_kl == scan.patched_kl == ()


def test_prompt_prefix_mismatch_is_recorded_before_alignment_or_forward() -> None:
    prompt = "prompt"
    continuation = " boundary"
    runtime = object.__new__(HuggingFaceMultiTokenKLReadoutRuntime)
    runtime.num_layers = 6
    runtime.tokenizer = _Tokenizer(
        {prompt: [1, 2, 3], prompt + continuation: [1, 2, 9, *range(10, 30)]}
    )
    runtime._tokenize_and_validate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("prefix mismatches must be recorded before alignment or model use")
    )

    scan = runtime.scan_pair(
        {"clean_text": prompt, "clean_continuation": continuation},
        teacher_forced_tokens=16,
        layer_window=(0, 6),
    )

    assert scan.available is False
    assert scan.invalid_reason == "clean_prompt_not_exact_token_prefix"
    assert scan.target_token_ids == scan.target_token_text == ()
    assert scan.untreated_kl == scan.patched_kl == ()


def test_checkpoint_name_is_content_addressed() -> None:
    record = _records()[0]
    path = readout_runner._checkpoint_path(
        Path("checkpoints"),
        record=record,
        runtime_fingerprint="a" * 64,
        protocol_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )

    assert path.parent == Path("checkpoints")
    assert path.stem != record["pair_id"]
    assert len(path.stem) == 64
    assert path.suffix == ".json"


def test_restoration_summary_is_the_single_source_for_means_and_validity() -> None:
    valid = summarize_restoration(
        untreated_kl=(1.0, 3.0, 5.0),
        patched_kl=(0.5, 1.5, 2.5),
        token_range=(2, 3),
        denominator_epsilon=1e-9,
    )
    assert valid.untreated_mean_kl == 4.0
    assert valid.patched_mean_kl == 2.0
    assert valid.value == 0.5
    assert valid.invalid_reason is None

    invalid = summarize_restoration(
        untreated_kl=(0.0, 0.0),
        patched_kl=(0.0, 0.0),
        token_range=(1, 2),
        denominator_epsilon=1e-9,
    )
    assert invalid.value is None
    assert invalid.invalid_reason == "denominator_le_1e-9"


def test_trajectory_svg_cycles_its_palette_for_additional_series() -> None:
    rows = [
        {
            "model": f"model-{setting}",
            "task": "fixture",
            "token_index": token,
            "median_raw_kl_reduction": float(setting + token),
        }
        for setting in range(7)
        for token in range(1, 17)
    ]

    svg = readout_runner._trajectory_svg(rows)

    assert svg.startswith("<svg")
    assert "model-6 / fixture" in svg


def test_runtime_appends_only_15_prefix_tokens_and_reads_all_16_predictions() -> None:
    runtime = object.__new__(HuggingFaceMultiTokenKLReadoutRuntime)
    runtime._torch = torch
    prompt_ids = torch.tensor([[1, 2, 3]])
    prompt_mask = torch.ones_like(prompt_ids)
    targets = tuple(range(10, 26))

    full_ids, full_mask = runtime._append_targets(prompt_ids, prompt_mask, targets)

    assert full_ids.tolist() == [[1, 2, 3, *range(10, 25)]]
    assert full_mask.tolist() == [[1] * 18]
    logits = torch.arange(18 * 2, dtype=torch.float32).reshape(1, 18, 2)
    selected = runtime._target_logits(
        SimpleNamespace(logits=logits),
        prompt_tokens=3,
        count=16,
    )
    assert selected.equal(logits[0, 2:18, :])


def _runner_config(tmp_path: Path, *, limit: int | None = 2) -> MultiTokenKLReadoutConfig:
    protocol_path = tmp_path / "protocol.yaml"
    _write_protocol(protocol_path)
    manifest_path = tmp_path / "manifest" / "pair_manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("fixture\n", encoding="utf-8")
    return MultiTokenKLReadoutConfig(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        teacher_forced_tokens=16,
        primary_token_range=(2, 16),
        gpu_id="3",
        output_dir=tmp_path / "output",
        limit_per_setting=limit,
    )


def test_runner_compiles_six_settings_excludes_near_zero_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    monkeypatch.setattr(readout_runner, "load_rebuttal_pair_manifest", lambda _path: records)
    _Runtime.calls = 0
    config = _runner_config(tmp_path)

    result = run_multitoken_kl_readout(config, runtime_factory=_Runtime)

    assert result.pairs == 12
    assert result.settings == 6
    assert result.primary_valid_pairs == 6
    assert result.target_available_pairs == 6
    assert result.trajectory_rows == 96
    assert _Runtime.calls == 12
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["confirmatory"] is False
    assert run["counts"] == {
        "pairs": 12,
        "primary_valid_pairs": 6,
        "records": 12,
        "setting_metrics": 6,
        "settings": 6,
        "target_available_pairs": 6,
        "trajectory_rows": 96,
    }
    records_out = [
        json.loads(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["readout_protocol_sha256"] == run["protocol_sha256"] for row in records_out)
    assert all(row["pair_manifest_sha256"] == run["pair_manifest_sha256"] for row in records_out)
    assert all(len(row["source_record_sha256"]) == 64 for row in records_out)
    assert sum(row["metrics"]["R_2:16"]["valid"] for row in records_out) == 6
    unavailable = [row for row in records_out if not row["readout_available"]]
    assert len(unavailable) == 6
    assert all(row["invalid_reason"] == "clean_continuation_lt_16" for row in unavailable)
    assert all(row["untreated_kl"] == [] and row["patched_kl"] == [] for row in unavailable)
    assert all(
        row["metrics"]["R_2:16"]["value"] == pytest.approx(0.75)
        for row in records_out
        if row["metrics"]["R_2:16"]["valid"]
    )
    assert result.trajectory_plot_path.read_text(encoding="utf-8").startswith("<svg")

    resumed = run_multitoken_kl_readout(
        replace(config, resume=True),
        runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not load a runtime")
        ),
    )
    assert resumed == result


def test_interrupted_run_retains_verified_checkpoint_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    monkeypatch.setattr(readout_runner, "load_rebuttal_pair_manifest", lambda _path: records)

    class FailingRuntime(_Runtime):
        attempts = 0

        def scan_pair(self, *args: object, **kwargs: object) -> MultiTokenKLScan:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("injected interruption")
            return super().scan_pair(*args, **kwargs)  # type: ignore[arg-type]

    config = _runner_config(tmp_path, limit=1)
    with pytest.raises(MultiTokenKLReadoutRunError):
        run_multitoken_kl_readout(config, runtime_factory=FailingRuntime)
    assert len(tuple((config.output_dir / "checkpoints").glob("*.json"))) == 1

    resumed = run_multitoken_kl_readout(
        replace(config, resume=True),
        runtime_factory=_Runtime,
    )
    assert resumed.pairs == 6
    assert resumed.settings == 6


def test_resume_rejects_a_tampered_pair_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readout_runner, "load_rebuttal_pair_manifest", lambda _path: _records())

    class FailAfterOne(_Runtime):
        attempts = 0

        def scan_pair(self, *args: object, **kwargs: object) -> MultiTokenKLScan:
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("injected interruption")
            return super().scan_pair(*args, **kwargs)  # type: ignore[arg-type]

    config = _runner_config(tmp_path, limit=1)
    with pytest.raises(MultiTokenKLReadoutRunError):
        run_multitoken_kl_readout(config, runtime_factory=FailAfterOne)
    (checkpoint,) = tuple((config.output_dir / "checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["scan"]["target_token_ids"][0] = 999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MultiTokenKLReadoutRunError) as exc_info:
        run_multitoken_kl_readout(
            replace(config, resume=True),
            runtime_factory=_Runtime,
        )
    assert "checkpoint provenance differs" in str(exc_info.value.__cause__)


def test_completed_resume_rejects_tampered_public_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readout_runner, "load_rebuttal_pair_manifest", lambda _path: _records())
    config = _runner_config(tmp_path, limit=1)
    result = run_multitoken_kl_readout(config, runtime_factory=_Runtime)
    result.summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output SHA-256 differs"):
        run_multitoken_kl_readout(
            replace(config, resume=True),
            runtime_factory=_Runtime,
        )

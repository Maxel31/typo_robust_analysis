"""Public contracts for the Appendix E typo-warning reproduction."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import typo_cot.cli as cli_module
import typo_cot.experiments.typo_warning_prompt.aggregation as warning_aggregation
import typo_cot.experiments.typo_warning_prompt.runner as warning_runner
import typo_cot.experiments.typo_warning_prompt.runtime as warning_runtime
from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.typo_warning_prompt import (
    ARM_ORDER,
    PROMPT_MARKERS,
    WARNING_TEXT,
    BuildTypoWarningSummaryConfig,
    TypoWarningSummaryInputError,
    WarningPromptArmScan,
    WarningPromptConfig,
    WarningPromptGeneration,
    WarningPromptInputUse,
    WarningPromptScan,
    WarningSourceBundle,
    WarningSourceCase,
    build_typo_warning_summary,
    build_warning_prompt_plan,
    exact_mcnemar,
    inject_typo_warning,
)
from typo_cot.experiments.typo_warning_prompt.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.typo_warning_prompt.protocol import PAPER_MODELS
from typo_cot.experiments.typo_warning_prompt.runtime import (
    HuggingFaceWarningPromptRuntime,
)
from typo_cot.experiments.typo_warning_prompt.scoring import extract_submitted_answer
from typo_cot.experiments.typo_warning_prompt.source import (
    DEFAULT_SUBMITTED_INPUTS,
    DEFAULT_SUBMITTED_INPUTS_SHA256,
    SourceFileSnapshot,
    canonical_sha256,
)
from typo_cot.experiments.typo_warning_prompt.submitted_inputs import (
    load_submitted_input_fixture,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _submitted_record(
    *,
    model: str,
    benchmark: str,
    sample_id: str,
    gold_answer: str,
) -> tuple[dict[str, object], str]:
    marker = PROMPT_MARKERS[benchmark]
    prompt = f"few-shot examples\n\n{marker}\n\nQuestion: typo {sample_id}\nReasoning:"
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    record = {
        "schema_version": "typo-warning-submitted-input/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "subset": "default",
        "gold_answer": gold_answer,
        "clean": {
            "prompt_sha256": "1" * 64,
            "editable_text_sha256": "2" * 64,
        },
        "edited": {
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
            "editable_text_sha256": "3" * 64,
        },
        "fixture_record_sha256": "4" * 64,
    }
    identity = canonical_sha256(
        {
            **record,
            "edited": {
                "prompt_sha256": prompt_sha,
                "editable_text_sha256": "3" * 64,
            },
        }
    )
    return record, identity


def _paper_source(
    *,
    model: str,
    benchmark: str,
    count: int = 300,
) -> WarningSourceBundle:
    fixture = load_submitted_input_fixture(DEFAULT_SUBMITTED_INPUTS)
    cases: list[WarningSourceCase] = []
    gold = "2" if benchmark == "gsm8k" else "A"
    for index in range(count):
        sample_id = f"{benchmark}_fixture_{index:04d}"
        record, record_sha = _submitted_record(
            model=model,
            benchmark=benchmark,
            sample_id=sample_id,
            gold_answer=gold,
        )
        cases.append(
            WarningSourceCase(
                sample_id=sample_id,
                submitted_record=record,
                submitted_record_sha256=record_sha,
                selector_record_sha256=canonical_sha256([sample_id, model, benchmark]),
            )
        )
    snapshot = SourceFileSnapshot(
        role="submitted-input-edit-manifest",
        path=DEFAULT_SUBMITTED_INPUTS,
        sha256=DEFAULT_SUBMITTED_INPUTS_SHA256,
    )
    revision = fixture.model_revisions[model]
    return WarningSourceBundle(
        model=model,
        benchmark=benchmark,
        model_revision=revision["revision"],
        model_revision_evidence=revision["evidence"],
        dataset_revision=fixture.dataset_revisions[benchmark],
        dataset_records_sha256=canonical_sha256([case.sample_id for case in cases]),
        dataset_sample_count=count,
        shared_sample_count=count,
        cases=tuple(cases),
        source_files=(snapshot,),
        source_sha256=canonical_sha256([case.identity for case in cases]),
        fixture_sha256=DEFAULT_SUBMITTED_INPUTS_SHA256,
        fixture_canonical_sha256=fixture.canonical_sha256,
        fixture_is_bundled_submitted_input=True,
        setting_records_sha256=canonical_sha256([case.identity for case in cases]),
        ordered_sample_ids_sha256=canonical_sha256([case.sample_id for case in cases]),
    )


def _input_use(arm: str, plan: object) -> WarningPromptInputUse:
    arm_plan = next(item for item in plan.arms if item.arm == arm)  # type: ignore[attr-defined]
    return WarningPromptInputUse(
        arm=arm,
        prompt_text_sha256=arm_plan.prompt_sha256,
        prompt_char_count=len(arm_plan.prompt),
        prompt_token_count=11,
        prompt_ids_sha256=("5" if arm == "without-warning" else "6") * 64,
    )


def _generation(
    *,
    answer: str,
    benchmark: str,
    gold: str,
) -> WarningPromptGeneration:
    text = f"The answer is {answer}."
    extraction = extract_submitted_answer(
        text,
        benchmark=benchmark,
        correct_answer=gold,
    )
    return WarningPromptGeneration(
        token_ids=(2,),
        text=text,
        value=extraction.value,
        is_extracted=extraction.is_extracted,
        is_correct=extraction.is_correct,
        method=extraction.method,
        primary_method=extraction.primary_method,
        stop_reason="eos_token",
        stop_token_id=2,
    )


def _record(
    *,
    case: WarningSourceCase,
    config: WarningPromptConfig,
    off: bool,
    on: bool,
) -> dict[str, object]:
    plan = build_warning_prompt_plan(
        case.submitted_record,
        source_record_sha256=case.submitted_record_sha256,
    )
    wrong = "3" if config.benchmark == "gsm8k" else "B"
    scan = WarningPromptScan(
        sample_id=case.sample_id,
        input_uses={arm: _input_use(arm, plan) for arm in ARM_ORDER},
        generations={
            "without-warning": _generation(
                answer=plan.gold_answer if off else wrong,
                benchmark=config.benchmark,
                gold=plan.gold_answer,
            ),
            "with-warning": _generation(
                answer=plan.gold_answer if on else wrong,
                benchmark=config.benchmark,
                gold=plan.gold_answer,
            ),
        },
    )
    return warning_runner._record(scan, case=case, plan=plan, config=config)


def _runtime_provenance(source: WarningSourceBundle, *, gpu_id: str = "0") -> dict[str, object]:
    return {
        "operation": "typo-warning-prompt",
        "runtime": "HuggingFaceWarningPromptRuntime",
        "python": "3.12.11",
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "model": source.model,
        "requested_revision": source.model_revision,
        "model_revision": source.model_revision,
        "model_revision_source": "model-config-metadata",
        "tokenizer_revision": source.model_revision,
        "tokenizer_revision_source": "tokenizer-init-metadata",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "cuda": "12.8",
        "cuda_visible_devices": gpu_id,
        "gpu_name": "Fixture GPU",
        "gpu_total_memory_bytes": 1024,
        "generation": dict(warning_runner._GENERATION),
        "implementation": warning_runner._IMPLEMENTATION,
        "implementation_code": implementation_code_identity(),
        "batching": dict(warning_runner._BATCHING),
        "answer_span_decoding": dict(warning_runner._ANSWER_SPAN_DECODING),
        "answer_extraction": warning_runner._ANSWER_EXTRACTION,
        "prompt_intervention": dict(warning_runner._PROMPT_INTERVENTION),
        "effective_eos_token_ids": [2],
        "effective_eos_token_ids_source": "fixture",
    }


def _outcomes(
    *,
    n: int,
    off_correct: int,
    on_correct: int,
    b10: int,
    b01: int,
) -> list[tuple[bool, bool]]:
    both = off_correct - b10
    neither = n - both - b10 - b01
    assert both >= 0 and neither >= 0 and both + b01 == on_correct
    return (
        [(True, True)] * both
        + [(True, False)] * b10
        + [(False, True)] * b01
        + [(False, False)] * neither
    )


def _write_completed_run(
    directory: Path,
    *,
    source: WarningSourceBundle,
    outcomes: Sequence[tuple[bool, bool]],
) -> None:
    directory.mkdir(parents=True)
    config = WarningPromptConfig(
        model=source.model,
        benchmark=source.benchmark,  # type: ignore[arg-type]
        output_dir=directory,
        gpu_id="0",
    )
    selected = source.cases
    rows = [
        _record(case=case, config=config, off=off, on=on)
        for case, (off, on) in zip(selected, outcomes, strict=True)
    ]
    records_path = directory / "warning_prompt_records.jsonl"
    _write_jsonl(records_path, rows)
    plan = warning_runner._plan_payload(source, selected, selected)
    summary = warning_runner._summary(
        rows,
        config=config,
        source=source,
        plan=plan,
    )
    summary_path = directory / "warning_prompt_summary.json"
    _write_json(summary_path, summary)
    outputs = {
        records_path.name: {
            "sha256": _sha256(records_path),
            "bytes": records_path.stat().st_size,
            "records": len(rows),
        },
        summary_path.name: {
            "sha256": _sha256(summary_path),
            "bytes": summary_path.stat().st_size,
            "records": 1,
        },
    }
    manifest = warning_runner._manifest(
        config=config,
        source=source,
        plan=plan,
        runtime=_runtime_provenance(source),
        status="completed",
        started_at="2026-08-09T00:00:00+00:00",
        completed_arms=len(rows) * 2,
        completed_pairs=len(rows),
        failures=(),
        checkpoints={},
        outputs=outputs,
    )
    _write_json(directory / "run.json", manifest)


def test_runner_and_runtime_share_one_protocol_constant_source() -> None:
    protocol = importlib.import_module("typo_cot.experiments.typo_warning_prompt.protocol")
    for protocol_name, runner_name, runtime_name in (
        ("ARM_ORDER", "ARM_ORDER", "ARM_ORDER"),
        ("GENERATION", "_GENERATION", "_GENERATION"),
        ("BATCHING", "_BATCHING", "_BATCHING"),
        ("ANSWER_EXTRACTION", "_ANSWER_EXTRACTION", "_ANSWER_EXTRACTION"),
        ("IMPLEMENTATION", "_IMPLEMENTATION", "_IMPLEMENTATION"),
    ):
        shared = getattr(protocol, protocol_name)
        assert getattr(warning_runner, runner_name) is shared
        assert getattr(warning_runtime, runtime_name) is shared


@pytest.mark.parametrize("benchmark", ("gsm8k", "mmlu"))
def test_warning_is_inserted_only_before_the_unique_final_marker(benchmark: str) -> None:
    marker = PROMPT_MARKERS[benchmark]
    prompt = f"few-shot\n\n{marker}\n\nQuestion: typo text\nReasoning:"

    warned = inject_typo_warning(prompt, benchmark=benchmark, enabled=True)

    prefix, suffix = prompt.split(marker)
    assert warned == prefix + WARNING_TEXT + "\n\n" + marker + suffix
    assert inject_typo_warning(prompt, benchmark=benchmark, enabled=False) == prompt


def test_warning_insertion_rejects_an_ambiguous_boundary() -> None:
    marker = PROMPT_MARKERS["gsm8k"]
    with pytest.raises(ValueError, match="exactly once"):
        inject_typo_warning(f"{marker} twice {marker}", benchmark="gsm8k", enabled=True)


def test_submitted_scorer_does_not_add_the_new_deterministic_fallback() -> None:
    extraction = extract_submitted_answer(
        "We computed 27 apples earlier, then stopped without a final answer.",
        benchmark="gsm8k",
        correct_answer="27",
    )

    assert extraction.value == ""
    assert extraction.is_extracted is False
    assert extraction.is_correct is False
    assert extraction.method == "no_match"


def test_bundled_fixture_has_the_six_submitted_settings_and_no_outputs() -> None:
    fixture = load_submitted_input_fixture(DEFAULT_SUBMITTED_INPUTS)

    assert fixture.sha256 == DEFAULT_SUBMITTED_INPUTS_SHA256
    assert len(fixture.settings) == 6
    assert all(setting.sample_count == 300 for setting in fixture.settings)
    assert fixture.provenance["contains_generated_outputs"] is False


def test_huggingface_runtime_generates_one_same_arm_batch_and_trims_eos(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        padding_side = "left"
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, value: object, **kwargs: object) -> dict[str, object]:
            texts = value if isinstance(value, list) else [value]
            rows = [[1, *[ord(character) for character in str(text)]] for text in texts]
            if not kwargs.get("padding"):
                return {"input_ids": rows[0]}
            width = max(map(len, rows))
            padded = [[0] * (width - len(row)) + row for row in rows]
            masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }

        def decode(self, token_ids: list[int], **_kwargs: object) -> str:
            visible = [token for token in token_ids if token != 99]
            return "The answer is 2." if visible[0] == 41 else "The answer is 3."

    class Model:
        def generate(self, *, input_ids: object, **kwargs: object) -> object:
            assert kwargs["max_new_tokens"] == 512
            assert kwargs["do_sample"] is False
            assert kwargs["eos_token_id"] == [99]
            suffix = torch.tensor([[41, 99, 0], [42, 43, 99]], device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    records = [
        _submitted_record(
            model=PAPER_MODELS[0],
            benchmark="gsm8k",
            sample_id=f"gsm8k_{index:05d}",
            gold_answer="2",
        )[0]
        for index in range(2)
    ]
    plans = [
        build_warning_prompt_plan(record, source_record_sha256=str(index + 1) * 64)
        for index, record in enumerate(records)
    ]
    runtime = object.__new__(HuggingFaceWarningPromptRuntime)
    runtime.config = WarningPromptConfig(
        model=PAPER_MODELS[0],
        benchmark="gsm8k",
        output_dir=tmp_path / "out",
    )
    runtime._torch = torch
    runtime.tokenizer = Tokenizer()
    runtime.model = Model()
    runtime.device = torch.device("cpu")
    runtime.effective_eos_token_ids = (99,)

    scans = runtime.scan_arm_batch(records, plans, arm="without-warning")

    assert [scan.sample_id for scan in scans] == [record["sample_id"] for record in records]
    assert scans[0].generation.token_ids == (41, 99)
    assert scans[1].generation.token_ids == (42, 43, 99)
    assert scans[0].generation.is_correct is True
    assert scans[1].generation.is_correct is False


def test_runtime_rejects_more_than_eight_or_unsorted_samples(tmp_path: Path) -> None:
    runtime = object.__new__(HuggingFaceWarningPromptRuntime)
    records = []
    plans = []
    for index in reversed(range(9)):
        record, _ = _submitted_record(
            model=PAPER_MODELS[0],
            benchmark="gsm8k",
            sample_id=f"gsm8k_{index:05d}",
            gold_answer="2",
        )
        records.append(record)
        plans.append(build_warning_prompt_plan(record, source_record_sha256="1" * 64))

    with pytest.raises(ValueError, match="batch exceeds"):
        runtime.scan_arm_batch(records, plans, arm="without-warning")
    with pytest.raises(ValueError, match="sorted"):
        runtime.scan_arm_batch(records[:2], plans[:2], arm="without-warning")


def test_catalog_exposes_the_submitted_input_command() -> None:
    spec = get_experiment("typo-warning-prompt")
    assert spec.status == "implemented"
    assert spec.required_arguments == ("--model", "--benchmark", "--output-dir")
    assert spec.outputs == (
        "warning_prompt_records.jsonl",
        "warning_prompt_summary.json",
        "run.json",
    )


def test_cli_parses_warning_setting_with_optional_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[WarningPromptConfig] = []
    result = SimpleNamespace(
        records_path=tmp_path / "out" / "warning_prompt_records.jsonl",
        summary_path=tmp_path / "out" / "warning_prompt_summary.json",
        run_path=tmp_path / "out" / "run.json",
        records=300,
    )
    monkeypatch.setattr(
        cli_module,
        "run_typo_warning_prompt",
        lambda config: captured.append(config) or result,
    )

    exit_code = main(
        [
            "typo-warning-prompt",
            "--model",
            PAPER_MODELS[0],
            "--benchmark",
            "gsm8k",
            "--input-fixture",
            "submitted.json",
            "--gpu-id",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert captured == [
        WarningPromptConfig(
            model=PAPER_MODELS[0],
            benchmark="gsm8k",
            input_fixture=Path("submitted.json"),
            output_dir=tmp_path / "out",
            gpu_id="1",
        )
    ]
    assert str(result.records_path) in capsys.readouterr().out


def test_cli_reports_warning_filesystem_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_config: WarningPromptConfig) -> object:
        raise OSError("source became unreadable")

    monkeypatch.setattr(cli_module, "run_typo_warning_prompt", fail)
    exit_code = main(
        [
            "typo-warning-prompt",
            "--model",
            PAPER_MODELS[0],
            "--benchmark",
            "gsm8k",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == ("typo-warning-prompt: error: source became unreadable\n")


def test_summary_builder_reproduces_the_printed_pooled_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        (model, benchmark): _paper_source(model=model, benchmark=benchmark)
        for model in PAPER_MODELS
        for benchmark in ("gsm8k", "mmlu")
    }

    def load(
        _fixture: Path | None,
        *,
        model: str,
        benchmark: str,
    ) -> WarningSourceBundle:
        return sources[(model, benchmark)]

    monkeypatch.setattr(warning_aggregation, "load_warning_sources", load)
    gsm = _outcomes(n=900, off_correct=541, on_correct=487, b10=121, b01=67)
    mmlu = _outcomes(n=900, off_correct=518, on_correct=506, b10=75, b01=63)
    runs = tmp_path / "runs"
    for benchmark, outcomes in (("gsm8k", gsm), ("mmlu", mmlu)):
        for index, model in enumerate(PAPER_MODELS):
            _write_completed_run(
                runs / model.rsplit("/", 1)[-1] / benchmark,
                source=sources[(model, benchmark)],
                outcomes=outcomes[index * 300 : (index + 1) * 300],
            )

    result = build_typo_warning_summary(
        BuildTypoWarningSummaryConfig(
            runs_root=runs,
            output_dir=tmp_path / "summary",
        )
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.settings == 6
    assert payload["pooled_by_benchmark"]["gsm8k"]["counts"] == {
        "without_warning_correct": 541,
        "with_warning_correct": 487,
    }
    assert payload["pooled_by_benchmark"]["gsm8k"]["mcnemar"] == exact_mcnemar(
        b10=121,
        b01=67,
    )
    assert payload["pooled_by_benchmark"]["mmlu"]["mcnemar"] == exact_mcnemar(
        b10=75,
        b01=63,
    )
    assert "one-pair-two-arm" not in payload["interpretation_limits"]
    assert "Model & Benchmark" in result.latex_path.read_text(encoding="utf-8")


def test_summary_builder_rejects_an_incomplete_grid_without_replacing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _paper_source(model=PAPER_MODELS[0], benchmark="gsm8k")
    monkeypatch.setattr(
        warning_aggregation,
        "load_warning_sources",
        lambda *_args, **_kwargs: source,
    )
    runs = tmp_path / "runs"
    _write_completed_run(
        runs / "one",
        source=source,
        outcomes=[(True, False)] * 300,
    )
    output = tmp_path / "summary"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_typo_warning_summary(BuildTypoWarningSummaryConfig(runs_root=runs, output_dir=output))

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_exact_mcnemar_uses_the_two_sided_binomial_probability() -> None:
    assert exact_mcnemar(b10=3, b01=0) == {
        "b10": 3,
        "b01": 0,
        "discordant": 3,
        "p_value": 0.25,
        "method": "exact-two-sided-mcnemar-binomial/v1",
    }
    assert exact_mcnemar(b10=0, b01=0)["p_value"] == 1.0


def test_summary_input_error_is_public() -> None:
    assert issubclass(TypoWarningSummaryInputError, ValueError)
    assert PAPER_SHA256 == "2cfb736e4636ee8db8dc6a92a6004c6e36914538a9acadcd66073289580a39d0"
    assert WarningPromptArmScan.__name__ == "WarningPromptArmScan"

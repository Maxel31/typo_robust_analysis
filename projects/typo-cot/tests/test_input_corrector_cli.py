"""CLI contracts for the Appendix E input-corrector producer and builder.

These tests intentionally precede the CLI implementation.  They keep argument
parsing and error reporting independent from the GPU runtime and from the
CPU-only aggregation internals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import typo_cot.cli as cli_module
from typo_cot.experiments.input_corrector_audit import (
    BuildInputCorrectorSummaryConfig,
    InputCorrectorAuditConfig,
    InputCorrectorAuditRunError,
    InputCorrectorSummaryInputError,
)


CORRECTORS = (
    "pyspellchecker",
    "t5-large-spell",
    "qwen2.5-7b-instruct",
)
BENCHMARKS = (
    "gsm8k",
    "mmlu",
    "mmlu-pro",
    "arc",
    "csqa",
    "math-500",
)
MODEL = "google/gemma-3-1b-it"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _command_parser(name: str) -> argparse.ArgumentParser:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices[name]


def _actions_by_destination(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions}


def _producer_argv(tmp_path: Path) -> list[str]:
    return [
        "input-corrector-audit",
        "--corrector",
        "t5-large-spell",
        "--model",
        MODEL,
        "--benchmark",
        "math-500",
        "--pairs",
        str(tmp_path / "pairs.jsonl"),
        "--gpu-id",
        "1",
        "--limit",
        "7",
        "--output-dir",
        str(tmp_path / "setting"),
        "--resume",
    ]


def _builder_argv(tmp_path: Path, *, include_math: bool = True) -> list[str]:
    argv = [
        "build-input-corrector-summary",
        "--runs-root",
        str(tmp_path / "core-runs"),
        "--output-dir",
        str(tmp_path / "published"),
    ]
    if include_math:
        argv[3:3] = ["--math-runs-root", str(tmp_path / "math-runs")]
    return argv


def test_both_public_readmes_prepare_the_complete_source_matrix_before_table12() -> None:
    for name in ("README.md", "README.ja.md"):
        readme = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        section = readme[readme.index("INPUT_CORRECTOR_SOURCE_BENCHMARKS=") :]
        assert "(gsm8k mmlu mmlu-pro arc csqa math-500)" in section
        assert section.index("typo-cot prepare-edited-pairs") < section.index(
            "typo-cot input-corrector-audit"
        )
        for argument in (
            "--targeting attribution-4",
            "--num-edits 4",
            "--seed 42",
            "--max-new-tokens 512",
            '--gpu-id "${GPU_ID}"',
        ):
            assert argument in section


def test_producer_parser_freezes_choices_types_and_defaults(tmp_path: Path) -> None:
    parser = _command_parser("input-corrector-audit")
    actions = _actions_by_destination(parser)

    assert tuple(actions["corrector"].choices) == CORRECTORS
    assert tuple(actions["benchmark"].choices) == BENCHMARKS
    assert actions["pairs"].type is Path
    assert actions["output_dir"].type is Path
    for required in ("corrector", "model", "benchmark", "pairs", "output_dir"):
        assert actions[required].required is True

    args = parser.parse_args(
        [
            "--corrector",
            "pyspellchecker",
            "--model",
            MODEL,
            "--benchmark",
            "gsm8k",
            "--pairs",
            str(tmp_path / "pairs.jsonl"),
            "--output-dir",
            str(tmp_path / "setting"),
        ]
    )

    assert args.pairs == tmp_path / "pairs.jsonl"
    assert args.output_dir == tmp_path / "setting"
    assert args.gpu_id == "0"
    assert args.limit is None
    assert args.resume is False


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--corrector", "unknown-corrector"),
        ("--benchmark", "unknown-benchmark"),
    ),
)
def test_producer_parser_rejects_values_outside_the_paper_contract(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    argv = [
        "--corrector",
        "pyspellchecker",
        "--model",
        MODEL,
        "--benchmark",
        "gsm8k",
        "--pairs",
        str(tmp_path / "pairs.jsonl"),
        "--output-dir",
        str(tmp_path / "setting"),
    ]
    argv[argv.index(option) + 1] = value

    with pytest.raises(SystemExit) as exc_info:
        _command_parser("input-corrector-audit").parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("limit", ("0", "-1"))
def test_producer_parser_requires_a_positive_limit(tmp_path: Path, limit: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _command_parser("input-corrector-audit").parse_args(
            [
                "--corrector",
                "pyspellchecker",
                "--model",
                MODEL,
                "--benchmark",
                "gsm8k",
                "--pairs",
                str(tmp_path / "pairs.jsonl"),
                "--limit",
                limit,
                "--output-dir",
                str(tmp_path / "setting"),
            ]
        )

    assert exc_info.value.code == 2


def test_builder_parser_has_an_optional_math_root_and_path_values(tmp_path: Path) -> None:
    parser = _command_parser("build-input-corrector-summary")
    actions = _actions_by_destination(parser)

    assert actions["runs_root"].required is True
    assert actions["math_runs_root"].required is False
    assert actions["output_dir"].required is True
    assert actions["runs_root"].type is Path
    assert actions["math_runs_root"].type is Path
    assert actions["output_dir"].type is Path

    args = parser.parse_args(
        [
            "--runs-root",
            str(tmp_path / "core-runs"),
            "--output-dir",
            str(tmp_path / "published"),
        ]
    )

    assert args.runs_root == tmp_path / "core-runs"
    assert args.math_runs_root is None
    assert args.output_dir == tmp_path / "published"


def test_producer_cli_constructs_exact_config_and_prints_all_result_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[InputCorrectorAuditConfig] = []
    output = tmp_path / "setting"
    result = SimpleNamespace(
        records=7,
        records_path=output / "corrector_records.jsonl",
        summary_path=output / "corrector_audit_summary.json",
        run_path=output / "run.json",
    )
    monkeypatch.setattr(
        cli_module,
        "run_input_corrector_audit",
        lambda config: captured.append(config) or result,
    )

    exit_code = cli_module.main(_producer_argv(tmp_path))

    assert exit_code == 0
    assert captured == [
        InputCorrectorAuditConfig(
            corrector="t5-large-spell",
            model=MODEL,
            benchmark="math-500",
            pairs=tmp_path / "pairs.jsonl",
            gpu_id="1",
            limit=7,
            output_dir=output,
            resume=True,
        )
    ]
    printed = capsys.readouterr().out
    for path in (result.records_path, result.summary_path, result.run_path):
        assert str(path) in printed


@pytest.mark.parametrize("include_math", (False, True), ids=("core-only", "with-math"))
def test_builder_cli_constructs_exact_config_and_prints_all_result_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    include_math: bool,
) -> None:
    captured: list[BuildInputCorrectorSummaryConfig] = []
    output = tmp_path / "published"
    result = SimpleNamespace(
        settings=75,
        summary_path=output / "input_corrector_summary.json",
        csv_path=output / "table12_input_correctors.csv",
        markdown_path=output / "table12_input_correctors.md",
        latex_path=output / "table12_input_correctors.tex",
        run_path=output / "run.json",
    )
    monkeypatch.setattr(
        cli_module,
        "run_build_input_corrector_summary",
        lambda config: captured.append(config) or result,
    )

    exit_code = cli_module.main(_builder_argv(tmp_path, include_math=include_math))

    assert exit_code == 0
    assert captured == [
        BuildInputCorrectorSummaryConfig(
            runs_root=tmp_path / "core-runs",
            math_runs_root=tmp_path / "math-runs" if include_math else None,
            output_dir=output,
        )
    ]
    printed = capsys.readouterr().out
    for path in (
        result.summary_path,
        result.csv_path,
        result.markdown_path,
        result.latex_path,
        result.run_path,
    ):
        assert str(path) in printed


@pytest.mark.parametrize(
    ("command", "error_type"),
    (
        ("producer", ValueError),
        ("producer", OSError),
        ("producer", InputCorrectorAuditRunError),
        ("builder", ValueError),
        ("builder", OSError),
        ("builder", InputCorrectorSummaryInputError),
    ),
    ids=(
        "producer-value",
        "producer-os",
        "producer-domain",
        "builder-value",
        "builder-os",
        "builder-domain",
    ),
)
def test_cli_converts_expected_failures_to_descriptive_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    error_type: type[Exception],
) -> None:
    message = f"synthetic {error_type.__name__} failure"

    def fail(_config: object) -> object:
        raise error_type(message)

    if command == "producer":
        monkeypatch.setattr(cli_module, "run_input_corrector_audit", fail)
        argv = _producer_argv(tmp_path)
        prefix = "input-corrector-audit"
    else:
        monkeypatch.setattr(cli_module, "run_build_input_corrector_summary", fail)
        argv = _builder_argv(tmp_path)
        prefix = "build-input-corrector-summary"

    exit_code = cli_module.main(argv)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == f"{prefix}: error: {message}\n"
    assert "Traceback" not in captured.err

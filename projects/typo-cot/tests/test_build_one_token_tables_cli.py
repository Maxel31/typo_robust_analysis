"""CLI contract tests for the CPU-only one-token table builder.

These tests intentionally precede the producer implementation.  They freeze the
README command shape and the fail-closed publication boundary without choosing
the internal aggregation API or importing the GPU/model runtime.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typo_cot.cli as cli_module
from typo_cot.cli import main
from typo_cot.experiments.catalog import get_experiment
from tests.test_build_one_token_tables_source import _write_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _command_parser() -> argparse.ArgumentParser:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices["build-one-token-tables"]


def test_catalog_exposes_the_table_builder_as_an_implemented_cpu_operation() -> None:
    """The artifact builder is a public CPU operation, not another GPU experiment."""
    spec = get_experiment("build-one-token-tables")

    assert spec.compute == "cpu"
    assert spec.status == "implemented"
    assert spec.target_command == (
        "uv run --project projects/typo-cot typo-cot build-one-token-tables"
    )
    assert spec.required_arguments == ("--runs-root", "--output-dir")
    assert spec.outputs == (
        "one_token_tables.json",
        "table10_one_token.csv",
        "table11_position_controls.csv",
        "one_token_tables.md",
        "one_token_tables.tex",
        "figure5_validation.json",
        "run.json",
    )


def test_parser_accepts_the_documented_table_builder_command() -> None:
    """Expected API: both README paths reach ``main`` as ``pathlib.Path`` values."""
    args = cli_module._parser().parse_args(
        [
            "build-one-token-tables",
            "--runs-root",
            "results/one-token-prefix-replacement",
            "--output-dir",
            "results/one-token-tables",
        ]
    )

    assert args.command == "build-one-token-tables"
    assert args.runs_root == Path("results/one-token-prefix-replacement")
    assert args.output_dir == Path("results/one-token-tables")


@pytest.mark.parametrize(
    "argv",
    (
        ["--output-dir", "results/one-token-tables"],
        ["--runs-root", "results/one-token-prefix-replacement"],
    ),
    ids=("missing-runs-root", "missing-output-dir"),
)
def test_parser_requires_both_input_and_output_directories(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _command_parser().parse_args(argv)

    assert exc_info.value.code == 2


def test_help_is_available_without_importing_model_libraries() -> None:
    """The completed CPU command must remain usable without the optional LRP stack."""
    script = r"""
import builtins

original_import = builtins.__import__
blocked = ("torch", "transformers", "tokenizers", "typo_cot.models", "typo_cot.lrp")

def reject_model_imports(name, *args, **kwargs):
    if any(name == root or name.startswith(root + ".") for root in blocked):
        raise RuntimeError(f"model library imported while rendering CPU help: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_model_imports

from typo_cot.cli import main

try:
    main(["build-one-token-tables", "--help"])
except SystemExit as exc:
    if exc.code != 0:
        raise
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(PROJECT_ROOT / "src"), environment.get("PYTHONPATH", "")),
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--runs-root" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--gpu-id" not in completed.stdout


def test_cli_refuses_to_overwrite_an_existing_output_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected result: exit 1, an intact directory, and no partial publication."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    output_dir = tmp_path / "published"
    output_dir.mkdir()
    sentinel = output_dir / "owned-by-user.txt"
    sentinel.write_text("do not replace\n", encoding="utf-8")

    result = main(
        [
            "build-one-token-tables",
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 1
    assert "already exists" in capsys.readouterr().err.lower()
    assert sentinel.read_text(encoding="utf-8") == "do not replace\n"
    assert set(output_dir.iterdir()) == {sentinel}


def test_documented_cli_executes_end_to_end_on_a_verified_partial_grid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "arbitrary-layout" / "one-setting")
    output_dir = tmp_path / "tables"

    result = main(
        [
            "build-one-token-tables",
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert "from 1 setting(s)" in capsys.readouterr().out
    assert (output_dir / "one_token_tables.json").is_file()
    assert (output_dir / "table10_one_token.csv").is_file()
    assert (output_dir / "table11_position_controls.csv").is_file()
    assert (output_dir / "one_token_tables.md").is_file()
    assert (output_dir / "one_token_tables.tex").is_file()
    assert (output_dir / "figure5_validation.json").is_file()
    assert (output_dir / "run.json").is_file()

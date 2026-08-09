"""CLI and bilingual command contracts for the Table 13 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import typo_cot.cli as cli_module
from typo_cot.experiments.restoration_order_accuracy import (
    BuildRestorationOrderTableConfig,
    RestorationOrderConfig,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    PAPER_BENCHMARKS,
    PAPER_BUDGETS,
    PAPER_ORDERS,
)


MODEL = "google/gemma-3-4b-it"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _command_parser(name: str) -> argparse.ArgumentParser:
    parser = cli_module._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices[name]


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions}


def _producer_argv(tmp_path: Path) -> list[str]:
    return [
        "restoration-order-accuracy",
        "--model",
        MODEL,
        "--benchmark",
        "gsm8k",
        "--pairs",
        str(tmp_path / "pairs.jsonl"),
        "--orders",
        *PAPER_ORDERS,
        "--budgets",
        *(str(value) for value in PAPER_BUDGETS),
        "--seed",
        "42",
        "--batch-size",
        "8",
        "--gpu-id",
        "1",
        "--limit",
        "3",
        "--output-dir",
        str(tmp_path / "setting"),
        "--resume",
    ]


def test_parser_exposes_exact_paper_grid_and_plural_order_budget_arguments(
    tmp_path: Path,
) -> None:
    parser = _command_parser("restoration-order-accuracy")
    actions = _actions(parser)
    assert tuple(actions["benchmark"].choices) == PAPER_BENCHMARKS
    assert tuple(actions["orders"].choices) == PAPER_ORDERS
    assert actions["orders"].nargs == "+"
    assert actions["budgets"].nargs == "+"
    assert actions["pairs"].type is Path
    assert actions["output_dir"].type is Path
    for name in ("orders", "budgets", "seed", "batch_size"):
        assert "paper" in str(actions[name].help).lower()
        assert "only" in str(actions[name].help).lower()

    args = parser.parse_args(_producer_argv(tmp_path)[1:])
    assert tuple(args.orders) == PAPER_ORDERS
    assert tuple(args.budgets) == PAPER_BUDGETS
    assert args.seed == 42
    assert args.batch_size == 8
    assert args.gpu_id == "1"
    assert args.limit == 3
    assert args.resume is True


def test_cli_builds_producer_and_builder_configs_and_prints_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    producer_configs: list[RestorationOrderConfig] = []
    producer_result = SimpleNamespace(
        records=3,
        records_path=tmp_path / "setting" / "restoration_order_records.jsonl",
        summary_path=tmp_path / "setting" / "restoration_order_summary.json",
        run_path=tmp_path / "setting" / "run.json",
    )
    monkeypatch.setattr(
        cli_module,
        "run_restoration_order_accuracy",
        lambda config: producer_configs.append(config) or producer_result,
    )
    assert cli_module.main(_producer_argv(tmp_path)) == 0
    assert producer_configs == [
        RestorationOrderConfig(
            model=MODEL,
            benchmark="gsm8k",
            pairs=tmp_path / "pairs.jsonl",
            orders=PAPER_ORDERS,
            budgets=PAPER_BUDGETS,
            seed=42,
            batch_size=8,
            gpu_id="1",
            limit=3,
            output_dir=tmp_path / "setting",
            resume=True,
        )
    ]
    assert str(producer_result.records_path) in capsys.readouterr().out

    builder_configs: list[BuildRestorationOrderTableConfig] = []
    table_dir = tmp_path / "table"
    builder_result = SimpleNamespace(
        settings=6,
        table_path=table_dir / "restoration_order_table.json",
        csv_path=table_dir / "table13_restoration_order.csv",
        markdown_path=table_dir / "table13_restoration_order.md",
        latex_path=table_dir / "table13_restoration_order.tex",
        run_path=table_dir / "run.json",
    )
    monkeypatch.setattr(
        cli_module,
        "build_restoration_order_table",
        lambda config: builder_configs.append(config) or builder_result,
    )
    assert (
        cli_module.main(
            [
                "build-restoration-order-table",
                "--runs-root",
                str(tmp_path / "runs"),
                "--output-dir",
                str(table_dir),
            ]
        )
        == 0
    )
    assert builder_configs == [
        BuildRestorationOrderTableConfig(
            runs_root=tmp_path / "runs",
            output_dir=table_dir,
        )
    ]
    printed = capsys.readouterr().out
    for path in (
        builder_result.table_path,
        builder_result.csv_path,
        builder_result.markdown_path,
        builder_result.latex_path,
        builder_result.run_path,
    ):
        assert str(path) in printed


def _table13_bash(readme: str) -> str:
    marker = "# Prepare the six complete Attribution-4 sources"
    marker_index = readme.index(marker)
    start = readme.rfind("```bash\n", 0, marker_index)
    end = readme.index("\n```", marker_index)
    assert start >= 0
    return readme[start + len("```bash\n") : end]


def test_english_and_japanese_readmes_publish_the_same_complete_grid_command() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")
    english_block = _table13_bash(english)
    japanese_block = _table13_bash(japanese)
    assert english_block == japanese_block
    for token in (
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "RESTORATION_BENCHMARKS=(gsm8k mmlu)",
        "--orders high-relevance-first seeded-random low-relevance-first",
        "--budgets 0 1 2 3 4",
        "--seed 42",
        "--batch-size 8",
        "build-restoration-order-table",
    ):
        assert token in english_block


def test_cli_reports_producer_or_builder_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "run_restoration_order_accuracy",
        lambda config: (_ for _ in ()).throw(ValueError("bad producer")),
    )
    assert cli_module.main(_producer_argv(tmp_path)) == 1
    assert "restoration-order-accuracy: error: bad producer" in capsys.readouterr().err

    monkeypatch.setattr(
        cli_module,
        "build_restoration_order_table",
        lambda config: (_ for _ in ()).throw(ValueError("bad builder")),
    )
    assert (
        cli_module.main(
            [
                "build-restoration-order-table",
                "--runs-root",
                str(tmp_path / "runs"),
                "--output-dir",
                str(tmp_path / "table"),
            ]
        )
        == 1
    )
    assert "build-restoration-order-table: error: bad builder" in capsys.readouterr().err

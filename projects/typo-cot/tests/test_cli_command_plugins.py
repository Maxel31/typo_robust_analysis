"""Optional command-package discovery contracts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from typo_cot import cli as cli_module


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    registrar: Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None]

    def load(
        self,
    ) -> Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None]:
        return self.registrar


def test_optional_command_plugin_is_loaded_and_dispatched(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def register(
        commands: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        parser = commands.add_parser("fixture-plugin")
        parser.add_argument("--value", required=True)

        def handle(args: argparse.Namespace) -> int:
            calls.append(args.value)
            return 7

        parser.set_defaults(_typo_cot_plugin_handler=handle)

    monkeypatch.setattr(
        cli_module.importlib.metadata,
        "entry_points",
        lambda *, group: [_EntryPoint(name="fixture", registrar=register)],
    )

    assert cli_module.main(["fixture-plugin", "--value", "observed"]) == 7
    assert calls == ["observed"]

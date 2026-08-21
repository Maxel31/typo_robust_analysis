"""Command-line interface for the standalone UCC-Inj reproductions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .exp6 import Exp6Config, run_exp6, write_exp6_results


def _load_config(path: Path) -> Exp6Config:
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("config must be a YAML mapping")
    section = loaded.get("exp6", loaded)
    if not isinstance(section, dict):
        raise TypeError("exp6 config must be a mapping")
    if "noise_levels" in section:
        section = {**section, "noise_levels": tuple(section["noise_levels"])}
    return Exp6Config(**section)


def _exp6(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    if args.model is not None:
        config = Exp6Config(**{**config.__dict__, "model": args.model})
    if args.limit is not None:
        config = Exp6Config(**{**config.__dict__, "limit": args.limit})
    if args.device is not None:
        config = Exp6Config(**{**config.__dict__, "device": args.device})
    records, summary = run_exp6(config, progress=lambda values: tqdm(values, desc="exp6"))
    write_exp6_results(output_dir=Path(args.output_dir), config=config, records=records, summary=summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    exp6 = subcommands.add_parser("exp6-cosine", help="run UCC-Inj experiment 6")
    exp6.add_argument("--config", required=True, help="YAML configuration")
    exp6.add_argument("--output-dir", required=True, help="new result directory")
    exp6.add_argument("--model", help="override config.model")
    exp6.add_argument("--device", help="override config.device, e.g. cuda:0")
    exp6.add_argument("--limit", type=int, help="run a deterministic prefix for a smoke test")
    exp6.set_defaults(handler=_exp6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

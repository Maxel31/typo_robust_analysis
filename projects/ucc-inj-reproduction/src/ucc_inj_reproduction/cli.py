"""Command-line interface for standalone UCC-Inj protocol adaptations."""

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
    if "exp6" in loaded:
        unexpected = sorted(set(loaded) - {"exp6"})
        if unexpected:
            raise ValueError(f"unexpected top-level config keys outside exp6: {unexpected}")
        section = loaded["exp6"]
    else:
        section = loaded
    if not isinstance(section, dict):
        raise TypeError("exp6 config must be a mapping")
    if "noise_levels" in section:
        section = {**section, "noise_levels": tuple(section["noise_levels"])}
    config = Exp6Config(**section)
    config.validate()
    return config


def _exp6(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    config = _load_config(Path(args.config))
    if args.limit is not None:
        config = Exp6Config(**{**config.__dict__, "limit": args.limit})
    if args.device is not None:
        config = Exp6Config(**{**config.__dict__, "device": args.device})
    config.validate()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"output path is already occupied: {output_dir}") from error
    completed = False
    try:
        records, summary, provenance = run_exp6(
            config,
            progress=lambda values: tqdm(values, desc="exp6"),
        )
        write_exp6_results(
            output_dir=output_dir,
            config=config,
            records=records,
            summary=summary,
            provenance=provenance,
            reserved_output_dir=True,
        )
        completed = True
    finally:
        if not completed:
            try:
                output_dir.rmdir()
            except OSError:
                pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    exp6 = subcommands.add_parser("exp6-cosine", help="run the UCC-Inj exp6 adaptation")
    exp6.add_argument("--config", required=True, help="YAML configuration")
    exp6.add_argument("--output-dir", required=True, help="new result directory")
    exp6.add_argument("--device", help="override config.device, e.g. cuda:0")
    exp6.add_argument("--limit", type=int, help="run a deterministic prefix for a smoke test")
    exp6.set_defaults(handler=_exp6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

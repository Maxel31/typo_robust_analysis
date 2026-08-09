"""Atomic CPU runner for the one-token paper artifact bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typo_cot.experiments.build_one_token_tables.aggregation import (
    assemble_one_token_tables,
)
from typo_cot.experiments.build_one_token_tables.protocol import (
    ANALYSIS_PROTOCOL,
    OUTPUT_NAMES,
    InferenceProtocol,
    PUBLIC_INFERENCE,
)
from typo_cot.experiments.build_one_token_tables.render import render_artifacts
from typo_cot.experiments.build_one_token_tables.source import (
    ValidatedOneTokenRun,
    discover_and_validate_runs,
)
from typo_cot.experiments.catalog import PAPER_SHA256


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class BuildOneTokenTablesConfig:
    """Paths for the deterministic CPU artifact build."""

    runs_root: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        runs = self.runs_root.resolve()
        output = self.output_dir.resolve()
        if output == runs or output in runs.parents or runs in output.parents:
            raise ValueError("runs_root and output_dir must be separate directory trees")


@dataclass(frozen=True, slots=True)
class BuildOneTokenTablesResult:
    output_dir: Path
    tables_path: Path
    table10_csv_path: Path
    table11_csv_path: Path
    markdown_path: Path
    latex_path: Path
    figure5_path: Path
    run_path: Path
    settings: int


def _input_inventory(runs: tuple[ValidatedOneTokenRun, ...]) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for run in runs:
        inventory.append(
            {
                "setting_id": run.setting_id,
                "model": run.model,
                "benchmark": run.benchmark,
                "cohort": run.cohort,
                "position_controls": list(run.position_controls),
                "run_dir": str(run.run_dir),
                "run_manifest_sha256": run.manifest_sha256,
                "producer_code_identity": dict(run.producer_code_identity),
                "files": {name: dict(metadata) for name, metadata in run.input_files.items()},
                "records": len(run.records),
            }
        )
    return sorted(inventory, key=lambda item: str(item["setting_id"]))


def run_build_one_token_tables(
    config: BuildOneTokenTablesConfig,
    *,
    inference: InferenceProtocol = PUBLIC_INFERENCE,
) -> BuildOneTokenTablesResult:
    """Validate all inputs, render privately, and atomically publish the bundle."""

    if not isinstance(config, BuildOneTokenTablesConfig):
        raise TypeError("config must be BuildOneTokenTablesConfig")
    if not isinstance(inference, InferenceProtocol):
        raise TypeError("inference must be InferenceProtocol")
    output = config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    runs = discover_and_validate_runs(config.runs_root)
    inventory = _input_inventory(runs)
    analysis = assemble_one_token_tables(runs, inference=inference)
    analysis = {
        **analysis,
        "paper_sha256": PAPER_SHA256,
        "analysis_protocol": {
            **ANALYSIS_PROTOCOL,
            "inference": inference.to_dict(),
        },
        "inputs": inventory,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        render_artifacts(temporary, analysis)
        expected = set(OUTPUT_NAMES)
        present = {path.name for path in temporary.iterdir()}
        if present != expected:
            raise RuntimeError(
                "renderer output set is incomplete: "
                f"expected {sorted(expected)}, found {sorted(present)}"
            )
        outputs = {
            name: {
                "sha256": _file_sha256(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in OUTPUT_NAMES
        }
        now = datetime.now(UTC).isoformat()
        manifest: Mapping[str, object] = {
            "schema_version": "build-one-token-tables-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "build-one-token-tables",
            "status": "completed",
            "arguments": {
                "runs_root": str(config.runs_root.resolve()),
                "output_dir": str(output),
            },
            "analysis_protocol": {
                **ANALYSIS_PROTOCOL,
                "inference": inference.to_dict(),
            },
            "analysis_protocol_sha256": _canonical_sha256(
                {**ANALYSIS_PROTOCOL, "inference": inference.to_dict()}
            ),
            "inputs": inventory,
            "coverage": analysis["coverage"],
            "outputs": outputs,
            "started_at": now,
            "updated_at": now,
        }
        _write_json(temporary / "run.json", manifest)
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        os.rename(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return BuildOneTokenTablesResult(
        output_dir=output,
        tables_path=output / "one_token_tables.json",
        table10_csv_path=output / "table10_one_token.csv",
        table11_csv_path=output / "table11_position_controls.csv",
        markdown_path=output / "one_token_tables.md",
        latex_path=output / "one_token_tables.tex",
        figure5_path=output / "figure5_validation.json",
        run_path=output / "run.json",
        settings=len(runs),
    )


__all__ = [
    "BuildOneTokenTablesConfig",
    "BuildOneTokenTablesResult",
    "run_build_one_token_tables",
]

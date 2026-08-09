"""Atomic CPU runner for Appendix C/Table 8."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.edit_count_sensitivity.aggregation import build_analysis
from typo_cot.experiments.edit_count_sensitivity.protocol import (
    ANALYSIS_PROTOCOL,
    ANALYSIS_PROTOCOL_SHA256,
)
from typo_cot.experiments.edit_count_sensitivity.render import render_artifacts
from typo_cot.experiments.edit_count_sensitivity.source import (
    discover_cot_swap_runs,
    discover_prepared_runs,
)

_OUTPUT_NAMES = (
    "edit_count_records.jsonl",
    "edit_count_summary.json",
    "table8_edit_count.csv",
    "table8_edit_count.md",
    "table8_edit_count.tex",
)


@dataclass(frozen=True, slots=True)
class EditCountSensitivityConfig:
    """Public arguments for the Table 8 artifact builder."""

    pairs_root: Path
    cot_swap_runs_root: Path
    edit_counts: tuple[int, ...]
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs_root", Path(self.pairs_root))
        object.__setattr__(self, "cot_swap_runs_root", Path(self.cot_swap_runs_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "edit_counts", tuple(self.edit_counts))
        if not self.edit_counts:
            raise ValueError("edit_counts must not be empty")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count not in {1, 2, 4}
            for count in self.edit_counts
        ):
            raise ValueError("edit_counts values must be 1, 2, or 4")
        if len(set(self.edit_counts)) != len(self.edit_counts):
            raise ValueError("edit_counts must be unique")
        if tuple(sorted(self.edit_counts)) != self.edit_counts:
            raise ValueError("edit_counts must be in ascending order")
        pairs = self.pairs_root.resolve()
        cot = self.cot_swap_runs_root.resolve()
        output = self.output_dir.resolve()
        if pairs == cot:
            raise ValueError("pairs_root and cot_swap_runs_root must be distinct")
        for source in (pairs, cot):
            if output == source or source in output.parents:
                raise ValueError("output_dir must be separate from both input trees")

    def public_arguments(self) -> dict[str, object]:
        return {
            "pairs_root": str(self.pairs_root.resolve()),
            "cot_swap_runs_root": str(self.cot_swap_runs_root.resolve()),
            "edit_counts": list(self.edit_counts),
            "output_dir": str(self.output_dir.resolve()),
        }


@dataclass(frozen=True, slots=True)
class EditCountSensitivityResult:
    """Published Table 8 artifact paths and validated coverage counts."""

    output_dir: Path
    records_path: Path
    summary_path: Path
    csv_path: Path
    markdown_path: Path
    latex_path: Path
    run_path: Path
    accuracy_settings: int
    restoration_settings: int


def _serialized_json(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(_serialized_json(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _code_identity() -> dict[str, object]:
    package = Path(__file__).resolve().parent
    files = sorted(package.glob("*.py"))
    digest = hashlib.sha256()
    entries: list[dict[str, object]] = []
    for path in files:
        relative = path.name
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return {
        "algorithm": "sorted-relative-path-and-file-sha256/v1",
        "python_file_count": len(entries),
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def _input_manifest(prepared: object, cot_swap: object) -> dict[str, object]:
    return {
        "prepared_runs": [
            {
                "model": run.model,
                "benchmark": run.benchmark,
                "edit_count": run.edit_count,
                "run": str(run.run_path),
                "run_sha256": run.run_sha256,
                "pairs": str(run.pairs_path),
                "pairs_sha256": run.pairs_sha256,
                "records": len(run.records),
            }
            for run in prepared  # type: ignore[union-attr]
        ],
        "cot_swap_runs": [
            {
                "model": run.model,
                "benchmark": run.benchmark,
                "edit_count": run.edit_count,
                "run": str(run.run_path),
                "run_sha256": run.run_sha256,
                "records": str(run.records_path),
                "records_sha256": run.records_sha256,
                "record_count": len(run.records),
            }
            for run in cot_swap  # type: ignore[union-attr]
        ],
    }


def run_edit_count_sensitivity(
    config: EditCountSensitivityConfig,
) -> EditCountSensitivityResult:
    """Validate producers, recompute Table 8, and publish exactly once."""
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    prepared = discover_prepared_runs(config.pairs_root, edit_counts=config.edit_counts)
    cot_swap = discover_cot_swap_runs(
        config.cot_swap_runs_root,
        edit_counts=config.edit_counts,
        prepared_runs=prepared,
    )
    rows, summary = build_analysis(
        prepared,
        cot_swap,
        edit_counts=config.edit_counts,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        _write_jsonl(staging / "edit_count_records.jsonl", rows)
        _write_json(staging / "edit_count_summary.json", summary)
        render_artifacts(staging, summary)
        outputs = {name: _metadata(staging / name) for name in _OUTPUT_NAMES}
        coverage = summary["coverage"]
        if not isinstance(coverage, dict):
            raise AssertionError("analysis coverage must be a JSON object")
        accuracy_coverage = coverage["accuracy"]
        restoration_coverage = coverage["restoration"]
        if not isinstance(accuracy_coverage, dict) or not isinstance(restoration_coverage, dict):
            raise AssertionError("analysis coverage sections must be JSON objects")
        manifest = {
            "schema_version": "edit-count-sensitivity-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "edit-count-sensitivity",
            "status": "completed",
            "arguments": config.public_arguments(),
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "analysis_protocol_sha256": ANALYSIS_PROTOCOL_SHA256,
            "implementation": _code_identity(),
            "inputs": _input_manifest(prepared, cot_swap),
            "counts": {
                "validated_prepare_runs": len(prepared),
                "validated_cot_swap_runs": len(cot_swap),
                "accuracy_settings": accuracy_coverage["complete_setting_count"],
                "restoration_settings": restoration_coverage["complete_setting_count"],
                "records": len(rows),
            },
            "comparability": summary["comparability"],
            "outputs": outputs,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(staging / "run.json", manifest)
        if output_dir.exists():
            raise FileExistsError(f"output directory appeared during publication: {output_dir}")
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    accuracy_settings = int(accuracy_coverage["complete_setting_count"])
    restoration_settings = int(restoration_coverage["complete_setting_count"])
    return EditCountSensitivityResult(
        output_dir=output_dir,
        records_path=output_dir / "edit_count_records.jsonl",
        summary_path=output_dir / "edit_count_summary.json",
        csv_path=output_dir / "table8_edit_count.csv",
        markdown_path=output_dir / "table8_edit_count.md",
        latex_path=output_dir / "table8_edit_count.tex",
        run_path=output_dir / "run.json",
        accuracy_settings=accuracy_settings,
        restoration_settings=restoration_settings,
    )


__all__ = [
    "EditCountSensitivityConfig",
    "EditCountSensitivityResult",
    "run_edit_count_sensitivity",
]

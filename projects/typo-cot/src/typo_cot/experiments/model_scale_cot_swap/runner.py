"""Atomic CPU artifact builder for Appendix C/Table 9."""

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
from typo_cot.experiments.model_scale_cot_swap.aggregation import build_analysis
from typo_cot.experiments.model_scale_cot_swap.protocol import (
    ANALYSIS_PROTOCOL,
    ANALYSIS_PROTOCOL_SHA256,
)
from typo_cot.experiments.model_scale_cot_swap.render import render_artifacts
from typo_cot.experiments.model_scale_cot_swap.source import (
    ModelScaleInputs,
    discover_model_scale_inputs,
)

_OUTPUT_NAMES = (
    "model_scale_records.jsonl",
    "model_scale_summary.json",
    "table9_model_scale.csv",
    "table9_model_scale.md",
    "table9_model_scale.tex",
)


@dataclass(frozen=True, slots=True)
class ModelScaleCotSwapConfig:
    """Public arguments for the Table 9 CPU builder."""

    pairs_root: Path
    cot_swap_runs_root: Path
    cohort: Path
    output_dir: Path

    def __post_init__(self) -> None:
        for field in ("pairs_root", "cot_swap_runs_root", "cohort", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        pairs = self.pairs_root.resolve()
        cot = self.cot_swap_runs_root.resolve()
        cohort = self.cohort.resolve()
        output = self.output_dir.resolve()
        if pairs == cot:
            raise ValueError("pairs_root and cot_swap_runs_root must be distinct")
        for source in (pairs, cot):
            if output == source or source in output.parents:
                raise ValueError("output_dir must be separate from both producer trees")
        if output == cohort or output in cohort.parents:
            raise ValueError("output_dir must not replace the cohort artifact")

    def public_arguments(self) -> dict[str, object]:
        return {
            "pairs_root": str(self.pairs_root.resolve()),
            "cot_swap_runs_root": str(self.cot_swap_runs_root.resolve()),
            "cohort": str(self.cohort.resolve()),
            "output_dir": str(self.output_dir.resolve()),
        }


@dataclass(frozen=True, slots=True)
class ModelScaleCotSwapResult:
    """Published Table 9 artifact paths and coverage count."""

    output_dir: Path
    records_path: Path
    summary_path: Path
    csv_path: Path
    markdown_path: Path
    latex_path: Path
    run_path: Path
    settings: int


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
    package = Path(__file__).resolve().parents[2]
    paths = sorted(
        package.rglob("*.py"),
        key=lambda path: path.relative_to(package).as_posix(),
    )
    digest = hashlib.sha256()
    files: list[dict[str, object]] = []
    for path in paths:
        relative_path = path.relative_to(package).as_posix()
        file_hash = _sha256(path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": file_hash,
            }
        )
    return {
        "algorithm": "sorted-typo-cot-python-relative-path-and-file-sha256/v1",
        "python_file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _input_manifest(inputs: ModelScaleInputs) -> dict[str, object]:
    cohort_payload = inputs.cohort.to_dict()
    cohort_payload.pop("sample_ids", None)
    return {
        "cohort": {
            "path": str(inputs.cohort.path),
            "artifact_sha256": inputs.cohort.artifact_sha256,
            **cohort_payload,
        },
        "prepared_runs": [
            {
                "model": run.model,
                "run": str(run.run_path),
                "run_sha256": run.run_sha256,
                "pairs": str(run.pairs_path),
                "pairs_sha256": run.pairs_sha256,
                "records": len(run.records),
            }
            for run in inputs.prepared_runs
        ],
        "cot_swap_runs": [
            {
                "model": run.model,
                "run": str(run.run_path),
                "run_sha256": run.run_sha256,
                "records": str(run.records_path),
                "records_sha256": run.records_sha256,
                "record_count": len(run.records),
            }
            for run in inputs.cot_swap_runs
        ],
    }


def run_model_scale_cot_swap(
    config: ModelScaleCotSwapConfig,
) -> ModelScaleCotSwapResult:
    """Validate producers, recompute Table 9, and publish exactly once."""
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    inputs = discover_model_scale_inputs(
        pairs_root=config.pairs_root,
        cot_swap_runs_root=config.cot_swap_runs_root,
        cohort_path=config.cohort,
    )
    rows, summary = build_analysis(inputs.cot_swap_runs)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        _write_jsonl(staging / "model_scale_records.jsonl", rows)
        _write_json(staging / "model_scale_summary.json", summary)
        render_artifacts(staging, rows, summary)
        outputs = {name: _metadata(staging / name) for name in _OUTPUT_NAMES}
        manifest = {
            "schema_version": "model-scale-cot-swap-run/v1",
            "paper_sha256": PAPER_SHA256,
            "operation": "model-scale-cot-swap",
            "status": "completed",
            "arguments": config.public_arguments(),
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "analysis_protocol_sha256": ANALYSIS_PROTOCOL_SHA256,
            "implementation": _code_identity(),
            "inputs": _input_manifest(inputs),
            "counts": {
                "validated_prepare_runs": len(inputs.prepared_runs),
                "validated_cot_swap_runs": len(inputs.cot_swap_runs),
                "settings": len(rows),
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

    return ModelScaleCotSwapResult(
        output_dir=output_dir,
        records_path=output_dir / "model_scale_records.jsonl",
        summary_path=output_dir / "model_scale_summary.json",
        csv_path=output_dir / "table9_model_scale.csv",
        markdown_path=output_dir / "table9_model_scale.md",
        latex_path=output_dir / "table9_model_scale.tex",
        run_path=output_dir / "run.json",
        settings=len(rows),
    )


__all__ = [
    "ModelScaleCotSwapConfig",
    "ModelScaleCotSwapResult",
    "run_model_scale_cot_swap",
]

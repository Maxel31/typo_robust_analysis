"""Strict, resumable writer for one layerwise KL-patching setting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.prepare_edited_pairs.runner import (
    PUBLIC_BENCHMARKS,
    TARGETING_CONDITIONS,
)

DIRECTION_NAMES = ("clean-to-edited", "edited-to-clean")
_DIRECTION_ORDER = {name: index for index, name in enumerate(DIRECTION_NAMES)}
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_PROTOCOL = {
    "schema_version": "layerwise-kl-patching-protocol/v1",
    "source_cohort": "clean-correct-edited-wrong",
    "intervention_site": "complete-decoder-block-residual-output",
    "intervention_positions": "all-aligned-edited-word-final-tokens",
    "readout": "prompt-final-logits-predicting-first-cot-token",
    "kl_orientation": "reference-to-recipient",
    "denominator_epsilon": 1e-9,
    "layer_grid": "all-decoder-layers-0-through-L-minus-1",
    "relative_depth": "layer-index-divided-by-L",
    "third_membership": "layer-center-(layer-plus-0.5)-divided-by-L",
    "layer_profile": "across-pair-median",
    "third_aggregation": "pair-layer-mean-then-setting-pair-median",
    "mcb": {
        "method": "paired-bootstrap-median-difference-to-observed-peak",
        "resamples": 2000,
        "seed": 42,
        "one_sided_upper_percentile": 95,
    },
    "dtype": "bfloat16",
    "use_cache": False,
}


class LayerwiseKLPatchingRunError(RuntimeError):
    """Raised after one or more runtime pairs fail, preserving checkpoints."""


@dataclass(frozen=True, slots=True)
class LayerwiseKLPatchingConfig:
    """Frozen public arguments for one model/benchmark/targeting setting."""

    model: str
    benchmark: str
    pairs: Path
    targeting: Literal["attribution-4", "random-4"]
    directions: tuple[str, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", Path(self.pairs))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in PUBLIC_BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {self.benchmark!r}")
        if self.targeting not in TARGETING_CONDITIONS:
            raise ValueError(f"unsupported targeting condition: {self.targeting!r}")
        if not self.directions:
            raise ValueError("directions must contain at least one direction")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("directions must not contain duplicates")
        unknown = [name for name in self.directions if name not in _DIRECTION_ORDER]
        if unknown:
            raise ValueError(f"unsupported direction: {unknown[0]!r}")
        object.__setattr__(
            self,
            "directions",
            tuple(sorted(self.directions, key=_DIRECTION_ORDER.__getitem__)),
        )
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""
        payload = asdict(self)
        payload["pairs"] = str(self.pairs.resolve())
        payload["output_dir"] = str(self.output_dir.resolve())
        payload["directions"] = list(self.directions)
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class DirectionScan:
    """Untreated denominator and one patched KL numerator per decoder layer."""

    denominator_kl: float
    patched_kl_by_layer: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairScan:
    """Runtime result for every requested direction of one source pair."""

    sample_id: str
    directions: Mapping[str, DirectionScan]


class LayerwiseKLPatchingRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(
        self, pair: dict[str, object], directions: tuple[str, ...]
    ) -> PairScan: ...


@dataclass(frozen=True, slots=True)
class LayerwiseKLPatchingResult:
    """Published paths and record counts for a completed setting."""

    layer_records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    included_grids: int
    layer_records: int


@dataclass(frozen=True, slots=True)
class _SourcePair:
    sample_id: str
    record: dict[str, object]
    fingerprint: str
    clean_correct: bool
    edited_wrong: bool
    aligned_count: int


@dataclass(frozen=True, slots=True)
class _Source:
    pairs: tuple[_SourcePair, ...]
    pairs_sha256: str
    run_sha256: str
    manifest: dict[str, object]
    model_revision: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_fingerprint(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_loads(text: str, *, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc


def _load_json(path: Path) -> dict[str, object]:
    payload = _strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _token_indices(
    word: Mapping[str, object],
    *,
    side: str,
    prompt_token_count: int,
    context: str,
) -> int:
    field = f"{side}_token_indices"
    raw_indices = word.get(field)
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError(f"{context}.{field} must be a non-empty list")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in raw_indices
    ):
        raise ValueError(f"{context}.{field} must contain non-negative integers")
    final_field = f"{side}_final_token"
    final = word.get(final_field)
    if not isinstance(final, int) or isinstance(final, bool) or final != raw_indices[-1]:
        raise ValueError(f"{context}.{final_field} must equal the final token-index entry")
    if final >= prompt_token_count:
        raise ValueError(f"{context}.{final_field} is outside the recorded prompt")
    return final


def _validate_pair_record(
    record: dict[str, object],
    *,
    config: LayerwiseKLPatchingConfig,
    line_number: int,
) -> tuple[str, bool, bool, int]:
    context = f"{config.pairs}:{line_number}"
    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise ValueError(f"{context}: unknown pair schema")
    sample_id = _nonempty_string(record.get("sample_id"), field=f"{context}.sample_id")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", config.targeting),
    ):
        if record.get(field) != expected:
            raise ValueError(f"{context}: record {field} does not match --{field}")
    if record.get("seed") != 42:
        raise ValueError(f"{context}: layerwise paper input requires seed 42")
    if record.get("num_edits_requested") != 4:
        raise ValueError(f"{context}: layerwise paper input requires four edits")

    clean = _mapping(record.get("clean"), field=f"{context}.clean")
    edited = _mapping(record.get("edited"), field=f"{context}.edited")
    prompt_counts: dict[str, int] = {}
    correctness: dict[str, bool] = {}
    for side, payload in (("clean", clean), ("edited", edited)):
        _nonempty_string(payload.get("prompt"), field=f"{context}.{side}.prompt")
        prompt_counts[side] = _positive_int(
            payload.get("prompt_token_count"),
            field=f"{context}.{side}.prompt_token_count",
        )
        answer = _mapping(payload.get("answer"), field=f"{context}.{side}.answer")
        is_correct = answer.get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(f"{context}.{side}.answer.is_correct must be boolean")
        correctness[side] = is_correct

    aligned_words = record.get("aligned_words")
    if not isinstance(aligned_words, list):
        raise ValueError(f"{context}.aligned_words must be a list")
    if record.get("num_aligned_words") != len(aligned_words):
        raise ValueError(f"{context}.num_aligned_words does not match aligned_words")
    final_positions: dict[str, list[int]] = {"clean": [], "edited": []}
    for index, raw_word in enumerate(aligned_words):
        word = _mapping(raw_word, field=f"{context}.aligned_words[{index}]")
        for side in ("clean", "edited"):
            final_positions[side].append(
                _token_indices(
                    word,
                    side=side,
                    prompt_token_count=prompt_counts[side],
                    context=f"{context}.aligned_words[{index}]",
                )
            )
    for side, positions in final_positions.items():
        if len(positions) != len(set(positions)):
            raise ValueError(f"{context}: duplicate {side} aligned final-token coordinate")

    return sample_id, correctness["clean"], not correctness["edited"], len(aligned_words)


def _load_source(config: LayerwiseKLPatchingConfig) -> _Source:
    pairs_path = config.pairs
    if not pairs_path.is_file():
        raise ValueError(f"pairs input is not a file: {pairs_path}")
    run_path = pairs_path.parent / "run.json"
    if not run_path.is_file():
        raise ValueError(f"pairs input is missing sibling run.json: {run_path}")
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise ValueError("source run has an unknown schema")
    if manifest.get("operation") != "prepare-edited-pairs":
        raise ValueError("source run has the wrong operation")
    if manifest.get("status") != "completed":
        raise ValueError("source run is not completed")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("source run paper SHA-256 does not match the final paper")
    arguments = _mapping(manifest.get("arguments"), field="source run arguments")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", config.targeting),
    ):
        if arguments.get(field) != expected:
            raise ValueError(f"source run {field} does not match the requested setting")
    if arguments.get("seed") != 42:
        raise ValueError("source run must use paper seed 42")
    if arguments.get("num_edits") != 4:
        raise ValueError("source run must request four edits")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("source pair preparation must not be limited")
    if arguments.get("max_new_tokens") != 512:
        raise ValueError("source run must use the paper generation cap 512")
    provenance = _mapping(manifest.get("provenance"), field="source run provenance")
    model_revision = _nonempty_string(
        provenance.get("model_revision"),
        field="source run provenance.model_revision",
    )
    counts = _mapping(manifest.get("counts"), field="source run counts")
    if counts.get("failed") != 0 or manifest.get("failures") not in ([], None):
        raise ValueError("source run contains failures")

    source_pairs: list[_SourcePair] = []
    previous_id: str | None = None
    with pairs_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {pairs_path}:{line_number}")
            payload = _strict_loads(line, context=f"{pairs_path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"pair record must be an object at {pairs_path}:{line_number}")
            sample_id, clean_correct, edited_wrong, aligned_count = _validate_pair_record(
                payload,
                config=config,
                line_number=line_number,
            )
            if previous_id is not None and sample_id <= previous_id:
                raise ValueError("pair sample IDs must be strictly sorted and unique")
            previous_id = sample_id
            source_pairs.append(
                _SourcePair(
                    sample_id=sample_id,
                    record=payload,
                    fingerprint=_record_fingerprint(line),
                    clean_correct=clean_correct,
                    edited_wrong=edited_wrong,
                    aligned_count=aligned_count,
                )
            )
    if not source_pairs:
        raise ValueError("pairs input contains no records")
    if counts.get("written") != len(source_pairs):
        raise ValueError("source run written count does not match pairs.jsonl")
    return _Source(
        pairs=tuple(source_pairs),
        pairs_sha256=_sha256(pairs_path),
        run_sha256=_sha256(run_path),
        manifest=manifest,
        model_revision=model_revision,
    )


def _checkpoint_path(checkpoints_dir: Path, sample_id: str) -> Path:
    return checkpoints_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"


def _process_direction(
    scan: DirectionScan,
    *,
    n_layers: int,
) -> dict[str, object]:
    from typo_cot.experiments.layerwise_kl_patching.metrics import normalized_kl_score

    denominator = float(scan.denominator_kl)
    if not math.isfinite(denominator):
        return {
            "status": "excluded",
            "exclusion_reason": "nonfinite_denominator",
            "denominator_kl": None,
            "expected_layers": n_layers,
            "observed_finite_layers": 0,
            "layers": [],
        }
    if denominator <= 1e-9:
        return {
            "status": "excluded",
            "exclusion_reason": "denominator_le_1e-9",
            "denominator_kl": denominator,
            "expected_layers": n_layers,
            "observed_finite_layers": 0,
            "layers": [],
        }
    values = tuple(float(value) for value in scan.patched_kl_by_layer)
    if len(values) != n_layers:
        return {
            "status": "excluded",
            "exclusion_reason": "incomplete_layer_grid",
            "denominator_kl": denominator,
            "expected_layers": n_layers,
            "observed_finite_layers": sum(math.isfinite(value) for value in values),
            "layers": [],
        }
    if not all(math.isfinite(value) for value in values):
        return {
            "status": "excluded",
            "exclusion_reason": "nonfinite_layer_value",
            "denominator_kl": denominator,
            "expected_layers": n_layers,
            "observed_finite_layers": sum(math.isfinite(value) for value in values),
            "layers": [],
        }
    layers = [
        {
            "layer_index": layer,
            "patched_kl": value,
            "normalized_kl": normalized_kl_score(
                denominator_kl=denominator,
                patched_kl=value,
            ),
        }
        for layer, value in enumerate(values)
    ]
    return {
        "status": "included",
        "exclusion_reason": None,
        "denominator_kl": denominator,
        "expected_layers": n_layers,
        "observed_finite_layers": n_layers,
        "layers": layers,
    }


def _process_pair_scan(
    source_pair: _SourcePair,
    scan: PairScan,
    *,
    config: LayerwiseKLPatchingConfig,
    n_layers: int,
    runtime_fingerprint: str,
) -> dict[str, object]:
    if scan.sample_id != source_pair.sample_id:
        raise ValueError("runtime PairScan sample_id does not match its source pair")
    if set(scan.directions) != set(config.directions):
        raise ValueError("runtime did not return exactly the requested directions")
    return {
        "schema_version": "layerwise-kl-patching-checkpoint/v1",
        "sample_id": source_pair.sample_id,
        "source_record_sha256": source_pair.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "num_layers": n_layers,
        "aligned_positions": source_pair.aligned_count,
        "directions": {
            direction: _process_direction(scan.directions[direction], n_layers=n_layers)
            for direction in config.directions
        },
    }


def _runtime_fingerprint(provenance: Mapping[str, object]) -> str:
    serialized = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _base_manifest(
    *,
    config: LayerwiseKLPatchingConfig,
    source: _Source,
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object] | None,
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    comparability = {
        "status": (
            "partial-smoke-run"
            if config.limit is not None
            else "fresh-paper-protocol-reproduction"
        ),
        "exact_historical_table5_ids": False,
        "note": (
            "Fresh corrected pair preparation does not recreate undocumented historical "
            "teacher-forcing eligibility gates."
        ),
    }
    payload: dict[str, object] = {
        "schema_version": "layerwise-kl-patching-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "layerwise-kl-patching",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "input": {
            "pairs_path": str(config.pairs.resolve()),
            "pairs_sha256": source.pairs_sha256,
            "source_run_sha256": source.run_sha256,
            "source_record_count": len(source.pairs),
            "source_schema": "prepare-edited-pairs/v1",
            "source_model_revision": source.model_revision,
        },
        "runtime": dict(runtime_provenance) if runtime_provenance is not None else None,
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": comparability,
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    *,
    config: LayerwiseKLPatchingConfig,
    source: _Source,
) -> str:
    if manifest.get("schema_version") != "layerwise-kl-patching-run/v1":
        raise ValueError("cannot resume an unknown layerwise run schema")
    if manifest.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    if manifest.get("paper_sha256") != PAPER_SHA256 or manifest.get("protocol") != _PROTOCOL:
        raise ValueError("resume paper or protocol fingerprint does not match")
    input_payload = _mapping(manifest.get("input"), field="resume input")
    if (
        input_payload.get("pairs_sha256") != source.pairs_sha256
        or input_payload.get("source_run_sha256") != source.run_sha256
        or input_payload.get("source_record_count") != len(source.pairs)
    ):
        raise ValueError("resume source input fingerprint does not match")
    started_at = manifest.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _completed_result(
    manifest: Mapping[str, object],
    *,
    output_dir: Path,
) -> LayerwiseKLPatchingResult:
    outputs = _mapping(manifest.get("outputs"), field="completed run outputs")
    expected_names = (
        "layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    )
    for name in expected_names:
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"completed output is missing or has changed: {path}")
    counts = _mapping(manifest.get("counts"), field="completed run counts")
    return LayerwiseKLPatchingResult(
        layer_records_path=output_dir / "layer_records.jsonl",
        pair_status_records_path=output_dir / "pair_status_records.jsonl",
        summary_path=output_dir / "setting_summary.json",
        run_path=output_dir / "run.json",
        included_grids=int(counts.get("included_grids", 0)),
        layer_records=int(counts.get("layer_records", 0)),
    )


def _load_checkpoint(
    path: Path,
    *,
    source_pair: _SourcePair,
    runtime_fingerprint: str,
    n_layers: int,
) -> dict[str, object]:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "layerwise-kl-patching-checkpoint/v1"
        or checkpoint.get("sample_id") != source_pair.sample_id
        or checkpoint.get("source_record_sha256") != source_pair.fingerprint
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("num_layers") != n_layers
    ):
        raise ValueError(f"checkpoint provenance does not match: {path}")
    return checkpoint


def _compile_outputs(
    *,
    config: LayerwiseKLPatchingConfig,
    source: _Source,
    selected: Sequence[_SourcePair],
    checkpoints_dir: Path,
    runtime_fingerprint: str,
    n_layers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from typo_cot.experiments.layerwise_kl_patching.metrics import summarize_direction

    layer_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    scores: dict[str, list[tuple[float, ...]]] = {
        direction: [] for direction in config.directions
    }
    exclusion_counts: dict[str, Counter[str]] = {
        direction: Counter() for direction in config.directions
    }
    for source_pair in selected:
        checkpoint = _load_checkpoint(
            _checkpoint_path(checkpoints_dir, source_pair.sample_id),
            source_pair=source_pair,
            runtime_fingerprint=runtime_fingerprint,
            n_layers=n_layers,
        )
        directions = _mapping(checkpoint.get("directions"), field="checkpoint directions")
        for direction in config.directions:
            result = _mapping(directions.get(direction), field=f"checkpoint {direction}")
            status = str(result.get("status"))
            reason = result.get("exclusion_reason")
            status_rows.append(
                {
                    "schema_version": "layerwise-kl-patching-pair-status/v1",
                    "sample_id": source_pair.sample_id,
                    "source_record_sha256": source_pair.fingerprint,
                    "model": config.model,
                    "benchmark": config.benchmark,
                    "targeting": config.targeting,
                    "direction": direction,
                    "status": status,
                    "exclusion_reason": reason,
                    "denominator_kl": result.get("denominator_kl"),
                    "expected_layers": n_layers,
                    "observed_finite_layers": result.get("observed_finite_layers"),
                    "aligned_positions": source_pair.aligned_count,
                }
            )
            if status != "included":
                exclusion_counts[direction][str(reason)] += 1
                continue
            raw_layers = result.get("layers")
            if not isinstance(raw_layers, list) or len(raw_layers) != n_layers:
                raise ValueError("included checkpoint lost its complete layer grid")
            pair_scores: list[float] = []
            for raw_layer in raw_layers:
                layer = _mapping(raw_layer, field="checkpoint layer")
                layer_index = int(layer["layer_index"])
                score = float(layer["normalized_kl"])
                pair_scores.append(score)
                layer_rows.append(
                    {
                        "schema_version": "layerwise-kl-patching-layer/v1",
                        "paper_sha256": PAPER_SHA256,
                        "model": config.model,
                        "benchmark": config.benchmark,
                        "targeting": config.targeting,
                        "sample_id": source_pair.sample_id,
                        "source_record_sha256": source_pair.fingerprint,
                        "direction": direction,
                        "layer_index": layer_index,
                        "num_layers": n_layers,
                        "relative_depth": layer_index / n_layers,
                        "layer_center_relative_depth": (layer_index + 0.5) / n_layers,
                        "aligned_positions": source_pair.aligned_count,
                        "denominator_kl": result["denominator_kl"],
                        "patched_kl": layer["patched_kl"],
                        "normalized_kl": score,
                    }
                )
            scores[direction].append(tuple(pair_scores))

    upstream = Counter()
    selected_failures = 0
    aligned_selected = 0
    for pair in source.pairs:
        if not pair.clean_correct:
            upstream["clean_not_correct"] += 1
        elif not pair.edited_wrong:
            upstream["edited_not_wrong"] += 1
        else:
            selected_failures += 1
            if pair.aligned_count:
                aligned_selected += 1
            else:
                upstream["no_aligned_words"] += 1

    direction_summaries: dict[str, object] = {}
    for direction in config.directions:
        direction_summary: dict[str, object] = {
            "included_pairs": len(scores[direction]),
            "excluded_pairs": len(selected) - len(scores[direction]),
            "exclusions": dict(sorted(exclusion_counts[direction].items())),
        }
        if scores[direction]:
            direction_summary.update(
                summarize_direction(
                    tuple(scores[direction]),
                    bootstrap_resamples=2000,
                    seed=42,
                )
            )
        else:
            direction_summary.update(
                {
                    "layer_profile": [],
                    "peak": None,
                    "mcb": None,
                    "depth_thirds": [],
                }
            )
        direction_summaries[direction] = direction_summary

    summary = {
        "schema_version": "layerwise-kl-patching-setting-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "layerwise-kl-patching",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "targeting": config.targeting,
            "num_layers": n_layers,
        },
        "protocol": _PROTOCOL,
        "population": {
            "input_pairs": len(source.pairs),
            "selected_failures": selected_failures,
            "aligned_selected_failures": aligned_selected,
            "eligible_before_limit": aligned_selected,
            "selected_by_limit": len(selected),
            "upstream_exclusions": dict(sorted(upstream.items())),
        },
        "directions": direction_summaries,
        "cross_setting_boundary": {
            "scope": "one-setting-only",
            "headline": "macro-average 30 setting summaries with equal setting weight",
            "all_32_sensitivity": True,
        },
    }
    return layer_rows, status_rows, summary


def run_layerwise_kl_patching(
    config: LayerwiseKLPatchingConfig,
    *,
    runtime: LayerwiseKLPatchingRuntime | None = None,
) -> LayerwiseKLPatchingResult:
    """Validate one source setting, scan all layers, and publish auditable outputs."""
    source = _load_source(config)
    output_dir = config.output_dir
    if output_dir.resolve() == config.pairs.parent.resolve():
        raise ValueError("output directory must not overwrite the source pair directory")
    run_path = output_dir / "run.json"
    layer_path = output_dir / "layer_records.jsonl"
    status_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "setting_summary.json"
    work_dir = output_dir / ".layerwise-kl-patching-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    previous: dict[str, object] | None = None
    if config.resume:
        previous = _load_json(run_path)
        started_at = _validate_resume_manifest(previous, config=config, source=source)
        if previous.get("status") == "completed":
            return _completed_result(previous, output_dir=output_dir)
    else:
        started_at = _now()

    eligible = [
        pair
        for pair in source.pairs
        if pair.clean_correct and pair.edited_wrong and pair.aligned_count > 0
    ]
    if not eligible:
        raise ValueError("source contains no aligned clean-correct, edited-wrong pairs")
    selected = eligible[: config.limit] if config.limit is not None else eligible

    if runtime is None:
        from typo_cot.experiments.layerwise_kl_patching.runtime import (
            HuggingFaceLayerwiseKLPatchingRuntime,
        )

        runtime = HuggingFaceLayerwiseKLPatchingRuntime(
            config,
            revision=source.model_revision,
        )
    if not isinstance(runtime.num_layers, int) or runtime.num_layers <= 0:
        raise ValueError("runtime num_layers must be a positive integer")
    runtime_provenance = dict(runtime.provenance())
    if runtime_provenance.get("num_decoder_layers") != runtime.num_layers:
        raise ValueError("runtime provenance num_decoder_layers does not match runtime")
    if runtime_provenance.get("model_revision") != source.model_revision:
        raise ValueError("runtime model revision does not match pair preparation")
    if runtime_provenance.get("tokenizer_revision") != source.model_revision:
        raise ValueError("runtime tokenizer revision does not match pair preparation")
    runtime_fp = _runtime_fingerprint(runtime_provenance)
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match the original run")

    # A model/environment initialization error must not leave a new output
    # directory that falsely looks resumable. Create checkpoints only after the
    # runtime and its frozen provenance have both validated.
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    existing = 0
    for pair in selected:
        path = _checkpoint_path(checkpoints_dir, pair.sample_id)
        if path.is_file():
            _load_checkpoint(
                path,
                source_pair=pair,
                runtime_fingerprint=runtime_fp,
                n_layers=runtime.num_layers,
            )
            existing += 1

    counts: dict[str, object] = {
        "input_pairs": len(source.pairs),
        "eligible_pairs": len(eligible),
        "selected_pairs": len(selected),
        "checkpointed_pairs": existing,
        "failed_pairs": 0,
        "included_grids": 0,
        "layer_records": 0,
    }
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            source=source,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            counts=counts,
            failures=[],
        ),
    )

    failures: list[dict[str, str]] = []
    for pair in tqdm(
        selected,
        desc="layerwise-kl-patching",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        checkpoint_path = _checkpoint_path(checkpoints_dir, pair.sample_id)
        if checkpoint_path.is_file():
            continue
        try:
            scan = runtime.scan_pair(pair.record, config.directions)
            checkpoint = _process_pair_scan(
                pair,
                scan,
                config=config,
                n_layers=runtime.num_layers,
                runtime_fingerprint=runtime_fp,
            )
            _write_json_atomic(checkpoint_path, checkpoint)
            counts["checkpointed_pairs"] = int(counts["checkpointed_pairs"]) + 1
        except Exception as exc:  # noqa: BLE001 - preserve all expensive completed pairs
            failures.append(
                {
                    "sample_id": pair.sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        counts["failed_pairs"] = len(failures)
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                source=source,
                status="running",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                counts=counts,
                failures=failures,
            ),
        )

    if failures:
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                source=source,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                counts=counts,
                failures=failures,
            ),
        )
        noun = "pair" if len(failures) == 1 else "pairs"
        raise LayerwiseKLPatchingRunError(
            f"{len(failures)} {noun} failed; inspect run.json and rerun with --resume"
        )

    layer_rows, status_rows, summary = _compile_outputs(
        config=config,
        source=source,
        selected=selected,
        checkpoints_dir=checkpoints_dir,
        runtime_fingerprint=runtime_fp,
        n_layers=runtime.num_layers,
    )
    _write_jsonl_atomic(layer_path, layer_rows)
    _write_jsonl_atomic(status_path, status_rows)
    _write_json_atomic(summary_path, summary)
    included_grids = sum(row["status"] == "included" for row in status_rows)
    counts["included_grids"] = included_grids
    counts["layer_records"] = len(layer_rows)
    output_metadata = {
        layer_path.name: {"sha256": _sha256(layer_path), "records": len(layer_rows)},
        status_path.name: {"sha256": _sha256(status_path), "records": len(status_rows)},
        summary_path.name: {"sha256": _sha256(summary_path)},
    }
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            source=source,
            status="completed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            counts=counts,
            failures=[],
            outputs=output_metadata,
        ),
    )
    shutil.rmtree(work_dir)
    return LayerwiseKLPatchingResult(
        layer_records_path=layer_path,
        pair_status_records_path=status_path,
        summary_path=summary_path,
        run_path=run_path,
        included_grids=included_grids,
        layer_records=len(layer_rows),
    )

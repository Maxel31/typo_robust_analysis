"""Strict, resumable writer for the paper's patch-position control scan."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tqdm.auto import tqdm

from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.patch_position_controls.planning import (
    POSITION_NAMES,
    PositionCoordinates,
)
from typo_cot.experiments.patch_position_controls.runtime import POSITION_RUNTIME_PROTOCOL

_POSITION_ORDER = {name: index for index, name in enumerate(POSITION_NAMES)}
_ALTERNATIVE_POSITIONS = frozenset(("prompt-final", "question-final"))
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_DENOMINATOR_EPSILON = 1e-9
_DENOMINATOR_ABS_TOLERANCE = 1e-12
_DENOMINATOR_REL_TOLERANCE = 1e-5
_SOURCE_OUTPUT_NAMES = (
    "layer_records.jsonl",
    "pair_status_records.jsonl",
    "setting_summary.json",
)
_PUBLIC_OUTPUT_NAMES = (
    "position_control_records.jsonl",
    "pair_status_records.jsonl",
    "position_control_summary.json",
)
_PROTOCOL = {
    "schema_version": "patch-position-controls-protocol/v1",
    "source_operation": "layerwise-kl-patching",
    "source_direction": "clean-to-edited",
    "source_cohort": "complete-included-layerwise-kl-grids",
    "intervention_site": "complete-decoder-block-residual-output",
    "intervention_positions": list(POSITION_NAMES),
    "readout": "prompt-final-logits-predicting-first-cot-token",
    "kl_orientation": "clean-to-patched-edited",
    "score": "1-KL(clean||patched-edited)/KL(clean||edited)",
    "denominator_epsilon": _DENOMINATOR_EPSILON,
    "runtime_denominator_tolerance": {
        "absolute": _DENOMINATOR_ABS_TOLERANCE,
        "relative": _DENOMINATOR_REL_TOLERANCE,
    },
    "layer_grid": "all-decoder-layers-0-through-L-minus-1",
    "layer_profile": "across-pair-median",
    "peak_ties": "retain-all-exact-ties",
    "pair_policy": "one-common-pair-set-with-pair-atomic-checkpoints",
    "design_status": "post_hoc_exploratory_descriptive",
    "dtype": "bfloat16",
    "use_cache": False,
}
_PUBLISHED_REFERENCE = {
    "cohort_pairs": 109,
    "edited-word": {
        "peak_layer": 2,
        "peak_normalized_kl_rounded": 0.751,
        "relative_depth_rounded": 0.059,
        "table5_mcb_source_metadata": "8*",
    },
    "prompt-final": {"layer_values_rounded": {"0": 0.001, "16": 0.25, "32": 0.98}},
    "question-final": {
        "all_layer_peak_normalized_kl_rounded": 0.011,
        "peak_layer_published": False,
    },
}


class PositionControlRunError(RuntimeError):
    """Raised after a runtime pair fails, while retaining valid checkpoints."""


@dataclass(frozen=True, slots=True)
class PositionControlConfig:
    """Frozen public arguments for the one published position-control setting."""

    model: str
    benchmark: str
    layerwise_kl_run: Path
    positions: tuple[str, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layerwise_kl_run", Path(self.layerwise_kl_run))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.model != "google/gemma-3-4b-it":
            raise ValueError(
                "patch-position-controls only supports the paper model " "google/gemma-3-4b-it"
            )
        if self.benchmark != "gsm8k":
            raise ValueError("patch-position-controls only supports gsm8k")
        if not self.positions:
            raise ValueError("positions must contain at least one position")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must not contain duplicates")
        unknown = [name for name in self.positions if name not in _POSITION_ORDER]
        if unknown:
            raise ValueError(f"unsupported position: {unknown[0]!r}")
        object.__setattr__(
            self,
            "positions",
            tuple(sorted(self.positions, key=_POSITION_ORDER.__getitem__)),
        )
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""
        payload = asdict(self)
        payload["layerwise_kl_run"] = str(self.layerwise_kl_run.resolve())
        payload["output_dir"] = str(self.output_dir.resolve())
        payload["positions"] = list(self.positions)
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class PositionControlResult:
    """Published paths and counts for a completed position scan."""

    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    records: int


class PositionControlRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def scan_pair(self, pair: dict[str, object], positions: tuple[str, ...]) -> object: ...


@dataclass(frozen=True, slots=True)
class _PreparedPair:
    sample_id: str
    record: dict[str, object]
    fingerprint: str
    clean_correct: bool
    edited_wrong: bool
    aligned_count: int
    aligned_coordinates: PositionCoordinates | None

    @property
    def eligible(self) -> bool:
        return self.clean_correct and self.edited_wrong and self.aligned_count > 0


@dataclass(frozen=True, slots=True)
class _ReferencePair:
    sample_id: str
    record: dict[str, object]
    source_record_sha256: str
    source_status_sha256: str
    source_layers_sha256: str
    denominator_kl: float
    aligned_coordinates: PositionCoordinates
    reference_layers: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ReferenceSource:
    pairs: tuple[_ReferencePair, ...]
    run_manifest: dict[str, object]
    run_sha256: str
    output_sha256: Mapping[str, str]
    source_pairs_path: Path
    source_pairs_sha256: str
    source_pair_run_sha256: str
    model_revision: str
    num_layers: int
    source_comparability: Mapping[str, object]
    upstream_population: Mapping[str, object]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    payload = _strict_loads(text, context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[tuple[dict[str, object], str]]:
    if not path.is_file():
        raise ValueError(f"JSONL input is not a file: {path}")
    rows: list[tuple[dict[str, object], str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = _strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append((payload, _record_fingerprint(line)))
    return rows


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
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


def _nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _span(value: object, *, field: str, prompt_length: int) -> tuple[int, int]:
    payload = _mapping(value, field=field)
    start, end = payload.get("start"), payload.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= prompt_length
    ):
        raise ValueError(f"{field} must be a valid prompt character span")
    return start, end


def _validate_prepared_pair(
    record: dict[str, object],
    *,
    config: PositionControlConfig,
    sample_id: str,
    fingerprint: str,
) -> _PreparedPair:
    context = f"source pair {sample_id}"
    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise ValueError(f"{context} has an unknown schema")
    for field, expected in (
        ("sample_id", sample_id),
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", "attribution-4"),
        ("seed", 42),
        ("num_edits_requested", 4),
    ):
        if record.get(field) != expected:
            raise ValueError(f"{context}.{field} does not match the paper setting")
    prompts: dict[str, Mapping[str, object]] = {}
    prompt_counts: dict[str, int] = {}
    correctness: dict[str, bool] = {}
    editable_texts: dict[str, str] = {}
    editable_spans: dict[str, tuple[int, int]] = {}
    for side in ("clean", "edited"):
        payload = _mapping(record.get(side), field=f"{context}.{side}")
        prompt = _nonempty_string(payload.get("prompt"), field=f"{context}.{side}.prompt")
        count = _positive_int(
            payload.get("prompt_token_count"),
            field=f"{context}.{side}.prompt_token_count",
        )
        editable_text = _nonempty_string(
            payload.get("editable_text"), field=f"{context}.{side}.editable_text"
        )
        editable_span = _span(
            payload.get("editable_prompt_span"),
            field=f"{context}.{side}.editable_prompt_span",
            prompt_length=len(prompt),
        )
        if prompt[editable_span[0] : editable_span[1]] != editable_text:
            raise ValueError(f"{context}.{side}.editable_text does not match editable_prompt_span")
        prompts[side] = payload
        prompt_counts[side] = count
        editable_texts[side] = editable_text
        editable_spans[side] = editable_span
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
    coordinates: dict[str, list[int]] = {"clean": [], "edited": []}
    previous_word_index: int | None = None
    for word_index, raw_word in enumerate(aligned_words):
        word = _mapping(raw_word, field=f"{context}.aligned_words[{word_index}]")
        recorded_word_index = word.get("word_index")
        if (
            not isinstance(recorded_word_index, int)
            or isinstance(recorded_word_index, bool)
            or recorded_word_index < 0
            or (previous_word_index is not None and recorded_word_index <= previous_word_index)
        ):
            raise ValueError(f"{context}.aligned_words word indices must be increasing")
        previous_word_index = recorded_word_index
        for side in ("clean", "edited"):
            final = word.get(f"{side}_final_token")
            indices = word.get(f"{side}_token_indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    not isinstance(index, int) or isinstance(index, bool) or index < 0
                    for index in indices
                )
                or any(index >= prompt_counts[side] for index in indices)
                or any(left >= right for left, right in zip(indices, indices[1:]))
                or not isinstance(final, int)
                or isinstance(final, bool)
                or final != indices[-1]
                or final >= prompt_counts[side]
            ):
                raise ValueError(
                    f"{context}.aligned_words[{word_index}].{side}_final_token "
                    "does not match valid token indices"
                )
            prompt_span = _span(
                word.get(f"{side}_prompt_span"),
                field=f"{context}.aligned_words[{word_index}].{side}_prompt_span",
                prompt_length=len(str(prompts[side]["prompt"])),
            )
            word_text = _nonempty_string(
                word.get(f"{side}_text"),
                field=f"{context}.aligned_words[{word_index}].{side}_text",
            )
            editable_word_span = _span(
                word.get(f"{side}_editable_span"),
                field=f"{context}.aligned_words[{word_index}].{side}_editable_span",
                prompt_length=len(editable_texts[side]),
            )
            editable_prompt_start, editable_prompt_end = editable_spans[side]
            expected_prompt_span = (
                editable_prompt_start + editable_word_span[0],
                editable_prompt_start + editable_word_span[1],
            )
            prompt = str(prompts[side]["prompt"])
            if (
                prompt_span != expected_prompt_span
                or prompt_span[1] > editable_prompt_end
                or prompt[prompt_span[0] : prompt_span[1]] != word_text
                or editable_texts[side][editable_word_span[0] : editable_word_span[1]] != word_text
            ):
                raise ValueError(
                    f"{context}.aligned_words[{word_index}] {side} text/span provenance "
                    "does not match the editable prompt"
                )
            coordinates[side].append(final)
    for side, values in coordinates.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{context} has duplicate {side} edited-word coordinates")
    aligned_coordinates = (
        PositionCoordinates(
            source_positions=tuple(coordinates["clean"]),
            destination_positions=tuple(coordinates["edited"]),
            locator="aligned-edited-word-final-tokens/v1",
        )
        if aligned_words
        else None
    )
    return _PreparedPair(
        sample_id=sample_id,
        record=record,
        fingerprint=fingerprint,
        clean_correct=correctness["clean"],
        edited_wrong=not correctness["edited"],
        aligned_count=len(aligned_words),
        aligned_coordinates=aligned_coordinates,
    )


def _validate_source_protocol(protocol: Mapping[str, object]) -> None:
    required = {
        "source_cohort": "clean-correct-edited-wrong",
        "intervention_site": "complete-decoder-block-residual-output",
        "intervention_positions": "all-aligned-edited-word-final-tokens",
        "readout": "prompt-final-logits-predicting-first-cot-token",
        "kl_orientation": "reference-to-recipient",
        "denominator_epsilon": _DENOMINATOR_EPSILON,
        "layer_grid": "all-decoder-layers-0-through-L-minus-1",
        "layer_profile": "across-pair-median",
    }
    for field, expected in required.items():
        if protocol.get(field) != expected:
            raise ValueError(f"source layerwise protocol field {field!r} does not match")


def _verify_source_outputs(run_dir: Path, manifest: Mapping[str, object]) -> dict[str, str]:
    outputs = _mapping(manifest.get("outputs"), field="source outputs")
    if set(outputs) != set(_SOURCE_OUTPUT_NAMES):
        raise ValueError("source run does not register the exact layerwise output set")
    hashes: dict[str, str] = {}
    for name in _SOURCE_OUTPUT_NAMES:
        metadata = _mapping(outputs.get(name), field=f"source output {name}")
        path = run_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"source output is missing or has changed: {path}")
        hashes[name] = str(metadata["sha256"])
    return hashes


def _load_source_pairs(
    *,
    config: PositionControlConfig,
    pairs_path: Path,
    expected_pairs_sha256: object,
    expected_run_sha256: object,
    expected_model_revision: str,
) -> dict[str, _PreparedPair]:
    if not pairs_path.is_file() or expected_pairs_sha256 != _sha256(pairs_path):
        raise ValueError("source pair file is missing or has changed")
    pair_run_path = pairs_path.parent / "run.json"
    if not pair_run_path.is_file() or expected_run_sha256 != _sha256(pair_run_path):
        raise ValueError("source pair-preparation run is missing or has changed")
    pair_manifest = _load_json(pair_run_path)
    if pair_manifest.get("schema_version") != "prepare-edited-pairs-run/v1":
        raise ValueError("source pair-preparation run has an unknown schema")
    if pair_manifest.get("operation") != "prepare-edited-pairs":
        raise ValueError("source pair-preparation run has the wrong operation")
    if pair_manifest.get("status") != "completed":
        raise ValueError("source pair-preparation run is not completed")
    if pair_manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("source pair-preparation paper SHA-256 does not match")
    pair_arguments = _mapping(
        pair_manifest.get("arguments"), field="source pair-preparation arguments"
    )
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", "attribution-4"),
        ("seed", 42),
        ("num_edits", 4),
        ("max_new_tokens", 512),
        ("limit", None),
    ):
        if field not in pair_arguments or pair_arguments.get(field) != expected:
            raise ValueError(
                f"source pair-preparation argument {field!r} does not match the paper setting"
            )
    pair_provenance = _mapping(
        pair_manifest.get("provenance"), field="source pair-preparation provenance"
    )
    if pair_provenance.get("model_revision") != expected_model_revision:
        raise ValueError("source pair-preparation model revision does not match layerwise run")
    pair_counts = _mapping(pair_manifest.get("counts"), field="source pair-preparation counts")
    if pair_counts.get("failed") != 0 or pair_manifest.get("failures") not in ([], None):
        raise ValueError("source pair-preparation run contains failures")
    records: dict[str, _PreparedPair] = {}
    previous: str | None = None
    for record, fingerprint in _load_jsonl(pairs_path):
        sample_id = _nonempty_string(record.get("sample_id"), field="pair sample_id")
        if previous is not None and sample_id <= previous:
            raise ValueError("source pair IDs must be strictly sorted and unique")
        previous = sample_id
        records[sample_id] = _validate_prepared_pair(
            record,
            config=config,
            sample_id=sample_id,
            fingerprint=fingerprint,
        )
    if pair_counts.get("written") != len(records):
        raise ValueError("source pair-preparation written count does not match pairs.jsonl")
    if pair_counts.get("discovered") != len(records):
        raise ValueError(
            "source pair-preparation discovered count does not match its failure-free output"
        )
    return records


def _load_reference_source(config: PositionControlConfig) -> _ReferenceSource:
    run_dir = config.layerwise_kl_run
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError(f"layerwise KL run is missing run.json: {run_dir}")
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "layerwise-kl-patching-run/v1":
        raise ValueError("source layerwise run has an unknown schema")
    if manifest.get("operation") != "layerwise-kl-patching":
        raise ValueError("source run has the wrong operation")
    if manifest.get("status") != "completed":
        raise ValueError("source layerwise run is not completed")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("source paper SHA-256 does not match the final paper")
    arguments = _mapping(manifest.get("arguments"), field="source arguments")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", "attribution-4"),
    ):
        if arguments.get(field) != expected:
            raise ValueError(f"source argument {field!r} does not match")
    directions = arguments.get("directions")
    if not isinstance(directions, list) or "clean-to-edited" not in directions:
        raise ValueError("source run must include the clean-to-edited direction")
    if len(directions) != len(set(directions)) or any(
        direction not in ("clean-to-edited", "edited-to-clean") for direction in directions
    ):
        raise ValueError("source run has an invalid direction plan")
    if "limit" not in arguments or arguments.get("limit") is not None:
        raise ValueError("source layerwise run must not be limited")
    _validate_source_protocol(_mapping(manifest.get("protocol"), field="source protocol"))
    output_hashes = _verify_source_outputs(run_dir, manifest)
    source_counts = _mapping(manifest.get("counts"), field="source counts")
    if source_counts.get("failed_pairs") != 0 or manifest.get("failures") not in ([], None):
        raise ValueError("completed source layerwise run contains failures")

    runtime = _mapping(manifest.get("runtime"), field="source runtime")
    num_layers = _positive_int(
        runtime.get("num_decoder_layers"), field="source runtime.num_decoder_layers"
    )
    model_revision = _nonempty_string(
        runtime.get("model_revision"), field="source runtime.model_revision"
    )
    if runtime.get("requested_revision") != model_revision:
        raise ValueError("source runtime requested revision does not match model revision")
    if runtime.get("tokenizer_revision") != model_revision:
        raise ValueError("source runtime tokenizer revision does not match model revision")
    if runtime.get("dtype") != "bfloat16":
        raise ValueError("source layerwise runtime must use bfloat16")

    input_payload = _mapping(manifest.get("input"), field="source input")
    pairs_path_value = _nonempty_string(
        input_payload.get("pairs_path"), field="source input.pairs_path"
    )
    pairs_argument = _nonempty_string(arguments.get("pairs"), field="source arguments.pairs")
    pairs_path = Path(pairs_path_value)
    if Path(pairs_argument).resolve() != pairs_path.resolve():
        raise ValueError("source run records inconsistent pair paths")
    if input_payload.get("source_model_revision") != model_revision:
        raise ValueError("source input model revision does not match runtime")
    pair_records = _load_source_pairs(
        config=config,
        pairs_path=pairs_path,
        expected_pairs_sha256=input_payload.get("pairs_sha256"),
        expected_run_sha256=input_payload.get("source_run_sha256"),
        expected_model_revision=model_revision,
    )
    if input_payload.get("source_record_count") != len(pair_records):
        raise ValueError("source layerwise input count does not match pairs.jsonl")
    eligible_ids = tuple(sample_id for sample_id, pair in pair_records.items() if pair.eligible)
    expected_status_keys = {
        (sample_id, direction) for sample_id in eligible_ids for direction in directions
    }

    raw_statuses = _load_jsonl(run_dir / "pair_status_records.jsonl")
    output_metadata = _mapping(manifest.get("outputs"), field="source outputs")
    status_metadata = _mapping(
        output_metadata.get("pair_status_records.jsonl"), field="source status metadata"
    )
    if status_metadata.get("records") != len(raw_statuses):
        raise ValueError("source status output record count does not match")
    statuses: dict[str, tuple[dict[str, object], str]] = {}
    all_statuses: dict[tuple[str, str], tuple[dict[str, object], str]] = {}
    seen_status_keys: set[tuple[str, str]] = set()
    for row, fingerprint in raw_statuses:
        if row.get("schema_version") != "layerwise-kl-patching-pair-status/v1":
            raise ValueError("source status row has an unknown schema")
        sample_id = _nonempty_string(row.get("sample_id"), field="source status sample_id")
        direction = _nonempty_string(row.get("direction"), field="source status direction")
        if direction not in directions:
            raise ValueError("source status row is outside the requested direction plan")
        key = (sample_id, direction)
        if key in seen_status_keys:
            raise ValueError("source status rows contain a duplicate pair-direction")
        seen_status_keys.add(key)
        pair_entry = pair_records.get(sample_id)
        if pair_entry is None:
            raise ValueError("source status row has no matching pair record")
        for field, expected in (
            ("model", config.model),
            ("benchmark", config.benchmark),
            ("targeting", "attribution-4"),
            ("source_record_sha256", pair_entry.fingerprint),
            ("expected_layers", num_layers),
            ("aligned_positions", pair_entry.aligned_count),
        ):
            if row.get(field) != expected:
                raise ValueError(f"source status field {field!r} does not match")
        if row.get("status") not in ("included", "excluded"):
            raise ValueError("source status row has an invalid status")
        all_statuses[key] = (row, fingerprint)
        if direction == "clean-to-edited" and row.get("status") == "included":
            statuses[sample_id] = (row, fingerprint)
    if seen_status_keys != expected_status_keys:
        raise ValueError("source status coverage does not match every eligible pair and direction")
    if not statuses:
        raise ValueError("source run has no included clean-to-edited pairs")

    raw_layers = _load_jsonl(run_dir / "layer_records.jsonl")
    layer_metadata = _mapping(
        output_metadata.get("layer_records.jsonl"), field="source layer metadata"
    )
    if layer_metadata.get("records") != len(raw_layers):
        raise ValueError("source layer output record count does not match")
    layers_by_id: dict[str, list[tuple[dict[str, object], str]]] = {
        sample_id: [] for sample_id in statuses
    }
    for row, fingerprint in raw_layers:
        if row.get("schema_version") != "layerwise-kl-patching-layer/v1":
            raise ValueError("source layer row has an unknown schema")
        sample_id = _nonempty_string(row.get("sample_id"), field="source layer sample_id")
        direction = _nonempty_string(row.get("direction"), field="source layer direction")
        status_entry = all_statuses.get((sample_id, direction))
        pair_entry = pair_records.get(sample_id)
        if (
            direction not in directions
            or status_entry is None
            or status_entry[0].get("status") != "included"
            or pair_entry is None
        ):
            raise ValueError("source layer row has no included source status")
        for field, expected in (
            ("paper_sha256", PAPER_SHA256),
            ("model", config.model),
            ("benchmark", config.benchmark),
            ("targeting", "attribution-4"),
            ("source_record_sha256", pair_entry.fingerprint),
            ("num_layers", num_layers),
            ("aligned_positions", pair_entry.aligned_count),
        ):
            if row.get(field) != expected:
                raise ValueError(f"source layer field {field!r} does not match")
        layer_index = row.get("layer_index")
        if (
            not isinstance(layer_index, int)
            or isinstance(layer_index, bool)
            or not 0 <= layer_index < num_layers
        ):
            raise ValueError("source layer index is outside the decoder grid")
        if direction != "clean-to-edited":
            continue
        if sample_id not in layers_by_id:
            raise ValueError("source layer row has no included clean-to-edited status")
        layers_by_id[sample_id].append((row, fingerprint))

    if source_counts.get("layer_records") != len(raw_layers):
        raise ValueError("source run layer-record count does not match its output")
    included_grid_count = sum(row.get("status") == "included" for row, _ in raw_statuses)
    if source_counts.get("included_grids") != included_grid_count:
        raise ValueError("source run included-grid count does not match its statuses")
    for field, expected in (
        ("input_pairs", len(pair_records)),
        ("eligible_pairs", len(eligible_ids)),
        ("selected_pairs", len(eligible_ids)),
        ("checkpointed_pairs", len(eligible_ids)),
        ("failed_pairs", 0),
    ):
        if source_counts.get(field) != expected:
            raise ValueError(f"source run count {field!r} does not match its cohort")

    reference_pairs: list[_ReferencePair] = []
    for sample_id in sorted(statuses):
        status, status_fingerprint = statuses[sample_id]
        pair_entry = pair_records.get(sample_id)
        if pair_entry is None:
            raise ValueError(f"source included pair {sample_id!r} is absent from pairs.jsonl")
        pair_record = pair_entry.record
        pair_fingerprint = pair_entry.fingerprint
        if status.get("source_record_sha256") != pair_fingerprint:
            raise ValueError("source status pair fingerprint does not match pairs.jsonl")
        denominator = _finite_float(
            status.get("denominator_kl"), field=f"source status {sample_id}.denominator_kl"
        )
        if denominator <= _DENOMINATOR_EPSILON:
            raise ValueError("source included denominator is not greater than 1e-9")
        if (
            status.get("expected_layers") != num_layers
            or status.get("observed_finite_layers") != num_layers
        ):
            raise ValueError("source included status does not claim a complete layer grid")
        if not pair_entry.eligible or pair_entry.aligned_coordinates is None:
            raise ValueError("source included pair is not clean-correct, edited-wrong, and aligned")
        aligned_coordinates = pair_entry.aligned_coordinates
        layer_entries = layers_by_id[sample_id]
        if len(layer_entries) != num_layers:
            raise ValueError(f"source pair {sample_id!r} does not have a complete layer grid")
        validated_layers: list[dict[str, object]] = []
        layer_fingerprints: list[str] = []
        for expected_layer, (row, fingerprint) in enumerate(layer_entries):
            if row.get("schema_version") != "layerwise-kl-patching-layer/v1":
                raise ValueError("source layer row has an unknown schema")
            for field, expected in (
                ("paper_sha256", PAPER_SHA256),
                ("model", config.model),
                ("benchmark", config.benchmark),
                ("targeting", "attribution-4"),
                ("sample_id", sample_id),
                ("source_record_sha256", pair_fingerprint),
                ("direction", "clean-to-edited"),
                ("layer_index", expected_layer),
                ("num_layers", num_layers),
            ):
                if row.get(field) != expected:
                    raise ValueError(
                        f"source pair {sample_id!r} layer {expected_layer} field {field!r} "
                        "does not match"
                    )
            row_denominator = _finite_float(
                row.get("denominator_kl"), field="source layer denominator_kl"
            )
            patched = _finite_float(row.get("patched_kl"), field="source layer patched_kl")
            score = _finite_float(row.get("normalized_kl"), field="source layer normalized_kl")
            if row_denominator != denominator:
                raise ValueError("source layer denominator differs from its included status")
            expected_score = 1.0 - patched / denominator
            if not math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("source normalized KL does not match its numerator")
            validated_layers.append(dict(row))
            layer_fingerprints.append(fingerprint)
        reference_pairs.append(
            _ReferencePair(
                sample_id=sample_id,
                record=pair_record,
                source_record_sha256=pair_fingerprint,
                source_status_sha256=status_fingerprint,
                source_layers_sha256=_fingerprint_json(layer_fingerprints),
                denominator_kl=denominator,
                aligned_coordinates=aligned_coordinates,
                reference_layers=tuple(validated_layers),
            )
        )

    summary = _load_json(run_dir / "setting_summary.json")
    if summary.get("schema_version") != "layerwise-kl-patching-setting-summary/v1":
        raise ValueError("source summary has an unknown schema")
    setting = _mapping(summary.get("setting"), field="source summary setting")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", "attribution-4"),
        ("num_layers", num_layers),
    ):
        if setting.get(field) != expected:
            raise ValueError(f"source summary setting field {field!r} does not match")
    direction_summaries = _mapping(summary.get("directions"), field="source directions")
    if set(direction_summaries) != set(directions):
        raise ValueError("source summary direction set does not match its run arguments")
    for direction in directions:
        direction_summary = _mapping(
            direction_summaries.get(direction), field=f"source {direction} summary"
        )
        included = sum(
            row.get("status") == "included"
            for (sample_id, row_direction), (row, _) in all_statuses.items()
            if sample_id in eligible_ids and row_direction == direction
        )
        if (
            direction_summary.get("included_pairs") != included
            or direction_summary.get("excluded_pairs") != len(eligible_ids) - included
        ):
            raise ValueError(f"source {direction} summary counts do not match statuses")
    clean_summary = _mapping(
        direction_summaries.get("clean-to-edited"), field="source clean-to-edited summary"
    )
    if clean_summary.get("included_pairs") != len(reference_pairs):
        raise ValueError("source summary included count does not match complete grids")
    selected_failures = 0
    aligned_selected_failures = 0
    upstream_exclusions: dict[str, int] = {}
    for pair in pair_records.values():
        if not pair.clean_correct:
            reason = "clean_not_correct"
        elif not pair.edited_wrong:
            reason = "edited_not_wrong"
        else:
            selected_failures += 1
            if pair.aligned_count:
                aligned_selected_failures += 1
                continue
            reason = "no_aligned_words"
        upstream_exclusions[reason] = upstream_exclusions.get(reason, 0) + 1
    population = _mapping(summary.get("population"), field="source population")
    for field, expected in (
        ("input_pairs", len(pair_records)),
        ("selected_failures", selected_failures),
        ("aligned_selected_failures", aligned_selected_failures),
        ("eligible_before_limit", len(eligible_ids)),
        ("selected_by_limit", len(eligible_ids)),
        ("upstream_exclusions", dict(sorted(upstream_exclusions.items()))),
    ):
        if population.get(field) != expected:
            raise ValueError(f"source summary population field {field!r} does not match")
    source_comparability = _mapping(manifest.get("comparability"), field="source comparability")
    if (
        source_comparability.get("status") != "fresh-paper-protocol-reproduction"
        or source_comparability.get("exact_historical_table5_ids") is not False
    ):
        raise ValueError("source layerwise v1 comparability must be a fresh protocol reproduction")
    return _ReferenceSource(
        pairs=tuple(reference_pairs),
        run_manifest=manifest,
        run_sha256=_sha256(run_path),
        output_sha256=output_hashes,
        source_pairs_path=pairs_path,
        source_pairs_sha256=str(input_payload["pairs_sha256"]),
        source_pair_run_sha256=str(input_payload["source_run_sha256"]),
        model_revision=model_revision,
        num_layers=num_layers,
        source_comparability={
            "status": "fresh-paper-protocol-reproduction",
            "exact_historical_table5_ids": False,
        },
        upstream_population=dict(population),
    )


def _checkpoint_path(checkpoints_dir: Path, sample_id: str) -> Path:
    return checkpoints_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"


def _runtime_fingerprint(provenance: Mapping[str, object] | None) -> str:
    return _fingerprint_json(provenance)


def _validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    config: PositionControlConfig,
    source: _ReferenceSource,
) -> None:
    expected = {
        "model": config.model,
        "operation": "patch-position-controls",
        "requested_revision": source.model_revision,
        "model_revision": source.model_revision,
        "tokenizer_revision": source.model_revision,
        "num_decoder_layers": source.num_layers,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "cuda_visible_devices": config.gpu_id,
        "position_controls": POSITION_RUNTIME_PROTOCOL,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"runtime provenance field {field!r} does not match")
    if provenance.get("runtime") not in (
        "HuggingFacePositionControlRuntime",
        "position-fixture",
    ):
        raise ValueError("runtime provenance has an unsupported runtime identifier")


def _selected_plan_fingerprint(pairs: Sequence[_ReferencePair], positions: Sequence[str]) -> str:
    return _fingerprint_json(
        {
            "schema_version": "patch-position-controls-plan/v1",
            "positions": list(positions),
            "pairs": [
                {
                    "sample_id": pair.sample_id,
                    "source_record_sha256": pair.source_record_sha256,
                    "source_status_sha256": pair.source_status_sha256,
                    "source_layers_sha256": pair.source_layers_sha256,
                    "denominator_kl": pair.denominator_kl,
                }
                for pair in pairs
            ],
        }
    )


def _validate_coordinates(
    coordinates: PositionCoordinates,
    *,
    pair: _ReferencePair,
    position: str,
) -> None:
    source = coordinates.source_positions
    destination = coordinates.destination_positions
    if not source or len(source) != len(destination):
        raise ValueError(f"{position} coordinates must be non-empty and paired")
    if len(source) != len(set(source)) or len(destination) != len(set(destination)):
        raise ValueError(f"{position} coordinates must not contain duplicates")
    clean = _mapping(pair.record.get("clean"), field="pair.clean")
    edited = _mapping(pair.record.get("edited"), field="pair.edited")
    clean_count = _positive_int(clean.get("prompt_token_count"), field="clean token count")
    edited_count = _positive_int(edited.get("prompt_token_count"), field="edited token count")
    if any(index < 0 or index >= clean_count for index in source):
        raise ValueError(f"{position} clean coordinate is outside the prompt")
    if any(index < 0 or index >= edited_count for index in destination):
        raise ValueError(f"{position} edited coordinate is outside the prompt")
    if position == "prompt-final" and (
        source != (clean_count - 1,) or destination != (edited_count - 1,)
    ):
        raise ValueError("prompt-final runtime coordinates are not final prompt tokens")
    expected_locator = {
        "prompt-final": "prompt-final-token/v1",
        "question-final": "editable-prompt-span-final-overlap/v1",
    }.get(position)
    if expected_locator is not None and coordinates.locator != expected_locator:
        raise ValueError(f"{position} runtime locator does not match the public protocol")


def _process_scan(
    pair: _ReferencePair,
    scan: object,
    *,
    alternatives: tuple[str, ...],
    num_layers: int,
    runtime_fingerprint: str,
) -> dict[str, object]:
    from typo_cot.experiments.patch_position_controls.runtime import (
        PositionControlPairScan,
    )

    if not isinstance(scan, PositionControlPairScan):
        raise ValueError("runtime did not return PositionControlPairScan")
    if scan.sample_id != pair.sample_id:
        raise ValueError("runtime sample_id does not match the source pair")
    if set(scan.positions) != set(alternatives):
        raise ValueError("runtime did not return exactly the requested alternative positions")
    runtime_denominator = float(scan.denominator_kl)
    if not math.isfinite(runtime_denominator):
        raise ValueError("runtime untreated denominator is nonfinite")
    delta = runtime_denominator - pair.denominator_kl
    tolerance = max(
        _DENOMINATOR_ABS_TOLERANCE,
        abs(pair.denominator_kl) * _DENOMINATOR_REL_TOLERANCE,
    )
    if abs(delta) > tolerance:
        raise ValueError(
            "runtime untreated denominator does not match the fixed source denominator: "
            f"source={pair.denominator_kl}, runtime={runtime_denominator}, "
            f"tolerance={tolerance}"
        )
    processed: dict[str, object] = {}
    for position in alternatives:
        position_scan = scan.positions[position]
        _validate_coordinates(position_scan.coordinates, pair=pair, position=position)
        values = tuple(float(value) for value in position_scan.patched_kl_by_layer)
        if len(values) != num_layers:
            raise ValueError(f"{position} runtime returned an incomplete layer grid")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{position} runtime returned a nonfinite layer value")
        processed[position] = {
            "coordinates": {
                "source_positions": list(position_scan.coordinates.source_positions),
                "destination_positions": list(position_scan.coordinates.destination_positions),
                "locator": position_scan.coordinates.locator,
            },
            "layers": [
                {
                    "layer_index": layer_index,
                    "patched_kl": patched,
                    "normalized_kl": 1.0 - patched / pair.denominator_kl,
                }
                for layer_index, patched in enumerate(values)
            ],
        }
    return {
        "schema_version": "patch-position-controls-checkpoint/v1",
        "sample_id": pair.sample_id,
        "source_record_sha256": pair.source_record_sha256,
        "source_status_sha256": pair.source_status_sha256,
        "source_layers_sha256": pair.source_layers_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "num_layers": num_layers,
        "source_denominator_kl": pair.denominator_kl,
        "runtime_denominator_kl": runtime_denominator,
        "runtime_denominator_delta": delta,
        "positions": processed,
    }


def _reference_only_checkpoint(
    pair: _ReferencePair, *, num_layers: int, runtime_fingerprint: str
) -> dict[str, object]:
    return {
        "schema_version": "patch-position-controls-checkpoint/v1",
        "sample_id": pair.sample_id,
        "source_record_sha256": pair.source_record_sha256,
        "source_status_sha256": pair.source_status_sha256,
        "source_layers_sha256": pair.source_layers_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "num_layers": num_layers,
        "source_denominator_kl": pair.denominator_kl,
        "runtime_denominator_kl": None,
        "runtime_denominator_delta": None,
        "positions": {},
    }


def _load_checkpoint(
    path: Path,
    *,
    pair: _ReferencePair,
    runtime_fingerprint: str,
    num_layers: int,
    alternatives: Sequence[str],
) -> dict[str, object]:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "patch-position-controls-checkpoint/v1"
        or checkpoint.get("sample_id") != pair.sample_id
        or checkpoint.get("source_record_sha256") != pair.source_record_sha256
        or checkpoint.get("source_status_sha256") != pair.source_status_sha256
        or checkpoint.get("source_layers_sha256") != pair.source_layers_sha256
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("num_layers") != num_layers
        or checkpoint.get("source_denominator_kl") != pair.denominator_kl
    ):
        raise ValueError(f"checkpoint provenance does not match: {path}")
    positions = _mapping(checkpoint.get("positions"), field="checkpoint positions")
    if set(positions) != set(alternatives):
        raise ValueError(f"checkpoint position plan does not match: {path}")
    runtime_denominator = checkpoint.get("runtime_denominator_kl")
    runtime_delta = checkpoint.get("runtime_denominator_delta")
    if alternatives:
        parsed_denominator = _finite_float(
            runtime_denominator, field="checkpoint runtime_denominator_kl"
        )
        parsed_delta = _finite_float(runtime_delta, field="checkpoint runtime_denominator_delta")
        tolerance = max(
            _DENOMINATOR_ABS_TOLERANCE,
            abs(pair.denominator_kl) * _DENOMINATOR_REL_TOLERANCE,
        )
        if (
            not math.isclose(
                parsed_delta,
                parsed_denominator - pair.denominator_kl,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or abs(parsed_delta) > tolerance
        ):
            raise ValueError("checkpoint runtime denominator comparison does not match")
    elif runtime_denominator is not None or runtime_delta is not None:
        raise ValueError("reference-only checkpoint contains a runtime denominator")
    for position in alternatives:
        result = _mapping(positions[position], field=f"checkpoint {position}")
        coordinates = _mapping(result.get("coordinates"), field="checkpoint coordinates")
        parsed = PositionCoordinates(
            tuple(coordinates.get("source_positions", ())),
            tuple(coordinates.get("destination_positions", ())),
            str(coordinates.get("locator", "")),
        )
        _validate_coordinates(parsed, pair=pair, position=position)
        layers = result.get("layers")
        if not isinstance(layers, list) or len(layers) != num_layers:
            raise ValueError(f"checkpoint has an incomplete {position} layer grid")
        for expected_layer, raw_layer in enumerate(layers):
            layer = _mapping(raw_layer, field=f"checkpoint {position} layer")
            if layer.get("layer_index") != expected_layer:
                raise ValueError(f"checkpoint {position} layer grid is not ordered")
            patched = _finite_float(layer.get("patched_kl"), field="checkpoint patched_kl")
            score = _finite_float(layer.get("normalized_kl"), field="checkpoint normalized_kl")
            expected_score = 1.0 - patched / pair.denominator_kl
            if not math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("checkpoint normalized KL does not match its numerator")
    return checkpoint


def _load_checkpoint_registry(
    manifest: Mapping[str, object],
    *,
    selected: Sequence[_ReferencePair],
    checkpoints_dir: Path,
) -> dict[str, dict[str, str]]:
    raw_registry = _mapping(manifest.get("checkpoints"), field="resume checkpoints")
    selected_by_id = {pair.sample_id: pair for pair in selected}
    registry: dict[str, dict[str, str]] = {}
    for sample_id, raw_metadata in raw_registry.items():
        if sample_id not in selected_by_id:
            raise ValueError("resume checkpoint registry contains an unselected pair")
        metadata = _mapping(raw_metadata, field=f"resume checkpoint {sample_id}")
        expected_name = _checkpoint_path(checkpoints_dir, sample_id).name
        if metadata.get("file") != expected_name:
            raise ValueError("resume checkpoint registry contains a noncanonical path")
        path = checkpoints_dir / expected_name
        expected_sha256 = metadata.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or not path.is_file()
            or _sha256(path) != expected_sha256
        ):
            raise ValueError(f"registered checkpoint is missing or has changed: {path}")
        registry[sample_id] = {"file": expected_name, "sha256": expected_sha256}
    return registry


def _cleanup_orphan_checkpoints(
    checkpoints_dir: Path, registry: Mapping[str, Mapping[str, str]]
) -> None:
    expected = {metadata["file"] for metadata in registry.values()}
    for child in checkpoints_dir.iterdir():
        if child.name in expected:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _cohort_fingerprint(pairs: Sequence[_ReferencePair]) -> str:
    return _fingerprint_json([pair.sample_id for pair in pairs])


def _position_summary(scores: Sequence[tuple[float, ...]], num_layers: int) -> dict[str, object]:
    profile = [
        {
            "layer_index": layer,
            "relative_depth": layer / num_layers,
            "pairs": len(scores),
            "median_normalized_kl": statistics.median(row[layer] for row in scores),
        }
        for layer in range(num_layers)
    ]
    peak_value = max(float(row["median_normalized_kl"]) for row in profile)
    tied = [
        int(row["layer_index"])
        for row in profile
        if float(row["median_normalized_kl"]) == peak_value
    ]
    peak_layer = tied[0]
    return {
        "included_pairs": len(scores),
        "layer_profile": profile,
        "peak": {
            "layer_index": peak_layer,
            "tied_layer_indices": tied,
            "relative_depth": peak_layer / num_layers,
            "median_normalized_kl": peak_value,
        },
    }


def _coordinates_payload(coordinates: PositionCoordinates) -> dict[str, object]:
    return {
        "source_positions": list(coordinates.source_positions),
        "destination_positions": list(coordinates.destination_positions),
        "locator": coordinates.locator,
    }


def _compile_outputs(
    *,
    config: PositionControlConfig,
    source: _ReferenceSource,
    selected: Sequence[_ReferencePair],
    checkpoints_dir: Path,
    runtime_fingerprint: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    alternatives = tuple(
        position for position in config.positions if position in _ALTERNATIVE_POSITIONS
    )
    records: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    scores: dict[str, list[tuple[float, ...]]] = {position: [] for position in config.positions}
    for pair in selected:
        checkpoint = _load_checkpoint(
            _checkpoint_path(checkpoints_dir, pair.sample_id),
            pair=pair,
            runtime_fingerprint=runtime_fingerprint,
            num_layers=source.num_layers,
            alternatives=alternatives,
        )
        alternative_results = _mapping(checkpoint.get("positions"), field="checkpoint positions")
        position_coordinates: dict[str, dict[str, object]] = {}
        for position in config.positions:
            if position == "edited-word":
                coordinates = pair.aligned_coordinates
                layer_values = pair.reference_layers
                execution = "verified-reference"
            else:
                result = _mapping(alternative_results.get(position), field=f"checkpoint {position}")
                raw_coordinates = _mapping(
                    result.get("coordinates"), field=f"checkpoint {position} coordinates"
                )
                coordinates = PositionCoordinates(
                    tuple(raw_coordinates.get("source_positions", ())),
                    tuple(raw_coordinates.get("destination_positions", ())),
                    str(raw_coordinates.get("locator", "")),
                )
                layer_values = tuple(
                    dict(_mapping(row, field=f"checkpoint {position} layer"))
                    for row in result["layers"]  # type: ignore[index]
                )
                execution = "executed"
            position_coordinates[position] = _coordinates_payload(coordinates)
            pair_scores: list[float] = []
            for layer_index, layer in enumerate(layer_values):
                patched = float(layer["patched_kl"])
                score = float(layer["normalized_kl"])
                pair_scores.append(score)
                records.append(
                    {
                        "schema_version": "patch-position-controls-record/v1",
                        "paper_sha256": PAPER_SHA256,
                        "model": config.model,
                        "benchmark": config.benchmark,
                        "targeting": "attribution-4",
                        "sample_id": pair.sample_id,
                        "source_record_sha256": pair.source_record_sha256,
                        "source_status_sha256": pair.source_status_sha256,
                        "source_layers_sha256": pair.source_layers_sha256,
                        "direction": "clean-to-edited",
                        "position": position,
                        "source_positions": list(coordinates.source_positions),
                        "destination_positions": list(coordinates.destination_positions),
                        "locator": coordinates.locator,
                        "layer_index": layer_index,
                        "num_layers": source.num_layers,
                        "relative_depth": layer_index / source.num_layers,
                        "denominator_kl": pair.denominator_kl,
                        "patched_kl": patched,
                        "normalized_kl": score,
                        "execution": execution,
                    }
                )
            scores[position].append(tuple(pair_scores))
        statuses.append(
            {
                "schema_version": "patch-position-controls-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": "attribution-4",
                "sample_id": pair.sample_id,
                "source_record_sha256": pair.source_record_sha256,
                "source_status_sha256": pair.source_status_sha256,
                "source_layers_sha256": pair.source_layers_sha256,
                "status": "included",
                "direction": "clean-to-edited",
                "source_denominator_kl": pair.denominator_kl,
                "runtime_denominator_kl": checkpoint.get("runtime_denominator_kl"),
                "runtime_denominator_delta": checkpoint.get("runtime_denominator_delta"),
                "expected_layers": source.num_layers,
                "complete_positions": list(config.positions),
                "positions": position_coordinates,
            }
        )

    full_protocol = config.limit is None and config.positions == POSITION_NAMES
    source_status = source.source_comparability.get("status", "fresh-paper-protocol-reproduction")
    comparability_status = str(source_status) if full_protocol else "partial-smoke-run"
    summary = {
        "schema_version": "patch-position-controls-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-position-controls",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "targeting": "attribution-4",
            "direction": "clean-to-edited",
            "num_layers": source.num_layers,
        },
        "protocol": _PROTOCOL,
        "design_status": "post_hoc_exploratory_descriptive",
        "population": {
            "source_common_pairs": len(source.pairs),
            "executed_common_pairs": len(selected),
            "source_id_fingerprint": _cohort_fingerprint(source.pairs),
            "executed_id_fingerprint": _cohort_fingerprint(selected),
            "id_fingerprint_algorithm": "sha256-canonical-json-sorted-sample-ids/v1",
            "upstream_layerwise_population": dict(source.upstream_population),
        },
        "positions": {
            position: _position_summary(scores[position], source.num_layers)
            for position in config.positions
        },
        "published_reference": _PUBLISHED_REFERENCE,
        "comparability": {
            "status": comparability_status,
            "exact_historical_table5_ids": source.source_comparability.get(
                "exact_historical_table5_ids", False
            ),
            "requested_all_paper_positions": config.positions == POSITION_NAMES,
            "limited": config.limit is not None,
        },
        "interpretation": {
            "claim": "intervention-reachability",
            "not_claimed": ["information-movement", "encoding-location", "mechanism"],
            "position_specific_ci_or_test": False,
            "new_three_position_mcb": False,
            "edited_word_table5_mcb_preserved_as_source_metadata": True,
        },
    }
    return records, statuses, summary


def _manifest_input(config: PositionControlConfig, source: _ReferenceSource) -> dict[str, object]:
    return {
        "layerwise_kl_run": str(config.layerwise_kl_run.resolve()),
        "layerwise_run_sha256": source.run_sha256,
        "layerwise_outputs_sha256": dict(source.output_sha256),
        "pairs_path": str(source.source_pairs_path.resolve()),
        "pairs_sha256": source.source_pairs_sha256,
        "pair_source_run_sha256": source.source_pair_run_sha256,
        "source_model_revision": source.model_revision,
        "source_common_pairs": len(source.pairs),
        "source_id_fingerprint": _cohort_fingerprint(source.pairs),
    }


def _manifest_plan(
    config: PositionControlConfig,
    selected: Sequence[_ReferencePair],
    plan_fingerprint: str,
) -> dict[str, object]:
    return {
        "fingerprint": plan_fingerprint,
        "selected_pairs": len(selected),
        "selected_id_fingerprint": _cohort_fingerprint(selected),
        "positions": list(config.positions),
    }


def _manifest_comparability(
    config: PositionControlConfig, source: _ReferenceSource
) -> dict[str, object]:
    full_protocol = config.limit is None and config.positions == POSITION_NAMES
    source_status = source.source_comparability.get("status", "fresh-paper-protocol-reproduction")
    return {
        "status": str(source_status) if full_protocol else "partial-smoke-run",
        "source_status": source_status,
        "exact_historical_table5_ids": source.source_comparability.get(
            "exact_historical_table5_ids", False
        ),
    }


def _base_manifest(
    *,
    config: PositionControlConfig,
    source: _ReferenceSource,
    selected: Sequence[_ReferencePair],
    plan_fingerprint: str,
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object] | None,
    checkpoints: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "patch-position-controls-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-position-controls",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "input": _manifest_input(config, source),
        "plan": _manifest_plan(config, selected, plan_fingerprint),
        "runtime": dict(runtime_provenance) if runtime_provenance is not None else None,
        "checkpoints": {
            sample_id: dict(metadata) for sample_id, metadata in sorted(checkpoints.items())
        },
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": _manifest_comparability(config, source),
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    *,
    config: PositionControlConfig,
    source: _ReferenceSource,
    selected: Sequence[_ReferencePair],
    plan_fingerprint: str,
) -> str:
    if manifest.get("schema_version") != "patch-position-controls-run/v1":
        raise ValueError("cannot resume an unknown position-control run schema")
    if manifest.get("operation") != "patch-position-controls":
        raise ValueError("resume run has the wrong operation")
    status = manifest.get("status")
    if status not in ("running", "failed", "completed"):
        raise ValueError("resume run has an invalid status")
    if manifest.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    if manifest.get("paper_sha256") != PAPER_SHA256 or manifest.get("protocol") != _PROTOCOL:
        raise ValueError("resume paper or protocol fingerprint does not match")
    input_payload = _mapping(manifest.get("input"), field="resume input")
    if input_payload != _manifest_input(config, source):
        raise ValueError("resume source input fingerprint does not match")
    plan = _mapping(manifest.get("plan"), field="resume plan")
    if plan != _manifest_plan(config, selected, plan_fingerprint):
        raise ValueError("resume deterministic plan fingerprint does not match")
    if manifest.get("comparability") != _manifest_comparability(config, source):
        raise ValueError("resume comparability label does not match the deterministic plan")

    alternatives = tuple(
        position for position in config.positions if position in _ALTERNATIVE_POSITIONS
    )
    recorded_runtime = manifest.get("runtime")
    if alternatives:
        runtime_payload = _mapping(recorded_runtime, field="resume runtime")
        _validate_runtime_provenance(runtime_payload, config=config, source=source)
    elif recorded_runtime is not None:
        raise ValueError("reference-only resume must not contain runtime provenance")

    registry = _mapping(manifest.get("checkpoints"), field="resume checkpoints")
    selected_by_id = {pair.sample_id: pair for pair in selected}
    if not set(registry).issubset(selected_by_id):
        raise ValueError("resume checkpoint registry contains an unselected pair")
    for sample_id, raw_metadata in registry.items():
        metadata = _mapping(raw_metadata, field=f"resume checkpoint {sample_id}")
        expected_file = hashlib.sha256(sample_id.encode()).hexdigest() + ".json"
        checksum = metadata.get("sha256")
        if (
            set(metadata) != {"file", "sha256"}
            or metadata.get("file") != expected_file
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ValueError("resume checkpoint registry metadata is invalid")

    counts = _mapping(manifest.get("counts"), field="resume counts")
    expected_count_fields = {
        "source_common_pairs",
        "selected_pairs",
        "checkpointed_pairs",
        "completed_pairs",
        "failed_pairs",
        "records",
    }
    if set(counts) != expected_count_fields:
        raise ValueError("resume count fields do not match the run schema")
    parsed_counts = {
        field: _nonnegative_int(counts.get(field), field=f"resume counts.{field}")
        for field in expected_count_fields
    }
    if (
        parsed_counts["source_common_pairs"] != len(source.pairs)
        or parsed_counts["selected_pairs"] != len(selected)
        or parsed_counts["checkpointed_pairs"] != len(registry)
        or parsed_counts["completed_pairs"] != len(registry)
    ):
        raise ValueError("resume counts do not match source, plan, and checkpoints")
    failures = manifest.get("failures")
    if not isinstance(failures, list):
        raise ValueError("resume failures must be a list")
    expected_records = len(selected) * len(config.positions) * source.num_layers
    if status == "completed":
        if (
            set(registry) != set(selected_by_id)
            or parsed_counts["failed_pairs"] != 0
            or parsed_counts["records"] != expected_records
            or failures
            or "outputs" not in manifest
        ):
            raise ValueError("completed resume counts or registry do not match the full plan")
    elif status == "running":
        if parsed_counts["failed_pairs"] != 0 or parsed_counts["records"] != 0 or failures:
            raise ValueError("running resume manifest contains failed or published results")
        if "outputs" in manifest:
            raise ValueError("running resume manifest must not register public outputs")
    else:
        if (
            parsed_counts["failed_pairs"] != 1
            or parsed_counts["records"] != 0
            or len(failures) != 1
            or "outputs" in manifest
        ):
            raise ValueError("failed resume manifest has inconsistent failure state")

    started_at = manifest.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("resume started_at must be a non-empty timestamp")
    if not isinstance(manifest.get("updated_at"), str) or not manifest.get("updated_at"):
        raise ValueError("resume updated_at must be a non-empty timestamp")
    return started_at


def _completed_result(
    manifest: Mapping[str, object],
    *,
    output_dir: Path,
    expected_pairs: int,
    expected_records: int,
) -> PositionControlResult:
    outputs = _mapping(manifest.get("outputs"), field="completed run outputs")
    if set(outputs) != set(_PUBLIC_OUTPUT_NAMES):
        raise ValueError("completed run output registry does not match")
    for name in _PUBLIC_OUTPUT_NAMES:
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"completed output is missing or has changed: {path}")
    if (
        _mapping(
            outputs.get("position_control_records.jsonl"),
            field="completed position records metadata",
        ).get("records")
        != expected_records
        or _mapping(
            outputs.get("pair_status_records.jsonl"),
            field="completed pair status metadata",
        ).get("records")
        != expected_pairs
    ):
        raise ValueError("completed output record counts do not match the deterministic plan")
    counts = _mapping(manifest.get("counts"), field="completed run counts")
    return PositionControlResult(
        records_path=output_dir / "position_control_records.jsonl",
        pair_status_records_path=output_dir / "pair_status_records.jsonl",
        summary_path=output_dir / "position_control_summary.json",
        run_path=output_dir / "run.json",
        pairs=_nonnegative_int(counts.get("completed_pairs"), field="completed pairs"),
        records=_nonnegative_int(counts.get("records"), field="completed records"),
    )


def _remove_public_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            path.unlink()


def run_patch_position_controls(
    config: PositionControlConfig,
    *,
    runtime: PositionControlRuntime | None = None,
) -> PositionControlResult:
    """Validate a layerwise source, execute alternative positions, and publish outputs."""
    source = _load_reference_source(config)
    output_dir = config.output_dir
    if output_dir.resolve() == config.layerwise_kl_run.resolve():
        raise ValueError("output directory must not overwrite the source layerwise run")
    run_path = output_dir / "run.json"
    records_path = output_dir / "position_control_records.jsonl"
    statuses_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "position_control_summary.json"
    public_paths = (records_path, statuses_path, summary_path)
    work_dir = output_dir / ".patch-position-controls-work"
    checkpoints_dir = work_dir / "checkpoints"

    output_is_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if config.resume and not run_path.is_file():
        raise ValueError(f"cannot resume without the original run.json: {output_dir}")
    if output_is_nonempty and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --resume only for this run"
        )

    selected = source.pairs[: config.limit] if config.limit is not None else source.pairs
    if not selected:
        raise ValueError("source run contains no selected common pairs")
    plan_fingerprint = _selected_plan_fingerprint(selected, config.positions)
    previous: dict[str, object] | None = None
    checkpoint_registry: dict[str, dict[str, str]] = {}
    if config.resume:
        previous = _load_json(run_path)
        started_at = _validate_resume_manifest(
            previous,
            config=config,
            source=source,
            selected=selected,
            plan_fingerprint=plan_fingerprint,
        )
        if previous.get("status") == "completed":
            return _completed_result(
                previous,
                output_dir=output_dir,
                expected_pairs=len(selected),
                expected_records=len(selected) * len(config.positions) * source.num_layers,
            )
        checkpoint_registry = _load_checkpoint_registry(
            previous,
            selected=selected,
            checkpoints_dir=checkpoints_dir,
        )
    else:
        started_at = _now()

    alternatives = tuple(
        position for position in config.positions if position in _ALTERNATIVE_POSITIONS
    )
    runtime_provenance: Mapping[str, object] | None
    if alternatives:
        if runtime is None:
            from typo_cot.experiments.patch_position_controls.runtime import (
                HuggingFacePositionControlRuntime,
            )

            runtime = HuggingFacePositionControlRuntime(
                config,
                revision=source.model_revision,
            )
        if not isinstance(runtime.num_layers, int) or runtime.num_layers <= 0:
            raise ValueError("runtime num_layers must be a positive integer")
        if runtime.num_layers != source.num_layers:
            raise ValueError("runtime decoder-layer count does not match the source run")
        runtime_provenance = dict(runtime.provenance())
        _validate_runtime_provenance(
            runtime_provenance,
            config=config,
            source=source,
        )
    else:
        runtime_provenance = None
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match the original run")
    runtime_fingerprint = _runtime_fingerprint(runtime_provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_orphan_checkpoints(checkpoints_dir, checkpoint_registry)
    _remove_public_outputs(public_paths)
    for pair in selected:
        if pair.sample_id not in checkpoint_registry:
            continue
        path = _checkpoint_path(checkpoints_dir, pair.sample_id)
        _load_checkpoint(
            path,
            pair=pair,
            runtime_fingerprint=runtime_fingerprint,
            num_layers=source.num_layers,
            alternatives=alternatives,
        )
    checkpointed = len(checkpoint_registry)
    counts: dict[str, object] = {
        "source_common_pairs": len(source.pairs),
        "selected_pairs": len(selected),
        "checkpointed_pairs": checkpointed,
        "completed_pairs": checkpointed,
        "failed_pairs": 0,
        "records": 0,
    }
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            source=source,
            selected=selected,
            plan_fingerprint=plan_fingerprint,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            checkpoints=checkpoint_registry,
            counts=counts,
            failures=[],
        ),
    )

    for pair in tqdm(
        selected,
        desc="patch-position-controls",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        checkpoint_path = _checkpoint_path(checkpoints_dir, pair.sample_id)
        if pair.sample_id in checkpoint_registry:
            continue
        try:
            if alternatives:
                assert runtime is not None
                checkpoint = _process_scan(
                    pair,
                    runtime.scan_pair(pair.record, alternatives),
                    alternatives=alternatives,
                    num_layers=source.num_layers,
                    runtime_fingerprint=runtime_fingerprint,
                )
            else:
                checkpoint = _reference_only_checkpoint(
                    pair,
                    num_layers=source.num_layers,
                    runtime_fingerprint=runtime_fingerprint,
                )
            _write_json_atomic(checkpoint_path, checkpoint)
            checkpoint_registry[pair.sample_id] = {
                "file": checkpoint_path.name,
                "sha256": _sha256(checkpoint_path),
            }
            counts["checkpointed_pairs"] = int(counts["checkpointed_pairs"]) + 1
            counts["completed_pairs"] = int(counts["completed_pairs"]) + 1
            _write_json_atomic(
                run_path,
                _base_manifest(
                    config=config,
                    source=source,
                    selected=selected,
                    plan_fingerprint=plan_fingerprint,
                    status="running",
                    started_at=started_at,
                    runtime_provenance=runtime_provenance,
                    checkpoints=checkpoint_registry,
                    counts=counts,
                    failures=[],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - retain prior expensive checkpoints
            failure = {
                "sample_id": pair.sample_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            counts["failed_pairs"] = 1
            _remove_public_outputs(public_paths)
            _write_json_atomic(
                run_path,
                _base_manifest(
                    config=config,
                    source=source,
                    selected=selected,
                    plan_fingerprint=plan_fingerprint,
                    status="failed",
                    started_at=started_at,
                    runtime_provenance=runtime_provenance,
                    checkpoints=checkpoint_registry,
                    counts=counts,
                    failures=[failure],
                ),
            )
            raise PositionControlRunError(
                f"pair {pair.sample_id} failed: {exc}; inspect run.json and rerun with --resume"
            ) from exc

    try:
        records, statuses, summary = _compile_outputs(
            config=config,
            source=source,
            selected=selected,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprint=runtime_fingerprint,
        )
        _write_jsonl_atomic(records_path, records)
        _write_jsonl_atomic(statuses_path, statuses)
        _write_json_atomic(summary_path, summary)
        counts["records"] = len(records)
        output_metadata = {
            records_path.name: {"sha256": _sha256(records_path), "records": len(records)},
            statuses_path.name: {
                "sha256": _sha256(statuses_path),
                "records": len(statuses),
            },
            summary_path.name: {"sha256": _sha256(summary_path)},
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                source=source,
                selected=selected,
                plan_fingerprint=plan_fingerprint,
                status="completed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[],
                outputs=output_metadata,
            ),
        )
    except Exception as exc:
        _remove_public_outputs(public_paths)
        counts["failed_pairs"] = 1
        counts["records"] = 0
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                source=source,
                selected=selected,
                plan_fingerprint=plan_fingerprint,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[
                    {
                        "sample_id": "<output-compilation>",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            ),
        )
        raise PositionControlRunError(f"output compilation failed: {exc}") from exc

    shutil.rmtree(work_dir)
    return PositionControlResult(
        records_path=records_path,
        pair_status_records_path=statuses_path,
        summary_path=summary_path,
        run_path=run_path,
        pairs=len(selected),
        records=len(records),
    )

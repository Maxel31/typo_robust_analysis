"""Strict, resumable runner for the paper's answer-level coordinate controls."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments import fixed_window_answer_patching as fixed_api
from typo_cot.experiments.catalog import PAPER_SHA256
from typo_cot.experiments.fixed_window_answer_patching import LayerWindow
from typo_cot.experiments.fixed_window_answer_patching import runner as fixed_runner

CONTROL_NAMES = ("correct", "offset-2", "cross-item", "self-copy")
_CONTROL_ORDER = {name: index for index, name in enumerate(CONTROL_NAMES)}
_DIAGNOSTIC_CONTROLS = frozenset(CONTROL_NAMES[1:])
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_PAPER_WINDOW = LayerWindow(0, 6)

_PROTOCOL = {
    "schema_version": "patch-coordinate-controls-protocol/v1",
    "reference_denominator": (
        "completed-fixed-window-clean-to-edited-included-pairs-at-window-0:6"
    ),
    "correct": "referenced-same-item-clean-edited-word-states-at-original-coordinates",
    "offset-2": {
        "source": "same-item-clean",
        "offset_tokens": 2,
        "rule": "shift-both-source-and-write-coordinates-retain-valid-non-edited-pairs",
    },
    "cross-item": {
        "source": "different-item-clean-edited-word-states",
        "strata": ["targeting", "aligned-word-count"],
        "assignment": "sample-id-sort-cyclic-shift-one/v1",
    },
    "self-copy": ("diagnostic-extension-copy-edited-states-back-to-identical-edited-coordinates"),
    "intervention_site": "complete-decoder-block-residual-output",
    "window": "0:6",
    "direction": "clean-to-edited",
    "generation": {
        "dtype": "bfloat16",
        "decoding": "greedy",
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": 512,
        "padding_side": "left",
        "use_cache": True,
        "patch_application": "all-window-layers-on-prompt-prefill-exactly-once",
    },
    "answer_extraction": (
        "task-primary-then-empty-only-deterministic-fallback-exact-canonical-comparison/v1"
    ),
    "restoration": "extracted-patched-answer-equals-regenerated-clean-answer",
    "unextractable": "failure-retained-in-the-paired-denominator",
    "inference": "two-sided-exact-mcnemar-conditional-binomial/v1",
}

_HISTORICAL_REFERENCE = {
    "table": "Table 7",
    "setting": {"model": "google/gemma-3-4b-it", "benchmark": "gsm8k"},
    "correct": {"successes": 129, "total": 172, "rate": 0.75},
    "offset-2": {
        "successes": 44,
        "total": 172,
        "rate": 44 / 172,
        "difference_from_correct": 85 / 172,
        "mcnemar_p_value": 5.0749031687767575e-20,
    },
    "cross-item": {
        "successes": 42,
        "total": 172,
        "rate": 42 / 172,
        "difference_from_correct": 87 / 172,
        "mcnemar_p_value": 1.3281749873159826e-18,
    },
    "comparison": "descriptive-only-historical-cohort",
    "historical_cohort_identity": False,
    "note": (
        "The unpublished historical IDs and the historical locator are not reused; "
        "fresh corrected-alignment runs must not be expected to equal 129/44/42."
    ),
}


class CoordinateControlRunError(RuntimeError):
    """Raised after runtime failures while retaining verified checkpoints."""


@dataclass(frozen=True, slots=True)
class CoordinateControlConfig:
    """Frozen public arguments for one coordinate-control comparison."""

    model: str
    benchmark: Literal["gsm8k"]
    fixed_window_run: Path
    layers: tuple[LayerWindow, ...]
    controls: tuple[str, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_window_run", Path(self.fixed_window_run))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark != "gsm8k":
            raise ValueError("benchmark must be gsm8k for the paper coordinate controls")
        if len(self.layers) != 1:
            raise ValueError("layers must contain exactly one layer window")
        if not isinstance(self.layers[0], LayerWindow):
            raise TypeError("layers must contain only LayerWindow values")
        if self.layers[0] != _PAPER_WINDOW:
            raise ValueError("coordinate controls require the paper window 0:6")
        if not self.controls:
            raise ValueError("controls must contain at least one control")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("controls must not contain duplicates")
        unknown = [name for name in self.controls if name not in _CONTROL_ORDER]
        if unknown:
            raise ValueError(f"unsupported control: {unknown[0]!r}")
        if "correct" not in self.controls:
            raise ValueError("controls must include correct as the paired reference")
        if not set(self.controls).intersection(_DIAGNOSTIC_CONTROLS):
            raise ValueError("controls must include at least one diagnostic control")
        object.__setattr__(
            self,
            "controls",
            tuple(sorted(self.controls, key=_CONTROL_ORDER.__getitem__)),
        )
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""

        payload = asdict(self)
        payload["fixed_window_run"] = str(self.fixed_window_run.resolve())
        payload["layers"] = [window.label for window in self.layers]
        payload["controls"] = list(self.controls)
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class ControlScan:
    """Diagnostic generations returned for one recipient."""

    sample_id: str
    generations: Mapping[str, fixed_runner.AnswerGeneration]


class CoordinateControlRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def regenerate_baseline(self, pair: dict[str, object]) -> fixed_runner.BaselineScan: ...

    def scan_controls(
        self,
        pair: dict[str, object],
        baseline: fixed_runner.BaselineScan,
        donor_pair: dict[str, object] | None,
        window: LayerWindow,
        controls: tuple[str, ...],
    ) -> ControlScan: ...


@dataclass(frozen=True, slots=True)
class CoordinateControlResult:
    """Published paths and counts for a completed coordinate-control run."""

    records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    pairs: int
    records: int


@dataclass(frozen=True, slots=True)
class _ReferenceCorrect:
    generation: fixed_runner.AnswerGeneration
    event: bool
    recipient_generation_identical: bool


@dataclass(frozen=True, slots=True)
class _ReferencePair:
    source: Any
    baseline: fixed_runner.BaselineScan
    correct: _ReferenceCorrect

    @property
    def key(self) -> tuple[str, str]:
        return self.source.targeting, self.source.sample_id


@dataclass(frozen=True, slots=True)
class _Reference:
    path: Path
    run_sha256: str
    manifest: Mapping[str, object]
    sources: Any
    pairs: tuple[_ReferencePair, ...]
    denominator_sha256: str
    output_metadata: Mapping[str, object]
    runtime: Mapping[str, object]
    num_layers: int


# Import remains lazy so invalid references and completed resumes never load torch/model code.
HuggingFaceCoordinateControlRuntime: Any = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _load_json(path: Path) -> dict[str, object]:
    return fixed_runner._load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            value = fixed_runner._strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _generation_from_record(
    value: object,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
) -> fixed_runner.AnswerGeneration:
    row = _mapping(value, field=field)
    token_ids = row.get("token_ids")
    if not isinstance(token_ids, list):
        raise TypeError(f"{field}.token_ids must be a list")
    generation = fixed_runner.AnswerGeneration(
        token_ids=tuple(token_ids),
        text=row.get("text"),  # type: ignore[arg-type]
        value=row.get("value"),  # type: ignore[arg-type]
        is_extracted=row.get("is_extracted"),  # type: ignore[arg-type]
        is_correct=row.get("is_correct"),  # type: ignore[arg-type]
        method=row.get("method"),  # type: ignore[arg-type]
        primary_method=row.get("primary_method"),  # type: ignore[arg-type]
    )
    fixed_runner._validate_generation(
        generation,
        field=field,
        benchmark=benchmark,
        gold_answer=gold_answer,
    )
    return generation


def _baseline_from_record(
    value: object,
    *,
    sample_id: str,
    benchmark: str,
    gold_answer: str,
) -> fixed_runner.BaselineScan:
    row = _mapping(value, field="reference baseline")
    return fixed_runner.BaselineScan(
        sample_id=sample_id,
        clean=_generation_from_record(
            row.get("clean"),
            field="reference baseline.clean",
            benchmark=benchmark,
            gold_answer=gold_answer,
        ),
        edited=_generation_from_record(
            row.get("edited"),
            field="reference baseline.edited",
            benchmark=benchmark,
            gold_answer=gold_answer,
        ),
    )


def _validate_output_registry(
    run_dir: Path,
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    expected = (
        "fixed_window_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    )
    outputs = _mapping(manifest.get("outputs"), field="fixed-window outputs")
    if set(outputs) != set(expected):
        raise ValueError("fixed-window output hash registry has unexpected entries")
    validated: dict[str, Mapping[str, object]] = {}
    for name in expected:
        metadata = _mapping(outputs.get(name), field=f"fixed-window output {name}")
        path = run_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"fixed-window output hash does not match: {path}")
        records = metadata.get("records")
        if not isinstance(records, int) or isinstance(records, bool) or records <= 0:
            raise ValueError(f"fixed-window output record count is invalid: {name}")
        validated[name] = metadata
    return validated


def _fixed_config_from_manifest(
    run_dir: Path,
    manifest: Mapping[str, object],
) -> fixed_runner.FixedWindowAnswerPatchingConfig:
    arguments = _mapping(manifest.get("arguments"), field="fixed-window arguments")
    raw_pairs = arguments.get("pairs")
    raw_layers = arguments.get("layers")
    raw_directions = arguments.get("directions")
    if not isinstance(raw_pairs, list) or not all(isinstance(path, str) for path in raw_pairs):
        raise ValueError("fixed-window arguments.pairs must be a list of paths")
    if not isinstance(raw_layers, list) or not all(isinstance(value, str) for value in raw_layers):
        raise ValueError("fixed-window arguments.layers must be a list")
    if not isinstance(raw_directions, list) or not all(
        isinstance(value, str) for value in raw_directions
    ):
        raise ValueError("fixed-window arguments.directions must be a list")
    output_dir = arguments.get("output_dir")
    if not isinstance(output_dir, str):
        raise TypeError("fixed-window arguments.output_dir must be a path")
    config = fixed_runner.FixedWindowAnswerPatchingConfig(
        model=_nonempty_string(arguments.get("model"), field="fixed-window model"),
        benchmark=arguments.get("benchmark"),  # type: ignore[arg-type]
        pairs=tuple(Path(path) for path in raw_pairs),
        layers=tuple(fixed_api.parse_layer_window(value) for value in raw_layers),
        directions=tuple(raw_directions),
        output_dir=Path(output_dir),
        gpu_id=_nonempty_string(arguments.get("gpu_id"), field="fixed-window gpu_id"),
        limit=arguments.get("limit"),  # type: ignore[arg-type]
    )
    if config.output_dir.resolve() != run_dir.resolve():
        raise ValueError("fixed-window run directory does not match its recorded output_dir")
    return config


def _load_reference(config: CoordinateControlConfig) -> _Reference:
    run_dir = config.fixed_window_run
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError(f"fixed-window reference is missing run.json: {run_dir}")
    manifest = _load_json(run_path)
    if manifest.get("schema_version") != "fixed-window-answer-patching-run/v1":
        raise ValueError("fixed-window reference has an unknown run schema")
    if manifest.get("operation") != "fixed-window-answer-patching":
        raise ValueError("fixed-window reference has the wrong operation")
    if manifest.get("status") != "completed" or manifest.get("failures") != []:
        raise ValueError("fixed-window reference is not a failure-free completed run")
    if manifest.get("paper_sha256") != PAPER_SHA256:
        raise ValueError("fixed-window reference paper SHA-256 does not match")
    if manifest.get("protocol") != fixed_runner._PROTOCOL:
        raise ValueError("fixed-window reference protocol fingerprint does not match")

    fixed_config = _fixed_config_from_manifest(run_dir, manifest)
    if fixed_config.model != config.model or fixed_config.benchmark != config.benchmark:
        raise ValueError("fixed-window reference model or benchmark does not match the request")
    if "clean-to-edited" not in fixed_config.directions:
        raise ValueError("fixed-window reference has no clean-to-edited denominator")
    if config.layers[0] not in fixed_config.layers:
        raise ValueError("fixed-window reference does not contain the requested window 0:6")
    output_metadata = _validate_output_registry(run_dir, manifest)

    runtime = _mapping(manifest.get("runtime"), field="fixed-window runtime")
    num_layers = runtime.get("num_decoder_layers")
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 6:
        raise ValueError("fixed-window runtime decoder-layer count is invalid")

    recorded_input = _mapping(manifest.get("input"), field="fixed-window input")
    recorded_targetings = _mapping(
        recorded_input.get("targeting_sources"),
        field="fixed-window input targeting_sources",
    )
    for source_path in fixed_config.pairs:
        matching = [
            _mapping(value, field=f"fixed-window input source {targeting}")
            for targeting, value in recorded_targetings.items()
            if isinstance(value, Mapping) and value.get("pairs_path") == str(source_path.resolve())
        ]
        if len(matching) != 1:
            raise ValueError("fixed-window input does not identify each prepared source path")
        metadata = matching[0]
        source_run_path = source_path.parent / "run.json"
        if (
            not source_path.is_file()
            or metadata.get("pairs_sha256") != _sha256(source_path)
            or not source_run_path.is_file()
            or metadata.get("source_run_sha256") != _sha256(source_run_path)
        ):
            raise ValueError(
                f"prepared source hash does not match fixed-window input: {source_path}"
            )

    sources = fixed_runner._load_sources(fixed_config)
    if manifest.get("input") != fixed_runner._input_manifest(sources):
        raise ValueError("fixed-window source input fingerprint does not match")
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if runtime.get(field) != sources.model_revision:
            raise ValueError(f"fixed-window runtime {field} does not match pair preparation")
    selected = fixed_runner._select_anchors(sources, limit=fixed_config.limit)
    if not selected:
        raise ValueError("fixed-window reference selected no anchors")

    status_path = run_dir / "pair_status_records.jsonl"
    statuses = _load_jsonl(status_path)
    if len(statuses) != output_metadata[status_path.name].get("records"):
        raise ValueError("fixed-window pair-status record count does not match its registry")
    if len(statuses) != len(selected):
        raise ValueError("fixed-window pair-status rows do not cover every selected anchor")
    status_by_key: dict[tuple[str, str], tuple[Any, fixed_runner.BaselineScan]] = {}
    for status, source in zip(statuses, selected, strict=True):
        key = (status.get("targeting"), status.get("sample_id"))
        if key != source.key:
            raise ValueError("fixed-window pair-status identity or order does not match selection")
        if (
            status.get("schema_version") != "fixed-window-answer-patching-pair-status/v1"
            or status.get("paper_sha256") != PAPER_SHA256
            or status.get("model") != config.model
            or status.get("benchmark") != config.benchmark
            or status.get("source_record_sha256") != source.fingerprint
            or status.get("aligned_positions") != source.aligned_count
        ):
            raise ValueError("fixed-window pair-status provenance does not match its source")
        gold = _nonempty_string(source.record.get("gold_answer"), field="source gold_answer")
        baseline = _baseline_from_record(
            status.get("baseline"),
            sample_id=source.sample_id,
            benchmark=config.benchmark,
            gold_answer=gold,
        )
        direction_status = _mapping(
            status.get("direction_status"), field="fixed-window direction_status"
        )
        clean_status = _mapping(
            direction_status.get("clean-to-edited"),
            field="fixed-window direction_status.clean-to-edited",
        )
        expected_included, expected_reason = fixed_runner._direction_status(
            baseline, "clean-to-edited"
        )
        if (
            clean_status.get("included") is not expected_included
            or clean_status.get("exclusion_reason") != expected_reason
        ):
            raise ValueError("fixed-window clean-to-edited denominator status is inconsistent")
        if expected_included:
            status_by_key[source.key] = (source, baseline)
    if not status_by_key:
        raise ValueError("fixed-window clean-to-edited denominator is empty")

    records_path = run_dir / "fixed_window_records.jsonl"
    records = _load_jsonl(records_path)
    if len(records) != output_metadata[records_path.name].get("records"):
        raise ValueError("fixed-window record count does not match its registry")
    record_keys: set[tuple[object, object, object, object]] = set()
    correct_by_key: dict[tuple[str, str], _ReferenceCorrect] = {}
    for row in records:
        row_key = (
            row.get("targeting"),
            row.get("sample_id"),
            row.get("direction"),
            row.get("window"),
        )
        if row_key in record_keys:
            raise ValueError("fixed-window records contain a duplicate pair/direction/window")
        record_keys.add(row_key)
        pair_key = (str(row.get("targeting")), str(row.get("sample_id")))
        if row.get("direction") != "clean-to-edited" or row.get("window") != "0:6":
            continue
        if pair_key not in status_by_key:
            raise ValueError("fixed-window target record is outside the restoration denominator")
        source, baseline = status_by_key[pair_key]
        if (
            row.get("schema_version") != "fixed-window-answer-patching-window/v1"
            or row.get("paper_sha256") != PAPER_SHA256
            or row.get("model") != config.model
            or row.get("benchmark") != config.benchmark
            or row.get("source_record_sha256") != source.fingerprint
            or row.get("aligned_positions") != source.aligned_count
            or row.get("window_start") != 0
            or row.get("window_stop") != 6
            or row.get("num_layers") != num_layers
            or row.get("baseline")
            != {
                "clean": fixed_runner._generation_record(baseline.clean),
                "edited": fixed_runner._generation_record(baseline.edited),
            }
        ):
            raise ValueError("fixed-window correct-coordinate record provenance is inconsistent")
        gold = _nonempty_string(source.record.get("gold_answer"), field="source gold_answer")
        generation = _generation_from_record(
            row.get("patched_answer"),
            field="fixed-window correct patched_answer",
            benchmark=config.benchmark,
            gold_answer=gold,
        )
        expected_event = bool(
            generation.is_extracted
            and answers_equal(
                generation.value,
                baseline.clean.value,
                benchmark=fixed_runner._evaluation_benchmark(config.benchmark),
            )
        )
        if row.get("event") is not expected_event:
            raise ValueError("fixed-window correct-coordinate event is inconsistent")
        expected_identical = generation.token_ids == baseline.edited.token_ids
        if row.get("recipient_generation_identical") is not expected_identical:
            raise ValueError("fixed-window recipient identity flag is inconsistent")
        correct_by_key[pair_key] = _ReferenceCorrect(
            generation=generation,
            event=expected_event,
            recipient_generation_identical=expected_identical,
        )
    if set(correct_by_key) != set(status_by_key):
        raise ValueError("fixed-window correct-coordinate grid does not match its denominator")

    summary = _load_json(run_dir / "setting_summary.json")
    directions = _mapping(summary.get("directions"), field="fixed-window summary directions")
    clean_summary = _mapping(
        directions.get("clean-to-edited"), field="fixed-window summary clean-to-edited"
    )
    windows = _mapping(clean_summary.get("windows"), field="fixed-window summary windows")
    window_summary = _mapping(windows.get("0:6"), field="fixed-window summary window 0:6")
    successes = sum(correct.event for correct in correct_by_key.values())
    if (
        clean_summary.get("included_pairs") != len(status_by_key)
        or window_summary.get("total") != len(status_by_key)
        or window_summary.get("successes") != successes
        or window_summary.get("rate") != successes / len(status_by_key)
    ):
        raise ValueError("fixed-window summary does not match the referenced correct records")

    ordered_pairs = tuple(
        _ReferencePair(
            source=status_by_key[source.key][0],
            baseline=status_by_key[source.key][1],
            correct=correct_by_key[source.key],
        )
        for source in selected
        if source.key in status_by_key
    )
    denominator_payload = [
        {
            "targeting": pair.source.targeting,
            "sample_id": pair.source.sample_id,
            "source_record_sha256": pair.source.fingerprint,
        }
        for pair in ordered_pairs
    ]
    return _Reference(
        path=run_dir.resolve(),
        run_sha256=_sha256(run_path),
        manifest=manifest,
        sources=sources,
        pairs=ordered_pairs,
        denominator_sha256=_canonical_sha256(denominator_payload),
        output_metadata=output_metadata,
        runtime=runtime,
        num_layers=num_layers,
    )


@dataclass(frozen=True, slots=True)
class _PairPlan:
    reference: _ReferencePair
    clean_positions: tuple[int, ...]
    edited_positions: tuple[int, ...]
    offset: Any | None
    donor_key: tuple[str, str] | None

    @property
    def key(self) -> tuple[str, str]:
        return self.reference.key


@dataclass(frozen=True, slots=True)
class _Plans:
    all_pairs: tuple[_PairPlan, ...]
    selected: tuple[_PairPlan, ...]
    donor_map_sha256: str | None


def _pair_positions(source: Any) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    record = source.record
    aligned = record.get("aligned_words")
    if not isinstance(aligned, list) or len(aligned) != source.aligned_count:
        raise ValueError(f"source aligned-word grid is invalid: {source.key}")
    clean_positions: list[int] = []
    edited_positions: list[int] = []
    for index, raw_word in enumerate(aligned):
        word = _mapping(raw_word, field=f"source aligned_words[{index}]")
        clean = word.get("clean_final_token")
        edited = word.get("edited_final_token")
        if (
            not isinstance(clean, int)
            or isinstance(clean, bool)
            or not isinstance(edited, int)
            or isinstance(edited, bool)
        ):
            raise TypeError(f"source aligned endpoint is invalid: {source.key}")
        clean_positions.append(clean)
        edited_positions.append(edited)
    clean_payload = _mapping(record.get("clean"), field="source clean")
    edited_payload = _mapping(record.get("edited"), field="source edited")
    clean_length = clean_payload.get("prompt_token_count")
    edited_length = edited_payload.get("prompt_token_count")
    if (
        not isinstance(clean_length, int)
        or isinstance(clean_length, bool)
        or not isinstance(edited_length, int)
        or isinstance(edited_length, bool)
    ):
        raise TypeError(f"source prompt token count is invalid: {source.key}")
    return tuple(clean_positions), tuple(edited_positions), clean_length, edited_length


def _build_plans(config: CoordinateControlConfig, reference: _Reference) -> _Plans:
    from typo_cot.experiments.patch_coordinate_controls.planning import (
        DonorItem,
        assign_cross_item_donors,
        plan_offset_positions,
    )

    positions: dict[tuple[str, str], tuple[tuple[int, ...], tuple[int, ...], int, int]] = {
        pair.key: _pair_positions(pair.source) for pair in reference.pairs
    }
    offsets: dict[tuple[str, str], Any] = {}
    if "offset-2" in config.controls:
        for pair in reference.pairs:
            clean, edited, clean_length, edited_length = positions[pair.key]
            offset = plan_offset_positions(
                clean,
                edited,
                clean_length,
                edited_length,
                offset=2,
            )
            if not offset.source_positions:
                raise ValueError(
                    "offset-2 leaves no valid coordinates for reference pair "
                    f"{pair.source.targeting}/{pair.source.sample_id}; refusing to change "
                    "the paired denominator"
                )
            offsets[pair.key] = offset

    donors: dict[tuple[str, str], str] = {}
    donor_map_sha256: str | None = None
    if "cross-item" in config.controls:
        donors = assign_cross_item_donors(
            DonorItem(pair.source.targeting, pair.source.sample_id, pair.source.aligned_count)
            for pair in reference.pairs
        )
        donor_map_sha256 = _canonical_sha256(
            [
                {
                    "targeting": targeting,
                    "recipient_sample_id": sample_id,
                    "donor_sample_id": donor,
                }
                for (targeting, sample_id), donor in sorted(donors.items())
            ]
        )

    plans = tuple(
        _PairPlan(
            reference=pair,
            clean_positions=positions[pair.key][0],
            edited_positions=positions[pair.key][1],
            offset=offsets.get(pair.key),
            donor_key=(pair.source.targeting, donors[pair.key]) if pair.key in donors else None,
        )
        for pair in reference.pairs
    )
    selected = plans if config.limit is None else plans[: config.limit]
    if not selected:
        raise ValueError("coordinate-control recipient selection is empty")
    return _Plans(
        all_pairs=plans,
        selected=selected,
        donor_map_sha256=donor_map_sha256,
    )


def _plan_payload(plan: _PairPlan) -> dict[str, object]:
    payload: dict[str, object] = {
        "targeting": plan.reference.source.targeting,
        "sample_id": plan.reference.source.sample_id,
        "source_record_sha256": plan.reference.source.fingerprint,
        "clean_positions": list(plan.clean_positions),
        "edited_positions": list(plan.edited_positions),
        "donor": (
            None
            if plan.donor_key is None
            else {"targeting": plan.donor_key[0], "sample_id": plan.donor_key[1]}
        ),
    }
    if plan.offset is None:
        payload["offset"] = None
    else:
        payload["offset"] = {
            "source_positions": list(plan.offset.source_positions),
            "destination_positions": list(plan.offset.destination_positions),
            "input_pairs": plan.offset.input_pairs,
            "dropped_pairs": plan.offset.dropped_pairs,
            "offset_tokens": 2,
        }
    return payload


def _reference_metadata(reference: _Reference) -> dict[str, object]:
    outputs = {
        name: {
            "sha256": metadata.get("sha256"),
            "records": metadata.get("records"),
        }
        for name, metadata in sorted(reference.output_metadata.items())
    }
    return {
        "path": str(reference.path),
        "run_sha256": reference.run_sha256,
        "outputs": outputs,
        "direction": "clean-to-edited",
        "window": "0:6",
        "full_denominator": len(reference.pairs),
        "denominator_sha256": reference.denominator_sha256,
        "historical_cohort_identity": False,
    }


def _comparability(
    config: CoordinateControlConfig,
    reference: _Reference,
) -> dict[str, object]:
    limitations: list[str] = []
    if config.model != "google/gemma-3-4b-it":
        limitations.append("model-is-not-paper-primary-gemma-3-4b-it")
    if set(config.controls) != set(CONTROL_NAMES):
        limitations.append("not-all-published-and-diagnostic-arms-requested")
    if config.limit is not None:
        limitations.append("limit-is-smoke-only")
    reference_comparability = _mapping(
        reference.manifest.get("comparability"), field="fixed-window comparability"
    )
    if reference_comparability.get("exact_historical_cohort_identity") is not False:
        limitations.append("reference-historical-identity-claim-is-not-explicitly-false")
    if config.limit is not None:
        status = "partial-smoke-run"
    elif config.model != "google/gemma-3-4b-it":
        status = "non-paper-setting"
    elif limitations:
        status = "partial-paper-protocol"
    else:
        status = "fresh-primary-coordinate-control-run"
    return {
        "status": status,
        "requirements": {
            "paper_model": config.model == "google/gemma-3-4b-it",
            "benchmark_gsm8k": config.benchmark == "gsm8k",
            "window_0:6": config.layers == (_PAPER_WINDOW,),
            "all_arms": set(config.controls) == set(CONTROL_NAMES),
            "unlimited": config.limit is None,
            "historical_cohort_identity": False,
        },
        "limitations": limitations,
        "reference_fixed_window_status": reference_comparability.get("status"),
        "exact_historical_cohort_identity": False,
        "note": (
            "This label describes a fresh run of the final-PDF protocol on the exact "
            "referenced fixed-window denominator; it does not claim the unpublished "
            "historical 172 sample identities or historical locator."
        ),
    }


def _runtime_fingerprint(provenance: Mapping[str, object]) -> str:
    return _canonical_sha256(provenance)


def _validate_runtime(
    runtime: CoordinateControlRuntime,
    *,
    reference: _Reference,
    config: CoordinateControlConfig,
) -> tuple[dict[str, object], str]:
    if not isinstance(runtime.num_layers, int) or isinstance(runtime.num_layers, bool):
        raise TypeError("runtime num_layers must be a positive integer")
    if runtime.num_layers <= 0 or config.layers[0].stop > runtime.num_layers:
        raise ValueError("coordinate-control window is outside the runtime decoder")
    provenance = dict(runtime.provenance())
    if provenance.get("num_decoder_layers") != runtime.num_layers:
        raise ValueError("runtime provenance num_decoder_layers does not match runtime")
    revision = reference.sources.model_revision
    for field in ("requested_revision", "model_revision", "tokenizer_revision"):
        if provenance.get(field) != revision:
            raise ValueError(f"runtime {field} does not match pair preparation")
    generation = _mapping(provenance.get("generation"), field="runtime generation")
    for field, expected in (
        ("do_sample", False),
        ("max_new_tokens", 512),
        ("use_cache", True),
        ("padding_side", "left"),
    ):
        if generation.get(field) != expected or type(generation.get(field)) is not type(expected):
            raise ValueError(f"runtime generation.{field} must be {expected!r}")
    return provenance, _runtime_fingerprint(provenance)


def _checkpoint_name(plan: _PairPlan) -> str:
    identity = f"{plan.key[0]}\0{plan.key[1]}".encode()
    return f"{hashlib.sha256(identity).hexdigest()}.json"


def _checkpoint_registry_entry(path: Path, plan: _PairPlan) -> dict[str, str]:
    return {
        "sha256": _sha256(path),
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
    }


def _registry_matches(path: Path, metadata: object, plan: _PairPlan) -> bool:
    return bool(
        path.is_file()
        and isinstance(metadata, Mapping)
        and metadata.get("targeting") == plan.key[0]
        and metadata.get("sample_id") == plan.key[1]
        and metadata.get("sha256") == _sha256(path)
    )


def _baseline_checkpoint_payload(
    plan: _PairPlan,
    baseline: fixed_runner.BaselineScan,
    *,
    reference: _Reference,
    runtime_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "patch-coordinate-controls-baseline-checkpoint/v1",
        "reference_run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "baseline": {
            "clean": fixed_runner._generation_record(baseline.clean),
            "edited": fixed_runner._generation_record(baseline.edited),
        },
    }


def _load_baseline_checkpoint(
    path: Path,
    plan: _PairPlan,
    *,
    reference: _Reference,
    runtime_fingerprint: str,
    config: CoordinateControlConfig,
) -> fixed_runner.BaselineScan:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "patch-coordinate-controls-baseline-checkpoint/v1"
        or checkpoint.get("reference_run_sha256") != reference.run_sha256
        or checkpoint.get("denominator_sha256") != reference.denominator_sha256
        or checkpoint.get("targeting") != plan.key[0]
        or checkpoint.get("sample_id") != plan.key[1]
        or checkpoint.get("source_record_sha256") != plan.reference.source.fingerprint
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
    ):
        raise ValueError(f"baseline checkpoint provenance does not match: {path}")
    gold = _nonempty_string(
        plan.reference.source.record.get("gold_answer"), field="source gold_answer"
    )
    baseline = _baseline_from_record(
        checkpoint.get("baseline"),
        sample_id=plan.key[1],
        benchmark=config.benchmark,
        gold_answer=gold,
    )
    if baseline != plan.reference.baseline:
        raise ValueError(f"baseline replay does not match fixed-window reference: {plan.key}")
    return baseline


def _control_checkpoint_payload(
    plan: _PairPlan,
    scan: ControlScan,
    *,
    reference: _Reference,
    runtime_fingerprint: str,
    config: CoordinateControlConfig,
    plans: _Plans,
) -> dict[str, object]:
    diagnostics = tuple(control for control in config.controls if control != "correct")
    return {
        "schema_version": "patch-coordinate-controls-control-checkpoint/v1",
        "reference_run_sha256": reference.run_sha256,
        "denominator_sha256": reference.denominator_sha256,
        "donor_map_sha256": plans.donor_map_sha256,
        "targeting": plan.key[0],
        "sample_id": plan.key[1],
        "source_record_sha256": plan.reference.source.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "window": config.layers[0].label,
        "controls": list(diagnostics),
        "plan_sha256": _canonical_sha256(_plan_payload(plan)),
        "generations": {
            control: fixed_runner._generation_record(scan.generations[control])
            for control in diagnostics
        },
    }


def _validate_control_scan(
    scan: ControlScan,
    plan: _PairPlan,
    *,
    baseline: fixed_runner.BaselineScan,
    config: CoordinateControlConfig,
) -> None:
    diagnostics = tuple(control for control in config.controls if control != "correct")
    if not isinstance(scan, ControlScan) or scan.sample_id != plan.key[1]:
        raise ValueError("runtime ControlScan identity does not match its recipient")
    if set(scan.generations) != set(diagnostics) or len(scan.generations) != len(diagnostics):
        raise ValueError("runtime did not return exactly the requested diagnostic controls")
    gold = _nonempty_string(
        plan.reference.source.record.get("gold_answer"), field="source gold_answer"
    )
    for control in diagnostics:
        fixed_runner._validate_generation(
            scan.generations[control],
            field=f"runtime {control}",
            benchmark=config.benchmark,
            gold_answer=gold,
        )
    if "self-copy" in diagnostics and scan.generations["self-copy"] != baseline.edited:
        raise ValueError(f"self-copy must be token-identical to the edited baseline: {plan.key}")


def _load_control_checkpoint(
    path: Path,
    plan: _PairPlan,
    *,
    baseline: fixed_runner.BaselineScan,
    reference: _Reference,
    runtime_fingerprint: str,
    config: CoordinateControlConfig,
    plans: _Plans,
) -> ControlScan:
    checkpoint = _load_json(path)
    diagnostics = tuple(control for control in config.controls if control != "correct")
    if (
        checkpoint.get("schema_version") != "patch-coordinate-controls-control-checkpoint/v1"
        or checkpoint.get("reference_run_sha256") != reference.run_sha256
        or checkpoint.get("denominator_sha256") != reference.denominator_sha256
        or checkpoint.get("donor_map_sha256") != plans.donor_map_sha256
        or checkpoint.get("targeting") != plan.key[0]
        or checkpoint.get("sample_id") != plan.key[1]
        or checkpoint.get("source_record_sha256") != plan.reference.source.fingerprint
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("window") != config.layers[0].label
        or checkpoint.get("controls") != list(diagnostics)
        or checkpoint.get("plan_sha256") != _canonical_sha256(_plan_payload(plan))
    ):
        raise ValueError(f"control checkpoint provenance does not match: {path}")
    generations = _mapping(checkpoint.get("generations"), field="checkpoint generations")
    if set(generations) != set(diagnostics):
        raise ValueError("control checkpoint generation grid is incomplete")
    gold = _nonempty_string(
        plan.reference.source.record.get("gold_answer"), field="source gold_answer"
    )
    scan = ControlScan(
        sample_id=plan.key[1],
        generations={
            control: _generation_from_record(
                generations.get(control),
                field=f"checkpoint {control}",
                benchmark=config.benchmark,
                gold_answer=gold,
            )
            for control in diagnostics
        },
    )
    _validate_control_scan(scan, plan, baseline=baseline, config=config)
    return scan


def _base_manifest(
    *,
    config: CoordinateControlConfig,
    reference: _Reference,
    plans: _Plans,
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object] | None,
    comparability: Mapping[str, object],
    baseline_registry: Mapping[str, Mapping[str, str]],
    control_registry: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "patch-coordinate-controls-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-coordinate-controls",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "reference": _reference_metadata(reference),
        "plans": {
            "donor_map_sha256": plans.donor_map_sha256,
            "full_plan_sha256": _canonical_sha256(
                [_plan_payload(plan) for plan in plans.all_pairs]
            ),
            "executed_plan_sha256": _canonical_sha256(
                [_plan_payload(plan) for plan in plans.selected]
            ),
        },
        "runtime": dict(runtime_provenance) if runtime_provenance is not None else None,
        "checkpoints": {
            "baselines": {
                name: dict(metadata) for name, metadata in sorted(baseline_registry.items())
            },
            "controls": {
                name: dict(metadata) for name, metadata in sorted(control_registry.items())
            },
        },
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": dict(comparability),
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _expected_plan_manifest(plans: _Plans) -> dict[str, object]:
    return {
        "donor_map_sha256": plans.donor_map_sha256,
        "full_plan_sha256": _canonical_sha256([_plan_payload(plan) for plan in plans.all_pairs]),
        "executed_plan_sha256": _canonical_sha256([_plan_payload(plan) for plan in plans.selected]),
    }


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    *,
    config: CoordinateControlConfig,
    reference: _Reference,
    plans: _Plans,
) -> str:
    if manifest.get("schema_version") != "patch-coordinate-controls-run/v1":
        raise ValueError("cannot resume an unknown coordinate-control run schema")
    if manifest.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("cannot resume a coordinate-control run with an unknown status")
    if manifest.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    if manifest.get("paper_sha256") != PAPER_SHA256 or manifest.get("protocol") != _PROTOCOL:
        raise ValueError("resume paper or protocol fingerprint does not match")
    if manifest.get("reference") != _reference_metadata(reference):
        raise ValueError("resume fixed-window reference fingerprint does not match")
    if manifest.get("plans") != _expected_plan_manifest(plans):
        raise ValueError("resume coordinate or donor plans do not match")
    checkpoints = _mapping(manifest.get("checkpoints"), field="resume checkpoints")
    _mapping(checkpoints.get("baselines"), field="resume baseline checkpoints")
    _mapping(checkpoints.get("controls"), field="resume control checkpoints")
    started_at = manifest.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _completed_result(
    manifest: Mapping[str, object],
    *,
    output_dir: Path,
) -> CoordinateControlResult:
    names = (
        "coordinate_control_records.jsonl",
        "pair_status_records.jsonl",
        "coordinate_control_summary.json",
    )
    try:
        outputs = _mapping(manifest.get("outputs"), field="completed run outputs")
        if set(outputs) != set(names):
            raise ValueError("completed output hash registry has unexpected entries")
        for name in names:
            metadata = _mapping(outputs.get(name), field=f"completed output {name}")
            path = output_dir / name
            if not path.is_file() or metadata.get("sha256") != _sha256(path):
                raise ValueError(f"completed output hash does not match: {path}")
        counts = _mapping(manifest.get("counts"), field="completed run counts")
        pairs = counts.get("executed_pairs")
        records = counts.get("records")
        if not isinstance(pairs, int) or not isinstance(records, int):
            raise TypeError("completed output counts are invalid")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise CoordinateControlRunError(f"completed output hash validation failed: {exc}") from exc
    return CoordinateControlResult(
        records_path=output_dir / names[0],
        pair_status_records_path=output_dir / names[1],
        summary_path=output_dir / names[2],
        run_path=output_dir / "run.json",
        pairs=pairs,
        records=records,
    )


def _event(
    generation: fixed_runner.AnswerGeneration,
    baseline: fixed_runner.BaselineScan,
    *,
    benchmark: str,
) -> bool:
    return bool(
        generation.is_extracted
        and answers_equal(
            generation.value,
            baseline.clean.value,
            benchmark=fixed_runner._evaluation_benchmark(benchmark),
        )
    )


def _coordinates_for_control(
    control: str,
    plan: _PairPlan,
    by_key: Mapping[tuple[str, str], _PairPlan],
) -> dict[str, object]:
    if control == "correct":
        return {
            "source": "same-item-clean-edited-word",
            "donor_targeting": plan.key[0],
            "donor_sample_id": plan.key[1],
            "source_positions": list(plan.clean_positions),
            "destination_positions": list(plan.edited_positions),
            "input_pairs": len(plan.clean_positions),
            "dropped_pairs": 0,
            "offset_tokens": 0,
        }
    if control == "offset-2":
        if plan.offset is None:
            raise ValueError("offset-2 plan is missing")
        return {
            "source": "same-item-clean-offset",
            "donor_targeting": plan.key[0],
            "donor_sample_id": plan.key[1],
            "source_positions": list(plan.offset.source_positions),
            "destination_positions": list(plan.offset.destination_positions),
            "input_pairs": plan.offset.input_pairs,
            "dropped_pairs": plan.offset.dropped_pairs,
            "offset_tokens": 2,
        }
    if control == "cross-item":
        if plan.donor_key is None or plan.donor_key not in by_key:
            raise ValueError("cross-item donor plan is missing")
        donor = by_key[plan.donor_key]
        return {
            "source": "different-item-clean-edited-word",
            "donor_targeting": donor.key[0],
            "donor_sample_id": donor.key[1],
            "source_positions": list(donor.clean_positions),
            "destination_positions": list(plan.edited_positions),
            "input_pairs": len(plan.edited_positions),
            "dropped_pairs": 0,
            "offset_tokens": 0,
        }
    if control == "self-copy":
        return {
            "source": "same-item-edited-identity",
            "donor_targeting": plan.key[0],
            "donor_sample_id": plan.key[1],
            "source_positions": list(plan.edited_positions),
            "destination_positions": list(plan.edited_positions),
            "input_pairs": len(plan.edited_positions),
            "dropped_pairs": 0,
            "offset_tokens": 0,
        }
    raise ValueError(f"unsupported control: {control!r}")


def _compile_outputs(
    *,
    config: CoordinateControlConfig,
    reference: _Reference,
    plans: _Plans,
    baseline_dir: Path,
    control_dir: Path,
    runtime_fingerprint: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from typo_cot.experiments.patch_coordinate_controls.metrics import exact_mcnemar

    by_key = {plan.key: plan for plan in plans.all_pairs}
    selected_keys = {plan.key for plan in plans.selected}
    records: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    event_grid: dict[str, list[bool]] = {control: [] for control in config.controls}
    targeting_counts: Counter[str] = Counter()

    selected_payloads: dict[tuple[str, str], tuple[fixed_runner.BaselineScan, ControlScan]] = {}
    for plan in plans.selected:
        baseline = _load_baseline_checkpoint(
            baseline_dir / _checkpoint_name(plan),
            plan,
            reference=reference,
            runtime_fingerprint=runtime_fingerprint,
            config=config,
        )
        scan = _load_control_checkpoint(
            control_dir / _checkpoint_name(plan),
            plan,
            baseline=baseline,
            reference=reference,
            runtime_fingerprint=runtime_fingerprint,
            config=config,
            plans=plans,
        )
        selected_payloads[plan.key] = (baseline, scan)

    for plan in plans.all_pairs:
        selected = plan.key in selected_keys
        baseline_replay = selected_payloads.get(plan.key, (None, None))[0]
        statuses.append(
            {
                "schema_version": "patch-coordinate-controls-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": plan.key[0],
                "sample_id": plan.key[1],
                "source_record_sha256": plan.reference.source.fingerprint,
                "reference_denominator_included": True,
                "selected_for_execution": selected,
                "execution_status": "completed" if selected else "not-selected-by-limit",
                "aligned_positions": len(plan.edited_positions),
                "coordinate_plan": _plan_payload(plan),
                "reference_baseline": {
                    "clean": fixed_runner._generation_record(plan.reference.baseline.clean),
                    "edited": fixed_runner._generation_record(plan.reference.baseline.edited),
                },
                "baseline_replay": (
                    None
                    if baseline_replay is None
                    else {
                        "matches_reference": baseline_replay == plan.reference.baseline,
                        "clean": fixed_runner._generation_record(baseline_replay.clean),
                        "edited": fixed_runner._generation_record(baseline_replay.edited),
                    }
                ),
            }
        )
        if not selected:
            continue
        baseline, scan = selected_payloads[plan.key]
        targeting_counts[plan.key[0]] += 1
        for control in config.controls:
            if control == "correct":
                generation = plan.reference.correct.generation
                event = plan.reference.correct.event
                source_kind = "fixed-window-reference"
            else:
                generation = scan.generations[control]
                event = _event(generation, baseline, benchmark=config.benchmark)
                source_kind = "coordinate-control-runtime"
            event_grid[control].append(event)
            coordinates = _coordinates_for_control(control, plan, by_key)
            records.append(
                {
                    "schema_version": "patch-coordinate-controls-record/v1",
                    "paper_sha256": PAPER_SHA256,
                    "model": config.model,
                    "benchmark": config.benchmark,
                    "targeting": plan.key[0],
                    "sample_id": plan.key[1],
                    "source_record_sha256": plan.reference.source.fingerprint,
                    "control": control,
                    "result_source": source_kind,
                    "denominator": "fixed-window-clean-to-edited",
                    "window": config.layers[0].label,
                    "window_start": config.layers[0].start,
                    "window_stop": config.layers[0].stop,
                    "num_layers": reference.num_layers,
                    "aligned_positions": len(plan.edited_positions),
                    "coordinates": coordinates,
                    "event": event,
                    "recipient_generation_identical": generation.token_ids
                    == baseline.edited.token_ids,
                    "baseline": {
                        "clean": fixed_runner._generation_record(baseline.clean),
                        "edited": fixed_runner._generation_record(baseline.edited),
                    },
                    "patched_answer": fixed_runner._generation_record(generation),
                }
            )

    controls_summary: dict[str, object] = {}
    for control in config.controls:
        events = event_grid[control]
        successes = sum(events)
        controls_summary[control] = {
            "successes": successes,
            "total": len(events),
            "rate": successes / len(events) if events else None,
            "wilson_95_interval": fixed_runner._wilson_interval(successes, len(events)),
        }
    comparisons: dict[str, object] = {}
    for control in ("offset-2", "cross-item"):
        if control in event_grid and "correct" in event_grid:
            comparisons[f"correct-vs-{control}"] = exact_mcnemar(
                event_grid["correct"], event_grid[control]
            )
    self_events = event_grid.get("self-copy")
    summary = {
        "schema_version": "patch-coordinate-controls-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "patch-coordinate-controls",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "window": config.layers[0].label,
            "controls": list(config.controls),
        },
        "protocol": _PROTOCOL,
        "population": {
            "reference_pairs": len(reference.pairs),
            "reference_denominator_sha256": reference.denominator_sha256,
            "executed_pairs": len(plans.selected),
            "executed_by_targeting": dict(sorted(targeting_counts.items())),
        },
        "controls": controls_summary,
        "paired_exact_mcnemar": comparisons,
        "self_copy_integrity": (
            None
            if self_events is None
            else {
                "role": "diagnostic-extension-not-a-published-Table-7-answer-row",
                "pairs": len(self_events),
                "token_identical": len(self_events),
                "all_token_identical": True,
            }
        ),
        "historical_reference": _HISTORICAL_REFERENCE,
    }
    return records, statuses, summary


def _runtime_class() -> Any:
    if HuggingFaceCoordinateControlRuntime is not None:
        return HuggingFaceCoordinateControlRuntime
    from typo_cot.experiments.patch_coordinate_controls.runtime import (
        HuggingFaceCoordinateControlRuntime as runtime_class,
    )

    return runtime_class


def _remove_public_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)


def _write_running_manifest(
    run_path: Path,
    *,
    config: CoordinateControlConfig,
    reference: _Reference,
    plans: _Plans,
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object],
    comparability: Mapping[str, object],
    baseline_registry: Mapping[str, Mapping[str, str]],
    control_registry: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
) -> None:
    fixed_runner._write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            reference=reference,
            plans=plans,
            status=status,
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=failures,
        ),
    )


def run_patch_coordinate_controls(
    config: CoordinateControlConfig,
    *,
    runtime: CoordinateControlRuntime | None = None,
) -> CoordinateControlResult:
    """Evaluate coordinate/donor controls on one completed fixed-window cohort."""

    reference = _load_reference(config)
    plans = _build_plans(config, reference)
    comparability = _comparability(config, reference)

    output_dir = config.output_dir
    if output_dir.resolve() == reference.path:
        raise ValueError("output directory must not overwrite the fixed-window reference")
    run_path = output_dir / "run.json"
    records_path = output_dir / "coordinate_control_records.jsonl"
    status_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "coordinate_control_summary.json"
    checkpoint_root = output_dir / "checkpoints"
    baseline_dir = checkpoint_root / "baselines"
    control_dir = checkpoint_root / "controls"
    public_paths = (records_path, status_path, summary_path)

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
        started_at = _validate_resume_manifest(
            previous,
            config=config,
            reference=reference,
            plans=plans,
        )
        if previous.get("status") == "completed":
            return _completed_result(previous, output_dir=output_dir)
    else:
        started_at = _now()

    if runtime is None:
        runtime = _runtime_class()(config, revision=reference.sources.model_revision)
    runtime_provenance, runtime_fp = _validate_runtime(
        runtime,
        reference=reference,
        config=config,
    )
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match the original run")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    _remove_public_outputs(public_paths)

    previous_checkpoints: Mapping[str, object] = (
        _mapping(previous.get("checkpoints"), field="resume checkpoints")
        if previous is not None
        else {}
    )
    previous_baselines: Mapping[str, object] = (
        _mapping(previous_checkpoints.get("baselines"), field="resume baseline checkpoints")
        if previous is not None
        else {}
    )
    previous_controls: Mapping[str, object] = (
        _mapping(previous_checkpoints.get("controls"), field="resume control checkpoints")
        if previous is not None
        else {}
    )
    selected_names = {_checkpoint_name(plan) for plan in plans.selected}
    for directory in (baseline_dir, control_dir):
        for orphan in directory.glob("*.json"):
            if orphan.name not in selected_names:
                orphan.unlink()

    baseline_registry: dict[str, dict[str, str]] = {}
    baselines: dict[tuple[str, str], fixed_runner.BaselineScan] = {}
    for plan in plans.selected:
        name = _checkpoint_name(plan)
        path = baseline_dir / name
        reusable = _registry_matches(path, previous_baselines.get(name), plan)
        if reusable:
            try:
                baselines[plan.key] = _load_baseline_checkpoint(
                    path,
                    plan,
                    reference=reference,
                    runtime_fingerprint=runtime_fp,
                    config=config,
                )
            except (KeyError, OSError, TypeError, ValueError):
                reusable = False
        if reusable:
            baseline_registry[name] = _checkpoint_registry_entry(path, plan)
        else:
            path.unlink(missing_ok=True)

    control_registry: dict[str, dict[str, str]] = {}
    for plan in plans.selected:
        if plan.key not in baselines:
            continue
        name = _checkpoint_name(plan)
        path = control_dir / name
        reusable = _registry_matches(path, previous_controls.get(name), plan)
        if reusable:
            try:
                _load_control_checkpoint(
                    path,
                    plan,
                    baseline=baselines[plan.key],
                    reference=reference,
                    runtime_fingerprint=runtime_fp,
                    config=config,
                    plans=plans,
                )
            except (KeyError, OSError, TypeError, ValueError):
                reusable = False
        if reusable:
            control_registry[name] = _checkpoint_registry_entry(path, plan)
        else:
            path.unlink(missing_ok=True)

    counts: dict[str, object] = {
        "reference_pairs": len(reference.pairs),
        "executed_pairs": len(plans.selected),
        "baseline_checkpoints": len(baseline_registry),
        "control_checkpoints": len(control_registry),
        "records": 0,
        "failed_pairs": 0,
    }
    _write_running_manifest(
        run_path,
        config=config,
        reference=reference,
        plans=plans,
        status="running",
        started_at=started_at,
        runtime_provenance=runtime_provenance,
        comparability=comparability,
        baseline_registry=baseline_registry,
        control_registry=control_registry,
        counts=counts,
        failures=[],
    )

    baseline_failures: list[dict[str, str]] = []
    for plan in tqdm(
        plans.selected,
        desc="patch-coordinate-controls baseline replay",
        unit="pair",
        total=len(plans.selected),
        disable=None,
    ):
        name = _checkpoint_name(plan)
        if name in baseline_registry:
            continue
        path = baseline_dir / name
        try:
            baseline = runtime.regenerate_baseline(plan.reference.source.record)
            if not isinstance(baseline, fixed_runner.BaselineScan):
                raise TypeError("runtime baseline result must be BaselineScan")
            gold = _nonempty_string(
                plan.reference.source.record.get("gold_answer"), field="source gold_answer"
            )
            if baseline.sample_id != plan.key[1]:
                raise ValueError("runtime baseline sample_id does not match its source pair")
            fixed_runner._validate_generation(
                baseline.clean,
                field="baseline replay.clean",
                benchmark=config.benchmark,
                gold_answer=gold,
            )
            fixed_runner._validate_generation(
                baseline.edited,
                field="baseline replay.edited",
                benchmark=config.benchmark,
                gold_answer=gold,
            )
            if baseline != plan.reference.baseline:
                raise ValueError(
                    "baseline replay does not exactly match the fixed-window reference"
                )
            fixed_runner._write_json_atomic(
                path,
                _baseline_checkpoint_payload(
                    plan,
                    baseline,
                    reference=reference,
                    runtime_fingerprint=runtime_fp,
                ),
            )
            baselines[plan.key] = _load_baseline_checkpoint(
                path,
                plan,
                reference=reference,
                runtime_fingerprint=runtime_fp,
                config=config,
            )
            baseline_registry[name] = _checkpoint_registry_entry(path, plan)
        except Exception as exc:  # noqa: BLE001 - audit every baseline before controls
            path.unlink(missing_ok=True)
            baseline_failures.append(
                {
                    "stage": "baseline-replay",
                    "targeting": plan.key[0],
                    "sample_id": plan.key[1],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        counts["baseline_checkpoints"] = len(baseline_registry)
        counts["failed_pairs"] = len(baseline_failures)
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=baseline_failures,
        )

    if baseline_failures:
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="failed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=baseline_failures,
        )
        raise CoordinateControlRunError(
            f"baseline replay failed for {len(baseline_failures)} pair(s): "
            f"{baseline_failures[0]['message']}; inspect run.json and rerun with --resume"
        )

    diagnostics = tuple(control for control in config.controls if control != "correct")
    plan_by_key = {plan.key: plan for plan in plans.all_pairs}
    control_failures: list[dict[str, str]] = []
    for plan in tqdm(
        plans.selected,
        desc="patch-coordinate-controls",
        unit="pair",
        total=len(plans.selected),
        disable=None,
    ):
        name = _checkpoint_name(plan)
        if name in control_registry:
            continue
        path = control_dir / name
        try:
            donor_pair = (
                None
                if plan.donor_key is None
                else plan_by_key[plan.donor_key].reference.source.record
            )
            scan = runtime.scan_controls(
                plan.reference.source.record,
                baselines[plan.key],
                donor_pair,
                config.layers[0],
                diagnostics,
            )
            _validate_control_scan(
                scan,
                plan,
                baseline=baselines[plan.key],
                config=config,
            )
            fixed_runner._write_json_atomic(
                path,
                _control_checkpoint_payload(
                    plan,
                    scan,
                    reference=reference,
                    runtime_fingerprint=runtime_fp,
                    config=config,
                    plans=plans,
                ),
            )
            _load_control_checkpoint(
                path,
                plan,
                baseline=baselines[plan.key],
                reference=reference,
                runtime_fingerprint=runtime_fp,
                config=config,
                plans=plans,
            )
            control_registry[name] = _checkpoint_registry_entry(path, plan)
        except Exception as exc:  # noqa: BLE001 - preserve preceding expensive pairs
            path.unlink(missing_ok=True)
            control_failures.append(
                {
                    "stage": "control-generation",
                    "targeting": plan.key[0],
                    "sample_id": plan.key[1],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        counts["control_checkpoints"] = len(control_registry)
        counts["failed_pairs"] = len(control_failures)
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=control_failures,
        )
        if control_failures:
            break

    if control_failures:
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="failed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=control_failures,
        )
        raise CoordinateControlRunError(
            f"{len(control_failures)} coordinate-control pair(s) failed: "
            f"{control_failures[0]['message']}; inspect run.json and rerun with --resume"
        )

    try:
        records, statuses, summary = _compile_outputs(
            config=config,
            reference=reference,
            plans=plans,
            baseline_dir=baseline_dir,
            control_dir=control_dir,
            runtime_fingerprint=runtime_fp,
        )
    except Exception as exc:
        failure = {
            "stage": "output-compilation",
            "targeting": "all",
            "sample_id": "*",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        counts["failed_pairs"] = 1
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="failed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=[failure],
        )
        raise CoordinateControlRunError(
            f"output compilation failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    counts["records"] = len(records)
    counts["failed_pairs"] = 0
    _remove_public_outputs(public_paths)
    try:
        fixed_runner._write_jsonl_atomic(records_path, records)
        fixed_runner._write_jsonl_atomic(status_path, statuses)
        fixed_runner._write_json_atomic(summary_path, summary)
        outputs = {
            records_path.name: {
                "sha256": _sha256(records_path),
                "records": len(records),
            },
            status_path.name: {
                "sha256": _sha256(status_path),
                "records": len(statuses),
            },
            summary_path.name: {"sha256": _sha256(summary_path), "records": 1},
        }
        fixed_runner._write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                reference=reference,
                plans=plans,
                status="completed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=comparability,
                baseline_registry=baseline_registry,
                control_registry=control_registry,
                counts=counts,
                failures=[],
                outputs=outputs,
            ),
        )
    except Exception as exc:
        _remove_public_outputs(public_paths)
        failure = {
            "stage": "output-finalization",
            "targeting": "all",
            "sample_id": "*",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_running_manifest(
            run_path,
            config=config,
            reference=reference,
            plans=plans,
            status="failed",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=comparability,
            baseline_registry=baseline_registry,
            control_registry=control_registry,
            counts=counts,
            failures=[failure],
        )
        raise CoordinateControlRunError(
            f"output finalization failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    return CoordinateControlResult(
        records_path=records_path,
        pair_status_records_path=status_path,
        summary_path=summary_path,
        run_path=run_path,
        pairs=len(plans.selected),
        records=len(records),
    )

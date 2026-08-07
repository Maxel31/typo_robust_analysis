"""Strict, resumable runner for fixed-window free-answer patching."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.catalog import PAPER_SHA256

DIRECTION_NAMES = ("clean-to-edited", "edited-to-clean")
_DIRECTION_ORDER = {name: index for index, name in enumerate(DIRECTION_NAMES)}
_TARGETING_NAMES = ("attribution-4", "random-4")
_TARGETING_ORDER = {name: index for index, name in enumerate(_TARGETING_NAMES)}
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_LAYER_WINDOW = re.compile(r"([0-9]+):([0-9]+)")
_SELECTION_SEED = 42
_MAX_PAPER_ANCHORS = 300
_BOOTSTRAP_RESAMPLES = 10_000

_TABLE6_MODELS = frozenset(
    {
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
)
_MMLU_PRO_MODELS = frozenset(
    {
        "Qwen/Qwen2.5-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
)
_TABLE6_REFERENCES: dict[tuple[str, str], dict[str, object]] = {
    ("gemma-3-4b-it", "gsm8k"): {
        "restoration": {"denominator": 172, "successes": 129, "rate": 0.750},
        "induction": {"denominator": 187, "successes": 116, "rate": 0.620},
    },
    ("Llama-3.2-3B-Instruct", "gsm8k"): {
        "restoration": {"denominator": 232, "successes": 161, "rate": 0.694},
        "induction": {"denominator": 270, "successes": 167, "rate": 0.619},
    },
    ("Mistral-7B-Instruct-v0.3", "gsm8k"): {
        "restoration": {"denominator": 197, "successes": 93, "rate": 0.472},
        "induction": {"denominator": 234, "successes": 146, "rate": 0.624},
    },
    ("gemma-3-4b-it", "mmlu"): {
        "restoration": {"denominator": 209, "successes": 135, "rate": 0.646},
        "induction": {"denominator": 257, "successes": 144, "rate": 0.560},
    },
    ("Llama-3.2-3B-Instruct", "mmlu"): {
        "restoration": {"denominator": 226, "successes": 142, "rate": 0.628},
        "induction": {"denominator": 261, "successes": 170, "rate": 0.651},
    },
    ("Mistral-7B-Instruct-v0.3", "mmlu"): {
        "restoration": {"denominator": 205, "successes": 140, "rate": 0.683},
        "induction": {"denominator": 249, "successes": 128, "rate": 0.514},
    },
}
_TABLE7_REFERENCES: dict[str, dict[str, object]] = {
    "Qwen2.5-3B-Instruct": {
        "denominator": 97,
        "early_window_successes": 51,
        "following_window_successes": 54,
        "difference_percentage_points": -3.1,
        "confidence_interval_percentage_points": [-14.4, 8.2],
    },
    "Mistral-7B-Instruct-v0.3": {
        "denominator": 120,
        "early_window_successes": 89,
        "following_window_successes": 67,
        "difference_percentage_points": 18.3,
        "confidence_interval_percentage_points": [8.3, 28.3],
    },
}

_PROTOCOL = {
    "schema_version": "fixed-window-answer-patching-protocol/v1",
    "source_anchor": "prepared-clean-correct-edited-wrong-with-aligned-word",
    "targeting_pool": "one-or-both-prepared-attribution-4-and-random-4-arms",
    "selection": {
        "seed": _SELECTION_SEED,
        "algorithm": "sample-id-sort-then-python-random-shuffle-per-targeting/v1",
        "maximum_total": _MAX_PAPER_ANCHORS,
        "maximum_for_one_targeting": _MAX_PAPER_ANCHORS,
        "maximum_per_targeting_for_two": _MAX_PAPER_ANCHORS // 2,
    },
    "intervention_site": "complete-decoder-block-residual-output",
    "intervention_positions": "all-aligned-edited-word-final-tokens",
    "intervention_window": "all-requested-blocks-in-one-prompt-prefill",
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
        "patch_application": "prompt-prefill-exactly-once-per-window-generation",
    },
    "answer_extraction": "task-primary-then-empty-only-deterministic-fallback/v1",
    "direction_denominators": {
        "clean-to-edited": "regenerated-clean-correct-and-regenerated-edited-wrong",
        "edited-to-clean": "all-regenerated-clean-correct-anchors",
    },
    "restoration": "extracted-patched-edited-answer-equals-regenerated-clean-answer",
    "induction": "extracted-patched-clean-answer-differs-from-regenerated-clean-answer",
    "unextractable": "failure-in-both-directions-retained-in-each-direction-denominator",
    "mmlu_pro_comparison": {
        "windows": ["0:6", "6:12"],
        "direction": "clean-to-edited",
        "method": "paired-percentile-bootstrap-binary-risk-difference/v1",
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": 42,
    },
}


class FixedWindowAnswerPatchingRunError(RuntimeError):
    """Raised after runtime failures while retaining valid pair checkpoints."""


@dataclass(frozen=True, slots=True, order=True)
class LayerWindow:
    """A half-open range of zero-based decoder-block indices."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.stop, bool)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("layer window must be a non-empty non-negative half-open range")

    @property
    def label(self) -> str:
        """Return the stable command-line representation."""

        return f"{self.start}:{self.stop}"


def parse_layer_window(value: str) -> LayerWindow:
    """Parse one ``START:STOP`` half-open decoder-block range."""

    if not isinstance(value, str):
        raise TypeError("layer window must be a START:STOP string")
    match = _LAYER_WINDOW.fullmatch(value)
    if match is None:
        raise ValueError("layer window must use the unambiguous START:STOP form")
    try:
        return LayerWindow(int(match.group(1)), int(match.group(2)))
    except ValueError as exc:
        raise ValueError(f"invalid layer window {value!r}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FixedWindowAnswerPatchingConfig:
    """Frozen public arguments for one model/benchmark fixed-window run."""

    model: str
    benchmark: Literal["gsm8k", "mmlu", "mmlu-pro"]
    pairs: tuple[Path, ...]
    layers: tuple[LayerWindow, ...]
    directions: tuple[str, ...]
    output_dir: Path
    gpu_id: str = "0"
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(Path(path) for path in self.pairs))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in {"gsm8k", "mmlu", "mmlu-pro"}:
            raise ValueError("benchmark must be gsm8k, mmlu, or mmlu-pro")
        if not 1 <= len(self.pairs) <= 2:
            raise ValueError("pairs must contain one or two prepared pair files")
        if len({path.resolve() for path in self.pairs}) != len(self.pairs):
            raise ValueError("pairs must contain one or two different files")
        if not self.layers:
            raise ValueError("layers must contain at least one layer window")
        if any(not isinstance(window, LayerWindow) for window in self.layers):
            raise TypeError("layers must contain only LayerWindow values")
        windows = tuple(sorted(self.layers))
        for previous, current in zip(windows, windows[1:], strict=False):
            if current.start < previous.stop:
                raise ValueError("layer windows must not overlap")
        object.__setattr__(self, "layers", windows)
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
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""

        payload = asdict(self)
        payload["pairs"] = [str(path.resolve()) for path in self.pairs]
        payload["layers"] = [window.label for window in self.layers]
        payload["directions"] = list(self.directions)
        payload["output_dir"] = str(self.output_dir.resolve())
        payload.pop("resume")
        return payload


@dataclass(frozen=True, slots=True)
class AnswerGeneration:
    """One generated continuation and its extracted-answer provenance."""

    token_ids: tuple[int, ...]
    text: str
    value: str
    is_extracted: bool
    is_correct: bool
    method: str
    primary_method: str


@dataclass(frozen=True, slots=True)
class BaselineScan:
    """Untreated clean and edited generations for one selected anchor."""

    sample_id: str
    clean: AnswerGeneration
    edited: AnswerGeneration


@dataclass(frozen=True, slots=True)
class DirectionWindowScan:
    """One independently generated answer for each requested layer window."""

    patched_by_window: Mapping[str, AnswerGeneration]


@dataclass(frozen=True, slots=True)
class PairWindowScan:
    """All requested direction/window generations for one selected anchor."""

    sample_id: str
    directions: Mapping[str, DirectionWindowScan]


class FixedWindowAnswerPatchingRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan: ...

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        windows: tuple[LayerWindow, ...],
        directions: tuple[str, ...],
    ) -> PairWindowScan: ...


@dataclass(frozen=True, slots=True)
class FixedWindowAnswerPatchingResult:
    """Published paths and record counts for a completed setting."""

    fixed_window_records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    included_direction_pairs: int
    window_records: int


@dataclass(frozen=True, slots=True)
class _SourcePair:
    targeting: str
    sample_id: str
    record: dict[str, object]
    fingerprint: str
    prepared_clean_correct: bool
    prepared_edited_wrong: bool
    aligned_count: int

    @property
    def key(self) -> tuple[str, str]:
        return self.targeting, self.sample_id


@dataclass(frozen=True, slots=True)
class _Source:
    targeting: str
    path: Path
    pairs: tuple[_SourcePair, ...]
    pairs_sha256: str
    run_sha256: str
    manifest: dict[str, object]
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int


@dataclass(frozen=True, slots=True)
class _Sources:
    by_targeting: Mapping[str, _Source]
    model_revision: str
    dataset_records_sha256: str
    dataset_sample_count: int


# The name deliberately exists at module import time so callers can monkeypatch
# model creation.  If left as ``None``, the production class is imported only
# after source and completed-resume integrity validation.
HuggingFaceFixedWindowAnswerPatchingRuntime: Any = None


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
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


def _evaluation_benchmark(benchmark: str) -> str:
    """Map the public CLI spelling to the evaluator's internal task key."""

    return "mmlu_pro" if benchmark == "mmlu-pro" else benchmark


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _token_index(
    word: Mapping[str, object],
    *,
    side: str,
    prompt_token_count: int,
    context: str,
) -> int:
    indices = word.get(f"{side}_token_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{context}.{side}_token_indices must be a non-empty list")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
        raise ValueError(f"{context}.{side}_token_indices must contain non-negative integers")
    final_field = f"{side}_final_token"
    final = word.get(final_field)
    if not isinstance(final, int) or isinstance(final, bool) or final != indices[-1]:
        raise ValueError(f"{context}.{final_field} must equal the final token-index entry")
    if final >= prompt_token_count:
        raise ValueError(f"{context}.{final_field} is outside the recorded prompt")
    return final


def _validate_pair_record(
    record: dict[str, object],
    *,
    config: FixedWindowAnswerPatchingConfig,
    targeting: str,
    path: Path,
    line_number: int,
) -> tuple[str, bool, bool, int]:
    context = f"{path}:{line_number}"
    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise ValueError(f"{context}: unknown pair schema")
    sample_id = _nonempty_string(record.get("sample_id"), field=f"{context}.sample_id")
    for field, expected in (
        ("model", config.model),
        ("benchmark", config.benchmark),
        ("targeting", targeting),
    ):
        if record.get(field) != expected:
            raise ValueError(f"{context}: record {field} does not match its requested source")
    if record.get("seed") != 42:
        raise ValueError(f"{context}: fixed-window answer input requires seed 42")
    if record.get("num_edits_requested") != 4:
        raise ValueError(f"{context}: fixed-window answer input requires four edits")
    _nonempty_string(record.get("gold_answer"), field=f"{context}.gold_answer")

    correctness: dict[str, bool] = {}
    prompt_counts: dict[str, int] = {}
    for side in ("clean", "edited"):
        payload = _mapping(record.get(side), field=f"{context}.{side}")
        _nonempty_string(payload.get("prompt"), field=f"{context}.{side}.prompt")
        prompt_counts[side] = _positive_int(
            payload.get("prompt_token_count"),
            field=f"{context}.{side}.prompt_token_count",
        )
        answer = _mapping(payload.get("answer"), field=f"{context}.{side}.answer")
        correct = answer.get("is_correct")
        if not isinstance(correct, bool):
            raise ValueError(f"{context}.{side}.answer.is_correct must be boolean")
        correctness[side] = correct

    aligned_words = record.get("aligned_words")
    if not isinstance(aligned_words, list):
        raise ValueError(f"{context}.aligned_words must be a list")
    if record.get("num_aligned_words") != len(aligned_words):
        raise ValueError(f"{context}.num_aligned_words does not match aligned_words")
    positions: dict[str, list[int]] = {"clean": [], "edited": []}
    for index, raw_word in enumerate(aligned_words):
        word = _mapping(raw_word, field=f"{context}.aligned_words[{index}]")
        for side in ("clean", "edited"):
            positions[side].append(
                _token_index(
                    word,
                    side=side,
                    prompt_token_count=prompt_counts[side],
                    context=f"{context}.aligned_words[{index}]",
                )
            )
    for side, side_positions in positions.items():
        if len(side_positions) != len(set(side_positions)):
            raise ValueError(f"{context}: duplicate {side} aligned final-token coordinate")
    return (
        sample_id,
        correctness["clean"],
        not correctness["edited"],
        len(aligned_words),
    )


def _source_targeting(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"pairs input is not a file: {path}")
    run_path = path.parent / "run.json"
    if not run_path.is_file():
        raise ValueError(f"pairs input is missing sibling run.json: {run_path}")
    manifest = _load_json(run_path)
    arguments = _mapping(manifest.get("arguments"), field="source run arguments")
    targeting = arguments.get("targeting")
    if targeting not in _TARGETING_ORDER:
        raise ValueError("source run targeting must be attribution-4 or random-4")
    return str(targeting)


def _load_source(
    path: Path,
    *,
    targeting: str,
    config: FixedWindowAnswerPatchingConfig,
) -> _Source:
    run_path = path.parent / "run.json"
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
        ("targeting", targeting),
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
    dataset_records_sha256 = _nonempty_string(
        provenance.get("dataset_records_sha256"),
        field="source run provenance.dataset_records_sha256",
    )
    dataset_sample_count = _positive_int(
        provenance.get("dataset_sample_count"),
        field="source run provenance.dataset_sample_count",
    )
    counts = _mapping(manifest.get("counts"), field="source run counts")
    discovered = _positive_int(counts.get("discovered"), field="source run counts.discovered")
    written = _positive_int(counts.get("written"), field="source run counts.written")
    if counts.get("failed") != 0 or manifest.get("failures") not in ([], None):
        raise ValueError("source run contains failures")
    if discovered != written or discovered != dataset_sample_count:
        raise ValueError(
            "source run must contain the complete dataset: discovered, written, "
            "and dataset_sample_count must match"
        )

    source_pairs: list[_SourcePair] = []
    previous_id: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = _strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"pair record must be an object at {path}:{line_number}")
            (
                sample_id,
                clean_correct,
                edited_wrong,
                aligned_count,
            ) = _validate_pair_record(
                payload,
                config=config,
                targeting=targeting,
                path=path,
                line_number=line_number,
            )
            if previous_id is not None and sample_id <= previous_id:
                raise ValueError("pair sample IDs must be strictly sorted and unique")
            previous_id = sample_id
            source_pairs.append(
                _SourcePair(
                    targeting=targeting,
                    sample_id=sample_id,
                    record=payload,
                    fingerprint=_record_fingerprint(line),
                    prepared_clean_correct=clean_correct,
                    prepared_edited_wrong=edited_wrong,
                    aligned_count=aligned_count,
                )
            )
    if not source_pairs:
        raise ValueError("pairs input contains no records")
    if written != len(source_pairs):
        raise ValueError("source run written count does not match pairs.jsonl")
    return _Source(
        targeting=targeting,
        path=path,
        pairs=tuple(source_pairs),
        pairs_sha256=_sha256(path),
        run_sha256=_sha256(run_path),
        manifest=manifest,
        model_revision=model_revision,
        dataset_records_sha256=dataset_records_sha256,
        dataset_sample_count=dataset_sample_count,
    )


def _load_sources(config: FixedWindowAnswerPatchingConfig) -> _Sources:
    detected = [(_source_targeting(path), path) for path in config.pairs]
    targetings = [targeting for targeting, _ in detected]
    if len(set(targetings)) != len(targetings):
        raise ValueError("pair sources contain a duplicate targeting arm")
    sources = {
        targeting: _load_source(path, targeting=targeting, config=config)
        for targeting, path in detected
    }
    revisions = {source.model_revision for source in sources.values()}
    if len(revisions) != 1:
        raise ValueError("source model revisions differ between targeting arms")
    dataset_hashes = {source.dataset_records_sha256 for source in sources.values()}
    dataset_counts = {source.dataset_sample_count for source in sources.values()}
    if len(dataset_hashes) != 1 or len(dataset_counts) != 1:
        raise ValueError("source dataset fingerprints differ between targeting arms")
    return _Sources(
        by_targeting=sources,
        model_revision=next(iter(revisions)),
        dataset_records_sha256=next(iter(dataset_hashes)),
        dataset_sample_count=next(iter(dataset_counts)),
    )


def _eligible(pair: _SourcePair) -> bool:
    return bool(pair.prepared_clean_correct and pair.prepared_edited_wrong and pair.aligned_count)


def _select_anchors(
    sources: _Sources,
    *,
    limit: int | None,
) -> tuple[_SourcePair, ...]:
    cap = _MAX_PAPER_ANCHORS if len(sources.by_targeting) == 1 else _MAX_PAPER_ANCHORS // 2
    selected_by_targeting: dict[str, list[_SourcePair]] = {}
    for targeting in sorted(sources.by_targeting, key=_TARGETING_ORDER.__getitem__):
        eligible = [pair for pair in sources.by_targeting[targeting].pairs if _eligible(pair)]
        eligible.sort(key=lambda pair: pair.sample_id)
        random.Random(_SELECTION_SEED).shuffle(eligible)
        selected_by_targeting[targeting] = eligible[:cap]
    selected: list[_SourcePair] = []
    if limit is not None:
        depth = 0
        targetings = sorted(selected_by_targeting, key=_TARGETING_ORDER.__getitem__)
        while len(selected) < limit:
            added = False
            for targeting in targetings:
                candidates = selected_by_targeting[targeting]
                if depth < len(candidates):
                    selected.append(candidates[depth])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            depth += 1
    else:
        selected = [
            pair
            for targeting in sorted(selected_by_targeting, key=_TARGETING_ORDER.__getitem__)
            for pair in selected_by_targeting[targeting]
        ]
    selected.sort(key=lambda pair: (_TARGETING_ORDER[pair.targeting], pair.sample_id))
    return tuple(selected)


def _input_manifest(sources: _Sources) -> dict[str, object]:
    return {
        "source_model_revision": sources.model_revision,
        "dataset_records_sha256": sources.dataset_records_sha256,
        "dataset_sample_count": sources.dataset_sample_count,
        "targeting_sources": {
            targeting: {
                "pairs_path": str(source.path.resolve()),
                "pairs_sha256": source.pairs_sha256,
                "source_run_sha256": source.run_sha256,
                "source_record_count": len(source.pairs),
                "source_schema": "prepare-edited-pairs/v1",
            }
            for targeting, source in sorted(
                sources.by_targeting.items(),
                key=lambda item: _TARGETING_ORDER[item[0]],
            )
        },
    }


def _checkpoint_path(checkpoints_dir: Path, pair: _SourcePair) -> Path:
    identity = f"{pair.targeting}\0{pair.sample_id}".encode("utf-8")
    return checkpoints_dir / f"{hashlib.sha256(identity).hexdigest()}.json"


def _generation_record(generation: AnswerGeneration) -> dict[str, object]:
    return {
        "token_ids": list(generation.token_ids),
        "text": generation.text,
        "value": generation.value,
        "is_extracted": generation.is_extracted,
        "is_correct": generation.is_correct,
        "method": generation.method,
        "primary_method": generation.primary_method,
    }


def _validate_generation(
    generation: AnswerGeneration,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
) -> None:
    if not isinstance(generation, AnswerGeneration):
        raise ValueError(f"{field} must be an AnswerGeneration")
    if not generation.token_ids or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in generation.token_ids
    ):
        raise ValueError(f"{field}.token_ids must contain non-negative integers")
    if not isinstance(generation.text, str):
        raise ValueError(f"{field}.text must be a string")
    if not isinstance(generation.value, str):
        raise ValueError(f"{field}.value must be a string")
    if generation.is_extracted is not bool(generation.value):
        raise ValueError(f"{field}.is_extracted must agree with the answer value")
    if not isinstance(generation.is_correct, bool) or generation.is_correct is not answers_equal(
        generation.value,
        gold_answer,
        benchmark=_evaluation_benchmark(benchmark),
    ):
        raise ValueError(f"{field}.is_correct does not match the recorded answer")
    if not generation.method or not generation.primary_method:
        raise ValueError(f"{field} extraction provenance must be non-empty")


def _direction_status(
    baseline: BaselineScan,
    direction: str,
) -> tuple[bool, str | None]:
    if not baseline.clean.is_correct:
        return False, "regenerated_clean_not_correct"
    if direction == "clean-to-edited" and baseline.edited.is_correct:
        return False, "regenerated_edited_not_wrong"
    return True, None


def _runtime_fingerprint(provenance: Mapping[str, object]) -> str:
    serialized = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _process_pair(
    source_pair: _SourcePair,
    *,
    baseline: BaselineScan,
    scan: PairWindowScan | None,
    included_directions: tuple[str, ...],
    config: FixedWindowAnswerPatchingConfig,
    n_layers: int,
    runtime_fingerprint: str,
) -> dict[str, object]:
    if baseline.sample_id != source_pair.sample_id:
        raise ValueError("runtime BaselineScan sample_id does not match its source pair")
    gold_answer = _nonempty_string(
        source_pair.record.get("gold_answer"),
        field="source pair gold_answer",
    )
    _validate_generation(
        baseline.clean,
        field="baseline.clean",
        benchmark=config.benchmark,
        gold_answer=gold_answer,
    )
    _validate_generation(
        baseline.edited,
        field="baseline.edited",
        benchmark=config.benchmark,
        gold_answer=gold_answer,
    )
    statuses = {
        direction: _direction_status(baseline, direction) for direction in config.directions
    }
    expected_included = tuple(
        direction for direction in config.directions if statuses[direction][0]
    )
    if included_directions != expected_included:
        raise ValueError("runtime direction request does not match regenerated denominators")

    directions: dict[str, object] = {}
    if included_directions:
        if scan is None:
            raise ValueError("included pair-direction is missing its window scan")
        if scan.sample_id != source_pair.sample_id:
            raise ValueError("runtime PairWindowScan sample_id does not match its source pair")
        if set(scan.directions) != set(included_directions):
            raise ValueError("runtime did not return exactly the included directions")
        for direction in included_directions:
            direction_scan = scan.directions[direction]
            if not isinstance(direction_scan, DirectionWindowScan):
                raise ValueError(f"runtime {direction} result must be DirectionWindowScan")
            generations = direction_scan.patched_by_window
            expected_labels = [window.label for window in config.layers]
            if set(generations) != set(expected_labels) or len(generations) != len(expected_labels):
                raise ValueError(f"runtime {direction} result is not the requested window grid")
            recipient = baseline.edited if direction == "clean-to-edited" else baseline.clean
            window_rows: list[dict[str, object]] = []
            for window in config.layers:
                generation = generations[window.label]
                _validate_generation(
                    generation,
                    field=f"{direction}.window[{window.label}]",
                    benchmark=config.benchmark,
                    gold_answer=gold_answer,
                )
                equal_to_clean = answers_equal(
                    generation.value,
                    baseline.clean.value,
                    benchmark=_evaluation_benchmark(config.benchmark),
                )
                event = bool(
                    generation.is_extracted
                    and (equal_to_clean if direction == "clean-to-edited" else not equal_to_clean)
                )
                window_rows.append(
                    {
                        "window": window.label,
                        "window_start": window.start,
                        "window_stop": window.stop,
                        "event": event,
                        "recipient_generation_identical": generation.token_ids
                        == recipient.token_ids,
                        "patched_answer": _generation_record(generation),
                    }
                )
            directions[direction] = {"windows": window_rows}
    elif scan is not None:
        raise ValueError("runtime returned a window scan for an excluded pair")

    return {
        "schema_version": "fixed-window-answer-patching-checkpoint/v1",
        "targeting": source_pair.targeting,
        "sample_id": source_pair.sample_id,
        "source_record_sha256": source_pair.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "num_layers": n_layers,
        "layer_windows": [window.label for window in config.layers],
        "aligned_positions": source_pair.aligned_count,
        "baseline": {
            "clean": _generation_record(baseline.clean),
            "edited": _generation_record(baseline.edited),
        },
        "direction_status": {
            direction: {
                "included": statuses[direction][0],
                "exclusion_reason": statuses[direction][1],
            }
            for direction in config.directions
        },
        "directions": directions,
    }


def _validate_checkpoint_generation(
    value: object,
    *,
    field: str,
    benchmark: str,
    gold_answer: str,
) -> Mapping[str, object]:
    generation = _mapping(value, field=field)
    token_ids = generation.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in token_ids
        )
    ):
        raise ValueError(f"{field}.token_ids must contain non-negative integers")
    text = generation.get("text")
    answer = generation.get("value")
    if not isinstance(text, str) or not isinstance(answer, str):
        raise ValueError(f"{field}.text and value must be strings")
    extracted = generation.get("is_extracted")
    if not isinstance(extracted, bool) or extracted is not bool(answer):
        raise ValueError(f"{field}.is_extracted must agree with the answer value")
    correct = generation.get("is_correct")
    if not isinstance(correct, bool) or correct is not answers_equal(
        answer,
        gold_answer,
        benchmark=_evaluation_benchmark(benchmark),
    ):
        raise ValueError(f"{field}.is_correct does not match the recorded answer")
    for provenance_field in ("method", "primary_method"):
        if not isinstance(generation.get(provenance_field), str) or not generation.get(
            provenance_field
        ):
            raise ValueError(f"{field}.{provenance_field} must be non-empty")
    return generation


def _load_checkpoint(
    path: Path,
    *,
    source_pair: _SourcePair,
    runtime_fingerprint: str,
    n_layers: int,
    config: FixedWindowAnswerPatchingConfig,
) -> dict[str, object]:
    checkpoint = _load_json(path)
    expected_windows = [window.label for window in config.layers]
    if (
        checkpoint.get("schema_version") != "fixed-window-answer-patching-checkpoint/v1"
        or checkpoint.get("targeting") != source_pair.targeting
        or checkpoint.get("sample_id") != source_pair.sample_id
        or checkpoint.get("source_record_sha256") != source_pair.fingerprint
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("num_layers") != n_layers
        or checkpoint.get("layer_windows") != expected_windows
        or checkpoint.get("aligned_positions") != source_pair.aligned_count
    ):
        raise ValueError(f"checkpoint provenance does not match: {path}")

    gold_answer = _nonempty_string(
        source_pair.record.get("gold_answer"),
        field="source pair gold_answer",
    )
    baseline = _mapping(checkpoint.get("baseline"), field="checkpoint baseline")
    clean = _validate_checkpoint_generation(
        baseline.get("clean"),
        field="checkpoint baseline.clean",
        benchmark=config.benchmark,
        gold_answer=gold_answer,
    )
    edited = _validate_checkpoint_generation(
        baseline.get("edited"),
        field="checkpoint baseline.edited",
        benchmark=config.benchmark,
        gold_answer=gold_answer,
    )
    direction_status = _mapping(
        checkpoint.get("direction_status"),
        field="checkpoint direction_status",
    )
    if set(direction_status) != set(config.directions):
        raise ValueError("checkpoint direction statuses do not match the request")
    included_directions: list[str] = []
    for direction in config.directions:
        status = _mapping(
            direction_status.get(direction),
            field=f"checkpoint direction_status.{direction}",
        )
        expected_included = clean.get("is_correct") is True and not (
            direction == "clean-to-edited" and edited.get("is_correct") is True
        )
        expected_reason = (
            None
            if expected_included
            else "regenerated_clean_not_correct"
            if clean.get("is_correct") is False
            else "regenerated_edited_not_wrong"
        )
        if (
            status.get("included") is not expected_included
            or status.get("exclusion_reason") != expected_reason
        ):
            raise ValueError(f"checkpoint {direction} denominator status is inconsistent")
        if expected_included:
            included_directions.append(direction)

    directions = _mapping(checkpoint.get("directions"), field="checkpoint directions")
    if set(directions) != set(included_directions):
        raise ValueError("checkpoint generations do not match included directions")
    for direction in included_directions:
        direction_payload = _mapping(
            directions.get(direction),
            field=f"checkpoint {direction}",
        )
        windows = direction_payload.get("windows")
        if not isinstance(windows, list) or len(windows) != len(config.layers):
            raise ValueError(f"checkpoint {direction} is not a complete window grid")
        recipient = edited if direction == "clean-to-edited" else clean
        for expected_window, raw_window in zip(config.layers, windows, strict=True):
            window = _mapping(raw_window, field=f"checkpoint {direction} window")
            if (
                window.get("window") != expected_window.label
                or window.get("window_start") != expected_window.start
                or window.get("window_stop") != expected_window.stop
            ):
                raise ValueError(f"checkpoint {direction} window grid is not exact and ordered")
            event = window.get("event")
            identical = window.get("recipient_generation_identical")
            if not isinstance(event, bool) or not isinstance(identical, bool):
                raise ValueError(f"checkpoint {direction} window flags must be boolean")
            patched = _validate_checkpoint_generation(
                window.get("patched_answer"),
                field=f"checkpoint {direction} window[{expected_window.label}]",
                benchmark=config.benchmark,
                gold_answer=gold_answer,
            )
            if identical is not (patched.get("token_ids") == recipient.get("token_ids")):
                raise ValueError(f"checkpoint {direction} recipient identity flag is inconsistent")
            equal_to_clean = answers_equal(
                str(patched.get("value")),
                str(clean.get("value")),
                benchmark=_evaluation_benchmark(config.benchmark),
            )
            expected_event = bool(
                patched.get("is_extracted")
                and (equal_to_clean if direction == "clean-to-edited" else not equal_to_clean)
            )
            if event is not expected_event:
                raise ValueError(f"checkpoint {direction} event is inconsistent")
    return checkpoint


def _checkpoint_registry_entry(
    path: Path,
    *,
    source_pair: _SourcePair,
) -> dict[str, str]:
    return {
        "sha256": _sha256(path),
        "targeting": source_pair.targeting,
        "sample_id": source_pair.sample_id,
    }


def _checkpoint_registry_matches(
    path: Path,
    *,
    source_pair: _SourcePair,
    metadata: object,
) -> bool:
    if not path.is_file() or not isinstance(metadata, Mapping):
        return False
    return (
        metadata.get("targeting") == source_pair.targeting
        and metadata.get("sample_id") == source_pair.sample_id
        and metadata.get("sha256") == _sha256(path)
    )


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    from typo_cot.experiments.fixed_window_answer_patching.metrics import (
        wilson_interval,
    )

    return list(wilson_interval(successes, total))


def _comparability(
    config: FixedWindowAnswerPatchingConfig,
    sources: _Sources,
    selected: Sequence[_SourcePair],
    *,
    included_by_direction: Mapping[str, int] | None = None,
) -> dict[str, object]:
    arms = set(sources.by_targeting)
    windows = tuple(window.label for window in config.layers)
    directions = tuple(config.directions)
    selected_counts = Counter(pair.targeting for pair in selected)

    if config.benchmark == "mmlu-pro":
        expected_model = config.model in _MMLU_PRO_MODELS
        expected_arms = arms == {"attribution-4"}
        expected_windows = windows == ("0:6", "6:12")
        expected_directions = directions == ("clean-to-edited",)
        paper_setting = expected_model
    else:
        expected_model = config.model in _TABLE6_MODELS
        expected_arms = arms == set(_TARGETING_NAMES)
        expected_windows = windows == ("0:6",)
        expected_directions = directions == DIRECTION_NAMES
        paper_setting = expected_model and config.benchmark in {"gsm8k", "mmlu"}

    limitations: list[str] = []
    if not expected_model:
        limitations.append("model-not-in-paper-fixed-window-settings")
    if not expected_arms:
        limitations.append("targeting-arms-do-not-match-prespecified-setting")
    if not expected_windows:
        limitations.append("layer-windows-do-not-match-prespecified-setting")
    if not expected_directions:
        limitations.append("directions-do-not-match-prespecified-setting")
    for targeting in sorted(arms, key=_TARGETING_ORDER.__getitem__):
        if selected_counts[targeting] == 0:
            limitations.append(f"no-selected-{targeting}-anchors")
    if included_by_direction is not None:
        for direction in config.directions:
            if included_by_direction.get(direction, 0) == 0:
                limitations.append(f"no-regenerated-{direction}-denominator")
    if config.limit is not None:
        limitations.append("limit-is-smoke-only")

    if config.limit is not None:
        status = "partial-smoke-run"
    elif not paper_setting:
        status = "non-paper-setting"
    elif limitations:
        status = "partial-paper-protocol"
    elif config.benchmark == "mmlu-pro":
        status = "prespecified-mmlu-pro-window-comparison"
    else:
        status = "fresh-paper-protocol-reproduction"
    return {
        "status": status,
        "requirements": {
            "paper_model": expected_model,
            "targeting_arms": expected_arms,
            "layer_windows": expected_windows,
            "directions": expected_directions,
            "unlimited": config.limit is None,
        },
        "limitations": limitations,
        "targeting_arms": sorted(arms, key=_TARGETING_ORDER.__getitem__),
        "selected_by_targeting": {
            targeting: selected_counts[targeting]
            for targeting in sorted(arms, key=_TARGETING_ORDER.__getitem__)
        },
        "included_by_direction": (
            None
            if included_by_direction is None
            else {
                direction: included_by_direction.get(direction, 0)
                for direction in config.directions
            }
        ),
        "exact_historical_table6_ids": False,
        "note": (
            "Fresh runs use final-PDF extracted-answer events. Published Table 6 "
            "values are descriptive metadata because historical induction aggregation "
            "treated some unextractable generations differently."
        ),
    }


def _base_manifest(
    *,
    config: FixedWindowAnswerPatchingConfig,
    sources: _Sources,
    status: str,
    started_at: str,
    runtime_provenance: Mapping[str, object] | None,
    comparability: Mapping[str, object],
    checkpoints: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, object],
    failures: Sequence[Mapping[str, str]],
    outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "fixed-window-answer-patching-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "fixed-window-answer-patching",
        "status": status,
        "arguments": config.public_arguments(),
        "protocol": _PROTOCOL,
        "input": _input_manifest(sources),
        "runtime": dict(runtime_provenance) if runtime_provenance is not None else None,
        "checkpoints": {name: dict(metadata) for name, metadata in sorted(checkpoints.items())},
        "counts": dict(counts),
        "failures": list(failures),
        "comparability": dict(comparability),
        "started_at": started_at,
        "updated_at": _now(),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    return payload


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    *,
    config: FixedWindowAnswerPatchingConfig,
    sources: _Sources,
) -> str:
    if manifest.get("schema_version") != "fixed-window-answer-patching-run/v1":
        raise ValueError("cannot resume an unknown fixed-window answer run schema")
    if manifest.get("status") not in {"running", "failed", "completed"}:
        raise ValueError("cannot resume a run with an unknown status")
    if manifest.get("arguments") != config.public_arguments():
        raise ValueError("resume arguments do not match the existing run.json")
    if manifest.get("paper_sha256") != PAPER_SHA256 or manifest.get("protocol") != _PROTOCOL:
        raise ValueError("resume paper or protocol fingerprint does not match")
    if manifest.get("input") != _input_manifest(sources):
        raise ValueError("resume source input fingerprint does not match")
    if not isinstance(manifest.get("checkpoints"), Mapping):
        raise ValueError("resume manifest has no checkpoint integrity registry")
    started_at = manifest.get("started_at")
    return started_at if isinstance(started_at, str) else _now()


def _completed_result(
    manifest: Mapping[str, object],
    *,
    output_dir: Path,
) -> FixedWindowAnswerPatchingResult:
    names = (
        "fixed_window_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
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
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise FixedWindowAnswerPatchingRunError(
            f"completed output hash validation failed: {exc}"
        ) from exc
    return FixedWindowAnswerPatchingResult(
        fixed_window_records_path=output_dir / names[0],
        pair_status_records_path=output_dir / names[1],
        summary_path=output_dir / names[2],
        run_path=output_dir / "run.json",
        included_direction_pairs=int(counts.get("included_direction_pairs", 0)),
        window_records=int(counts.get("window_records", 0)),
    )


def _historical_reference(config: FixedWindowAnswerPatchingConfig) -> dict[str, object]:
    basename = config.model.rsplit("/", 1)[-1]
    if config.benchmark == "mmlu-pro":
        return {
            "table": "Table 7",
            "published_setting": _TABLE7_REFERENCES.get(basename),
            "comparison": "descriptive-only-fresh-public-cohort",
            "historical_bootstrap_discrepancy": True,
            "induction_unextractable_discrepancy": False,
            "note": (
                "Fresh runs use the stated 10,000 pair-resampled percentile bootstrap; "
                "the published interval is retained as historical reference metadata."
            ),
        }
    table6 = _TABLE6_REFERENCES.get((basename, config.benchmark))
    return {
        "table": "Table 6",
        "published_setting": table6,
        "comparison": "descriptive-only-fresh-public-cohort",
        "historical_bootstrap_discrepancy": False,
        "induction_unextractable_discrepancy": True,
        "note": (
            "The historical induction aggregate counted some unextractable patched "
            "answers as changes; this runner follows the final PDF and records them as failures."
        ),
    }


def _compile_outputs(
    *,
    config: FixedWindowAnswerPatchingConfig,
    sources: _Sources,
    selected: Sequence[_SourcePair],
    checkpoints_dir: Path,
    runtime_fingerprint: str,
    n_layers: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    Counter[str],
]:
    window_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    event_grids: dict[str, dict[str, list[bool]]] = {
        direction: {window.label: [] for window in config.layers} for direction in config.directions
    }
    included_by_direction: Counter[str] = Counter()
    included_by_targeting: dict[str, Counter[str]] = {
        direction: Counter() for direction in config.directions
    }
    exclusions: dict[str, Counter[str]] = {direction: Counter() for direction in config.directions}

    for pair in selected:
        checkpoint = _load_checkpoint(
            _checkpoint_path(checkpoints_dir, pair),
            source_pair=pair,
            runtime_fingerprint=runtime_fingerprint,
            n_layers=n_layers,
            config=config,
        )
        direction_status = _mapping(
            checkpoint.get("direction_status"),
            field="checkpoint direction_status",
        )
        status_rows.append(
            {
                "schema_version": "fixed-window-answer-patching-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": pair.targeting,
                "sample_id": pair.sample_id,
                "source_record_sha256": pair.fingerprint,
                "aligned_positions": pair.aligned_count,
                "direction_status": direction_status,
                "baseline": checkpoint.get("baseline"),
            }
        )
        directions = _mapping(checkpoint.get("directions"), field="checkpoint directions")
        for direction in config.directions:
            status = _mapping(
                direction_status.get(direction),
                field=f"checkpoint direction_status.{direction}",
            )
            if status.get("included") is not True:
                exclusions[direction][str(status.get("exclusion_reason"))] += 1
                continue
            included_by_direction[direction] += 1
            included_by_targeting[direction][pair.targeting] += 1
            direction_payload = _mapping(
                directions.get(direction),
                field=f"checkpoint {direction}",
            )
            windows = direction_payload.get("windows")
            if not isinstance(windows, list) or len(windows) != len(config.layers):
                raise ValueError("included checkpoint lost its complete window grid")
            for raw_window in windows:
                window = _mapping(raw_window, field="checkpoint window")
                label = str(window.get("window"))
                event = window.get("event")
                if label not in event_grids[direction] or not isinstance(event, bool):
                    raise ValueError("checkpoint window label or event is invalid")
                event_grids[direction][label].append(event)
                window_rows.append(
                    {
                        "schema_version": "fixed-window-answer-patching-window/v1",
                        "paper_sha256": PAPER_SHA256,
                        "model": config.model,
                        "benchmark": config.benchmark,
                        "targeting": pair.targeting,
                        "sample_id": pair.sample_id,
                        "source_record_sha256": pair.fingerprint,
                        "direction": direction,
                        "window": label,
                        "window_start": window.get("window_start"),
                        "window_stop": window.get("window_stop"),
                        "num_layers": n_layers,
                        "aligned_positions": pair.aligned_count,
                        "event": event,
                        "recipient_generation_identical": window.get(
                            "recipient_generation_identical"
                        ),
                        "baseline": checkpoint.get("baseline"),
                        "patched_answer": window.get("patched_answer"),
                    }
                )

    direction_summaries: dict[str, object] = {}
    for direction in config.directions:
        summaries: dict[str, object] = {}
        for window in config.layers:
            events = event_grids[direction][window.label]
            successes = sum(events)
            total = len(events)
            summaries[window.label] = {
                "window_start": window.start,
                "window_stop": window.stop,
                "successes": successes,
                "total": total,
                "rate": successes / total if total else None,
                "wilson_95_interval": _wilson_interval(successes, total),
            }
        direction_summaries[direction] = {
            "included_pairs": included_by_direction[direction],
            "included_by_targeting": {
                targeting: included_by_targeting[direction][targeting]
                for targeting in sorted(sources.by_targeting, key=_TARGETING_ORDER.__getitem__)
            },
            "exclusions": dict(sorted(exclusions[direction].items())),
            "windows": summaries,
        }

    comparison: dict[str, object] | None = None
    if (
        config.benchmark == "mmlu-pro"
        and "clean-to-edited" in event_grids
        and {"0:6", "6:12"}.issubset(event_grids["clean-to-edited"])
        and event_grids["clean-to-edited"]["0:6"]
    ):
        from typo_cot.experiments.fixed_window_answer_patching.metrics import (
            paired_binary_difference,
        )

        comparison = {
            "left_window": "0:6",
            "right_window": "6:12",
            **paired_binary_difference(
                event_grids["clean-to-edited"]["0:6"],
                event_grids["clean-to-edited"]["6:12"],
                resamples=_BOOTSTRAP_RESAMPLES,
                seed=42,
            ),
        }

    selected_by_targeting = Counter(pair.targeting for pair in selected)
    source_counts: dict[str, int] = {}
    prepared_eligible: dict[str, int] = {}
    upstream_exclusions: dict[str, dict[str, int]] = {}
    for targeting in sorted(sources.by_targeting, key=_TARGETING_ORDER.__getitem__):
        source = sources.by_targeting[targeting]
        source_counts[targeting] = len(source.pairs)
        prepared_eligible[targeting] = sum(_eligible(pair) for pair in source.pairs)
        counts: Counter[str] = Counter()
        for pair in source.pairs:
            if not pair.prepared_clean_correct:
                counts["prepared_clean_not_correct"] += 1
            elif not pair.prepared_edited_wrong:
                counts["prepared_edited_not_wrong"] += 1
            elif not pair.aligned_count:
                counts["no_aligned_words"] += 1
        upstream_exclusions[targeting] = dict(sorted(counts.items()))

    summary = {
        "schema_version": "fixed-window-answer-patching-setting-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "fixed-window-answer-patching",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "num_layers": n_layers,
            "layer_windows": [window.label for window in config.layers],
        },
        "protocol": _PROTOCOL,
        "population": {
            "source_pairs": source_counts,
            "prepared_eligible_by_targeting": prepared_eligible,
            "selected_anchors": len(selected),
            "selected_by_targeting": {
                targeting: selected_by_targeting[targeting]
                for targeting in sorted(sources.by_targeting, key=_TARGETING_ORDER.__getitem__)
            },
            "direction_denominators": {
                direction: included_by_direction[direction] for direction in config.directions
            },
            "upstream_exclusions": upstream_exclusions,
        },
        "directions": direction_summaries,
        "prespecified_mmlu_pro_window_comparison": comparison,
        "historical_reference": _historical_reference(config),
    }
    return window_rows, status_rows, summary, included_by_direction


def _runtime_class() -> Any:
    if HuggingFaceFixedWindowAnswerPatchingRuntime is not None:
        return HuggingFaceFixedWindowAnswerPatchingRuntime
    from typo_cot.experiments.fixed_window_answer_patching.runtime import (
        HuggingFaceFixedWindowAnswerPatchingRuntime as runtime_class,
    )

    return runtime_class


def _remove_public_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)


def run_fixed_window_answer_patching(
    config: FixedWindowAnswerPatchingConfig,
    *,
    runtime: FixedWindowAnswerPatchingRuntime | None = None,
) -> FixedWindowAnswerPatchingResult:
    """Regenerate baselines and patch each requested block window independently."""

    sources = _load_sources(config)
    selected = _select_anchors(sources, limit=config.limit)
    if not selected:
        raise ValueError(
            "pair sources contain no aligned prepared clean-correct/edited-wrong anchors"
        )
    initial_comparability = _comparability(config, sources, selected)

    output_dir = config.output_dir
    if output_dir.resolve() in {
        source.path.parent.resolve() for source in sources.by_targeting.values()
    }:
        raise ValueError("output directory must not overwrite a source pair directory")
    run_path = output_dir / "run.json"
    records_path = output_dir / "fixed_window_records.jsonl"
    status_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "setting_summary.json"
    checkpoints_dir = output_dir / "checkpoints"
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
        started_at = _validate_resume_manifest(previous, config=config, sources=sources)
        if previous.get("status") == "completed":
            return _completed_result(previous, output_dir=output_dir)
    else:
        started_at = _now()

    if runtime is None:
        runtime = _runtime_class()(config, revision=sources.model_revision)
    if not isinstance(runtime.num_layers, int) or runtime.num_layers <= 0:
        raise ValueError("runtime num_layers must be a positive integer")
    for window in config.layers:
        if window.stop > runtime.num_layers:
            raise ValueError(
                f"layer window {window.label} is outside the runtime decoder layer "
                f"count {runtime.num_layers}"
            )
    runtime_provenance = dict(runtime.provenance())
    if runtime_provenance.get("num_decoder_layers") != runtime.num_layers:
        raise ValueError("runtime provenance num_decoder_layers does not match runtime")
    if runtime_provenance.get("requested_revision") != sources.model_revision:
        raise ValueError("runtime requested revision does not match pair preparation")
    if runtime_provenance.get("model_revision") != sources.model_revision:
        raise ValueError("runtime model revision does not match pair preparation")
    if runtime_provenance.get("tokenizer_revision") != sources.model_revision:
        raise ValueError("runtime tokenizer revision does not match pair preparation")
    runtime_fp = _runtime_fingerprint(runtime_provenance)
    if previous is not None and previous.get("runtime") != runtime_provenance:
        raise ValueError("resume runtime provenance does not match the original run")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _remove_public_outputs(public_paths)
    previous_registry: Mapping[str, object] = (
        _mapping(previous.get("checkpoints"), field="resume checkpoint registry")
        if previous is not None
        else {}
    )
    checkpoint_registry: dict[str, dict[str, str]] = {}
    selected_checkpoint_names = {_checkpoint_path(checkpoints_dir, pair).name for pair in selected}
    for orphan in checkpoints_dir.glob("*.json"):
        if orphan.name not in selected_checkpoint_names:
            orphan.unlink()
    for pair in selected:
        path = _checkpoint_path(checkpoints_dir, pair)
        reusable = _checkpoint_registry_matches(
            path,
            source_pair=pair,
            metadata=previous_registry.get(path.name),
        )
        if reusable:
            try:
                _load_checkpoint(
                    path,
                    source_pair=pair,
                    runtime_fingerprint=runtime_fp,
                    n_layers=runtime.num_layers,
                    config=config,
                )
            except (KeyError, OSError, TypeError, ValueError):
                reusable = False
        if reusable:
            checkpoint_registry[path.name] = _checkpoint_registry_entry(
                path,
                source_pair=pair,
            )
        elif path.exists():
            path.unlink()

    counts: dict[str, object] = {
        "selected_anchors": len(selected),
        "checkpointed_pairs": len(checkpoint_registry),
        "failed_pairs": 0,
        "included_direction_pairs": 0,
        "window_records": 0,
    }
    _write_json_atomic(
        run_path,
        _base_manifest(
            config=config,
            sources=sources,
            status="running",
            started_at=started_at,
            runtime_provenance=runtime_provenance,
            comparability=initial_comparability,
            checkpoints=checkpoint_registry,
            counts=counts,
            failures=[],
        ),
    )

    failures: list[dict[str, str]] = []
    for pair in tqdm(
        selected,
        desc="fixed-window-answer-patching",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        checkpoint_path = _checkpoint_path(checkpoints_dir, pair)
        if checkpoint_path.name in checkpoint_registry:
            continue
        try:
            baseline = runtime.regenerate_baseline(pair.record)
            if not isinstance(baseline, BaselineScan):
                raise ValueError("runtime baseline result must be BaselineScan")
            if baseline.sample_id != pair.sample_id:
                raise ValueError("runtime BaselineScan sample_id does not match its source pair")
            gold_answer = _nonempty_string(
                pair.record.get("gold_answer"),
                field="source pair gold_answer",
            )
            _validate_generation(
                baseline.clean,
                field="baseline.clean",
                benchmark=config.benchmark,
                gold_answer=gold_answer,
            )
            _validate_generation(
                baseline.edited,
                field="baseline.edited",
                benchmark=config.benchmark,
                gold_answer=gold_answer,
            )
            included_directions = tuple(
                direction
                for direction in config.directions
                if _direction_status(baseline, direction)[0]
            )
            scan = (
                runtime.scan_pair(
                    pair.record,
                    baseline,
                    config.layers,
                    included_directions,
                )
                if included_directions
                else None
            )
            checkpoint = _process_pair(
                pair,
                baseline=baseline,
                scan=scan,
                included_directions=included_directions,
                config=config,
                n_layers=runtime.num_layers,
                runtime_fingerprint=runtime_fp,
            )
            _write_json_atomic(checkpoint_path, checkpoint)
            _load_checkpoint(
                checkpoint_path,
                source_pair=pair,
                runtime_fingerprint=runtime_fp,
                n_layers=runtime.num_layers,
                config=config,
            )
            checkpoint_registry[checkpoint_path.name] = _checkpoint_registry_entry(
                checkpoint_path,
                source_pair=pair,
            )
            counts["checkpointed_pairs"] = len(checkpoint_registry)
        except Exception as exc:  # noqa: BLE001 - preserve other expensive pairs
            if checkpoint_path.name not in checkpoint_registry:
                checkpoint_path.unlink(missing_ok=True)
            failures.append(
                {
                    "targeting": pair.targeting,
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
                sources=sources,
                status="running",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=initial_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=failures,
            ),
        )

    if failures:
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=initial_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=failures,
            ),
        )
        noun = "pair" if len(failures) == 1 else "pairs"
        raise FixedWindowAnswerPatchingRunError(
            f"{len(failures)} {noun} failed: {failures[0]['message']}; "
            "inspect run.json and rerun with --resume"
        )

    try:
        window_rows, status_rows, summary, included_by_direction = _compile_outputs(
            config=config,
            sources=sources,
            selected=selected,
            checkpoints_dir=checkpoints_dir,
            runtime_fingerprint=runtime_fp,
            n_layers=runtime.num_layers,
        )
    except Exception as exc:  # noqa: BLE001 - make the manifest authoritative
        compilation_failure = {
            "targeting": "all",
            "sample_id": "*",
            "error_type": "OutputCompilationError",
            "message": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=initial_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[compilation_failure],
            ),
        )
        raise FixedWindowAnswerPatchingRunError(
            f"output compilation failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    included_direction_pairs = sum(included_by_direction.values())
    counts["included_direction_pairs"] = included_direction_pairs
    counts["window_records"] = len(window_rows)
    final_comparability = _comparability(
        config,
        sources,
        selected,
        included_by_direction=included_by_direction,
    )
    if included_direction_pairs == 0:
        empty_cohort_failure = {
            "targeting": "all",
            "sample_id": "*",
            "error_type": "EmptyDirectionCohort",
            "message": "no selected anchor remained in any requested regenerated denominator",
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=final_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[empty_cohort_failure],
            ),
        )
        raise FixedWindowAnswerPatchingRunError(empty_cohort_failure["message"])

    _remove_public_outputs(public_paths)
    try:
        _write_jsonl_atomic(records_path, window_rows)
        _write_jsonl_atomic(status_path, status_rows)
        _write_json_atomic(summary_path, summary)
        output_metadata = {
            records_path.name: {
                "sha256": _sha256(records_path),
                "records": len(window_rows),
            },
            status_path.name: {
                "sha256": _sha256(status_path),
                "records": len(status_rows),
            },
            summary_path.name: {"sha256": _sha256(summary_path), "records": 1},
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="completed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=final_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[],
                outputs=output_metadata,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - remove partially published output set
        _remove_public_outputs(public_paths)
        finalization_failure = {
            "targeting": "all",
            "sample_id": "*",
            "error_type": "OutputFinalizationError",
            "message": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=final_comparability,
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[finalization_failure],
            ),
        )
        raise FixedWindowAnswerPatchingRunError(
            f"output finalization failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    return FixedWindowAnswerPatchingResult(
        fixed_window_records_path=records_path,
        pair_status_records_path=status_path,
        summary_path=summary_path,
        run_path=run_path,
        included_direction_pairs=included_direction_pairs,
        window_records=len(window_rows),
    )

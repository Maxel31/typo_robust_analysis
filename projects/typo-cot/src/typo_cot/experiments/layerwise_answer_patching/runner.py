"""Strict, resumable runner for one free-answer layerwise patching setting."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from tqdm.auto import tqdm

from typo_cot.evaluation.fallback import answers_equal
from typo_cot.experiments.catalog import PAPER_SHA256

DIRECTION_NAMES = ("clean-to-edited", "edited-to-clean")
_DIRECTION_ORDER = {name: index for index, name in enumerate(DIRECTION_NAMES)}
_TARGETING_NAMES = ("attribution-4", "random-4")
_TARGETING_ORDER = {name: index for index, name in enumerate(_TARGETING_NAMES)}
_GPU_ID = re.compile(r"0|[1-9][0-9]*")
_SELECTION_SEED = 42
_MAX_PAPER_ANCHORS = 300
_BOOTSTRAP_RESAMPLES = 2_000
_PROTOCOL = {
    "schema_version": "layerwise-answer-patching-protocol/v1",
    "source_anchor": "prepared-clean-correct-edited-wrong-with-aligned-word",
    "targeting_pool": "attribution-4-and-random-4-balanced-before-current-flip-recheck",
    "selection": {
        "seed": _SELECTION_SEED,
        "algorithm": "sample-id-sort-then-python-random-shuffle-per-targeting/v1",
        "maximum_total": _MAX_PAPER_ANCHORS,
        "maximum_per_targeting": _MAX_PAPER_ANCHORS // 2,
    },
    "fixed_cohort": "regenerated-clean-correct-and-regenerated-edited-wrong",
    "intervention_site": "complete-decoder-block-residual-output",
    "intervention_positions": "all-aligned-edited-word-final-tokens",
    "layer_grid": "all-decoder-layers-0-through-L-minus-1",
    "generation": {
        "dtype": "bfloat16",
        "decoding": "greedy",
        "max_new_tokens": 512,
        "padding_side": "left",
        "use_cache": True,
        "patch_application": "prompt-prefill-exactly-once",
    },
    "answer_extraction": "task-primary-then-empty-only-deterministic-fallback/v1",
    "restoration": "extracted-patched-edited-answer-equals-regenerated-clean-answer",
    "induction": "extracted-patched-clean-answer-differs-from-regenerated-clean-answer",
    "unextractable": "failure-in-both-directions-retained-in-denominator",
    "relative_depth": "layer-index-divided-by-L",
    "layer_profile": "binary-event-rate-on-one-fixed-cohort",
    "mcb": {
        "method": "paired-bootstrap-binary-risk-difference-hsu-mcb/v1",
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": 42,
        "one_sided_upper_percentile": 95,
    },
}

_HISTORICAL_REFERENCES = {
    ("gemma-3-4b-it", "gsm8k"): (172, 4, 0),
    ("Llama-3.2-3B-Instruct", "gsm8k"): (220, 2, 0),
    ("Mistral-7B-Instruct-v0.3", "gsm8k"): (179, 1, 0),
    ("Qwen2.5-3B-Instruct", "gsm8k"): (94, 8, 0),
    ("gemma-3-4b-it", "mmlu"): (209, 5, 0),
    ("Llama-3.2-3B-Instruct", "mmlu"): (226, 0, 0),
    ("Mistral-7B-Instruct-v0.3", "mmlu"): (205, 1, 1),
    ("Qwen2.5-3B-Instruct", "mmlu"): (209, 5, 0),
}
_PAPER_MODELS = frozenset(
    {
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-3B-Instruct",
    }
)


class LayerwiseAnswerPatchingRunError(RuntimeError):
    """Raised after runtime failures while retaining valid pair checkpoints."""


@dataclass(frozen=True, slots=True)
class LayerwiseAnswerPatchingConfig:
    """Frozen public arguments for one model/benchmark answer scan."""

    model: str
    benchmark: Literal["gsm8k", "mmlu"]
    attribution_pairs: Path
    random_pairs: Path
    directions: tuple[str, ...]
    max_pairs: int
    output_dir: Path
    gpu_id: str = "0"
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "attribution_pairs", Path(self.attribution_pairs))
        object.__setattr__(self, "random_pairs", Path(self.random_pairs))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.benchmark not in {"gsm8k", "mmlu"}:
            raise ValueError("layerwise-answer-patching supports only gsm8k and mmlu")
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
        if (
            isinstance(self.max_pairs, bool)
            or not isinstance(self.max_pairs, int)
            or self.max_pairs <= 0
            or self.max_pairs % 2
        ):
            raise ValueError("max_pairs must be a positive even integer")
        if self.max_pairs > _MAX_PAPER_ANCHORS:
            raise ValueError("max_pairs must be at most 300")
        if _GPU_ID.fullmatch(self.gpu_id) is None:
            raise ValueError("gpu_id must be a single non-negative integer")
        if self.attribution_pairs.resolve() == self.random_pairs.resolve():
            raise ValueError("attribution and random pair inputs must be different files")

    def public_arguments(self) -> dict[str, object]:
        """Return stable arguments, excluding the transport-only resume flag."""

        payload = asdict(self)
        for field in ("attribution_pairs", "random_pairs", "output_dir"):
            payload[field] = str(getattr(self, field).resolve())
        payload["directions"] = list(self.directions)
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
class DirectionAnswerScan:
    """One independently generated answer for every decoder layer."""

    patched_by_layer: tuple[AnswerGeneration, ...]


@dataclass(frozen=True, slots=True)
class PairAnswerScan:
    """All requested direction grids for one fixed-cohort pair."""

    sample_id: str
    directions: Mapping[str, DirectionAnswerScan]


class LayerwiseAnswerPatchingRuntime(Protocol):
    """GPU runtime seam used by production and CPU integration fixtures."""

    num_layers: int

    def provenance(self) -> Mapping[str, object]: ...

    def regenerate_baseline(self, pair: dict[str, object]) -> BaselineScan: ...

    def scan_pair(
        self,
        pair: dict[str, object],
        baseline: BaselineScan,
        directions: tuple[str, ...],
    ) -> PairAnswerScan: ...


@dataclass(frozen=True, slots=True)
class LayerwiseAnswerPatchingResult:
    """Published paths and record counts for a completed setting."""

    answer_layer_records_path: Path
    pair_status_records_path: Path
    summary_path: Path
    run_path: Path
    fixed_pairs: int
    layer_records: int


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
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
    config: LayerwiseAnswerPatchingConfig,
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
        raise ValueError(f"{context}: layerwise answer input requires seed 42")
    if record.get("num_edits_requested") != 4:
        raise ValueError(f"{context}: layerwise answer input requires four edits")
    _nonempty_string(record.get("gold_answer"), field=f"{context}.gold_answer")

    correctness: dict[str, bool] = {}
    prompt_counts: dict[str, int] = {}
    for side in ("clean", "edited"):
        payload = _mapping(record.get(side), field=f"{context}.{side}")
        _nonempty_string(payload.get("prompt"), field=f"{context}.{side}.prompt")
        prompt_counts[side] = _positive_int(
            payload.get("prompt_token_count"), field=f"{context}.{side}.prompt_token_count"
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
    return sample_id, correctness["clean"], not correctness["edited"], len(aligned_words)


def _load_source(
    path: Path,
    *,
    targeting: str,
    config: LayerwiseAnswerPatchingConfig,
) -> _Source:
    if not path.is_file():
        raise ValueError(f"pairs input is not a file: {path}")
    run_path = path.parent / "run.json"
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
        provenance.get("model_revision"), field="source run provenance.model_revision"
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
            "source run must contain the complete dataset: "
            "discovered, written, and dataset_sample_count must match"
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
            sample_id, clean_correct, edited_wrong, aligned_count = _validate_pair_record(
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


def _load_sources(config: LayerwiseAnswerPatchingConfig) -> _Sources:
    sources = {
        "attribution-4": _load_source(
            config.attribution_pairs,
            targeting="attribution-4",
            config=config,
        ),
        "random-4": _load_source(
            config.random_pairs,
            targeting="random-4",
            config=config,
        ),
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


def _select_anchors(sources: _Sources, *, max_pairs: int) -> tuple[_SourcePair, ...]:
    per_targeting = max_pairs // len(_TARGETING_NAMES)
    selected: list[_SourcePair] = []
    for targeting in _TARGETING_NAMES:
        eligible = [
            pair
            for pair in sources.by_targeting[targeting].pairs
            if pair.prepared_clean_correct and pair.prepared_edited_wrong and pair.aligned_count
        ]
        eligible.sort(key=lambda pair: pair.sample_id)
        random.Random(_SELECTION_SEED).shuffle(eligible)
        selected.extend(eligible[:per_targeting])
    selected.sort(key=lambda pair: (_TARGETING_ORDER[pair.targeting], pair.sample_id))
    return tuple(selected)


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


def _validate_generation(generation: AnswerGeneration, *, field: str) -> None:
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
    if not isinstance(generation.is_correct, bool):
        raise ValueError(f"{field}.is_correct must be boolean")
    if not generation.method or not generation.primary_method:
        raise ValueError(f"{field} extraction provenance must be non-empty")


def _process_pair(
    source_pair: _SourcePair,
    *,
    baseline: BaselineScan,
    scan: PairAnswerScan | None,
    config: LayerwiseAnswerPatchingConfig,
    n_layers: int,
    runtime_fingerprint: str,
) -> dict[str, object]:
    if baseline.sample_id != source_pair.sample_id:
        raise ValueError("runtime BaselineScan sample_id does not match its source pair")
    _validate_generation(baseline.clean, field="baseline.clean")
    _validate_generation(baseline.edited, field="baseline.edited")
    exclusion: str | None = None
    if not baseline.clean.is_correct:
        exclusion = "regenerated_clean_not_correct"
    elif baseline.edited.is_correct:
        exclusion = "regenerated_edited_not_wrong"

    directions: dict[str, object] = {}
    if exclusion is None:
        if scan is None:
            raise ValueError("current-flip pair is missing its layer scan")
        if scan.sample_id != source_pair.sample_id:
            raise ValueError("runtime PairAnswerScan sample_id does not match its source pair")
        if set(scan.directions) != set(config.directions):
            raise ValueError("runtime did not return exactly the requested directions")
        for direction in config.directions:
            layer_scan = scan.directions[direction]
            if not isinstance(layer_scan, DirectionAnswerScan):
                raise ValueError(f"runtime {direction} result must be DirectionAnswerScan")
            generations = tuple(layer_scan.patched_by_layer)
            if len(generations) != n_layers:
                raise ValueError(f"runtime {direction} result is not a complete layer grid")
            recipient = baseline.edited if direction == "clean-to-edited" else baseline.clean
            layer_rows: list[dict[str, object]] = []
            for layer_index, generation in enumerate(generations):
                _validate_generation(generation, field=f"{direction}.layer[{layer_index}]")
                extracted = generation.is_extracted
                equal_to_clean = answers_equal(
                    generation.value,
                    baseline.clean.value,
                    benchmark=config.benchmark,
                )
                event = bool(
                    extracted
                    and (equal_to_clean if direction == "clean-to-edited" else not equal_to_clean)
                )
                layer_rows.append(
                    {
                        "layer_index": layer_index,
                        "event": event,
                        "recipient_generation_identical": generation.token_ids
                        == recipient.token_ids,
                        "patched_answer": _generation_record(generation),
                    }
                )
            if generations[-1].token_ids != recipient.token_ids:
                raise ValueError(f"runtime {direction} final-layer patch is not a structural no-op")
            directions[direction] = {"layers": layer_rows}
    elif scan is not None:
        raise ValueError("runtime returned a layer scan for an excluded regenerated baseline")

    return {
        "schema_version": "layerwise-answer-patching-checkpoint/v1",
        "targeting": source_pair.targeting,
        "sample_id": source_pair.sample_id,
        "source_record_sha256": source_pair.fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "num_layers": n_layers,
        "aligned_positions": source_pair.aligned_count,
        "status": "included" if exclusion is None else "excluded",
        "exclusion_reason": exclusion,
        "baseline": {
            "clean": _generation_record(baseline.clean),
            "edited": _generation_record(baseline.edited),
        },
        "directions": directions,
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
                sources.by_targeting.items(), key=lambda item: _TARGETING_ORDER[item[0]]
            )
        },
    }


def _comparability(
    config: LayerwiseAnswerPatchingConfig,
    selected: Sequence[_SourcePair],
    *,
    fixed_by_targeting: Mapping[str, int] | None = None,
) -> dict[str, object]:
    selected_counts = Counter(pair.targeting for pair in selected)
    requirements: dict[str, bool | None] = {
        "paper_model": config.model in _PAPER_MODELS,
        "paper_benchmark": config.benchmark in {"gsm8k", "mmlu"},
        "both_directions": set(config.directions) == set(DIRECTION_NAMES),
        "max_pairs_300": config.max_pairs == _MAX_PAPER_ANCHORS,
        "selected_from_both_targeting_arms": all(
            selected_counts[targeting] > 0 for targeting in _TARGETING_NAMES
        ),
        "fixed_cohort_from_both_targeting_arms": (
            None
            if fixed_by_targeting is None
            else all(fixed_by_targeting.get(targeting, 0) > 0 for targeting in _TARGETING_NAMES)
        ),
    }
    limitations: list[str] = []
    if not requirements["paper_model"]:
        limitations.append("model-not-in-paper-eight-settings")
    if not requirements["both_directions"]:
        limitations.append("directions-not-both")
    if not requirements["max_pairs_300"]:
        limitations.append("max-pairs-below-300")
    for targeting in _TARGETING_NAMES:
        if selected_counts[targeting] == 0:
            limitations.append(f"no-selected-{targeting}-anchors")
        if fixed_by_targeting is not None and fixed_by_targeting.get(targeting, 0) == 0:
            limitations.append(f"no-fixed-{targeting}-anchors")

    if not requirements["max_pairs_300"]:
        status = "partial-smoke-run"
    elif not requirements["paper_model"]:
        status = "non-paper-setting"
    elif limitations:
        status = "partial-paper-protocol"
    elif fixed_by_targeting is None:
        status = "pending-current-flip-recheck"
    else:
        status = "fresh-paper-protocol-reproduction"
    return {
        "status": status,
        "requirements": requirements,
        "limitations": limitations,
        "exact_historical_figure2_ids": False,
        "historical_qwen_targeting_discrepancy": True,
        "historical_unextractable_induction_discrepancy": True,
        "note": (
            "The public default follows the final PDF: both targeting sources are "
            "required and still-unextractable patched answers fail both readouts. "
            "Historical Qwen Figure 2 records contain Attribution-4 only, and the "
            "historical plotting audit counted unextractable induction as a change."
        ),
    }


def _base_manifest(
    *,
    config: LayerwiseAnswerPatchingConfig,
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
        "schema_version": "layerwise-answer-patching-run/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "layerwise-answer-patching",
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
    config: LayerwiseAnswerPatchingConfig,
    sources: _Sources,
) -> str:
    if manifest.get("schema_version") != "layerwise-answer-patching-run/v1":
        raise ValueError("cannot resume an unknown layerwise answer run schema")
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
) -> LayerwiseAnswerPatchingResult:
    outputs = _mapping(manifest.get("outputs"), field="completed run outputs")
    names = (
        "answer_layer_records.jsonl",
        "pair_status_records.jsonl",
        "setting_summary.json",
    )
    for name in names:
        metadata = _mapping(outputs.get(name), field=f"completed output {name}")
        path = output_dir / name
        if not path.is_file() or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"completed output is missing or has changed: {path}")
    counts = _mapping(manifest.get("counts"), field="completed run counts")
    return LayerwiseAnswerPatchingResult(
        answer_layer_records_path=output_dir / names[0],
        pair_status_records_path=output_dir / names[1],
        summary_path=output_dir / names[2],
        run_path=output_dir / "run.json",
        fixed_pairs=int(counts.get("fixed_current_flips", 0)),
        layer_records=int(counts.get("layer_records", 0)),
    )


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
        benchmark=benchmark,
    ):
        raise ValueError(f"{field}.is_correct does not match the recorded answer")
    if not isinstance(generation.get("method"), str) or not generation.get("method"):
        raise ValueError(f"{field}.method must be non-empty")
    if not isinstance(generation.get("primary_method"), str) or not generation.get(
        "primary_method"
    ):
        raise ValueError(f"{field}.primary_method must be non-empty")
    return generation


def _load_checkpoint(
    path: Path,
    *,
    source_pair: _SourcePair,
    runtime_fingerprint: str,
    n_layers: int,
    config: LayerwiseAnswerPatchingConfig,
) -> dict[str, object]:
    checkpoint = _load_json(path)
    if (
        checkpoint.get("schema_version") != "layerwise-answer-patching-checkpoint/v1"
        or checkpoint.get("targeting") != source_pair.targeting
        or checkpoint.get("sample_id") != source_pair.sample_id
        or checkpoint.get("source_record_sha256") != source_pair.fingerprint
        or checkpoint.get("runtime_fingerprint") != runtime_fingerprint
        or checkpoint.get("num_layers") != n_layers
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
    status = checkpoint.get("status")
    exclusion = checkpoint.get("exclusion_reason")
    directions = _mapping(checkpoint.get("directions"), field="checkpoint directions")
    if status == "excluded":
        expected_exclusion = (
            "regenerated_clean_not_correct"
            if clean.get("is_correct") is False
            else "regenerated_edited_not_wrong"
            if edited.get("is_correct") is True
            else None
        )
        if exclusion != expected_exclusion or directions:
            raise ValueError("excluded checkpoint has inconsistent baseline status")
        return checkpoint
    if status != "included" or exclusion is not None:
        raise ValueError("checkpoint status must be a consistent included or excluded value")
    if clean.get("is_correct") is not True or edited.get("is_correct") is not False:
        raise ValueError("included checkpoint baseline is not a current answer flip")
    if set(directions) != set(config.directions):
        raise ValueError("included checkpoint directions do not match the requested grid")

    for direction in config.directions:
        direction_payload = _mapping(
            directions.get(direction),
            field=f"checkpoint {direction}",
        )
        layers = direction_payload.get("layers")
        if not isinstance(layers, list) or len(layers) != n_layers:
            raise ValueError(f"checkpoint {direction} is not a complete layer grid")
        recipient = edited if direction == "clean-to-edited" else clean
        for expected_layer, raw_layer in enumerate(layers):
            layer = _mapping(raw_layer, field=f"checkpoint {direction} layer")
            if layer.get("layer_index") != expected_layer:
                raise ValueError(f"checkpoint {direction} layer indices are not exact and ordered")
            event = layer.get("event")
            identical = layer.get("recipient_generation_identical")
            if not isinstance(event, bool) or not isinstance(identical, bool):
                raise ValueError(f"checkpoint {direction} layer flags must be boolean")
            patched = _validate_checkpoint_generation(
                layer.get("patched_answer"),
                field=f"checkpoint {direction} layer[{expected_layer}].patched_answer",
                benchmark=config.benchmark,
                gold_answer=gold_answer,
            )
            actual_identical = patched.get("token_ids") == recipient.get("token_ids")
            if identical is not actual_identical:
                raise ValueError(f"checkpoint {direction} recipient identity flag is inconsistent")
            equal_to_clean = answers_equal(
                str(patched.get("value")),
                str(clean.get("value")),
                benchmark=config.benchmark,
            )
            expected_event = bool(
                patched.get("is_extracted")
                and (equal_to_clean if direction == "clean-to-edited" else not equal_to_clean)
            )
            if event is not expected_event:
                raise ValueError(f"checkpoint {direction} event is inconsistent")
        if layers[-1].get("recipient_generation_identical") is not True:
            raise ValueError(f"checkpoint {direction} final layer is not a structural no-op")
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


def _compile_outputs(
    *,
    config: LayerwiseAnswerPatchingConfig,
    sources: _Sources,
    selected: Sequence[_SourcePair],
    checkpoints_dir: Path,
    runtime_fingerprint: str,
    n_layers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from typo_cot.experiments.layerwise_answer_patching.metrics import (
        summarize_binary_layers,
    )

    layer_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    event_grids: dict[str, list[tuple[bool, ...]]] = {
        direction: [] for direction in config.directions
    }
    fixed_by_targeting: Counter[str] = Counter()
    baseline_exclusions: Counter[str] = Counter()
    for pair in selected:
        checkpoint = _load_checkpoint(
            _checkpoint_path(checkpoints_dir, pair),
            source_pair=pair,
            runtime_fingerprint=runtime_fingerprint,
            n_layers=n_layers,
            config=config,
        )
        status = str(checkpoint.get("status"))
        exclusion = checkpoint.get("exclusion_reason")
        status_rows.append(
            {
                "schema_version": "layerwise-answer-patching-pair-status/v1",
                "paper_sha256": PAPER_SHA256,
                "model": config.model,
                "benchmark": config.benchmark,
                "targeting": pair.targeting,
                "sample_id": pair.sample_id,
                "source_record_sha256": pair.fingerprint,
                "status": status,
                "exclusion_reason": exclusion,
                "aligned_positions": pair.aligned_count,
                "baseline": checkpoint.get("baseline"),
            }
        )
        if status != "included":
            baseline_exclusions[str(exclusion)] += 1
            continue
        fixed_by_targeting[pair.targeting] += 1
        directions = _mapping(checkpoint.get("directions"), field="checkpoint directions")
        for direction in config.directions:
            direction_payload = _mapping(directions.get(direction), field=f"checkpoint {direction}")
            layers = direction_payload.get("layers")
            if not isinstance(layers, list) or len(layers) != n_layers:
                raise ValueError("included checkpoint lost its complete layer grid")
            pair_events: list[bool] = []
            for raw_layer in layers:
                layer = _mapping(raw_layer, field="checkpoint layer")
                layer_index = int(layer["layer_index"])
                event = layer.get("event")
                if not isinstance(event, bool):
                    raise ValueError("checkpoint layer event must be boolean")
                pair_events.append(event)
                layer_rows.append(
                    {
                        "schema_version": "layerwise-answer-patching-layer/v1",
                        "paper_sha256": PAPER_SHA256,
                        "model": config.model,
                        "benchmark": config.benchmark,
                        "targeting": pair.targeting,
                        "sample_id": pair.sample_id,
                        "source_record_sha256": pair.fingerprint,
                        "direction": direction,
                        "layer_index": layer_index,
                        "num_layers": n_layers,
                        "relative_depth": layer_index / n_layers,
                        "layer_center_relative_depth": (layer_index + 0.5) / n_layers,
                        "aligned_positions": pair.aligned_count,
                        "event": event,
                        "recipient_generation_identical": layer.get(
                            "recipient_generation_identical"
                        ),
                        "baseline": checkpoint.get("baseline"),
                        "patched_answer": layer.get("patched_answer"),
                    }
                )
            event_grids[direction].append(tuple(pair_events))

    direction_summaries: dict[str, object] = {}
    for direction in config.directions:
        if event_grids[direction]:
            direction_summaries[direction] = summarize_binary_layers(
                event_grids[direction],
                bootstrap_resamples=_BOOTSTRAP_RESAMPLES,
                seed=42,
            )
        else:
            direction_summaries[direction] = {
                "included_pairs": 0,
                "num_layers": n_layers,
                "layer_profile": [],
                "peak": None,
                "mcb": None,
            }

    prepared_eligible: dict[str, int] = {}
    selected_by_targeting = Counter(pair.targeting for pair in selected)
    upstream: dict[str, dict[str, int]] = {}
    for targeting in _TARGETING_NAMES:
        counts: Counter[str] = Counter()
        eligible_count = 0
        for pair in sources.by_targeting[targeting].pairs:
            if not pair.prepared_clean_correct:
                counts["prepared_clean_not_correct"] += 1
            elif not pair.prepared_edited_wrong:
                counts["prepared_edited_not_wrong"] += 1
            elif not pair.aligned_count:
                counts["no_aligned_words"] += 1
            else:
                eligible_count += 1
        prepared_eligible[targeting] = eligible_count
        upstream[targeting] = dict(sorted(counts.items()))

    basename = config.model.rsplit("/", 1)[-1]
    reference = _HISTORICAL_REFERENCES.get((basename, config.benchmark))
    historical_reference = (
        {
            "fixed_current_flips": reference[0],
            "restoration_peak_layer": reference[1],
            "induction_peak_layer": reference[2],
            "comparison": "descriptive-only-fresh-public-cohort",
        }
        if reference is not None
        else None
    )
    summary = {
        "schema_version": "layerwise-answer-patching-setting-summary/v1",
        "paper_sha256": PAPER_SHA256,
        "operation": "layerwise-answer-patching",
        "setting": {
            "model": config.model,
            "benchmark": config.benchmark,
            "num_layers": n_layers,
        },
        "protocol": _PROTOCOL,
        "population": {
            "source_pairs": {
                targeting: len(sources.by_targeting[targeting].pairs)
                for targeting in _TARGETING_NAMES
            },
            "prepared_eligible_by_targeting": prepared_eligible,
            "selected_anchors": len(selected),
            "selected_by_targeting": {
                targeting: selected_by_targeting[targeting] for targeting in _TARGETING_NAMES
            },
            "fixed_current_flips": sum(fixed_by_targeting.values()),
            "fixed_by_targeting": {
                targeting: fixed_by_targeting[targeting] for targeting in _TARGETING_NAMES
            },
            "upstream_exclusions": upstream,
            "regenerated_baseline_exclusions": dict(sorted(baseline_exclusions.items())),
        },
        "directions": direction_summaries,
        "historical_figure2_reference": historical_reference,
        "cross_setting_boundary": {
            "scope": "one-setting-only",
            "paper_claim": "count peak layers across eight setting summaries; do not pool rates",
        },
    }
    return layer_rows, status_rows, summary


def run_layerwise_answer_patching(
    config: LayerwiseAnswerPatchingConfig,
    *,
    runtime: LayerwiseAnswerPatchingRuntime | None = None,
) -> LayerwiseAnswerPatchingResult:
    """Regenerate baselines, freeze current flips, and scan all requested layers."""

    sources = _load_sources(config)
    selected = _select_anchors(sources, max_pairs=config.max_pairs)
    if not selected:
        raise ValueError(
            "pair sources contain no aligned prepared clean-correct/edited-wrong anchors"
        )
    initial_comparability = _comparability(config, selected)

    output_dir = config.output_dir
    if output_dir.resolve() in {
        config.attribution_pairs.parent.resolve(),
        config.random_pairs.parent.resolve(),
    }:
        raise ValueError("output directory must not overwrite either source pair directory")
    run_path = output_dir / "run.json"
    layer_path = output_dir / "answer_layer_records.jsonl"
    status_path = output_dir / "pair_status_records.jsonl"
    summary_path = output_dir / "setting_summary.json"
    work_dir = output_dir / ".layerwise-answer-patching-work"
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
        started_at = _validate_resume_manifest(previous, config=config, sources=sources)
        if previous.get("status") == "completed":
            return _completed_result(previous, output_dir=output_dir)
    else:
        started_at = _now()

    if runtime is None:
        from typo_cot.experiments.layerwise_answer_patching.runtime import (
            HuggingFaceLayerwiseAnswerPatchingRuntime,
        )

        runtime = HuggingFaceLayerwiseAnswerPatchingRuntime(
            config,
            revision=sources.model_revision,
        )
    if not isinstance(runtime.num_layers, int) or runtime.num_layers <= 0:
        raise ValueError("runtime num_layers must be a positive integer")
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
        "fixed_current_flips": 0,
        "layer_records": 0,
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
        desc="layerwise-answer-patching",
        unit="pair",
        total=len(selected),
        disable=None,
    ):
        checkpoint_path = _checkpoint_path(checkpoints_dir, pair)
        if checkpoint_path.name in checkpoint_registry:
            continue
        try:
            baseline = runtime.regenerate_baseline(pair.record)
            scan = (
                runtime.scan_pair(pair.record, baseline, config.directions)
                if baseline.clean.is_correct and not baseline.edited.is_correct
                else None
            )
            checkpoint = _process_pair(
                pair,
                baseline=baseline,
                scan=scan,
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
        raise LayerwiseAnswerPatchingRunError(
            f"{len(failures)} {noun} failed; inspect run.json and rerun with --resume"
        )

    try:
        layer_rows, status_rows, summary = _compile_outputs(
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
        raise LayerwiseAnswerPatchingRunError(
            f"output compilation failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    fixed_pairs = sum(row["status"] == "included" for row in status_rows)
    counts["fixed_current_flips"] = fixed_pairs
    counts["layer_records"] = len(layer_rows)
    final_fixed_by_targeting = Counter(
        str(row["targeting"]) for row in status_rows if row["status"] == "included"
    )
    if fixed_pairs == 0:
        empty_cohort_failure = {
            "targeting": "all",
            "sample_id": "*",
            "error_type": "EmptyFixedCohort",
            "message": (
                "no selected anchor remained a regenerated clean-correct, edited-wrong pair"
            ),
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="failed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=_comparability(
                    config,
                    selected,
                    fixed_by_targeting=final_fixed_by_targeting,
                ),
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[empty_cohort_failure],
            ),
        )
        raise LayerwiseAnswerPatchingRunError(empty_cohort_failure["message"])

    public_paths = (layer_path, status_path, summary_path)
    for path in public_paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    try:
        _write_jsonl_atomic(layer_path, layer_rows)
        _write_jsonl_atomic(status_path, status_rows)
        _write_json_atomic(summary_path, summary)
        output_metadata = {
            layer_path.name: {"sha256": _sha256(layer_path), "records": len(layer_rows)},
            status_path.name: {
                "sha256": _sha256(status_path),
                "records": len(status_rows),
            },
            summary_path.name: {"sha256": _sha256(summary_path)},
        }
        _write_json_atomic(
            run_path,
            _base_manifest(
                config=config,
                sources=sources,
                status="completed",
                started_at=started_at,
                runtime_provenance=runtime_provenance,
                comparability=_comparability(
                    config,
                    selected,
                    fixed_by_targeting=final_fixed_by_targeting,
                ),
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[],
                outputs=output_metadata,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - clean a partially published output set
        for path in public_paths:
            path.unlink(missing_ok=True)
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
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
                comparability=_comparability(
                    config,
                    selected,
                    fixed_by_targeting=final_fixed_by_targeting,
                ),
                checkpoints=checkpoint_registry,
                counts=counts,
                failures=[finalization_failure],
            ),
        )
        raise LayerwiseAnswerPatchingRunError(
            f"output finalization failed; inspect run.json and rerun with --resume: {exc}"
        ) from exc

    try:
        shutil.rmtree(work_dir)
    except OSError:
        # Published hashes and the completed manifest remain authoritative; a
        # stale private work directory is harmless and can be removed manually.
        pass
    return LayerwiseAnswerPatchingResult(
        answer_layer_records_path=layer_path,
        pair_status_records_path=status_path,
        summary_path=summary_path,
        run_path=run_path,
        fixed_pairs=fixed_pairs,
        layer_records=len(layer_rows),
    )

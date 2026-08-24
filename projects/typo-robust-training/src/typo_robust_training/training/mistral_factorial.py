"""Frozen packed-data stream for the Mistral state-free factorial.

The five factorial arms must differ only in adapter placement and output target
scope.  This module therefore materializes clean/noisy pairs *before* training,
including deterministic rejection/replacement decisions.  Every arm for one
seed consumes the same byte-identical ``pairs.jsonl`` artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.perturb import TypoGenerator
from typo_robust_training.data.records import TypoEdit
from typo_robust_training.training.encoding import encode_training_pair
from typo_robust_training.training.filesystem import (
    publish_directory_noreplace,
    reject_path_symlink_components,
)
from typo_robust_training.training.kojima_faithful import (
    FINEWEB_DATASET,
    FINEWEB_DATA_FILE,
    FINEWEB_REVISION,
    MATCHED_REPLICATION_SEEDS,
    MAX_SEQUENCE_LENGTH,
    MISTRAL_MODEL,
    MISTRAL_REVISION,
    PACKED_EXAMPLES,
    PACKING_POLICY,
    TARGET_USABLE_EXAMPLES,
    KojimaFaithfulDataBundle,
    load_kojima_faithful_data_bundle,
)
from typo_robust_training.training.pairs import TrainingPair, materialize_training_pair


FACTORIAL_METHOD_IDENTITY = "mistral-state-free-probe-factorial/v1"
FACTORIAL_PAIRING_POLICY = "exact-alternating-clean-noisy-precomputed/v1"
FACTORIAL_NOISE_POLICY = "record-local-three-operation-fixed-mixture/v1"
FACTORIAL_RUNTIME_POLICY = "hash-attested-prevalidated-8000-pair-stream/v1"
FACTORIAL_INITIALIZATION_POLICY = "sha256-layer-keyed-kaiming-a-zero-b/v1"
FACTORIAL_PARENT_PREFIX_POLICY = "faithful-parent-prefix-no-added-special-tokens/v1"
FACTORIAL_OPERATIONS = MappingProxyType(
    {
        "deletion": 1.0 / 3.0,
        "duplication": 1.0 / 3.0,
        "keyboard-neighbor-substitution": 1.0 / 3.0,
    }
)
FACTORIAL_EDIT_COUNTS = MappingProxyType({"1": 0.50, "2": 0.30, "3-4": 0.20})

_PAIR_SCHEMA = "mistral-state-free-factorial-pair/v1"
_SKIP_SCHEMA = "mistral-state-free-factorial-skip/v1"
_MANIFEST_SCHEMA = "mistral-state-free-factorial-data-manifest/v1"
_RUN_SCHEMA = "prepare-mistral-state-free-factorial-data-run/v1"
_SOURCE_DIR = "packed_source"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _object(path: Path) -> tuple[Mapping[str, object], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"factorial data artifact is not one regular file: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"factorial data artifact is not UTF-8: {path}") from exc
    value = strict_loads(text, context=str(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"factorial data artifact must contain an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _reject_path_symlink_components(path: Path, *, artifact: str) -> None:
    reject_path_symlink_components(path, artifact=f"factorial {artifact}")


def _reject_tree_links(root: Path, *, artifact: str) -> None:
    """Reject symlinks, hard-linked files, and special nodes in an artifact tree."""

    if root.is_symlink():
        raise ValueError(f"factorial {artifact} root cannot be a symlink")
    if not root.is_dir():
        raise ValueError(f"factorial {artifact} root must be a directory")
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"factorial {artifact} tree contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ValueError(f"factorial {artifact} tree contains a hard link: {path}")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"factorial {artifact} tree contains a special node: {path}")


def _publish_directory_noreplace(staged: Path, output: Path) -> None:
    """Publish with the factorial-specific race error kept stable for callers."""

    try:
        publish_directory_noreplace(staged, output)
    except FileExistsError as exc:
        raise FileExistsError(f"factorial output appeared during preparation: {output}") from exc


def _token_ids_sha256(values: tuple[int, ...]) -> str:
    return hashlib.sha256(
        b"".join(int(token).to_bytes(4, "big", signed=False) for token in values)
    ).hexdigest()


def _encode_factorial_pair(
    pair: TrainingPair,
    *,
    source: object,
    tokenizer: Any,
) -> object:
    """Encode against the faithful prefix without silently prepending another BOS."""

    encoding = encode_training_pair(
        pair,
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
        require_all_edits_visible=True,
        require_downstream_targets=not pair.is_noop,
        add_special_tokens=False,
    )
    metadata = getattr(source, "metadata", None)
    expected_prefix = (
        metadata.get("clean_prefix_token_ids_sha256") if isinstance(metadata, Mapping) else None
    )
    if _SHA256.fullmatch(str(expected_prefix)) is None or (
        _token_ids_sha256(encoding.clean_input_ids) != expected_prefix
    ):
        raise ValueError("factorial clean context differs from its faithful parent prefix")
    if encoding.student_tokens != MAX_SEQUENCE_LENGTH:
        raise ValueError("factorial packed pair does not fill the 8,192-token context")
    return encoding


def _load_attested_factorial_tokenizer() -> tuple[Any, Mapping[str, object]]:
    """Load the exact tokenizer under the frozen snapshot attestation contract."""

    from typo_cot.models.tokenizer_attestation import (
        load_attested_tokenizer,
        preflight_frozen_tokenizer_attestation,
        require_frozen_tokenizer_attestation,
    )

    frozen = preflight_frozen_tokenizer_attestation(
        expected_model=MISTRAL_MODEL,
        expected_revision=MISTRAL_REVISION,
    )
    tokenizer, loaded = load_attested_tokenizer(MISTRAL_MODEL, MISTRAL_REVISION)
    required = require_frozen_tokenizer_attestation(
        SimpleNamespace(tokenizer_snapshot_attestation=loaded),
        expected_model=MISTRAL_MODEL,
        expected_revision=MISTRAL_REVISION,
    )
    if (
        loaded.provenance_dict() != frozen.provenance_dict()
        or required.provenance_dict() != frozen.provenance_dict()
    ):
        raise ValueError("factorial tokenizer attestation changed after preflight")
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("factorial preparation requires the exact fast tokenizer")
    return tokenizer, MappingProxyType(loaded.provenance_dict())


def _validate_tokenizer_provenance(value: object) -> Mapping[str, object]:
    from typo_cot.models.tokenizer_attestation import (
        validate_tokenizer_attestation_provenance,
    )

    return MappingProxyType(
        validate_tokenizer_attestation_provenance(
            value,
            expected_model=MISTRAL_MODEL,
            expected_revision=MISTRAL_REVISION,
        ).provenance_dict()
    )


def _edit_to_dict(edit: TypoEdit) -> dict[str, object]:
    return {
        "operation": edit.operation,
        "clean_word": edit.clean_word,
        "typo_word": edit.typo_word,
        "clean_char_span": list(edit.clean_char_span),
        "typo_char_span": list(edit.typo_char_span),
    }


def _edit_from_dict(value: object) -> TypoEdit:
    fields = {
        "operation",
        "clean_word",
        "typo_word",
        "clean_char_span",
        "typo_char_span",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("factorial typo edit fields differ")
    clean_span = value["clean_char_span"]
    typo_span = value["typo_char_span"]
    if not isinstance(clean_span, list) or not isinstance(typo_span, list):
        raise ValueError("factorial typo edit spans must be lists")
    return TypoEdit(
        operation=str(value["operation"]),
        clean_word=str(value["clean_word"]),
        typo_word=str(value["typo_word"]),
        clean_char_span=tuple(clean_span),  # type: ignore[arg-type]
        typo_char_span=tuple(typo_span),  # type: ignore[arg-type]
    )


def _pair_to_row(
    pair: TrainingPair,
    *,
    seed: int,
    usable_index: int,
    attempt_index: int,
) -> dict[str, object]:
    row = {
        "schema_version": _PAIR_SCHEMA,
        "seed": seed,
        "usable_index": usable_index,
        "attempt_index": attempt_index,
        "source_record_id": pair.record_id,
        "clean_text": pair.clean_text,
        "typo_text": pair.typo_text,
        "clean_sha256": hashlib.sha256(pair.clean_text.encode()).hexdigest(),
        "typo_sha256": hashlib.sha256(pair.typo_text.encode()).hexdigest(),
        "is_noop": pair.is_noop,
        "edits": [_edit_to_dict(edit) for edit in pair.edits],
    }
    return {**row, "pair_sha256": _canonical_sha(row)}


def _pair_from_row(value: object, *, expected_seed: int, expected_index: int) -> TrainingPair:
    fields = {
        "schema_version",
        "seed",
        "usable_index",
        "attempt_index",
        "source_record_id",
        "clean_text",
        "typo_text",
        "clean_sha256",
        "typo_sha256",
        "is_noop",
        "edits",
        "pair_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("factorial pair fields differ")
    unsigned = {key: value[key] for key in fields - {"pair_sha256"}}
    clean_text = value["clean_text"]
    typo_text = value["typo_text"]
    is_noop = value["is_noop"]
    if (
        value["schema_version"] != _PAIR_SCHEMA
        or value["seed"] != expected_seed
        or value["usable_index"] != expected_index
        or isinstance(value["attempt_index"], bool)
        or not isinstance(value["attempt_index"], int)
        or value["attempt_index"] < expected_index
        or not isinstance(value["source_record_id"], str)
        or _SHA256.fullmatch(value["source_record_id"]) is None
        or not isinstance(clean_text, str)
        or not clean_text
        or not isinstance(typo_text, str)
        or not typo_text
        or value["clean_sha256"] != hashlib.sha256(clean_text.encode()).hexdigest()
        or value["typo_sha256"] != hashlib.sha256(typo_text.encode()).hexdigest()
        or type(is_noop) is not bool
        or value["pair_sha256"] != _canonical_sha(unsigned)
        or not isinstance(value["edits"], list)
    ):
        raise ValueError("factorial pair identity differs")
    edits = tuple(_edit_from_dict(edit) for edit in value["edits"])
    if is_noop != (expected_index % 2 == 0) or is_noop != (not edits):
        raise ValueError("factorial pair clean/noisy alternation differs")
    if is_noop != (clean_text == typo_text):
        raise ValueError("factorial pair no-op text identity differs")
    for edit in edits:
        if (
            clean_text[slice(*edit.clean_char_span)] != edit.clean_word
            or typo_text[slice(*edit.typo_char_span)] != edit.typo_word
            or edit.operation not in FACTORIAL_OPERATIONS
        ):
            raise ValueError("factorial pair edit does not bind to its text")
    return TrainingPair(
        record_id=str(value["source_record_id"]),
        clean_text=clean_text,
        typo_text=typo_text,
        task=None,
        answer=None,
        metadata=MappingProxyType(
            {
                "mistral_state_free_factorial": True,
                "usable_index": expected_index,
                "attempt_index": int(value["attempt_index"]),
                "pair_sha256": str(value["pair_sha256"]),
            }
        ),
        edits=edits,
        is_noop=bool(is_noop),
        epoch=0,
        variant=0,
    )


@dataclass(frozen=True, slots=True)
class PrepareMistralFactorialDataConfig:
    seed: int
    packed_source_dir: Path
    output_dir: Path
    target_usable_examples: int = TARGET_USABLE_EXAMPLES

    def __post_init__(self) -> None:
        if self.seed not in MATCHED_REPLICATION_SEEDS:
            raise ValueError("factorial seed must be one of the matched seeds 42/43/44")
        if (
            isinstance(self.target_usable_examples, bool)
            or not isinstance(self.target_usable_examples, int)
            or self.target_usable_examples <= 0
            or self.target_usable_examples > PACKED_EXAMPLES
            or self.target_usable_examples % 2
        ):
            raise ValueError("factorial usable count must be positive, even, and within attempts")


@dataclass(frozen=True, slots=True)
class PrepareMistralFactorialDataResult:
    manifest_path: Path
    pairs_path: Path
    skips_path: Path
    run_path: Path
    usable_examples: int
    student_tokens: int


@dataclass(frozen=True, slots=True)
class MistralFactorialDataBundle:
    root: Path
    pairs: tuple[TrainingPair, ...]
    data_identity_sha256: str
    training_data_sha256: str
    artifact_sha256: Mapping[str, str]
    seed: int
    source_revision: str
    source_order_sha256: str
    packing_policy: str
    pairing_policy: str
    noise_policy: str
    target_usable_examples: int
    target_student_tokens: int
    packed_attempts: int
    consumed_attempts: int
    tokenizer_snapshot_attestation: Mapping[str, object]


def _prepare_mistral_factorial_data_in_directory(
    config: PrepareMistralFactorialDataConfig,
    *,
    output: Path,
    parent_root: Path,
    parent: KojimaFaithfulDataBundle,
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, object],
) -> PrepareMistralFactorialDataResult:
    """Write and round-trip one staged factorial artifact directory."""

    source_copy = output / _SOURCE_DIR
    source_copy.mkdir()
    for name in ("manifest.json", "packed_sources.jsonl", "run.json"):
        source = parent_root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"factorial packed source artifact differs: {name}")
        shutil.copyfile(source, source_copy / name)

    generator = TypoGenerator(
        seed=config.seed,
        operation_weights=FACTORIAL_OPERATIONS,
        edit_count_weights=FACTORIAL_EDIT_COUNTS,
        explicit_clean_pair_probability=0.0,
        minimum_word_letters=2,
    )
    pairs_path = output / "pairs.jsonl"
    skips_path = output / "skips.jsonl"
    pairs: list[TrainingPair] = []
    skip_rows: list[dict[str, object]] = []
    operation_counts: Counter[str] = Counter()
    edit_count_counts: Counter[str] = Counter()
    last_attempt_index = -1
    with (
        pairs_path.open("w", encoding="utf-8", newline="\n") as pair_handle,
        skips_path.open("w", encoding="utf-8", newline="\n") as skip_handle,
    ):
        for attempt_index, source in enumerate(parent.sources):
            if len(pairs) >= config.target_usable_examples:
                break
            force_noop = len(pairs) % 2 == 0
            last_attempt_index = attempt_index
            pair = materialize_training_pair(
                source,
                generator=generator,
                epoch=0,
                variant=0,
                force_noop=force_noop,
            )
            try:
                _encode_factorial_pair(
                    pair,
                    source=source,
                    tokenizer=tokenizer,
                )
            except (ValueError, RuntimeError) as exc:
                skip = {
                    "schema_version": _SKIP_SCHEMA,
                    "seed": config.seed,
                    "attempt_index": attempt_index,
                    "source_record_id": source.record_id,
                    "intended_is_noop": force_noop,
                    "reason_type": type(exc).__name__,
                    "reason_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                }
                skip_handle.write(
                    json.dumps(skip, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                )
                skip_rows.append(skip)
                continue
            row = _pair_to_row(
                pair,
                seed=config.seed,
                usable_index=len(pairs),
                attempt_index=attempt_index,
            )
            pair_handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            pairs.append(pair)
            if not pair.is_noop:
                operation_counts.update(edit.operation for edit in pair.edits)
                count = len(pair.edits)
                edit_count_counts["3-4" if count >= 3 else str(count)] += 1
    if len(pairs) != config.target_usable_examples:
        raise ValueError("factorial packed attempts were exhausted before the usable budget")

    pairs_sha = _sha256_file(pairs_path)
    skips_sha = _sha256_file(skips_path)
    copied_hashes = {
        f"{_SOURCE_DIR}/{name}": _sha256_file(source_copy / name)
        for name in ("manifest.json", "packed_sources.jsonl", "run.json")
    }
    student_tokens = len(pairs) * MAX_SEQUENCE_LENGTH
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "method_identity": FACTORIAL_METHOD_IDENTITY,
        "seed": config.seed,
        "model": {"id": MISTRAL_MODEL, "revision": MISTRAL_REVISION},
        "dataset": {
            "id": FINEWEB_DATASET,
            "revision": FINEWEB_REVISION,
            "data_file": FINEWEB_DATA_FILE,
            "data_file_sha256": parent.source_file_sha256,
        },
        "tokenizer": {
            "id": MISTRAL_MODEL,
            "revision": MISTRAL_REVISION,
            "snapshot_attestation": dict(tokenizer_provenance),
        },
        "packing": {
            "policy": PACKING_POLICY,
            "source_order_sha256": parent.source_order_sha256,
            "packed_attempts": len(parent.sources),
            "consumed_attempts": max(
                last_attempt_index + 1,
                0,
            ),
            "usable_examples": len(pairs),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "target_student_tokens": student_tokens,
        },
        "pairing": {
            "policy": FACTORIAL_PAIRING_POLICY,
            "clean_probability": 0.5,
            "first_row": "clean",
            "realized_clean_examples": sum(pair.is_noop for pair in pairs),
            "realized_noisy_examples": sum(not pair.is_noop for pair in pairs),
        },
        "noise": {
            "policy": FACTORIAL_NOISE_POLICY,
            "operations": dict(FACTORIAL_OPERATIONS),
            "edit_count_distribution": dict(FACTORIAL_EDIT_COUNTS),
            "minimum_word_letters": 2,
            "realized_operation_counts": {
                operation: operation_counts[operation] for operation in sorted(FACTORIAL_OPERATIONS)
            },
            "realized_edit_count_counts": {
                bucket: edit_count_counts[bucket] for bucket in ("1", "2", "3-4")
            },
        },
        "runtime": {
            "policy": FACTORIAL_RUNTIME_POLICY,
            "parent_prefix_policy": FACTORIAL_PARENT_PREFIX_POLICY,
            "prevalidated_downstream_offsets": [2, 16],
            "skipped_attempts": len(skip_rows),
        },
        "parent": {
            "data_identity_sha256": parent.data_identity_sha256,
            "artifacts": copied_hashes,
        },
        "artifacts": {
            "pairs.jsonl": {"sha256": pairs_sha},
            "skips.jsonl": {"sha256": skips_sha},
            **{name: {"sha256": digest} for name, digest in copied_hashes.items()},
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)
    run = {
        "schema_version": _RUN_SCHEMA,
        "status": "completed",
        "seed": config.seed,
        "manifest_sha256": manifest_sha,
        "outputs": {
            "manifest.json": {"sha256": manifest_sha},
            "pairs.jsonl": {"sha256": pairs_sha},
            "skips.jsonl": {"sha256": skips_sha},
            **{name: {"sha256": digest} for name, digest in copied_hashes.items()},
        },
    }
    run_path = output / "run.json"
    _write_json(run_path, run)
    # Re-open the complete artifact before publication is considered successful.
    loaded = load_mistral_factorial_data_bundle(output, seed=config.seed)
    if len(loaded.pairs) != len(pairs) or loaded.target_student_tokens != student_tokens:
        raise RuntimeError("factorial data publication did not round-trip")
    return PrepareMistralFactorialDataResult(
        manifest_path=manifest_path,
        pairs_path=pairs_path,
        skips_path=skips_path,
        run_path=run_path,
        usable_examples=len(pairs),
        student_tokens=student_tokens,
    )


def prepare_mistral_factorial_data(
    config: PrepareMistralFactorialDataConfig,
) -> PrepareMistralFactorialDataResult:
    """Atomically freeze one attested pair stream shared by all five arms."""

    raw_parent = Path(config.packed_source_dir)
    raw_output = Path(config.output_dir)
    _reject_path_symlink_components(raw_parent, artifact="packed source")
    _reject_path_symlink_components(raw_output, artifact="output")
    parent_root = raw_parent.resolve()
    output = raw_output.resolve()
    if output.exists():
        raise FileExistsError(f"factorial output already exists: {output}")
    if parent_root == output or parent_root in output.parents or output in parent_root.parents:
        raise ValueError("factorial output cannot contain its source artifact")
    _reject_tree_links(parent_root, artifact="packed source")
    parent = load_kojima_faithful_data_bundle(parent_root, seed=config.seed)
    if len(parent.sources) < config.target_usable_examples:
        raise ValueError("factorial packed source has too few attempts for its usable target")
    if (
        config.target_usable_examples == TARGET_USABLE_EXAMPLES
        and len(parent.sources) != PACKED_EXAMPLES
    ):
        raise ValueError("production factorial requires exactly 8,800 packed attempts")
    tokenizer, tokenizer_provenance = _load_attested_factorial_tokenizer()

    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_path_symlink_components(raw_output, artifact="output")
    staged = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    published = False
    try:
        staged_result = _prepare_mistral_factorial_data_in_directory(
            config,
            output=staged,
            parent_root=parent_root,
            parent=parent,
            tokenizer=tokenizer,
            tokenizer_provenance=tokenizer_provenance,
        )
        _publish_directory_noreplace(staged, output)
        published = True
        return PrepareMistralFactorialDataResult(
            manifest_path=output / staged_result.manifest_path.name,
            pairs_path=output / staged_result.pairs_path.name,
            skips_path=output / staged_result.skips_path.name,
            run_path=output / staged_result.run_path.name,
            usable_examples=staged_result.usable_examples,
            student_tokens=staged_result.student_tokens,
        )
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def load_mistral_factorial_data_bundle(
    root: Path,
    *,
    seed: int,
) -> MistralFactorialDataBundle:
    """Load one closed-world factorial pair stream and reject any drift."""

    raw_root = Path(root)
    _reject_path_symlink_components(raw_root, artifact="data")
    resolved = raw_root.resolve()
    _reject_tree_links(resolved, artifact="data")
    if seed not in MATCHED_REPLICATION_SEEDS:
        raise ValueError("factorial data seed differs")
    run_path = resolved / "run.json"
    manifest_path = resolved / "manifest.json"
    pairs_path = resolved / "pairs.jsonl"
    skips_path = resolved / "skips.jsonl"
    expected_tree_files = {
        "run.json",
        "manifest.json",
        "pairs.jsonl",
        "skips.jsonl",
        f"{_SOURCE_DIR}/manifest.json",
        f"{_SOURCE_DIR}/packed_sources.jsonl",
        f"{_SOURCE_DIR}/run.json",
    }
    actual_tree_files = {
        str(path.relative_to(resolved)) for path in resolved.rglob("*") if path.is_file()
    }
    actual_tree_dirs = {
        str(path.relative_to(resolved)) for path in resolved.rglob("*") if path.is_dir()
    }
    if actual_tree_files != expected_tree_files or actual_tree_dirs != {_SOURCE_DIR}:
        raise ValueError("factorial data closed-world tree inventory differs")
    run, run_sha = _object(run_path)
    manifest, manifest_sha = _object(manifest_path)
    expected_artifacts = {
        "pairs.jsonl",
        "skips.jsonl",
        f"{_SOURCE_DIR}/manifest.json",
        f"{_SOURCE_DIR}/packed_sources.jsonl",
        f"{_SOURCE_DIR}/run.json",
    }
    if (
        run.get("schema_version") != _RUN_SCHEMA
        or run.get("status") != "completed"
        or run.get("seed") != seed
        or manifest.get("schema_version") != _MANIFEST_SCHEMA
        or manifest.get("method_identity") != FACTORIAL_METHOD_IDENTITY
        or manifest.get("seed") != seed
    ):
        raise ValueError("factorial data run identity differs")
    outputs = run.get("outputs")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != {"manifest.json", *expected_artifacts}
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != expected_artifacts
    ):
        raise ValueError("factorial data closed-world artifact inventory differs")
    if run.get("manifest_sha256") != manifest_sha or outputs.get("manifest.json") != {
        "sha256": manifest_sha
    }:
        raise ValueError("factorial manifest hash differs")
    artifact_hashes: dict[str, str] = {"manifest.json": manifest_sha}
    for name in sorted(expected_artifacts):
        path = resolved / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"factorial data artifact differs: {name}")
        digest = _sha256_file(path)
        if outputs.get(name) != {"sha256": digest} or artifacts.get(name) != {"sha256": digest}:
            raise ValueError(f"factorial data artifact hash differs: {name}")
        artifact_hashes[name] = digest

    parent = load_kojima_faithful_data_bundle(resolved / _SOURCE_DIR, seed=seed)
    model = manifest.get("model")
    dataset = manifest.get("dataset")
    tokenizer = manifest.get("tokenizer")
    packing = manifest.get("packing")
    pairing = manifest.get("pairing")
    noise = manifest.get("noise")
    runtime = manifest.get("runtime")
    parent_fields = manifest.get("parent")
    tokenizer_provenance = (
        _validate_tokenizer_provenance(tokenizer.get("snapshot_attestation"))
        if isinstance(tokenizer, Mapping)
        else None
    )
    if (
        model != {"id": MISTRAL_MODEL, "revision": MISTRAL_REVISION}
        or dataset
        != {
            "id": FINEWEB_DATASET,
            "revision": FINEWEB_REVISION,
            "data_file": FINEWEB_DATA_FILE,
            "data_file_sha256": parent.source_file_sha256,
        }
        or not isinstance(tokenizer, Mapping)
        or set(tokenizer) != {"id", "revision", "snapshot_attestation"}
        or tokenizer.get("id") != MISTRAL_MODEL
        or tokenizer.get("revision") != MISTRAL_REVISION
        or tokenizer_provenance is None
        or not isinstance(packing, Mapping)
        or set(packing)
        != {
            "policy",
            "source_order_sha256",
            "packed_attempts",
            "consumed_attempts",
            "usable_examples",
            "max_sequence_length",
            "target_student_tokens",
        }
        or packing.get("policy") != PACKING_POLICY
        or packing.get("source_order_sha256") != parent.source_order_sha256
        or packing.get("packed_attempts") != len(parent.sources)
        or isinstance(packing.get("consumed_attempts"), bool)
        or not isinstance(packing.get("consumed_attempts"), int)
        or not 0 < int(packing["consumed_attempts"]) <= len(parent.sources)
        or isinstance(packing.get("usable_examples"), bool)
        or not isinstance(packing.get("usable_examples"), int)
        or int(packing["usable_examples"]) <= 0
        or isinstance(packing.get("target_student_tokens"), bool)
        or not isinstance(packing.get("target_student_tokens"), int)
        or packing.get("max_sequence_length") != MAX_SEQUENCE_LENGTH
        or not isinstance(pairing, Mapping)
        or set(pairing)
        != {
            "policy",
            "clean_probability",
            "first_row",
            "realized_clean_examples",
            "realized_noisy_examples",
        }
        or pairing.get("policy") != FACTORIAL_PAIRING_POLICY
        or pairing.get("clean_probability") != 0.5
        or pairing.get("first_row") != "clean"
        or not isinstance(noise, Mapping)
        or set(noise)
        != {
            "policy",
            "operations",
            "edit_count_distribution",
            "minimum_word_letters",
            "realized_operation_counts",
            "realized_edit_count_counts",
        }
        or noise.get("policy") != FACTORIAL_NOISE_POLICY
        or noise.get("operations") != dict(FACTORIAL_OPERATIONS)
        or noise.get("edit_count_distribution") != dict(FACTORIAL_EDIT_COUNTS)
        or noise.get("minimum_word_letters") != 2
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "policy",
            "parent_prefix_policy",
            "prevalidated_downstream_offsets",
            "skipped_attempts",
        }
        or runtime.get("policy") != FACTORIAL_RUNTIME_POLICY
        or runtime.get("parent_prefix_policy") != FACTORIAL_PARENT_PREFIX_POLICY
        or runtime.get("prevalidated_downstream_offsets") != [2, 16]
        or not isinstance(parent_fields, Mapping)
        or set(parent_fields) != {"data_identity_sha256", "artifacts"}
        or parent_fields.get("data_identity_sha256") != parent.data_identity_sha256
        or parent_fields.get("artifacts")
        != {
            name: artifact_hashes[name]
            for name in sorted(expected_artifacts)
            if name.startswith(f"{_SOURCE_DIR}/")
        }
    ):
        raise ValueError("factorial frozen source/noise contract differs")

    attested_tokenizer, observed_tokenizer_provenance = _load_attested_factorial_tokenizer()
    if dict(tokenizer_provenance) != dict(observed_tokenizer_provenance):
        raise ValueError("factorial tokenizer differs from its external frozen attestation")

    generator = TypoGenerator(
        seed=seed,
        operation_weights=FACTORIAL_OPERATIONS,
        edit_count_weights=FACTORIAL_EDIT_COUNTS,
        explicit_clean_pair_probability=0.0,
        minimum_word_letters=2,
    )
    rows: list[TrainingPair] = []
    previous_attempt = -1
    for line_number, line in read_lf_jsonl_lines(pairs_path, context="factorial pairs"):
        value = strict_loads(line, context=f"{pairs_path}:{line_number}")
        pair = _pair_from_row(value, expected_seed=seed, expected_index=len(rows))
        attempt = int(pair.metadata["attempt_index"])
        if attempt <= previous_attempt:
            raise ValueError("factorial pair attempt order differs")
        previous_attempt = attempt
        rows.append(pair)
    realized_clean = sum(pair.is_noop for pair in rows)
    realized_noisy = len(rows) - realized_clean
    realized_operation_counts = Counter(edit.operation for pair in rows for edit in pair.edits)
    realized_edit_count_counts = Counter(
        "3-4" if len(pair.edits) >= 3 else str(len(pair.edits)) for pair in rows if not pair.is_noop
    )
    if (
        not rows
        or packing.get("usable_examples") != len(rows)
        or packing.get("target_student_tokens") != len(rows) * MAX_SEQUENCE_LENGTH
        or packing.get("consumed_attempts") != previous_attempt + 1
        or len(rows) % 2
        or pairing.get("realized_clean_examples") != realized_clean
        or pairing.get("realized_noisy_examples") != realized_noisy
        or realized_clean != realized_noisy
        or noise.get("realized_operation_counts")
        != {
            operation: realized_operation_counts[operation]
            for operation in sorted(FACTORIAL_OPERATIONS)
        }
        or noise.get("realized_edit_count_counts")
        != {bucket: realized_edit_count_counts[bucket] for bucket in ("1", "2", "3-4")}
    ):
        raise ValueError("factorial usable pair/token accounting differs")

    skipped_attempts: list[int] = []
    skip_rows: dict[int, Mapping[str, object]] = {}
    skip_lines = (
        read_lf_jsonl_lines(skips_path, context="factorial skips")
        if skips_path.stat().st_size
        else ()
    )
    for line_number, line in skip_lines:
        value = strict_loads(line, context=f"{skips_path}:{line_number}")
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "schema_version",
                "seed",
                "attempt_index",
                "source_record_id",
                "intended_is_noop",
                "reason_type",
                "reason_sha256",
            }
            or value.get("schema_version") != _SKIP_SCHEMA
            or value.get("seed") != seed
            or isinstance(value.get("attempt_index"), bool)
            or not isinstance(value.get("attempt_index"), int)
            or not 0 <= int(value["attempt_index"]) < int(packing["consumed_attempts"])
            or type(value.get("intended_is_noop")) is not bool
            or not isinstance(value.get("reason_type"), str)
            or not isinstance(value.get("source_record_id"), str)
            or _SHA256.fullmatch(str(value["source_record_id"])) is None
            or not isinstance(value.get("reason_sha256"), str)
            or _SHA256.fullmatch(str(value["reason_sha256"])) is None
        ):
            raise ValueError("factorial skip ledger differs")
        attempt = int(value["attempt_index"])
        skipped_attempts.append(attempt)
        skip_rows[attempt] = value
    used_attempts = {int(pair.metadata["attempt_index"]) for pair in rows}
    expected_attempts = set(range(int(packing["consumed_attempts"])))
    if (
        len(skipped_attempts) != len(set(skipped_attempts))
        or used_attempts.intersection(skipped_attempts)
        or used_attempts.union(skipped_attempts) != expected_attempts
        or isinstance(runtime.get("skipped_attempts"), bool)
        or not isinstance(runtime.get("skipped_attempts"), int)
        or runtime.get("skipped_attempts") != len(skipped_attempts)
    ):
        raise ValueError("factorial skip/replacement decisions are incomplete")
    used_by_attempt = {int(pair.metadata["attempt_index"]): pair for pair in rows}
    accepted = 0
    for attempt in range(int(packing["consumed_attempts"])):
        intended_is_noop = accepted % 2 == 0
        expected_pair = materialize_training_pair(
            parent.sources[attempt],
            generator=generator,
            epoch=0,
            variant=0,
            force_noop=intended_is_noop,
        )
        encoding_error: ValueError | RuntimeError | None = None
        try:
            _encode_factorial_pair(
                expected_pair,
                source=parent.sources[attempt],
                tokenizer=attested_tokenizer,
            )
        except (ValueError, RuntimeError) as exc:
            encoding_error = exc
        if attempt in used_by_attempt:
            pair = used_by_attempt[attempt]
            if encoding_error is not None:
                raise ValueError("factorial accepted attempt fails tokenizer revalidation")
            if (
                pair.record_id != expected_pair.record_id
                or pair.clean_text != expected_pair.clean_text
                or pair.typo_text != expected_pair.typo_text
                or pair.edits != expected_pair.edits
                or pair.is_noop != expected_pair.is_noop
            ):
                raise ValueError("factorial pair differs from its deterministic source realization")
            if pair.is_noop != intended_is_noop:
                raise ValueError("factorial usable attempt changes the alternating group")
            accepted += 1
            continue
        skip = skip_rows[attempt]
        if (
            encoding_error is None
            or skip.get("reason_type") != type(encoding_error).__name__
            or skip.get("reason_sha256") != hashlib.sha256(str(encoding_error).encode()).hexdigest()
            or skip.get("intended_is_noop") != intended_is_noop
            or skip.get("source_record_id") != parent.sources[attempt].record_id
        ):
            raise ValueError("factorial skip ledger differs from its source attempt")

    _reject_tree_links(resolved, artifact="data")
    final_tree_files = {
        str(path.relative_to(resolved)) for path in resolved.rglob("*") if path.is_file()
    }
    final_tree_dirs = {
        str(path.relative_to(resolved)) for path in resolved.rglob("*") if path.is_dir()
    }
    if final_tree_files != expected_tree_files or final_tree_dirs != {_SOURCE_DIR}:
        raise ValueError("factorial data tree changed during validation")
    if _sha256_file(run_path) != run_sha or any(
        _sha256_file(resolved / name) != digest for name, digest in artifact_hashes.items()
    ):
        raise ValueError("factorial data changed during validation")

    identity = _canonical_sha(
        {
            "manifest_sha256": manifest_sha,
            "pairs_sha256": artifact_hashes["pairs.jsonl"],
            "skips_sha256": artifact_hashes["skips.jsonl"],
            "seed": seed,
        }
    )
    return MistralFactorialDataBundle(
        root=resolved,
        pairs=tuple(rows),
        data_identity_sha256=identity,
        training_data_sha256=identity,
        artifact_sha256=MappingProxyType(artifact_hashes),
        seed=seed,
        source_revision=FINEWEB_REVISION,
        source_order_sha256=parent.source_order_sha256,
        packing_policy=PACKING_POLICY,
        pairing_policy=FACTORIAL_PAIRING_POLICY,
        noise_policy=FACTORIAL_NOISE_POLICY,
        target_usable_examples=len(rows),
        target_student_tokens=len(rows) * MAX_SEQUENCE_LENGTH,
        packed_attempts=int(packing["packed_attempts"]),
        consumed_attempts=int(packing["consumed_attempts"]),
        tokenizer_snapshot_attestation=tokenizer_provenance,
    )


__all__ = [
    "FACTORIAL_EDIT_COUNTS",
    "FACTORIAL_INITIALIZATION_POLICY",
    "FACTORIAL_METHOD_IDENTITY",
    "FACTORIAL_NOISE_POLICY",
    "FACTORIAL_OPERATIONS",
    "FACTORIAL_PARENT_PREFIX_POLICY",
    "FACTORIAL_PAIRING_POLICY",
    "FACTORIAL_RUNTIME_POLICY",
    "MistralFactorialDataBundle",
    "PrepareMistralFactorialDataConfig",
    "PrepareMistralFactorialDataResult",
    "load_mistral_factorial_data_bundle",
    "prepare_mistral_factorial_data",
]

"""Hash-attested data path for a faithful Kojima et al. Mistral reproduction.

The public implementation builds one 8,192-token example by shuffling FineWeb
documents, concatenating them with the tokenizer BOS string until the buffer is
at least 500 tokenizer tokens over length, removing NBSP, and round-tripping
through tokenize -> ids -> decode.  This module freezes that otherwise implicit
stream into a seed-specific artifact before training.  Training still generates
the four public typo operations on the fly, but the runtime RNG and the exact
packed-source cursor are checkpointed by the shared runner.

The upstream repository pins ``datasets`` but not PyTorch, so its DataLoader
shuffle is not portable across PyTorch releases.  This reproduction deliberately
uses ``torch.randperm`` from the project's pinned PyTorch version and records the
version in the artifact.  It also replaces invalid aligned pairs before forming
a fixed-size accumulation batch, whereas public optimizer boundaries are tied to
the raw attempt index.  These and the project-locked dependency versions are
attested departures; this path must not be
described as bit-exact.  Packing, noise, masking, and teacher semantics otherwise
match the public implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from typo_robust_training.data.config import strict_loads
from typo_robust_training.data.jsonl import read_lf_jsonl_lines
from typo_robust_training.data.records import TypoEdit
from typo_robust_training.training.encoding import PairedEncoding
from typo_robust_training.training.pairs import TrainingPair, TrainingSource


FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB_DATA_FILE = "sample/10BT/000_00000.parquet"
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.1"
MISTRAL_REVISION = "7231864981174d9bee8c7687c24c8344414eae6b"
UPSTREAM_REPOSITORY = "https://github.com/characternlp/charnoise"
UPSTREAM_REVISION = "4cb90b28e9f6976046a6e93aec2dcab27e76555d"
PUBLIC_ANCHOR_SEED = 1
MATCHED_REPLICATION_SEEDS = (42, 43, 44)
ALLOWED_SEEDS = (PUBLIC_ANCHOR_SEED, *MATCHED_REPLICATION_SEEDS)
MAX_SEQUENCE_LENGTH = 8192
PACKING_OVERFILL_TOKENS = 500
PACKED_EXAMPLES = 8800
TARGET_USABLE_EXAMPLES = 8000
TARGET_STUDENT_TOKENS = TARGET_USABLE_EXAMPLES * MAX_SEQUENCE_LENGTH
UPSTREAM_VALIDATION_DOCUMENTS = 20
PACKING_POLICY = "kojima-bos-overfill500-canonicalize-truncate8192/v2"
SHUFFLE_POLICY = "torch-randperm-pinned-project-runtime/v1"
NOISE_POLICY = "kojima-python-random-global-four-operation/v1"
SKIP_POLICY = "skip-unusable-and-replace-before-usable-accumulation/v1"
_MANIFEST_SCHEMA = "kojima-faithful-packed-fineweb-manifest/v1"
_ROW_SCHEMA = "kojima-faithful-packed-fineweb-row/v1"
_RUN_SCHEMA = "prepare-kojima-faithful-packed-fineweb-run/v1"


def _noise_manifest() -> dict[str, object]:
    return {
        "policy": NOISE_POLICY,
        "document_clean_probability": 0.5,
        "noisy_document_frequency_distribution": "uniform-[0,1)",
        "minimum_word_length": 3,
        "ascii_only": True,
        "operations": ["delete", "swap", "addition", "random-replace"],
        "operation_probability": 0.25,
    }


def _reproduction_departure() -> dict[str, object]:
    return {
        "row_order": {
            "field": "shuffled FineWeb row order",
            "upstream_unpinned_dependency": "torch",
            "resolution": SHUFFLE_POLICY,
            "semantic_effect": "row order only; public packing and noise rules are unchanged",
        },
        "attempt_boundary": {
            "upstream_behavior": "8800 j-index attempts with optimizer boundaries tied to j",
            "resolution": SKIP_POLICY,
            "semantic_effect": (
                "unusable attempts advance the frozen source cursor but do not consume "
                "a micro-step or token; each update contains 16 usable examples"
            ),
        },
        "dependency_stack": {
            "upstream_pinned": {
                "datasets": "3.2.0",
                "transformers": "4.44.2",
                "peft": "0.14.0",
                "torch": "unversioned",
            },
            "reproduction_pinned": _runtime_versions(),
            "semantic_effect": (
                "project-locked dependency behavior is hash-attested but not claimed "
                "bit-identical to the public environment"
            ),
        },
    }


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
        ).encode()
    ).hexdigest()


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "datasets": _version("datasets"),
        "transformers": _version("transformers"),
        "peft": _version("peft"),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tokenize_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if not isinstance(tokens, list) or not isinstance(ids, list) or len(tokens) != len(ids):
        raise ValueError("Kojima tokenizer tokenize/id conversion differs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
        raise ValueError("Kojima tokenizer returned invalid token IDs")
    return tuple(ids)


def _tokenizer_revision(tokenizer: Any) -> str:
    kwargs = getattr(tokenizer, "init_kwargs", None)
    revision = kwargs.get("_commit_hash") if isinstance(kwargs, Mapping) else None
    if revision != MISTRAL_REVISION:
        raise ValueError("Kojima tokenizer revision differs from the frozen Mistral revision")
    return str(revision)


class FineWebTextProvider(Protocol):
    """Minimal provider used by the real builder and deterministic tests."""

    source_file_sha256: str

    def texts(self) -> Sequence[str]: ...


class HuggingFaceFineWebTextProvider:
    """Download exactly one pinned FineWeb parquet and expose its row order."""

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=FINEWEB_DATASET,
                repo_type="dataset",
                filename=FINEWEB_DATA_FILE,
                revision=FINEWEB_REVISION,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        ).resolve()
        if not path.is_file():
            raise RuntimeError("pinned FineWeb parquet download is unavailable")
        self.path = path
        self.source_file_sha256 = _sha256_file(path)

    def texts(self) -> Sequence[str]:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(self.path), split="train")
        if "text" not in dataset.column_names:
            raise ValueError("pinned FineWeb parquet has no text column")
        return dataset["text"]


@dataclass(frozen=True, slots=True)
class PrepareKojimaFaithfulDataConfig:
    seed: int
    output_dir: Path
    cache_dir: Path | None = None
    packed_examples: int = PACKED_EXAMPLES

    def __post_init__(self) -> None:
        if self.seed not in ALLOWED_SEEDS:
            raise ValueError("Kojima data seed is outside anchor/matched replication inventory")
        if (
            isinstance(self.packed_examples, bool)
            or not isinstance(self.packed_examples, int)
            or self.packed_examples <= 0
        ):
            raise ValueError("Kojima packed example count must be positive")


@dataclass(frozen=True, slots=True)
class PrepareKojimaFaithfulDataResult:
    manifest_path: Path
    packed_sources_path: Path
    run_path: Path
    packed_examples: int
    student_tokens: int


@dataclass(frozen=True, slots=True)
class KojimaFaithfulDataBundle:
    root: Path
    sources: tuple[TrainingSource, ...]
    generator: "KojimaFaithfulNoiseGenerator"
    data_identity_sha256: str
    training_data_sha256: str
    artifact_sha256: Mapping[str, str]
    seed: int
    source_file_sha256: str
    source_revision: str
    source_order_sha256: str
    packing_policy: str
    rng_policy: str
    packed_attempts: int
    target_usable_examples: int


class UnusableKojimaFaithfulPairError(ValueError):
    """A public clean/noisy valid-position mismatch that must be skipped."""


def _shuffled_indices(count: int, *, seed: int) -> tuple[int, ...]:
    import torch

    if count <= UPSTREAM_VALIDATION_DOCUMENTS:
        raise ValueError("FineWeb inventory is too small after upstream validation holdout")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return tuple(int(index) for index in torch.randperm(count, generator=generator).tolist())


def _clean_raw_document(text: object, *, bos_token: str) -> str:
    if not isinstance(text, str):
        raise ValueError("FineWeb text must be a string")
    return text.strip("\n").strip(" ") + bos_token


def _canonicalize_packed_document(tokenizer: Any, text: str) -> tuple[str, tuple[int, ...]]:
    without_nbsp = text.replace("\xa0", "")
    ids = _tokenize_ids(tokenizer, without_nbsp)
    canonical = tokenizer.decode(list(ids), add_special_tokens=False)
    canonical_ids = _tokenize_ids(tokenizer, canonical)
    if canonical_ids != ids:
        raise ValueError("Kojima tokenize/decode canonicalization is not idempotent")
    if len(canonical_ids) < MAX_SEQUENCE_LENGTH:
        raise ValueError("Kojima packed document is shorter than the fixed context")
    return canonical, canonical_ids[:MAX_SEQUENCE_LENGTH]


def iter_packed_documents(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    seed: int,
    packed_examples: int,
) -> Iterator[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Yield canonical packed text, first 8192 IDs, and consumed raw row IDs."""

    if seed not in ALLOWED_SEEDS:
        raise ValueError("Kojima packing seed differs")
    bos_token = getattr(tokenizer, "bos_token", None) or getattr(tokenizer, "eos_token", None)
    if not isinstance(bos_token, str) or not bos_token:
        raise ValueError("Kojima tokenizer has no BOS/EOS separator string")
    order = _shuffled_indices(len(texts), seed=seed)
    cursor = UPSTREAM_VALIDATION_DOCUMENTS
    for _packed_index in range(packed_examples):
        clean_document = bos_token
        consumed: list[int] = []
        while len(_tokenize_ids(tokenizer, clean_document)) < (
            MAX_SEQUENCE_LENGTH + PACKING_OVERFILL_TOKENS
        ):
            if cursor >= len(order):
                raise ValueError("pinned FineWeb file exhausted before the frozen token budget")
            row_index = order[cursor]
            cursor += 1
            consumed.append(row_index)
            clean_document += _clean_raw_document(texts[row_index], bos_token=bos_token)
        canonical, prefix_ids = _canonicalize_packed_document(tokenizer, clean_document)
        yield canonical, prefix_ids, tuple(consumed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_kojima_faithful_data(
    config: PrepareKojimaFaithfulDataConfig,
    *,
    provider: FineWebTextProvider | None = None,
    tokenizer: Any | None = None,
) -> PrepareKojimaFaithfulDataResult:
    """Freeze a seed-specific, hash-bound faithful FineWeb packing stream."""

    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Kojima data output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if provider is None:
        provider = HuggingFaceFineWebTextProvider(cache_dir=config.cache_dir)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MISTRAL_MODEL,
            revision=MISTRAL_REVISION,
            cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        )
    tokenizer_revision = _tokenizer_revision(tokenizer)
    bos_token = getattr(tokenizer, "bos_token", None) or getattr(tokenizer, "eos_token", None)
    if not isinstance(bos_token, str) or not bos_token:
        raise ValueError("Kojima tokenizer has no separator token")
    texts = provider.texts()
    packed_path = output / "packed_sources.jsonl"
    order_rows: list[dict[str, object]] = []
    with packed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for packed_index, (clean_text, prefix_ids, raw_rows) in enumerate(
            iter_packed_documents(
                texts,
                tokenizer=tokenizer,
                seed=config.seed,
                packed_examples=config.packed_examples,
            )
        ):
            token_sha = hashlib.sha256(
                b"".join(int(token).to_bytes(4, "big", signed=False) for token in prefix_ids)
            ).hexdigest()
            record_id = _canonical_sha(
                {
                    "dataset": FINEWEB_DATASET,
                    "revision": FINEWEB_REVISION,
                    "file_sha256": provider.source_file_sha256,
                    "seed": config.seed,
                    "packed_index": packed_index,
                    "raw_rows": raw_rows,
                    "clean_prefix_token_ids_sha256": token_sha,
                }
            )
            row = {
                "schema_version": _ROW_SCHEMA,
                "record_id": record_id,
                "packed_index": packed_index,
                "seed": config.seed,
                "clean_text": clean_text,
                "clean_text_sha256": hashlib.sha256(clean_text.encode()).hexdigest(),
                "clean_prefix_token_ids_sha256": token_sha,
                "student_tokens": MAX_SEQUENCE_LENGTH,
                "raw_row_indices": list(raw_rows),
            }
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            order_rows.append(
                {"record_id": record_id, "raw_row_indices": list(raw_rows)}
            )
    packed_sha = _sha256_file(packed_path)
    source_order_sha = _canonical_sha(order_rows)
    target_usable_examples = (
        TARGET_USABLE_EXAMPLES
        if config.packed_examples == PACKED_EXAMPLES
        else config.packed_examples
    )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "seed": config.seed,
        "seed_role": (
            "public-anchor" if config.seed == PUBLIC_ANCHOR_SEED else "matched-replication"
        ),
        "matched_inference_pool": list(MATCHED_REPLICATION_SEEDS),
        "public_anchor_excluded_from_matched_inference": True,
        "dataset": {
            "id": FINEWEB_DATASET,
            "revision": FINEWEB_REVISION,
            "data_file": FINEWEB_DATA_FILE,
            "data_file_sha256": provider.source_file_sha256,
        },
        "tokenizer": {
            "id": MISTRAL_MODEL,
            "revision": tokenizer_revision,
            "bos_token": bos_token,
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
        },
        "packing": {
            "policy": PACKING_POLICY,
            "shuffle_policy": SHUFFLE_POLICY,
            "upstream_validation_documents_skipped": UPSTREAM_VALIDATION_DOCUMENTS,
            "overfill_tokens": PACKING_OVERFILL_TOKENS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "packed_attempts": config.packed_examples,
            "target_usable_examples": target_usable_examples,
            "packed_attempt_tokens": config.packed_examples * MAX_SEQUENCE_LENGTH,
            "target_student_tokens": target_usable_examples * MAX_SEQUENCE_LENGTH,
            "source_order_sha256": source_order_sha,
        },
        "noise": _noise_manifest(),
        "runtime_versions": _runtime_versions(),
        "reproduction_departure": _reproduction_departure(),
        "artifacts": {"packed_sources.jsonl": {"sha256": packed_sha}},
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
            "packed_sources.jsonl": {"sha256": packed_sha},
        },
    }
    run_path = output / "run.json"
    _write_json(run_path, run)
    return PrepareKojimaFaithfulDataResult(
        manifest_path=manifest_path,
        packed_sources_path=packed_path,
        run_path=run_path,
        packed_examples=config.packed_examples,
        student_tokens=target_usable_examples * MAX_SEQUENCE_LENGTH,
    )


def _load_object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"Kojima data artifact is missing: {path}")
    value = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"Kojima data artifact must be an object: {path}")
    return value


def _packed_sources(
    path: Path,
    *,
    manifest: Mapping[str, object],
    seed: int,
    source_file_sha256: str,
) -> tuple[TrainingSource, ...]:
    packing = manifest.get("packing")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(packing, Mapping) or not isinstance(tokenizer, Mapping):
        raise ValueError("Kojima data packing/tokenizer manifest differs")
    expected_count = packing.get("packed_attempts")
    rows: list[TrainingSource] = []
    order_rows: list[dict[str, object]] = []
    previous_raw_row: set[int] = set()
    for line_number, line in read_lf_jsonl_lines(path, context="Kojima packed sources"):
        row = strict_loads(line, context=f"{path}:{line_number}")
        fields = {
            "schema_version",
            "record_id",
            "packed_index",
            "seed",
            "clean_text",
            "clean_text_sha256",
            "clean_prefix_token_ids_sha256",
            "student_tokens",
            "raw_row_indices",
        }
        if not isinstance(row, Mapping) or set(row) != fields or row.get("schema_version") != _ROW_SCHEMA:
            raise ValueError("Kojima packed source fields differ")
        index = len(rows)
        clean_text = row.get("clean_text")
        raw_rows = row.get("raw_row_indices")
        if (
            row.get("packed_index") != index
            or row.get("seed") != seed
            or not isinstance(clean_text, str)
            or not clean_text
            or row.get("clean_text_sha256") != hashlib.sha256(clean_text.encode()).hexdigest()
            or row.get("student_tokens") != MAX_SEQUENCE_LENGTH
            or not isinstance(raw_rows, list)
            or not raw_rows
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw_rows)
            or previous_raw_row.intersection(raw_rows)
        ):
            raise ValueError("Kojima packed source identity/order differs")
        previous_raw_row.update(raw_rows)
        token_sha = row.get("clean_prefix_token_ids_sha256")
        record_id = row.get("record_id")
        if (
            not _is_sha256(record_id)
            or not _is_sha256(token_sha)
        ):
            raise ValueError("Kojima packed source hash fields differ")
        expected_record_id = _canonical_sha(
            {
                "dataset": FINEWEB_DATASET,
                "revision": FINEWEB_REVISION,
                "file_sha256": source_file_sha256,
                "seed": seed,
                "packed_index": index,
                "raw_rows": raw_rows,
                "clean_prefix_token_ids_sha256": token_sha,
            }
        )
        if record_id != expected_record_id:
            raise ValueError("Kojima packed source record_id is not derivable from its source")
        rows.append(
            TrainingSource(
                kind="clean",
                record_id=record_id,
                source=FINEWEB_DATASET,
                source_revision=FINEWEB_REVISION,
                source_split=FINEWEB_DATA_FILE,
                source_id=f"packed-{index:06d}",
                group_id=record_id,
                clean_text=clean_text,
                typo_text=None,
                task=None,
                answer=None,
                operation=None,
                metadata={
                    "kojima_faithful": True,
                    "packed_index": index,
                    "clean_prefix_token_ids_sha256": token_sha,
                    "bos_token": tokenizer.get("bos_token"),
                    "raw_row_indices": tuple(raw_rows),
                },
                token_count=MAX_SEQUENCE_LENGTH,
            )
        )
        order_rows.append({"record_id": record_id, "raw_row_indices": raw_rows})
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or expected_count != len(rows)
    ):
        raise ValueError("Kojima packed source count differs from the 64M-token protocol")
    if packing.get("source_order_sha256") != _canonical_sha(order_rows):
        raise ValueError("Kojima packed source order hash differs")
    return tuple(rows)


def load_kojima_faithful_data_bundle(
    root: Path,
    *,
    seed: int,
) -> KojimaFaithfulDataBundle:
    """Load only a complete seed-specific packing bound to the frozen source."""

    resolved = Path(root).resolve()
    run_path = resolved / "run.json"
    manifest_path = resolved / "manifest.json"
    packed_path = resolved / "packed_sources.jsonl"
    run = _load_object(run_path)
    manifest = _load_object(manifest_path)
    run_fields = {
        "schema_version",
        "status",
        "seed",
        "manifest_sha256",
        "outputs",
    }
    manifest_fields = {
        "schema_version",
        "seed",
        "seed_role",
        "matched_inference_pool",
        "public_anchor_excluded_from_matched_inference",
        "dataset",
        "tokenizer",
        "upstream",
        "packing",
        "noise",
        "runtime_versions",
        "reproduction_departure",
        "artifacts",
    }
    if (
        set(run) != run_fields
        or set(manifest) != manifest_fields
        or run.get("schema_version") != _RUN_SCHEMA
        or run.get("status") != "completed"
        or manifest.get("schema_version") != _MANIFEST_SCHEMA
        or run.get("seed") != seed
        or manifest.get("seed") != seed
        or seed not in ALLOWED_SEEDS
    ):
        raise ValueError("Kojima data run/seed identity differs")
    outputs = run.get("outputs")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != {"manifest.json", "packed_sources.jsonl"}
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"packed_sources.jsonl"}
    ):
        raise ValueError("Kojima data output inventory differs")
    manifest_sha = _sha256_file(manifest_path)
    packed_sha = _sha256_file(packed_path)
    if (
        run.get("manifest_sha256") != manifest_sha
        or outputs.get("manifest.json") != {"sha256": manifest_sha}
        or outputs.get("packed_sources.jsonl") != {"sha256": packed_sha}
        or artifacts.get("packed_sources.jsonl") != {"sha256": packed_sha}
    ):
        raise ValueError("Kojima data artifact hash differs")
    dataset = manifest.get("dataset")
    tokenizer = manifest.get("tokenizer")
    upstream = manifest.get("upstream")
    packing = manifest.get("packing")
    noise = manifest.get("noise")
    runtime_versions = manifest.get("runtime_versions")
    reproduction_departure = manifest.get("reproduction_departure")
    source_file_sha256 = (
        dataset.get("data_file_sha256") if isinstance(dataset, Mapping) else None
    )
    expected_dataset = {
        "id": FINEWEB_DATASET,
        "revision": FINEWEB_REVISION,
        "data_file": FINEWEB_DATA_FILE,
        "data_file_sha256": source_file_sha256,
    }
    packing_fields = {
        "policy",
        "shuffle_policy",
        "upstream_validation_documents_skipped",
        "overfill_tokens",
        "max_sequence_length",
        "packed_attempts",
        "target_usable_examples",
        "packed_attempt_tokens",
        "target_student_tokens",
        "source_order_sha256",
    }
    if (
        not isinstance(dataset, Mapping)
        or set(dataset) != set(expected_dataset)
        or dataset != expected_dataset
        or not _is_sha256(source_file_sha256)
        or not isinstance(tokenizer, Mapping)
        or set(tokenizer) != {"id", "revision", "bos_token"}
        or tokenizer.get("id") != MISTRAL_MODEL
        or tokenizer.get("revision") != MISTRAL_REVISION
        or not isinstance(tokenizer.get("bos_token"), str)
        or not tokenizer.get("bos_token")
        or upstream != {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION}
        or not isinstance(packing, Mapping)
        or set(packing) != packing_fields
        or packing.get("policy") != PACKING_POLICY
        or packing.get("shuffle_policy") != SHUFFLE_POLICY
        or packing.get("upstream_validation_documents_skipped")
        != UPSTREAM_VALIDATION_DOCUMENTS
        or packing.get("overfill_tokens") != PACKING_OVERFILL_TOKENS
        or packing.get("max_sequence_length") != MAX_SEQUENCE_LENGTH
        or isinstance(packing.get("packed_attempts"), bool)
        or not isinstance(packing.get("packed_attempts"), int)
        or int(packing["packed_attempts"]) <= 0
        or isinstance(packing.get("target_usable_examples"), bool)
        or not isinstance(packing.get("target_usable_examples"), int)
        or not 0 < int(packing["target_usable_examples"]) <= int(packing["packed_attempts"])
        or packing.get("packed_attempt_tokens")
        != int(packing["packed_attempts"]) * MAX_SEQUENCE_LENGTH
        or packing.get("target_student_tokens")
        != int(packing["target_usable_examples"]) * MAX_SEQUENCE_LENGTH
        or not _is_sha256(packing.get("source_order_sha256"))
        or noise != _noise_manifest()
        or runtime_versions != _runtime_versions()
        or reproduction_departure != _reproduction_departure()
    ):
        raise ValueError("Kojima data frozen source/packing/noise identity differs")
    expected_seed_role = "public-anchor" if seed == PUBLIC_ANCHOR_SEED else "matched-replication"
    if (
        manifest.get("seed_role") != expected_seed_role
        or manifest.get("matched_inference_pool") != list(MATCHED_REPLICATION_SEEDS)
        or manifest.get("public_anchor_excluded_from_matched_inference") is not True
    ):
        raise ValueError("Kojima anchor/matched inference roles differ")
    sources = _packed_sources(
        packed_path,
        manifest=manifest,
        seed=seed,
        source_file_sha256=source_file_sha256,
    )
    identity = _canonical_sha(
        {
            "manifest_sha256": manifest_sha,
            "packed_sources_sha256": packed_sha,
            "seed": seed,
        }
    )
    return KojimaFaithfulDataBundle(
        root=resolved,
        sources=sources,
        generator=KojimaFaithfulNoiseGenerator(
            seed=seed,
            bos_token=str(tokenizer["bos_token"]),
        ),
        data_identity_sha256=identity,
        training_data_sha256=identity,
        artifact_sha256={"manifest.json": manifest_sha, "packed_sources.jsonl": packed_sha},
        seed=seed,
        source_file_sha256=source_file_sha256,
        source_revision=FINEWEB_REVISION,
        source_order_sha256=str(packing["source_order_sha256"]),
        packing_policy=PACKING_POLICY,
        rng_policy=NOISE_POLICY,
        packed_attempts=int(packing["packed_attempts"]),
        target_usable_examples=int(packing["target_usable_examples"]),
    )


def _noise_word(word: str) -> tuple[str, str]:
    operation = random.randint(1, 4)
    characters = list(word)
    if operation == 1:
        characters[random.choice(range(len(characters)))] = ""
        return "".join(characters), "delete"
    if operation == 2:
        left, right = random.sample(range(len(characters)), 2)
        characters[left], characters[right] = characters[right], characters[left]
        return "".join(characters), "swap"
    target = random.choice(range(len(characters)))
    replacement = chr(random.choice(range(97, 123)))
    if operation == 3:
        characters[target] = characters[target] + replacement
        return "".join(characters), "addition"
    characters[target] = replacement
    return "".join(characters), "random-replace"


def _join_with_spans(
    clean_words: Sequence[str],
    noisy_words: Sequence[str],
    operations: Sequence[str | None],
    *,
    separator: str,
    clean_start: int,
    typo_start: int,
) -> tuple[str, str, tuple[TypoEdit, ...]]:
    clean_parts: list[str] = []
    typo_parts: list[str] = []
    edits: list[TypoEdit] = []
    clean_cursor = clean_start
    typo_cursor = typo_start
    for index, (clean_word, noisy_word, operation) in enumerate(
        zip(clean_words, noisy_words, operations, strict=True)
    ):
        if index:
            clean_parts.append(separator)
            typo_parts.append(separator)
            clean_cursor += len(separator)
            typo_cursor += len(separator)
        clean_parts.append(clean_word)
        typo_parts.append(noisy_word)
        if operation is not None and clean_word != noisy_word:
            edits.append(
                TypoEdit(
                    operation=operation,
                    clean_word=clean_word,
                    typo_word=noisy_word,
                    clean_char_span=(clean_cursor, clean_cursor + len(clean_word)),
                    typo_char_span=(typo_cursor, typo_cursor + len(noisy_word)),
                )
            )
        clean_cursor += len(clean_word)
        typo_cursor += len(noisy_word)
    return "".join(clean_parts), "".join(typo_parts), tuple(edits)


class KojimaFaithfulNoiseGenerator:
    """Public document-level 50% clean / U(0,1) four-operation process."""

    def __init__(self, *, seed: int, bos_token: str) -> None:
        if seed not in ALLOWED_SEEDS or not isinstance(bos_token, str) or not bos_token:
            raise ValueError("Kojima noise generator identity differs")
        self.seed = seed
        self.bos_token = bos_token

    def materialize(self, source: TrainingSource, *, epoch: int) -> TrainingPair:
        if epoch != 0:
            raise ValueError("Kojima faithful 64M stream must not repeat packed examples")
        if source.metadata.get("kojima_faithful") is not True:
            raise ValueError("Kojima generator received a generic project source")
        clean_docs = source.clean_text.split(self.bos_token)
        noisy_docs: list[str] = []
        clean_rebuilt_docs: list[str] = []
        all_edits: list[TypoEdit] = []
        clean_document_cursor = typo_document_cursor = 0
        for doc_index, clean_doc in enumerate(clean_docs):
            frequency = 0.0 if random.random() <= 0.5 else random.random()
            clean_lines = clean_doc.split("\n")
            noisy_lines: list[str] = []
            rebuilt_lines: list[str] = []
            line_edits: list[TypoEdit] = []
            clean_line_cursor = clean_document_cursor
            typo_line_cursor = typo_document_cursor
            for line_index, clean_line in enumerate(clean_lines):
                clean_words = clean_line.split(" ")
                noisy_words: list[str] = []
                operations: list[str | None] = []
                for word in clean_words:
                    if random.random() < frequency and len(word) >= 3 and word.isascii():
                        noisy_word, operation = _noise_word(word)
                    else:
                        noisy_word, operation = word, None
                    noisy_words.append(noisy_word)
                    operations.append(operation)
                rebuilt, noisy, edits = _join_with_spans(
                    clean_words,
                    noisy_words,
                    operations,
                    separator=" ",
                    clean_start=clean_line_cursor,
                    typo_start=typo_line_cursor,
                )
                rebuilt_lines.append(rebuilt)
                noisy_lines.append(noisy)
                line_edits.extend(edits)
                clean_line_cursor += len(rebuilt)
                typo_line_cursor += len(noisy)
                if line_index + 1 < len(clean_lines):
                    clean_line_cursor += 1
                    typo_line_cursor += 1
            clean_rebuilt = "\n".join(rebuilt_lines)
            noisy_doc = "\n".join(noisy_lines)
            clean_rebuilt_docs.append(clean_rebuilt)
            noisy_docs.append(noisy_doc)
            all_edits.extend(line_edits)
            clean_document_cursor += len(clean_rebuilt)
            typo_document_cursor += len(noisy_doc)
            if doc_index + 1 < len(clean_docs):
                clean_document_cursor += len(self.bos_token)
                typo_document_cursor += len(self.bos_token)
        clean_text = self.bos_token.join(clean_rebuilt_docs)
        typo_text = self.bos_token.join(noisy_docs)
        if clean_text != source.clean_text:
            raise RuntimeError("Kojima word splitting failed to reconstruct canonical clean text")
        return TrainingPair(
            record_id=source.record_id,
            clean_text=clean_text,
            typo_text=typo_text,
            task=None,
            answer=None,
            metadata=source.metadata,
            edits=tuple(all_edits),
            is_noop=not all_edits,
            epoch=epoch,
        )


def _word_list(text: str, *, bos_token: str) -> tuple[str, ...]:
    return tuple(
        word
        for document in text.split(bos_token)
        for line in document.split("\n")
        for word in line.split(" ")
    )


def _valid_target_positions(
    tokenizer: Any,
    *,
    text: str,
    words: Sequence[str],
    changed: Sequence[bool],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if not isinstance(tokens, list) or not isinstance(ids, list) or len(tokens) != len(ids):
        raise ValueError("Kojima tokenizer token/ID inventory differs")
    labels = [1] * len(tokens)
    start = 0
    end = 1
    for word, is_changed in zip(words, changed, strict=True):
        while True:
            if end > len(tokens):
                raise ValueError("Kojima public word-to-token mask alignment failed")
            decoded = tokenizer.decode(ids[start:end], add_special_tokens=False)
            if word.replace(" ", "") in decoded.replace(" ", ""):
                if is_changed:
                    for index in range(start, end):
                        labels[index] = 0
                    if end < len(tokens) and tokens[end] == "Ċ":
                        labels[end] = 0
                break
            end += 1
        if word != "":
            start = end
            end = start + 1
    truncated_ids = tuple(int(value) for value in ids[:MAX_SEQUENCE_LENGTH])
    if len(truncated_ids) != MAX_SEQUENCE_LENGTH:
        raise ValueError("Kojima faithful encoding does not fill the fixed context")
    full_valid_count = sum(labels)
    positions = tuple(
        index for index, valid in enumerate(labels[:MAX_SEQUENCE_LENGTH][1:]) if valid == 1
    )
    return truncated_ids, positions, full_valid_count


def encode_kojima_faithful_pair(pair: TrainingPair, *, tokenizer: Any) -> PairedEncoding:
    """Reproduce the public token mask and order-based clean/noisy alignment."""

    if pair.metadata.get("kojima_faithful") is not True:
        raise ValueError("Kojima encoding requires a faithful packed pair")
    bos_token = pair.metadata.get("bos_token")
    expected_prefix_sha = pair.metadata.get("clean_prefix_token_ids_sha256")
    if not isinstance(bos_token, str) or not isinstance(expected_prefix_sha, str):
        raise ValueError("Kojima packed pair is missing tokenizer attestation")
    clean_words = _word_list(pair.clean_text, bos_token=bos_token)
    noisy_words = _word_list(pair.typo_text, bos_token=bos_token)
    if len(clean_words) != len(noisy_words):
        raise ValueError("Kojima clean/noisy word cardinality differs")
    changed = tuple(clean != noisy for clean, noisy in zip(clean_words, noisy_words, strict=True))
    clean_ids, clean_positions, clean_valid_count = _valid_target_positions(
        tokenizer,
        text=pair.clean_text,
        words=clean_words,
        changed=changed,
    )
    typo_ids, typo_positions, typo_valid_count = _valid_target_positions(
        tokenizer,
        text=pair.typo_text,
        words=noisy_words,
        changed=changed,
    )
    token_sha = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "big", signed=False) for token in clean_ids)
    ).hexdigest()
    if token_sha != expected_prefix_sha:
        raise ValueError("Kojima clean prefix token hash differs from prepared data")
    if clean_valid_count != typo_valid_count:
        raise UnusableKojimaFaithfulPairError(
            "Kojima clean/noisy valid target alignment differs"
        )
    token_count = min(len(clean_positions), len(typo_positions))
    if token_count <= 0:
        raise UnusableKojimaFaithfulPairError("Kojima pair has no valid aligned target")
    pairs = tuple(zip(clean_positions[:token_count], typo_positions[:token_count], strict=True))
    ones = (1,) * MAX_SEQUENCE_LENGTH
    return PairedEncoding(
        record_id=pair.record_id,
        clean_input_ids=clean_ids,
        typo_input_ids=typo_ids,
        clean_attention_mask=ones,
        typo_attention_mask=ones,
        output_logit_pairs=pairs,
        global_state_token_pairs=(),
        clean_edit_positions=(),
        typo_edit_positions=(),
        answer_targets=(),
        student_tokens=MAX_SEQUENCE_LENGTH,
        is_noop=pair.is_noop,
    )


__all__ = [
    "ALLOWED_SEEDS",
    "FINEWEB_DATASET",
    "FINEWEB_DATA_FILE",
    "FINEWEB_REVISION",
    "KojimaFaithfulDataBundle",
    "KojimaFaithfulNoiseGenerator",
    "MATCHED_REPLICATION_SEEDS",
    "MISTRAL_MODEL",
    "MISTRAL_REVISION",
    "PACKED_EXAMPLES",
    "PUBLIC_ANCHOR_SEED",
    "TARGET_STUDENT_TOKENS",
    "TARGET_USABLE_EXAMPLES",
    "UnusableKojimaFaithfulPairError",
    "PrepareKojimaFaithfulDataConfig",
    "PrepareKojimaFaithfulDataResult",
    "encode_kojima_faithful_pair",
    "iter_packed_documents",
    "load_kojima_faithful_data_bundle",
    "prepare_kojima_faithful_data",
]

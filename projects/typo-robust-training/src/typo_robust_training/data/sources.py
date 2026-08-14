"""Pinned public-dataset adapters and exact tokenizer accounting."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from typo_robust_training.data.config import DatasetSource, TrainingDataProtocol, strict_loads
from typo_robust_training.data.natural_typos import parse_github_typo_commit
from typo_robust_training.data.records import CleanRecord, NaturalTypoRecord


_DOLMA_REMOTE_SHARDS = 16
_DOLMA_ROWS_PER_SHARD = 256
_REMOTE_TIMEOUT_SECONDS = 60
_DOLMA_DUPLICATE_IDENTITY_POLICY = "drop-exact-text-duplicates-fail-on-conflict/v1"
_DOLMA_BLANK_TEXT_POLICY = "skip-blank-string/v1"


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset field {field} must be a non-empty string")
    return value.strip()


def _choices(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        labels, texts = value.get("label"), value.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return tuple((str(label), str(text)) for label, text in zip(labels, texts, strict=True))
    if isinstance(value, list):
        return tuple((chr(65 + index), str(text)) for index, text in enumerate(value))
    raise ValueError("multiple-choice dataset row has invalid choices")


def _multiple_choice_text(question: str, choices: tuple[tuple[str, str], ...]) -> str:
    return question + "\n" + "\n".join(f"{label}. {text}" for label, text in choices)


def _answer_choice_text(
    choices: tuple[tuple[str, str], ...],
    *,
    answer: str,
) -> str:
    matches = tuple(text for label, text in choices if label == answer)
    if len(matches) != 1:
        raise ValueError("multiple-choice answer must identify exactly one choice")
    return matches[0]


def segment_document(record: CleanRecord, *, character_limit: int) -> CleanRecord:
    if not isinstance(record, CleanRecord) or record.task is not None:
        raise TypeError("document segmentation requires a non-task CleanRecord")
    if (
        isinstance(character_limit, bool)
        or not isinstance(character_limit, int)
        or character_limit <= 0
    ):
        raise ValueError("document character limit must be a positive integer")
    text = record.text
    if len(text) <= character_limit:
        start, end, segment = 0, len(text), text
    else:
        available_starts = len(text) - character_limit + 1
        start = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big") % available_starts
        if start > 0 and not text[start - 1].isspace():
            while start < len(text) and not text[start].isspace():
                start += 1
        while start < len(text) and text[start].isspace():
            start += 1
        end = min(start + character_limit, len(text))
        if end < len(text) and not text[end].isspace():
            while end > start and not text[end - 1].isspace():
                end -= 1
        while end > start and text[end - 1].isspace():
            end -= 1
        segment = text[start:end]
        if not segment:
            raise ValueError(f"document window contains no complete word: {record.source_id}")
    metadata = {
        **dict(record.metadata),
        "original_character_count": len(text),
        "document_window": [start, end],
        "document_character_limit": character_limit,
    }
    return replace(record, text=segment, metadata=metadata)


def _format_huggingface_record(
    source_name: str,
    source: DatasetSource,
    split: str,
    index: int,
    row: Mapping[str, object],
) -> CleanRecord:
    metadata: dict[str, object] = {"dataset_row_index": index}
    if source_name == "fineweb_edu":
        source_id = str(row.get("id") or f"{split}-{index}")
        text = _text(row.get("text"), field="text")
        group_id = str(row.get("id") or row.get("url") or source_id)
        metadata.update(
            {
                key: row[key]
                for key in ("url", "dump", "date", "language", "score", "token_count")
                if key in row and isinstance(row[key], (str, int, float, bool, type(None)))
            }
        )
        answer = None
    elif source_name == "gsm8k":
        source_id = f"{split}-{index}"
        text = _text(row.get("question"), field="question")
        reference_solution = _text(row.get("answer"), field="answer")
        matches = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", reference_solution)
        if len(matches) != 1:
            raise ValueError("GSM8K answer must contain exactly one canonical #### number")
        answer = matches[0].replace(",", "")
        group_id = source_id
        metadata["reference_solution"] = reference_solution
    elif source_name == "mmlu":
        source_id = f"{split}-{index}"
        choices = _choices(row.get("choices"))
        text = _multiple_choice_text(_text(row.get("question"), field="question"), choices)
        raw_answer = row.get("answer")
        if (
            isinstance(raw_answer, bool)
            or not isinstance(raw_answer, int)
            or not 0 <= raw_answer < len(choices)
        ):
            raise ValueError("MMLU answer index is invalid")
        answer = choices[raw_answer][0]
        group_id = source_id
        metadata["subject"] = row.get("subject")
        metadata["answer_choice_text"] = choices[raw_answer][1]
    elif source_name in {"arc", "commonsense_qa"}:
        source_id = str(row.get("id") or f"{split}-{index}")
        choices = _choices(row.get("choices"))
        text = _multiple_choice_text(_text(row.get("question"), field="question"), choices)
        answer = _text(row.get("answerKey"), field="answerKey")
        group_id = source_id
        metadata["answer_choice_text"] = _answer_choice_text(choices, answer=answer)
    elif source_name == "mmlu_pro":
        source_id = str(row.get("question_id") or row.get("id") or f"{split}-{index}")
        choices = _choices(row.get("options") or row.get("choices"))
        text = _multiple_choice_text(_text(row.get("question"), field="question"), choices)
        raw_answer = row.get("answer")
        if isinstance(raw_answer, int) and not isinstance(raw_answer, bool):
            if not 0 <= raw_answer < len(choices):
                raise ValueError("MMLU-Pro answer index is invalid")
            answer = choices[raw_answer][0]
        else:
            answer = _text(raw_answer, field="answer")
        group_id = source_id
        metadata["category"] = row.get("category")
        metadata["answer_choice_text"] = _answer_choice_text(choices, answer=answer)
    elif source_name == "math_500":
        source_id = str(row.get("unique_id") or row.get("id") or f"{split}-{index}")
        text = _text(row.get("problem"), field="problem")
        answer = _text(row.get("answer") or row.get("solution"), field="answer")
        group_id = source_id
        metadata.update(
            {
                key: row[key]
                for key in ("subject", "level")
                if key in row and isinstance(row[key], (str, int, float, bool, type(None)))
            }
        )
    elif source_name == "dolma":
        source_id = str(row.get("id") or f"{split}-{index}")
        text = _text(row.get("text"), field="text")
        group_id = source_id
        answer = None
        metadata["origin_metadata"] = row.get("metadata")
        metadata["dolma_source"] = row.get("source")
    else:
        raise ValueError(f"unsupported Hugging Face source: {source_name}")
    return CleanRecord(
        source=source_name,
        source_revision=source.revision,
        source_split=split,
        source_id=f"{source_name}:{source_id}",
        group_id=f"{source_name}:{group_id}",
        text=text,
        task=source.task,
        answer=answer,
        metadata=metadata,
    )


def _approved_repositories(path: Path) -> Mapping[str, str]:
    if not path.is_file():
        raise ValueError(
            f"GitHub Typo Corpus requires a local approved-repositories JSON file; missing: {path}"
        )
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, Mapping) or any(
        not isinstance(repository, str)
        or not repository
        or not isinstance(license_name, str)
        or not license_name
        for repository, license_name in payload.items()
    ):
        raise ValueError("approved-repositories file must map repository URLs to licenses")
    return {str(repository): str(license_name) for repository, license_name in payload.items()}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_json_lines(path: Path, *, corpus: str) -> Iterable[Mapping[str, object]]:
    opener: Any = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {corpus} JSON at line {line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid {corpus} object at line {line_number}")
            yield payload


class HuggingFaceDataSourceProvider:
    """Load pinned dataset rows, plus a locally licensed natural-typo archive."""

    def __init__(
        self,
        *,
        protocol: TrainingDataProtocol,
        natural_corpus_path: Path | None = None,
        approved_repositories_path: Path | None = None,
        dolma_corpus_path: Path | None = None,
    ) -> None:
        self.protocol = protocol
        data_root = Path(__file__).resolve().parents[3] / "data"
        self.natural_corpus_path = Path(
            natural_corpus_path
            or os.environ.get(
                "TYPO_GITHUB_CORPUS_PATH",
                data_root / "github-typo-corpus.v1.0.0.jsonl.gz",
            )
        ).resolve()
        self.approved_repositories_path = Path(
            approved_repositories_path
            or os.environ.get(
                "TYPO_GITHUB_APPROVED_REPOSITORIES",
                data_root / "github-typo-approved-repositories.json",
            )
        ).resolve()
        configured_dolma = dolma_corpus_path or os.environ.get("TYPO_DOLMA_CORPUS_PATH")
        self.dolma_corpus_path = (
            Path(configured_dolma).resolve() if configured_dolma is not None else None
        )

    def provenance(self) -> Mapping[str, object]:
        dolma_source = self.protocol.sources["dolma"]
        if self.dolma_corpus_path is None:
            dolma_urls, dolma_inventory_sha256 = self._remote_dolma_urls(dolma_source)
        else:
            dolma_urls, dolma_inventory_sha256 = (), None
        return {
            "provider": "pinned-public-datasets/v1",
            "datasets_version": self._datasets_version(),
            "natural_corpus_path": str(self.natural_corpus_path),
            "natural_corpus_sha256": _sha256_file(self.natural_corpus_path),
            "approved_repositories_path": str(self.approved_repositories_path),
            "approved_repositories_sha256": _sha256_file(self.approved_repositories_path),
            "dolma_mode": (
                "local-jsonl" if self.dolma_corpus_path is not None else "pinned-remote-shards"
            ),
            "dolma_corpus_path": (
                str(self.dolma_corpus_path) if self.dolma_corpus_path is not None else None
            ),
            "dolma_corpus_sha256": (
                _sha256_file(self.dolma_corpus_path) if self.dolma_corpus_path is not None else None
            ),
            "dolma_url_inventory_sha256": dolma_inventory_sha256,
            "dolma_selected_urls": list(dolma_urls),
            "dolma_duplicate_identity_policy": _DOLMA_DUPLICATE_IDENTITY_POLICY,
            "dolma_blank_text_policy": _DOLMA_BLANK_TEXT_POLICY,
            "source_revisions": {
                name: source.revision for name, source in self.protocol.sources.items()
            },
        }

    @staticmethod
    def _datasets_version() -> str:
        import datasets

        return datasets.__version__

    def _iter_natural(self, source: DatasetSource) -> Iterable[NaturalTypoRecord]:
        if not self.natural_corpus_path.is_file():
            raise ValueError(
                "GitHub Typo Corpus v1.0.0 archive is unavailable; set "
                "TYPO_GITHUB_CORPUS_PATH to the locally obtained JSONL or JSONL.GZ file"
            )
        approved = _approved_repositories(self.approved_repositories_path)
        for payload in _iter_json_lines(
            self.natural_corpus_path,
            corpus="GitHub Typo Corpus",
        ):
            yield from parse_github_typo_commit(
                payload,
                approved_repositories=approved,
                source_revision=source.revision,
            )

    def _remote_dolma_urls(self, source: DatasetSource) -> tuple[tuple[str, ...], str]:
        from huggingface_hub import hf_hub_download

        if source.subset is None:
            raise ValueError("Dolma source requires a pinned URL-inventory name")
        inventory_path = Path(
            hf_hub_download(
                source.dataset,
                f"urls/{source.subset}.txt",
                repo_type="dataset",
                revision=source.revision,
            )
        )
        raw = inventory_path.read_bytes()
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError("Dolma URL inventory is not UTF-8") from exc
        urls = tuple(line.strip() for line in lines if line.strip())
        if (
            not urls
            or len(set(urls)) != len(urls)
            or any(not url.startswith("https://") for url in urls)
        ):
            raise ValueError("Dolma URL inventory is empty, duplicated, or non-HTTPS")

        by_domain: dict[str, list[str]] = defaultdict(list)
        for url in urls:
            path_parts = tuple(part for part in urlsplit(url).path.split("/") if part)
            domain = path_parts[-2] if len(path_parts) >= 2 else "unknown"
            by_domain[domain].append(url)
        ordered_domains = sorted(by_domain)
        for domain in ordered_domains:
            by_domain[domain].sort(
                key=lambda url: hashlib.sha256(
                    f"dolma-shard/v1\0{self.protocol.seed}\0{url}".encode()
                ).hexdigest()
            )
        selected: list[str] = []
        depth = 0
        limit = min(_DOLMA_REMOTE_SHARDS, len(urls))
        while len(selected) < limit:
            emitted = False
            for domain in ordered_domains:
                if depth < len(by_domain[domain]):
                    selected.append(by_domain[domain][depth])
                    emitted = True
                    if len(selected) == limit:
                        break
            if not emitted:
                break
            depth += 1
        return tuple(selected), hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _remote_dolma_rows(url: str) -> Iterable[Mapping[str, object]]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "typo-robust-training/0.1 (+research reproducibility)"},
        )
        with urllib.request.urlopen(request, timeout=_REMOTE_TIMEOUT_SECONDS) as response:
            with gzip.GzipFile(fileobj=response) as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    if line_number > _DOLMA_ROWS_PER_SHARD:
                        break
                    if not raw_line.strip():
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid remote Dolma JSON at {url}:{line_number}"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise ValueError(f"invalid remote Dolma object at {url}:{line_number}")
                    yield payload

    def _iter_dolma(self, source: DatasetSource) -> Iterable[CleanRecord]:
        if self.dolma_corpus_path is not None and not self.dolma_corpus_path.is_file():
            raise ValueError(
                "TYPO_DOLMA_CORPUS_PATH does not point to a JSONL or JSONL.GZ file: "
                f"{self.dolma_corpus_path}"
            )
        if self.dolma_corpus_path is not None:
            payloads = _iter_json_lines(self.dolma_corpus_path, corpus="Dolma")
        else:
            urls, _ = self._remote_dolma_urls(source)
            payloads = (payload for url in urls for payload in self._remote_dolma_rows(url))
        seen_text_by_record_id: dict[str, str] = {}
        for index, payload in enumerate(payloads):
            payload_text = payload.get("text")
            if isinstance(payload_text, str) and not payload_text.strip():
                continue
            record = _format_huggingface_record("dolma", source, "train", index, payload)
            previous_text = seen_text_by_record_id.get(record.record_id)
            if previous_text is not None:
                if previous_text != record.text:
                    raise ValueError(
                        "Dolma duplicated source identity with conflicting text: "
                        f"{record.source_id}"
                    )
                continue
            seen_text_by_record_id[record.record_id] = record.text
            yield segment_document(
                record,
                character_limit=self.protocol.document_character_window,
            )

    def iter_records(
        self,
        source_name: str,
        source: DatasetSource,
    ) -> Iterable[CleanRecord | NaturalTypoRecord]:
        if source_name == "github_typo_corpus":
            yield from self._iter_natural(source)
            return
        if source_name == "dolma":
            yield from self._iter_dolma(source)
            return
        from datasets import load_dataset

        for split in source.splits:
            dataset = load_dataset(
                source.dataset,
                name=source.subset,
                split=split,
                revision=source.revision,
                streaming=source.streaming,
            )
            if source.streaming and hasattr(dataset, "shuffle"):
                dataset = dataset.shuffle(
                    seed=self.protocol.seed,
                    buffer_size=self.protocol.shuffle_buffer_size,
                )
            for index, row in enumerate(dataset):
                if not isinstance(row, Mapping):
                    raise ValueError(f"dataset {source_name}/{split} emitted a non-object row")
                record = _format_huggingface_record(source_name, source, split, index, row)
                if source_name == "fineweb_edu":
                    record = segment_document(
                        record,
                        character_limit=self.protocol.document_character_window,
                    )
                yield record


class HuggingFaceTokenCounter:
    """Count exact non-padding tokens with the pinned student tokenizer."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        max_sequence_length: int,
        tokenizer: object | None = None,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer = tokenizer
        self.model = model
        self.revision = revision
        self.max_sequence_length = max_sequence_length

    def prepare_record(self, record: CleanRecord) -> CleanRecord:
        if not isinstance(record, CleanRecord):
            raise TypeError("token preparation requires a CleanRecord")
        if record.task is not None:
            return record
        encoded = self.tokenizer(
            record.text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded.get("input_ids")
        offsets = encoded.get("offset_mapping")
        if not isinstance(input_ids, list) or not isinstance(offsets, list):
            raise ValueError("tokenizer must return input_ids and exact offset_mapping")
        if not input_ids or len(input_ids) != len(offsets):
            raise ValueError("tokenizer returned incomplete document offsets")
        try:
            retained_end = max(int(offset[1]) for offset in offsets)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("tokenizer returned malformed document offsets") from exc
        if retained_end <= 0 or retained_end > len(record.text):
            raise ValueError("tokenizer document offsets are out of bounds")
        if (
            retained_end < len(record.text)
            and not record.text[retained_end - 1].isspace()
            and not record.text[retained_end].isspace()
        ):
            while retained_end > 0 and not record.text[retained_end - 1].isspace():
                retained_end -= 1
        prepared_text = record.text[:retained_end].rstrip()
        if not prepared_text:
            raise ValueError(f"token preparation retained no complete word: {record.source_id}")
        token_count = self(prepared_text)
        metadata = {
            **dict(record.metadata),
            "pretokenized_character_count": len(record.text),
            "prepared_character_count": len(prepared_text),
            "prepared_token_count": token_count,
            "tokenizer_model": self.model,
            "tokenizer_revision": self.revision,
        }
        return replace(record, text=prepared_text, metadata=metadata)

    def __call__(self, text: str) -> int:
        token_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_sequence_length,
        )["input_ids"]
        return len(token_ids)


__all__ = ["HuggingFaceDataSourceProvider", "HuggingFaceTokenCounter", "segment_document"]

"""Pinned public-source adapter and local-corpus provenance contracts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from typo_robust_training.data import sources as sources_module
from typo_robust_training.data.config import load_training_data_config
from typo_robust_training.data.records import CleanRecord
from typo_robust_training.data.sources import (
    HuggingFaceDataSourceProvider,
    HuggingFaceTokenCounter,
    _format_huggingface_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-sanity.yaml"


def test_observed_public_dataset_schemas_format_without_losing_answers() -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    rows = {
        "fineweb_edu": {
            "id": "doc-1",
            "text": "An educational document with enough useful words.",
            "url": "https://example.test/doc",
            "dump": "CC-MAIN",
            "language": "en",
            "score": 4.0,
            "token_count": 9,
        },
        "gsm8k": {"question": "What is 1 + 1?", "answer": "#### 2"},
        "mmlu": {
            "question": "Choose one.",
            "choices": ["zero", "one", "two", "three"],
            "answer": 2,
            "subject": "fixture",
        },
        "arc": {
            "id": "arc-1",
            "question": "Which answer?",
            "choices": {"label": ["A", "B"], "text": ["first", "second"]},
            "answerKey": "B",
        },
        "mmlu_pro": {
            "question_id": 7,
            "question": "Choose one.",
            "options": ["zero", "one"],
            "answer": "B",
            "answer_index": 1,
            "category": "fixture",
        },
        "math_500": {
            "unique_id": "math-1",
            "problem": "Compute 1+1.",
            "answer": "2",
            "solution": "It is 2.",
            "subject": "algebra",
            "level": 1,
        },
        "commonsense_qa": {
            "id": "csqa-1",
            "question": "Where?",
            "choices": {"label": ["A", "B"], "text": ["here", "there"]},
            "answerKey": "A",
        },
    }
    expected_answers = {
        "fineweb_edu": None,
        "gsm8k": "2",
        "mmlu": "C",
        "arc": "B",
        "mmlu_pro": "B",
        "math_500": "2",
        "commonsense_qa": "A",
    }
    for source_name, row in rows.items():
        record = _format_huggingface_record(
            source_name,
            protocol.sources[source_name],
            protocol.sources[source_name].splits[0],
            0,
            row,
        )
        assert record.answer == expected_answers[source_name]
        assert record.source_revision == protocol.sources[source_name].revision
        if source_name in {"mmlu", "arc", "mmlu_pro", "commonsense_qa"}:
            expected_choice_text = {
                "mmlu": "two",
                "arc": "second",
                "mmlu_pro": "one",
                "commonsense_qa": "here",
            }[source_name]
            assert record.metadata["answer_choice_text"] == expected_choice_text
    gsm8k = _format_huggingface_record(
        "gsm8k",
        protocol.sources["gsm8k"],
        "train",
        0,
        {"question": "What is 1 + 1?", "answer": "Add them.\n#### 2"},
    )
    assert gsm8k.answer == "2"
    assert gsm8k.metadata["reference_solution"] == "Add them.\n#### 2"


def test_provider_passes_pinned_revision_and_declared_split_to_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    calls: list[dict[str, object]] = []

    def fake_load_dataset(dataset: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append({"dataset": dataset, **kwargs})
        return [{"question": "What is 1 + 1?", "answer": "#### 2"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    provider = HuggingFaceDataSourceProvider(protocol=protocol)
    records = tuple(provider.iter_records("gsm8k", protocol.sources["gsm8k"]))

    assert len(records) == len(protocol.sources["gsm8k"].splits)
    assert [call["split"] for call in calls] == list(protocol.sources["gsm8k"].splits)
    assert all(call["revision"] == protocol.sources["gsm8k"].revision for call in calls)
    assert all(call["name"] == "main" for call in calls)


def test_dolma_local_cache_is_hashed(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "dolma-1",
                    "text": "A held-out domain document with sufficient words.",
                    "source": "pes2o",
                    "metadata": {"fixture": True},
                }
            )
            + "\n"
        )
    provider = HuggingFaceDataSourceProvider(
        protocol=protocol,
        dolma_corpus_path=archive,
    )
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))
    assert len(records) == 1
    assert records[0].source == "dolma"
    assert records[0].text.startswith("A held-out domain")
    assert (
        provider.provenance()["dolma_corpus_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert (
        provider.provenance()["dolma_duplicate_identity_policy"]
        == "drop-normalized-text-duplicates-fail-on-conflict/v1"
    )
    assert provider.provenance()["dolma_blank_text_policy"] == "skip-blank-string/v1"
    assert (
        provider.provenance()["dolma_unsegmentable_text_policy"]
        == "skip-unsegmentable-document/v1"
    )
    assert (
        provider.provenance()["dolma_document_duplicate_policy"]
        == "drop-identical-normalized-documents-first-wins/v1"
    )
    assert (
        provider.provenance()["dolma_usable_text_policy"]
        == "clean-corpus-segmentation-only/v1"
    )
    assert (
        provider.provenance()["document_segmentation_policy"]
        == "maximal-complete-word-window-hash-tiebreak/v4"
    )


def test_dolma_exact_duplicate_identity_is_not_emitted_twice(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    duplicate = {
        "id": "duplicated-upstream-id",
        "text": "The same upstream document appears twice in the pinned sample.",
        "source": "gutenberg",
        "metadata": {"fixture": True},
    }
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")
        handle.write(json.dumps({**duplicate, "metadata": {"fixture": "duplicate"}}) + "\n")

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert len(records) == 1
    assert records[0].source_id == "dolma:duplicated-upstream-id"
    assert records[0].metadata["dataset_row_index"] == 0
    assert records[0].metadata["origin_metadata"] == {"fixture": True}


def test_dolma_duplicate_identity_with_conflicting_text_fails_closed(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for text in ("First document text.", "Conflicting document text."):
            handle.write(
                json.dumps(
                    {
                        "id": "conflicting-upstream-id",
                        "text": text,
                        "source": "gutenberg",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    with pytest.raises(ValueError, match="duplicated source identity with conflicting text"):
        tuple(provider.iter_records("dolma", protocol.sources["dolma"]))


def test_dolma_duplicate_identity_compares_emitted_normalized_text(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for text in ("Alpha held-out document.", "Alpha held-out document.\n\n   "):
            handle.write(
                json.dumps(
                    {
                        "id": "same-upstream-id",
                        "text": text,
                        "source": "gutenberg",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert len(records) == 1
    assert records[0].source_id == "dolma:same-upstream-id"
    assert records[0].text == "Alpha held-out document."


def test_dolma_blank_text_rows_are_not_emitted(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for source_id, text in (
            ("blank", "  \n"),
            ("usable", "A non-empty held-out document remains available."),
        ):
            handle.write(
                json.dumps(
                    {
                        "id": source_id,
                        "text": text,
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert [record.source_id for record in records] == ["dolma:usable"]


def test_dolma_unsegmentable_rows_are_not_emitted(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for source_id, text in (
            ("blob", "A" * (protocol.document_character_window * 2)),
            ("usable", "A non-empty held-out document remains available."),
        ):
            handle.write(
                json.dumps(
                    {
                        "id": source_id,
                        "text": text,
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert [record.source_id for record in records] == ["dolma:usable"]


def test_dolma_word_rich_documents_survive_unlucky_hashed_windows(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    limit = protocol.document_character_window
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for index in range(20):
            handle.write(
                json.dumps(
                        {
                            "id": f"word-rich-{index}",
                            "text": (f"alpha{index} beta{index} gamma{index} " * 1_000)
                            + f"document-{index} "
                            + chr(65 + index) * (limit * 2),
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert len(records) == 20
    assert all(
        record.metadata["document_window_strategy"]
        == "maximal-complete-word-hash-tiebreak"
        for record in records
    )
    assert all(len(record.text) > limit * 0.9 for record in records)
    assert len({record.text for record in records}) == len(records)


@pytest.mark.parametrize(
    "flat_prefix",
    (
        "1234 5678 9012 3456 ",
        "Menu Menu Menu Menu ",
    ),
)
def test_dolma_clean_documents_survive_hashed_flat_regions(
    tmp_path: Path,
    flat_prefix: str,
) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    limit = protocol.document_character_window
    rich = " ".join(
        f"paragraph{index} contains varied prose about robustness research"
        for index in range(120)
    )
    text = flat_prefix * (limit * 20 // len(flat_prefix)) + rich
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "flat-prefix-with-prose",
                    "text": text,
                    "source": "common-crawl",
                    "metadata": {},
                }
            )
            + "\n"
        )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert [record.source_id for record in records] == ["dolma:flat-prefix-with-prose"]


def test_dolma_clean_corpus_does_not_require_typo_targets(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for source_id, text in (
            ("one-word", "a"),
            ("two-letter-words", "ab ba ab ba"),
        ):
            handle.write(
                json.dumps(
                    {
                        "id": source_id,
                        "text": text,
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert [record.source_id for record in records] == [
        "dolma:one-word",
        "dolma:two-letter-words",
    ]


def test_dolma_preserves_distinct_documents_with_identical_emitted_segments(
    tmp_path: Path,
) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    limit = protocol.document_character_window
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(
                json.dumps(
                    {
                        "id": f"boilerplate-{index}",
                        "text": "Copyright notice all rights reserved "
                        + chr(65 + index) * (limit * 2),
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert len(records) == 5
    assert [record.source_id for record in records] == [
        f"dolma:boilerplate-{index}" for index in range(5)
    ]
    assert {record.text for record in records} == {"Copyright notice all rights reserved"}


def test_dolma_prefers_full_prose_over_nearby_single_token(tmp_path: Path) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    limit = protocol.document_character_window
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for index in range(20):
            prose = " ".join(
                f"paragraph{index}-{paragraph} contains varied prose about robustness"
                for paragraph in range(300)
            )
            handle.write(
                json.dumps(
                    {
                        "id": f"prose-rich-{index}",
                        "text": prose
                        + " "
                        + "A" * (3 * limit)
                        + "\n1\n"
                        + "B" * (50 * limit),
                        "source": "common-crawl",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert len(records) == 20
    assert all(len(record.text) > limit * 0.9 for record in records)
    assert all("varied prose" in record.text for record in records)


def test_dolma_deduplicates_identical_full_documents_across_source_ids(
    tmp_path: Path,
) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    archive = tmp_path / "dolma.jsonl.gz"
    text = "A held-out document appears verbatim under two upstream identities."
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        for source_id in ("upstream-a", "upstream-b"):
            handle.write(
                json.dumps(
                    {
                        "id": source_id,
                        "text": text,
                        "source": "gutenberg",
                        "metadata": {},
                    }
                )
                + "\n"
            )

    provider = HuggingFaceDataSourceProvider(protocol=protocol, dolma_corpus_path=archive)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))

    assert [record.source_id for record in records] == ["dolma:upstream-a"]


def test_dolma_default_streams_selected_shards_from_pinned_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_training_data_config(DEFAULT_CONFIG)
    monkeypatch.delenv("TYPO_DOLMA_CORPUS_PATH", raising=False)
    provider = HuggingFaceDataSourceProvider(protocol=protocol)
    url = "https://olmo-data.example/dolma-v1_5-sample/wiki-0000.json.gz"
    monkeypatch.setattr(
        provider,
        "_remote_dolma_urls",
        lambda source: ((url,), "c" * 64),
    )
    compressed = gzip.compress(
        (
            json.dumps(
                {
                    "id": "dolma-remote-1",
                    "text": "A remote held-out domain document with enough useful words.",
                    "source": "wiki",
                    "metadata": {"fixture": True},
                }
            )
            + "\n"
        ).encode()
    )

    class _Response(io.BytesIO):
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: object, *, timeout: int) -> _Response:
        assert getattr(request, "full_url") == url
        assert "typo-robust-training" in request.get_header("User-agent")
        assert timeout == 60
        return _Response(compressed)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    records = tuple(provider.iter_records("dolma", protocol.sources["dolma"]))
    assert [record.source_id for record in records] == ["dolma:dolma-remote-1"]
    provenance = provider.provenance()
    assert provenance["dolma_url_inventory_sha256"] == "c" * 64
    assert provenance["dolma_selected_urls"] == [url]


def test_long_document_window_is_content_stable_and_records_boundaries() -> None:
    segment_document = getattr(sources_module, "segment_document", None)
    assert callable(segment_document)
    text = " ".join(f"educational-{index}" for index in range(2_000))

    def record(source_id: str) -> CleanRecord:
        return CleanRecord(
            source="fineweb_edu",
            source_revision="a" * 40,
            source_split="train",
            source_id=source_id,
            group_id=source_id,
            text=text,
            task=None,
            answer=None,
            metadata={},
        )

    first = segment_document(record("first"), character_limit=8_192)
    second = segment_document(record("second"), character_limit=8_192)
    assert first.text == second.text
    assert len(first.text) <= 8_192
    assert first.metadata["document_window"] == second.metadata["document_window"]
    assert first.metadata["original_character_count"] == len(text)


def test_token_preparation_keeps_only_complete_words_within_model_limit() -> None:
    class _Tokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
            max_length: int,
            return_offsets_mapping: bool = False,
        ) -> dict[str, object]:
            import re

            spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
            offsets = ([(0, 0)] if add_special_tokens else []) + spans
            if truncation:
                offsets = offsets[:max_length]
            result: dict[str, object] = {"input_ids": list(range(len(offsets)))}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            return result

    original = CleanRecord(
        source="fineweb_edu",
        source_revision="a" * 40,
        source_split="train",
        source_id="long",
        group_id="long",
        text="alpha beta gamma delta epsilon zeta eta",
        task=None,
        answer=None,
        metadata={},
    )
    counter = HuggingFaceTokenCounter(
        model="fixture/model",
        revision="b" * 40,
        max_sequence_length=5,
        tokenizer=_Tokenizer(),
    )
    prepared = counter.prepare_record(original)
    assert prepared.text == "alpha beta gamma delta"
    assert prepared.metadata["prepared_token_count"] == 5
    assert counter(prepared.text) == 5

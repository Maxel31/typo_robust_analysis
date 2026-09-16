from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors

from typo_cot.experiments.rebuttal_v2.artifacts import ArtifactSnapshots
from typo_cot.experiments.rebuttal_v2.input_tokenization import (
    OFFSET_CONVENTION,
    TOKENIZER_LOCK_SCHEMA,
    load_tokenizer_lock,
    reconstruct_token_alignment,
)
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError, canonical_sha256, sha256_text


MODEL_ID = "fixture/local-model"
REVISION = "a" * 40


def _tokenizer() -> Tokenizer:
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "Q",
                ":",
                "long",
                "##word",
                "##x",
                "short",
                "caf",
                "##é",
                "!",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(
        [AddedToken("[CLS]", special=True), AddedToken("[SEP]", special=True)]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tokenizer


def _write_lock(tmp_path: Path, tokenizer: Tokenizer | None = None) -> Path:
    tokenizer = _tokenizer() if tokenizer is None else tokenizer
    raw = tokenizer.to_str().encode("utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(raw)
    serialized = json.loads(raw)
    entry = {
        "tokenizer_id": "fixture/local-tokenizer",
        "revision": REVISION,
        "tokenizer_files": {
            "tokenizer.json": {"path": "tokenizer.json", "sha256": hashlib.sha256(raw).hexdigest()}
        },
        "library": {"name": "tokenizers", "version": "0.22.2"},
        "special_token_ids": {
            row["content"]: row["id"] for row in serialized["added_tokens"] if row["special"]
        },
        "chat_template": None,
        "add_special_tokens": True,
        "normalization": serialized["normalizer"],
        "padding_side": "left",
        "offset_convention": OFFSET_CONVENTION,
    }
    path = tmp_path / "tokenizer_lock.json"
    path.write_text(
        json.dumps({"schema_version": TOKENIZER_LOCK_SCHEMA, "models": {MODEL_ID: entry}})
    )
    return path


def _edit_lock(path: Path, change) -> None:
    payload = json.loads(path.read_text())
    change(payload["models"][MODEL_ID])
    path.write_text(json.dumps(payload))


def _align(path: Path, prompt: str = "longword", **kwargs):
    lock = load_tokenizer_lock(path)
    return reconstruct_token_alignment(
        prompt,
        lock.entries[MODEL_ID],
        kwargs.pop("spans", [[0, len(prompt)]]),
        query_text=kwargs.pop("query_text", prompt),
        query_span=kwargs.pop("query_span", [0, len(prompt)]),
        **kwargs,
    )


def test_complete_prompt_uses_explicit_final_query_not_first_substring(tmp_path):
    path = _write_lock(tmp_path)
    prompt = "Q: longword\nQ:  longword!"
    start = len("Q: longword\nQ:  ")
    audit = _align(
        path, prompt, query_text="longword!", query_span=[start, len(prompt)], spans=[[0, 8]]
    )
    assert audit.eligibility == "valid"
    assert audit.prompt_token_ids == [1, 3, 4, 5, 6, 3, 4, 5, 6, 11, 2]
    assert audit.word_endpoints == [8]
    assert audit.word_alignments == [
        {
            "query_span": [0, 8],
            "full_prompt_span": [start, start + 8],
            "token_indices": [7, 8],
            "token_span": [7, 9],
            "endpoint": 8,
        }
    ]
    assert audit.prompt_token_ids_sha256 == canonical_sha256(audit.prompt_token_ids)
    assert audit.special_tokens_mask[0] == audit.special_tokens_mask[-1] == 1
    assert audit.offset_mapping[0] == audit.offset_mapping[-1] == [0, 0]


def test_variable_subtoken_counts_are_retained_and_last_token_is_endpoint(tmp_path):
    path = _write_lock(tmp_path)
    clean = _align(path, "longword")
    typo = _align(path, "longxword")
    assert clean.eligibility == typo.eligibility == "valid"
    assert clean.word_alignments[0]["token_indices"] == [1, 2]
    assert typo.word_alignments[0]["token_indices"] == [1, 2, 3]
    assert clean.word_endpoints == [2]
    assert typo.word_endpoints == [3]


def test_unicode_offsets_are_original_codepoints_not_bytes(tmp_path):
    path = _write_lock(tmp_path)
    audit = _align(path, "  café!", query_text="café!", query_span=[2, 7], spans=[[0, 4]])
    assert audit.eligibility == "valid"
    assert audit.offset_mapping == [[0, 0], [2, 5], [5, 6], [6, 7], [0, 0]]
    assert audit.word_endpoints == [2]


def test_byte_fallback_duplicate_offsets_keep_all_subtokens(tmp_path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(vocab={value: index for index, value in enumerate(alphabet)}, merges=[])
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    path = _write_lock(tmp_path, tokenizer)
    audit = _align(path, "é")
    assert audit.eligibility == "valid"
    assert audit.offset_mapping == [[0, 1], [0, 1]]
    assert audit.word_alignments[0]["token_indices"] == [0, 1]
    assert audit.word_endpoints == [1]


def test_normalization_is_pinned_and_original_offsets_survive_casefolding(tmp_path):
    tokenizer = _tokenizer()
    tokenizer.normalizer = normalizers.Lowercase()
    path = _write_lock(tmp_path, tokenizer)
    audit = _align(path, "LONGWORD")
    assert audit.eligibility == "valid"
    assert audit.prompt_token_ids == [1, 5, 6, 2]
    assert audit.offset_mapping[1:3] == [[0, 4], [4, 8]]


def test_special_insertion_policy_is_used_and_chat_template_not_rendered(tmp_path):
    path = _write_lock(tmp_path)
    _edit_lock(
        path,
        lambda entry: entry.update(
            add_special_tokens=False,
            chat_template={
                "text": "do not render {{ message }}",
                "sha256": sha256_text("do not render {{ message }}"),
            },
        ),
    )
    audit = _align(path)
    assert audit.eligibility == "valid"
    assert audit.prompt_token_ids == [5, 6]
    assert audit.word_endpoints == [1]


@pytest.mark.parametrize("revision,expected", [(None, None), (REVISION, True), ("b" * 40, False)])
def test_archived_agreement_is_independent_of_alignment(tmp_path, revision, expected):
    path = _write_lock(tmp_path)
    audit = _align(path, archived_tokenizer_revision=revision)
    assert audit.eligibility == "valid"
    assert audit.archived_revision_match is expected
    assert audit.matches_archived_tokenizer is (False if expected is False else None)


@pytest.mark.parametrize(
    "known,changes,expected",
    [
        ([], {}, None),
        (["revision"], {}, None),
        (["id"], {}, None),
        (["sha256"], {}, None),
        (["revision", "id"], {}, None),
        (["revision", "id", "sha256"], {}, True),
        (["id"], {"id": "different/tokenizer"}, False),
        (["sha256"], {"sha256": "0" * 64}, False),
        (["revision", "id", "sha256"], {"revision": "b" * 40}, False),
    ],
)
def test_full_archived_tokenizer_agreement_requires_all_identity_evidence(
    tmp_path, known, changes, expected
):
    path = _write_lock(tmp_path)
    values = {
        "revision": REVISION,
        "id": "fixture/local-tokenizer",
        "sha256": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
    }
    values.update(changes)
    arguments = {f"archived_tokenizer_{name}": values[name] for name in known}
    audit = _align(path, **arguments)
    assert audit.eligibility == "valid"
    assert audit.matches_archived_tokenizer is expected


@pytest.mark.parametrize("field", ["revision", "id", "sha256"])
@pytest.mark.parametrize(
    "unavailable",
    [
        pytest.param(None, id="absent"),
        pytest.param({}, id="dict"),
        pytest.param([], id="list"),
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param("", id="empty"),
        pytest.param(" \t\n", id="whitespace"),
    ],
)
def test_invalid_or_blank_archived_identity_is_unknown(tmp_path, field, unavailable):
    path = _write_lock(tmp_path)
    values = {
        "revision": REVISION,
        "id": "fixture/local-tokenizer",
        "sha256": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
    }
    values[field] = unavailable
    audit = _align(
        path,
        **{f"archived_tokenizer_{name}": value for name, value in values.items()},
    )
    assert audit.eligibility == "valid"
    assert audit.matches_archived_tokenizer is None
    assert audit.archived_revision_match is (None if field == "revision" else True)


@pytest.mark.parametrize(
    "unknown_field,mismatch_field",
    [("revision", "id"), ("id", "sha256"), ("sha256", "revision")],
)
def test_known_archived_mismatch_dominates_unknown_identity(
    tmp_path, unknown_field, mismatch_field
):
    path = _write_lock(tmp_path)
    values = {
        "revision": REVISION,
        "id": "fixture/local-tokenizer",
        "sha256": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
    }
    values[unknown_field] = {}
    values[mismatch_field] = "known-mismatch"
    audit = _align(
        path,
        **{f"archived_tokenizer_{name}": value for name, value in values.items()},
    )
    assert audit.matches_archived_tokenizer is False
    expected_revision_match = None if unknown_field == "revision" else mismatch_field != "revision"
    assert audit.archived_revision_match is expected_revision_match


def test_nonblank_archived_identity_is_compared_without_normalization(tmp_path):
    path = _write_lock(tmp_path)
    audit = _align(
        path,
        archived_tokenizer_revision=REVISION,
        archived_tokenizer_id=" fixture/local-tokenizer ",
        archived_tokenizer_sha256=hashlib.sha256(
            (tmp_path / "tokenizer.json").read_bytes()
        ).hexdigest(),
    )
    assert audit.matches_archived_tokenizer is False
    assert audit.archived_revision_match is True


def test_missing_model_entry_keeps_unknown_not_whole_audit_error(tmp_path):
    path = _write_lock(tmp_path)
    lock = load_tokenizer_lock(path)
    audit = reconstruct_token_alignment(
        "longword", lock.entries.get("absent"), [[0, 8]], query_text="longword", query_span=[0, 8]
    )
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["tokenizer_lock_entry_missing"]
    assert audit.prompt_token_ids is None


@pytest.mark.parametrize("query_span", [None, [0, 8]])
def test_never_infers_query_location_from_a_matching_substring(tmp_path, query_span):
    path = _write_lock(tmp_path)
    audit = _align(
        path, "Q: longword", query_text="longword", query_span=query_span, spans=[[0, 8]]
    )
    assert audit.eligibility == ("unknown" if query_span is None else "invalid")
    assert audit.prompt_token_ids is not None
    assert audit.word_endpoints == []


def test_unknown_spans_still_capture_complete_encoding(tmp_path):
    path = _write_lock(tmp_path)
    audit = _align(path, spans=[])
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["changed_word_spans_unavailable"]
    assert audit.prompt_token_ids == [1, 5, 6, 2]


def test_missing_prompt_is_unknown_without_encoding(tmp_path):
    entry = load_tokenizer_lock(_write_lock(tmp_path)).entries[MODEL_ID]
    audit = reconstruct_token_alignment(None, entry, [], query_text=None, query_span=None)
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["exact_full_prompt_unavailable"]
    assert audit.prompt_token_ids is None


@pytest.mark.parametrize("span", [[0, 99], [-1, 8], [True, 8], [8, 2], [0]])
def test_malformed_word_span_is_rejected(tmp_path, span):
    with pytest.raises(IntakeError):
        _align(_write_lock(tmp_path), spans=[span])


def test_duplicate_word_spans_are_not_silently_deduplicated(tmp_path):
    with pytest.raises(IntakeError, match="deduplicated"):
        _align(_write_lock(tmp_path), spans=[[0, 8], [0, 8]])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("revision", "main", "immutable"),
        ("revision", "a" * 39, "immutable"),
        ("revision", "A" * 40, "immutable"),
        ("offset_convention", "utf8-byte", "must be one of"),
        ("add_special_tokens", 1, "boolean"),
        ("padding_side", "middle", "must be one of"),
        ("normalization", {"type": "Lowercase"}, "normalization disagrees"),
        ("special_token_ids", {}, "omits"),
        ("special_token_ids", {"bos": 1, "eos": 2, "invalid": 100}, "absent"),
        ("chat_template", {"text": "hello", "sha256": "0" * 64}, "SHA-256 mismatch"),
    ],
)
def test_inconsistent_lock_is_fatal(tmp_path, field, value, message):
    path = _write_lock(tmp_path)
    _edit_lock(path, lambda entry: entry.update({field: value}))
    with pytest.raises(IntakeError, match=message):
        load_tokenizer_lock(path)


def test_named_special_roles_can_share_a_special_id_or_be_null(tmp_path):
    path = _write_lock(tmp_path)
    _edit_lock(
        path,
        lambda entry: entry.update(
            special_token_ids={"bos": 1, "cls": 1, "eos": 2, "sep": 2, "pad": None}
        ),
    )
    lock = load_tokenizer_lock(path)
    assert lock.entries[MODEL_ID].descriptor["special_token_ids"] == {
        "bos": 1,
        "cls": 1,
        "eos": 2,
        "sep": 2,
        "pad": None,
    }
    assert _align(path).eligibility == "valid"


@pytest.mark.parametrize("ordinary_added_token", [False, True])
@pytest.mark.parametrize("library_version", ["0.22.2", "not-installed"])
def test_special_roles_cannot_declare_an_ordinary_vocabulary_id(
    tmp_path, ordinary_added_token, library_version
):
    tokenizer = _tokenizer()
    if ordinary_added_token:
        tokenizer.add_tokens([AddedToken("<ordinary>", special=False)])
        ordinary_id = tokenizer.token_to_id("<ordinary>")
    else:
        ordinary_id = tokenizer.token_to_id("long")
    path = _write_lock(tmp_path, tokenizer)

    def declare_ordinary_special(entry):
        entry["special_token_ids"]["pad"] = ordinary_id
        entry["library"]["version"] = library_version

    _edit_lock(path, declare_ordinary_special)
    with pytest.raises(IntakeError, match="not marked special"):
        load_tokenizer_lock(path)


@pytest.mark.parametrize("mode", ["padding", "truncation"])
def test_complete_prompt_lock_cannot_enable_padding_or_truncation(tmp_path, mode):
    tokenizer = _tokenizer()
    if mode == "padding":
        tokenizer.enable_padding(length=128)
    else:
        tokenizer.enable_truncation(max_length=4)
    with pytest.raises(IntakeError, match="disable padding and truncation"):
        load_tokenizer_lock(_write_lock(tmp_path, tokenizer))


def test_library_version_mismatch_is_unknown_but_bad_ids_remain_fatal(tmp_path):
    path = _write_lock(tmp_path)
    _edit_lock(path, lambda entry: entry["library"].update(version="not-installed"))
    audit = _align(path)
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["tokenizer_library_version_mismatch"]
    _edit_lock(path, lambda entry: entry["special_token_ids"].update(bad=999))
    with pytest.raises(IntakeError, match="absent"):
        load_tokenizer_lock(path)


def test_unavailable_tokenizer_library_preserves_unknown_audit(tmp_path, monkeypatch):
    path = _write_lock(tmp_path)

    def unavailable(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", unavailable)
    audit = _align(path)
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["tokenizer_library_unavailable"]


def test_local_cpu_lock_loading_does_not_import_model_frameworks(tmp_path):
    path = _write_lock(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from typo_cot.experiments.rebuttal_v2.input_tokenization import load_tokenizer_lock; "
            "load_tokenizer_lock(sys.argv[1]); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_hash_mismatch_and_previous_snapshot_changes_are_fatal(tmp_path):
    path = _write_lock(tmp_path)
    with pytest.raises(IntakeError, match="previous capture"):
        load_tokenizer_lock(path, snapshots={path.resolve(): "0" * 64})
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(IntakeError, match="SHA-256 mismatch"):
        load_tokenizer_lock(path)


def test_encoding_uses_captured_verified_bytes_not_a_reopened_tokenizer(tmp_path, monkeypatch):
    path = _write_lock(tmp_path)
    original = ArtifactSnapshots.reference

    def replace_after_capture(self, reference, containing_file):
        result = original(self, reference, containing_file)
        result[0].write_text("{}")
        return result

    monkeypatch.setattr(ArtifactSnapshots, "reference", replace_after_capture)
    audit = _align(path)
    assert audit.eligibility == "valid"
    assert audit.prompt_token_ids == [1, 5, 6, 2]
    assert (tmp_path / "tokenizer.json").read_text() == "{}"


@pytest.mark.parametrize(
    "offsets,masks,reason",
    [
        ([[0, 4], [3, 2]], [0, 0], "token_offset_out_of_bounds"),
        ([[4, 8], [0, 4]], [0, 0], "nonmonotone_token_offsets"),
        ([[0, 4], [5, 8]], [0, 0], "changed_word_offset_coverage_gap"),
        ([[0, 0], [0, 8]], [0, 0], "zero_width_non_special_token"),
        ([[0, 4], [0, 0], [4, 8]], [0, 1, 0], "noncontiguous_changed_word_tokens"),
    ],
)
def test_unstable_offset_mapping_is_unknown_not_valid(tmp_path, offsets, masks, reason):
    lock = load_tokenizer_lock(_write_lock(tmp_path))
    encoded = SimpleNamespace(
        ids=list(range(len(offsets))), offsets=offsets, special_tokens_mask=masks
    )
    entry = replace(
        lock.entries[MODEL_ID], _backend=SimpleNamespace(encode=lambda *args, **kwargs: encoded)
    )
    audit = reconstruct_token_alignment(
        "longword", entry, [[0, 8]], query_text="longword", query_span=[0, 8]
    )
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == [reason]
    assert audit.prompt_token_ids is not None


def test_token_crossing_into_another_word_is_unknown(tmp_path):
    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    path = _write_lock(tmp_path, tokenizer)
    audit = _align(path, "longword short", spans=[[0, 8]])
    assert audit.eligibility == "unknown"
    assert audit.reason_codes == ["token_crosses_word_boundary"]


def test_padding_side_is_metadata_but_does_not_add_padding_tokens(tmp_path):
    path = _write_lock(tmp_path)
    left = _align(path)
    _edit_lock(path, lambda entry: entry.update(padding_side="right"))
    right = _align(path)
    assert left.prompt_token_ids == right.prompt_token_ids == [1, 5, 6, 2]


def test_lock_snapshot_contains_every_consumed_file_and_preserves_inputs(tmp_path):
    path = _write_lock(tmp_path)
    snapshots = {tmp_path / "other.json": "b" * 64}
    lock = load_tokenizer_lock(path, snapshots=snapshots)
    assert snapshots == {tmp_path / "other.json": "b" * 64}
    assert set(lock.snapshots) == {path, tmp_path / "tokenizer.json", tmp_path / "other.json"}
    assert lock.reference == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_strict_json_unknown_fields_and_duplicate_keys_are_rejected(tmp_path):
    path = _write_lock(tmp_path)
    _edit_lock(path, lambda entry: entry.update(typo_setting=True))
    with pytest.raises(IntakeError, match="unknown fields"):
        load_tokenizer_lock(path)
    path.write_text('{"schema_version":"rebuttal-tokenizer-lock/v2","models":{},"models":{}}')
    with pytest.raises(IntakeError, match="duplicate"):
        load_tokenizer_lock(path)

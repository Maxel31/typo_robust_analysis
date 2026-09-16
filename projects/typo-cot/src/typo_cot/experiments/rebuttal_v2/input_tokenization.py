"""Pinned, local-only tokenization of exact complete archived prompts.

No model, Transformers wrapper, network lookup, chat rendering, or standalone
query encoding participates in this audit.  Missing provenance is an unknown
audit; a contradictory or corrupt lock is an invalid input artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
import re
from typing import Any
import unicodedata

from .artifacts import ArtifactSnapshots
from .schemas import (
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_bool,
    require_enum,
    require_int,
    require_keys,
    require_object,
    require_sha256,
    require_string,
    sha256_text,
    strict_loads,
)


TOKENIZER_LOCK_SCHEMA = "rebuttal-tokenizer-lock/v2"
ALIGNMENT_ALGORITHM_VERSION = "full-prompt-word-overlap/v2.0.0"
OFFSET_CONVENTION = "unicode-codepoint-half-open"
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ENTRY_FIELDS = {
    "tokenizer_id",
    "revision",
    "tokenizer_files",
    "library",
    "special_token_ids",
    "chat_template",
    "add_special_tokens",
    "normalization",
    "padding_side",
    "offset_convention",
}


@dataclass(frozen=True)
class LockedTokenizer:
    descriptor: dict[str, Any]
    reference: dict[str, str]
    unavailable_reason: str | None
    _backend: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class TokenizerLock:
    path: Path
    payload: dict[str, Any]
    entries: dict[str, LockedTokenizer]
    snapshots: dict[Path, str]

    @property
    def reference(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}


@dataclass(frozen=True)
class AlignmentAudit:
    eligibility: str
    reason_codes: list[str]
    explanation: str
    prompt_token_ids: list[int] | None
    prompt_token_ids_sha256: str | None
    offset_mapping: list[list[int]] | None
    special_tokens_mask: list[int] | None
    word_alignments: list[dict[str, Any]]
    word_endpoints: list[int]
    matches_archived_tokenizer: bool | None
    archived_revision_match: bool | None


def _json_object(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise IntakeError(f"tokenizer artifact is not UTF-8: {path}") from exc
    return require_object(strict_loads(text, str(path)), str(path))


def _load_entry(
    model_id: str, entry: object, lock_path: Path, captured: ArtifactSnapshots
) -> LockedTokenizer:
    context = f"tokenizer lock model {model_id}"
    entry = require_object(entry, context)
    require_keys(entry, _ENTRY_FIELDS, context=context)
    require_string(entry["tokenizer_id"], f"{context}.tokenizer_id")
    revision = require_string(entry["revision"], f"{context}.revision")
    if _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise IntakeError(f"{context}.revision must be a lowercase 40-hex immutable HF commit")
    require_bool(entry["add_special_tokens"], f"{context}.add_special_tokens")
    require_enum(entry["padding_side"], {"left", "right"}, f"{context}.padding_side")
    require_enum(entry["offset_convention"], {OFFSET_CONVENTION}, f"{context}.offset_convention")
    library = require_object(entry["library"], f"{context}.library")
    require_keys(library, {"name", "version"}, context=f"{context}.library")
    require_enum(library["name"], {"tokenizers"}, f"{context}.library.name")
    version = require_string(library["version"], f"{context}.library.version")
    chat = entry["chat_template"]
    if chat is not None:
        chat = require_object(chat, f"{context}.chat_template")
        require_keys(chat, {"text", "sha256"}, context=f"{context}.chat_template")
        require_string(chat["text"], f"{context}.chat_template.text", allow_empty=True)
        require_sha256(chat["sha256"], f"{context}.chat_template.sha256")
        if sha256_text(chat["text"]) != chat["sha256"]:
            raise IntakeError(f"{context}.chat_template SHA-256 mismatch")
    files = require_object(entry["tokenizer_files"], f"{context}.tokenizer_files")
    require_keys(files, {"tokenizer.json"}, context=f"{context}.tokenizer_files")
    tokenizer_path, tokenizer_bytes = captured.reference(files["tokenizer.json"], lock_path)
    serialized = _json_object(tokenizer_bytes, tokenizer_path)
    for name in ("normalizer", "padding", "truncation", "added_tokens"):
        if name not in serialized:
            raise IntakeError(f"tokenizer.json is missing {name}: {tokenizer_path}")
    if canonical_json(serialized["normalizer"]) != canonical_json(entry["normalization"]):
        raise IntakeError(f"{context}.normalization disagrees with tokenizer.json")
    if serialized["padding"] is not None or serialized["truncation"] is not None:
        raise IntakeError("tokenizer.json must disable padding and truncation for complete prompts")
    special_ids = require_object(entry["special_token_ids"], f"{context}.special_token_ids")
    declared_ids = set()
    for name, token_id in special_ids.items():
        require_string(name, f"{context}.special_token_ids key")
        if token_id is not None:
            declared_ids.add(require_int(token_id, f"special token {name}", minimum=0))
    added = serialized["added_tokens"]
    if not isinstance(added, list):
        raise IntakeError("tokenizer.json added_tokens must be an array")
    actual_special_ids = set()
    added_ids = set()
    for token in added:
        token = require_object(token, "tokenizer.json added token")
        token_id = require_int(token.get("id"), "added token id", minimum=0)
        added_ids.add(token_id)
        if require_bool(token.get("special"), "added token special"):
            actual_special_ids.add(token_id)
    if not actual_special_ids.issubset(declared_ids):
        raise IntakeError(f"{context}.special_token_ids omits tokenizer.json special token IDs")
    model = require_object(serialized.get("model"), "tokenizer.json model")
    vocabulary = model.get("vocab")
    if isinstance(vocabulary, dict):
        vocabulary_ids = {
            require_int(value, "tokenizer vocabulary ID", minimum=0)
            for value in vocabulary.values()
        }
    elif isinstance(vocabulary, list):
        # Unigram vocabularies have implicit IDs equal to array positions.
        vocabulary_ids = set(range(len(vocabulary)))
    else:
        raise IntakeError("tokenizer.json model must contain a vocabulary")
    if not declared_ids.issubset(vocabulary_ids | added_ids):
        raise IntakeError(f"{context}.special_token_ids contains IDs absent from vocabulary")
    if not declared_ids.issubset(actual_special_ids):
        raise IntakeError(
            f"{context}.special_token_ids contains IDs not marked special in tokenizer.json"
        )
    backend = None
    unavailable = None
    try:
        installed_version = metadata.version("tokenizers")
    except metadata.PackageNotFoundError:
        unavailable = "tokenizer_library_unavailable"
    else:
        if installed_version != version:
            unavailable = "tokenizer_library_version_mismatch"
        else:
            from tokenizers import Tokenizer

            try:
                # The bytes being parsed are exactly the bytes checked above.
                backend = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
            except Exception as exc:
                raise IntakeError(f"cannot reconstruct locked tokenizer {model_id}: {exc}") from exc
            vocab_ids = set(backend.get_vocab(with_added_tokens=True).values())
            if not declared_ids.issubset(vocab_ids):
                raise IntakeError(
                    f"{context}.special_token_ids contains IDs absent from vocabulary"
                )
    return LockedTokenizer(
        descriptor=entry,
        reference={"path": str(tokenizer_path), "sha256": captured.snapshots[tokenizer_path]},
        unavailable_reason=unavailable,
        _backend=backend,
    )


def load_tokenizer_lock(
    path: str | Path, snapshots: dict[Path, str] | None = None
) -> TokenizerLock:
    """Load a complete local lock, optionally binding already-captured input hashes.

    ``snapshots`` is not mutated.  The returned hashes include the lock and every
    referenced tokenizer file, and must be revalidated before publication.
    """

    path = Path(path).resolve()
    captured = ArtifactSnapshots()
    payload = _json_object(captured.capture(path), path)
    require_keys(payload, {"schema_version", "models"}, context="tokenizer lock")
    require_enum(
        payload["schema_version"], {TOKENIZER_LOCK_SCHEMA}, "tokenizer lock schema_version"
    )
    models = require_object(payload["models"], "tokenizer lock models")
    entries = {}
    for model_id, entry in sorted(models.items()):
        require_string(model_id, "tokenizer lock model ID")
        entries[model_id] = _load_entry(model_id, entry, path, captured)
    combined = {Path(p).resolve(): digest for p, digest in (snapshots or {}).items()}
    for input_path, digest in captured.snapshots.items():
        if input_path in combined and combined[input_path] != digest:
            raise IntakeError(f"input changed since previous capture: {input_path}")
        combined[input_path] = digest
    return TokenizerLock(path, payload, entries, combined)


def _span(value: object, length: int, context: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise IntakeError(f"{context} must be a [start,end] array")
    start = require_int(value[0], f"{context} start", minimum=0)
    end = require_int(value[1], f"{context} end", minimum=0)
    if not start <= end <= length:
        raise IntakeError(f"{context} lies outside its text")
    return [start, end]


def reconstruct_token_alignment(
    prompt: str | None,
    entry: LockedTokenizer | None,
    spans: list[list[int]],
    *,
    query_text: str | None,
    query_span: list[int] | None,
    archived_tokenizer_revision: object = None,
    archived_tokenizer_id: object = None,
    archived_tokenizer_sha256: object = None,
) -> AlignmentAudit:
    """Map all changed-word subtokens in one exact full prompt.

    Query and word coordinates are supplied explicitly; repeated strings never
    select an occurrence.  Equal offsets from byte-level Unicode subtokens are
    valid, but nonmonotone, uncovered, or colliding mappings remain unknown.
    Historical identity agreement requires the explicit tokenizer ID, revision,
    and tokenizer.json hash. It does not certify historical wrapper/runtime
    settings such as special-token insertion. These comparisons never control
    independent audit validity.
    """

    matches = None
    revision_match = None
    if entry is not None:
        comparisons = []
        for observed, expected in (
            (archived_tokenizer_revision, entry.descriptor["revision"]),
            (archived_tokenizer_id, entry.descriptor["tokenizer_id"]),
            (archived_tokenizer_sha256, entry.reference["sha256"]),
        ):
            comparisons.append(
                None
                if not isinstance(observed, str) or not observed.strip()
                else observed == expected
            )
        revision_match = comparisons[0]
        if False in comparisons:
            matches = False
        elif all(value is True for value in comparisons):
            matches = True
    ids = offsets = special_mask = None
    mappings: list[dict[str, Any]] = []

    def result(eligibility: str, reasons: list[str], explanation: str) -> AlignmentAudit:
        return AlignmentAudit(
            eligibility,
            reasons,
            explanation,
            ids,
            canonical_sha256(ids) if ids is not None else None,
            offsets,
            special_mask,
            mappings,
            [item["endpoint"] for item in mappings],
            matches,
            revision_match,
        )

    missing = []
    if entry is None:
        missing.append("tokenizer_lock_entry_missing")
    elif entry.unavailable_reason is not None:
        missing.append(entry.unavailable_reason)
    if prompt is None:
        missing.append("exact_full_prompt_unavailable")
    if missing:
        return result(
            "unknown", missing, "The pinned tokenizer or exact complete prompt is unavailable."
        )
    require_string(prompt, "exact full prompt", allow_empty=True)
    try:
        encoded = entry._backend.encode(
            prompt, add_special_tokens=entry.descriptor["add_special_tokens"]
        )
    except Exception as exc:
        return result(
            "unknown", ["tokenizer_encoding_failed"], f"Pinned local encoding failed: {exc}"
        )
    ids = list(encoded.ids)
    offsets = [list(item) for item in encoded.offsets]
    special_mask = list(encoded.special_tokens_mask)
    if len(ids) != len(offsets) or len(ids) != len(special_mask):
        return result(
            "unknown",
            ["tokenizer_offset_length_mismatch"],
            "Offset and token array lengths differ.",
        )
    previous = [0, 0]
    usable: list[tuple[int, int, int]] = []
    for index, (token_id, offset, special) in enumerate(
        zip(ids, offsets, special_mask, strict=True)
    ):
        require_int(token_id, "encoded token ID", minimum=0)
        if special not in (0, 1):
            return result(
                "unknown", ["invalid_special_token_mask"], "A special-token mask value is invalid."
            )
        try:
            start, end = _span(offset, len(prompt), "token offset")
        except IntakeError:
            return result(
                "unknown",
                ["token_offset_out_of_bounds"],
                "A token offset is outside the exact prompt.",
            )
        if start == end:
            if special:
                continue
            return result(
                "unknown",
                ["zero_width_non_special_token"],
                "An ordinary token has no source character span.",
            )
        if start < previous[0] or end < previous[1]:
            return result(
                "unknown",
                ["nonmonotone_token_offsets"],
                "Token offsets do not preserve source order.",
            )
        previous = [start, end]
        usable.append((index, start, end))
    if query_text is None or query_span is None:
        return result(
            "unknown",
            ["explicit_query_span_unavailable"],
            "Exact query text and its explicit full-prompt span are required.",
        )
    require_string(query_text, "query text", allow_empty=True)
    try:
        query_start, query_end = _span(query_span, len(prompt), "query span")
    except IntakeError:
        return result(
            "invalid",
            ["query_span_out_of_bounds"],
            "The declared query span is outside the exact prompt.",
        )
    if prompt[query_start:query_end] != query_text:
        return result(
            "invalid",
            ["query_span_text_mismatch"],
            "The declared query span does not contain the exact query text.",
        )
    if not isinstance(spans, list):
        raise IntakeError("changed-word spans must be an array")
    word_spans = [_span(span, len(query_text), "changed-word span") for span in spans]
    if not word_spans:
        return result(
            "unknown",
            ["changed_word_spans_unavailable"],
            "No independently aligned changed-word spans are available.",
        )
    if len({tuple(span) for span in word_spans}) != len(word_spans):
        raise IntakeError("changed-word spans must be explicitly deduplicated")
    for word_start, word_end in word_spans:
        start, end = query_start + word_start, query_start + word_end
        if start == end:
            return result(
                "unknown",
                ["empty_changed_word_span"],
                "A changed word has no source characters on this side.",
            )
        overlaps = [(index, s, e) for index, s, e in usable if s < end and start < e]
        if not overlaps:
            return result(
                "unknown",
                ["changed_word_has_no_tokens"],
                "No ordinary token overlaps a changed word.",
            )
        for _, s, e in overlaps:
            outside = prompt[s:start] + prompt[end:e]
            if any(unicodedata.category(char)[0] in {"L", "N", "M"} for char in outside):
                return result(
                    "unknown",
                    ["token_crosses_word_boundary"],
                    "An overlapping token also contains characters from another word.",
                )
        covered_until = start
        for _, s, e in overlaps:
            if s > covered_until:
                break
            covered_until = max(covered_until, e)
        if covered_until < end:
            return result(
                "unknown",
                ["changed_word_offset_coverage_gap"],
                "Token offsets do not cover every character of a changed word.",
            )
        indices = [item[0] for item in overlaps]
        if indices != list(range(indices[0], indices[-1] + 1)):
            return result(
                "unknown",
                ["noncontiguous_changed_word_tokens"],
                "Changed-word subtokens are not contiguous in the full prompt.",
            )
        mappings.append(
            {
                "query_span": [word_start, word_end],
                "full_prompt_span": [start, end],
                "token_indices": indices,
                "token_span": [indices[0], indices[-1] + 1],
                "endpoint": indices[-1],
            }
        )
    if len({item["endpoint"] for item in mappings}) != len(mappings):
        return result(
            "unknown",
            ["changed_word_endpoint_collision"],
            "Distinct changed words share a token endpoint.",
        )
    return result(
        "valid",
        [],
        "All changed words map to complete full-prompt subtoken spans and final-token endpoints.",
    )

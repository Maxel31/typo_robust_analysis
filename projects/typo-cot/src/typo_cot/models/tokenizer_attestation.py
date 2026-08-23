"""Fail-closed Hugging Face tokenizer revision attestation.

``transformers`` does not guarantee that the resolved Hub commit remains in a
tokenizer's ``init_kwargs``.  A requested revision therefore is not evidence
that the object returned by ``from_pretrained`` came from that revision.  This
module resolves the immutable snapshot, inventories its tokenizer assets, and
compares the loaded tokenizer with a second tokenizer loaded from that exact
local snapshot before making the commit observable to downstream consumers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils.hub import cached_file, extract_commit_hash

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# This fixed inventory covers the files used by the supported SentencePiece,
# BPE, and fast-tokenizer families.  Absence is recorded explicitly, so the
# attestation captures both file presence and content without relying on a
# possibly incomplete directory listing of the local Hub cache.
TOKENIZER_ASSET_FILENAMES: tuple[str, ...] = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)

_BEHAVIOR_FLAGS: tuple[str, ...] = (
    "add_bos_token",
    "add_eos_token",
    "add_prefix_space",
    "byte_fallback",
    "clean_up_tokenization_spaces",
    "do_basic_tokenize",
    "do_lower_case",
    "keep_accents",
    "legacy",
    "model_max_length",
    "padding_side",
    "spaces_between_special_tokens",
    "split_special_tokens",
    "strip_accents",
    "tokenize_chinese_chars",
    "truncation_side",
    "use_default_system_prompt",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    """Convert tokenizer metadata into deterministic JSON-compatible data."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]

    # AddedToken is intentionally handled structurally.  Its repr is not a
    # stable serialization contract and may omit behavior-changing flags.
    if hasattr(value, "content"):
        fields = {
            "content": getattr(value, "content"),
            "single_word": getattr(value, "single_word", False),
            "lstrip": getattr(value, "lstrip", False),
            "rstrip": getattr(value, "rstrip", False),
            "normalized": getattr(value, "normalized", True),
            "special": getattr(value, "special", False),
        }
        return {"type": type(value).__name__, **_canonical_value(fields)}
    raise TypeError(f"unsupported tokenizer attestation value: {type(value).__name__}")


def tokenizer_fingerprint_payload(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Return the canonical, behavior-relevant tokenizer object payload."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("tokenizer backend canonical serialization is not observable")
    try:
        backend_payload = json.loads(backend.to_str())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("tokenizer backend serialization is not valid JSON") from exc

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        raise ValueError("tokenizer added vocabulary is not observable")

    special_map = getattr(tokenizer, "special_tokens_map_extended", None)
    if not isinstance(special_map, Mapping):
        special_map = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special_map, Mapping):
        raise ValueError("tokenizer special token map is not observable")

    special_token_ids: dict[str, Any] = {}
    for name in sorted(str(key) for key in special_map):
        id_name = (
            "additional_special_tokens_ids"
            if name == "additional_special_tokens"
            else f"{name}_id"
        )
        special_token_ids[id_name] = getattr(tokenizer, id_name, None)

    flags: dict[str, Any] = {}
    for name in _BEHAVIOR_FLAGS:
        if hasattr(tokenizer, name):
            flags[name] = getattr(tokenizer, name)

    return {
        "schema": "typo-cot-tokenizer-object-fingerprint/v1",
        "class": {
            "module": type(tokenizer).__module__,
            "qualname": type(tokenizer).__qualname__,
        },
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "backend": _canonical_value(backend_payload),
        "added_vocab": _canonical_value(get_added_vocab()),
        "special_tokens_map": _canonical_value(special_map),
        "special_token_ids": _canonical_value(special_token_ids),
        "all_special_tokens": _canonical_value(
            getattr(tokenizer, "all_special_tokens", None)
        ),
        "all_special_ids": _canonical_value(getattr(tokenizer, "all_special_ids", None)),
        "chat_template": _canonical_value(getattr(tokenizer, "chat_template", None)),
        "behavior_flags": _canonical_value(flags),
    }


def tokenizer_fingerprint_sha256(tokenizer: PreTrainedTokenizerBase) -> str:
    """Hash the canonical tokenizer object payload."""

    return hashlib.sha256(_canonical_json_bytes(tokenizer_fingerprint_payload(tokenizer))).hexdigest()


@dataclass(frozen=True)
class TokenizerAssetAttestation:
    """Presence and content digest of one exact-snapshot asset."""

    filename: str
    present: bool
    sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "present": self.present,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TokenizerSnapshotAttestation:
    """Portable attestation for a loaded tokenizer and immutable Hub snapshot."""

    model_name: str
    requested_revision: str
    observed_commit: str
    assets: tuple[TokenizerAssetAttestation, ...]
    tokenizer_fingerprint_sha256: str

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "typo-cot-tokenizer-snapshot-attestation/v1",
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "observed_commit": self.observed_commit,
            "assets": [asset.as_dict() for asset in self.assets],
            "tokenizer_fingerprint_sha256": self.tokenizer_fingerprint_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._payload())).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "attestation_sha256": self.sha256}


@dataclass(frozen=True)
class _ResolvedTokenizerSnapshot:
    model_name: str
    requested_revision: str
    observed_commit: str
    snapshot_dir: Path
    assets: tuple[TokenizerAssetAttestation, ...]


def _snapshot_root(resolved_file: str, observed_commit: str) -> Path:
    path = Path(resolved_file)
    for candidate in path.parents:
        if candidate.name == observed_commit and candidate.parent.name == "snapshots":
            return candidate
    raise ValueError("resolved tokenizer asset is not inside the observed Hub snapshot")


def _verified_asset_sha256(
    resolved_file: str,
    *,
    snapshot_dir: Path,
    filename: str,
) -> str:
    """Verify snapshot link, cache blob identity, and return the content SHA256."""

    resolved_path = Path(resolved_file)
    if _snapshot_root(resolved_file, snapshot_dir.name) != snapshot_dir:
        raise ValueError(f"tokenizer asset {filename} is outside the exact snapshot")
    expected_path = snapshot_dir / filename
    if not expected_path.is_symlink():
        raise ValueError(f"tokenizer asset {filename} is not an auditable snapshot symlink")
    if not resolved_path.exists() or not os.path.samefile(resolved_path, expected_path):
        raise ValueError(f"tokenizer asset {filename} does not match its exact-snapshot path")

    try:
        blob_path = expected_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"tokenizer asset {filename} has a broken cache-blob link") from exc
    expected_blob_dir = (snapshot_dir.parent.parent / "blobs").resolve(strict=True)
    if blob_path.parent != expected_blob_dir:
        raise ValueError(f"tokenizer asset {filename} points outside the model cache blobs")

    blob_name = blob_path.name
    sha256 = _sha256_file(blob_path)
    if re.fullmatch(r"[0-9a-f]{64}", blob_name):
        if sha256 != blob_name:
            raise ValueError(f"tokenizer asset {filename} fails SHA256 blob integrity")
    elif re.fullmatch(r"[0-9a-f]{40}", blob_name):
        if _git_blob_sha1(blob_path) != blob_name:
            raise ValueError(f"tokenizer asset {filename} fails Git-blob SHA1 integrity")
    else:
        raise ValueError(f"tokenizer asset {filename} has an unrecognized cache-blob identity")
    return sha256


def _resolve_tokenizer_snapshot(
    model_name: str,
    requested_revision: str,
) -> _ResolvedTokenizerSnapshot:
    """Resolve a revision and inventory tokenizer assets at its exact commit."""

    anchor = cached_file(
        model_name,
        "tokenizer_config.json",
        revision=requested_revision,
    )
    if anchor is None:  # pragma: no cover - cached_file normally raises first
        raise ValueError("tokenizer_config.json could not be resolved")
    observed_commit = extract_commit_hash(anchor, None)
    if not isinstance(observed_commit, str) or _COMMIT_RE.fullmatch(observed_commit) is None:
        raise ValueError("tokenizer snapshot commit is not observable as an exact 40-hex revision")
    if _COMMIT_RE.fullmatch(requested_revision) and requested_revision != observed_commit:
        raise ValueError("resolved tokenizer snapshot differs from requested exact revision")
    snapshot_dir = _snapshot_root(anchor, observed_commit)
    repo_files = frozenset(HfApi().list_repo_files(model_name, revision=observed_commit))
    if "tokenizer_config.json" not in repo_files:
        raise ValueError("resolved snapshot inventory omits tokenizer_config.json")

    assets: list[TokenizerAssetAttestation] = []
    for filename in TOKENIZER_ASSET_FILENAMES:
        if filename not in repo_files:
            assets.append(TokenizerAssetAttestation(filename, False, None))
            continue
        resolved = cached_file(
            model_name,
            filename,
            revision=observed_commit,
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved is None:
            raise ValueError(f"snapshot tokenizer asset is present but not locally cached: {filename}")
        asset_commit = extract_commit_hash(resolved, None)
        if asset_commit != observed_commit:
            raise ValueError(f"tokenizer asset {filename} resolved from a different commit")
        sha256 = _verified_asset_sha256(
            resolved,
            snapshot_dir=snapshot_dir,
            filename=filename,
        )
        assets.append(TokenizerAssetAttestation(filename, True, sha256))

    return _ResolvedTokenizerSnapshot(
        model_name=model_name,
        requested_revision=requested_revision,
        observed_commit=observed_commit,
        snapshot_dir=snapshot_dir,
        assets=tuple(assets),
    )


def _retained_commit(tokenizer: PreTrainedTokenizerBase) -> str | None:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if init_kwargs is None:
        return None
    if not isinstance(init_kwargs, dict):
        raise ValueError("tokenizer init_kwargs are not mutable metadata")
    value = init_kwargs.get("_commit_hash")
    if value is None:
        return None
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("loaded tokenizer retained an invalid commit revision")
    return value


def load_attested_tokenizer(
    model_name: str,
    requested_revision: str,
    *,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedTokenizerBase, TokenizerSnapshotAttestation]:
    """Load and attest a tokenizer before exposing its resolved commit.

    The first object is loaded through the public repository identifier at the
    resolved immutable commit.  The reference object is loaded independently
    from the exact local snapshot directory.  Only an exact object fingerprint
    match permits ``_commit_hash`` to be written to the returned object.
    """

    resolved = _resolve_tokenizer_snapshot(model_name, requested_revision)
    actual = AutoTokenizer.from_pretrained(
        model_name,
        revision=resolved.observed_commit,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    retained = _retained_commit(actual)
    if retained is not None and retained != resolved.observed_commit:
        raise ValueError("loaded tokenizer retained a different commit revision")

    reference = AutoTokenizer.from_pretrained(
        str(resolved.snapshot_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    actual_fingerprint = tokenizer_fingerprint_sha256(actual)
    reference_fingerprint = tokenizer_fingerprint_sha256(reference)
    if actual_fingerprint != reference_fingerprint:
        raise ValueError(
            "loaded tokenizer does not match the independently loaded exact-snapshot reference"
        )

    init_kwargs = getattr(actual, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        raise ValueError("tokenizer init_kwargs are not mutable metadata")
    # This is deliberately the observed commit, never the caller's request.
    init_kwargs["_commit_hash"] = resolved.observed_commit
    attestation = TokenizerSnapshotAttestation(
        model_name=model_name,
        requested_revision=requested_revision,
        observed_commit=resolved.observed_commit,
        assets=resolved.assets,
        tokenizer_fingerprint_sha256=actual_fingerprint,
    )
    return actual, attestation

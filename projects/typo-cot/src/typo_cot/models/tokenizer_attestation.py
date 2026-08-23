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
import importlib.metadata
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKENIZER_ATTESTATION_MANIFEST_ENV = "TYPO_COT_TOKENIZER_ATTESTATION_MANIFEST"

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
    "vocab.txt",
    "chat_template.jinja",
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
    transformers_version: str
    tokenizers_version: str
    source_manifest_sha256: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "typo-cot-tokenizer-snapshot-attestation/v1",
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "observed_commit": self.observed_commit,
            "assets": [asset.as_dict() for asset in self.assets],
            "tokenizer_fingerprint_sha256": self.tokenizer_fingerprint_sha256,
            "transformers_version": self.transformers_version,
            "tokenizers_version": self.tokenizers_version,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._payload())).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "attestation_sha256": self.sha256}

    def provenance_dict(self) -> dict[str, Any]:
        """Return exact attestation details plus the frozen manifest file digest."""

        if self.source_manifest_sha256 is None:
            raise ValueError("tokenizer attestation is not bound to a frozen manifest")
        return {
            **self.as_dict(),
            "manifest_file_sha256": self.source_manifest_sha256,
        }


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
    *,
    expected: TokenizerSnapshotAttestation | None = None,
) -> _ResolvedTokenizerSnapshot:
    """Resolve a revision and inventory tokenizer assets at its exact commit."""

    if expected is not None:
        if expected.model_name != model_name or expected.requested_revision != requested_revision:
            raise ValueError("frozen tokenizer manifest request identity differs")
        observed_commit = expected.observed_commit
        anchor = cached_file(
            model_name,
            "tokenizer_config.json",
            revision=observed_commit,
            local_files_only=True,
        )
    else:
        anchor = cached_file(
            model_name,
            "tokenizer_config.json",
            revision=requested_revision,
        )
    if anchor is None:  # pragma: no cover - cached_file normally raises first
        raise ValueError("tokenizer_config.json could not be resolved")
    extracted_commit = extract_commit_hash(anchor, None)
    if expected is None:
        observed_commit = extracted_commit
    elif extracted_commit != observed_commit:
        raise ValueError("locally resolved tokenizer anchor differs from frozen manifest")
    if not isinstance(observed_commit, str) or _COMMIT_RE.fullmatch(observed_commit) is None:
        raise ValueError("tokenizer snapshot commit is not observable as an exact 40-hex revision")
    if _COMMIT_RE.fullmatch(requested_revision) and requested_revision != observed_commit:
        raise ValueError("resolved tokenizer snapshot differs from requested exact revision")
    snapshot_dir = _snapshot_root(anchor, observed_commit)
    if expected is None:
        repo_files = frozenset(HfApi().list_repo_files(model_name, revision=observed_commit))
        if "tokenizer_config.json" not in repo_files:
            raise ValueError("resolved snapshot inventory omits tokenizer_config.json")
        expected_assets: dict[str, TokenizerAssetAttestation] | None = None
    else:
        expected_assets = {asset.filename: asset for asset in expected.assets}
        if tuple(expected_assets) != TOKENIZER_ASSET_FILENAMES:
            raise ValueError("frozen tokenizer manifest asset inventory differs")
        repo_files = frozenset(
            filename for filename, asset in expected_assets.items() if asset.present
        )

    assets: list[TokenizerAssetAttestation] = []
    for filename in TOKENIZER_ASSET_FILENAMES:
        if filename not in repo_files:
            # A locally injected file can be consumed by AutoTokenizer even if
            # the authoritative inventory says it does not exist.
            if os.path.lexists(snapshot_dir / filename):
                raise ValueError(
                    f"tokenizer asset {filename} exists locally but is absent from inventory"
                )
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
        if expected_assets is not None and sha256 != expected_assets[filename].sha256:
            raise ValueError(f"tokenizer asset {filename} differs from frozen manifest")
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


def _normalize_padding_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Apply the wrapper's historical pad-token fallback before fingerprinting."""

    if getattr(tokenizer, "pad_token", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("tokenizer has neither a pad token nor an EOS fallback")
    tokenizer.pad_token = eos_token


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate tokenizer manifest key: {key}")
        output[key] = value
    return output


def _attestation_from_payload(
    payload: object,
    *,
    source_manifest_sha256: str,
) -> TokenizerSnapshotAttestation:
    """Validate exact manifest data and bind it to its source-file digest."""

    if not isinstance(payload, dict):
        raise ValueError("frozen tokenizer attestation manifest must be an object")
    expected_keys = {
        "schema",
        "model_name",
        "requested_revision",
        "observed_commit",
        "assets",
        "tokenizer_fingerprint_sha256",
        "transformers_version",
        "tokenizers_version",
        "attestation_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("frozen tokenizer attestation manifest fields differ")
    if payload.get("schema") != "typo-cot-tokenizer-snapshot-attestation/v1":
        raise ValueError("frozen tokenizer attestation manifest schema differs")
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != len(TOKENIZER_ASSET_FILENAMES):
        raise ValueError("frozen tokenizer attestation asset inventory differs")
    assets: list[TokenizerAssetAttestation] = []
    for filename, raw_asset in zip(TOKENIZER_ASSET_FILENAMES, raw_assets, strict=True):
        if not isinstance(raw_asset, dict) or set(raw_asset) != {"filename", "present", "sha256"}:
            raise ValueError("frozen tokenizer attestation asset entry differs")
        present = raw_asset.get("present")
        sha256 = raw_asset.get("sha256")
        if raw_asset.get("filename") != filename or not isinstance(present, bool):
            raise ValueError("frozen tokenizer attestation asset identity differs")
        if present:
            if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
                raise ValueError("frozen tokenizer attestation asset SHA256 differs")
        elif sha256 is not None:
            raise ValueError("absent frozen tokenizer asset must not have a SHA256")
        assets.append(TokenizerAssetAttestation(filename, present, sha256))
    tokenizer_config = next(
        asset for asset in assets if asset.filename == "tokenizer_config.json"
    )
    if not tokenizer_config.present:
        raise ValueError("frozen tokenizer attestation omits tokenizer_config.json")

    string_fields = (
        "model_name",
        "requested_revision",
        "observed_commit",
        "tokenizer_fingerprint_sha256",
        "transformers_version",
        "tokenizers_version",
        "attestation_sha256",
    )
    if any(not isinstance(payload.get(field), str) or not payload[field] for field in string_fields):
        raise ValueError("frozen tokenizer attestation string field differs")
    if _COMMIT_RE.fullmatch(payload["observed_commit"]) is None or _SHA256_RE.fullmatch(
        payload["tokenizer_fingerprint_sha256"]
    ) is None:
        raise ValueError("frozen tokenizer attestation digest or revision differs")
    attestation = TokenizerSnapshotAttestation(
        model_name=payload["model_name"],
        requested_revision=payload["requested_revision"],
        observed_commit=payload["observed_commit"],
        assets=tuple(assets),
        tokenizer_fingerprint_sha256=payload["tokenizer_fingerprint_sha256"],
        transformers_version=payload["transformers_version"],
        tokenizers_version=payload["tokenizers_version"],
        source_manifest_sha256=source_manifest_sha256,
    )
    if payload["attestation_sha256"] != attestation.sha256:
        raise ValueError("frozen tokenizer attestation aggregate SHA256 differs")
    return attestation


def _attestation_from_manifest(path: Path) -> TokenizerSnapshotAttestation:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("frozen tokenizer attestation manifest is not readable") from exc
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("frozen tokenizer attestation manifest is invalid JSON") from exc
    return _attestation_from_payload(
        payload,
        source_manifest_sha256=manifest_sha256,
    )


def load_tokenizer_attestation_manifest(path: str | os.PathLike[str]) -> TokenizerSnapshotAttestation:
    """Load one frozen audit manifest without contacting the Hub."""

    return _attestation_from_manifest(Path(path))


def frozen_tokenizer_attestation_from_environment() -> TokenizerSnapshotAttestation | None:
    """Read and validate the infrastructure manifest named by the environment."""

    raw_path = os.environ.get(TOKENIZER_ATTESTATION_MANIFEST_ENV)
    if raw_path is None:
        return None
    if not raw_path:
        raise ValueError(f"{TOKENIZER_ATTESTATION_MANIFEST_ENV} must not be empty")
    return load_tokenizer_attestation_manifest(raw_path)


def tokenizer_attestation_manifest_bytes(attestation: TokenizerSnapshotAttestation) -> bytes:
    """Serialize an audit manifest deterministically for pre-run freezing."""

    return json.dumps(
        attestation.as_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"


def validate_tokenizer_attestation_provenance(
    value: object,
    *,
    expected_model: str,
    expected_revision: str,
) -> TokenizerSnapshotAttestation:
    """Validate the exact attestation embedded in a scientific artifact."""

    if not isinstance(value, Mapping):
        raise ValueError("tokenizer attestation provenance must be an object")
    payload = dict(value)
    manifest_sha256 = payload.pop("manifest_file_sha256", None)
    if not isinstance(manifest_sha256, str) or _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("tokenizer attestation manifest-file SHA256 differs")
    attestation = _attestation_from_payload(
        payload,
        source_manifest_sha256=manifest_sha256,
    )
    if (
        attestation.model_name != expected_model
        or attestation.requested_revision != expected_revision
        or attestation.observed_commit != expected_revision
        or attestation.provenance_dict() != dict(value)
    ):
        raise ValueError("tokenizer attestation provenance identity differs")
    return attestation


def require_frozen_tokenizer_attestation(
    wrapper: object,
    *,
    expected_model: str,
    expected_revision: str,
) -> TokenizerSnapshotAttestation:
    """Require a wrapper attestation that was checked against a frozen manifest."""

    attestation = getattr(wrapper, "tokenizer_snapshot_attestation", None)
    if not isinstance(attestation, TokenizerSnapshotAttestation):
        raise ValueError("loaded tokenizer has no snapshot attestation")
    expected = frozen_tokenizer_attestation_from_environment()
    if expected is None:
        raise ValueError(
            f"scientific runtime requires {TOKENIZER_ATTESTATION_MANIFEST_ENV}"
        )
    if attestation.source_manifest_sha256 is None:
        raise ValueError("loaded tokenizer attestation is not frozen-manifest-backed")
    if (
        attestation.model_name != expected_model
        or attestation.requested_revision != expected_revision
        or attestation.observed_commit != expected_revision
        or _SHA256_RE.fullmatch(attestation.source_manifest_sha256) is None
    ):
        raise ValueError("loaded tokenizer frozen attestation identity differs")
    if attestation.provenance_dict() != expected.provenance_dict():
        raise ValueError("loaded tokenizer attestation differs from frozen manifest")
    return attestation


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
    if trust_remote_code:
        raise ValueError("attested tokenizer loads forbid trust_remote_code=True")
    expected = frozen_tokenizer_attestation_from_environment()
    if expected is not None and (
        expected.transformers_version != importlib.metadata.version("transformers")
        or expected.tokenizers_version != importlib.metadata.version("tokenizers")
    ):
        raise ValueError("tokenizer library versions differ from frozen manifest")
    resolved = _resolve_tokenizer_snapshot(
        model_name,
        requested_revision,
        expected=expected,
    )
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
    _normalize_padding_token(actual)
    _normalize_padding_token(reference)
    actual_fingerprint = tokenizer_fingerprint_sha256(actual)
    reference_fingerprint = tokenizer_fingerprint_sha256(reference)
    if actual_fingerprint != reference_fingerprint:
        raise ValueError(
            "loaded tokenizer does not match the independently loaded exact-snapshot reference"
        )
    if expected is not None and actual_fingerprint != expected.tokenizer_fingerprint_sha256:
        raise ValueError("loaded tokenizer fingerprint differs from frozen manifest")

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
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
        source_manifest_sha256=(None if expected is None else expected.source_manifest_sha256),
    )
    if expected is not None and attestation.as_dict() != expected.as_dict():
        raise ValueError("runtime tokenizer attestation differs from frozen manifest")
    return actual, attestation

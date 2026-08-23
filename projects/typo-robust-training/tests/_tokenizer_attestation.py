"""Deterministic frozen-tokenizer attestation fixtures for scientific runtimes."""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import replace
from pathlib import Path

from typo_cot.models.tokenizer_attestation import (
    TOKENIZER_ASSET_FILENAMES,
    TokenizerAssetAttestation,
    TokenizerSnapshotAttestation,
    tokenizer_attestation_manifest_bytes,
)


def frozen_tokenizer_attestation(
    model: str,
    revision: str,
) -> tuple[bytes, TokenizerSnapshotAttestation]:
    assets = tuple(
        TokenizerAssetAttestation(
            filename=filename,
            present=filename == "tokenizer_config.json",
            sha256="b" * 64 if filename == "tokenizer_config.json" else None,
        )
        for filename in TOKENIZER_ASSET_FILENAMES
    )
    dynamic = TokenizerSnapshotAttestation(
        model_name=model,
        requested_revision=revision,
        observed_commit=revision,
        assets=assets,
        tokenizer_fingerprint_sha256="c" * 64,
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
    )
    raw = tokenizer_attestation_manifest_bytes(dynamic)
    frozen = replace(
        dynamic,
        source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return raw, frozen


def tokenizer_attestation_provenance(model: str, revision: str) -> dict[str, object]:
    return frozen_tokenizer_attestation(model, revision)[1].provenance_dict()


def write_tokenizer_attestation_manifest(
    path: Path,
    *,
    model: str,
    revision: str,
) -> TokenizerSnapshotAttestation:
    raw, frozen = frozen_tokenizer_attestation(model, revision)
    path.write_bytes(raw)
    return frozen

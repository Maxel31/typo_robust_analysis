"""Produce and verify the tokenizer manifest used by scientific runtimes.

The consumer-facing file deliberately uses the existing
``typo-cot-tokenizer-snapshot-attestation/v1`` schema.  This module only adds a
closed-world producer record beside that file so the Hub identity, executing
source, and runtime that created it are auditable without weakening the shared
consumer contract.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from typo_cot.models.tokenizer_attestation import (
    TOKENIZER_ATTESTATION_MANIFEST_ENV,
    TokenizerSnapshotAttestation,
    load_attested_tokenizer,
    load_tokenizer_attestation_manifest,
    tokenizer_attestation_manifest_bytes,
)

from typo_robust_training.probe.attestation import (
    RuntimeCheckoutAttestation,
    attest_runtime_checkout,
)


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_SCHEMA = "freeze-tokenizer-attestation-run/v1"
_PROVIDER = "hugging-face-hub-exact-tokenizer-snapshot/v1"
_ATTESTATION_FILENAME = "tokenizer-attestation.json"
_RUN_FILENAME = "tokenizer-attestation-freeze-run.json"
_RUNTIME_PACKAGES = ("huggingface-hub", "tokenizers", "transformers")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_identity() -> dict[str, object]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in _RUNTIME_PACKAGES},
    }


def _producer_payload(
    *,
    model: str,
    revision: str,
    attestation: TokenizerSnapshotAttestation,
    attestation_file_sha256: str,
    checkout: RuntimeCheckoutAttestation,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": _RUN_SCHEMA,
        "provider": {
            "identity": _PROVIDER,
            "repository_id": model,
            "requested_revision": revision,
            "observed_commit": attestation.observed_commit,
            "trust_remote_code": False,
        },
        "code": checkout.as_dict(),
        "runtime": dict(runtime),
        "output": {
            "path": _ATTESTATION_FILENAME,
            "sha256": attestation_file_sha256,
            "attestation_sha256": attestation.sha256,
        },
    }


def _with_record_sha256(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["record_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate tokenizer freeze-run key: {key}")
        output[key] = value
    return output


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("tokenizer freeze-run manifest is not readable") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("tokenizer freeze-run manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("tokenizer freeze-run manifest must be an object")
    return value


def _validate_runtime(value: object) -> dict[str, object]:
    expected = _runtime_identity()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("tokenizer freeze runtime identity differs")
    return expected


def _validate_checkout(
    value: object,
    *,
    expected_revision: str,
) -> RuntimeCheckoutAttestation:
    observed = attest_runtime_checkout(expected_revision)
    if not isinstance(value, Mapping) or dict(value) != observed.as_dict():
        raise ValueError("tokenizer freeze code attestation differs")
    return observed


@dataclass(frozen=True, slots=True)
class TokenizerAttestationFreezeResult:
    """Paths and immutable identity of one completed freeze operation."""

    attestation_path: Path
    run_manifest_path: Path
    attestation: TokenizerSnapshotAttestation
    run_sha256: str


def load_tokenizer_attestation_freeze_bundle(
    run_manifest_path: Path,
    *,
    expected_model: str,
    expected_revision: str,
    expected_code_revision: str,
) -> TokenizerAttestationFreezeResult:
    """Fail closed over a frozen tokenizer file and its producer record."""

    if not expected_model:
        raise ValueError("expected tokenizer model must not be empty")
    if _REVISION.fullmatch(expected_revision) is None:
        raise ValueError("expected tokenizer revision must be one exact 40-hex commit")
    if _REVISION.fullmatch(expected_code_revision) is None:
        raise ValueError("expected code revision must be one exact 40-hex commit")

    payload = _read_json_object(run_manifest_path)
    expected_top_level = {
        "schema_version",
        "provider",
        "code",
        "runtime",
        "output",
        "record_sha256",
    }
    if set(payload) != expected_top_level or payload.get("schema_version") != _RUN_SCHEMA:
        raise ValueError("tokenizer freeze-run manifest fields or schema differ")

    record_sha256 = payload.get("record_sha256")
    if not isinstance(record_sha256, str) or _SHA256.fullmatch(record_sha256) is None:
        raise ValueError("tokenizer freeze-run aggregate SHA256 differs")
    unsigned = dict(payload)
    del unsigned["record_sha256"]
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != record_sha256:
        raise ValueError("tokenizer freeze-run aggregate SHA256 differs")

    expected_provider = {
        "identity": _PROVIDER,
        "repository_id": expected_model,
        "requested_revision": expected_revision,
        "observed_commit": expected_revision,
        "trust_remote_code": False,
    }
    provider = payload.get("provider")
    if not isinstance(provider, Mapping) or dict(provider) != expected_provider:
        raise ValueError("tokenizer freeze provider identity differs")

    _validate_checkout(payload.get("code"), expected_revision=expected_code_revision)
    _validate_runtime(payload.get("runtime"))

    output = payload.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "path",
        "sha256",
        "attestation_sha256",
    }:
        raise ValueError("tokenizer freeze output identity differs")
    if output.get("path") != _ATTESTATION_FILENAME:
        raise ValueError("tokenizer freeze output path differs")
    output_sha256 = output.get("sha256")
    output_attestation_sha256 = output.get("attestation_sha256")
    if (
        not isinstance(output_sha256, str)
        or _SHA256.fullmatch(output_sha256) is None
        or not isinstance(output_attestation_sha256, str)
        or _SHA256.fullmatch(output_attestation_sha256) is None
    ):
        raise ValueError("tokenizer freeze output digest differs")

    attestation_path = run_manifest_path.parent / _ATTESTATION_FILENAME
    try:
        attestation_raw = attestation_path.read_bytes()
    except OSError as exc:
        raise ValueError("frozen tokenizer attestation output is not readable") from exc
    if _sha256_bytes(attestation_raw) != output_sha256:
        raise ValueError("frozen tokenizer attestation file SHA256 differs")
    attestation = load_tokenizer_attestation_manifest(attestation_path)
    if (
        attestation.model_name != expected_model
        or attestation.requested_revision != expected_revision
        or attestation.observed_commit != expected_revision
        or attestation.sha256 != output_attestation_sha256
    ):
        raise ValueError("frozen tokenizer attestation identity differs")
    return TokenizerAttestationFreezeResult(
        attestation_path=attestation_path,
        run_manifest_path=run_manifest_path,
        attestation=attestation,
        run_sha256=record_sha256,
    )


def freeze_tokenizer_attestation(
    *,
    model: str,
    revision: str,
    code_revision: str,
    output_dir: Path,
) -> TokenizerAttestationFreezeResult:
    """Resolve an exact Hub tokenizer and atomically freeze its shared manifest."""

    if not model:
        raise ValueError("tokenizer model must not be empty")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("tokenizer revision must be one exact 40-hex commit")
    if _REVISION.fullmatch(code_revision) is None:
        raise ValueError("code revision must be one exact 40-hex commit")
    if os.environ.get(TOKENIZER_ATTESTATION_MANIFEST_ENV) is not None:
        raise ValueError(
            f"{TOKENIZER_ATTESTATION_MANIFEST_ENV} must be unset while freezing a new manifest"
        )
    if output_dir.exists():
        raise FileExistsError(f"tokenizer attestation output already exists: {output_dir}")

    checkout = attest_runtime_checkout(code_revision)
    runtime = _runtime_identity()
    _tokenizer, attestation = load_attested_tokenizer(model, revision)
    if (
        attestation.model_name != model
        or attestation.requested_revision != revision
        or attestation.observed_commit != revision
        or attestation.source_manifest_sha256 is not None
    ):
        raise ValueError("resolved tokenizer identity differs from the freeze request")

    attestation_raw = tokenizer_attestation_manifest_bytes(attestation)
    producer = _producer_payload(
        model=model,
        revision=revision,
        attestation=attestation,
        attestation_file_sha256=_sha256_bytes(attestation_raw),
        checkout=checkout,
        runtime=runtime,
    )
    run_raw = _pretty_json_bytes(_with_record_sha256(producer))

    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        raise FileExistsError(f"tokenizer attestation temporary output exists: {temporary}")
    try:
        temporary.mkdir(parents=False)
        (temporary / _ATTESTATION_FILENAME).write_bytes(attestation_raw)
        (temporary / _RUN_FILENAME).write_bytes(run_raw)
        temporary.rename(output_dir)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise

    return load_tokenizer_attestation_freeze_bundle(
        output_dir / _RUN_FILENAME,
        expected_model=model,
        expected_revision=revision,
        expected_code_revision=code_revision,
    )


__all__ = [
    "TokenizerAttestationFreezeResult",
    "freeze_tokenizer_attestation",
    "load_tokenizer_attestation_freeze_bundle",
]

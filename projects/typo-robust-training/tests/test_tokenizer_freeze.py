"""Adversarial tests for the production tokenizer-attestation freeze CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typo_cot.models import tokenizer_attestation as tokenizer_module
from typo_cot.models.tokenizer_attestation import (
    TOKENIZER_ASSET_FILENAMES,
    TokenizerAssetAttestation,
    TokenizerSnapshotAttestation,
    load_tokenizer_attestation_manifest,
    preflight_frozen_tokenizer_attestation,
    require_frozen_tokenizer_attestation,
    tokenizer_attestation_manifest_bytes,
    validate_tokenizer_attestation_provenance,
)

from typo_robust_training.cli import register_commands
from typo_robust_training.probe.attestation import RuntimeCheckoutAttestation
from typo_robust_training import tokenizer_freeze


MODEL = "org/exact-model"
REVISION = "a" * 40
CODE_REVISION = "b" * 40
RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.12.9",
    "platform": "Linux-test",
    "packages": {
        "huggingface-hub": "0.36.2",
        "tokenizers": "0.22.1",
        "transformers": "4.57.6",
    },
}
CHECKOUT = RuntimeCheckoutAttestation(
    revision=CODE_REVISION,
    typo_robust_training_tree="c" * 40,
    typo_cot_tree="d" * 40,
    typo_cot_runtime_sources=(
        "projects/typo-cot/src/typo_cot/__init__.py",
        "projects/typo-cot/src/typo_cot/models/wrapper.py",
    ),
)


def _attestation(*, model: str = MODEL, revision: str = REVISION) -> TokenizerSnapshotAttestation:
    return TokenizerSnapshotAttestation(
        model_name=model,
        requested_revision=revision,
        observed_commit=revision,
        assets=tuple(
            TokenizerAssetAttestation(
                filename=filename,
                present=filename in {"config.json", "tokenizer_config.json", "tokenizer.json"},
                sha256=(
                    hashlib.sha256(filename.encode()).hexdigest()
                    if filename in {"config.json", "tokenizer_config.json", "tokenizer.json"}
                    else None
                ),
            )
            for filename in TOKENIZER_ASSET_FILENAMES
        ),
        tokenizer_fingerprint_sha256="e" * 64,
        transformers_version="4.57.6",
        tokenizers_version="0.22.1",
    )


@pytest.fixture
def frozen_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tokenizer_module.TOKENIZER_ATTESTATION_MANIFEST_ENV, raising=False)
    monkeypatch.setattr(tokenizer_freeze, "_runtime_identity", lambda: dict(RUNTIME))
    monkeypatch.setattr(
        tokenizer_freeze,
        "attest_runtime_checkout",
        lambda revision: (
            CHECKOUT
            if revision == CODE_REVISION
            else (_ for _ in ()).throw(ValueError("revision differs"))
        ),
    )
    monkeypatch.setattr(
        tokenizer_freeze,
        "load_attested_tokenizer",
        lambda model, revision: (
            object(),
            _attestation(model=model, revision=revision),
        ),
    )


def _freeze(tmp_path: Path) -> tokenizer_freeze.TokenizerAttestationFreezeResult:
    return tokenizer_freeze.freeze_tokenizer_attestation(
        model=MODEL,
        revision=REVISION,
        code_revision=CODE_REVISION,
        output_dir=tmp_path / "frozen-tokenizer",
    )


def _rehash_run(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    unsigned = dict(payload)
    unsigned.pop("record_sha256", None)
    payload["record_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_freeze_is_deterministic_and_hashes_config_assets(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    first = _freeze(tmp_path / "first")
    second = tokenizer_freeze.freeze_tokenizer_attestation(
        model=MODEL,
        revision=REVISION,
        code_revision=CODE_REVISION,
        output_dir=tmp_path / "second" / "frozen-tokenizer",
    )

    assert first.attestation_path.read_bytes() == second.attestation_path.read_bytes()
    assert first.run_manifest_path.read_bytes() == second.run_manifest_path.read_bytes()
    assets = {asset.filename: asset for asset in first.attestation.assets}
    assert assets["config.json"].sha256 is not None
    assert assets["tokenizer_config.json"].sha256 is not None
    assert assets["tokenizer.json"].sha256 is not None
    run = json.loads(first.run_manifest_path.read_text())
    assert run["provider"] == {
        "identity": "hugging-face-hub-exact-tokenizer-snapshot/v1",
        "repository_id": MODEL,
        "requested_revision": REVISION,
        "observed_commit": REVISION,
        "trust_remote_code": False,
    }
    assert run["code"] == CHECKOUT.as_dict()
    assert run["runtime"] == RUNTIME


def test_freeze_rejects_nonexact_revision_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        tokenizer_freeze,
        "load_attested_tokenizer",
        lambda *_args: (_ for _ in ()).throw(AssertionError("provider accessed")),
    )
    with pytest.raises(ValueError, match="exact 40-hex"):
        tokenizer_freeze.freeze_tokenizer_attestation(
            model=MODEL,
            revision="main",
            code_revision=CODE_REVISION,
            output_dir=tmp_path / "output",
        )


def test_bundle_rejects_revision_mismatch(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)
    with pytest.raises(ValueError, match="provider identity differs"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision="f" * 40,
            expected_code_revision=CODE_REVISION,
        )


@pytest.mark.parametrize("field", ["extra", "missing"])
def test_bundle_rejects_extra_or_missing_run_fields_even_after_rehash(
    tmp_path: Path,
    frozen_runtime: None,
    field: str,
) -> None:
    result = _freeze(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        if field == "extra":
            payload["unexpected"] = True
        else:
            del payload["runtime"]

    _rehash_run(result.run_manifest_path, mutate)
    with pytest.raises(ValueError, match="fields or schema differ"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision=REVISION,
            expected_code_revision=CODE_REVISION,
        )


def test_bundle_rejects_local_cache_substituted_for_provider_identity(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        provider = payload["provider"]
        assert isinstance(provider, dict)
        provider["identity"] = "unverified-local-cache/v1"

    _rehash_run(result.run_manifest_path, mutate)
    with pytest.raises(ValueError, match="provider identity differs"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision=REVISION,
            expected_code_revision=CODE_REVISION,
        )


def test_bundle_rejects_self_rehashed_runtime_tamper(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        runtime = payload["runtime"]
        assert isinstance(runtime, dict)
        runtime["python_version"] = "9.9.9"

    _rehash_run(result.run_manifest_path, mutate)
    with pytest.raises(ValueError, match="runtime identity differs"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision=REVISION,
            expected_code_revision=CODE_REVISION,
        )


def test_bundle_rejects_self_rehashed_code_tree_tamper(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        code = payload["code"]
        assert isinstance(code, dict)
        code["typo_cot_tree"] = "0" * 40

    _rehash_run(result.run_manifest_path, mutate)
    with pytest.raises(ValueError, match="code attestation differs"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision=REVISION,
            expected_code_revision=CODE_REVISION,
        )


def test_bundle_rejects_self_rehashed_tokenizer_identity_tamper(
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)
    replacement = _attestation(model="attacker/rehashed-model")
    replacement_raw = tokenizer_attestation_manifest_bytes(replacement)
    result.attestation_path.write_bytes(replacement_raw)

    def mutate(payload: dict[str, object]) -> None:
        output = payload["output"]
        assert isinstance(output, dict)
        output["sha256"] = hashlib.sha256(replacement_raw).hexdigest()
        output["attestation_sha256"] = replacement.sha256

    _rehash_run(result.run_manifest_path, mutate)
    with pytest.raises(ValueError, match="attestation identity differs"):
        tokenizer_freeze.load_tokenizer_attestation_freeze_bundle(
            result.run_manifest_path,
            expected_model=MODEL,
            expected_revision=REVISION,
            expected_code_revision=CODE_REVISION,
        )


@pytest.mark.parametrize("field", ["extra", "missing"])
def test_existing_attestation_validator_rejects_closed_world_field_tamper(
    tmp_path: Path,
    frozen_runtime: None,
    field: str,
) -> None:
    result = _freeze(tmp_path)
    payload = json.loads(result.attestation_path.read_text())
    if field == "extra":
        payload["unexpected"] = True
    else:
        del payload["tokenizers_version"]
    result.attestation_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="manifest fields differ"):
        load_tokenizer_attestation_manifest(result.attestation_path)


def test_generated_manifest_is_accepted_by_shared_scientific_consumer_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_runtime: None,
) -> None:
    result = _freeze(tmp_path)
    monkeypatch.setenv(
        tokenizer_module.TOKENIZER_ATTESTATION_MANIFEST_ENV,
        str(result.attestation_path),
    )
    monkeypatch.setattr(
        tokenizer_module.importlib.metadata,
        "version",
        lambda package: RUNTIME["packages"][package],
    )

    preflight = preflight_frozen_tokenizer_attestation(
        expected_model=MODEL,
        expected_revision=REVISION,
    )
    loaded = load_tokenizer_attestation_manifest(result.attestation_path)
    accepted = require_frozen_tokenizer_attestation(
        SimpleNamespace(tokenizer_snapshot_attestation=loaded),
        expected_model=MODEL,
        expected_revision=REVISION,
    )
    embedded = validate_tokenizer_attestation_provenance(
        accepted.provenance_dict(),
        expected_model=MODEL,
        expected_revision=REVISION,
    )
    assert preflight.provenance_dict() == embedded.provenance_dict()


def test_cli_registers_exact_freeze_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = SimpleNamespace(
        attestation_path=tmp_path / "tokenizer-attestation.json",
        run_manifest_path=tmp_path / "tokenizer-attestation-freeze-run.json",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        tokenizer_freeze,
        "freeze_tokenizer_attestation",
        lambda **kwargs: calls.append(kwargs) or result,
    )
    parser = argparse.ArgumentParser()
    register_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "freeze-tokenizer-attestation",
            "--model",
            MODEL,
            "--revision",
            REVISION,
            "--code-revision",
            CODE_REVISION,
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args._typo_cot_plugin_handler(args) == 0
    assert calls == [
        {
            "model": MODEL,
            "revision": REVISION,
            "code_revision": CODE_REVISION,
            "output_dir": tmp_path / "output",
        }
    ]

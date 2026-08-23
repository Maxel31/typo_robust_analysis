"""Adversarial tests for exact-snapshot tokenizer attestation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typo_cot.models import tokenizer_attestation as module
from typo_cot.models.tokenizer_attestation import (
    TokenizerAssetAttestation,
    TokenizerSnapshotAttestation,
    _ResolvedTokenizerSnapshot,
    load_attested_tokenizer,
    require_frozen_tokenizer_attestation,
    tokenizer_attestation_manifest_bytes,
    tokenizer_fingerprint_sha256,
)


class _Backend:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_str(self) -> str:
        # Deliberately non-canonical ordering/spacing: the implementation must
        # parse and canonicalize rather than hash this presentation directly.
        return json.dumps(self.payload, sort_keys=False, indent=2)


class _FakeTokenizer:
    is_fast = True
    add_bos_token = True
    add_eos_token = False
    special_tokens_map_extended = {"eos_token": "</s>", "pad_token": "<pad>"}
    eos_token_id = 1
    pad_token_id = 0
    all_special_tokens = ["<pad>", "</s>"]
    all_special_ids = [0, 1]
    chat_template = "{{ messages }}"
    truncation_side = "right"
    clean_up_tokenization_spaces = False
    split_special_tokens = False
    add_prefix_space = False

    def __init__(
        self,
        *,
        backend_marker: str = "exact",
        padding_side: str = "right",
        added_token_id: int = 0,
        add_bos_token: bool = True,
        retained_commit: str | None = None,
        pad_token: str | None = "<pad>",
        eos_token: str = "</s>",
        special_marker: str = "exact",
    ) -> None:
        self.backend_tokenizer = _Backend(
            {"model": {"type": "Unigram", "marker": backend_marker}, "version": "1.0"}
        )
        self.padding_side = padding_side
        self.added_token_id = added_token_id
        self.add_bos_token = add_bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.pad_token_id = 0 if pad_token is not None else None
        self.eos_token_id = 1
        self.special_tokens_map_extended = {
            "eos_token": eos_token,
            "pad_token": pad_token,
            "additional_special_tokens": [f"<{special_marker}>"],
        }
        self.additional_special_tokens_ids = [2]
        self.all_special_tokens = [token for token in (pad_token, eos_token) if token]
        self.all_special_tokens.append(f"<{special_marker}>")
        self.all_special_ids = [0, 1, 2] if pad_token is not None else [1, 2]
        self.model_max_length = 8192
        self.init_kwargs: dict[str, Any] = {}
        if retained_commit is not None:
            self.init_kwargs["_commit_hash"] = retained_commit

    def get_added_vocab(self) -> dict[str, int]:
        return {"<pad>": self.added_token_id}


def _resolved(tmp_path: Path, *, commit: str = "a" * 40) -> _ResolvedTokenizerSnapshot:
    snapshot = tmp_path / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    return _ResolvedTokenizerSnapshot(
        model_name="org/model",
        requested_revision="main",
        observed_commit=commit,
        snapshot_dir=snapshot,
        assets=(TokenizerAssetAttestation("tokenizer.json", True, "c" * 64),),
    )


def _assets(*, tokenizer_config_sha256: str) -> tuple[TokenizerAssetAttestation, ...]:
    return tuple(
        TokenizerAssetAttestation(
            filename,
            filename == "tokenizer_config.json",
            tokenizer_config_sha256 if filename == "tokenizer_config.json" else None,
        )
        for filename in module.TOKENIZER_ASSET_FILENAMES
    )


def _expected_attestation(
    *,
    commit: str,
    tokenizer_config_sha256: str,
    fingerprint: str,
    requested_revision: str | None = None,
) -> TokenizerSnapshotAttestation:
    return TokenizerSnapshotAttestation(
        model_name="org/model",
        requested_revision=requested_revision or commit,
        observed_commit=commit,
        assets=_assets(tokenizer_config_sha256=tokenizer_config_sha256),
        tokenizer_fingerprint_sha256=fingerprint,
        transformers_version="4.57.6",
        tokenizers_version="0.22.1",
    )


def _snapshot_asset(model_root: Path, commit: str, filename: str, content: bytes) -> Path:
    blob_hash = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content,
        usedforsecurity=False,
    ).hexdigest()
    blob = model_root / "blobs" / blob_hash
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    snapshot_path = model_root / "snapshots" / commit / filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to(blob)
    return snapshot_path


def test_fingerprint_canonicalizes_backend_json() -> None:
    first = _FakeTokenizer()
    second = _FakeTokenizer()
    second.backend_tokenizer = _Backend(
        {"version": "1.0", "model": {"marker": "exact", "type": "Unigram"}}
    )

    assert tokenizer_fingerprint_sha256(first) == tokenizer_fingerprint_sha256(second)


def test_attestation_stamps_only_observed_commit_after_independent_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    actual = _FakeTokenizer()
    reference = _FakeTokenizer()
    calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(module, "_resolve_tokenizer_snapshot", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "test")

    def fake_from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
        calls.append((name, kwargs))
        return actual if len(calls) == 1 else reference

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    loaded, attestation = load_attested_tokenizer("org/model", "main")

    assert loaded is actual
    assert actual.init_kwargs["_commit_hash"] == "a" * 40
    assert calls[0][1]["revision"] == "a" * 40
    assert calls[0][1]["local_files_only"] is True
    assert calls[1][0] == str(resolved.snapshot_dir)
    assert calls[1][1]["local_files_only"] is True
    assert attestation.observed_commit == "a" * 40
    assert len(attestation.sha256) == 64


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (_FakeTokenizer(backend_marker="wrong"), "does not match"),
        (_FakeTokenizer(padding_side="left"), "does not match"),
        (_FakeTokenizer(pad_token="<wrong-pad>"), "does not match"),
        (_FakeTokenizer(added_token_id=99), "does not match"),
        (_FakeTokenizer(add_bos_token=False), "does not match"),
        (_FakeTokenizer(retained_commit="b" * 40), "different commit revision"),
    ],
)
def test_attestation_rejects_wrong_backend_flags_or_retained_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    actual: _FakeTokenizer,
    message: str,
) -> None:
    resolved = _resolved(tmp_path)
    reference = _FakeTokenizer()
    loaded = iter((actual, reference))
    monkeypatch.setattr(module, "_resolve_tokenizer_snapshot", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "test")
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: next(loaded),
    )

    with pytest.raises(ValueError, match=message):
        load_attested_tokenizer("org/model", "main")

    # A failed comparison must never make an unverifiable revision observable.
    assert actual.init_kwargs.get("_commit_hash") != resolved.observed_commit


def test_snapshot_inventory_rejects_asset_from_wrong_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_commit = "a" * 40
    wrong_commit = "b" * 40
    model_root = tmp_path / "models--org--model"
    anchor = _snapshot_asset(model_root, requested_commit, "tokenizer_config.json", b"{}")
    wrong_asset = _snapshot_asset(model_root, wrong_commit, "tokenizer.json", b"wrong")

    def fake_cached_file(_model: str, filename: str, **_kwargs: Any) -> str | None:
        if filename == "tokenizer_config.json":
            return str(anchor)
        if filename == "tokenizer.json":
            return str(wrong_asset)
        return None

    def fake_extract(path: str | None, _fallback: None) -> str | None:
        if path is None:
            return None
        return wrong_commit if Path(path) == wrong_asset else requested_commit

    monkeypatch.setattr(module, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        module.HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["tokenizer_config.json", "tokenizer.json"],
    )
    monkeypatch.setattr(module, "extract_commit_hash", fake_extract)

    with pytest.raises(ValueError, match="tokenizer.json resolved from a different commit"):
        module._resolve_tokenizer_snapshot("org/model", requested_commit)


def test_snapshot_inventory_records_exact_presence_and_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", b"config")
    tokenizer_json = _snapshot_asset(model_root, commit, "tokenizer.json", b"backend")

    def fake_cached_file(_model: str, filename: str, **_kwargs: Any) -> str | None:
        return {
            "tokenizer_config.json": str(anchor),
            "tokenizer.json": str(tokenizer_json),
        }.get(filename)

    monkeypatch.setattr(module, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        module.HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["tokenizer_config.json", "tokenizer.json"],
    )
    monkeypatch.setattr(
        module,
        "extract_commit_hash",
        lambda path, _fallback: None if path is None else commit,
    )

    resolved = module._resolve_tokenizer_snapshot("org/model", commit)
    inventory = {asset.filename: asset for asset in resolved.assets}

    assert inventory["tokenizer.json"].present is True
    assert inventory["tokenizer.json"].sha256 == (
        "10e08a419e850eba1ebba18fdd28eb7ec1b7e8baa9bcc3b973e2b8891ec726be"
    )
    assert inventory["added_tokens.json"].present is False
    assert inventory["added_tokens.json"].sha256 is None


def test_snapshot_inventory_rejects_missing_resolved_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", b"config")
    missing = model_root / "snapshots" / commit / "tokenizer.json"

    def fake_cached_file(_model: str, filename: str, **_kwargs: Any) -> str | None:
        if filename == "tokenizer_config.json":
            return str(anchor)
        if filename == "tokenizer.json":
            return str(missing)
        return None

    monkeypatch.setattr(module, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        module.HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["tokenizer_config.json", "tokenizer.json"],
    )
    monkeypatch.setattr(
        module,
        "extract_commit_hash",
        lambda path, _fallback: None if path is None else commit,
    )

    with pytest.raises(ValueError, match="not an auditable snapshot symlink"):
        module._resolve_tokenizer_snapshot("org/model", commit)


def test_snapshot_inventory_rejects_redirected_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", b"config")
    content = b"redirected"
    external_hash = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content,
        usedforsecurity=False,
    ).hexdigest()
    external_blob = tmp_path / "attacker" / external_hash
    external_blob.parent.mkdir()
    external_blob.write_bytes(content)
    redirected = model_root / "snapshots" / commit / "tokenizer.json"
    redirected.symlink_to(external_blob)

    def fake_cached_file(_model: str, filename: str, **_kwargs: Any) -> str | None:
        if filename == "tokenizer_config.json":
            return str(anchor)
        if filename == "tokenizer.json":
            return str(redirected)
        return None

    monkeypatch.setattr(module, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        module.HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["tokenizer_config.json", "tokenizer.json"],
    )
    monkeypatch.setattr(
        module,
        "extract_commit_hash",
        lambda path, _fallback: None if path is None else commit,
    )

    with pytest.raises(ValueError, match="points outside the model cache blobs"):
        module._resolve_tokenizer_snapshot("org/model", commit)


def test_frozen_inventory_rejects_same_cache_valid_blob_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    original = b"trusted config"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", original)
    expected = _expected_attestation(
        commit=commit,
        tokenizer_config_sha256=hashlib.sha256(original).hexdigest(),
        fingerprint="f" * 64,
    )

    replacement = b"different but internally valid cache blob"
    replacement_name = hashlib.sha1(
        f"blob {len(replacement)}\0".encode() + replacement,
        usedforsecurity=False,
    ).hexdigest()
    replacement_blob = model_root / "blobs" / replacement_name
    replacement_blob.write_bytes(replacement)
    anchor.unlink()
    anchor.symlink_to(replacement_blob)

    monkeypatch.setattr(
        module,
        "cached_file",
        lambda _model, filename, **kwargs: str(anchor)
        if filename == "tokenizer_config.json"
        else None,
    )
    monkeypatch.setattr(module, "extract_commit_hash", lambda _path, _fallback: commit)

    with pytest.raises(ValueError, match="differs from frozen manifest"):
        module._resolve_tokenizer_snapshot(
            "org/model",
            commit,
            expected=expected,
        )


def test_frozen_inventory_rejects_injected_locally_absent_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    config = b"trusted config"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", config)
    _snapshot_asset(model_root, commit, "tokenizer.json", b"injected")
    expected = _expected_attestation(
        commit=commit,
        tokenizer_config_sha256=hashlib.sha256(config).hexdigest(),
        fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        module,
        "cached_file",
        lambda _model, filename, **kwargs: str(anchor)
        if filename == "tokenizer_config.json"
        else None,
    )
    monkeypatch.setattr(module, "extract_commit_hash", lambda _path, _fallback: commit)

    with pytest.raises(ValueError, match="exists locally but is absent"):
        module._resolve_tokenizer_snapshot(
            "org/model",
            commit,
            expected=expected,
        )


def test_attestation_rejects_special_token_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    actual = _FakeTokenizer(special_marker="wrong")
    reference = _FakeTokenizer()
    loaded = iter((actual, reference))
    monkeypatch.setattr(module, "_resolve_tokenizer_snapshot", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "test")
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: next(loaded),
    )

    with pytest.raises(ValueError, match="does not match"):
        load_attested_tokenizer("org/model", "main")


def test_attestation_normalizes_padding_before_fingerprint_and_returns_it_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    actual = _FakeTokenizer(pad_token=None)
    reference = _FakeTokenizer(pad_token=None)
    loaded = iter((actual, reference))
    monkeypatch.setattr(module, "_resolve_tokenizer_snapshot", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "test")
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: next(loaded),
    )

    tokenizer, attestation = load_attested_tokenizer("org/model", "main")

    assert tokenizer.pad_token == tokenizer.eos_token
    assert attestation.tokenizer_fingerprint_sha256 == tokenizer_fingerprint_sha256(tokenizer)


def test_frozen_manifest_loads_from_complete_cache_while_hub_is_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model_root = tmp_path / "models--org--model"
    config = b"trusted config"
    anchor = _snapshot_asset(model_root, commit, "tokenizer_config.json", config)
    actual = _FakeTokenizer()
    reference = _FakeTokenizer()
    fingerprint = tokenizer_fingerprint_sha256(actual)
    expected = _expected_attestation(
        commit=commit,
        tokenizer_config_sha256=hashlib.sha256(config).hexdigest(),
        fingerprint=fingerprint,
    )
    manifest_path = tmp_path / "tokenizer-attestation.json"
    manifest_path.write_bytes(tokenizer_attestation_manifest_bytes(expected))
    monkeypatch.setenv(module.TOKENIZER_ATTESTATION_MANIFEST_ENV, str(manifest_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        module.HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    def fake_cached_file(_model: str, filename: str, **kwargs: Any) -> str | None:
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == commit
        return str(anchor) if filename == "tokenizer_config.json" else None

    monkeypatch.setattr(module, "cached_file", fake_cached_file)
    monkeypatch.setattr(module, "extract_commit_hash", lambda _path, _fallback: commit)
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: "4.57.6" if name == "transformers" else "0.22.1",
    )
    loaded = iter((actual, reference))
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: next(loaded),
    )

    tokenizer, attestation = load_attested_tokenizer("org/model", commit)

    assert tokenizer is actual
    assert attestation.provenance_dict()["manifest_file_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_attested_load_rejects_remote_code_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_resolve_tokenizer_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolved")),
    )
    with pytest.raises(ValueError, match="forbid trust_remote_code"):
        load_attested_tokenizer("org/model", "a" * 40, trust_remote_code=True)


def test_scientific_consumer_rejects_empty_or_dynamic_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="no snapshot attestation"):
        require_frozen_tokenizer_attestation(
            SimpleNamespace(tokenizer_snapshot_attestation=None),
            expected_model="org/model",
            expected_revision="a" * 40,
        )

    dynamic = _expected_attestation(
        commit="a" * 40,
        tokenizer_config_sha256="b" * 64,
        fingerprint="c" * 64,
    )
    monkeypatch.delenv(module.TOKENIZER_ATTESTATION_MANIFEST_ENV, raising=False)
    with pytest.raises(ValueError, match=module.TOKENIZER_ATTESTATION_MANIFEST_ENV):
        require_frozen_tokenizer_attestation(
            SimpleNamespace(tokenizer_snapshot_attestation=dynamic),
            expected_model="org/model",
            expected_revision="a" * 40,
        )

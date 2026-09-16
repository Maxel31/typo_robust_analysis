"""CPU-only acceptance tests for the rebuttal-v2 runtime lock."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typo_cot.experiments.rebuttal_v2.runtime_lock import (
    load_runtime_lock,
    scientific_config_sha256,
    scientific_projection,
    tokenizer_projection,
    validate_runtime_inputs,
)
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError, canonical_sha256, sha256_text

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/rebuttal_v2/protocol.json"
PAIR_A = "a" * 64
PAIR_B = "b" * 64
MODEL = "google/gemma-3-4b-it"


@pytest.fixture
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _descriptor(tokenizer_ref: dict[str, str]) -> dict:
    return {
        "tokenizer_id": "fixture/tokenizer",
        "revision": "2" * 40,
        "tokenizer_files": {"tokenizer.json": tokenizer_ref},
        "library": {"name": "tokenizers", "version": "0.22.2"},
        "special_token_ids": {"eos": 1, "pad": 0},
        "chat_template": None,
        "add_special_tokens": True,
        "normalization": None,
        "padding_side": "left",
        "offset_convention": "unicode-codepoint-half-open",
    }


def _entry(tokenizer_ref: dict[str, str], protocol: dict) -> dict:
    descriptor = _descriptor(tokenizer_ref)
    return {
        "model_id": MODEL,
        "model_revision": "1" * 40,
        "model_files": {"model.safetensors": "3" * 64},
        "tokenizer_id": descriptor.pop("tokenizer_id"),
        "tokenizer_revision": descriptor.pop("revision"),
        "tokenizer_policy": descriptor,
        "generation": {**protocol["generation"], "temperature": 0.0},
        "attention_backend": "sdpa",
        "cache_behavior": {"use_cache": True, "implementation": "dynamic"},
        "effective_eos_ids": [1, 2],
        "library_versions": {
            "tokenizers": "0.22.2",
            "transformers": "4.57.6",
            "torch": "2.10.0",
        },
        "code_commit": "4" * 40,
        "code_tree_sha256": "5" * 64,
        "device_identity": {
            "type": "cuda",
            "model": "Fixture GPU",
            "compute_capability": "8.0",
        },
        "compute_settings": {
            "cuda_version": "13.0",
            "driver_version": "999.0",
            "deterministic_algorithms": True,
            "allow_tf32": False,
            "matmul_precision": "highest",
        },
        "prompt_sha256": {},
        "prompt_token_ids_sha256": {},
        "donor_prompt_sha256": {},
        "donor_prompt_token_ids_sha256": {},
    }


def _write_lock(
    root: Path,
    protocol: dict,
    *,
    token_name: str = "tokenizer.json",
    provenance: dict | None = None,
    mutate=None,
) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / token_name
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_bytes(b'{"fixture":true}')
    token_ref = {
        "path": token_name,
        "sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
    }
    entry = _entry(token_ref, protocol)
    if mutate is not None:
        mutate(entry)
    payload = {
        "schema_version": "rebuttal-runtime-lock/v2",
        "settings": {"R3": entry},
    }
    if provenance is not None:
        payload["provenance"] = provenance
    path = root / "runtime_lock.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, entry


def _load(root: Path, protocol: dict, **kwargs):
    path, _ = _write_lock(root, protocol, **kwargs)
    return load_runtime_lock(path, protocol, ["R3"])


def _pair(pair_id: str, clean: str, typo: str | None = "typo") -> dict:
    return {
        "pair_id": pair_id,
        "model_id": MODEL,
        "task": "gsm8k",
        "clean_prompt": clean,
        "typo_prompt": typo,
    }


def _audit_row(pair: dict, clean_ids: list[int], typo_ids: list[int] | None) -> dict:
    def side(ids: list[int] | None) -> dict:
        return {
            "prompt_token_ids": ids,
            "prompt_token_ids_sha256": None if ids is None else canonical_sha256(ids),
        }

    return {
        "pair_id": pair["pair_id"],
        "alignment": {
            "eligibility": "invalid",
            "clean": side(clean_ids),
            "typo": side(typo_ids),
        },
    }


def _audit_bundle(protocol: dict, descriptor: dict, rows: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        protocol=protocol,
        tokenizer_lock={
            "schema_version": "rebuttal-tokenizer-lock/v2",
            "models": {MODEL: descriptor},
        },
        rows=rows,
    )


def _bind_pair(entry: dict, pair: dict, clean_ids: list[int], typo_ids: list[int] | None) -> None:
    entry["prompt_sha256"][pair["pair_id"]] = {
        "clean": sha256_text(pair["clean_prompt"]),
        "typo": None if pair["typo_prompt"] is None else sha256_text(pair["typo_prompt"]),
    }
    entry["prompt_token_ids_sha256"][pair["pair_id"]] = {
        "clean": canonical_sha256(clean_ids),
        "typo": None if typo_ids is None else canonical_sha256(typo_ids),
    }


def test_paths_telemetry_shards_and_physical_gpu_indices_do_not_change_hash(
    tmp_path: Path, protocol: dict
) -> None:
    first = _load(
        tmp_path / "a",
        protocol,
        token_name="tokenizer.json",
        provenance={"path": "/host/a", "telemetry": {"temperature": 51}, "shard_count": 1},
    )
    second = _load(
        tmp_path / "b",
        protocol,
        token_name="copied/tokenizer.json",
        provenance={
            "path": "/host/b",
            "telemetry": {"temperature": 99},
            "shard_count": 6,
            "physical_gpu_indices": [7],
        },
    )

    assert scientific_projection(first, protocol, ["R3"]) == scientific_projection(
        second, protocol, ["R3"]
    )
    assert scientific_config_sha256(first, protocol, ["R3"]) == scientific_config_sha256(
        second, protocol, ["R3"]
    )
    assert len(first.snapshots) == len(second.snapshots) == 2
    projected = tokenizer_projection(_descriptor({"path": "elsewhere", "sha256": "6" * 64}))
    assert projected["tokenizer_files"]["tokenizer.json"] == {"sha256": "6" * 64}


@pytest.mark.parametrize(
    "change",
    [
        lambda entry: entry["device_identity"].update(model="Different GPU"),
        lambda entry: entry["compute_settings"].update(allow_tf32=True),
        lambda entry: entry["generation"].update(repetition_penalty=1.1),
    ],
)
def test_device_precision_and_extra_generation_controls_change_scientific_hash(
    tmp_path: Path, protocol: dict, change
) -> None:
    baseline = _load(tmp_path / "baseline", protocol)
    changed = _load(tmp_path / "changed", protocol, mutate=change)

    assert scientific_config_sha256(baseline, protocol, ["R3"]) != scientific_config_sha256(
        changed, protocol, ["R3"]
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda entry: entry["generation"].update(quantization=0),
        lambda entry: entry["cache_behavior"].update(use_cache=1),
        lambda entry: entry.update(effective_eos_ids=[True]),
        lambda entry: entry["compute_settings"].update(allow_tf32=0),
    ],
)
def test_bool_and_integer_fields_are_strict(tmp_path: Path, protocol: dict, change) -> None:
    path, _ = _write_lock(tmp_path, protocol, mutate=change)
    with pytest.raises(IntakeError):
        load_runtime_lock(path, protocol, ["R3"])


def test_missing_selected_entry_and_unknown_scientific_key_are_fatal(
    tmp_path: Path, protocol: dict
) -> None:
    path, _ = _write_lock(tmp_path / "missing", protocol)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["settings"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntakeError, match="missing selected settings: R3"):
        load_runtime_lock(path, protocol, ["R3"])

    unknown, _ = _write_lock(
        tmp_path / "unknown", protocol, mutate=lambda entry: entry.update(shard_count=3)
    )
    with pytest.raises(IntakeError, match="unknown fields: shard_count"):
        load_runtime_lock(unknown, protocol, ["R3"])

    valid, _ = _write_lock(tmp_path / "empty", protocol)
    with pytest.raises(IntakeError, match="experiments must not be empty"):
        load_runtime_lock(valid, protocol, [])


@pytest.mark.parametrize(
    "change",
    [
        lambda entry: entry["compute_settings"].update(cuda_version=None),
        lambda entry: entry["compute_settings"].update(driver_version=None),
        lambda entry: entry.update(model_files={"../model.safetensors": "3" * 64}),
        lambda entry: entry.update(model_files={"/model.safetensors": "3" * 64}),
    ],
)
def test_cuda_versions_and_model_filenames_are_fully_resolved(
    tmp_path: Path, protocol: dict, change
) -> None:
    path, _ = _write_lock(tmp_path, protocol, mutate=change)
    with pytest.raises(IntakeError):
        load_runtime_lock(path, protocol, ["R3"])


def test_tokenizer_mismatch_is_reported_even_with_no_selected_pairs(
    tmp_path: Path, protocol: dict
) -> None:
    bundle = _load(tmp_path, protocol)
    descriptor = _descriptor({"path": "irrelevant", "sha256": "7" * 64})
    audit = _audit_bundle(protocol, descriptor, [])

    assert validate_runtime_inputs(bundle, audit, []) == [
        {
            "pair_id": None,
            "setting": "R3",
            "reason_codes": ["input_audit_tokenizer_descriptor_mismatch"],
        }
    ]


def test_all_available_prompt_blockers_are_collected(tmp_path: Path, protocol: dict) -> None:
    pair_a = _pair(PAIR_A, "clean-a", "typo-a")
    pair_b = _pair(PAIR_B, "clean-b", "typo-b")

    def bind(entry: dict) -> None:
        _bind_pair(entry, pair_a, [1, 2], [3, 4])
        _bind_pair(entry, pair_b, [5, 6], [7, 8])
        entry["prompt_sha256"][PAIR_A]["clean"] = "0" * 64
        entry["prompt_token_ids_sha256"][PAIR_A]["typo"] = None
        entry["prompt_token_ids_sha256"][PAIR_B]["clean"] = "1" * 64

    bundle = _load(tmp_path, protocol, mutate=bind)
    runtime_descriptor = _full_descriptor(bundle.settings["R3"])
    audit = _audit_bundle(
        protocol,
        runtime_descriptor,
        [_audit_row(pair_a, [1, 2], [3, 4]), _audit_row(pair_b, [5, 6], [7, 8])],
    )

    blockers = validate_runtime_inputs(bundle, audit, [pair_a, pair_b])

    assert {row["pair_id"] for row in blockers} == {PAIR_A, PAIR_B}
    by_pair = {row["pair_id"]: row["reason_codes"] for row in blockers}
    assert by_pair[PAIR_A] == [
        "clean_runtime_prompt_sha256_mismatch",
        "typo_runtime_prompt_token_ids_sha256_missing",
    ]
    assert by_pair[PAIR_B] == ["clean_runtime_prompt_token_ids_sha256_mismatch"]


def _full_descriptor(entry: dict) -> dict:
    return {
        "tokenizer_id": entry["tokenizer_id"],
        "revision": entry["tokenizer_revision"],
        **copy.deepcopy(entry["tokenizer_policy"]),
    }


def test_invalid_alignment_with_intact_complete_prompt_ids_passes(
    tmp_path: Path, protocol: dict
) -> None:
    pair = _pair(PAIR_A, "clean", "typo")
    bundle = _load(
        tmp_path,
        protocol,
        mutate=lambda entry: _bind_pair(entry, pair, [1, 2], [3, 4]),
    )
    audit = _audit_bundle(
        protocol,
        _full_descriptor(bundle.settings["R3"]),
        [_audit_row(pair, [1, 2], [3, 4])],
    )

    assert validate_runtime_inputs(bundle, audit, [pair]) == []


def test_unavailable_prompt_is_not_a_global_blocker(tmp_path: Path, protocol: dict) -> None:
    pair = _pair(PAIR_A, "clean", None)
    bundle = _load(
        tmp_path,
        protocol,
        mutate=lambda entry: _bind_pair(entry, pair, [1, 2], None),
    )
    audit = _audit_bundle(
        protocol,
        _full_descriptor(bundle.settings["R3"]),
        [_audit_row(pair, [1, 2], None)],
    )

    assert validate_runtime_inputs(bundle, audit, [pair]) == []


def test_known_empty_complete_prompt_ids_are_hash_checked_not_called_missing(
    tmp_path: Path, protocol: dict
) -> None:
    pair = _pair(PAIR_A, "", "")
    bundle = _load(
        tmp_path,
        protocol,
        mutate=lambda entry: _bind_pair(entry, pair, [], []),
    )
    audit = _audit_bundle(
        protocol,
        _full_descriptor(bundle.settings["R3"]),
        [_audit_row(pair, [], [])],
    )

    assert validate_runtime_inputs(bundle, audit, [pair]) == []


def test_every_matching_donor_is_checked_even_without_a_recipient_assignment(
    tmp_path: Path, protocol: dict
) -> None:
    donor_id = "c" * 64
    donor = _pair(donor_id, "donor-clean", None)
    donor_ids = [9, 10]

    def bind(entry: dict) -> None:
        entry["donor_prompt_sha256"][donor_id] = sha256_text(donor["clean_prompt"])
        entry["donor_prompt_token_ids_sha256"][donor_id] = canonical_sha256(donor_ids)

    bundle = _load(tmp_path, protocol, mutate=bind)
    descriptor = _full_descriptor(bundle.settings["R3"])
    audit = _audit_bundle(protocol, descriptor, [])
    bank = SimpleNamespace(
        rows=[{"pair_id": donor_id, "model_id": MODEL, "task": "gsm8k"}],
        pairs=[donor],
        audit_rows_by_pair_id={donor_id: _audit_row(donor, donor_ids, None)},
        tokenizer_descriptors_by_pair_id={donor_id: descriptor},
    )

    assert validate_runtime_inputs(bundle, audit, [], donor_bank=bank) == []

    changed = copy.deepcopy(bank.audit_rows_by_pair_id[donor_id])
    changed["alignment"]["clean"]["prompt_token_ids"] = [99]
    bank.audit_rows_by_pair_id[donor_id] = changed
    blockers = validate_runtime_inputs(bundle, audit, [], donor_bank=bank)
    assert blockers == [
        {
            "pair_id": donor_id,
            "setting": "R3",
            "reason_codes": [
                "clean_audit_prompt_token_ids_sha256_mismatch",
                "clean_runtime_prompt_token_ids_sha256_mismatch",
            ],
        }
    ]

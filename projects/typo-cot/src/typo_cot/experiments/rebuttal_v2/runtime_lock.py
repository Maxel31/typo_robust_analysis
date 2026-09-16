"""Strict CPU-only validation and hashing of fresh-runtime declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import artifact_io
from .artifacts import ArtifactSnapshots
from .input_tokenization import OFFSET_CONVENTION
from .schemas import (
    PROTOCOL_SHA256,
    IntakeError,
    canonical_json,
    canonical_sha256,
    require_bool,
    require_int,
    require_keys,
    require_object,
    require_sha256,
    require_string,
    sha256_text,
    strict_loads,
)

RUNTIME_LOCK_SCHEMA = "rebuttal-runtime-lock/v2"
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PAIR_ID = re.compile(r"[0-9a-f]{64}\Z")
_TOKENIZER_POLICY_FIELDS = {
    "tokenizer_files",
    "library",
    "special_token_ids",
    "chat_template",
    "add_special_tokens",
    "normalization",
    "padding_side",
    "offset_convention",
}
_ENTRY_FIELDS = {
    "model_id",
    "model_revision",
    "model_files",
    "tokenizer_id",
    "tokenizer_revision",
    "tokenizer_policy",
    "generation",
    "attention_backend",
    "cache_behavior",
    "effective_eos_ids",
    "library_versions",
    "code_commit",
    "code_tree_sha256",
    "device_identity",
    "compute_settings",
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "donor_prompt_sha256",
    "donor_prompt_token_ids_sha256",
}


@dataclass(frozen=True)
class RuntimeLockBundle:
    path: Path
    payload: dict[str, Any]
    settings: dict[str, dict[str, Any]]
    snapshots: dict[Path, str]

    @property
    def reference(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.snapshots[self.path]}


def _revision(value: object, context: str) -> str:
    result = require_string(value, context)
    if _REVISION.fullmatch(result) is None:
        raise IntakeError(f"{context} must be a lowercase 40-hex immutable commit")
    return result


def _json_copy(value: object, context: str) -> Any:
    """Validate JSON types and detach the normalized value from caller input."""

    return strict_loads(canonical_json(value), context)


def _string_map(value: object, context: str, *, required: set[str] = frozenset()) -> dict[str, str]:
    obj = require_object(value, context)
    if not obj:
        raise IntakeError(f"{context} must not be empty")
    result = {
        require_string(key, f"{context} key"): require_string(item, f"{context}.{key}")
        for key, item in obj.items()
    }
    missing = required.difference(result)
    if missing:
        raise IntakeError(f"{context} missing required fields: {', '.join(sorted(missing))}")
    return result


def _sha_map(value: object, context: str, *, allow_empty: bool = True) -> dict[str, str]:
    obj = require_object(value, context)
    if not allow_empty and not obj:
        raise IntakeError(f"{context} must not be empty")
    result = {}
    for key, item in obj.items():
        filename = require_string(key, f"{context} filename")
        posix = PurePosixPath(filename)
        windows = PureWindowsPath(filename)
        if (
            not posix.parts
            or posix.is_absolute()
            or windows.is_absolute()
            or ".." in posix.parts
            or ".." in windows.parts
        ):
            raise IntakeError(f"{context} filename must be a safe relative model path")
        result[filename] = require_sha256(item, f"{context}.{key}")
    return result


def _artifact_ref(value: object, context: str) -> dict[str, str]:
    obj = require_object(value, context)
    require_keys(obj, {"path", "sha256"}, context=context)
    return {
        "path": require_string(obj["path"], f"{context}.path"),
        "sha256": require_sha256(obj["sha256"], f"{context}.sha256"),
    }


def tokenizer_projection(descriptor: object) -> dict[str, Any]:
    """Remove only tokenizer ArtifactRef locations while preserving their hashes."""

    value = require_object(descriptor, "tokenizer descriptor")

    def project(item: object, context: str) -> Any:
        if isinstance(item, dict):
            if set(item) == {"path", "sha256"}:
                ref = _artifact_ref(item, context)
                return {"sha256": ref["sha256"]}
            return {key: project(child, f"{context}.{key}") for key, child in item.items()}
        if isinstance(item, list):
            return [project(child, f"{context}[{index}]") for index, child in enumerate(item)]
        return _json_copy(item, context)

    return project(value, "tokenizer descriptor")


def _tokenizer_policy(
    value: object,
    context: str,
    *,
    containing_file: Path,
    captured: ArtifactSnapshots,
) -> dict[str, Any]:
    policy = require_object(value, context)
    require_keys(policy, _TOKENIZER_POLICY_FIELDS, context=context)
    files = require_object(policy["tokenizer_files"], f"{context}.tokenizer_files")
    require_keys(files, {"tokenizer.json"}, context=f"{context}.tokenizer_files")
    tokenizer_files = {}
    for name, ref in files.items():
        parsed_ref = _artifact_ref(ref, f"{context}.tokenizer_files.{name}")
        captured.reference(parsed_ref, containing_file)
        tokenizer_files[name] = parsed_ref
    library = require_object(policy["library"], f"{context}.library")
    require_keys(library, {"name", "version"}, context=f"{context}.library")
    if require_string(library["name"], f"{context}.library.name") != "tokenizers":
        raise IntakeError(f"{context}.library.name must be tokenizers")
    library_value = {
        "name": "tokenizers",
        "version": require_string(library["version"], f"{context}.library.version"),
    }
    special = require_object(policy["special_token_ids"], f"{context}.special_token_ids")
    special_ids: dict[str, int | None] = {}
    for name, token_id in special.items():
        key = require_string(name, f"{context}.special_token_ids key")
        special_ids[key] = (
            None
            if token_id is None
            else require_int(token_id, f"{context}.special_token_ids.{key}", minimum=0)
        )
    chat = policy["chat_template"]
    if chat is not None:
        chat_obj = require_object(chat, f"{context}.chat_template")
        require_keys(chat_obj, {"text", "sha256"}, context=f"{context}.chat_template")
        text = require_string(chat_obj["text"], f"{context}.chat_template.text", allow_empty=True)
        digest = require_sha256(chat_obj["sha256"], f"{context}.chat_template.sha256")
        if sha256_text(text) != digest:
            raise IntakeError(f"{context}.chat_template SHA-256 mismatch")
        chat = {"text": text, "sha256": digest}
    add_special = require_bool(policy["add_special_tokens"], f"{context}.add_special_tokens")
    padding = require_string(policy["padding_side"], f"{context}.padding_side")
    if padding != "left":
        raise IntakeError(f"{context}.padding_side must be left")
    offset = require_string(policy["offset_convention"], f"{context}.offset_convention")
    if offset != OFFSET_CONVENTION:
        raise IntakeError(f"{context}.offset_convention must be {OFFSET_CONVENTION}")
    return {
        "tokenizer_files": tokenizer_files,
        "library": library_value,
        "special_token_ids": special_ids,
        "chat_template": chat,
        "add_special_tokens": add_special,
        "normalization": _json_copy(policy["normalization"], f"{context}.normalization"),
        "padding_side": padding,
        "offset_convention": offset,
    }


def _pair_hash_map(value: object, context: str) -> dict[str, dict[str, str | None]]:
    obj = require_object(value, context)
    result: dict[str, dict[str, str | None]] = {}
    for pair_id, raw_sides in obj.items():
        if _PAIR_ID.fullmatch(pair_id) is None:
            raise IntakeError(f"{context} key must be a lowercase 64-hex pair ID")
        sides = require_object(raw_sides, f"{context}.{pair_id}")
        require_keys(sides, {"clean", "typo"}, context=f"{context}.{pair_id}")
        result[pair_id] = {
            side: None
            if sides[side] is None
            else require_sha256(sides[side], f"{context}.{pair_id}.{side}")
            for side in ("clean", "typo")
        }
    return result


def _donor_hash_map(value: object, context: str) -> dict[str, str]:
    obj = require_object(value, context)
    result: dict[str, str] = {}
    for pair_id, digest in obj.items():
        if _PAIR_ID.fullmatch(pair_id) is None:
            raise IntakeError(f"{context} key must be a lowercase 64-hex pair ID")
        result[pair_id] = require_sha256(digest, f"{context}.{pair_id}")
    return result


def _protocol_settings(protocol: object) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = require_object(protocol, "protocol")
    if value.get("schema_version") != "rebuttal-protocol/v2":
        raise IntakeError("unsupported protocol schema_version")
    if canonical_sha256(value) != PROTOCOL_SHA256:
        raise IntakeError("protocol content does not match frozen revision 2.0.0")
    raw_settings = value.get("settings")
    if not isinstance(raw_settings, list):
        raise IntakeError("protocol.settings must be a JSON array")
    settings: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_settings):
        entry = require_object(raw, f"protocol.settings[{index}]")
        setting_id = require_string(entry.get("id"), f"protocol.settings[{index}].id")
        require_string(entry.get("model"), f"protocol.settings[{index}].model")
        require_string(entry.get("task"), f"protocol.settings[{index}].task")
        if setting_id in settings:
            raise IntakeError(f"protocol contains duplicate setting ID {setting_id}")
        settings[setting_id] = entry
    require_object(value.get("generation"), "protocol.generation")
    return value, settings


def _selected_experiments(
    experiments: object, protocol_settings: dict[str, dict[str, Any]]
) -> list[str]:
    if not isinstance(experiments, list):
        raise IntakeError("experiments must be a JSON array")
    selected = [
        require_string(item, f"experiments[{index}]") for index, item in enumerate(experiments)
    ]
    if len(selected) != len(set(selected)):
        raise IntakeError("experiments must not contain duplicates")
    if not selected:
        raise IntakeError("experiments must not be empty")
    unknown = set(selected).difference(protocol_settings)
    if unknown:
        raise IntakeError(f"unknown selected experiments: {', '.join(sorted(unknown))}")
    return sorted(selected)


def _entry(
    value: object,
    setting_id: str,
    protocol_setting: dict[str, Any],
    protocol_generation: dict[str, Any],
    lock_path: Path,
    captured: ArtifactSnapshots,
) -> dict[str, Any]:
    context = f"runtime lock setting {setting_id}"
    entry = require_object(value, context)
    require_keys(entry, _ENTRY_FIELDS, context=context)
    model_id = require_string(entry["model_id"], f"{context}.model_id")
    if canonical_json(model_id) != canonical_json(protocol_setting["model"]):
        raise IntakeError(f"{context}.model_id disagrees with selected protocol setting")
    policy = _tokenizer_policy(
        entry["tokenizer_policy"],
        f"{context}.tokenizer_policy",
        containing_file=lock_path,
        captured=captured,
    )
    generation = require_object(entry["generation"], f"{context}.generation")
    for key, expected in protocol_generation.items():
        if key not in generation:
            raise IntakeError(f"{context}.generation missing protocol field {key}")
        if canonical_json(generation[key]) != canonical_json(expected):
            raise IntakeError(f"{context}.generation.{key} disagrees with protocol")
    generation_value = _json_copy(generation, f"{context}.generation")
    if generation_value.get("padding_side") != "left":
        raise IntakeError(f"{context}.generation.padding_side must be left")
    cache = require_object(entry["cache_behavior"], f"{context}.cache_behavior")
    if not cache:
        raise IntakeError(f"{context}.cache_behavior must not be empty")
    if "use_cache" not in cache:
        raise IntakeError(f"{context}.cache_behavior missing required field use_cache")
    require_bool(cache["use_cache"], f"{context}.cache_behavior.use_cache")
    cache_value = _json_copy(cache, f"{context}.cache_behavior")
    eos_raw = entry["effective_eos_ids"]
    if not isinstance(eos_raw, list) or not eos_raw:
        raise IntakeError(f"{context}.effective_eos_ids must be a nonempty JSON array")
    eos_ids = [
        require_int(item, f"{context}.effective_eos_ids[{index}]", minimum=0)
        for index, item in enumerate(eos_raw)
    ]
    if len(eos_ids) != len(set(eos_ids)):
        raise IntakeError(f"{context}.effective_eos_ids must contain unique IDs")
    libraries = _string_map(
        entry["library_versions"],
        f"{context}.library_versions",
        required={"tokenizers", "transformers", "torch"},
    )
    if libraries["tokenizers"] != policy["library"]["version"]:
        raise IntakeError(f"{context}.library_versions.tokenizers disagrees with tokenizer policy")
    device = require_object(entry["device_identity"], f"{context}.device_identity")
    require_keys(
        device,
        {"type", "model", "compute_capability"},
        context=f"{context}.device_identity",
    )
    device_type = require_string(device["type"], f"{context}.device_identity.type")
    device_model = require_string(device["model"], f"{context}.device_identity.model")
    capability = device["compute_capability"]
    if device_type.lower() == "cuda":
        capability = require_string(capability, f"{context}.device_identity.compute_capability")
    elif capability is not None:
        raise IntakeError(
            f"{context}.device_identity.compute_capability must be null for non-CUDA devices"
        )
    compute = require_object(entry["compute_settings"], f"{context}.compute_settings")
    if not compute:
        raise IntakeError(f"{context}.compute_settings must not be empty")
    for key in ("cuda_version", "driver_version"):
        if key not in compute:
            raise IntakeError(f"{context}.compute_settings missing required field {key}")
        if compute[key] is not None:
            require_string(compute[key], f"{context}.compute_settings.{key}")
        elif device_type.lower() == "cuda":
            raise IntakeError(f"{context}.compute_settings.{key} must be known for CUDA")
    for key in ("deterministic_algorithms", "allow_tf32"):
        if key not in compute:
            raise IntakeError(f"{context}.compute_settings missing required field {key}")
        require_bool(compute[key], f"{context}.compute_settings.{key}")
    compute_value = _json_copy(compute, f"{context}.compute_settings")
    return {
        "model_id": model_id,
        "model_revision": _revision(entry["model_revision"], f"{context}.model_revision"),
        "model_files": _sha_map(entry["model_files"], f"{context}.model_files", allow_empty=False),
        "tokenizer_id": require_string(entry["tokenizer_id"], f"{context}.tokenizer_id"),
        "tokenizer_revision": _revision(
            entry["tokenizer_revision"], f"{context}.tokenizer_revision"
        ),
        "tokenizer_policy": policy,
        "generation": generation_value,
        "attention_backend": require_string(
            entry["attention_backend"], f"{context}.attention_backend"
        ),
        "cache_behavior": cache_value,
        "effective_eos_ids": eos_ids,
        "library_versions": libraries,
        "code_commit": _revision(entry["code_commit"], f"{context}.code_commit"),
        "code_tree_sha256": require_sha256(
            entry["code_tree_sha256"], f"{context}.code_tree_sha256"
        ),
        "device_identity": {
            "type": device_type,
            "model": device_model,
            "compute_capability": capability,
        },
        "compute_settings": compute_value,
        "prompt_sha256": _pair_hash_map(entry["prompt_sha256"], f"{context}.prompt_sha256"),
        "prompt_token_ids_sha256": _pair_hash_map(
            entry["prompt_token_ids_sha256"], f"{context}.prompt_token_ids_sha256"
        ),
        "donor_prompt_sha256": _donor_hash_map(
            entry["donor_prompt_sha256"], f"{context}.donor_prompt_sha256"
        ),
        "donor_prompt_token_ids_sha256": _donor_hash_map(
            entry["donor_prompt_token_ids_sha256"],
            f"{context}.donor_prompt_token_ids_sha256",
        ),
    }


def load_runtime_lock(
    path: str | Path,
    protocol: dict,
    experiments: list[str],
) -> RuntimeLockBundle:
    """Load selected runtime declarations without importing model frameworks."""

    protocol_value, protocol_settings = _protocol_settings(protocol)
    selected = _selected_experiments(experiments, protocol_settings)
    lock_path = Path(path).resolve()
    captured = ArtifactSnapshots()
    raw = captured.capture(lock_path)
    try:
        payload = require_object(strict_loads(raw.decode("utf-8"), str(lock_path)), str(lock_path))
    except UnicodeError as exc:
        raise IntakeError(f"runtime lock is not UTF-8: {lock_path}") from exc
    require_keys(payload, {"schema_version", "settings"}, {"provenance"}, "runtime lock")
    if payload["schema_version"] != RUNTIME_LOCK_SCHEMA:
        raise IntakeError("unsupported runtime lock schema_version")
    if "provenance" in payload:
        provenance = require_object(payload["provenance"], "runtime lock provenance")
        _json_copy(provenance, "runtime lock provenance")
    raw_settings = require_object(payload["settings"], "runtime lock settings")
    unknown = set(raw_settings).difference(protocol_settings)
    if unknown:
        raise IntakeError(f"runtime lock has unknown settings: {', '.join(sorted(unknown))}")
    missing = set(selected).difference(raw_settings)
    if missing:
        raise IntakeError(f"runtime lock missing selected settings: {', '.join(sorted(missing))}")
    generation = require_object(protocol_value["generation"], "protocol.generation")
    settings = {
        setting_id: _entry(
            raw_settings[setting_id],
            setting_id,
            protocol_settings[setting_id],
            generation,
            lock_path,
            captured,
        )
        for setting_id in selected
    }
    artifact_io.revalidate_snapshots(captured.snapshots)
    return RuntimeLockBundle(lock_path, payload, settings, dict(captured.snapshots))


def scientific_projection(
    bundle: RuntimeLockBundle,
    protocol: dict,
    experiments: list[str],
) -> dict[str, Any]:
    """Return the path-, telemetry-, provenance-, and shard-independent lock payload."""

    protocol_value, protocol_settings = _protocol_settings(protocol)
    selected = _selected_experiments(experiments, protocol_settings)
    missing = set(selected).difference(bundle.settings)
    if missing:
        raise IntakeError(f"runtime bundle missing selected settings: {', '.join(sorted(missing))}")
    settings: dict[str, dict[str, Any]] = {}
    for setting_id in selected:
        entry = _json_copy(bundle.settings[setting_id], f"runtime setting {setting_id}")
        entry["tokenizer_policy"] = tokenizer_projection(entry["tokenizer_policy"])
        settings[setting_id] = entry
    return {"protocol_sha256": canonical_sha256(protocol_value), "settings": settings}


def scientific_config_sha256(
    bundle: RuntimeLockBundle,
    protocol: dict,
    experiments: list[str],
) -> str:
    return canonical_sha256(scientific_projection(bundle, protocol, experiments))


def _full_tokenizer_descriptor(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer_id": entry["tokenizer_id"],
        "revision": entry["tokenizer_revision"],
        **entry["tokenizer_policy"],
    }


def _row_map(rows: object, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise IntakeError(f"{context} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"{context}[{index}]")
        pair_id = require_string(row.get("pair_id"), f"{context}[{index}].pair_id")
        if pair_id in result:
            raise IntakeError(f"{context} contains duplicate pair_id {pair_id}")
        result[pair_id] = row
    return result


def _prompt_reasons(
    pair: dict[str, Any],
    audit: dict[str, Any] | None,
    entry: dict[str, Any],
    *,
    donor: bool,
) -> list[str]:
    pair_id = pair["pair_id"]
    sides = ("clean",) if donor else ("clean", "typo")
    reasons: set[str] = set()
    text_map = entry["donor_prompt_sha256"] if donor else entry["prompt_sha256"]
    ids_map = entry["donor_prompt_token_ids_sha256"] if donor else entry["prompt_token_ids_sha256"]
    runtime_text = {"clean": text_map.get(pair_id)} if donor else text_map.get(pair_id, {})
    runtime_ids = {"clean": ids_map.get(pair_id)} if donor else ids_map.get(pair_id, {})
    if audit is None:
        reasons.add("input_audit_row_missing")
    for side in sides:
        prompt = pair.get(f"{side}_prompt")
        if prompt is None:
            continue
        if not isinstance(prompt, str):
            raise IntakeError(f"selected pair {pair_id} {side}_prompt must be a string or null")
        expected_text = runtime_text.get(side)
        if expected_text is None:
            reasons.add(f"{side}_runtime_prompt_sha256_missing")
        elif expected_text != sha256_text(prompt):
            reasons.add(f"{side}_runtime_prompt_sha256_mismatch")
        if audit is None:
            continue
        alignment = audit.get("alignment")
        side_audit = alignment.get(side) if isinstance(alignment, dict) else None
        if not isinstance(side_audit, dict):
            reasons.add(f"{side}_prompt_token_ids_missing")
            continue
        token_ids = side_audit.get("prompt_token_ids")
        stored_digest = side_audit.get("prompt_token_ids_sha256")
        if not isinstance(token_ids, list):
            reasons.add(f"{side}_prompt_token_ids_missing")
            continue
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise IntakeError(f"input audit {pair_id} {side} prompt token IDs are malformed")
        recomputed = canonical_sha256(token_ids)
        if not isinstance(stored_digest, str) or stored_digest != recomputed:
            reasons.add(f"{side}_audit_prompt_token_ids_sha256_mismatch")
        expected_ids = runtime_ids.get(side)
        if expected_ids is None:
            reasons.add(f"{side}_runtime_prompt_token_ids_sha256_missing")
        elif expected_ids != recomputed:
            reasons.add(f"{side}_runtime_prompt_token_ids_sha256_mismatch")
    return sorted(reasons)


def validate_runtime_inputs(
    bundle: RuntimeLockBundle,
    audit_bundle: object,
    selected_pairs: list[dict],
    *,
    donor_bank: object = None,
) -> list[dict[str, Any]]:
    """Collect every tokenizer and complete-prompt blocker for selected inputs."""

    if not isinstance(selected_pairs, list):
        raise IntakeError("selected_pairs must be a list")
    audit_rows = _row_map(getattr(audit_bundle, "rows", None), "input audit rows")
    tokenizer_lock = require_object(
        getattr(audit_bundle, "tokenizer_lock", None), "input audit tokenizer lock"
    )
    audit_descriptors = require_object(
        tokenizer_lock.get("models"), "input audit tokenizer lock models"
    )
    blockers: list[dict[str, Any]] = []
    settings_by_model_task: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for setting_id, entry in sorted(bundle.settings.items()):
        task = next(
            (
                setting["task"]
                for setting in audit_bundle.protocol["settings"]
                if setting["id"] == setting_id
            ),
            None,
        )
        settings_by_model_task[(entry["model_id"], task)] = (setting_id, entry)
        audit_descriptor = audit_descriptors.get(entry["model_id"])
        reasons = []
        if audit_descriptor is None:
            reasons.append("input_audit_tokenizer_descriptor_missing")
        elif canonical_json(tokenizer_projection(audit_descriptor)) != canonical_json(
            tokenizer_projection(_full_tokenizer_descriptor(entry))
        ):
            reasons.append("input_audit_tokenizer_descriptor_mismatch")
        if reasons:
            blockers.append({"pair_id": None, "setting": setting_id, "reason_codes": reasons})

    recipient_rows = _row_map(selected_pairs, "selected_pairs")
    for pair_id, pair in recipient_rows.items():
        key = (pair.get("model_id"), pair.get("task"))
        selected = settings_by_model_task.get(key)
        if selected is None:
            blockers.append(
                {
                    "pair_id": pair_id,
                    "setting": None,
                    "reason_codes": ["selected_pair_setting_mismatch"],
                }
            )
            continue
        setting_id, entry = selected
        reasons = _prompt_reasons(pair, audit_rows.get(pair_id), entry, donor=False)
        if reasons:
            blockers.append({"pair_id": pair_id, "setting": setting_id, "reason_codes": reasons})

    if donor_bank is not None:
        donor_rows = _row_map(getattr(donor_bank, "rows", None), "donor bank rows")
        donor_pairs = _row_map(getattr(donor_bank, "pairs", None), "donor bank pairs")
        donor_audits = require_object(
            getattr(donor_bank, "audit_rows_by_pair_id", None), "donor audit rows"
        )
        donor_descriptors = require_object(
            getattr(donor_bank, "tokenizer_descriptors_by_pair_id", None),
            "donor tokenizer descriptors",
        )
        for pair_id, row in donor_rows.items():
            selected = settings_by_model_task.get((row.get("model_id"), row.get("task")))
            if selected is None:
                continue
            setting_id, entry = selected
            reasons: set[str] = set()
            descriptor = donor_descriptors.get(pair_id)
            if descriptor is None:
                reasons.add("donor_tokenizer_descriptor_missing")
            elif canonical_json(tokenizer_projection(descriptor)) != canonical_json(
                tokenizer_projection(_full_tokenizer_descriptor(entry))
            ):
                reasons.add("donor_tokenizer_descriptor_mismatch")
            pair = donor_pairs.get(pair_id)
            if pair is None:
                reasons.add("donor_pair_missing")
            else:
                reasons.update(_prompt_reasons(pair, donor_audits.get(pair_id), entry, donor=True))
            if reasons:
                blockers.append(
                    {
                        "pair_id": pair_id,
                        "setting": setting_id,
                        "reason_codes": sorted(reasons),
                    }
                )
    return sorted(
        blockers,
        key=lambda row: (row["setting"] or "", row["pair_id"] or "", row["reason_codes"]),
    )

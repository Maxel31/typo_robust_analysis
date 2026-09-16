"""CPU-only JSON, validation and artifact primitives for rebuttal v2 intake."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path


class IntakeError(ValueError):
    """An input does not satisfy the declared rebuttal v2 artifact contract."""


PROTOCOL_SCHEMA_VERSION = "rebuttal-protocol/v2"
PROTOCOL_REVISION = "2.0.0"
PROTOCOL_SHA256 = "d13794e9d29f1c1750c66f996475ba05f7932e239768963ef1014b562868e70b"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


def require_string(value: object, context: str, *, allow_empty: bool = False) -> str:
    """Validate an exact UTF-8 string without trimming or normalizing it."""

    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "" if allow_empty else "non-empty "
        raise IntakeError(f"{context} must be a {qualifier}string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise IntakeError(f"{context} must contain valid UTF-8 text") from exc
    return value


def require_nullable_string(value: object, context: str) -> str | None:
    return None if value is None else require_string(value, context)


def require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise IntakeError(f"{context} must be a JSON object")
    for key in value:
        require_string(key, f"{context} object key", allow_empty=True)
    return value


def require_keys(
    value: dict[str, object],
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str = "record",
) -> None:
    """Reject both missing fields and undeclared schema extensions."""

    obj = require_object(value, context)
    required_keys, optional_keys = set(required), set(optional)
    missing = required_keys.difference(obj)
    unexpected = set(obj).difference(required_keys | optional_keys)
    if missing:
        raise IntakeError(f"{context} missing required fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise IntakeError(f"{context} has unknown fields: {', '.join(sorted(unexpected))}")


def require_int(value: object, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise IntakeError(f"{context} must be an integer{bound}, not a boolean")
    return value


def require_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise IntakeError(f"{context} must be a boolean")
    return value


def require_enum(value: object, choices: Iterable[str], context: str) -> str:
    result = require_string(value, context)
    allowed = tuple(choices)
    if result not in allowed:
        raise IntakeError(f"{context} must be one of {allowed!r}")
    return result


def require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntakeError(f"{context} must be a lowercase SHA-256 digest")
    return value


def require_string_list(value: object, context: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise IntakeError(f"{context} must be a JSON array")
    result = [require_string(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise IntakeError(f"{context} contains duplicate values")
    return result


def require_timestamp(value: object, context: str) -> str:
    result = require_string(value, context)
    if _UTC_TIMESTAMP.fullmatch(result) is None:
        raise IntakeError(f"{context} must be a UTC RFC 3339 timestamp ending in Z")
    try:
        datetime.fromisoformat(result)
    except ValueError as exc:
        raise IntakeError(f"{context} is not a valid calendar timestamp") from exc
    return result


def require_date(value: object, context: str) -> str:
    result = require_string(value, context)
    if _DATE.fullmatch(result) is None:
        raise IntakeError(f"{context} must be a YYYY-MM-DD date")
    try:
        date.fromisoformat(result)
    except ValueError as exc:
        raise IntakeError(f"{context} is not a valid calendar date") from exc
    return result


def _canonical_value(value: object, context: str) -> object:
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        return require_string(value, context, allow_empty=True)
    if type(value) is float:
        if not math.isfinite(value):
            raise IntakeError(f"{context} contains a nonfinite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_canonical_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        obj = require_object(value, context)
        return {key: _canonical_value(item, f"{context}.{key}") for key, item in obj.items()}
    raise IntakeError(f"{context} contains a non-JSON value of type {type(value).__name__}")


def canonical_json(payload: object) -> str:
    """Serialize the v2 canonical form, preserving integer/float distinctions."""

    try:
        return json.dumps(
            _canonical_value(payload, "canonical JSON"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, ValueError) as exc:
        if isinstance(exc, IntakeError):
            raise
        raise IntakeError(f"invalid canonical JSON: {exc}") from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        require_string(text, "text", allow_empty=True).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise IntakeError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return sha256_text(canonical_json(payload))


def strict_loads(text: str, context: str = "JSON") -> object:
    """Reject duplicate keys, nonstandard constants, and overflowing JSON floats."""

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise IntakeError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise IntakeError(f"nonfinite JSON constant {value}")

    try:
        require_string(text, context, allow_empty=True)
        payload = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)
        # parse_constant does not catch standard number syntax that overflows to inf.
        _canonical_value(payload, context)
        return payload
    except (ValueError, RecursionError) as exc:
        raise IntakeError(f"invalid JSON in {context}: {exc}") from exc


def load_json_object(path: Path) -> dict[str, object]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            payload = strict_loads(handle.read(), context=str(path))
    except (OSError, UnicodeError) as exc:
        raise IntakeError(f"cannot read JSON artifact {path}: {exc}") from exc
    return require_object(payload, str(path))


def iter_jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield physical line numbers and objects; blank JSONL lines are permitted."""

    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                context = f"{path}:{line_number}"
                yield line_number, require_object(strict_loads(line, context), context)
    except (OSError, UnicodeError) as exc:
        raise IntakeError(f"cannot read JSONL artifact {path}: {exc}") from exc


def validate_artifact_ref(ref: object, context: str = "artifact reference") -> dict[str, str]:
    obj = require_object(ref, context)
    require_keys(obj, ("path", "sha256"), context=context)
    return {
        "path": require_string(obj["path"], f"{context}.path"),
        "sha256": require_sha256(obj["sha256"], f"{context}.sha256"),
    }


def make_ref(path: Path, relative_to_file: Path) -> dict[str, str]:
    """Hash existing bytes and express their path relative to the containing file."""

    try:
        target = Path(path).resolve()
        base = Path(relative_to_file).resolve().parent
        relative = os.path.relpath(target, base)
    except (OSError, ValueError, RuntimeError) as exc:
        raise IntakeError(f"cannot resolve artifact reference for {path}: {exc}") from exc
    return {"path": relative, "sha256": sha256_file(target)}


def verify_ref(ref: object, relative_to_file: Path) -> Path:
    """Resolve against the containing file and verify the exact referenced bytes."""

    parsed = validate_artifact_ref(ref)
    try:
        path = Path(parsed["path"])
        if not path.is_absolute():
            path = Path(relative_to_file).resolve().parent / path
        path = path.resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise IntakeError(f"cannot resolve artifact reference: {exc}") from exc
    observed = sha256_file(path)
    if observed != parsed["sha256"]:
        raise IntakeError(
            f"artifact SHA-256 mismatch at {path}: expected {parsed['sha256']}, got {observed}"
        )
    return path


def validate_record_ref(ref: object, context: str = "record reference") -> dict[str, object]:
    """Validate shape; resolving unique record membership is the consumer's job."""

    obj = require_object(ref, context)
    require_keys(obj, ("artifact", "record_id"), context=context)
    return {
        "artifact": validate_artifact_ref(obj["artifact"], f"{context}.artifact"),
        "record_id": require_string(obj["record_id"], f"{context}.record_id"),
    }


def load_protocol(path: Path) -> dict[str, object]:
    """Accept the known protocol by content, independent of its path or formatting."""

    payload = load_json_object(path)
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise IntakeError(f"unsupported protocol schema_version at {path}")
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise IntakeError(f"unsupported protocol_revision at {path}")
    if canonical_sha256(payload) != PROTOCOL_SHA256:
        raise IntakeError(
            f"protocol content does not match frozen revision {PROTOCOL_REVISION}: {path}"
        )
    return payload

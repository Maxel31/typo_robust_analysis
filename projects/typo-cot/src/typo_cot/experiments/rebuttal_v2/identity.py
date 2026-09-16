"""Outcome-independent identities and evidence-backed aliases for rebuttal v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import (
    IntakeError,
    canonical_json,
    load_json_object,
    require_keys,
    require_nullable_string,
    require_object,
    require_string,
    sha256_text,
    validate_artifact_ref,
    verify_ref,
)

PAIR_IDENTITY_FIELDS = ("source_namespace", "source_pair_key")
GROUP_IDENTITY_FIELDS = ("dataset_id", "split", "original_problem_id")
ALIASES_SCHEMA_VERSION = "rebuttal-identity-aliases/v2"


def _validate_identity_value(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if isinstance(value, list):
        for item in value:
            _validate_identity_value(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_identity_value(item)
        return
    raise IntakeError("identity payload must contain JSON values without floats")


def identity_hash(domain: str, payload: object) -> str:
    """Hash the explicit record-identity domain and its canonical payload."""

    require_string(domain, "identity domain")
    if "\n" in domain or "\r" in domain:
        raise IntakeError("identity domain must be a single line")
    try:
        _validate_identity_value(payload)
    except RecursionError as exc:
        raise IntakeError("identity payload contains a cycle or excessive nesting") from exc
    return sha256_text(domain + "\n" + canonical_json(payload))


def pair_id_for(source_namespace: str, source_pair_key: str) -> str:
    return identity_hash(
        "rebuttal-pair/v2",
        {
            "source_namespace": require_string(source_namespace, "source_namespace"),
            "source_pair_key": require_string(source_pair_key, "source_pair_key"),
        },
    )


def original_problem_group_id_for(
    dataset_id: str | None,
    split: str | None,
    original_problem_id: str | None,
) -> str | None:
    payload = {
        "dataset_id": require_nullable_string(dataset_id, "dataset_id"),
        "split": require_nullable_string(split, "split"),
        "original_problem_id": require_nullable_string(original_problem_id, "original_problem_id"),
    }
    if any(value is None for value in payload.values()):
        return None
    return identity_hash("rebuttal-original-problem/v2", payload)


@dataclass(frozen=True)
class AliasResolution:
    """Preserve original identity and verified, absolute evidence references."""

    original: dict[str, str | None]
    canonical: dict[str, str | None]
    evidence_refs: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _AliasLink:
    canonical: tuple[str, ...]
    evidence_ref: dict[str, str]


class IdentityAliases:
    """Resolve a prevalidated alias graph without deriving identity from outcomes."""

    def __init__(
        self,
        pairs: dict[tuple[str, ...], _AliasLink] | None = None,
        groups: dict[tuple[str, ...], _AliasLink] | None = None,
    ) -> None:
        self._pairs = {} if pairs is None else dict(pairs)
        self._groups = {} if groups is None else dict(groups)

    def resolve_pair(self, source_namespace: str, source_pair_key: str) -> AliasResolution:
        payload = {
            "source_namespace": require_string(source_namespace, "source_namespace"),
            "source_pair_key": require_string(source_pair_key, "source_pair_key"),
        }
        return self._resolve(payload, PAIR_IDENTITY_FIELDS, self._pairs)

    def resolve_group(
        self,
        dataset_id: str | None,
        split: str | None,
        original_problem_id: str | None,
    ) -> AliasResolution:
        payload = {
            "dataset_id": require_nullable_string(dataset_id, "dataset_id"),
            "split": require_nullable_string(split, "split"),
            "original_problem_id": require_nullable_string(original_problem_id, "original_problem_id"),
        }
        return self._resolve(payload, GROUP_IDENTITY_FIELDS, self._groups)

    @staticmethod
    def _resolve(
        payload: dict[str, str | None],
        fields: tuple[str, ...],
        links: dict[tuple[str, ...], _AliasLink],
    ) -> AliasResolution:
        if any(value is None for value in payload.values()):
            return AliasResolution(dict(payload), dict(payload), ())
        key = tuple(require_string(payload[field], field) for field in fields)
        evidence: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()
        while key in links:
            if key in seen:
                raise IntakeError("identity alias cycle")
            seen.add(key)
            link = links[key]
            evidence.append(dict(link.evidence_ref))
            key = link.canonical
        return AliasResolution(dict(payload), dict(zip(fields, key)), tuple(evidence))


def _identity_tuple(value: object, fields: tuple[str, ...], context: str) -> tuple[str, ...]:
    obj = require_object(value, context)
    require_keys(obj, fields, context=context)
    return tuple(require_string(obj[field], f"{context}.{field}") for field in fields)


def _load_links(
    rows: object,
    fields: tuple[str, ...],
    path: Path,
    context: str,
) -> dict[tuple[str, ...], _AliasLink]:
    if not isinstance(rows, list):
        raise IntakeError(f"{context} must be a JSON array")
    result: dict[tuple[str, ...], _AliasLink] = {}
    for index, row in enumerate(rows):
        location = f"{context}[{index}]"
        obj = require_object(row, location)
        require_keys(obj, ("alias", "canonical", "evidence_ref"), context=location)
        alias = _identity_tuple(obj["alias"], fields, f"{location}.alias")
        canonical = _identity_tuple(obj["canonical"], fields, f"{location}.canonical")
        if alias in result:
            raise IntakeError(f"{location} has a duplicate alias key or conflicting canonical target")
        evidence = validate_artifact_ref(obj["evidence_ref"], f"{location}.evidence_ref")
        evidence_path = verify_ref(evidence, relative_to_file=path)
        result[alias] = _AliasLink(
            canonical,
            {"path": str(evidence_path), "sha256": evidence["sha256"]},
        )
    for alias in result:
        seen: set[tuple[str, ...]] = set()
        current = alias
        while current in result:
            if current in seen:
                raise IntakeError(f"{context} contains an identity alias cycle")
            seen.add(current)
            current = result[current].canonical
    return result


def parse_identity_aliases(payload: object, source_path: Path) -> IdentityAliases:
    """Validate captured alias content without reopening the source artifact.

    Evidence references are still resolved and verified relative to source_path.
    Callers binding an input hash must pass the object decoded from those bytes.
    """

    obj = require_object(payload, str(source_path))
    require_keys(obj, ("schema_version", "pair_aliases", "group_aliases"), context=str(source_path))
    if obj["schema_version"] != ALIASES_SCHEMA_VERSION:
        raise IntakeError(f"unsupported identity aliases schema_version at {source_path}")
    return IdentityAliases(
        _load_links(obj["pair_aliases"], PAIR_IDENTITY_FIELDS, source_path, "pair_aliases"),
        _load_links(obj["group_aliases"], GROUP_IDENTITY_FIELDS, source_path, "group_aliases"),
    )


def load_identity_aliases(path: Path | None) -> IdentityAliases:
    """Verify every alias/evidence entry, including aliases unused by this intake."""

    if path is None:
        return IdentityAliases()
    return parse_identity_aliases(load_json_object(path), source_path=path)

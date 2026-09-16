"""CPU-only, immutable scoring views over the archived generation manifest.

The public proxy is an explicitly labeled reproduction of the checked-in public
extractor plus its empty-result fallback. It is never historical parser code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typo_cot.evaluation import audit_extractor, extractor, fallback
from typo_cot.evaluation.audit_extractor import (
    AuditExtraction,
    ExtractionUnavailable,
    Score,
    canonicalize_answer,
    extract_explicit_answer,
    score_answer,
)
from typo_cot.experiments.rebuttal_v2.identity import identity_hash
from typo_cot.experiments.rebuttal_v2.schemas import canonical_json, canonical_sha256, sha256_file


PRIMARY = "explicit_answer_v2"
PROXY = "public_v1_proxy"
PARSERS = (PRIMARY, PROXY)
PROXY_VERSION = "public-extractor-fallback/v1"
SAMPLING_POLICY = {
    "schema_version": "rebuttal-answer-blind-sampling/v2",
    "seed": 42,
    "selection": "sha256-within-extraction-status-and-canonical-value-stratum",
    "disagreement_cap_per_stratum": 32,
    "agreement_cap": 8,
    "uses_gold_or_correctness": False,
    "human_judgments_received": False,
    "stratum_definition": "parser unavailable IDs; differing extraction statuses; differing canonical values; shared ambiguous/unextractable; extracted agreement",
}


def _parser_descriptors() -> dict[str, dict[str, Any]]:
    primary_files = {"audit_extractor.py": sha256_file(Path(audit_extractor.__file__))}
    proxy_files = {
        "extractor.py": sha256_file(Path(extractor.__file__)),
        "fallback.py": sha256_file(Path(fallback.__file__)),
    }
    return {
        PRIMARY: {
            "parser_id": PRIMARY,
            "parser_version": audit_extractor.PARSER_VERSION,
            "parser_code_sha256": canonical_sha256(primary_files),
            "parser_code_files": primary_files,
            "historical_parser_identity": "not_applicable",
            "score_rule": "exact-canonical-answer/v2",
        },
        PROXY: {
            "parser_id": PROXY,
            "parser_version": PROXY_VERSION,
            "parser_code_sha256": canonical_sha256(proxy_files),
            "parser_code_files": proxy_files,
            "historical_parser_identity": "unavailable-public-proxy-only",
            "score_rule": "fallback.answers_equal/public-v1",
            "auxiliary_score_rule": "exact-canonical-answer/v2",
        },
    }


def _code_identity(snapshots: dict[Path, str]) -> dict[str, Any]:
    base = Path(__file__).resolve().parent
    paths = [
        base / name
        for name in ("answer_audit.py", "artifacts.py", "identity.py", "schemas.py", "cli.py")
    ]
    paths += [Path(module.__file__).resolve() for module in (audit_extractor, extractor, fallback)]
    paths.append(base.parents[1] / "cli.py")
    refs = []
    for path in paths:
        digest = sha256_file(path)
        snapshots[path] = digest
        refs.append({"path": str(path), "sha256": digest})
    commit, dirty, reasons = None, None, []
    try:
        result = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        candidate = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate):
            commit = candidate
        else:
            reasons.append("code_commit_unknown")
        status = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = bool(status.stdout)
    except (OSError, subprocess.SubprocessError):
        reasons.append("code_git_identity_unavailable")
    return {
        "source_refs": sorted(refs, key=lambda item: item["path"]),
        "git_commit": commit,
        "git_dirty": dirty,
        "reason_codes": reasons,
    }


def _labels_supported(labels: Any) -> bool:
    return labels is None or (
        isinstance(labels, (list, tuple))
        and all(isinstance(label, str) and re.fullmatch(r"[A-Z]", label) for label in labels)
        and len(set(labels)) == len(labels)
    )


def _proxy_extract(record: dict[str, Any]) -> tuple[AuditExtraction, str]:
    task = record["task"]
    if task not in {"gsm8k", "mmlu-pro"}:
        raise ExtractionUnavailable("unsupported_task")
    benchmark = task.replace("-", "_")
    primary = extractor.create_extractor(benchmark).extract(record["raw_text"])
    value = primary.extracted_answer
    method = f"primary:{primary.extraction_method}"
    if not value:
        value, fallback_method = fallback.fallback_answer(
            record["raw_text"],
            benchmark=benchmark,
            allow_positional=record["termination"] != "length-cap",
        )
        method = f"fallback:{fallback_method}" if value else "unextractable"
    reasons = ["public_proxy_not_identified_historical_parser"]
    reasons.append("public_proxy_candidate_spans_not_reported")
    try:
        if not _labels_supported(record.get("valid_labels")):
            raise ExtractionUnavailable("unsupported_item_label_set")
        canonical = canonicalize_answer(value, task, record.get("valid_labels")) if value else None
    except ExtractionUnavailable as exc:
        canonical = None
        reasons.append(getattr(exc, "reason_code", str(exc)))
    if value and canonical is None:
        reasons.append("public_proxy_noncanonical_value")
    if not value:
        reasons.append("public_proxy_no_answer")
    return (
        AuditExtraction(
            task=task,
            extraction_status="extracted" if value else "unextractable",
            raw_value=value or None,
            canonical_value=canonical,
            candidate_spans=[],
            reasons=reasons,
            termination=record["termination"],
        ),
        method,
    )


def compare_parsers(generation_record: dict[str, Any]) -> dict[str, Any]:
    """Compare both parsers on one exact raw output; extraction receives no gold.

    The input includes task, labels and gold supplied by the verified pair join.
    The proxy score reproduces the public fallback module's string-equality rule
    only when archived raw gold is a nonempty string. Its separate auxiliary
    canonical score isolates extraction differences without changing that rule.
    """
    raw = generation_record["raw_text"]
    if not isinstance(raw, str):
        raise ValueError("compare_parsers requires present raw text")
    result: dict[str, Any] = {"generation_id": generation_record["generation_id"], "parsers": {}}
    for parser_id in PARSERS:
        try:
            if generation_record["task"] not in {"gsm8k", "mmlu-pro"}:
                raise ExtractionUnavailable("unsupported_task")
            if parser_id == PRIMARY:
                if not _labels_supported(generation_record.get("valid_labels")):
                    raise ExtractionUnavailable("unsupported_item_label_set")
                extraction = extract_explicit_answer(
                    raw,
                    generation_record["task"],
                    generation_record.get("valid_labels"),
                    generation_record["termination"],
                )
                method = "explicit-final-line-full-match"
            else:
                extraction, method = _proxy_extract(generation_record)
        except ExtractionUnavailable as exc:
            result["parsers"][parser_id] = {
                "available": False,
                "reason_codes": [getattr(exc, "reason_code", str(exc))],
            }
            continue
        score = (
            Score("not_scored", None, ["public_proxy_canonical_value_unavailable"])
            if parser_id == PROXY
            and extraction.raw_value is not None
            and extraction.canonical_value is None
            else score_answer(extraction, generation_record.get("canonical_gold"))
        )
        row = {
            "available": True,
            **asdict(extraction),
            **asdict(score),
            "extraction_method": method,
            "reason_codes": sorted(set(extraction.reasons + score.reasons)),
            "legacy_score_status": "not_applicable",
            "legacy_gold_match": None,
            "canonical_score_status": score.score_status,
            "canonical_gold_match": score.gold_match,
        }
        row.pop("reasons")
        if parser_id == PROXY:
            raw_gold = generation_record.get("gold")
            has_gold = isinstance(raw_gold, str) and bool(raw_gold.strip())
            row["legacy_score_status"] = "scored" if has_gold else "not_scored"
            row["legacy_gold_match"] = (
                fallback.answers_equal(
                    extraction.raw_value or "",
                    raw_gold,
                    benchmark=generation_record["task"].replace("-", "_"),
                )
                if has_gold
                else None
            )
            if not has_gold:
                row["reason_codes"].append("legacy_gold_unavailable")
            row["score_status"] = row["legacy_score_status"]
            row["gold_match"] = row["legacy_gold_match"]
            if extraction.raw_value is not None and extraction.canonical_value is None:
                row["canonical_score_status"] = "not_scored"
                row["canonical_gold_match"] = None
        result["parsers"][parser_id] = row
    return result


def _blind_stratum(record: dict[str, Any]) -> str:
    left, right = (record["parsers"][name] for name in PARSERS)
    if not left["available"] or not right["available"]:
        unavailable = [name for name in PARSERS if not record["parsers"][name]["available"]]
        return "parser_unavailable:" + ",".join(unavailable)
    statuses = [row["extraction_status"] for row in (left, right)]
    if statuses[0] != statuses[1]:
        return "status:" + "/".join(statuses)
    if left["canonical_value"] != right["canonical_value"]:
        return "canonical_value_disagreement"
    if statuses[0] != "extracted":
        return "shared_" + statuses[0]
    return "agreement"


def export_blind_disagreements(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze an outcome-independent sample with a strict reviewer allowlist."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        gid = record["generation_id"]
        if gid in seen:
            raise ValueError("Duplicate generation in blind review input")
        seen.add(gid)
        groups[_blind_stratum(record)].append(record)
    reviewer, private, strata = [], [], []
    for stratum, members in sorted(groups.items()):
        ordered = sorted(
            members,
            key=lambda item: identity_hash(
                "rebuttal-answer-blind-order/v2",
                {
                    "seed": SAMPLING_POLICY["seed"],
                    "stratum": stratum,
                    "generation_id": item["generation_id"],
                },
            ),
        )
        cap = SAMPLING_POLICY[
            "agreement_cap" if stratum == "agreement" else "disagreement_cap_per_stratum"
        ]
        for record in ordered[:cap]:
            blind_id = identity_hash(
                "rebuttal-answer-blind-id/v2",
                {"seed": SAMPLING_POLICY["seed"], "generation_id": record["generation_id"]},
            )
            reviewer.append(
                {
                    "schema_version": "rebuttal-answer-blind/v2",
                    "blind_id": blind_id,
                    "raw_text": record["raw_text"],
                }
            )
            private.append(
                {
                    "schema_version": "rebuttal-answer-blind-private/v2",
                    "blind_id": blind_id,
                    "sampling_stratum": stratum,
                    **{
                        key: record.get(key)
                        for key in (
                            "generation_id",
                            "pair_id",
                            "arm",
                            "window",
                            "model_id",
                            "task",
                            "gold",
                            "canonical_gold",
                            "cohort_membership",
                            "parsers",
                        )
                    },
                }
            )
        strata.append(
            {
                "stratum": stratum,
                "eligible_count": len(ordered),
                "selected_count": min(cap, len(ordered)),
                "unaudited_count": len(ordered) - min(cap, len(ordered)),
                "cap": cap,
            }
        )
    return {
        "annotation_records": sorted(reviewer, key=lambda item: item["blind_id"]),
        "private_mapping": sorted(private, key=lambda item: item["blind_id"]),
        "sampling_policy": {
            **SAMPLING_POLICY,
            "strata": strata,
            "selection_frozen_before_judgments": True,
        },
    }


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _ref(name: str, raw: bytes) -> dict[str, str]:
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        cells = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (dict, list)):
                value = canonical_json(value)
            if value is None:
                value = "NA"
            if isinstance(value, str) and re.match(r"^[\s\x00-\x20]*[=+@-]", value):
                value = "'" + value
            cells[field] = value
        writer.writerow(cells)
    return handle.getvalue().encode("utf-8")


def _publish_files(output: Path, files: dict[str, bytes], bundle: Any) -> None:
    from typo_cot.experiments.rebuttal_v2.artifacts import revalidate_inputs

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.answer-audit-", dir=output.parent))
    try:
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        revalidate_inputs(bundle)
        if output.exists():
            output.rmdir()
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def _memberships(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return pair["cohort_membership"] or [
        {"cohort_id": None, "membership": "unknown", "source_ref": None}
    ]


def _slot_membership(pair: dict[str, Any], link: dict[str, Any]) -> str:
    expected = pair.get("expected_generations")
    if expected is None:
        return "unknown"
    return (
        "expected"
        if any(item["arm"] == link["arm"] and item["window"] == link["window"] for item in expected)
        else "extra"
    )


def _failure_artifact(
    parser: str, reports: dict[str, Any], descriptor: dict[str, Any]
) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = {}
    member_ids, extra_ids, unknown_ids = set(), set(), set()
    states = []
    for row in reports["baseline"]:
        state = row["primary_state" if parser == PRIMARY else "proxy_state"]
        states.append(
            {
                "pair_id": row["pair_id"],
                "cohort_membership": row["cohort_membership"],
                "state": state,
            }
        )
        for membership in _memberships(row):
            identity = {
                "cohort_id": membership["cohort_id"],
                "membership": membership["membership"],
            }
            key = canonical_json(identity)
            group = strata.setdefault(
                key,
                {
                    **identity,
                    "eligible_pair_ids": [],
                    "ineligible_pair_ids": [],
                    "unknown_pair_ids": [],
                },
            )
            group[f"{state}_pair_ids"].append(row["pair_id"])
            if state == "eligible":
                {"member": member_ids, "extra": extra_ids, "unknown": unknown_ids}[
                    membership["membership"]
                ].add(row["pair_id"])
    return {
        "schema_version": "rebuttal-archived-baseline-failure-ids/v2",
        "parser_id": parser,
        "score_rule": descriptor["score_rule"],
        "comparison_kind": "archived-baselines-not-fresh",
        "pair_ids": sorted(member_ids),
        "pair_ids_policy": "eligible members of an explicitly recovered original cohort only; consult cohort_strata for each cohort",
        "extra_pair_ids": sorted(extra_ids),
        "unverified_pair_ids": sorted(unknown_ids),
        "cohort_strata": [strata[key] for key in sorted(strata)],
        "pair_states": states,
    }


def _eligibility(
    clean: dict[str, Any] | None, typo: dict[str, Any] | None, *, key: str = "gold_match"
) -> str:
    if clean is None or typo is None or clean.get(key) is None or typo.get(key) is None:
        return "unknown"
    return "eligible" if clean[key] is True and typo[key] is False else "ineligible"


def _reports(
    pairs: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_slot = {
        (row["pair_id"], row["arm"], canonical_json(row["window"]), row["parser_id"]): row
        for row in audit_rows
    }
    by_id = {row["pair_id"]: row for row in pairs}
    baseline, failures = [], {name: [] for name in PARSERS}
    for pair in pairs:
        states, canonical_states = {}, {}
        for parser in PARSERS:
            clean = by_slot.get((pair["pair_id"], "clean", "null", parser))
            typo = by_slot.get((pair["pair_id"], "typo", "null", parser))
            states[parser] = _eligibility(clean, typo)
            canonical_states[parser] = _eligibility(clean, typo, key="canonical_gold_match")
            if states[parser] == "eligible":
                failures[parser].append(pair["pair_id"])
        baseline.append(
            {
                "pair_id": pair["pair_id"],
                "cohort_membership": pair["cohort_membership"],
                "source_kind": pair["source_kind"],
                "primary_state": states[PRIMARY],
                "proxy_state": states[PROXY],
                "proxy_canonical_state": canonical_states[PROXY],
                "comparison_kind": "archived-baselines-not-fresh",
            }
        )
    transitions = []
    for pair in pairs:
        for link in pair["archived_generations"]:
            values = {}
            for parser in PARSERS:
                row = by_slot.get(
                    (pair["pair_id"], link["arm"], canonical_json(link["window"]), parser)
                )
                values[parser] = (
                    None
                    if row is None
                    else {
                        key: row[key]
                        for key in (
                            "extraction_status",
                            "raw_value",
                            "canonical_value",
                            "score_status",
                            "gold_match",
                            "score_rule",
                            "canonical_score_status",
                            "canonical_gold_match",
                        )
                    }
                )
            transitions.append(
                {
                    "pair_id": pair["pair_id"],
                    "generation_id": link["generation_id"],
                    "arm": link["arm"],
                    "window": link["window"],
                    "source_slot_membership": _slot_membership(pair, link),
                    "cohort_membership": pair["cohort_membership"],
                    "primary": values[PRIMARY],
                    "proxy": values[PROXY],
                }
            )
    aggregates: dict[str, dict[str, Any]] = {}
    for cov in coverage:
        slot, parser = cov["source_slot"], cov["parser_id"]
        pair = by_id[slot["pair_id"]]
        scored = by_slot.get((pair["pair_id"], slot["arm"], canonical_json(slot["window"]), parser))
        for membership in _memberships(pair):
            identity = {
                "cohort_id": membership["cohort_id"],
                "membership": membership["membership"],
                "task": pair["task"],
                "model_id": pair["model_id"],
                "source_kind": pair["source_kind"],
                "arm": slot["arm"],
                "window": slot["window"],
                "parser_id": parser,
                "source_slot_membership": cov["source_slot_membership"],
            }
            key = canonical_json(identity)
            if key not in aggregates:
                aggregates[key] = {
                    **identity,
                    "known_slot_count": 0,
                    "present_output_count": 0,
                    "scoring_status_counts": Counter(),
                    "extraction_status_counts": Counter(),
                    "termination_counts": Counter(),
                    "correct_count": 0,
                    "incorrect_count": 0,
                    "legacy_correct_count": 0,
                    "legacy_scored_count": 0,
                    "recovery_count": 0,
                    "recovery_comparable_count": 0,
                    "recovery_unknown_count": 0,
                }
            entry = aggregates[key]
            entry["known_slot_count"] += 1
            entry["scoring_status_counts"][cov["scoring_status"]] += 1
            if cov["generation_ref"] is not None:
                entry["present_output_count"] += 1
                entry["termination_counts"][generations[cov["generation_id"]]["termination"]] += 1
            if scored is not None:
                entry["extraction_status_counts"][scored["extraction_status"]] += 1
                entry["correct_count"] += int(scored["gold_match"] is True)
                entry["incorrect_count"] += int(scored["gold_match"] is False)
                entry["legacy_correct_count"] += int(scored["legacy_gold_match"] is True)
                entry["legacy_scored_count"] += int(scored["legacy_gold_match"] is not None)
            if slot["arm"] not in {"clean", "typo"}:
                typo = by_slot.get((pair["pair_id"], "typo", "null", parser))
                if (
                    scored is None
                    or typo is None
                    or scored["gold_match"] is None
                    or typo["gold_match"] is None
                ):
                    entry["recovery_unknown_count"] += 1
                else:
                    entry["recovery_comparable_count"] += 1
                    entry["recovery_count"] += int(
                        typo["gold_match"] is False and scored["gold_match"] is True
                    )
    return {
        "baseline": baseline,
        "transitions": transitions,
        "failures": failures,
        "aggregates": [aggregates[key] for key in sorted(aggregates)],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _add_rates(
    reports: dict[str, Any], pairs: list[dict[str, Any]], cohorts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rates are diagnostic views; missing slots never become model errors."""
    cohort_by_id = {row["cohort_id"]: row for row in cohorts}
    for row in reports["aggregates"]:
        scored = row["scoring_status_counts"].get("scored", 0)
        executed = sum(row["extraction_status_counts"].values())
        present = row["present_output_count"]
        cohort = cohort_by_id.get(row["cohort_id"], {})
        original_n = cohort.get("expected_count")
        original_complete = (
            row["membership"] == "member"
            and row["source_slot_membership"] == "expected"
            and cohort.get("expected_source_pair_keys") is not None
            and cohort.get("missing_count") == 0
            and cohort.get("recovered_count") == original_n
            and isinstance(original_n, int)
            and original_n > 0
            and original_n == row["known_slot_count"] == scored
        )
        row["rates"] = {
            "answer_success": {
                "value": _rate(row["correct_count"], scored),
                "denominator": scored,
                "scope": "scored outputs only; missing gold/output/parser excluded",
            },
            "extraction": {
                "values": {
                    status: _rate(row["extraction_status_counts"].get(status, 0), executed)
                    for status in ("extracted", "unextractable", "ambiguous")
                },
                "denominator": executed,
                "scope": "executed parser outputs only",
            },
            "termination": {
                "values": {
                    status: _rate(row["termination_counts"].get(status, 0), present)
                    for status in ("eos", "length-cap", "unknown")
                },
                "denominator": present,
                "scope": "present raw outputs, including parser-unavailable outputs",
            },
            "scoring_coverage": {
                "value": _rate(scored, row["known_slot_count"]),
                "denominator": row["known_slot_count"],
                "scope": "known source slots in this cohort and slot-membership stratum",
            },
            "recovery": {
                "value": _rate(row["recovery_count"], row["recovery_comparable_count"]),
                "denominator": row["recovery_comparable_count"],
                "scope": "patch/typo pairs with both scores available; clean/typo arms have no recovery rate",
            },
            "full_original_cohort_answer_success": {
                "value": _rate(row["correct_count"], original_n) if original_complete else None,
                "denominator": original_n,
                "scope": "available only when the entire exact original cohort has this expected slot and is scored",
                "complete": original_complete,
            },
        }
    by_id = {pair["pair_id"]: pair for pair in pairs}
    groups: dict[str, dict[str, Any]] = {}
    for transition in reports["transitions"]:
        pair = by_id[transition["pair_id"]]
        for membership in _memberships(pair):
            identity = {
                "cohort_id": membership["cohort_id"],
                "membership": membership["membership"],
                "task": pair["task"],
                "model_id": pair["model_id"],
                "source_kind": pair["source_kind"],
                "arm": transition["arm"],
                "window": transition["window"],
                "source_slot_membership": transition["source_slot_membership"],
            }
            key = canonical_json(identity)
            group = groups.setdefault(
                key,
                {
                    **identity,
                    "known_slot_count": 0,
                    "paired_scored_count": 0,
                    "unknown_count": 0,
                    "primary_correct_count": 0,
                    "proxy_correct_count": 0,
                },
            )
            group["known_slot_count"] += 1
            left, right = transition["primary"], transition["proxy"]
            if (
                left is None
                or right is None
                or left["gold_match"] is None
                or right["gold_match"] is None
            ):
                group["unknown_count"] += 1
            else:
                group["paired_scored_count"] += 1
                group["primary_correct_count"] += int(left["gold_match"])
                group["proxy_correct_count"] += int(right["gold_match"])
    for group in groups.values():
        group["primary_minus_proxy_success_count"] = (
            group["primary_correct_count"] - group["proxy_correct_count"]
        )
        group["primary_minus_proxy_success_rate"] = _rate(
            group["primary_minus_proxy_success_count"], group["paired_scored_count"]
        )
        group["denominator_scope"] = (
            "identical raw generations with both parser-specific scores available, not the full original cohort"
        )
        group["primary_score_rule"] = "exact-canonical-answer/v2"
        group["proxy_score_rule"] = "fallback.answers_equal/public-v1"
    return [groups[key] for key in sorted(groups)]


def run_answer_audit(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Verify a PR01 manifest and atomically publish parser-sensitivity artifacts."""
    from typo_cot.experiments.rebuttal_v2.artifacts import load_manifest

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    bundle = load_manifest(manifest_path)
    code_identity = _code_identity(bundle.snapshots)
    parsers = _parser_descriptors()
    generations, generation_refs = {}, {}
    for slot in bundle.slots:
        if slot.generation is not None:
            gid = slot.generation["generation_id"]
            generations[gid] = {**slot.generation, "raw_text": slot.raw_text}
            generation_refs[gid] = {
                "artifact": {
                    "path": str(slot.generation_path),
                    "sha256": bundle.snapshots[slot.generation_path],
                },
                "record_id": gid,
            }
    audits, coverage, comparisons = [], [], []
    for pair in bundle.pairs:
        pair_ref = {"artifact": bundle.manifest_ref, "record_id": pair["pair_id"]}
        for link in pair["archived_generations"]:
            generation = generations.get(link["generation_id"])
            comparison = None
            if generation is not None and isinstance(generation.get("raw_text"), str):
                joined = {**pair, **generation}
                comparison = {**joined, **compare_parsers(joined)}
                comparisons.append(comparison)
            for parser_id in PARSERS:
                slot = {"pair_id": pair["pair_id"], "arm": link["arm"], "window": link["window"]}
                descriptor = parsers[parser_id]
                cov_identity = {
                    "plan_id": None,
                    "source_slot": slot,
                    "parser_id": parser_id,
                    "parser_version": descriptor["parser_version"],
                }
                cov = {
                    "schema_version": "rebuttal-scoring-coverage/v2",
                    "coverage_id": identity_hash("rebuttal-scoring-coverage/v2", cov_identity),
                    **cov_identity,
                    "source_slot_membership": _slot_membership(pair, link),
                    "manifest_ref": pair_ref,
                    "generation_id": link["generation_id"],
                    "generation_ref": None,
                    "score_ref": None,
                    "scoring_status": "output_missing",
                    "reason_codes": list(link["reason_codes"]),
                }
                coverage.append(cov)
                if comparison is None:
                    cov["reason_codes"] = sorted(
                        set(cov["reason_codes"] + ["raw_generation_unavailable"])
                    )
                    continue
                cov["generation_ref"] = generation_refs[link["generation_id"]]
                result = comparison["parsers"][parser_id]
                if not result["available"]:
                    cov["scoring_status"] = "parser_unavailable"
                    cov["reason_codes"] = sorted(set(cov["reason_codes"] + result["reason_codes"]))
                    continue
                audit_id = identity_hash(
                    "rebuttal-answer-audit/v2",
                    {
                        "generation_id": link["generation_id"],
                        "parser_id": parser_id,
                        "parser_version": descriptor["parser_version"],
                        "parser_code_sha256": descriptor["parser_code_sha256"],
                    },
                )
                row = {
                    "schema_version": "rebuttal-answer-audit/v2",
                    "audit_id": audit_id,
                    "generation_id": link["generation_id"],
                    "generation_ref": cov["generation_ref"],
                    "pair_id": pair["pair_id"],
                    "manifest_ref": pair_ref,
                    "arm": link["arm"],
                    "window": link["window"],
                    "source_slot_membership": _slot_membership(pair, link),
                    "cohort_membership": pair["cohort_membership"],
                    "source_kind": pair["source_kind"],
                    "model_id": pair["model_id"],
                    "valid_labels": pair["valid_labels"],
                    "raw_text_sha256": generation["raw_text_sha256"],
                    "gold_ref": pair_ref,
                    "gold_canonicalization_version": pair["gold_canonicalization_version"],
                    **descriptor,
                    **{key: value for key, value in result.items() if key != "available"},
                }
                audits.append(row)
                cov["scoring_status"] = row["score_status"]
                cov["reason_codes"] = sorted(set(cov["reason_codes"] + row["reason_codes"]))
                cov["_audit_id"] = audit_id
    audits.sort(key=lambda item: item["audit_id"])
    coverage.sort(key=lambda item: item["coverage_id"])
    files = {
        "protocol.json": bundle.protocol_bytes,
        "answer_audit_records.jsonl": _jsonl_bytes(audits),
    }
    score_ref = _ref("answer_audit_records.jsonl", files["answer_audit_records.jsonl"])
    for row in coverage:
        audit_id = row.pop("_audit_id", None)
        if audit_id is not None:
            row["score_ref"] = {"artifact": score_ref, "record_id": audit_id}
    files["scoring_coverage.jsonl"] = _jsonl_bytes(coverage)
    reports = _reports(bundle.pairs, audits, coverage, generations)
    paired_differences = _add_rates(reports, bundle.pairs, bundle.source_audit["cohorts"])
    files["parser_transition_table.json"] = _json_bytes(reports["transitions"])
    files["parser_transition_table.csv"] = _csv_bytes(
        reports["transitions"],
        [
            "pair_id",
            "generation_id",
            "arm",
            "window",
            "source_slot_membership",
            "cohort_membership",
            "primary",
            "proxy",
        ],
    )
    files["baseline_transition_table.json"] = _json_bytes(reports["baseline"])
    files["baseline_transition_table.csv"] = _csv_bytes(
        reports["baseline"],
        [
            "pair_id",
            "cohort_membership",
            "source_kind",
            "primary_state",
            "proxy_state",
            "proxy_canonical_state",
            "comparison_kind",
        ],
    )
    for parser, filename in (
        (PRIMARY, "primary_parser_archived_baseline_failure_ids.json"),
        (PROXY, "public_v1_proxy_archived_baseline_failure_ids.json"),
    ):
        files[filename] = _json_bytes(_failure_artifact(parser, reports, parsers[parser]))
    blind = export_blind_disagreements(comparisons)
    files["parser_disagreement_blind.jsonl"] = _jsonl_bytes(blind["annotation_records"])
    files["parser_disagreement_private.jsonl"] = _jsonl_bytes(blind["private_mapping"])
    files["blind_sampling_policy.json"] = _json_bytes(blind["sampling_policy"])
    summary = {
        "schema_version": "rebuttal-parser-audit-summary/v2",
        "experiment_id": "R1",
        "run_status": "complete",
        "experiment_status": "complete",
        "model_execution_performed": False,
        "parser_independent_semantic_claim": "inconclusive",
        "machine_score_claim": "answer success under the fixed extraction rule",
        "parsers": parsers,
        "source_cohort_coverage": bundle.source_audit["cohorts"],
        "pair_count": len(bundle.pairs),
        "available_generation_count": len(comparisons),
        "audit_record_count": len(audits),
        "scoring_coverage_count": len(coverage),
        "scoring_status_counts": dict(Counter(row["scoring_status"] for row in coverage)),
        "expected_grid_unknown_pair_ids": sorted(
            pair["pair_id"] for pair in bundle.pairs if pair.get("expected_generations") is None
        ),
        "arm_parser_counts": reports["aggregates"],
        "paired_parser_differences": paired_differences,
        "blind_sampling": blind["sampling_policy"],
        "reason_codes": [
            "historical_parser_code_unavailable_public_proxy_used",
            "no_independent_complete_human_scoring",
            "archived_baselines_not_fresh",
        ],
    }
    files["parser_audit_summary.json"] = _json_bytes(summary)
    run = {
        "schema_version": "rebuttal-run/v2",
        "command": "answer-audit",
        "mode": "cpu-audit",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "invocation": {
            "manifest": str(Path(manifest_path).resolve()),
            "output_dir": str(Path(output_dir).resolve()),
        },
        "code_identity": code_identity,
        "manifest_metadata_ref": bundle.metadata_ref,
        "protocol_revision": bundle.protocol["protocol_revision"],
        "protocol_sha256": canonical_sha256(bundle.protocol),
        "input_refs": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(bundle.snapshots.items(), key=lambda item: str(item[0]))
        ],
        "config_refs": [
            _ref("protocol.json", files["protocol.json"]),
            _ref("blind_sampling_policy.json", files["blind_sampling_policy.json"]),
        ],
        "output_refs": {name: _ref(name, raw) for name, raw in sorted(files.items())},
        "run_status": "complete",
        "experiment_status": "complete",
        "experiment_id": "R1",
        "expected_ids": [row["coverage_id"] for row in coverage],
        "completed_ids": [
            row["coverage_id"]
            for row in coverage
            if row["scoring_status"] in {"scored", "not_scored"}
        ],
        "failed_ids": [],
        "missing_ids": [
            row["coverage_id"]
            for row in coverage
            if row["scoring_status"] not in {"scored", "not_scored"}
        ],
        "reason_codes": summary["reason_codes"],
        "model_execution_performed": False,
    }
    files["run.json"] = _json_bytes(run)
    _publish_files(output, files, bundle)
    return run

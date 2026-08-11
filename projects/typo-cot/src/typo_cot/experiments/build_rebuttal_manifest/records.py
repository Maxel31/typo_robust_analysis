"""Strict schemas for prepared inputs and normalized rebuttal pair records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from typo_cot.data.matched_donors import MatchedDonorCandidate, plan_cyclic_derangement
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.build_rebuttal_manifest.planning import (
    WindowSplitCandidate,
    plan_strict_offset_control,
    plan_window_split,
)
from typo_cot.experiments.build_rebuttal_manifest.protocol import (
    REBUTTAL_MANIFEST_PROTOCOL,
    REBUTTAL_SETTINGS,
    RebuttalSetting,
)
from typo_cot.experiments.catalog import PAPER_SHA256

PAIR_MANIFEST_SCHEMA = "rebuttal-pair-manifest/v1"
REBUTTAL_RUN_SCHEMA = "build-rebuttal-manifest-run/v1"
REBUTTAL_COHORT_SCHEMA = "rebuttal-cohort-ids/v1"
REBUTTAL_AUDIT_SCHEMA = "rebuttal-source-audit/v1"
TARGET_RULES = REBUTTAL_MANIFEST_PROTOCOL.target_rules
_TARGET_RULE_ORDER = {name: index for index, name in enumerate(TARGET_RULES)}
_SETTING_KEYS = {setting.key for setting in REBUTTAL_SETTINGS}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_loads(text: str, *, context: str) -> object:
    """Load standards-compliant JSON while rejecting duplicate object keys."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {context}: {exc}") from exc


def load_json_object(path: Path) -> dict[str, object]:
    payload = strict_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def iter_jsonl_objects(path: Path) -> Iterator[tuple[int, str, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            payload = strict_loads(line, context=f"{path}:{line_number}")
            if not isinstance(payload, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield line_number, line, payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _sha256_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _termination(value: object, *, field: str) -> str:
    termination = _nonempty_string(value, field=field)
    if termination not in {"eos", "length-cap"}:
        raise ValueError(f"{field} must be 'eos' or 'length-cap'")
    return termination


def _span(value: object, *, field: str, text: str) -> tuple[int, int]:
    payload = _mapping(value, field=field)
    start, end = payload.get("start"), payload.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(f"{field} must be a valid half-open character span")
    return start, end


def _list_span(value: object, *, field: str, upper_bound: int) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-integer half-open span")
    start, end = value
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= upper_bound
    ):
        raise ValueError(f"{field} must be a valid half-open span")
    return start, end


def _token_indices(
    value: object,
    *,
    field: str,
    token_count: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a token-index list")
    indices = tuple(value)
    if not indices and not allow_empty:
        raise ValueError(f"{field} must be a non-empty token-index list")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < token_count
        for index in indices
    ):
        raise ValueError(f"{field} contains an out-of-range token index")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError(f"{field} must be strictly increasing and unique")
    return indices


def pair_id_for(*, model: str, task: str, target_rule: str, sample_id: str) -> str:
    """Return the stable identity used by every downstream cohort artifact."""

    return canonical_sha256(
        {
            "model": model,
            "sample_id": sample_id,
            "target_rule": target_rule,
            "task": task,
        }
    )


def _side(
    record: Mapping[str, object],
    *,
    name: str,
    task: str,
    gold_answer: str,
) -> dict[str, object]:
    payload = _mapping(record.get(name), field=name)
    prompt = _nonempty_string(payload.get("prompt"), field=f"{name}.prompt")
    prompt_token_count = _positive_int(
        payload.get("prompt_token_count"),
        field=f"{name}.prompt_token_count",
    )
    continuation = payload.get("continuation")
    if not isinstance(continuation, str):
        raise ValueError(f"{name}.continuation must be a string")
    answer = _mapping(payload.get("answer"), field=f"{name}.answer")
    stored_correct = answer.get("is_correct")
    if not isinstance(stored_correct, bool):
        raise ValueError(f"{name}.answer.is_correct must be boolean")
    stored_extracted = answer.get("is_extracted")
    if not isinstance(stored_extracted, bool):
        raise ValueError(f"{name}.answer.is_extracted must be boolean")
    stored_value = answer.get("value")
    if not isinstance(stored_value, str):
        raise ValueError(f"{name}.answer.value must be a string")
    termination = _termination(payload.get("termination"), field=f"{name}.termination")
    extracted = extract_with_fallback(
        continuation,
        benchmark=task,
        correct_answer=gold_answer,
        allow_positional=termination == "eos",
    )
    if extracted.value != stored_value:
        raise ValueError(f"{name}.answer.value differs from deterministic re-extraction")
    if extracted.is_correct != stored_correct:
        raise ValueError(f"{name}.answer.is_correct differs from deterministic re-extraction")
    if extracted.is_extracted != stored_extracted:
        raise ValueError(f"{name}.answer.is_extracted differs from deterministic re-extraction")
    editable_span = _span(
        payload.get("editable_prompt_span"),
        field=f"{name}.editable_prompt_span",
        text=prompt,
    )
    return {
        "prompt": prompt,
        "prompt_token_count": prompt_token_count,
        "continuation": continuation,
        "answer": extracted.value,
        "correct": extracted.is_correct,
        "termination": termination,
        "editable_span": editable_span,
    }


def normalize_prepared_pair(
    record: Mapping[str, object],
    *,
    source_record_sha256: str,
    prepared_pairs_path: str,
    prepared_pairs_sha256: str,
    prepared_run_sha256: str,
    model_revision: str,
) -> dict[str, object]:
    """Validate and normalize one ``prepare-edited-pairs/v1`` source record."""

    if record.get("schema_version") != "prepare-edited-pairs/v1":
        raise ValueError("prepared pair has an unknown schema_version")
    sample_id = _nonempty_string(record.get("sample_id"), field="sample_id")
    model = _nonempty_string(record.get("model"), field="model")
    task = _nonempty_string(record.get("benchmark"), field="benchmark")
    if task not in {"gsm8k", "mmlu"}:
        raise ValueError(f"rebuttal pair task must be gsm8k or mmlu, got {task!r}")
    target_rule = _nonempty_string(record.get("targeting"), field="targeting")
    if target_rule not in TARGET_RULES:
        raise ValueError(f"unsupported target rule: {target_rule!r}")
    if (
        record.get("seed") != REBUTTAL_MANIFEST_PROTOCOL.source_seed
        or record.get("num_edits_requested") != REBUTTAL_MANIFEST_PROTOCOL.requested_edits
    ):
        raise ValueError("rebuttal pair seed or requested edit count differs from the protocol")
    gold_answer = _nonempty_string(record.get("gold_answer"), field="gold_answer")
    clean = _side(record, name="clean", task=task, gold_answer=gold_answer)
    typo = _side(record, name="edited", task=task, gold_answer=gold_answer)

    raw_words = record.get("aligned_words")
    if not isinstance(raw_words, list):
        raise ValueError("aligned_words must be a list")
    if record.get("num_aligned_words") != len(raw_words):
        raise ValueError("num_aligned_words differs from aligned_words")
    edits: list[dict[str, object]] = []
    clean_final_tokens: list[int] = []
    typo_final_tokens: list[int] = []
    all_clean_tokens: list[int] = []
    all_typo_tokens: list[int] = []
    for index, raw_word in enumerate(raw_words):
        word = _mapping(raw_word, field=f"aligned_words[{index}]")
        clean_span = _span(
            word.get("clean_prompt_span"),
            field=f"aligned_words[{index}].clean_prompt_span",
            text=str(clean["prompt"]),
        )
        typo_span = _span(
            word.get("edited_prompt_span"),
            field=f"aligned_words[{index}].edited_prompt_span",
            text=str(typo["prompt"]),
        )
        clean_text = _nonempty_string(
            word.get("clean_text"),
            field=f"aligned_words[{index}].clean_text",
        )
        typo_text = _nonempty_string(
            word.get("edited_text"),
            field=f"aligned_words[{index}].edited_text",
        )
        if str(clean["prompt"])[slice(*clean_span)] != clean_text:
            raise ValueError(f"aligned_words[{index}] clean text differs from its span")
        if str(typo["prompt"])[slice(*typo_span)] != typo_text:
            raise ValueError(f"aligned_words[{index}] typo text differs from its span")
        clean_tokens = _token_indices(
            word.get("clean_token_indices"),
            field=f"aligned_words[{index}].clean_token_indices",
            token_count=int(clean["prompt_token_count"]),
        )
        typo_tokens = _token_indices(
            word.get("edited_token_indices"),
            field=f"aligned_words[{index}].edited_token_indices",
            token_count=int(typo["prompt_token_count"]),
        )
        if word.get("clean_final_token") != clean_tokens[-1]:
            raise ValueError(f"aligned_words[{index}].clean_final_token differs")
        if word.get("edited_final_token") != typo_tokens[-1]:
            raise ValueError(f"aligned_words[{index}].edited_final_token differs")
        clean_final_tokens.append(clean_tokens[-1])
        typo_final_tokens.append(typo_tokens[-1])
        all_clean_tokens.extend(clean_tokens)
        all_typo_tokens.extend(typo_tokens)
        edits.append(
            {
                "clean_word": clean_text,
                "typo_word": typo_text,
                "clean_char_span": list(clean_span),
                "typo_char_span": list(typo_span),
                "clean_token_span": [clean_tokens[0], clean_tokens[-1] + 1],
                "typo_token_span": [typo_tokens[0], typo_tokens[-1] + 1],
                "clean_token_indices": list(clean_tokens),
                "typo_token_indices": list(typo_tokens),
                "clean_word_final_token": clean_tokens[-1],
                "typo_word_final_token": typo_tokens[-1],
            }
        )
    if len(clean_final_tokens) != len(set(clean_final_tokens)):
        raise ValueError("clean aligned words have duplicate final-token coordinates")
    if len(typo_final_tokens) != len(set(typo_final_tokens)):
        raise ValueError("typo aligned words have duplicate final-token coordinates")

    return {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "paper_sha256": PAPER_SHA256,
        "pair_id": pair_id_for(
            model=model,
            task=task,
            target_rule=target_rule,
            sample_id=sample_id,
        ),
        "sample_id": sample_id,
        "task": task,
        "model": model,
        "target_rule": target_rule,
        "gold_answer": gold_answer,
        "clean_text": clean["prompt"],
        "typo_text": typo["prompt"],
        "clean_continuation": clean["continuation"],
        "typo_continuation": typo["continuation"],
        "clean_termination": clean["termination"],
        "typo_termination": typo["termination"],
        "clean_answer": clean["answer"],
        "typo_answer": typo["answer"],
        "clean_correct": clean["correct"],
        "typo_correct": typo["correct"],
        "clean_prompt_token_count": clean["prompt_token_count"],
        "typo_prompt_token_count": typo["prompt_token_count"],
        "number_of_aligned_words": len(edits),
        "edits": edits,
        "_planning": {
            "clean_final_tokens": clean_final_tokens,
            "typo_final_tokens": typo_final_tokens,
            "clean_edited_tokens": sorted(set(all_clean_tokens)),
            "typo_edited_tokens": sorted(set(all_typo_tokens)),
        },
        "source": {
            "prepared_pairs_path": prepared_pairs_path,
            "prepared_pairs_sha256": prepared_pairs_sha256,
            "prepared_run_sha256": prepared_run_sha256,
            "source_record_sha256": source_record_sha256,
            "model_revision": model_revision,
        },
    }


def _validate_manifest_edit(
    value: object,
    *,
    context: str,
    clean_text: str,
    typo_text: str,
    clean_token_count: int,
    typo_token_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    edit = _mapping(value, field=context)
    clean_word = _nonempty_string(edit.get("clean_word"), field=f"{context}.clean_word")
    typo_word = _nonempty_string(edit.get("typo_word"), field=f"{context}.typo_word")
    clean_char_span = _list_span(
        edit.get("clean_char_span"),
        field=f"{context}.clean_char_span",
        upper_bound=len(clean_text),
    )
    typo_char_span = _list_span(
        edit.get("typo_char_span"),
        field=f"{context}.typo_char_span",
        upper_bound=len(typo_text),
    )
    if clean_text[slice(*clean_char_span)] != clean_word:
        raise ValueError(f"{context}: clean word differs from its character span")
    if typo_text[slice(*typo_char_span)] != typo_word:
        raise ValueError(f"{context}: typo word differs from its character span")
    clean_tokens = _token_indices(
        edit.get("clean_token_indices"),
        field=f"{context}.clean_token_indices",
        token_count=clean_token_count,
    )
    typo_tokens = _token_indices(
        edit.get("typo_token_indices"),
        field=f"{context}.typo_token_indices",
        token_count=typo_token_count,
    )
    if list(clean_tokens[:1] + (clean_tokens[-1] + 1,)) != edit.get("clean_token_span"):
        raise ValueError(f"{context}: clean token span differs from its indices")
    if list(typo_tokens[:1] + (typo_tokens[-1] + 1,)) != edit.get("typo_token_span"):
        raise ValueError(f"{context}: typo token span differs from its indices")
    if edit.get("clean_word_final_token") != clean_tokens[-1]:
        raise ValueError(f"{context}: clean final-token coordinate differs")
    if edit.get("typo_word_final_token") != typo_tokens[-1]:
        raise ValueError(f"{context}: typo final-token coordinate differs")
    return clean_tokens, typo_tokens


def _validate_manifest_record(record: dict[str, object], *, context: str) -> None:
    if record.get("schema_version") != PAIR_MANIFEST_SCHEMA:
        raise ValueError(f"{context}: unknown rebuttal pair schema")
    if record.get("paper_sha256") != PAPER_SHA256:
        raise ValueError(f"{context}: paper SHA-256 differs")
    if record.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256():
        raise ValueError(f"{context}: manifest protocol SHA-256 differs")
    identity = {
        "model": _nonempty_string(record.get("model"), field=f"{context}.model"),
        "sample_id": _nonempty_string(record.get("sample_id"), field=f"{context}.sample_id"),
        "target_rule": _nonempty_string(record.get("target_rule"), field=f"{context}.target_rule"),
        "task": _nonempty_string(record.get("task"), field=f"{context}.task"),
    }
    expected_pair_id = canonical_sha256(identity)
    if record.get("pair_id") != expected_pair_id:
        raise ValueError(f"{context}: pair_id differs from canonical identity")
    if (identity["model"], identity["task"]) not in _SETTING_KEYS:
        raise ValueError(f"{context}: model/task is outside the six-setting contract")
    if identity["target_rule"] not in TARGET_RULES:
        raise ValueError(f"{context}: target rule is outside the manifest contract")

    clean_text = _nonempty_string(record.get("clean_text"), field=f"{context}.clean_text")
    typo_text = _nonempty_string(record.get("typo_text"), field=f"{context}.typo_text")
    for field in ("clean_continuation", "typo_continuation"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"{context}.{field} must be a string")
    gold_answer = _nonempty_string(record.get("gold_answer"), field=f"{context}.gold_answer")
    clean_correct = _boolean(record.get("clean_correct"), field=f"{context}.clean_correct")
    typo_correct = _boolean(record.get("typo_correct"), field=f"{context}.typo_correct")
    for prefix, stored_correct in (("clean", clean_correct), ("typo", typo_correct)):
        answer = record.get(f"{prefix}_answer")
        if not isinstance(answer, str):
            raise ValueError(f"{context}.{prefix}_answer must be a string")
        termination = _termination(
            record.get(f"{prefix}_termination"),
            field=f"{context}.{prefix}_termination",
        )
        extracted = extract_with_fallback(
            str(record[f"{prefix}_continuation"]),
            benchmark=identity["task"],
            correct_answer=gold_answer,
            allow_positional=termination == "eos",
        )
        if extracted.value != answer or extracted.is_correct != stored_correct:
            raise ValueError(f"{context}: {prefix} answer differs from deterministic re-extraction")
    clean_token_count = _positive_int(
        record.get("clean_prompt_token_count"),
        field=f"{context}.clean_prompt_token_count",
    )
    typo_token_count = _positive_int(
        record.get("typo_prompt_token_count"),
        field=f"{context}.typo_prompt_token_count",
    )
    aligned_count = _nonnegative_int(
        record.get("number_of_aligned_words"),
        field=f"{context}.number_of_aligned_words",
    )
    edits = record.get("edits")
    if not isinstance(edits, list) or len(edits) != aligned_count:
        raise ValueError(f"{context}: edits differ from number_of_aligned_words")
    clean_final_tokens: list[int] = []
    typo_final_tokens: list[int] = []
    clean_edited_tokens: list[int] = []
    typo_edited_tokens: list[int] = []
    for index, edit in enumerate(edits):
        clean_tokens, typo_tokens = _validate_manifest_edit(
            edit,
            context=f"{context}.edits[{index}]",
            clean_text=clean_text,
            typo_text=typo_text,
            clean_token_count=clean_token_count,
            typo_token_count=typo_token_count,
        )
        clean_final_tokens.append(clean_tokens[-1])
        typo_final_tokens.append(typo_tokens[-1])
        clean_edited_tokens.extend(clean_tokens)
        typo_edited_tokens.extend(typo_tokens)
    if len(clean_final_tokens) != len(set(clean_final_tokens)):
        raise ValueError(f"{context}: duplicate clean final-token coordinate")
    if len(typo_final_tokens) != len(set(typo_final_tokens)):
        raise ValueError(f"{context}: duplicate typo final-token coordinate")

    patch_eligible = clean_correct and aligned_count > 0
    cohorts = _mapping(record.get("cohorts"), field=f"{context}.cohorts")
    restoration = _boolean(
        cohorts.get("restoration"),
        field=f"{context}.cohorts.restoration",
    )
    if restoration and (not patch_eligible or typo_correct):
        raise ValueError(f"{context}: restoration cohort is outside prepared typo failures")
    expected_cohorts = {
        "full_clean_correct": clean_correct,
        "patch_eligible_clean_correct": patch_eligible,
        "alignment_ineligible_clean_correct": clean_correct and not patch_eligible,
        "harm": patch_eligible and typo_correct,
    }
    for field, expected in expected_cohorts.items():
        if _boolean(cohorts.get(field), field=f"{context}.cohorts.{field}") != expected:
            raise ValueError(f"{context}: cohort {field} differs from pair outcomes")
    selection = _boolean(
        cohorts.get("window_selection"),
        field=f"{context}.cohorts.window_selection",
    )
    evaluation = _boolean(
        cohorts.get("window_evaluation"),
        field=f"{context}.cohorts.window_evaluation",
    )
    if selection and evaluation:
        raise ValueError(f"{context}: window split memberships overlap")
    if restoration != (selection or evaluation):
        raise ValueError(f"{context}: window split does not partition restoration pairs")

    controls = _mapping(record.get("controls"), field=f"{context}.controls")
    correct = _mapping(controls.get("correct"), field=f"{context}.controls.correct")
    if _boolean(correct.get("valid"), field=f"{context}.controls.correct.valid") != bool(
        aligned_count
    ):
        raise ValueError(f"{context}: correct-coordinate validity differs")
    correct_source = _token_indices(
        correct.get("source_positions"),
        field=f"{context}.controls.correct.source_positions",
        token_count=clean_token_count,
        allow_empty=not aligned_count,
    )
    correct_destination = _token_indices(
        correct.get("destination_positions"),
        field=f"{context}.controls.correct.destination_positions",
        token_count=typo_token_count,
        allow_empty=not aligned_count,
    )
    if correct_source != tuple(clean_final_tokens) or correct_destination != tuple(
        typo_final_tokens
    ):
        raise ValueError(f"{context}: correct-coordinate plan differs from aligned edits")

    expected_offset = plan_strict_offset_control(
        clean_final_tokens,
        typo_final_tokens,
        sorted(set(clean_edited_tokens)),
        sorted(set(typo_edited_tokens)),
        clean_token_count,
        typo_token_count,
        offset=REBUTTAL_MANIFEST_PROTOCOL.offset_tokens,
    )
    offset = _mapping(controls.get("offset_2"), field=f"{context}.controls.offset_2")
    expected_offset_payload = {
        "valid": expected_offset.valid,
        "source_positions": list(expected_offset.source_positions),
        "destination_positions": list(expected_offset.destination_positions),
        "input_pairs": expected_offset.input_pairs,
        "offset_tokens": expected_offset.offset_tokens,
        "invalid_reason": expected_offset.invalid_reason,
        "validity_rule": "all-prompt-interior-non-edited-coordinates/v1",
    }
    for field, expected in expected_offset_payload.items():
        if offset.get(field) != expected:
            raise ValueError(f"{context}: offset plan field {field} differs")

    cross_item = _mapping(
        controls.get("cross_item"),
        field=f"{context}.controls.cross_item",
    )
    cross_valid = _boolean(
        cross_item.get("valid"),
        field=f"{context}.controls.cross_item.valid",
    )
    donor_pair_id = cross_item.get("donor_pair_id")
    if cross_valid:
        donor_digest = _sha256_digest(
            donor_pair_id,
            field=f"{context}.controls.cross_item.donor_pair_id",
        )
        if not restoration or donor_digest == expected_pair_id:
            raise ValueError(f"{context}: cross-item donor is invalid")
        if cross_item.get("invalid_reason") is not None:
            raise ValueError(f"{context}: valid cross-item donor has an invalid reason")
    elif donor_pair_id is not None:
        raise ValueError(f"{context}: invalid cross-item control names a donor")
    if cross_item.get("matching_rule") != (
        "task-model-target-rule-aligned-word-count-cyclic-derangement/v1"
    ):
        raise ValueError(f"{context}: cross-item matching rule differs")
    common_valid = _boolean(
        controls.get("common_valid"),
        field=f"{context}.controls.common_valid",
    )
    if common_valid != (restoration and expected_offset.valid and cross_valid):
        raise ValueError(f"{context}: common-valid membership differs")

    fixed = _mapping(record.get("fixed_window"), field=f"{context}.fixed_window")
    if (
        _boolean(
            fixed.get("in_reference_denominator"),
            field=f"{context}.fixed_window.in_reference_denominator",
        )
        != restoration
    ):
        raise ValueError(f"{context}: fixed-window denominator membership differs")
    fixed_selected = _boolean(
        fixed.get("in_selected_anchors"),
        field=f"{context}.fixed_window.in_selected_anchors",
    )
    if restoration and not fixed_selected:
        raise ValueError(f"{context}: paper denominator is outside selected anchors")
    exclusion_reason = fixed.get("exclusion_reason")
    if fixed_selected and not restoration:
        _nonempty_string(
            exclusion_reason,
            field=f"{context}.fixed_window.exclusion_reason",
        )
    elif exclusion_reason is not None:
        raise ValueError(f"{context}: non-excluded pair has a fixed-window exclusion reason")
    expected_fixed = {
        "direction": REBUTTAL_MANIFEST_PROTOCOL.fixed_direction if restoration else None,
        "window": REBUTTAL_MANIFEST_PROTOCOL.fixed_window if restoration else None,
    }
    for field, expected in expected_fixed.items():
        if fixed.get(field) != expected:
            raise ValueError(f"{context}: fixed-window {field} differs")
    if restoration:
        _boolean(fixed.get("event"), field=f"{context}.fixed_window.event")
    elif fixed.get("event") is not None:
        raise ValueError(f"{context}: non-restoration pair has a fixed-window event")
    _nonempty_string(fixed.get("run_path"), field=f"{context}.fixed_window.run_path")
    _sha256_digest(fixed.get("run_sha256"), field=f"{context}.fixed_window.run_sha256")

    extra_cohorts = {
        "fixed_selected_anchor": fixed_selected,
        "fixed_excluded_anchor": fixed_selected and not restoration,
        "prepared_typo_wrong": patch_eligible and not typo_correct,
        "prepared_typo_wrong_outside_restoration": (
            patch_eligible and not typo_correct and not restoration
        ),
        "repair_harm_composite": restoration or expected_cohorts["harm"],
    }
    for field, expected in extra_cohorts.items():
        if _boolean(cohorts.get(field), field=f"{context}.cohorts.{field}") != expected:
            raise ValueError(f"{context}: cohort {field} differs from pair outcomes")

    source = _mapping(record.get("source"), field=f"{context}.source")
    _nonempty_string(
        source.get("prepared_pairs_path"),
        field=f"{context}.source.prepared_pairs_path",
    )
    for field in (
        "prepared_pairs_sha256",
        "prepared_run_sha256",
        "source_record_sha256",
    ):
        _sha256_digest(source.get(field), field=f"{context}.source.{field}")
    _nonempty_string(source.get("model_revision"), field=f"{context}.source.model_revision")


def _validate_manifest_relations(records: Sequence[dict[str, object]]) -> None:
    by_pair_id = {str(record["pair_id"]): record for record in records}
    restoration = [record for record in records if record["cohorts"]["restoration"]]
    setting_counts: dict[tuple[str, str], Counter[str]] = {
        setting.key: Counter() for setting in REBUTTAL_SETTINGS
    }
    for record in restoration:
        counts = setting_counts[(str(record["model"]), str(record["task"]))]
        counts["pairs"] += 1
        counts["successes"] += int(record["fixed_window"]["event"] is True)
    for setting in REBUTTAL_SETTINGS:
        counts = setting_counts[setting.key]
        if (
            counts["pairs"] != setting.paper_denominator
            or counts["successes"] != setting.paper_successes
        ):
            raise ValueError(f"manifest paper totals differ for {setting.slug}")

    donor_candidates = [
        MatchedDonorCandidate(
            stratum=(
                str(record["task"]),
                str(record["model"]),
                str(record["target_rule"]),
                str(record["number_of_aligned_words"]),
            ),
            identity=(str(record["pair_id"]),),
        )
        for record in restoration
    ]
    expected_donors = plan_cyclic_derangement(
        donor_candidates,
        reject_singletons=False,
    )
    expected_assignments = {
        recipient[0]: donor[0] for recipient, donor in expected_donors.assignments
    }
    expected_singletons = {identity[0] for identity in expected_donors.singleton_identities}
    actual_assignments: dict[str, str] = {}
    actual_singletons: set[str] = set()
    for record in restoration:
        pair_id = str(record["pair_id"])
        cross_item = record["controls"]["cross_item"]
        if cross_item["valid"]:
            donor_pair_id = str(cross_item["donor_pair_id"])
            if donor_pair_id not in by_pair_id:
                raise ValueError(f"manifest donor is absent: {donor_pair_id}")
            donor = by_pair_id[donor_pair_id]
            recipient_stratum = (
                record["task"],
                record["model"],
                record["target_rule"],
                record["number_of_aligned_words"],
            )
            donor_stratum = (
                donor["task"],
                donor["model"],
                donor["target_rule"],
                donor["number_of_aligned_words"],
            )
            if donor_stratum != recipient_stratum:
                raise ValueError(f"manifest donor stratum differs for {pair_id}")
            actual_assignments[pair_id] = donor_pair_id
        elif cross_item.get("invalid_reason") == "singleton-matching-stratum":
            actual_singletons.add(pair_id)
    if actual_assignments != expected_assignments or actual_singletons != expected_singletons:
        raise ValueError("manifest cross-item donor plan differs from the protocol")

    expected_split = plan_window_split(
        (
            WindowSplitCandidate(
                str(record["pair_id"]),
                (
                    str(record["model"]),
                    str(record["task"]),
                    str(record["target_rule"]),
                ),
            )
            for record in restoration
        ),
        seed=REBUTTAL_MANIFEST_PROTOCOL.window_split_seed,
    )
    actual_selection = {
        str(record["pair_id"]) for record in restoration if record["cohorts"]["window_selection"]
    }
    actual_evaluation = {
        str(record["pair_id"]) for record in restoration if record["cohorts"]["window_evaluation"]
    }
    if actual_selection != set(expected_split.selection_pair_ids) or actual_evaluation != set(
        expected_split.evaluation_pair_ids
    ):
        raise ValueError("manifest held-out split differs from the protocol")


def _load_and_validate_artifact_set(
    pair_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if pair_manifest_path.name != "pair_manifest.jsonl":
        raise ValueError("rebuttal pair manifest must retain its canonical file name")
    run_path = pair_manifest_path.parent / "run.json"
    cohort_path = pair_manifest_path.parent / "cohort_ids.json"
    audit_path = pair_manifest_path.parent / "source_audit.json"
    for path in (run_path, cohort_path, audit_path):
        if not path.is_file():
            raise ValueError(f"rebuttal artifact set is incomplete: {path}")

    run = load_json_object(run_path)
    if (
        run.get("schema_version") != REBUTTAL_RUN_SCHEMA
        or run.get("paper_sha256") != PAPER_SHA256
        or run.get("operation") != "build-rebuttal-manifest"
        or run.get("status") != "completed"
    ):
        raise ValueError(f"rebuttal run manifest contract differs: {run_path}")
    protocol = _mapping(run.get("protocol"), field=f"{run_path} protocol")
    if (
        protocol.get("manifest_contract") != REBUTTAL_MANIFEST_PROTOCOL.as_dict()
        or protocol.get("manifest_contract_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
    ):
        raise ValueError(f"rebuttal run protocol differs: {run_path}")

    paths = {
        "pair_manifest.jsonl": pair_manifest_path,
        "cohort_ids.json": cohort_path,
        "source_audit.json": audit_path,
    }
    outputs = _mapping(run.get("outputs"), field=f"{run_path} outputs")
    if set(outputs) != set(paths):
        raise ValueError(f"rebuttal run output inventory differs: {run_path}")
    for name, path in paths.items():
        metadata = _mapping(outputs.get(name), field=f"{run_path} outputs.{name}")
        if _sha256_digest(
            metadata.get("sha256"),
            field=f"{run_path} outputs.{name}.sha256",
        ) != sha256_file(path):
            raise ValueError(f"rebuttal output SHA-256 differs: {path}")
        output_records = _positive_int(
            metadata.get("records"),
            field=f"{run_path} outputs.{name}.records",
        )
        if name != "pair_manifest.jsonl" and output_records != 1:
            raise ValueError(f"rebuttal singleton artifact record count differs: {path}")

    cohorts = load_json_object(cohort_path)
    if (
        cohorts.get("schema_version") != REBUTTAL_COHORT_SCHEMA
        or cohorts.get("paper_sha256") != PAPER_SHA256
        or cohorts.get("manifest_protocol_sha256") != REBUTTAL_MANIFEST_PROTOCOL.sha256()
        or cohorts.get("pair_manifest_sha256") != sha256_file(pair_manifest_path)
    ):
        raise ValueError(f"rebuttal cohort artifact contract differs: {cohort_path}")
    if cohorts.get("identity") != "sha256-canonical-model-task-target-rule-sample-id/v1":
        raise ValueError(f"rebuttal cohort identity rule differs: {cohort_path}")
    if _mapping(cohorts.get("window_split"), field=f"{cohort_path} window_split") != {
        "algorithm": "sha256-order-first-floor-half-per-model-task-target-rule/v1",
        "seed": REBUTTAL_MANIFEST_PROTOCOL.window_split_seed,
        "outcome_independent": True,
    }:
        raise ValueError(f"rebuttal cohort split rule differs: {cohort_path}")
    audit = load_json_object(audit_path)
    if (
        audit.get("schema_version") != REBUTTAL_AUDIT_SCHEMA
        or audit.get("paper_sha256") != PAPER_SHA256
        or _mapping(audit.get("protocol"), field=f"{audit_path} protocol") != protocol
    ):
        raise ValueError(f"rebuttal source audit contract differs: {audit_path}")
    return run, cohorts, audit


def _digest_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    digests = tuple(_sha256_digest(item, field=f"{field}[]") for item in value)
    if digests != tuple(sorted(set(digests))):
        raise ValueError(f"{field} must be sorted and unique")
    return digests


def _validate_artifact_relations(
    records: Sequence[dict[str, object]],
    *,
    run: Mapping[str, object],
    cohorts: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    expected_cohorts = {
        "restoration": sorted(
            str(record["pair_id"]) for record in records if record["cohorts"]["restoration"]
        ),
        "harm": sorted(str(record["pair_id"]) for record in records if record["cohorts"]["harm"]),
        "full_clean_correct": sorted(
            str(record["pair_id"]) for record in records if record["cohorts"]["full_clean_correct"]
        ),
        "patch_eligible_clean_correct": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["patch_eligible_clean_correct"]
        ),
        "alignment_ineligible_clean_correct": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["alignment_ineligible_clean_correct"]
        ),
        "fixed_selected_anchor": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["fixed_selected_anchor"]
        ),
        "fixed_excluded_anchor": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["fixed_excluded_anchor"]
        ),
        "prepared_typo_wrong": sorted(
            str(record["pair_id"]) for record in records if record["cohorts"]["prepared_typo_wrong"]
        ),
        "prepared_typo_wrong_outside_restoration": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["prepared_typo_wrong_outside_restoration"]
        ),
        "repair_harm_composite": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["repair_harm_composite"]
        ),
        "window_selection": sorted(
            str(record["pair_id"]) for record in records if record["cohorts"]["window_selection"]
        ),
        "window_evaluation": sorted(
            str(record["pair_id"]) for record in records if record["cohorts"]["window_evaluation"]
        ),
        "offset_valid": sorted(
            str(record["pair_id"])
            for record in records
            if record["cohorts"]["restoration"] and record["controls"]["offset_2"]["valid"]
        ),
        "cross_item_valid": sorted(
            str(record["pair_id"])
            for record in records
            if record["controls"]["cross_item"]["valid"]
        ),
        "common_valid": sorted(
            str(record["pair_id"]) for record in records if record["controls"]["common_valid"]
        ),
    }
    stored_cohorts = _mapping(cohorts.get("cohorts"), field="cohort_ids.cohorts")
    if set(stored_cohorts) != set(expected_cohorts):
        raise ValueError("cohort artifact inventory differs from the pair manifest")
    for name, expected in expected_cohorts.items():
        if (
            list(_digest_list(stored_cohorts.get(name), field=f"cohort_ids.cohorts.{name}"))
            != expected
        ):
            raise ValueError(f"cohort artifact {name} differs from the pair manifest")

    expected_donors = [
        {
            "recipient_pair_id": str(record["pair_id"]),
            "donor_pair_id": str(record["controls"]["cross_item"]["donor_pair_id"]),
        }
        for record in records
        if record["controls"]["cross_item"]["valid"]
    ]
    expected_donors.sort(key=lambda item: item["recipient_pair_id"])
    if cohorts.get("cross_item_donors") != expected_donors:
        raise ValueError("cohort donor assignments differ from the pair manifest")
    invalid_cross = sorted(
        str(record["pair_id"])
        for record in records
        if record["cohorts"]["restoration"] and not record["controls"]["cross_item"]["valid"]
    )
    if (
        list(
            _digest_list(
                cohorts.get("invalid_cross_item_recipients"),
                field="cohort_ids.invalid_cross_item_recipients",
            )
        )
        != invalid_cross
    ):
        raise ValueError("invalid donor recipients differ from the pair manifest")

    counts = _mapping(run.get("counts"), field="run.counts")
    expected_counts = {
        "pair_records": len(records),
        "restoration_pairs": len(expected_cohorts["restoration"]),
        "fixed_window_successes": sum(
            int(record["fixed_window"]["event"] is True) for record in records
        ),
        "harm_pairs": len(expected_cohorts["harm"]),
        "fixed_selected_anchors": len(expected_cohorts["fixed_selected_anchor"]),
        "fixed_excluded_anchors": len(expected_cohorts["fixed_excluded_anchor"]),
        "prepared_wrong_outside_restoration": len(
            expected_cohorts["prepared_typo_wrong_outside_restoration"]
        ),
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"run count {field} differs from the artifact set")
    outputs = _mapping(run.get("outputs"), field="run.outputs")
    pair_metadata = _mapping(
        outputs.get("pair_manifest.jsonl"),
        field="run.outputs.pair_manifest.jsonl",
    )
    if pair_metadata.get("records") != len(records):
        raise ValueError("pair manifest record count differs from run metadata")

    paper_totals = _mapping(audit.get("paper_totals"), field="source_audit.paper_totals")
    if paper_totals != {
        "restoration_pairs": REBUTTAL_MANIFEST_PROTOCOL.restoration_pairs,
        "fixed_window_successes": REBUTTAL_MANIFEST_PROTOCOL.fixed_window_successes,
    }:
        raise ValueError("source audit paper totals differ from the protocol")

    expected_settings: list[dict[str, object]] = []
    for setting in REBUTTAL_SETTINGS:
        setting_records = [
            record for record in records if (record["model"], record["task"]) == setting.key
        ]

        def cohort_count(name: str) -> int:
            return sum(int(record["cohorts"][name] is True) for record in setting_records)

        restoration_records = [
            record for record in setting_records if record["cohorts"]["restoration"]
        ]
        expected_settings.append(
            {
                "model": setting.model,
                "task": setting.task,
                "paper_denominator": setting.paper_denominator,
                "paper_successes": setting.paper_successes,
                "prepared_pairs": len(setting_records),
                "full_clean_correct": cohort_count("full_clean_correct"),
                "patch_eligible_clean_correct": cohort_count("patch_eligible_clean_correct"),
                "alignment_ineligible_clean_correct": cohort_count(
                    "alignment_ineligible_clean_correct"
                ),
                "restoration_pairs": len(restoration_records),
                "fixed_window_successes": sum(
                    int(record["fixed_window"]["event"] is True) for record in restoration_records
                ),
                "harm_pairs": cohort_count("harm"),
                "fixed_selected_anchors": cohort_count("fixed_selected_anchor"),
                "fixed_excluded_anchors": cohort_count("fixed_excluded_anchor"),
                "prepared_typo_wrong": cohort_count("prepared_typo_wrong"),
                "prepared_typo_wrong_outside_restoration": cohort_count(
                    "prepared_typo_wrong_outside_restoration"
                ),
                "offset_valid": sum(
                    int(record["controls"]["offset_2"]["valid"] is True)
                    for record in restoration_records
                ),
                "cross_item_valid": sum(
                    int(record["controls"]["cross_item"]["valid"] is True)
                    for record in restoration_records
                ),
                "common_valid": sum(
                    int(record["controls"]["common_valid"] is True)
                    for record in restoration_records
                ),
            }
        )
    if audit.get("settings") != expected_settings:
        raise ValueError("source audit setting counts differ from the pair manifest")


def load_rebuttal_pair_manifest(path: Path) -> tuple[dict[str, object], ...]:
    """Read a normalized manifest fail-closed for downstream experiment runners."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"rebuttal pair manifest is not a file: {resolved}")
    run, cohorts, audit = _load_and_validate_artifact_set(resolved)
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    previous_sort_key: tuple[str, str, int, str] | None = None
    for line_number, _line, record in iter_jsonl_objects(resolved):
        context = f"{resolved}:{line_number}"
        _validate_manifest_record(record, context=context)
        pair_id = str(record["pair_id"])
        if pair_id in seen:
            raise ValueError(f"{context}: duplicate pair_id {pair_id}")
        seen.add(pair_id)
        sort_key = (
            str(record["task"]),
            str(record["model"]),
            _TARGET_RULE_ORDER[str(record["target_rule"])],
            str(record["sample_id"]),
        )
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ValueError(f"{context}: manifest records are not strictly sorted")
        previous_sort_key = sort_key
        records.append(record)
    if not records:
        raise ValueError("rebuttal pair manifest contains no records")
    _validate_manifest_relations(records)
    _validate_artifact_relations(
        records,
        run=run,
        cohorts=cohorts,
        audit=audit,
    )
    return tuple(records)


__all__ = [
    "PAIR_MANIFEST_SCHEMA",
    "REBUTTAL_AUDIT_SCHEMA",
    "REBUTTAL_COHORT_SCHEMA",
    "REBUTTAL_RUN_SCHEMA",
    "REBUTTAL_SETTINGS",
    "RebuttalSetting",
    "canonical_sha256",
    "iter_jsonl_objects",
    "load_json_object",
    "load_rebuttal_pair_manifest",
    "normalize_prepared_pair",
    "pair_id_for",
    "sha256_file",
    "strict_loads",
]

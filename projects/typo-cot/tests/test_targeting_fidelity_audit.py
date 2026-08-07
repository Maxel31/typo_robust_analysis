"""Tests for the paper Appendix A targeting-fidelity audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from typo_cot.cli import main
from typo_cot.experiments.catalog import PAPER_SHA256, get_experiment
from typo_cot.experiments.prepare_edited_pairs.protocol import seeded_character_edit
from typo_cot.experiments.targeting_fidelity_audit import (
    TargetingFidelityAuditConfig,
    TargetingFidelityAuditError,
    run_targeting_fidelity_audit,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _span(text: str, needle: str, *, occurrence: int = 1) -> dict[str, int]:
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.index(needle, search_from)
        search_from = start + len(needle)
    return {"start": start, "end": start + len(needle)}


def _attempt(
    rank: int,
    *,
    faithful: bool = True,
    operation: str = "substitution",
    attribution_rank: int | None = None,
) -> dict[str, object]:
    return {
        "selection_rank": rank,
        "attribution_rank": attribution_rank if attribution_rank is not None else rank,
        "target_token_index": 100 + rank,
        "target_token_text": f" token-{rank}",
        "relevance": 1.0 / rank,
        "intended_prompt_span": {"start": rank, "end": rank + 1},
        "intended_editable_span": {"start": rank, "end": rank + 1},
        "landed_editable_span_before": {"start": rank, "end": rank + 1},
        "landed_text_before": "x",
        "landed_origin_index": rank if faithful else None,
        "landed_on_intended_token": faithful,
        "intended_word_index": rank,
        "landed_word_index": rank,
        "operation": operation,
        "character_index": 0,
        "original_character": "x",
        "new_character": "y" if operation != "deletion" else None,
        "edited_token_text": "y" if operation != "deletion" else "",
    }


def _aligned_word(
    editable: str,
    needle: str,
    *,
    word_index: int,
    target_ranks: tuple[int, ...],
    occurrence: int = 1,
) -> dict[str, object]:
    clean_span = _span(editable, needle, occurrence=occurrence)
    return {
        "word_index": word_index,
        "clean_text": needle,
        "edited_text": f"{needle}x",
        "clean_editable_span": clean_span,
        "edited_editable_span": clean_span,
        "clean_prompt_span": clean_span,
        "edited_prompt_span": clean_span,
        "target_ranks": list(target_ranks),
        "target_token_indices": [100 + rank for rank in target_ranks],
        "clean_token_indices": [200 + word_index],
        "edited_token_indices": [200 + word_index],
        "clean_final_token": 200 + word_index,
        "edited_final_token": 200 + word_index,
    }


def _word_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, character in enumerate(text):
        if character.isspace():
            if start is not None:
                spans.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _word_index_at(text: str, position: int) -> int:
    return next(
        index for index, (start, end) in enumerate(_word_spans(text)) if start <= position < end
    )


def _materialize_edits(
    editable: str,
    attempt_specs: list[dict[str, object]],
    word_specs: list[dict[str, object]],
    *,
    sample_id: str,
    seed: int,
    editable_prompt_start: int,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    """Search producer-valid candidates matching compact test expectations."""
    clean_spans = _word_spans(editable)
    rank_to_word: dict[int, int] = {}
    for word in word_specs:
        planned_span = word["clean_editable_span"]
        assert isinstance(planned_span, dict)
        start = int(planned_span["start"])
        clean_word_index = next(
            index
            for index, (word_start, word_end) in enumerate(clean_spans)
            if word_start <= start < word_end
        )
        for rank in word["target_ranks"]:
            rank_to_word[int(rank)] = clean_word_index

    current = editable
    origins: list[int | None] = list(range(len(editable)))
    cumulative_shift = 0
    attempts: list[dict[str, object]] = []
    for rank, spec in enumerate(attempt_specs, 1):
        desired_landed_word = rank_to_word[rank]
        desired_operation = str(spec["operation"])
        desired_faithful = bool(spec["landed_on_intended_token"])
        candidate_spans: list[tuple[int, int, int]] = []
        for intended_word_index, (word_start, word_end) in enumerate(clean_spans):
            for intended_start in range(word_start, word_end):
                for intended_end in range(word_end, intended_start, -1):
                    if any(
                        character.isascii() and character.isalpha()
                        for character in editable[intended_start:intended_end]
                    ):
                        candidate_spans.append((intended_word_index, intended_start, intended_end))
        candidate_spans.sort(key=lambda item: item[0] != desired_landed_word)

        selected: tuple[int, int, int, int, object, int | None, int] | None = None
        for intended_word_index, intended_start, intended_end in candidate_spans:
            landed_start = intended_start + cumulative_shift
            landed_end = intended_end + cumulative_shift
            if not 0 <= landed_start < landed_end <= len(current):
                continue
            landed_text = current[landed_start:landed_end]
            target_text = editable[intended_start:intended_end]
            for target_token_index in range(rank * 1000, rank * 1000 + 2000):
                edit = seeded_character_edit(
                    landed_text,
                    _stable_seed(
                        seed,
                        sample_id,
                        target_token_index,
                        target_text,
                    ),
                )
                if edit is None or edit.operation != desired_operation:
                    continue
                affected_position = landed_start + edit.character_index
                landed_origin = origins[affected_position]
                faithful = (
                    landed_origin is not None and intended_start <= landed_origin < intended_end
                )
                landed_word_index = _word_index_at(current, affected_position)
                if faithful == desired_faithful and landed_word_index == desired_landed_word:
                    selected = (
                        intended_word_index,
                        intended_start,
                        intended_end,
                        target_token_index,
                        edit,
                        landed_origin,
                        landed_word_index,
                    )
                    break
            if selected is not None:
                break
        assert selected is not None, (rank, spec, cumulative_shift)
        (
            intended_word_index,
            intended_start,
            intended_end,
            target_token_index,
            raw_edit,
            landed_origin,
            landed_word_index,
        ) = selected
        edit = raw_edit
        assert hasattr(edit, "operation")
        landed_start = intended_start + cumulative_shift
        landed_end = intended_end + cumulative_shift
        landed_text = current[landed_start:landed_end]
        faithful = bool(spec["landed_on_intended_token"])
        attempts.append(
            {
                "selection_rank": rank,
                "attribution_rank": int(spec["attribution_rank"]),
                "target_token_index": target_token_index,
                "target_token_text": editable[intended_start:intended_end],
                "relevance": float(spec["relevance"]),
                "intended_prompt_span": {
                    "start": editable_prompt_start + intended_start,
                    "end": editable_prompt_start + intended_end,
                },
                "intended_editable_span": {"start": intended_start, "end": intended_end},
                "landed_editable_span_before": {
                    "start": landed_start,
                    "end": landed_end,
                },
                "landed_text_before": landed_text,
                "landed_origin_index": landed_origin,
                "landed_on_intended_token": faithful,
                "intended_word_index": intended_word_index,
                "landed_word_index": landed_word_index,
                "operation": edit.operation,
                "character_index": edit.character_index,
                "original_character": edit.original_character,
                "new_character": edit.new_character,
                "edited_token_text": edit.edited,
            }
        )
        local_origins = origins[landed_start:landed_end]
        if edit.operation == "substitution":
            replacement_origins = local_origins
        elif edit.operation == "duplication":
            replacement_origins = (
                local_origins[: edit.character_index + 1]
                + [None]
                + local_origins[edit.character_index + 1 :]
            )
        else:
            replacement_origins = (
                local_origins[: edit.character_index] + local_origins[edit.character_index + 1 :]
            )
        origins = origins[:landed_start] + replacement_origins + origins[landed_end:]
        current = current[:landed_start] + edit.edited + current[landed_end:]
        cumulative_shift += len(edit.edited) - len(landed_text)

    edited_spans = _word_spans(current)
    aligned: list[dict[str, object]] = []
    for word_index, ((clean_start, clean_end), (edited_start, edited_end)) in enumerate(
        zip(clean_spans, edited_spans, strict=True)
    ):
        clean_text = editable[clean_start:clean_end]
        edited_text = current[edited_start:edited_end]
        if clean_text == edited_text:
            continue
        matching = [attempt for attempt in attempts if attempt["landed_word_index"] == word_index]
        aligned.append(
            {
                "word_index": word_index,
                "clean_text": clean_text,
                "edited_text": edited_text,
                "clean_editable_span": {"start": clean_start, "end": clean_end},
                "edited_editable_span": {"start": edited_start, "end": edited_end},
                "clean_prompt_span": {
                    "start": editable_prompt_start + clean_start,
                    "end": editable_prompt_start + clean_end,
                },
                "edited_prompt_span": {
                    "start": editable_prompt_start + edited_start,
                    "end": editable_prompt_start + edited_end,
                },
                "target_ranks": [int(attempt["selection_rank"]) for attempt in matching],
                "target_token_indices": [
                    int(attempt["target_token_index"]) for attempt in matching
                ],
                "clean_token_indices": [200 + word_index],
                "edited_token_indices": [200 + word_index],
                "clean_final_token": 200 + word_index,
                "edited_final_token": 200 + word_index,
            }
        )
    return current, attempts, aligned


def _pair(
    sample_id: str,
    *,
    model: str,
    benchmark: str,
    targeting: str,
    question: str,
    choices: list[str] | None,
    gold_answer: str,
    attempts: list[dict[str, object]],
    aligned_words: list[dict[str, object]],
    seed: int = 42,
) -> dict[str, object]:
    if choices:
        options = " ".join(
            f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(choices)
        )
        editable = f"{question}\n{options}"
    else:
        editable = question
    editable_prompt_start = len("PREFIX:")
    clean_prompt = "PREFIX:" + editable
    edited_editable, attempts, aligned_words = _materialize_edits(
        editable,
        attempts,
        aligned_words,
        sample_id=sample_id,
        seed=seed,
        editable_prompt_start=editable_prompt_start,
    )
    excluded_attribution_tokens = (
        [
            {
                "token_index": 9000 + rank,
                "text": f"excluded-{rank}",
                "relevance": 10.0 - rank,
                "prompt_start": editable_prompt_start,
                "prompt_end": editable_prompt_start + 1,
                "attribution_rank": rank,
            }
            for rank in range(1, 5)
        ]
        if targeting == "random-4"
        else []
    )
    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": model,
        "benchmark": benchmark,
        "targeting": targeting,
        "seed": seed,
        "num_edits_requested": 4,
        "num_candidates": 12,
        "num_target_attempts": len(attempts),
        "num_aligned_words": len(aligned_words),
        "gold_answer": gold_answer,
        "subset": None,
        "clean": {
            "question": question,
            "choices": choices,
            "editable_text": editable,
            "editable_prompt_span": {
                "start": editable_prompt_start,
                "end": editable_prompt_start + len(editable),
            },
            "prompt": clean_prompt,
        },
        "edited": {
            "editable_text": edited_editable,
            "editable_prompt_span": {
                "start": editable_prompt_start,
                "end": editable_prompt_start + len(edited_editable),
            },
            "prompt": "PREFIX:" + edited_editable,
        },
        "excluded_attribution_tokens": excluded_attribution_tokens,
        "target_attempts": attempts,
        "aligned_words": aligned_words,
    }


def _write_source(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    model: str,
    benchmark: str,
    targeting: str,
    seed: int = 42,
    status: str = "completed",
    paper_sha256: str = PAPER_SHA256,
    limit: int | None = None,
) -> None:
    directory.mkdir(parents=True)
    pairs_path = directory / "pairs.jsonl"
    pairs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    failed = 0 if status == "completed" else 1
    dataset_identity = json.dumps(
        [
            {
                "sample_id": row["sample_id"],
                "question": row["clean"]["question"],
                "choices": row["clean"]["choices"],
                "correct_answer": row["gold_answer"],
                "subset": row["subset"],
            }
            for row in sorted(rows, key=lambda row: str(row["sample_id"]))
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    benchmark_loader = {
        "gsm8k": "gsm8k",
        "math-500": "math",
        "mmlu": "mmlu",
        "mmlu-pro": "mmlu_pro",
        "arc": "arc",
        "csqa": "commonsense_qa",
    }[benchmark]
    _write_json(
        directory / "run.json",
        {
            "schema_version": "prepare-edited-pairs-run/v1",
            "paper_sha256": paper_sha256,
            "operation": "prepare-edited-pairs",
            "status": status,
            "arguments": {
                "model": model,
                "benchmark": benchmark,
                "targeting": targeting,
                "num_edits": 4,
                "output_dir": str(directory),
                "seed": seed,
                "max_new_tokens": 512,
                "gpu_id": "0",
                "limit": limit,
            },
            "counts": {
                "discovered": len(rows) + failed,
                "written": len(rows),
                "failed": failed,
            },
            "failures": [] if failed == 0 else [{"sample_id": "failed"}],
            "decoding": {
                "strategy": "greedy",
                "dtype": "bfloat16",
                "padding_side": "left",
                "max_new_tokens": 512,
            },
            "provenance": {
                "fixture": True,
                "model": model,
                "model_revision": "fixture-model-revision",
                "benchmark_dataset_loader": benchmark_loader,
                "dataset_sample_count": len(rows),
                "dataset_records_sha256": hashlib.sha256(dataset_identity).hexdigest(),
                "random_seed_algorithm": "sha256-first-64-bits/v1",
                "target_position": "maximum-logit-after-first-cot-token",
                "alignment": "actual-edited-word-final-token",
            },
        },
    )


def _fixture_inputs(root: Path) -> None:
    model = "test/model"
    question = "Which animal?"
    choices = ["cat", "blue whale", "dog", "fox"]
    editable = f"{question}\n(A) cat (B) blue whale (C) dog (D) fox"
    attribution_rows = [
        _pair(
            "mc-gold-edited",
            model=model,
            benchmark="mmlu",
            targeting="attribution-4",
            question=question,
            choices=choices,
            gold_answer="B",
            attempts=[
                _attempt(1, operation="duplication"),
                _attempt(2, faithful=False, operation="substitution"),
                _attempt(3, operation="deletion"),
                _attempt(4, operation="substitution"),
            ],
            aligned_words=[
                _aligned_word(editable, "blue", word_index=3, target_ranks=(2, 3)),
                _aligned_word(editable, "dog", word_index=5, target_ranks=(1,)),
                _aligned_word(editable, "fox", word_index=6, target_ranks=(4,)),
            ],
        ),
        _pair(
            "free-answer",
            model=model,
            benchmark="gsm8k",
            targeting="attribution-4",
            question="alpha beta gamma delta",
            choices=None,
            gold_answer="4",
            attempts=[
                _attempt(1),
                _attempt(2, operation="duplication"),
                _attempt(3, operation="deletion"),
                _attempt(4),
            ],
            aligned_words=[
                _aligned_word(
                    "alpha beta gamma delta", word, word_index=index, target_ranks=(index + 1,)
                )
                for index, word in enumerate(("alpha", "beta", "gamma", "delta"))
            ],
        ),
    ]
    _write_source(
        root / "z-attribution-mmlu",
        [attribution_rows[0]],
        model=model,
        benchmark="mmlu",
        targeting="attribution-4",
    )
    _write_source(
        root / "a-attribution-gsm8k",
        [attribution_rows[1]],
        model=model,
        benchmark="gsm8k",
        targeting="attribution-4",
    )

    random_attempts = [
        _attempt(rank, operation="duplication", attribution_rank=rank + 4) for rank in range(1, 5)
    ]
    random_editable = f"{question}\n(A) cat (B) blue whale (C) dog (D) fox"
    random_row = _pair(
        "mc-gold-edited",
        model=model,
        benchmark="mmlu",
        targeting="random-4",
        question=question,
        choices=choices,
        gold_answer="B",
        attempts=random_attempts,
        aligned_words=[
            _aligned_word(random_editable, "Which", word_index=0, target_ranks=(1,)),
            _aligned_word(random_editable, "cat", word_index=2, target_ranks=(2,)),
            _aligned_word(random_editable, "dog", word_index=5, target_ranks=(3,)),
            _aligned_word(random_editable, "fox", word_index=6, target_ranks=(4,)),
        ],
    )
    _write_source(
        root / "m-random-mmlu",
        [random_row],
        model=model,
        benchmark="mmlu",
        targeting="random-4",
    )


def _break_cumulative_shift_invariant(manifest: dict[str, object], pair: dict[str, object]) -> None:
    del manifest
    attempt = pair["target_attempts"][0]
    intended = attempt["intended_editable_span"]
    intended_prompt = attempt["intended_prompt_span"]
    assert isinstance(intended, dict) and isinstance(intended_prompt, dict)
    intended["end"] = int(intended["end"]) + 1
    intended_prompt["end"] = int(intended_prompt["end"]) + 1


def _change_seed_input_but_preserve_declared_alignment(
    manifest: dict[str, object], pair: dict[str, object]
) -> None:
    del manifest
    attempt = pair["target_attempts"][0]
    original_index = int(attempt["target_token_index"])
    landed_text = str(attempt["landed_text_before"])
    sample_id = str(pair["sample_id"])
    seed = int(pair["seed"])
    target_text = str(attempt["target_token_text"])
    declared = (
        attempt["operation"],
        attempt["character_index"],
        attempt["new_character"],
        attempt["edited_token_text"],
    )
    replacement = original_index + 1
    while True:
        edit = seeded_character_edit(
            landed_text,
            _stable_seed(seed, sample_id, replacement, target_text),
        )
        assert edit is not None
        replayed = (
            edit.operation,
            edit.character_index,
            edit.new_character,
            edit.edited,
        )
        if replayed != declared:
            break
        replacement += 1
    attempt["target_token_index"] = replacement
    for word in pair["aligned_words"]:
        ranks = list(word["target_ranks"])
        if 1 in ranks:
            rank_position = ranks.index(1)
            word["target_token_indices"][rank_position] = replacement
            break


def test_audit_keeps_paper_denominators_separate_and_writes_provenance(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "pairs"
    _fixture_inputs(pairs_root)
    output_dir = tmp_path / "audit"

    result = run_targeting_fidelity_audit(
        TargetingFidelityAuditConfig(pairs_root=pairs_root, output_dir=output_dir)
    )

    assert result.items == 3
    assert result.settings == 2
    assert result.input_cells == 3
    assert set(path.name for path in output_dir.iterdir()) == {
        "operation_counts.json",
        "run.json",
        "targeting_fidelity.csv",
        "targeting_fidelity_records.jsonl",
    }

    with (output_dir / "targeting_fidelity.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    pooled = {row["targeting"]: row for row in csv_rows if row["row_type"] == "targeting"}
    attribution = pooled["attribution-4"]
    assert attribution["items"] == "2"
    assert attribution["four_distinct_word_items"] == "1"
    assert float(attribution["four_distinct_word_rate"]) == 0.5
    assert attribution["target_attempts"] == "8"
    assert attribution["faithful_target_attempts"] == "7"
    assert float(attribution["targeting_fidelity_rate"]) == 0.875
    assert float(attribution["targeting_miss_rate"]) == 0.125
    assert attribution["selection_rank_1_attempts"] == "2"
    assert attribution["selection_rank_1_faithful_attempts"] == "2"
    assert float(attribution["selection_rank_1_fidelity_rate"]) == 1.0
    assert attribution["prepared_multiple_choice_items"] == "1"
    assert attribution["prepared_gold_option_edited_items"] == "1"
    assert float(attribution["prepared_pair_gold_option_edit_rate"]) == 1.0

    random_control = pooled["random-4"]
    assert random_control["items"] == "1"
    assert float(random_control["four_distinct_word_rate"]) == 1.0
    assert random_control["selection_rank_1_attempts"] == "0"
    assert random_control["selection_rank_1_fidelity_rate"] == ""
    assert random_control["prepared_gold_option_edited_items"] == "0"

    records = [
        json.loads(line)
        for line in (output_dir / "targeting_fidelity_records.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in records] == [
        "free-answer",
        "mc-gold-edited",
        "mc-gold-edited",
    ]
    assert records[0]["gold_option_applicable"] is False
    assert records[0]["gold_option_edited"] is None
    assert records[1]["gold_option_edited"] is True
    assert records[2]["gold_option_edited"] is False

    operations = json.loads((output_dir / "operation_counts.json").read_text())
    assert operations["schema_version"] == "targeting-fidelity-operation-counts/v1"
    assert operations["overall"]["target_attempts"] == 12
    assert operations["overall"]["counts"] == {
        "deletion": 2,
        "duplication": 6,
        "substitution": 4,
    }

    run = json.loads((output_dir / "run.json").read_text())
    assert run["schema_version"] == "targeting-fidelity-audit-run/v1"
    assert run["paper_sha256"] == PAPER_SHA256
    assert run["status"] == "completed"
    assert run["counts"] == {
        "input_cells": 3,
        "input_files": 3,
        "items": 3,
        "settings": 2,
    }
    assert run["paper_reference_values"]["four_distinct_words"] == {
        "settings": 42,
        "attribution-4": {"items_with_four": 56141, "items": 68660, "rate": 0.818},
        "random-4": {"items_with_four": 65702, "items": 68660, "rate": 0.957},
    }
    assert run["paper_reference_values"]["top_selected_attribution_attempt"] == {
        "misplaced": 0,
        "attempts": 68650,
    }
    assert run["paper_reference_values"]["all_evaluable_target_miss_rate"] == {
        "rate": 0.302,
        "misplaced": 163043,
        "attempts": 540724,
        "legacy_unevaluable_attempts": 7589,
        "targeting_conditions": ["attribution-4", "random-4"],
    }
    assert run["paper_reference_values"]["conditional_gold_option_edit_rate"] == {
        "rate": 0.215,
        "numerator": 3501,
        "denominator": 16316,
        "settings": 20,
        "cohort": "Attribution-4 CoT-swap included multiple-choice items",
        "computable_from_prepared_pairs_alone": False,
    }
    assert run["paper_reference_values"]["sources"] == {
        "final_pdf": "rounded published rates and 0/68,650 rank-1 result",
        "archival_reanalysis": (
            "exact numerators, denominators, rank 2-4 breakdowns, and cohort counts"
        ),
    }
    assert run["paper_reference_values"]["source_by_metric"]["all_evaluable_target_miss_rate"] == {
        "rounded_rate": "final_pdf",
        "exact_counts": "archival_reanalysis",
    }
    assert "gold_option_edit_rate" not in run["paper_reference_values"]
    assert len(run["inputs"]) == 3
    assert run["paper_comparison"]["status"] == "not_comparable"
    assert run["paper_comparison"]["expected_model_benchmark_settings"] == 42
    assert run["paper_comparison"]["observed_model_benchmark_settings"] == 2
    assert run["paper_comparison"]["checks"] == {
        "paper_seed_42": True,
        "exact_42_setting_84_cell_grid": False,
        "exact_per_cell_item_counts": False,
        "paired_targeting_arms": False,
        "paper_generation_cap_512": True,
        "exact_arm_item_totals": False,
    }
    assert len(run["paper_comparison"]["cell_count_mismatches"]) == 3
    assert run["paper_comparison"]["expected_cell_counts_source"] == (
        "archival_reanalysis/source_provenance.csv"
    )
    assert run["metric_protocol"]["rank_metric"] == (
        "successful Attribution-4 application selection_rank"
    )
    assert run["metric_protocol"]["landing_metric"] == (
        "public-v1 origin-based Boolean for every target attempt"
    )
    for source in run["inputs"]:
        pairs_path = pairs_root / source["pairs_path"]
        manifest_path = pairs_root / source["manifest_path"]
        assert source["pairs_sha256"] == hashlib.sha256(pairs_path.read_bytes()).hexdigest()
        assert source["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for output in run["outputs"]:
        path = output_dir / output["path"]
        assert output["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_gold_option_uses_the_exact_formatted_choice_span_not_text_search(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    question = "Why is blue whale repeated here?"
    choices = ["cat", "blue whale", "dog", "fox"]
    editable = f"{question}\n(A) cat (B) blue whale (C) dog (D) fox"
    rows = [
        _pair(
            "question-only",
            model="test/model",
            benchmark="mmlu",
            targeting="attribution-4",
            question=question,
            choices=choices,
            gold_answer="B",
            attempts=[_attempt(1)],
            aligned_words=[
                _aligned_word(editable, "blue", word_index=2, target_ranks=(1,), occurrence=1)
            ],
        ),
        _pair(
            "gold-choice",
            model="test/model",
            benchmark="mmlu",
            targeting="attribution-4",
            question=question,
            choices=choices,
            gold_answer="B",
            attempts=[_attempt(1)],
            aligned_words=[
                _aligned_word(editable, "blue", word_index=7, target_ranks=(1,), occurrence=2)
            ],
        ),
    ]
    rows.sort(key=lambda row: str(row["sample_id"]))
    _write_source(
        pairs_root / "cell",
        rows,
        model="test/model",
        benchmark="mmlu",
        targeting="attribution-4",
    )

    output_dir = tmp_path / "audit"
    run_targeting_fidelity_audit(
        TargetingFidelityAuditConfig(pairs_root=pairs_root, output_dir=output_dir)
    )

    records = {
        row["sample_id"]: row
        for row in map(
            json.loads,
            (output_dir / "targeting_fidelity_records.jsonl").read_text().splitlines(),
        )
    }
    assert records["question-only"]["gold_option_edited"] is False
    assert records["gold-choice"]["gold_option_edited"] is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda manifest, pair: manifest.update(status="running"), "not completed"),
        (
            lambda manifest, pair: manifest.update(paper_sha256="0" * 64),
            "paper SHA-256",
        ),
        (
            lambda manifest, pair: manifest["arguments"].update(limit=1),
            "partial --limit",
        ),
        (
            lambda manifest, pair: manifest["arguments"].update(num_edits=2),
            "four edits",
        ),
        (lambda manifest, pair: pair.update(model="other/model"), "model"),
        (
            lambda manifest, pair: pair["target_attempts"][0].update(operation="swap"),
            "operation",
        ),
        (
            lambda manifest, pair: pair.update(num_aligned_words=99),
            "num_aligned_words",
        ),
        (
            lambda manifest, pair: manifest["provenance"].update(
                random_seed_algorithm="python-hash"
            ),
            "random_seed_algorithm",
        ),
        (
            lambda manifest, pair: manifest["provenance"].update(dataset_records_sha256="0" * 64),
            "dataset records SHA-256",
        ),
        (
            lambda manifest, pair: manifest["decoding"].update(strategy="sampled"),
            "greedy",
        ),
        (
            lambda manifest, pair: pair["target_attempts"][0].update(
                landed_on_intended_token=False
            ),
            "landing flag",
        ),
        (
            lambda manifest, pair: pair["edited"].update(editable_text="corrupted"),
            "replayed edited text",
        ),
        (
            lambda manifest, pair: pair["aligned_words"][0].update(target_ranks=[]),
            "target_ranks",
        ),
        (
            lambda manifest, pair: pair["target_attempts"][0].update(relevance="inf"),
            "finite relevance",
        ),
        (_break_cumulative_shift_invariant, "cumulative-shift"),
        (_change_seed_input_but_preserve_declared_alignment, "seeded Table 4"),
    ),
)
def test_invalid_or_partial_input_is_rejected_without_publishing_output(
    tmp_path: Path,
    mutation: object,
    error: str,
) -> None:
    pairs_root = tmp_path / "pairs"
    question = "alpha beta gamma delta"
    row = _pair(
        "one",
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        question=question,
        choices=None,
        gold_answer="4",
        attempts=[_attempt(rank) for rank in range(1, 5)],
        aligned_words=[
            _aligned_word(question, word, word_index=index, target_ranks=(index + 1,))
            for index, word in enumerate(question.split())
        ],
    )
    source = pairs_root / "cell"
    _write_source(
        source,
        [row],
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
    )
    manifest = json.loads((source / "run.json").read_text())
    pair = json.loads((source / "pairs.jsonl").read_text())
    mutation(manifest, pair)  # type: ignore[operator]
    _write_json(source / "run.json", manifest)
    (source / "pairs.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
    output_dir = tmp_path / "audit"

    with pytest.raises(TargetingFidelityAuditError, match=error):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(pairs_root=pairs_root, output_dir=output_dir)
        )

    assert not output_dir.exists()


def test_duplicate_setting_sources_are_rejected(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    question = "alpha"
    for source_name, sample_id in (("copy-a", "a"), ("copy-b", "b")):
        row = _pair(
            sample_id,
            model="test/model",
            benchmark="gsm8k",
            targeting="attribution-4",
            question=question,
            choices=None,
            gold_answer="1",
            attempts=[_attempt(1)],
            aligned_words=[_aligned_word(question, "alpha", word_index=0, target_ranks=(1,))],
        )
        _write_source(
            pairs_root / source_name,
            [row],
            model="test/model",
            benchmark="gsm8k",
            targeting="attribution-4",
        )

    with pytest.raises(TargetingFidelityAuditError, match="duplicate input cell"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "audit",
            )
        )


def test_normalized_model_aliases_cannot_hide_a_duplicate_cell(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    question = "alpha"
    for source_name, model in (("qualified", "org/model"), ("basename", "model")):
        row = _pair(
            source_name,
            model=model,
            benchmark="gsm8k",
            targeting="attribution-4",
            question=question,
            choices=None,
            gold_answer="1",
            attempts=[_attempt(1)],
            aligned_words=[_aligned_word(question, "alpha", word_index=0, target_ranks=(1,))],
        )
        _write_source(
            pairs_root / source_name,
            [row],
            model=model,
            benchmark="gsm8k",
            targeting="attribution-4",
        )

    with pytest.raises(TargetingFidelityAuditError, match="normalized input cell"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "audit",
            )
        )


def test_random_control_excludes_attribution_ranks_one_through_four(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "pairs"
    _fixture_inputs(pairs_root)
    source = pairs_root / "m-random-mmlu" / "pairs.jsonl"
    pair = json.loads(source.read_text())
    pair["target_attempts"][0]["attribution_rank"] = 4
    source.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    with pytest.raises(TargetingFidelityAuditError, match="Random-4.*top four"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "audit",
            )
        )


def test_paired_targeting_arms_require_matching_dataset_and_model_provenance(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "pairs"
    _fixture_inputs(pairs_root)
    manifest_path = pairs_root / "m-random-mmlu" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["model_revision"] = "other-model-revision"
    _write_json(manifest_path, manifest)

    with pytest.raises(TargetingFidelityAuditError, match="paired targeting provenance"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "audit",
            )
        )


def test_jsonl_rejects_nonstandard_constants_and_duplicate_keys(tmp_path: Path) -> None:
    for corruption, expected_error in (
        ("constant", "non-finite JSON constant"),
        ("key", "duplicate JSON key"),
    ):
        case_root = tmp_path / corruption / "pairs"
        question = "alpha beta"
        row = _pair(
            "one",
            model="test/model",
            benchmark="gsm8k",
            targeting="attribution-4",
            question=question,
            choices=None,
            gold_answer="1",
            attempts=[_attempt(1)],
            aligned_words=[_aligned_word(question, "alpha", word_index=0, target_ranks=(1,))],
        )
        source = case_root / "cell"
        _write_source(
            source,
            [row],
            model="test/model",
            benchmark="gsm8k",
            targeting="attribution-4",
        )
        serialized = (source / "pairs.jsonl").read_text(encoding="utf-8")
        if corruption == "constant":
            serialized = serialized.replace('"relevance": 1.0', '"relevance": NaN', 1)
        else:
            serialized = serialized.replace(
                '"sample_id": "one"',
                '"sample_id": "one", "sample_id": "duplicate"',
                1,
            )
        (source / "pairs.jsonl").write_text(serialized, encoding="utf-8")

        with pytest.raises(TargetingFidelityAuditError, match=expected_error):
            run_targeting_fidelity_audit(
                TargetingFidelityAuditConfig(
                    pairs_root=case_root,
                    output_dir=tmp_path / corruption / "audit",
                )
            )


def test_pair_rows_must_preserve_the_producer_sample_order(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    question = "alpha beta"
    rows = [
        _pair(
            sample_id,
            model="test/model",
            benchmark="gsm8k",
            targeting="attribution-4",
            question=question,
            choices=None,
            gold_answer="1",
            attempts=[_attempt(1)],
            aligned_words=[_aligned_word(question, "alpha", word_index=0, target_ranks=(1,))],
        )
        for sample_id in ("z", "a")
    ]
    _write_source(
        pairs_root / "cell",
        rows,
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
    )

    with pytest.raises(TargetingFidelityAuditError, match="strictly sorted"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "audit",
            )
        )


def test_expected_seed_is_enforced_and_an_existing_output_is_never_overwritten(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "pairs"
    question = "alpha"
    row = _pair(
        "one",
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        question=question,
        choices=None,
        gold_answer="1",
        attempts=[_attempt(1)],
        aligned_words=[_aligned_word(question, "alpha", word_index=0, target_ranks=(1,))],
        seed=7,
    )
    _write_source(
        pairs_root / "cell",
        [row],
        model="test/model",
        benchmark="gsm8k",
        targeting="attribution-4",
        seed=7,
    )

    with pytest.raises(TargetingFidelityAuditError, match="expected seed 42"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=tmp_path / "wrong-seed",
            )
        )

    sensitivity_output = tmp_path / "sensitivity"
    run_targeting_fidelity_audit(
        TargetingFidelityAuditConfig(
            pairs_root=pairs_root,
            output_dir=sensitivity_output,
            expected_seed=7,
        )
    )
    sensitivity_manifest = json.loads((sensitivity_output / "run.json").read_text())
    assert sensitivity_manifest["paper_comparison"]["status"] == "not_comparable"
    assert sensitivity_manifest["paper_comparison"]["checks"]["paper_seed_42"] is False

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(
                pairs_root=pairs_root,
                output_dir=existing,
                expected_seed=7,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_a_dangling_output_symlink_is_treated_as_an_existing_target(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    _fixture_inputs(pairs_root)
    output_dir = tmp_path / "audit"
    os.symlink(tmp_path / "missing-target", output_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        run_targeting_fidelity_audit(
            TargetingFidelityAuditConfig(pairs_root=pairs_root, output_dir=output_dir)
        )

    assert output_dir.is_symlink()


def test_machine_readable_outputs_are_deterministic_across_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first" / "pairs"
    second_root = tmp_path / "second" / "pairs"
    _fixture_inputs(first_root)
    _fixture_inputs(second_root)
    first_output = tmp_path / "first-audit"
    second_output = tmp_path / "second-audit"

    run_targeting_fidelity_audit(
        TargetingFidelityAuditConfig(pairs_root=first_root, output_dir=first_output)
    )
    run_targeting_fidelity_audit(
        TargetingFidelityAuditConfig(pairs_root=second_root, output_dir=second_output)
    )

    for filename in (
        "targeting_fidelity_records.jsonl",
        "targeting_fidelity.csv",
        "operation_counts.json",
    ):
        assert (first_output / filename).read_bytes() == (second_output / filename).read_bytes()


def test_cli_runs_the_cpu_audit_and_catalog_marks_it_implemented(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs_root = tmp_path / "pairs"
    _fixture_inputs(pairs_root)
    output_dir = tmp_path / "audit"

    assert (
        main(
            [
                "targeting-fidelity-audit",
                "--pairs-root",
                str(pairs_root),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert "audited 3 pair(s) across 2 setting(s)" in capsys.readouterr().out
    assert get_experiment("targeting-fidelity-audit").status == "implemented"


def test_cli_reports_validation_errors_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "targeting-fidelity-audit",
                "--pairs-root",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "audit"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "targeting-fidelity-audit: error:" in captured.err
    assert "does not exist" in captured.err

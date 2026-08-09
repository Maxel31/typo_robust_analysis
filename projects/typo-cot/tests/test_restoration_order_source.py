"""Source-cohort contracts for the Appendix E restoration-order audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import typo_cot.experiments.restoration_order_accuracy.source as source_module
from typo_cot.experiments.input_corrector_audit.source import InputCorrectorSourceError
from typo_cot.experiments.restoration_order_accuracy.source import (
    load_restoration_order_source,
)

MODEL = "google/gemma-3-4b-it"
BENCHMARK = "gsm8k"


def _attempt(token_index: int, relevance: float) -> dict[str, object]:
    return {
        "selection_rank": token_index,
        "target_token_index": token_index,
        "relevance": relevance,
    }


def _record(
    sample_id: str,
    *,
    clean_correct: bool = True,
    edited_correct: bool = False,
    clean_text: str = "aa bb cc dd ee",
    edited_text: str = "ax bx cx dx ee",
    attempts: tuple[dict[str, object], ...] | None = None,
    fresh_k0_correct: bool = False,
    fresh_k4_correct: bool = True,
) -> dict[str, object]:
    if attempts is None:
        attempts = tuple(
            _attempt(token_index, relevance)
            for token_index, relevance in zip(
                (10, 20, 30, 40),
                (4.0, 3.0, 2.0, 1.0),
                strict=True,
            )
        )
    clean_value = "2" if clean_correct else "3"
    edited_value = "2" if edited_correct else "3"

    def source_arm(editable_text: str, value: str, *, correct: bool) -> dict[str, object]:
        return {
            "editable_text": editable_text,
            "continuation": f"Reasoning. The answer is {value}.",
            "continuation_token_count": 8,
            "termination": "eos",
            "answer": {
                "value": value,
                "is_extracted": True,
                "is_correct": correct,
                "method": "primary:pattern_1",
                "primary_method": "pattern_1",
            },
        }

    return {
        "schema_version": "prepare-edited-pairs/v1",
        "sample_id": sample_id,
        "model": MODEL,
        "benchmark": BENCHMARK,
        "targeting": "attribution-4",
        "seed": 42,
        "num_edits_requested": 4,
        "num_target_attempts": len(attempts),
        "target_attempts": list(attempts),
        "gold_answer": "2",
        "clean": source_arm(clean_text, clean_value, correct=clean_correct),
        "edited": source_arm(edited_text, edited_value, correct=edited_correct),
        "answer_changed": clean_value != edited_value,
        # These simulate newly generated endpoint diagnostics. They must never
        # redefine the cohort selected from the archived source outcomes above.
        "fresh_endpoint_diagnostics": {
            "k0_is_correct": fresh_k0_correct,
            "k4_is_correct": fresh_k4_correct,
        },
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _StrictPreparedSource:
    records: tuple[dict[str, object], ...]
    fail_on_assert: bool = False
    assert_calls: int = 0
    model: str = MODEL
    benchmark: str = BENCHMARK
    model_revision: str = "1" * 40
    pairs_path: Path = Path("/strict/source/pairs.jsonl")
    run_path: Path = Path("/strict/source/run.json")
    pairs_sha256: str = "2" * 64
    run_sha256: str = "3" * 64
    ordered_sample_ids_sha256: str = "4" * 64
    dataset_records_sha256: str = "5" * 64
    assertion_log: list[str] = field(default_factory=list)

    def assert_unchanged(self) -> None:
        self.assert_calls += 1
        self.assertion_log.append("assert_unchanged")
        if self.fail_on_assert:
            raise InputCorrectorSourceError("strict prepared source changed")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "model_revision": self.model_revision,
            "records": len(self.records),
            "pairs_sha256": self.pairs_sha256,
            "run_sha256": self.run_sha256,
            "ordered_sample_ids_sha256": self.ordered_sample_ids_sha256,
            "dataset_records_sha256": self.dataset_records_sha256,
        }


def _install_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
    strict_source: _StrictPreparedSource,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def load(
        pairs_path: Path,
        *,
        model: str,
        benchmark: str,
        require_paper_cohort_size: bool = True,
    ) -> _StrictPreparedSource:
        calls.append(
            {
                "pairs_path": Path(pairs_path),
                "model": model,
                "benchmark": benchmark,
                "require_paper_cohort_size": require_paper_cohort_size,
            }
        )
        return strict_source

    monkeypatch.setattr(source_module, "load_input_corrector_source", load)
    return calls


def test_loader_delegates_to_the_strict_prepare_source_contract_and_rechecks_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = _StrictPreparedSource(records=(_record("eligible"),))
    calls = _install_strict_loader(monkeypatch, strict)
    pairs_path = tmp_path / "pairs.jsonl"

    source = load_restoration_order_source(
        pairs_path,
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert calls == [
        {
            "pairs_path": pairs_path,
            "model": MODEL,
            "benchmark": BENCHMARK,
            "require_paper_cohort_size": True,
        }
    ]
    assert strict.assert_calls == 1
    assert [record["sample_id"] for record in source.records] == ["eligible"]
    source.assert_unchanged()
    assert strict.assert_calls == 2


def test_loader_fails_if_the_strict_source_changes_during_derived_cohort_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = _StrictPreparedSource(records=(_record("eligible"),), fail_on_assert=True)
    _install_strict_loader(monkeypatch, strict)

    with pytest.raises(InputCorrectorSourceError, match="source changed"):
        load_restoration_order_source(
            tmp_path / "pairs.jsonl",
            model=MODEL,
            benchmark=BENCHMARK,
        )


def test_selection_uses_only_source_clean_correct_and_four_edit_wrong_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record(
            "archived-eligible-fresh-endpoints-disagree",
            fresh_k0_correct=True,
            fresh_k4_correct=False,
        ),
        _record("clean-wrong", clean_correct=False, edited_correct=False),
        _record("four-edit-still-correct", clean_correct=True, edited_correct=True),
    )
    strict = _StrictPreparedSource(records=records)
    _install_strict_loader(monkeypatch, strict)

    source = load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert source.source_selected_count == 1
    assert source.separable_count == 1
    assert source.to_dict()["source_selected"] == 1
    assert "archived_selected" not in source.to_dict()
    assert [record["sample_id"] for record in source.records] == [
        "archived-eligible-fresh-endpoints-disagree"
    ]
    assert source.records[0]["fresh_endpoint_diagnostics"] == {
        "k0_is_correct": True,
        "k4_is_correct": False,
    }


@pytest.mark.parametrize(
    ("termination", "token_count"),
    (("unknown", 8), ("length-cap", 8)),
)
def test_loader_rejects_invalid_source_termination_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    token_count: int,
) -> None:
    record = _record("invalid-termination")
    clean = record["clean"]
    assert isinstance(clean, dict)
    clean["termination"] = termination
    clean["continuation_token_count"] = token_count
    strict = _StrictPreparedSource(records=(record,))
    _install_strict_loader(monkeypatch, strict)

    with pytest.raises(ValueError, match="termination|length-cap"):
        load_restoration_order_source(
            tmp_path / "pairs.jsonl",
            model=MODEL,
            benchmark=BENCHMARK,
        )


def test_loader_recomputes_final_pdf_source_outcomes_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("stale-primary-only-metadata")
    clean = record["clean"]
    assert isinstance(clean, dict)
    clean["continuation"] = "Reasoning only. **2**"
    clean["continuation_token_count"] = 4
    clean["answer"] = {
        "value": "",
        "is_extracted": False,
        "is_correct": False,
        "method": "unextractable",
        "primary_method": "no_match",
    }
    strict = _StrictPreparedSource(records=(record,))
    _install_strict_loader(monkeypatch, strict)

    with pytest.raises(ValueError, match="final-PDF|recomputed|source answer"):
        load_restoration_order_source(
            tmp_path / "pairs.jsonl",
            model=MODEL,
            benchmark=BENCHMARK,
        )


def test_loader_revalidates_edited_outcome_when_clean_is_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("clean-wrong-stale-edited", clean_correct=False)
    edited = record["edited"]
    assert isinstance(edited, dict)
    edited["continuation"] = "Reasoning. The answer is 2."
    strict = _StrictPreparedSource(records=(record,))
    _install_strict_loader(monkeypatch, strict)

    with pytest.raises(ValueError, match="edited.*final-PDF|edited.*source answer"):
        load_restoration_order_source(
            tmp_path / "pairs.jsonl",
            model=MODEL,
            benchmark=BENCHMARK,
        )


def test_loader_allows_positional_fallback_when_eos_is_the_512th_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("eos-exactly-at-cap")
    clean = record["clean"]
    assert isinstance(clean, dict)
    clean["continuation"] = (
        "The computation is complete and the final total is 2 dollars."
    )
    clean["continuation_token_count"] = 512
    clean["termination"] = "eos"
    clean["answer"] = {
        "value": "2",
        "is_extracted": True,
        "is_correct": True,
        "method": "fallback:N5_tail_number",
        "primary_method": "no_match",
    }
    strict = _StrictPreparedSource(records=(record,))
    _install_strict_loader(monkeypatch, strict)

    source = load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert source.source_selected_count == 1


def test_source_plan_uses_the_protocol_seed_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = _StrictPreparedSource(records=(_record("eligible"),))
    _install_strict_loader(monkeypatch, strict)
    observed: list[int] = []
    real_builder = source_module.build_restoration_plan

    def capture_seed(**kwargs: object) -> object:
        observed.append(int(kwargs["seed"]))
        return real_builder(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(source_module, "PAPER_SEED", 7, raising=False)
    monkeypatch.setattr(source_module, "build_restoration_plan", capture_seed)

    load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert observed == [7]


def test_inseparable_source_selected_items_are_excluded_with_explicit_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inseparable = _record(
        "inseparable",
        clean_text="abXYef",
        edited_text="abPQef",
        attempts=(_attempt(10, 2.0), _attempt(20, 1.0)),
    )
    two_realized = _record(
        "two-realized",
        clean_text="aa bb cc",
        edited_text="ax bx cc",
        attempts=(_attempt(10, 2.0), _attempt(20, 1.0)),
    )
    strict = _StrictPreparedSource(records=(inseparable, two_realized))
    _install_strict_loader(monkeypatch, strict)

    source = load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert source.source_selected_count == 2
    assert source.separable_count == 1
    assert [record["sample_id"] for record in source.records] == ["two-realized"]
    assert any(
        exclusion.sample_id == "inseparable"
        and exclusion.reason == "inseparable-edit-groups"
        for exclusion in source.exclusions
    )


def test_more_than_four_edit_groups_are_an_explicit_source_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(
        "five-groups",
        clean_text="aa bb cc dd ee ff",
        edited_text="ax bx cx dx ex ff",
        attempts=tuple(_attempt(index * 10, float(6 - index)) for index in range(1, 6)),
    )
    strict = _StrictPreparedSource(records=(record,))
    _install_strict_loader(monkeypatch, strict)

    source = load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
    )

    assert source.source_selected_count == 1
    assert source.separable_count == 0
    assert source.records == ()
    assert [item.reason for item in source.exclusions] == ["inseparable-edit-groups"]


def test_limit_is_applied_after_archived_selection_and_inseparable_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record("00-ineligible", clean_correct=False),
        _record(
            "01-inseparable",
            clean_text="abXYef",
            edited_text="abPQef",
            attempts=(_attempt(10, 2.0), _attempt(20, 1.0)),
        ),
        _record("02-first-runnable"),
        _record("03-second-runnable"),
    )
    strict = _StrictPreparedSource(records=records)
    _install_strict_loader(monkeypatch, strict)

    source = load_restoration_order_source(
        tmp_path / "pairs.jsonl",
        model=MODEL,
        benchmark=BENCHMARK,
        limit=1,
    )

    assert source.source_selected_count == 3
    assert source.separable_count == 2
    assert [record["sample_id"] for record in source.records] == ["02-first-runnable"]
    assert source.selected_sample_ids_sha256 == _canonical_sha256(["02-first-runnable"])

"""Tests for quant_typo_neuron.robustness_evaluation.schema (TDD – written before implementation)."""
from __future__ import annotations

import pytest

from quant_typo_neuron.contracts import ItemResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fp16_item() -> ItemResult:
    return ItemResult(
        model="llama-3.2-1b-instruct",
        method="fp16",
        bit=None,
        typo_type="sub_keyboard",
        eps=1,
        dataset="gsm8k",
        seed=0,
        item_id="gsm8k-0001",
        correct_clean=1,
        correct_typo=0,
        conf=0.85,
    )


def _make_gptq_item() -> ItemResult:
    return ItemResult(
        model="llama-3.1-8b-instruct",
        method="gptq",
        bit=4,
        typo_type="insert",
        eps=0.10,
        dataset="mmlu",
        seed=1,
        item_id="mmlu-0042",
        correct_clean=0,
        correct_typo=1,
        conf=None,
    )


# ---------------------------------------------------------------------------
# Test: write_items / read_items roundtrip
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_path):
    from quant_typo_neuron.robustness_evaluation.schema import read_items, write_items

    items = [_make_fp16_item(), _make_gptq_item()]
    out = tmp_path / "subdir" / "items.jsonl"
    write_items(out, items)

    assert out.exists(), "write_items must create the file"
    recovered = read_items(out)
    assert recovered == items, "read_items must reproduce the original list"


def test_write_creates_parent_dirs(tmp_path):
    from quant_typo_neuron.robustness_evaluation.schema import read_items, write_items

    items = [_make_fp16_item()]
    deep = tmp_path / "a" / "b" / "c" / "items.jsonl"
    write_items(deep, items)
    assert deep.exists()
    assert read_items(deep) == items


# ---------------------------------------------------------------------------
# Test: to_long_records length and structure
# ---------------------------------------------------------------------------

def test_to_long_records_length():
    from quant_typo_neuron.robustness_evaluation.schema import to_long_records

    items = [_make_fp16_item(), _make_gptq_item()]
    records = to_long_records(items)
    assert len(records) == 4, "to_long_records must return 2 rows per item"


def test_to_long_records_fp16_row():
    """fp16 item -> quantization=0, correct mapped from correct_clean/correct_typo."""
    from quant_typo_neuron.robustness_evaluation.schema import to_long_records

    item = _make_fp16_item()
    rows = to_long_records([item])
    assert len(rows) == 2

    clean_row = next(r for r in rows if r["typo_present"] == 0)
    typo_row = next(r for r in rows if r["typo_present"] == 1)

    # quantization flag: fp16 -> 0
    assert clean_row["quantization"] == 0
    assert typo_row["quantization"] == 0

    # correct mapping
    assert clean_row["correct"] == item.correct_clean  # 1
    assert typo_row["correct"] == item.correct_typo    # 0

    # shared fields forwarded
    for row in rows:
        assert row["model"] == item.model
        assert row["method"] == item.method
        assert row["bit"] == item.bit
        assert row["dataset"] == item.dataset
        assert row["typo_type"] == item.typo_type
        assert row["eps"] == item.eps
        assert row["seed"] == item.seed
        assert row["item_id"] == item.item_id


def test_to_long_records_gptq_row():
    """gptq item -> quantization=1."""
    from quant_typo_neuron.robustness_evaluation.schema import to_long_records

    item = _make_gptq_item()
    rows = to_long_records([item])

    for row in rows:
        assert row["quantization"] == 1, "non-fp16 method must yield quantization=1"

    clean_row = next(r for r in rows if r["typo_present"] == 0)
    typo_row = next(r for r in rows if r["typo_present"] == 1)

    assert clean_row["correct"] == item.correct_clean  # 0
    assert typo_row["correct"] == item.correct_typo    # 1


# ---------------------------------------------------------------------------
# Test: to_long_df shape and columns
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "typo_present",
    "correct",
    "quantization",
    "model",
    "method",
    "bit",
    "dataset",
    "typo_type",
    "eps",
    "seed",
    "item_id",
}


def test_to_long_df_shape_and_columns():
    from quant_typo_neuron.robustness_evaluation.schema import to_long_df

    items = [_make_fp16_item(), _make_gptq_item()]
    df = to_long_df(items)

    assert df.shape == (4, len(EXPECTED_COLUMNS)), (
        f"Expected shape (4, {len(EXPECTED_COLUMNS)}), got {df.shape}"
    )
    assert set(df.columns) == EXPECTED_COLUMNS


def test_to_long_df_empty():
    from quant_typo_neuron.robustness_evaluation.schema import to_long_df

    df = to_long_df([])
    assert df.shape[0] == 0
    assert set(df.columns) == EXPECTED_COLUMNS

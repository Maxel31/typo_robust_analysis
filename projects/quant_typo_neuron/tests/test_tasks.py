"""Tests for quant_typo_neuron.data.tasks — NO network, local fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_typo_neuron.data.tasks import TaskItem, available_tasks, load_task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture()
def gsm8k_fixture(tmp_path: Path) -> Path:
    """Minimal gsm8k-style jsonl with question/answer keys."""
    records = [
        {"id": "gsm8k-0000", "question": "What is 1+1?", "answer": "2"},
        {"id": "gsm8k-0001", "question": "What is 2+2?", "answer": "4"},
        {"id": "gsm8k-0002", "question": "What is 3+3?", "answer": "6"},
    ]
    split_file = tmp_path / "gsm8k" / "test.jsonl"
    _write_jsonl(split_file, records)
    return tmp_path


@pytest.fixture()
def prompt_label_fixture(tmp_path: Path) -> Path:
    """Fixture using prompt/label keys (alternate field names)."""
    records = [
        {"item_id": "bbh-0", "prompt": "Solve: 5-3=?", "label": "2"},
        {"item_id": "bbh-1", "prompt": "Solve: 9-4=?", "label": "5"},
    ]
    split_file = tmp_path / "bbh" / "test.jsonl"
    _write_jsonl(split_file, records)
    return tmp_path


@pytest.fixture()
def target_fixture(tmp_path: Path) -> Path:
    """Fixture using target key for the answer field."""
    records = [
        {"prompt": "Define: cat", "target": "a small domesticated animal"},
    ]
    split_file = tmp_path / "mmlu" / "test.jsonl"
    _write_jsonl(split_file, records)
    return tmp_path


# ---------------------------------------------------------------------------
# available_tasks
# ---------------------------------------------------------------------------


def test_available_tasks_contains_all_five():
    tasks = available_tasks()
    assert isinstance(tasks, tuple)
    expected = {"gsm8k", "bbh", "mmlu", "longgen", "wordnet_id"}
    assert expected <= set(tasks), f"missing tasks: {expected - set(tasks)}"


# ---------------------------------------------------------------------------
# load_task — field mapping
# ---------------------------------------------------------------------------


def test_load_task_question_answer_keys(gsm8k_fixture: Path):
    """question -> prompt, answer -> answer, id -> item_id."""
    items = load_task("gsm8k", split="test", data_dir=gsm8k_fixture)
    assert len(items) == 3
    item = items[0]
    assert isinstance(item, TaskItem)
    assert item.prompt == "What is 1+1?"
    assert item.answer == "2"
    assert item.item_id == "gsm8k-0000"
    assert item.task == "gsm8k"


def test_load_task_prompt_label_keys(prompt_label_fixture: Path):
    """prompt -> prompt, label -> answer, item_id -> item_id."""
    items = load_task("bbh", split="test", data_dir=prompt_label_fixture)
    assert len(items) == 2
    assert items[0].prompt == "Solve: 5-3=?"
    assert items[0].answer == "2"
    assert items[0].item_id == "bbh-0"
    assert items[0].task == "bbh"


def test_load_task_target_key(target_fixture: Path):
    """target -> answer; fallback item_id is index when no id field."""
    items = load_task("mmlu", split="test", data_dir=target_fixture)
    assert len(items) == 1
    assert items[0].answer == "a small domesticated animal"
    assert items[0].item_id == "0"  # index fallback


# ---------------------------------------------------------------------------
# load_task — limit
# ---------------------------------------------------------------------------


def test_load_task_limit_truncates(gsm8k_fixture: Path):
    items = load_task("gsm8k", split="test", limit=2, data_dir=gsm8k_fixture)
    assert len(items) == 2


def test_load_task_limit_none_returns_all(gsm8k_fixture: Path):
    items = load_task("gsm8k", split="test", limit=None, data_dir=gsm8k_fixture)
    assert len(items) == 3


# ---------------------------------------------------------------------------
# load_task — unknown name raises ValueError
# ---------------------------------------------------------------------------


def test_load_task_unknown_name_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown task"):
        load_task("not_a_real_task", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# TaskItem structure
# ---------------------------------------------------------------------------


def test_task_item_is_dataclass(gsm8k_fixture: Path):
    items = load_task("gsm8k", split="test", data_dir=gsm8k_fixture)
    item = items[0]
    assert hasattr(item, "item_id")
    assert hasattr(item, "prompt")
    assert hasattr(item, "answer")
    assert hasattr(item, "task")

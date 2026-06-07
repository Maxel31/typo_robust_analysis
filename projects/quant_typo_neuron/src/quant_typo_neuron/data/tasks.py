"""Uniform task dataset loaders for M2 evaluation harness.

Supports: gsm8k, bbh, mmlu, longgen, wordnet_id.

Local JSONL files are preferred; HuggingFace datasets is used as a lazy
fallback when a local file is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from typo_utils.data.loaders import load_jsonl
from typo_utils.paths import datasets_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWN_TASKS: tuple[str, ...] = ("gsm8k", "bbh", "mmlu", "longgen", "wordnet_id")

# HuggingFace dataset identifiers for known tasks (used only as lazy fallback)
_HF_DATASET_IDS: dict[str, str] = {
    "gsm8k": "openai/gsm8k",
    "bbh": "lukaemon/bbh",
    "mmlu": "cais/mmlu",
    "longgen": "tau/scrolls",
    # wordnet_id is local-only; no HF fallback
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TaskItem:
    """A single evaluation item."""

    item_id: str
    prompt: str
    answer: str
    task: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_tasks() -> tuple[str, ...]:
    """Return the tuple of supported task names."""
    return _KNOWN_TASKS


def load_task(
    name: str,
    split: str = "test",
    limit: int | None = None,
    data_dir: str | Path | None = None,
) -> list[TaskItem]:
    """Load a task dataset as a list of :class:`TaskItem`.

    Resolution order:
    1. Local JSONL at ``<data_dir or datasets_dir()>/<name>/<split>.jsonl``.
    2. HuggingFace ``datasets`` (lazy import) — only for tasks in
       ``_HF_DATASET_IDS`` and only when no local file is found.

    Parameters
    ----------
    name:
        Task name; must be one of :func:`available_tasks`.
    split:
        Dataset split (default ``"test"``).
    limit:
        If given, truncate the loaded items to at most *limit* records.
    data_dir:
        Override the base datasets directory. Useful in tests.

    Returns
    -------
    list[TaskItem]

    Raises
    ------
    ValueError
        When *name* is not in :func:`available_tasks`.
    FileNotFoundError
        When the local file is absent and no HF fallback is available.
    """
    if name not in _KNOWN_TASKS:
        raise ValueError(
            f"unknown task: {name!r}. Available: {_KNOWN_TASKS}"
        )

    base = Path(data_dir) if data_dir is not None else datasets_dir()
    local_path = base / name / f"{split}.jsonl"

    if local_path.exists():
        raw_records = load_jsonl(local_path)
        items = [_record_to_item(rec, idx, name) for idx, rec in enumerate(raw_records)]
    else:
        # Lazy HuggingFace fallback — never imported in tests (local file always present)
        if name not in _HF_DATASET_IDS:
            raise FileNotFoundError(
                f"No local file at {local_path} and no HuggingFace fallback for task {name!r}."
            )
        items = _load_from_hf(name, split)

    if limit is not None:
        items = items[:limit]

    return items


def apply_typo_to_prompts(
    items: Sequence[TaskItem],
    typo_type: str,
    eps: float | int,
    seed: int,
) -> list[TaskItem]:
    """Apply typo perturbations to the prompt of each :class:`TaskItem`.

    The ``typo_utils.data.typo_real`` module is imported lazily so that tests
    (and code on branches where that module may not yet exist) are not broken.

    Parameters
    ----------
    items:
        Source task items.
    typo_type:
        One of the real-typo types (e.g. ``"sub_keyboard"``, ``"insert"``,
        ``"delete"``, ``"transpose"``).
    eps:
        Perturbation intensity — either a character count (int >= 1) or a
        proportion (float in (0, 1]).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[TaskItem]
        New TaskItem objects with perturbed prompts; other fields unchanged.
    """
    try:
        from typo_utils.data.typo_real import apply_typo  # type: ignore[import]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "typo_utils.data.typo_real is not available yet. "
            "Make sure the robustness_evaluation-typo-generators branch has been merged."
        ) from exc

    return [
        TaskItem(
            item_id=item.item_id,
            prompt=apply_typo(item.prompt, typo_type=typo_type, eps=eps, seed=seed),
            answer=item.answer,
            task=item.task,
        )
        for item in items
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_to_item(record: dict, idx: int, task: str) -> TaskItem:
    """Map a raw JSONL record dict to a :class:`TaskItem`.

    Field priority:
    - ``item_id``: ``id`` > ``item_id`` > str(idx)
    - ``prompt``: ``question`` > ``prompt``
    - ``answer``: ``answer`` > ``label`` > ``target``
    """
    # item_id
    if "id" in record:
        item_id = str(record["id"])
    elif "item_id" in record:
        item_id = str(record["item_id"])
    else:
        item_id = str(idx)

    # prompt
    if "question" in record:
        prompt = str(record["question"])
    elif "prompt" in record:
        prompt = str(record["prompt"])
    else:
        raise KeyError(
            f"Record at index {idx} has no 'question' or 'prompt' field: {list(record.keys())}"
        )

    # answer
    if "answer" in record:
        answer = str(record["answer"])
    elif "label" in record:
        answer = str(record["label"])
    elif "target" in record:
        answer = str(record["target"])
    else:
        raise KeyError(
            f"Record at index {idx} has no 'answer', 'label', or 'target' field: {list(record.keys())}"
        )

    return TaskItem(item_id=item_id, prompt=prompt, answer=answer, task=task)


def _load_from_hf(name: str, split: str) -> list[TaskItem]:  # pragma: no cover
    """Lazy HuggingFace dataset fallback -- only called when local file is absent."""
    import datasets as hf_datasets  # type: ignore[import]  # noqa: PLC0415

    hf_id = _HF_DATASET_IDS[name]
    ds = hf_datasets.load_dataset(hf_id, split=split)
    items = []
    for idx, record in enumerate(ds):
        items.append(_record_to_item(dict(record), idx, name))
    return items

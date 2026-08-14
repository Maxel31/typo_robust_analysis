"""The concrete training monitor executes both frozen corpus axes."""

from __future__ import annotations

import re
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from typo_robust_training.data.records import TypoEdit
from typo_robust_training.evaluation.data import EvaluationCorpusRecord
from typo_robust_training.training.runtime import HuggingFaceAdapterTrainingRuntime


class _Tokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"<bos>": 1}

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[object]]:
        pieces = [
            (match.group(), match.span()) for match in re.finditer(r"[A-Za-z]+|[^\w\s]", text)
        ]
        ids = [1]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for piece, span in pieces:
            ids.append(self.vocabulary.setdefault(piece, len(self.vocabulary) + 1))
            offsets.append(span)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


class _LogitModel(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32)
        logits[..., 0] = input_ids.sum(dim=1, keepdim=True).float() * self.scale
        return SimpleNamespace(logits=logits)


def _record(*, natural: bool) -> EvaluationCorpusRecord:
    clean = "The airport works."
    typo = "The arport works."
    edits = (
        (
            TypoEdit(
                operation="deletion",
                clean_word="airport",
                typo_word="arport",
                clean_char_span=(4, 11),
                typo_char_span=(4, 10),
            ),
        )
        if natural
        else ()
    )
    return EvaluationCorpusRecord(
        record_id=("b" if natural else "a") * 64,
        kind="natural" if natural else "clean-corpus",
        source="github_typo_corpus" if natural else "fineweb_edu",
        source_revision="c" * 40,
        source_split="train",
        source_id="natural" if natural else "clean",
        group_id="group",
        role="tune",
        clean_text=clean,
        typo_text=typo if natural else None,
        edits=edits,
        metadata=MappingProxyType({}),
    )


def test_concrete_monitor_runs_and_caches_frozen_teacher_natural_gap() -> None:
    runtime = HuggingFaceAdapterTrainingRuntime.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime.protocol = SimpleNamespace(max_sequence_length=32)
    runtime.tokenizer = _Tokenizer()
    runtime.teacher = _LogitModel(0.1)
    runtime.student = _LogitModel(0.2)
    runtime.student.train()
    runtime._monitor_base_clean = None
    runtime._monitor_base_natural = None
    records = (_record(natural=False), _record(natural=True))

    first = runtime.monitor(records)
    first_teacher_calls = runtime.teacher.calls
    second = runtime.monitor(records)

    assert first["clean_documents"] == 1.0
    assert first["natural_pairs"] == 1.0
    assert first["clean_kl_nats_per_token"] >= 0.0
    assert first["fineweb_edu_ppl_ratio"] > 0.0
    assert second["base_natural_clean_typo_kl"] == pytest.approx(
        first["base_natural_clean_typo_kl"]
    )
    assert first_teacher_calls == 3
    assert runtime.teacher.calls == first_teacher_calls + 1
    assert runtime.student.training is True

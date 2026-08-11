"""Runtime helpers preserve token coordinates, KL direction, and whole-window patching."""

from __future__ import annotations

import re

import pytest
import torch

from typo_robust_training.evaluation.prompting import EvaluationPrompts
from typo_robust_training.evaluation.runtime import (
    prompt_tokenization_profile,
    teacher_forced_kl_readout,
    window_patched_forward,
)
from typo_robust_training.localization.prompting import PromptSide


class _PieceTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        del kwargs
        pieces: list[tuple[str, tuple[int, int]]] = []
        for match in re.finditer(r"[A-Za-z]+|[^\w\s]", text):
            word = match.group()
            start, stop = match.span()
            if word == "airport":
                pieces.extend([("air", (start, start + 3)), ("port", (start + 3, stop))])
            elif word == "arport":
                pieces.extend(
                    [
                        ("ar", (start, start + 2)),
                        ("po", (start + 2, start + 4)),
                        ("rt", (start + 4, stop)),
                    ]
                )
            else:
                pieces.append((word, (start, stop)))
        return {
            "input_ids": [1, *range(2, len(pieces) + 2)],
            "attention_mask": [1] * (len(pieces) + 1),
            "offset_mapping": [(0, 0), *(span for _piece, span in pieces)],
        }


def test_prompt_profile_counts_subtokens_and_resolves_final_coordinates() -> None:
    prompts = EvaluationPrompts(
        record_id="a" * 64,
        answer=None,
        task_for_prompt=None,
        task_for_extractor=None,
        clean=PromptSide("The airport works.", ((4, 11),)),
        typo=PromptSide("The arport works.", ((4, 10),)),
    )

    profile = prompt_tokenization_profile(_PieceTokenizer(), prompts=prompts, max_tokens=32)

    assert profile.clean_positions == (3,)
    assert profile.typo_positions == (4,)
    assert profile.clean_subtoken_counts == (2,)
    assert profile.typo_subtoken_counts == (3,)
    assert profile.tokenization_stratum == "fragmentation-increased"
    assert len(profile.clean_input_ids) == 6
    assert len(profile.typo_input_ids) == 7


def test_teacher_forced_kl_is_clean_to_candidate_and_excludes_first_token() -> None:
    clean = torch.zeros(16, 3)
    candidate = clean.clone()
    candidate[0, 0] = 10.0
    candidate[1:, 1] = 2.0

    readout = teacher_forced_kl_readout(clean, candidate, teacher_forced_tokens=16)

    assert len(readout) == 15
    assert all(value > 0.0 for value in readout)
    assert teacher_forced_kl_readout(clean, clean, teacher_forced_tokens=16) == pytest.approx(
        (0.0,) * 15,
        abs=1e-7,
    )


class _Add(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value


def test_window_patch_overwrites_every_selected_layer_only_at_edited_position() -> None:
    layers = [_Add(1.0), _Add(2.0), _Add(4.0)]
    hidden = torch.zeros(1, 3, 1)
    donors = (
        torch.tensor([[100.0]]),
        torch.tensor([[250.0]]),
    )

    output = window_patched_forward(
        layers,
        layer_indices=(0, 1),
        positions=(1,),
        donor_values=donors,
        forward=lambda: layers[2](layers[1](layers[0](hidden))),
    )

    assert output[0, 1, 0].item() == pytest.approx(254.0)
    assert output[0, 0, 0].item() == pytest.approx(7.0)
    assert output[0, 2, 0].item() == pytest.approx(7.0)

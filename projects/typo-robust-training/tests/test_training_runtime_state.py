"""Runtime checkpoints restore CUDA RNG from CPU byte tensors."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.training.runtime import _cpu_cuda_rng_states


def test_cuda_rng_states_are_normalized_to_cpu_byte_tensors() -> None:
    states = _cpu_cuda_rng_states([torch.tensor([1, 2, 3], dtype=torch.uint8)])

    assert isinstance(states, tuple)
    assert len(states) == 1
    assert states[0].device.type == "cpu"
    assert states[0].dtype == torch.uint8
    assert states[0].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "states",
    [[], [torch.tensor([1], dtype=torch.int64)], ["not-a-tensor"]],
)
def test_cuda_rng_state_normalization_rejects_invalid_payloads(states: object) -> None:
    with pytest.raises(ValueError, match="CUDA RNG"):
        _cpu_cuda_rng_states(states)

"""Runtime checkpoints restore CUDA RNG from CPU byte tensors."""

from __future__ import annotations

import pytest
import torch

from typo_robust_training.training.runtime import (
    HuggingFaceAdapterTrainingRuntime,
    _cpu_cuda_rng_states,
)


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


def test_gpu_peak_telemetry_names_its_since_start_scope() -> None:
    class _Cuda:
        @staticmethod
        def memory_allocated() -> int:
            return 10

        @staticmethod
        def max_memory_allocated() -> int:
            return 20

        @staticmethod
        def memory_reserved() -> int:
            return 30

        @staticmethod
        def max_memory_reserved() -> int:
            return 40

    runtime = object.__new__(HuggingFaceAdapterTrainingRuntime)
    runtime._torch = type("_Torch", (), {"cuda": _Cuda()})()

    assert runtime.telemetry() == {
        "gpu_memory_allocated_bytes": 10,
        "gpu_peak_memory_allocated_bytes_since_start": 20,
        "gpu_memory_reserved_bytes": 30,
        "gpu_peak_memory_reserved_bytes_since_start": 40,
    }

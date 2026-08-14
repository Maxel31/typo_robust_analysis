"""Activation buffering has exact, deterministic resume boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from typo_robust_training.sae.runtime import HuggingFaceSaeRuntime


def _runtime() -> HuggingFaceSaeRuntime:
    runtime = object.__new__(HuggingFaceSaeRuntime)
    runtime._torch = torch
    runtime.protocol = SimpleNamespace(
        shuffle_buffer_activations=5,
        config_sha256="a" * 64,
    )

    def capture(text: str, *, layer_indices: tuple[int, ...]):
        length = int(text)
        base = torch.arange(length, dtype=torch.bfloat16).unsqueeze(1)
        return {layer: base + layer * 100 for layer in layer_indices}

    runtime._capture = capture
    return runtime


def test_activation_buffers_resume_inside_a_document_without_duplication() -> None:
    sources = (SimpleNamespace(clean_text="3"), SimpleNamespace(clean_text="6"))
    runtime = _runtime()
    buffers = tuple(
        runtime.iter_activation_buffers(
            sources,
            layer_indices=(5, 20),
            target_tokens=8,
        )
    )
    assert [buffer.tokens for buffer in buffers] == [5, 3]
    assert buffers[0].next_source_index == 1
    assert buffers[0].next_source_offset == 2
    assert buffers[1].next_source_index == 1
    assert buffers[1].next_source_offset == 5

    resumed = tuple(
        runtime.iter_activation_buffers(
            sources,
            layer_indices=(5, 20),
            target_tokens=3,
            start_source_index=buffers[0].next_source_index,
            start_source_offset=buffers[0].next_source_offset,
            start_buffer_index=1,
        )
    )
    assert len(resumed) == 1
    assert torch.equal(
        resumed[0].activations_by_layer[5],
        buffers[1].activations_by_layer[5],
    )

"""One-shot decoder-block patching across a complete layer window."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import ExitStack
from itertools import pairwise
from types import TracebackType
from typing import Self

import torch
from torch import nn

from typo_cot.experiments.layerwise_answer_patching.patching import (
    PrefillBlockOutputPatch,
)


class PrefillBlockOutputWindowPatch:
    """Patch every selected decoder block during one prompt prefill.

    All hooks are installed before the caller starts generation.  Each layer
    uses the same recipient token coordinates and the donor residual-stream
    rows captured after that corresponding decoder block.  The component
    :class:`PrefillBlockOutputPatch` instances deliberately ignore subsequent
    one-token cached-decoding forwards.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        layer_indices: Sequence[int],
        positions: Sequence[int],
        donor_values: Sequence[torch.Tensor],
    ) -> None:
        try:
            normalized_layers = tuple(layers)
        except TypeError as exc:
            raise TypeError("layers must be a sequence of torch modules") from exc
        if not normalized_layers or any(
            not isinstance(layer, nn.Module) for layer in normalized_layers
        ):
            raise ValueError("layers must contain one or more torch modules")

        try:
            normalized_indices = tuple(layer_indices)
        except TypeError as exc:
            raise TypeError("layer_indices must be a sequence of integers") from exc
        if not normalized_indices:
            raise ValueError("layer_indices must not be empty")
        if any(
            isinstance(layer_index, bool) or not isinstance(layer_index, int)
            for layer_index in normalized_indices
        ):
            raise TypeError("layer_indices must contain only integers")
        if any(left >= right for left, right in pairwise(normalized_indices)):
            raise ValueError("layer_indices must be strictly increasing")

        try:
            normalized_donors = tuple(donor_values)
        except TypeError as exc:
            raise TypeError("donor cache must be a sequence of tensors") from exc
        if len(normalized_donors) != len(normalized_indices):
            raise ValueError("donor cache must contain exactly one tensor per selected layer")

        self._layer_indices = normalized_indices
        self._patches = tuple(
            PrefillBlockOutputPatch(
                normalized_layers,
                layer_index=layer_index,
                positions=positions,
                donor_values=donor,
            )
            for layer_index, donor in zip(
                normalized_indices,
                normalized_donors,
                strict=True,
            )
        )
        self._stack: ExitStack | None = None

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return the decoder-layer indices patched by this context."""

        return self._layer_indices

    @property
    def applications(self) -> tuple[int, ...]:
        """Return the number of prompt-prefill applications for every layer."""

        return tuple(patch.applications for patch in self._patches)

    def __enter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("PrefillBlockOutputWindowPatch is already active")
        stack = ExitStack()
        try:
            for patch in self._patches:
                stack.enter_context(patch)
        except BaseException:
            # Pass the active exception through so a partially installed patch
            # is removed without reporting a misleading "did not run" error.
            stack.__exit__(*sys.exc_info())
            raise
        self._stack = stack
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._stack is None:
            raise RuntimeError("PrefillBlockOutputWindowPatch is not active")
        stack = self._stack
        self._stack = None
        return stack.__exit__(exc_type, exc, traceback)

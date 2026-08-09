"""Generation-specific one-shot decoder-block patching."""

from __future__ import annotations

from typing import Any

import torch

from typo_cot.experiments.layerwise_kl_patching.patching import BlockOutputPatch


class PrefillBlockOutputPatch(BlockOutputPatch):
    """Patch one prompt prefill and ignore subsequent cached decode steps.

    A second forward whose sequence still contains the prompt coordinates means
    generation is not using the expected KV-cache protocol and fails closed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._applications = 0

    @property
    def applications(self) -> int:
        return self._applications

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            hidden = output
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            hidden = output[0]
        else:
            # Delegate the precise unsupported-output diagnostic to the parent.
            return super()._hook(module, inputs, output)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            # Delegate the precise diagnostic to the parent implementation.
            return super()._hook(module, inputs, output)
        # Cached autoregressive decoding executes the block on one new token.
        # Once the prompt prefill has been patched, that one-token forward must
        # remain untreated even when an aligned prompt coordinate happens to be 0.
        if self._applications and hidden.shape[1] == 1:
            return output
        if hidden.shape[1] <= max(self._positions):
            return output
        if self._applications:
            raise RuntimeError("prompt-coordinate patch would run more than once")
        self._applications += 1
        return super()._hook(module, inputs, output)

    def __enter__(self) -> PrefillBlockOutputPatch:
        self._applications = 0
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        suppress = super().__exit__(exc_type, exc, traceback)
        if exc_type is None and self._applications != 1:
            raise RuntimeError("activation patch did not run during prompt prefill")
        return suppress

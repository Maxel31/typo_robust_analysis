"""Exact pre-projection capture and patching for neurons and attention heads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from typo_robust_training.localization.components import ComponentRef


def _modules(value: Sequence[nn.Module]) -> tuple[nn.Module, ...]:
    try:
        modules = tuple(value)
    except TypeError as exc:
        raise TypeError("modules must be a sequence") from exc
    if not modules or any(not isinstance(module, nn.Module) for module in modules):
        raise ValueError("modules must contain torch modules")
    return modules


def _positions(value: Sequence[int]) -> tuple[int, ...]:
    try:
        positions = tuple(value)
    except TypeError as exc:
        raise TypeError("positions must be a sequence") from exc
    if not positions or any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0
        for position in positions
    ):
        raise ValueError("positions must contain non-negative integers")
    if len(set(positions)) != len(positions):
        raise ValueError("positions must not be duplicated")
    return positions


def _input_tensor(inputs: tuple[Any, ...]) -> torch.Tensor:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise TypeError("component module input must start with a tensor")
    hidden = inputs[0]
    if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
        raise ValueError("component module input must have shape [1, sequence, features]")
    if not torch.is_floating_point(hidden):
        raise TypeError("component module input must be floating point")
    return hidden


def capture_module_inputs(
    modules: Sequence[nn.Module],
    *,
    positions: Sequence[int],
    forward: Callable[[], Any],
) -> tuple[torch.Tensor, ...]:
    """Capture full feature rows entering each module in exactly one forward."""

    normalized_modules = _modules(modules)
    normalized_positions = _positions(positions)
    captured: list[torch.Tensor | None] = [None] * len(normalized_modules)
    handles: list[RemovableHandle] = []

    def make_hook(index: int) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            hidden = _input_tensor(inputs)
            if max(normalized_positions) >= int(hidden.shape[1]):
                raise IndexError("component capture position is outside the sequence")
            if captured[index] is not None:
                raise RuntimeError("component capture module ran more than once")
            captured[index] = (
                hidden[0, normalized_positions, :].detach().to(device="cpu", copy=True)
            )

        return hook

    try:
        for index, module in enumerate(normalized_modules):
            handles.append(module.register_forward_pre_hook(make_hook(index)))
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
    missing = [index for index, value in enumerate(captured) if value is None]
    if missing:
        raise RuntimeError(f"forward did not execute component module(s): {missing}")
    return tuple(value for value in captured if value is not None)


class ComponentInputPatch:
    """Patch one post-SwiGLU neuron or one pre-o_proj attention-head slice."""

    def __init__(
        self,
        module: nn.Module,
        *,
        component: ComponentRef,
        positions: Sequence[int],
        donor_values: torch.Tensor,
        attention_head_dim: int,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("component patch module must be a torch module")
        if not isinstance(component, ComponentRef):
            raise TypeError("component must be ComponentRef")
        if (
            isinstance(attention_head_dim, bool)
            or not isinstance(attention_head_dim, int)
            or attention_head_dim <= 0
        ):
            raise ValueError("attention_head_dim must be a positive integer")
        self.module = module
        self.component = component
        self.positions = _positions(positions)
        self.attention_head_dim = attention_head_dim
        if component.kind == "mlp-neuron":
            self.start, self.stop = component.index, component.index + 1
        else:
            self.start = component.index * attention_head_dim
            self.stop = self.start + attention_head_dim
        if not isinstance(donor_values, torch.Tensor) or donor_values.ndim != 2:
            raise ValueError("component donor must have shape [positions, component-width]")
        if tuple(donor_values.shape) != (len(self.positions), self.stop - self.start):
            raise ValueError("component donor shape differs from patch positions and width")
        if not torch.is_floating_point(donor_values):
            raise TypeError("component donor must be floating point")
        self.donor_values = donor_values.detach().clone()
        self._handle: RemovableHandle | None = None
        self._applications = 0

    @property
    def applications(self) -> int:
        return self._applications

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        hidden = _input_tensor(inputs)
        if self._applications and int(hidden.shape[1]) == 1:
            return None
        if int(hidden.shape[1]) <= max(self.positions):
            return None
        if self._applications:
            raise RuntimeError("component prompt patch would run more than once")
        if self.stop > int(hidden.shape[2]):
            raise IndexError("component index is outside the module input feature dimension")
        self._applications += 1
        patched = hidden.clone()
        patched[0, self.positions, self.start : self.stop] = self.donor_values.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return (patched, *inputs[1:])

    def __enter__(self) -> Self:
        if self._handle is not None:
            raise RuntimeError("ComponentInputPatch is already active")
        self._applications = 0
        self._handle = self.module.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc_type is None and self._applications != 1:
            raise RuntimeError("component patch did not run exactly once")
        return False


__all__ = ["ComponentInputPatch", "capture_module_inputs"]

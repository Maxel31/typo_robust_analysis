"""Safe decoder-block activation capture and patching primitives.

The paper intervention copies the residual-stream value *after* one complete
text-decoder block.  These helpers therefore hook only the modules exposed as
``model.get_decoder().layers``; guessing from arbitrary ``named_modules`` can
silently select a vision stack in multimodal models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def _config_layer_counts(model: Any, decoder: Any) -> list[tuple[str, int]]:
    """Return all explicitly declared text-decoder layer counts."""

    candidates = (
        ("decoder.config", getattr(decoder, "config", None)),
        ("model.config", getattr(model, "config", None)),
        (
            "model.config.text_config",
            getattr(getattr(model, "config", None), "text_config", None),
        ),
    )
    counts: list[tuple[str, int]] = []
    for label, config in candidates:
        if config is None or not hasattr(config, "num_hidden_layers"):
            continue
        value = getattr(config, "num_hidden_layers")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label}.num_hidden_layers must be a positive integer, got {value!r}")
        counts.append((label, value))
    return counts


def find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the unambiguous text-decoder block stack.

    Discovery deliberately requires ``get_decoder().layers`` and validates its
    length against ``num_hidden_layers``.  This fail-closed behavior prevents a
    multimodal model's vision blocks from being patched by accident.
    """

    get_decoder = getattr(model, "get_decoder", None)
    if not callable(get_decoder):
        raise ValueError("text decoder discovery requires a callable model.get_decoder()")
    try:
        decoder = get_decoder()
    except Exception as exc:  # noqa: BLE001 - convert discovery failures uniformly
        raise ValueError("model.get_decoder() failed while finding the text decoder") from exc

    raw_layers = getattr(decoder, "layers", None)
    if raw_layers is None:
        raise ValueError("text decoder returned by get_decoder() has no .layers stack")
    try:
        layers = list(raw_layers)
    except TypeError as exc:
        raise ValueError("text decoder .layers must be an iterable of modules") from exc
    if not layers or any(not isinstance(layer, nn.Module) for layer in layers):
        raise ValueError("text decoder .layers must contain one or more torch modules")

    declared_counts = _config_layer_counts(model, decoder)
    if not declared_counts:
        raise ValueError("text decoder config must declare num_hidden_layers")
    distinct_counts = {count for _, count in declared_counts}
    if len(distinct_counts) != 1:
        details = ", ".join(f"{label}={count}" for label, count in declared_counts)
        raise ValueError(f"conflicting num_hidden_layers declarations: {details}")
    (declared_count,) = distinct_counts
    if len(layers) != declared_count:
        raise ValueError(
            "text decoder .layers length does not match num_hidden_layers: "
            f"{len(layers)} != {declared_count}"
        )
    return layers


def _positions_tuple(positions: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(positions)
    except TypeError as exc:
        raise TypeError("positions must be a sequence of token indices") from exc
    if not normalized:
        raise ValueError("positions must not be empty")
    if any(isinstance(position, bool) or not isinstance(position, int) for position in normalized):
        raise TypeError("positions must contain only integer token indices")
    if any(position < 0 for position in normalized):
        raise ValueError("positions must contain only non-negative token indices")
    if len(set(normalized)) != len(normalized):
        raise ValueError("positions must not contain duplicate token indices")
    return normalized


def _hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("decoder block tuple/list output must start with a hidden-state tensor")
        return output[0]
    raise TypeError(
        f"decoder block output must be a Tensor, tuple, or list; got {type(output).__name__}"
    )


def _with_hidden(output: Any, hidden: torch.Tensor) -> Any:
    """Return an output of the same container kind without mutating ``output``."""

    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    # Callers always run _hidden_from_output first; retain a defensive error.
    raise TypeError(f"unsupported decoder block output: {type(output).__name__}")


def _validate_hidden(hidden: torch.Tensor, positions: tuple[int, ...]) -> None:
    if hidden.ndim != 3:
        raise ValueError("decoder block hidden states must have shape [batch, sequence, hidden]")
    if hidden.shape[0] != 1:
        raise ValueError("decoder block capture and patching require batch size 1")
    if max(positions) >= hidden.shape[1]:
        raise IndexError(
            f"token position {max(positions)} is outside sequence length {hidden.shape[1]}"
        )


def _validated_layers(layers: Sequence[nn.Module]) -> tuple[nn.Module, ...]:
    try:
        normalized = tuple(layers)
    except TypeError as exc:
        raise TypeError("layers must be a sequence of torch modules") from exc
    if not normalized or any(not isinstance(layer, nn.Module) for layer in normalized):
        raise ValueError("layers must contain one or more torch modules")
    return normalized


def capture_block_outputs(
    layers: Sequence[nn.Module],
    *,
    positions: Sequence[int],
    forward: Callable[[], Any],
) -> list[torch.Tensor]:
    """Capture selected residual-stream rows from every block in one forward.

    Returned tensors have shape ``[len(positions), hidden_size]`` and are
    detached CPU clones, so they neither retain the model graph nor alias block
    outputs.  Every hook is removed even when the supplied forward raises.
    """

    captured, _output = capture_block_outputs_with_forward(
        layers,
        positions=positions,
        forward=forward,
    )
    return captured


def capture_block_outputs_with_forward(
    layers: Sequence[nn.Module],
    *,
    positions: Sequence[int],
    forward: Callable[[], Any],
) -> tuple[list[torch.Tensor], Any]:
    """Capture residual rows and return the same forward's model output.

    This is the single-pass form of :func:`capture_block_outputs`.  It keeps
    donor capture and downstream logits tied to one identical forward without
    retaining activation graphs in the detached CPU donor tensors.
    """

    decoder_layers = _validated_layers(layers)
    token_positions = _positions_tuple(positions)
    if not callable(forward):
        raise TypeError("forward must be callable")

    captured: list[torch.Tensor | None] = [None] * len(decoder_layers)
    handles: list[RemovableHandle] = []

    def make_hook(layer_index: int) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _hidden_from_output(output)
            _validate_hidden(hidden, token_positions)
            rows = hidden[0, token_positions, :]
            captured[layer_index] = rows.detach().to(device="cpu", copy=True)

        return hook

    try:
        for layer_index, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(make_hook(layer_index)))
        with torch.no_grad():
            output = forward()
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index, values in enumerate(captured) if values is None]
    if missing:
        raise RuntimeError(f"forward did not execute decoder block(s): {missing}")
    return [values for values in captured if values is not None], output


class BlockOutputPatch:
    """Temporarily patch selected positions after one complete decoder block."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        layer_index: int,
        positions: Sequence[int],
        donor_values: torch.Tensor,
    ) -> None:
        self._layers = _validated_layers(layers)
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if layer_index < 0 or layer_index >= len(self._layers):
            raise IndexError(f"layer_index {layer_index} is outside [0, {len(self._layers)})")
        self._layer_index = layer_index
        self._positions = _positions_tuple(positions)
        if not isinstance(donor_values, torch.Tensor):
            raise TypeError("donor_values must be a torch.Tensor")
        if donor_values.ndim != 2:
            raise ValueError("donor_values must have shape [positions, hidden]")
        if donor_values.shape[0] != len(self._positions):
            raise ValueError(
                "donor_values row count must match positions: "
                f"{donor_values.shape[0]} != {len(self._positions)}"
            )
        if donor_values.shape[1] == 0:
            raise ValueError("donor_values hidden dimension must not be empty")
        if not torch.is_floating_point(donor_values):
            raise TypeError("donor_values must have a floating-point dtype")
        self._donor_values = donor_values.detach().clone()
        self._handle: RemovableHandle | None = None

    def _hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden = _hidden_from_output(output)
        _validate_hidden(hidden, self._positions)
        if not torch.is_floating_point(hidden):
            raise TypeError("decoder block hidden states must have a floating-point dtype")
        if hidden.shape[2] != self._donor_values.shape[1]:
            raise ValueError(
                "donor_values hidden dimension does not match decoder output: "
                f"{self._donor_values.shape[1]} != {hidden.shape[2]}"
            )

        donor = self._donor_values.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[0, self._positions, :] = donor
        return _with_hidden(output, patched)

    def __enter__(self) -> BlockOutputPatch:
        if self._handle is not None:
            raise RuntimeError("BlockOutputPatch is already active")
        layer = self._layers[self._layer_index]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

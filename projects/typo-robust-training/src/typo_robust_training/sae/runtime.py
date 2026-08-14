"""Single-GPU activation collection and splice evaluation for SAEs."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from typo_robust_training.sae.config import SaeProtocol
from typo_robust_training.training.pairs import TrainingSource


@dataclass(frozen=True, slots=True)
class ActivationBuffer:
    activations_by_layer: Mapping[int, Any]
    source_start: int
    source_stop: int
    next_source_index: int
    next_source_offset: int
    tokens: int
    buffer_index: int


def _buffer_seed(*, config_sha256: str, buffer_index: int) -> int:
    digest = hashlib.sha256(
        f"sae-activation-buffer-order/v1\0{config_sha256}\0{buffer_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class HuggingFaceSaeRuntime:
    """Frozen LM runtime; only separately created SAE modules may receive gradients."""

    def __init__(self, *, protocol: SaeProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, SaeProtocol):
            raise TypeError("SAE runtime protocol must be SaeProtocol")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("SAE runtime requires exactly one requested CUDA GPU")
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        self.protocol = protocol
        self.gpu_id = gpu_id
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self.model = self.wrapper.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = "right"
        self.layers = tuple(find_decoder_layers(self.model))
        if len(self.layers) != protocol.decoder_layers:
            raise ValueError("SAE runtime decoder-layer count differs")
        text_config = getattr(self.model.config, "text_config", self.model.config)
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size != protocol.d_model:
            raise ValueError("SAE runtime hidden size differs from config")
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != protocol.model_revision:
            raise ValueError("SAE runtime model revision differs from config")
        self.device = next(self.model.parameters()).device

    def _tokenize(self, text: str) -> tuple[Any, Any]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.protocol.max_sequence_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("SAE tokenizer output must be an object")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if (
            input_ids is None
            or attention_mask is None
            or input_ids.ndim != 2
            or input_ids.shape != attention_mask.shape
            or int(input_ids.shape[0]) != 1
            or not bool(attention_mask.all())
        ):
            raise ValueError("SAE tokenizer must return one unpadded sequence")
        return input_ids.to(self.device), attention_mask.to(self.device)

    def _capture(self, text: str, *, layer_indices: Sequence[int]) -> Mapping[int, Any]:
        torch = self._torch
        selected = tuple(layer_indices)
        if not selected or tuple(sorted(set(selected))) != selected:
            raise ValueError("SAE capture layers must be sorted and unique")
        captured: dict[int, Any] = {}
        handles: list[Any] = []

        def make_hook(layer_index: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor) or value.ndim != 3:
                    raise TypeError("SAE block output must be a rank-three tensor")
                if layer_index in captured:
                    raise RuntimeError("SAE decoder layer ran more than once")
                captured[layer_index] = value.detach()

            return hook

        try:
            for layer_index in selected:
                handles.append(
                    self.layers[layer_index].register_forward_hook(make_hook(layer_index))
                )
            input_ids, attention_mask = self._tokenize(text)
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(selected):
            raise RuntimeError("SAE model forward skipped a requested decoder layer")
        return {
            layer: captured[layer][0].to(device="cpu", dtype=torch.bfloat16).contiguous()
            for layer in selected
        }

    def iter_activation_buffers(
        self,
        sources: Sequence[TrainingSource],
        *,
        layer_indices: Sequence[int],
        target_tokens: int,
        start_source_index: int = 0,
        start_source_offset: int = 0,
        start_buffer_index: int = 0,
    ) -> Iterator[ActivationBuffer]:
        torch = self._torch
        rows = tuple(sources)
        if (
            not rows
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens <= 0
            or not 0 <= start_source_index < len(rows)
            or isinstance(start_source_offset, bool)
            or not isinstance(start_source_offset, int)
            or start_source_offset < 0
        ):
            raise ValueError("SAE activation stream dimensions are invalid")
        selected = tuple(layer_indices)
        buffer_target = min(self.protocol.shuffle_buffer_activations, target_tokens)
        pieces: dict[int, list[Any]] = {layer: [] for layer in selected}
        buffered = 0
        emitted = 0
        source_start = start_source_index
        buffer_index = start_buffer_index
        for source_index in range(start_source_index, len(rows)):
            captured = self._capture(rows[source_index].clean_text, layer_indices=selected)
            length = int(next(iter(captured.values())).shape[0])
            if any(int(value.shape[0]) != length for value in captured.values()):
                raise RuntimeError("SAE captured layers have different sequence lengths")
            offset = start_source_offset if source_index == start_source_index else 0
            if offset >= length:
                raise ValueError("SAE resume source offset is outside the tokenized document")
            while offset < length:
                remaining_total = target_tokens - emitted - buffered
                if remaining_total <= 0:
                    break
                take = min(length - offset, buffer_target - buffered, remaining_total)
                for layer in selected:
                    pieces[layer].append(captured[layer][offset : offset + take])
                offset += take
                buffered += take
                if buffered == buffer_target or emitted + buffered == target_tokens:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(
                        _buffer_seed(
                            config_sha256=self.protocol.config_sha256,
                            buffer_index=buffer_index,
                        )
                    )
                    permutation = torch.randperm(buffered, generator=generator)
                    joined = {
                        layer: torch.cat(pieces[layer], dim=0)[permutation].contiguous()
                        for layer in selected
                    }
                    yield ActivationBuffer(
                        activations_by_layer=joined,
                        source_start=source_start,
                        source_stop=source_index + 1,
                        next_source_index=(source_index if offset < length else source_index + 1),
                        next_source_offset=(offset if offset < length else 0),
                        tokens=buffered,
                        buffer_index=buffer_index,
                    )
                    emitted += buffered
                    buffer_index += 1
                    source_start = source_index if offset < length else source_index + 1
                    pieces = {layer: [] for layer in selected}
                    buffered = 0
                    buffer_target = min(
                        self.protocol.shuffle_buffer_activations,
                        target_tokens - emitted,
                    )
                    if emitted == target_tokens:
                        return
            if emitted + buffered == target_tokens:
                break
        if emitted != target_tokens:
            raise ValueError(
                f"SAE source stream produced {emitted} activation tokens, expected {target_tokens}"
            )

    @contextmanager
    def _splice_hook(self, *, layer: int, sae: Any) -> Iterator[None]:
        torch = self._torch

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise TypeError("SAE splice target must be a rank-three tensor")
            with torch.inference_mode():
                reconstructed, _features = sae(value[0].float())
            replacement = reconstructed.to(dtype=value.dtype).unsqueeze(0)
            if isinstance(output, tuple):
                return (replacement, *output[1:])
            return replacement

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def splice_kl(self, text: str, *, layer: int, sae: Any) -> tuple[float, ...]:
        torch = self._torch
        input_ids, attention_mask = self._tokenize(text)
        with torch.inference_mode():
            reference = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits
            with self._splice_hook(layer=layer, sae=sae):
                spliced = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
        reference_log = torch.log_softmax(reference[0, :-1].float(), dim=-1)
        spliced_log = torch.log_softmax(spliced[0, :-1].float(), dim=-1)
        kl = (reference_log.exp() * (reference_log - spliced_log)).sum(dim=-1).clamp_min(0.0)
        return tuple(float(value) for value in kl.detach().cpu().tolist())

    def provenance(self) -> Mapping[str, object]:
        return {
            "runtime": "huggingface-sae-single-gpu/v1",
            "model": self.protocol.model,
            "model_revision": self.protocol.model_revision,
            "gpu_id": self.gpu_id,
            "decoder_layers": len(self.layers),
            "d_model": self.protocol.d_model,
        }


__all__ = ["ActivationBuffer", "HuggingFaceSaeRuntime"]

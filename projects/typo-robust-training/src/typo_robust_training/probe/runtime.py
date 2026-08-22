"""Read-only Hugging Face activation runtime for the probe producer."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from typo_robust_training.localization.prompting import word_final_token_positions
from typo_robust_training.probe.config import ProbeProducerProtocol
from typo_robust_training.probe.producer import ProbeCohortRecord


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _block_tensor(output: object) -> Any:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not hasattr(value, "ndim") or value.ndim != 3 or int(value.shape[0]) != 1:
        raise ValueError("decoder block output is not one [1, sequence, hidden] tensor")
    return value


class HuggingFaceProbeActivationProvider:
    """Extract complete-block residuals without enabling any model gradient."""

    def __init__(self, *, protocol: ProbeProducerProtocol, gpu_id: str) -> None:
        if not isinstance(protocol, ProbeProducerProtocol):
            raise TypeError("protocol must be ProbeProducerProtocol")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("select-probe-transition requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "select-probe-transition requires exactly one visible GPU; "
                "set CUDA_VISIBLE_DEVICES to one physical device"
            )
        from typo_cot.experiments.layerwise_kl_patching.patching import find_decoder_layers
        from typo_cot.models.wrapper import create_model_wrapper

        wrapper = create_model_wrapper(
            model_name=protocol.model,
            gpu_id=gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=protocol.model_revision,
        )
        self._model = wrapper.model
        self._model.eval()
        self._model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise RuntimeError("probe activation runtime failed to freeze the base model")
        self.tokenizer = wrapper.tokenizer
        if getattr(self.tokenizer, "is_fast", False) is not True:
            raise ValueError("probe activation runtime requires a fast tokenizer")
        self.layers = tuple(find_decoder_layers(self._model))
        self.device = next(self._model.parameters()).device
        self._torch = torch
        self.gpu_id = gpu_id
        self.model = protocol.model
        self.model_revision = protocol.model_revision
        self.decoder_layers = len(self.layers)
        raw_hidden = getattr(self._model.config, "hidden_size", None)
        if raw_hidden is None and hasattr(self._model.config, "text_config"):
            raw_hidden = getattr(self._model.config.text_config, "hidden_size", None)
        if isinstance(raw_hidden, bool) or not isinstance(raw_hidden, int):
            raise ValueError("loaded model does not expose an integer hidden size")
        self.hidden_size = raw_hidden
        self.base_model_frozen = True
        actual_revision = getattr(self._model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != protocol.model_revision:
            raise ValueError("loaded model revision differs from the preregistration")
        if (
            self.decoder_layers != protocol.decoder_layers
            or self.hidden_size != protocol.hidden_size
        ):
            raise ValueError("loaded model architecture differs from the preregistration")

    def _one_activation(self, record: ProbeCohortRecord, *, side: str) -> np.ndarray:
        if side == "clean":
            text = record.clean_text
            span = record.clean_word_char_span
        elif side == "typo":
            if record.typo_text is None or record.typo_word_char_span is None:
                raise ValueError("typo activation requested for an unpaired fit record")
            text = record.typo_text
            span = record.typo_word_char_span
        else:
            raise ValueError("probe activation side must be clean or typo")
        position = word_final_token_positions(
            self.tokenizer,
            text=text,
            spans=(span,),
        )[0]
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("probe tokenizer must return mapping-like tensors")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("probe tokenizer must return one unpadded input sequence")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise ValueError("probe texts must be unpadded and fully attended")
        if position >= int(input_ids.shape[1]):
            raise ValueError("edited-word final token lies outside the probe text")

        captured: list[Any | None] = [None] * self.decoder_layers
        handles: list[Any] = []
        for layer_index, layer in enumerate(self.layers):
            def capture(
                _module: object,
                _inputs: object,
                output: object,
                *,
                index: int = layer_index,
            ) -> None:
                captured[index] = _block_tensor(output)[0, position].detach().float().cpu()

            handles.append(layer.register_forward_hook(capture))
        try:
            with self._torch.inference_mode():
                self._model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("probe runtime did not capture every decoder layer")
        stacked = self._torch.stack(captured)  # type: ignore[arg-type]
        if stacked.shape != (self.decoder_layers, self.hidden_size):
            raise ValueError("captured probe activation shape differs")
        return np.ascontiguousarray(stacked.numpy(), dtype=np.float32)

    def activations(
        self,
        records: Sequence[ProbeCohortRecord],
        *,
        side: str,
    ) -> np.ndarray:
        if not records:
            raise ValueError("probe activation inventory must not be empty")
        return np.stack(
            [self._one_activation(record, side=side) for record in records],
            axis=0,
        )

    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "hugging-face-complete-block-residual/v1",
            "model": self.model,
            "model_revision": self.model_revision,
            "decoder_layers": self.decoder_layers,
            "hidden_size": self.hidden_size,
            "base_model_frozen": self.base_model_frozen,
            "gpu_id": self.gpu_id,
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
        }


__all__ = ["HuggingFaceProbeActivationProvider"]

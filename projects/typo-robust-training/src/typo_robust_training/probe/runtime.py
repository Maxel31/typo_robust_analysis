"""Read-only Hugging Face activation runtime for the probe producer."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from typo_robust_training.localization.prompting import word_final_token_positions
from typo_robust_training.probe.config import ProbeProducerProtocol
from typo_robust_training.probe.producer import ProbeCohortRecord


_REVISION = re.compile(r"[0-9a-f]{40}")


def _checkout_code_revision() -> str:
    module_path = Path(__file__).resolve()
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=module_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("probe runtime cannot locate its git checkout")
    checkout_root = Path(root_result.stdout.strip()).resolve()
    try:
        module_relative = module_path.relative_to(checkout_root)
        package_relative = module_path.parents[1].relative_to(checkout_root)
    except ValueError as exc:
        raise RuntimeError("probe runtime module is outside the attested checkout") from exc

    tracked_result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", module_relative.as_posix()],
        cwd=checkout_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked_result.returncode != 0:
        raise RuntimeError("probe runtime module is not tracked by the attested checkout")
    dirty_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            package_relative.as_posix(),
        ],
        cwd=checkout_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty_result.returncode != 0 or dirty_result.stdout.strip():
        raise RuntimeError("probe runtime source tree is not clean")

    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = head_result.stdout.strip()
    if head_result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        raise RuntimeError("probe runtime cannot attest the executing code revision")
    return revision


def _require_exact_model_revision(
    *,
    model_config: object,
    tokenizer: object,
    expected: str,
) -> str:
    model_candidates: list[str] = []
    for config in (
        model_config,
        getattr(model_config, "text_config", None),
    ):
        revision = getattr(config, "_commit_hash", None)
        if isinstance(revision, str) and revision:
            model_candidates.append(revision)
    if not model_candidates:
        raise ValueError("loaded model revision is not observable")
    if any(revision != expected for revision in model_candidates):
        raise ValueError("loaded model revision differs from the preregistration")

    tokenizer_candidates: list[str] = []
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        revision = init_kwargs.get("_commit_hash")
        if isinstance(revision, str) and revision:
            tokenizer_candidates.append(revision)
    if any(revision != expected for revision in tokenizer_candidates):
        raise ValueError("loaded tokenizer revision differs from the preregistration")
    return expected


def _inflation_bucket(delta: int) -> str:
    if delta <= -2:
        return "minus-two-or-more"
    if delta == -1:
        return "minus-one"
    if delta == 0:
        return "same"
    if delta == 1:
        return "plus-one"
    return "plus-two-or-more"


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
        self.code_revision = _checkout_code_revision()
        if self.code_revision != protocol.code_revision:
            raise ValueError("executing code revision differs from the preregistration")
        self.decoder_layers = len(self.layers)
        raw_hidden = getattr(self._model.config, "hidden_size", None)
        if raw_hidden is None and hasattr(self._model.config, "text_config"):
            raw_hidden = getattr(self._model.config.text_config, "hidden_size", None)
        if isinstance(raw_hidden, bool) or not isinstance(raw_hidden, int):
            raise ValueError("loaded model does not expose an integer hidden size")
        self.hidden_size = raw_hidden
        self.base_model_frozen = True
        _require_exact_model_revision(
            model_config=self._model.config,
            tokenizer=self.tokenizer,
            expected=protocol.model_revision,
        )
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

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("probe tokenizer must return a token mapping")
        input_ids = encoded.get("input_ids")
        if hasattr(input_ids, "ndim"):
            if input_ids.ndim == 1:
                return int(input_ids.shape[0])
            if input_ids.ndim == 2 and int(input_ids.shape[0]) == 1:
                return int(input_ids.shape[1])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                if len(input_ids) != 1:
                    raise ValueError("probe tokenizer returned multiple sequences")
                input_ids = input_ids[0]
            if all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids):
                return len(input_ids)
        raise ValueError("probe tokenizer returned an invalid token inventory")

    def token_inflation_bucket(self, record: ProbeCohortRecord) -> str:
        if record.typo_text is None:
            raise ValueError("token inflation requires a paired probe record")
        return _inflation_bucket(
            self._token_count(record.typo_text) - self._token_count(record.clean_text)
        )

    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "hugging-face-complete-block-residual/v1",
            "model": self.model,
            "model_revision": self.model_revision,
            "code_revision": self.code_revision,
            "decoder_layers": self.decoder_layers,
            "hidden_size": self.hidden_size,
            "base_model_frozen": self.base_model_frozen,
            "gpu_id": self.gpu_id,
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
        }


__all__ = [
    "HuggingFaceProbeActivationProvider",
    "_inflation_bucket",
    "_require_exact_model_revision",
]

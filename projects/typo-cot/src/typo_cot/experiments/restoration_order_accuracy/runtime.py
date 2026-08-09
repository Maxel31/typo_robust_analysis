"""Pinned Hugging Face generation adapter for the Table 13 producer."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import random
import re
from collections.abc import Callable, Mapping, Sequence

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.restoration_order_accuracy.integrity import (
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    GENERATION,
    PROTOCOL_SHA256,
    canonical_sha256,
)

_SINGLE_GPU = re.compile(r"0|[1-9][0-9]*")
_BENCHMARK_ALIASES = {"gsm8k": "gsm8k", "mmlu": "mmlu"}
_ANSWER_EXTRACTION = "task-primary-then-empty-only-fallback-symmetric-cap-aware/v1"
_GENERATION_ARGUMENTS = {
    "max_new_tokens": int(GENERATION["max_new_tokens"]),
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "use_cache": True,
    "return_dict_in_generate": False,
    "output_scores": False,
}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _configure_visible_gpu(gpu_id: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != gpu_id:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
            f"environment={visible!r}, argument={gpu_id!r}"
        )
    if _SINGLE_GPU.fullmatch(gpu_id) is None:
        raise ValueError("restoration-order runtime requires exactly one physical GPU ID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def _require_single_cuda(torch: object) -> None:
    cuda = torch.cuda  # type: ignore[attr-defined]
    if not cuda.is_available():
        raise RuntimeError("restoration-order runtime requires the requested CUDA GPU")
    if cuda.device_count() != 1:
        raise RuntimeError("restoration-order runtime requires exactly one visible GPU")


def _seed_runtime(torch: object) -> None:
    random.seed(42)
    try:
        import numpy as np

        np.random.seed(42)
    except ImportError:  # pragma: no cover - NumPy is a project dependency
        pass
    torch.manual_seed(42)  # type: ignore[attr-defined]
    torch.cuda.manual_seed_all(42)  # type: ignore[attr-defined]


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    result = tuple(value)
    if any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in result
    ):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return result


class HuggingFaceRestorationRuntime:
    """Generate one condition-major batch with a pinned evaluator revision."""

    def __init__(
        self,
        config: object,
        *,
        revision: str,
        wrapper_factory: Callable[..., object] | None = None,
        torch_module: object | None = None,
    ) -> None:
        if not revision:
            raise ValueError("restoration-order generation requires a pinned source revision")
        self.config = config
        self.revision = revision
        self.gpu_id = str(getattr(config, "gpu_id"))
        self.batch_size = int(getattr(config, "batch_size"))
        if self.batch_size != int(GENERATION["batch_size"]):
            raise ValueError(
                f"restoration-order batch_size must be {GENERATION['batch_size']}"
            )
        _configure_visible_gpu(self.gpu_id)
        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module
        _require_single_cuda(self._torch)
        _seed_runtime(self._torch)
        if wrapper_factory is None:
            from typo_cot.models.wrapper import create_model_wrapper

            wrapper_factory = create_model_wrapper
        self.wrapper = wrapper_factory(
            model_name=str(getattr(config, "model")),
            gpu_id=self.gpu_id,
            dtype=self._torch.bfloat16,  # type: ignore[attr-defined]
            wrap_for_lxt=False,
            revision=revision,
        )
        self.model = self.wrapper.model  # type: ignore[attr-defined]
        self.model.eval()
        self.tokenizer = self.wrapper.tokenizer  # type: ignore[attr-defined]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError("evaluator tokenizer exposes neither pad nor EOS token ID")
        self.device = next(self.model.parameters()).device
        self.effective_eos_token_ids, self.eos_source = self._resolve_eos()
        benchmark = str(getattr(config, "benchmark"))
        if benchmark not in _BENCHMARK_ALIASES:
            raise ValueError(f"restoration-order generation is unsupported for {benchmark!r}")
        self.internal_benchmark = _BENCHMARK_ALIASES[benchmark]
        self._closed = False

    @staticmethod
    def _normalized_eos(value: object) -> tuple[int, ...]:
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        return tuple(
            sorted(
                {
                    token
                    for token in values
                    if isinstance(token, int) and not isinstance(token, bool) and token >= 0
                }
            )
        )

    def _resolve_eos(self) -> tuple[tuple[int, ...], str]:
        generation_config = self.model.generation_config
        if getattr(generation_config, "stop_strings", None):
            raise ValueError("restoration-order generation does not support stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("restoration-order generation does not support forced_eos_token_id")
        values = self._normalized_eos(getattr(generation_config, "eos_token_id", None))
        if values:
            return values, "model-generation-config"
        values = self._normalized_eos(getattr(self.tokenizer, "eos_token_id", None))
        if values:
            return values, "tokenizer-fallback"
        raise ValueError("evaluator exposes no EOS token ID")

    def _encode_unpadded(self, prompt: str, *, field: str) -> tuple[int, ...]:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError(f"tokenizer returned no input_ids for {field}")
        return _token_ids(encoded["input_ids"], field=field)

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> Sequence[object]:
        """Generate and score one non-empty, aligned condition-major batch."""
        if self._closed:
            raise RuntimeError("restoration-order runtime is closed")
        if not prompts or not (
            len(prompts) == len(sample_ids) == len(gold_answers) <= self.batch_size
        ):
            raise ValueError(
                "restoration-order batch must be non-empty, aligned, and within batch_size"
            )
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("restoration-order prompts must be strings")
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
            raise ValueError("restoration-order sample IDs must be non-empty strings")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("restoration-order sample IDs must be unique within a batch")
        if any(not isinstance(answer, str) for answer in gold_answers):
            raise TypeError("restoration-order gold answers must be strings")

        prompt_ids = [
            self._encode_unpadded(prompt, field=f"{sample_id} prompt")
            for prompt, sample_id in zip(prompts, sample_ids, strict=True)
        ]
        encoded = self.tokenizer(
            list(prompts),
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer returned no batched input_ids")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if (
            not hasattr(input_ids, "ndim")
            or input_ids.ndim != 2
            or int(input_ids.shape[0]) != len(prompts)
            or attention_mask is None
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("tokenizer returned an invalid left-padded batch")
        for index, expected in enumerate(prompt_ids):
            actual = tuple(
                int(token)
                for token, attended in zip(
                    input_ids[index].detach().cpu().tolist(),
                    attention_mask[index].detach().cpu().tolist(),
                    strict=True,
                )
                if int(attended) == 1
            )
            if actual != expected:
                raise ValueError(f"batched tokenization differs for row {index}")

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with self._torch.inference_mode():  # type: ignore[attr-defined]
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **_GENERATION_ARGUMENTS,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        prompt_width = int(input_ids.shape[1])
        max_new_tokens = int(_GENERATION_ARGUMENTS["max_new_tokens"])
        if (
            not hasattr(output_ids, "ndim")
            or output_ids.ndim != 2
            or int(output_ids.shape[0]) != len(prompts)
            or int(output_ids.shape[1]) <= prompt_width
            or int(output_ids.shape[1]) > prompt_width + max_new_tokens
        ):
            raise ValueError("generation returned an invalid capped batch")
        if not bool(output_ids[:, :prompt_width].equal(input_ids)):
            raise ValueError("generation output does not preserve the padded prompt prefix")

        from typo_cot.experiments.restoration_order_accuracy.runner import (
            RestorationGeneration,
        )

        generations: list[RestorationGeneration] = []
        for index, (sample_id, gold_answer) in enumerate(
            zip(sample_ids, gold_answers, strict=True)
        ):
            raw = tuple(
                int(token) for token in output_ids[index, prompt_width:].detach().cpu().tolist()
            )
            eos_index = next(
                (
                    position
                    for position, token in enumerate(raw)
                    if token in self.effective_eos_token_ids
                ),
                None,
            )
            if eos_index is None:
                if len(raw) != max_new_tokens:
                    raise ValueError(f"{sample_id} stopped without EOS before the token cap")
                continuation = raw
                termination = "length-cap"
                allow_positional = False
            else:
                continuation = raw[: eos_index + 1]
                termination = "eos"
                allow_positional = True
            if not continuation:
                raise ValueError(f"{sample_id} generated no continuation token")
            text = self.tokenizer.decode(
                list(continuation),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            extraction = extract_with_fallback(
                text,
                benchmark=self.internal_benchmark,
                correct_answer=gold_answer,
                allow_positional=allow_positional,
            )
            generations.append(
                RestorationGeneration(
                    sample_id=sample_id,
                    token_ids=continuation,
                    text=text,
                    termination=termination,
                    extracted_answer=extraction.value,
                    is_extracted=extraction.is_extracted,
                    is_correct=extraction.is_correct,
                    method=extraction.method,
                    primary_method=extraction.primary_method,
                )
            )
        return tuple(generations)

    def provenance(self) -> dict[str, object]:
        """Describe the resolved evaluator, environment, and executable code."""
        if self._closed or self.model is None or self.tokenizer is None:
            raise RuntimeError("restoration-order runtime is closed")
        model_metadata = getattr(self.model.config, "_commit_hash", None)
        tokenizer_metadata = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        if model_metadata is not None and model_metadata != self.revision:
            raise ValueError("resolved evaluator model revision differs from the requested pin")
        if tokenizer_metadata is not None and tokenizer_metadata != self.revision:
            raise ValueError("resolved evaluator tokenizer revision differs from the requested pin")
        model_revision = model_metadata or self.revision
        tokenizer_revision = tokenizer_metadata or self.revision
        torch = self._torch
        generation_config = self.model.generation_config.to_dict()
        if not isinstance(generation_config, Mapping):
            raise ValueError("evaluator generation config must serialize to an object")
        base_generation_config_sha256 = canonical_sha256(dict(generation_config))
        effective_generation_config_sha256 = canonical_sha256(
            {
                "base_generation_config_sha256": base_generation_config_sha256,
                "overrides": _GENERATION_ARGUMENTS,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": list(self.effective_eos_token_ids),
            }
        )
        provenance = {
            "operation": "restoration-order-accuracy",
            "runtime": "HuggingFaceRestorationRuntime",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": str(getattr(self.config, "model")),
            "requested_revision": self.revision,
            "model_revision": model_revision,
            "model_revision_source": (
                "model-config-metadata" if model_metadata else "explicit-load-revision"
            ),
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_revision_source": (
                "tokenizer-init-metadata" if tokenizer_metadata else "explicit-load-revision"
            ),
            "dtype": "bfloat16",
            "device": "cuda:0",
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "protocol_sha256": PROTOCOL_SHA256,
            "generation": dict(GENERATION),
            "base_generation_config_sha256": base_generation_config_sha256,
            "effective_generation_config_sha256": (
                effective_generation_config_sha256
            ),
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.eos_source,
            "answer_extraction": _ANSWER_EXTRACTION,
            "benchmark_extractor": self.internal_benchmark,
            "historical_extraction_difference": "submitted-table13-primary-only",
            "implementation_code": implementation_code_identity(),
        }
        validate_paper_runtime_environment(provenance)
        return provenance

    def recover_after_batch_failure(self) -> None:
        """Release exception-held Python garbage and cached CUDA allocations."""
        gc.collect()
        if self._torch.cuda.is_available():  # type: ignore[attr-defined]
            self._torch.cuda.empty_cache()  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return
        self.model = None
        self.tokenizer = None
        if self.wrapper is not None:
            if hasattr(self.wrapper, "_model"):
                self.wrapper._model = None
            if hasattr(self.wrapper, "_tokenizer"):
                self.wrapper._tokenizer = None
        self.wrapper = None
        self._closed = True
        gc.collect()
        if self._torch.cuda.is_available():  # type: ignore[attr-defined]
            self._torch.cuda.empty_cache()  # type: ignore[attr-defined]


__all__ = ["HuggingFaceRestorationRuntime"]

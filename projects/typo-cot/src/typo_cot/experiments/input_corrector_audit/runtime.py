"""Production correction and duplicate-prompt generation adapters."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.input_corrector_audit.correctors import create_corrector
from typo_cot.experiments.input_corrector_audit.integrity import (
    implementation_code_identity,
    validate_paper_runtime_environment,
)
from typo_cot.experiments.input_corrector_audit.protocol import (
    CORRECTOR_MODELS,
    GENERATION,
    PROTOCOL_SHA256,
)

_SINGLE_GPU = re.compile(r"0|[1-9][0-9]*")
_BENCHMARK_ALIASES = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "mmlu-pro": "mmlu_pro",
    "arc": "arc",
    "csqa": "commonsense_qa",
}
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
        raise ValueError("input-corrector runtime requires exactly one physical GPU ID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def _require_single_cuda(torch: object) -> None:
    cuda = torch.cuda  # type: ignore[attr-defined]
    if not cuda.is_available():
        raise RuntimeError("input-corrector neural runtime requires the requested CUDA GPU")
    if cuda.device_count() != 1:
        raise RuntimeError("input-corrector neural runtime requires exactly one visible GPU")


def _seed_runtime(torch: object) -> None:
    random.seed(42)
    try:
        import numpy as np

        np.random.seed(42)
    except ImportError:  # pragma: no cover - NumPy is a project dependency
        pass
    torch.manual_seed(42)  # type: ignore[attr-defined]
    torch.cuda.manual_seed_all(42)  # type: ignore[attr-defined]


def _dictionary_sha256(corrector: object) -> str:
    spellchecker = getattr(corrector, "_spellchecker", None)
    word_frequency = getattr(spellchecker, "word_frequency", None)
    dictionary = getattr(word_frequency, "dictionary", None)
    if isinstance(dictionary, Mapping):
        payload: object = sorted(
            (str(word), int(frequency)) for word, frequency in dictionary.items()
        )
    else:
        payload = {
            "unavailable_in_injected_runtime": (
                f"{type(corrector).__module__}.{type(corrector).__qualname__}"
            )
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProductionCorrectionRuntime:
    """Adapt one frozen corrector to the runner's auditable outcome API."""

    def __init__(
        self,
        config: object,
        *,
        corrector_factory: Callable[..., object] = create_corrector,
        torch_module: object | None = None,
    ) -> None:
        self.config = config
        self.corrector_id = str(getattr(config, "corrector"))
        self.gpu_id = str(getattr(config, "gpu_id"))
        if self.corrector_id not in CORRECTOR_MODELS:
            raise ValueError(f"unsupported input corrector: {self.corrector_id!r}")
        _configure_visible_gpu(self.gpu_id)
        specification = CORRECTOR_MODELS[self.corrector_id]
        self.requested_revision = specification["revision"]
        self._torch = torch_module
        kwargs: dict[str, object] = {}
        if self.requested_revision is not None:
            if self._torch is None:
                import torch

                self._torch = torch
            _require_single_cuda(self._torch)
            _seed_runtime(self._torch)
            kwargs = {
                "revision": self.requested_revision,
                "device": "cuda",
            }
        self.corrector = corrector_factory(self.corrector_id, **kwargs)
        self._closed = False

    def correct(self, text: str) -> Any:
        """Correct one editable string while preserving Qwen parser metadata."""
        from typo_cot.experiments.input_corrector_audit.runner import CorrectionOutcome

        if self._closed or self.corrector is None:
            raise RuntimeError("correction runtime is closed")
        if self.corrector_id == "qwen2.5-7b-instruct":
            corrected, metadata = self.corrector.correct_with_meta(text)  # type: ignore[attr-defined]
            if not isinstance(metadata, Mapping):
                raise TypeError("Qwen corrector metadata must be a mapping")
            return CorrectionOutcome(
                corrected_text=corrected,
                parse_failed=metadata.get("parse_failed"),
                n_calls=metadata.get("n_calls"),
                raw_response=metadata.get("raw_response"),
            )
        corrected = self.corrector.correct(text)  # type: ignore[attr-defined]
        return CorrectionOutcome(
            corrected_text=corrected,
            parse_failed=False,
            n_calls=1,
            raw_response=None,
        )

    def provenance(self) -> dict[str, object]:
        """Describe exact model/dictionary, package, and executable identities."""
        if self._closed or self.corrector is None:
            raise RuntimeError("correction runtime is closed")
        common: dict[str, object] = {
            "operation": "input-corrector-audit",
            "runtime": "ProductionCorrectionRuntime",
            "corrector": self.corrector_id,
            "python": platform.python_version(),
            "requested_revision": self.requested_revision,
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_code": implementation_code_identity(),
        }
        if self.corrector_id == "pyspellchecker":
            common.update(
                {
                    "profile": "pyspellchecker",
                    "pyspellchecker": _package_version("pyspellchecker"),
                    "device": "cpu",
                    "dictionary_language": "en",
                    "dictionary_sha256": _dictionary_sha256(self.corrector),
                    "model_revision": None,
                    "tokenizer_revision": None,
                }
            )
            validate_paper_runtime_environment(
                common,
                profile="pyspellchecker",
                field_prefix="input-corrector correction environment",
            )
            return common

        model_metadata = None
        resolved = getattr(self.corrector, "resolved_revision", None)
        if callable(resolved):
            model_metadata = resolved()
        model_revision = model_metadata or self.requested_revision
        tokenizer = getattr(self.corrector, "_tokenizer", None)
        tokenizer_metadata = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if model_metadata is not None and model_metadata != self.requested_revision:
            raise ValueError(
                "resolved correction model revision differs from the requested public pin"
            )
        if tokenizer_metadata is not None and tokenizer_metadata != self.requested_revision:
            raise ValueError(
                "resolved correction tokenizer revision differs from the requested public pin"
            )
        tokenizer_revision = tokenizer_metadata or self.requested_revision
        torch = self._torch
        assert torch is not None
        common.update(
            {
                "profile": "neural",
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "accelerate": _package_version("accelerate"),
                "model_revision": model_revision,
                "model_revision_source": (
                    "model-config-metadata" if model_metadata else "explicit-load-revision"
                ),
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_revision_source": (
                    "tokenizer-init-metadata" if tokenizer_metadata else "explicit-load-revision"
                ),
                "device": "cuda:0",
                "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
        validate_paper_runtime_environment(
            common,
            profile="neural",
            field_prefix="input-corrector correction environment",
        )
        return common

    def close(self) -> None:
        if self._closed:
            return
        corrector = self.corrector
        self.corrector = None
        self._closed = True
        if corrector is not None:
            corrector.close()  # type: ignore[attr-defined]
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():  # type: ignore[attr-defined]
            self._torch.cuda.empty_cache()  # type: ignore[attr-defined]


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    result = tuple(value)
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return result


class HuggingFaceSamePromptRuntime:
    """Generate one adjacent duplicate-prompt batch with a pinned evaluator."""

    def __init__(
        self,
        config: object,
        *,
        revision: str,
        wrapper_factory: Callable[..., object] | None = None,
        torch_module: object | None = None,
    ) -> None:
        if not revision:
            raise ValueError("Same-prompt generation requires a pinned source revision")
        self.config = config
        self.revision = revision
        self.gpu_id = str(getattr(config, "gpu_id"))
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
            raise ValueError(f"Same-prompt generation is unsupported for {benchmark!r}")
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
            raise ValueError("Same-prompt generation does not support stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("Same-prompt generation does not support forced_eos_token_id")
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

    @staticmethod
    def _validate_duplicate_inputs(
        prompts: Sequence[str],
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> None:
        if len(prompts) not in {2, 4} or not (len(prompts) == len(sample_ids) == len(gold_answers)):
            raise ValueError("Same-prompt runtime requires exactly 2 or 4 aligned rows")
        for index in range(0, len(prompts), 2):
            if not all(isinstance(prompt, str) for prompt in prompts[index : index + 2]):
                raise TypeError("Same-prompt prompts must be strings")
            if prompts[index].encode("utf-8") != prompts[index + 1].encode("utf-8"):
                raise ValueError("adjacent duplicate prompt bytes differ")
            if sample_ids[index] != sample_ids[index + 1]:
                raise ValueError("sample IDs in an adjacent duplicate pair differ")
            if gold_answers[index] != gold_answers[index + 1]:
                raise ValueError("gold answers in an adjacent duplicate pair differ")

    def generate_duplicate_batch(
        self,
        prompts: Sequence[str],
        *,
        sample_ids: Sequence[str],
        gold_answers: Sequence[str],
    ) -> Sequence[Any]:
        """Generate `[p,p]` or `[p,p,q,q]` in one model call."""
        self._validate_duplicate_inputs(prompts, sample_ids, gold_answers)
        if self._closed:
            raise RuntimeError("Same-prompt runtime is closed")
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

        from typo_cot.experiments.input_corrector_audit.runner import SamePromptGeneration

        generations: list[SamePromptGeneration] = []
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
                allow_positional = False
            else:
                continuation = raw[: eos_index + 1]
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
                SamePromptGeneration(
                    sample_id=sample_id,
                    token_ids=continuation,
                    text=text,
                    extracted_answer=extraction.value,
                    is_extracted=extraction.is_extracted,
                    is_correct=extraction.is_correct,
                    method=extraction.method,
                    primary_method=extraction.primary_method,
                )
            )
        return tuple(generations)

    def provenance(self) -> dict[str, object]:
        if self._closed or self.model is None or self.tokenizer is None:
            raise RuntimeError("Same-prompt runtime is closed")
        model_metadata = getattr(self.model.config, "_commit_hash", None)
        tokenizer_metadata = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        if model_metadata is not None and model_metadata != self.revision:
            raise ValueError(
                "resolved evaluator model revision differs from the requested source revision"
            )
        if tokenizer_metadata is not None and tokenizer_metadata != self.revision:
            raise ValueError(
                "resolved evaluator tokenizer revision differs from the requested source revision"
            )
        model_revision = model_metadata or self.revision
        tokenizer_revision = tokenizer_metadata or self.revision
        torch = self._torch
        provenance = {
            "operation": "input-corrector-audit",
            "runtime": "HuggingFaceSamePromptRuntime",
            "profile": "neural",
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
            "generation": {"padding_side": "left", **_GENERATION_ARGUMENTS},
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.eos_source,
            "answer_extraction": _ANSWER_EXTRACTION,
            "benchmark_extractor": self.internal_benchmark,
            "implementation_code": implementation_code_identity(),
        }
        validate_paper_runtime_environment(
            provenance,
            profile="neural",
            field_prefix="input-corrector generation environment",
        )
        return provenance

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


__all__ = ["HuggingFaceSamePromptRuntime", "ProductionCorrectionRuntime"]

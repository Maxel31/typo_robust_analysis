"""Hugging Face GPU runtime for independent clean-prefix budget points."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.clean_prefix_scan.planning import (
    CleanCotAlignment,
    PrefixInputPlan,
    align_clean_cot_suffixes,
    build_prefix_input_plan,
)
from typo_cot.experiments.clean_prefix_scan.protocol import (
    ANSWER_DECODING,
    ANSWER_EXTRACTION,
    BENCHMARK_DATASET_NAMES,
    GENERATION,
    IMPLEMENTATION,
    INPUT_ASSEMBLY,
    PRE_ANSWER_BOUNDARY,
    SCAN_BOUNDARY,
    SELECTION_ALIGNMENT,
)

if TYPE_CHECKING:
    from typo_cot.experiments.clean_prefix_scan.runner import (
        CleanPrefixPointScan,
        CleanPrefixScanConfig,
    )

_TRIGGER = re.compile(r"[Tt]he answer is")


class CleanPrefixBoundaryInvalid(ValueError):
    """Selection-aligned target that fails the later edited-prompt audit."""

    def __init__(
        self,
        message: str,
        *,
        alignment: CleanCotAlignment,
        edited_prompt_ids: tuple[int, ...],
        edited_full_ids: tuple[int, ...],
    ) -> None:
        super().__init__(message)
        self.alignment = alignment
        self.edited_prompt_ids = edited_prompt_ids
        self.edited_full_ids = edited_full_ids


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _token_ids(value: object, *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    ids = tuple(value)
    if not ids and not allow_empty:
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return ids


class HuggingFaceCleanPrefixScanRuntime:
    """Load one pinned causal LM and run every prefix point independently."""

    def __init__(self, config: CleanPrefixScanConfig, *, revision: str) -> None:
        if not revision:
            raise ValueError("clean-prefix scan requires a pinned source revision")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != config.gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={config.gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("clean-prefix scan requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "clean-prefix scan requires exactly one visible GPU; set "
                "CUDA_VISIBLE_DEVICES to one physical device"
            )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        from typo_cot.models.wrapper import create_model_wrapper

        self.config = config
        self.revision = revision
        self._torch = torch
        self.wrapper = create_model_wrapper(
            model_name=config.model,
            gpu_id=config.gpu_id,
            dtype=torch.bfloat16,
            wrap_for_lxt=False,
            revision=revision,
        )
        self.model = self.wrapper.model
        self.model.eval()
        self.tokenizer = self.wrapper.tokenizer
        self.tokenizer.padding_side = str(GENERATION["padding_side"])
        self.device = next(self.model.parameters()).device
        self.effective_eos_token_ids, self.effective_eos_token_ids_source = (
            self._resolve_eos_token_ids()
        )

    @staticmethod
    def _normalized_token_ids(value: object) -> tuple[int, ...]:
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

    def _resolve_eos_token_ids(self) -> tuple[tuple[int, ...], str]:
        generation_config = self.model.generation_config
        if getattr(generation_config, "stop_strings", None):
            raise ValueError("clean-prefix scan does not support stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("clean-prefix scan does not support forced_eos_token_id")
        resolved = self._normalized_token_ids(getattr(generation_config, "eos_token_id", None))
        if resolved:
            return resolved, "model-generation-config"
        resolved = self._normalized_token_ids(getattr(self.tokenizer, "eos_token_id", None))
        if resolved:
            return resolved, "tokenizer-fallback"
        raise ValueError("model generation config and tokenizer expose no EOS token ID")

    def _encode(self, text: str, *, field: str) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError(f"tokenizer returned no input_ids for {field}")
        return _token_ids(encoded["input_ids"], field=field)

    def prepare_pair(self, pair: Mapping[str, object]) -> PrefixInputPlan:
        """Tokenize one source record and apply both legacy boundary stages."""

        clean = _mapping(pair.get("clean"), field="pair.clean")
        edited = _mapping(pair.get("edited"), field="pair.edited")
        clean_prompt = _text(clean.get("prompt"), field="pair.clean.prompt")
        edited_prompt = _text(edited.get("prompt"), field="pair.edited.prompt")
        continuation = _text(clean.get("continuation"), field="pair.clean.continuation")
        trigger = _TRIGGER.search(continuation)
        if trigger is None:
            raise ValueError("clean continuation has no submitted pre-answer trigger")
        pre_answer = continuation[: trigger.start()]
        if not pre_answer.strip():
            raise ValueError("clean continuation has an empty pre-answer prefix")

        clean_prompt_ids = self._encode(clean_prompt, field="clean prompt")
        edited_prompt_ids = self._encode(edited_prompt, field="edited prompt")
        recorded_clean_count = clean.get("prompt_token_count")
        recorded_edited_count = edited.get("prompt_token_count")
        if recorded_clean_count != len(clean_prompt_ids):
            raise ValueError("clean prompt token count differs from the prepared pair")
        if recorded_edited_count != len(edited_prompt_ids):
            raise ValueError("edited prompt token count differs from the prepared pair")
        clean_full_ids = self._encode(clean_prompt + pre_answer, field="clean full input")
        edited_full_ids = self._encode(edited_prompt + pre_answer, field="edited full input")
        alignment = align_clean_cot_suffixes(
            clean_prompt_ids=clean_prompt_ids,
            clean_full_ids=clean_full_ids,
            edited_prompt_ids=edited_prompt_ids,
            edited_full_ids=edited_full_ids,
        )
        try:
            return build_prefix_input_plan(
                edited_prompt_ids=edited_prompt_ids,
                edited_full_ids=edited_full_ids,
                alignment=alignment,
            )
        except ValueError as exc:
            raise CleanPrefixBoundaryInvalid(
                str(exc),
                alignment=alignment,
                edited_prompt_ids=edited_prompt_ids,
                edited_full_ids=edited_full_ids,
            ) from exc

    def generate_point(
        self,
        plan: PrefixInputPlan,
        k: int,
        *,
        gold_answer: str,
    ) -> CleanPrefixPointScan:
        """Generate only the continuation for one exact token-prefix input."""

        from typo_cot.experiments.clean_prefix_scan.runner import (
            CleanPrefixGeneration,
            CleanPrefixInputUse,
            CleanPrefixPointScan,
        )

        if not isinstance(plan, PrefixInputPlan):
            raise TypeError("plan must be a PrefixInputPlan")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("gold_answer must be a non-empty string")
        input_token_ids = plan.input_ids(k)
        input_ids = self._torch.tensor(
            [input_token_ids],
            dtype=self._torch.long,
            device=self.device,
        )
        attention_mask = self._torch.ones_like(input_ids)
        generation_arguments = {
            key: value
            for key, value in GENERATION.items()
            if key not in {"padding_side", "batch_size"}
        }
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_arguments,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        if output_ids.ndim != 2 or int(output_ids.shape[0]) != 1:
            raise ValueError("model.generate returned an invalid clean-prefix batch")
        output_row = tuple(int(token) for token in output_ids[0].detach().cpu().tolist())
        if output_row[: len(input_token_ids)] != input_token_ids:
            raise ValueError("model.generate changed the fixed clean-prefix input")
        generated_ids = output_row[len(input_token_ids) :]
        ended_with_eos = bool(generated_ids and generated_ids[-1] in self.effective_eos_token_ids)
        stop_reason = "eos_token" if ended_with_eos else "max_new_tokens"
        stop_token_id = generated_ids[-1] if ended_with_eos else None
        text = self.tokenizer.decode(
            list(generated_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        extraction = extract_with_fallback(
            text,
            benchmark=BENCHMARK_DATASET_NAMES[self.config.benchmark],
            correct_answer=gold_answer,
            allow_positional=ended_with_eos,
        )
        return CleanPrefixPointScan(
            k=k,
            input_use=CleanPrefixInputUse(
                k=k,
                input_ids=input_token_ids,
                prompt_token_count=len(plan.edited_prompt_ids),
                prefix_token_count=k,
            ),
            generation=CleanPrefixGeneration(
                token_ids=generated_ids,
                text=text,
                value=extraction.value,
                is_extracted=extraction.is_extracted,
                is_correct=extraction.is_correct,
                method=extraction.method,
                primary_method=extraction.primary_method,
                stop_reason=stop_reason,  # type: ignore[arg-type]
                stop_token_id=stop_token_id,
            ),
        )

    def provenance(self) -> Mapping[str, object]:
        """Return immutable model/tokenizer and explicit decoding provenance."""

        model_revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        tokenizer_revision = self.tokenizer.init_kwargs.get("_commit_hash") or self.revision
        return {
            "runtime": IMPLEMENTATION,
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "numpy": _package_version("numpy"),
            "requested_revision": self.revision,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "dtype": "bfloat16",
            "device": str(self.device),
            "cuda": self._torch.version.cuda,
            "gpu_name": self._torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "generation": dict(GENERATION),
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.effective_eos_token_ids_source,
            "answer_decoding": ANSWER_DECODING,
            "answer_extraction": ANSWER_EXTRACTION,
            "pre_answer_boundary": PRE_ANSWER_BOUNDARY,
            "selection_alignment": SELECTION_ALIGNMENT,
            "scan_boundary": SCAN_BOUNDARY,
            "input_assembly": INPUT_ASSEMBLY,
        }


__all__ = ["CleanPrefixBoundaryInvalid", "HuggingFaceCleanPrefixScanRuntime"]

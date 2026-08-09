"""Hugging Face GPU runtime for complete-text CoT swapping."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import typo_cot.experiments.cot_swap.protocol as cot_swap_protocol
from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.cot_swap.planning import CELL_ORDER, CellPlan, CotSwapPlan

if TYPE_CHECKING:
    from typo_cot.experiments.cot_swap.runner import (
        CotSwapConfig,
        CotSwapGeneration,
        CotSwapInputUse,
        CotSwapScan,
    )

_ANSWER_EXTRACTION = cot_swap_protocol.ANSWER_EXTRACTION
_ANSWER_SPAN_DECODING = cot_swap_protocol.ANSWER_SPAN_DECODING
_BATCHING = cot_swap_protocol.BATCHING
_BENCHMARK_NAMES = cot_swap_protocol.BENCHMARK_DATASET_NAMES
_GENERATION = cot_swap_protocol.GENERATION
_IMPLEMENTATION = cot_swap_protocol.IMPLEMENTATION
_TEXT_INTERVENTION = cot_swap_protocol.TEXT_INTERVENTION


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    """Normalize one unbatched tokenizer result without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    ids = tuple(value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return ids


def _ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        list(token_ids),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class HuggingFaceCotSwapRuntime:
    """Load one pinned causal LM and regenerate the four answer-span cells in a batch."""

    def __init__(self, config: CotSwapConfig, *, revision: str) -> None:
        if not revision:
            raise ValueError("cot-swap requires a pinned source revision")
        self.revision = revision
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != config.gpu_id:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES conflicts with --gpu-id: "
                f"environment={visible!r}, argument={config.gpu_id!r}"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cot-swap requires the requested CUDA GPU")
        requested_device_count = len(config.gpu_id.split(","))
        if torch.cuda.device_count() != requested_device_count:
            raise RuntimeError(
                "cot-swap visible GPU count does not match --gpu-id: "
                f"expected {requested_device_count}, found {torch.cuda.device_count()}"
            )

        from typo_cot.models.wrapper import create_model_wrapper

        self.config = config
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
        self.tokenizer.padding_side = str(_GENERATION["padding_side"])
        self.device = next(self.model.parameters()).device
        (
            self.effective_eos_token_ids,
            self.effective_eos_token_ids_source,
        ) = self._resolve_eos_token_ids()

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
            raise ValueError("cot-swap does not support generation-config stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("cot-swap does not support forced_eos_token_id")
        resolved = self._normalized_token_ids(getattr(generation_config, "eos_token_id", None))
        if resolved:
            return resolved, "model-generation-config"
        resolved = self._normalized_token_ids(getattr(self.tokenizer, "eos_token_id", None))
        if resolved:
            return resolved, "tokenizer-fallback"
        raise ValueError("model generation config and tokenizer expose no EOS token ID")

    @property
    def _extraction_benchmark(self) -> str:
        return _BENCHMARK_NAMES[self.config.benchmark]

    def _answer_generation(
        self,
        token_ids: tuple[int, ...],
        text: str,
        gold_answer: str,
        *,
        stop_reason: str,
        stop_token_id: int | None,
    ) -> CotSwapGeneration:
        from typo_cot.experiments.cot_swap.runner import CotSwapGeneration

        extraction = extract_with_fallback(
            text,
            benchmark=self._extraction_benchmark,
            correct_answer=gold_answer,
            allow_positional=stop_reason == "eos_token",
        )
        return CotSwapGeneration(
            token_ids=token_ids,
            text=text,
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
            stop_reason=stop_reason,
            stop_token_id=stop_token_id,
        )

    def _encode_unpadded(self, text: str, *, field: str) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError(f"tokenizer returned no input_ids for {field}")
        return _token_ids(encoded["input_ids"], field=field)

    def _generate_cell_batch(
        self,
        cell_plans: tuple[CellPlan, ...],
    ) -> dict[str, tuple[CotSwapInputUse, tuple[int, ...], str, str, int | None]]:
        from typo_cot.experiments.cot_swap.runner import CotSwapInputUse

        if tuple(cell.cell for cell in cell_plans) != CELL_ORDER:
            raise ValueError("CoT-swap runtime requires A, B, C, and D in paper order")

        prompt_ids: dict[str, tuple[int, ...]] = {}
        full_ids: dict[str, tuple[int, ...]] = {}
        for cell in cell_plans:
            prompt_ids[cell.cell] = self._encode_unpadded(
                cell.prompt,
                field=f"cell {cell.cell} prompt",
            )
            if len(prompt_ids[cell.cell]) != cell.prompt_token_count:
                raise ValueError(
                    f"runtime cell {cell.cell} prompt token count differs from pair record"
                )
            full_ids[cell.cell] = self._encode_unpadded(
                cell.full_input,
                field=f"cell {cell.cell} full input",
            )

        encoded = self.tokenizer(
            [cell.full_input for cell in cell_plans],
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer returned no batched input_ids")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != len(CELL_ORDER):
            raise ValueError("tokenizer must return one batched sequence for every CoT-swap cell")
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer returned an invalid batched attention mask")
        for index, cell in enumerate(cell_plans):
            batch_ids = tuple(
                int(token)
                for token, attended in zip(
                    input_ids[index].detach().cpu().tolist(),
                    attention_mask[index].detach().cpu().tolist(),
                    strict=True,
                )
                if int(attended) == 1
            )
            if batch_ids != full_ids[cell.cell]:
                raise ValueError(
                    f"batched tokenization differs from exact full input for cell {cell.cell}"
                )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        generation_arguments = {
            key: value for key, value in _GENERATION.items() if key != "padding_side"
        }
        max_new_tokens = int(_GENERATION["max_new_tokens"])
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_arguments,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.effective_eos_token_ids),
            )
        if (
            not hasattr(output_ids, "ndim")
            or output_ids.ndim != 2
            or int(output_ids.shape[0]) != len(CELL_ORDER)
            or int(output_ids.shape[1]) <= int(input_ids.shape[1])
            or int(output_ids.shape[1]) > int(input_ids.shape[1]) + max_new_tokens
        ):
            raise ValueError("generation must return a non-empty answer span for every cell")

        generated: dict[
            str,
            tuple[CotSwapInputUse, tuple[int, ...], str, str, int | None],
        ] = {}
        prompt_width = int(input_ids.shape[1])
        for index, cell in enumerate(cell_plans):
            raw_continuation = tuple(
                int(token) for token in output_ids[index, prompt_width:].detach().cpu().tolist()
            )
            if not raw_continuation:
                raise ValueError(f"cell {cell.cell} generated no continuation token")
            eos_index = next(
                (
                    position
                    for position, token in enumerate(raw_continuation)
                    if token in self.effective_eos_token_ids
                ),
                None,
            )
            if eos_index is None:
                if len(raw_continuation) != max_new_tokens:
                    raise ValueError(
                        f"cell {cell.cell} stopped without EOS before the "
                        f"{max_new_tokens}-token cap"
                    )
                continuation = raw_continuation
                stop_reason = "max_new_tokens"
                stop_token_id = None
            else:
                continuation = raw_continuation[: eos_index + 1]
                stop_reason = "eos_token"
                stop_token_id = continuation[-1]
            text = self.tokenizer.decode(
                list(continuation),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            use = CotSwapInputUse(
                cell=cell.cell,
                prompt_text_sha256=cell.prompt_sha256,
                pre_answer_text_sha256=cell.pre_answer_sha256,
                full_input_text_sha256=cell.full_input_sha256,
                prompt_char_count=len(cell.prompt),
                pre_answer_char_count=len(cell.pre_answer_text),
                full_input_char_count=len(cell.full_input),
                prompt_token_count=len(prompt_ids[cell.cell]),
                full_input_token_count=len(full_ids[cell.cell]),
                full_input_ids_sha256=_ids_sha256(full_ids[cell.cell]),
                prompt_prefix_token_stable=(
                    full_ids[cell.cell][: len(prompt_ids[cell.cell])] == prompt_ids[cell.cell]
                ),
            )
            generated[cell.cell] = (
                use,
                continuation,
                text,
                stop_reason,
                stop_token_id,
            )
        return generated

    def scan_pair(self, pair: dict[str, object], plan: CotSwapPlan) -> CotSwapScan:
        from typo_cot.experiments.cot_swap.runner import CotSwapScan

        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("pair sample_id must be non-empty")
        if plan.sample_id != sample_id:
            raise ValueError("CoT-swap plan sample_id does not match the pair")
        if not plan.eligible:
            raise ValueError("CoT-swap runtime received an excluded pair")
        if tuple(cell.cell for cell in plan.cells) != CELL_ORDER:
            raise ValueError("CoT-swap plan cells are not in paper order")
        gold_answer = pair.get("gold_answer")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("pair gold_answer must be non-empty")

        generated = self._generate_cell_batch(plan.cells)
        input_uses = {cell: generated[cell][0] for cell in CELL_ORDER}
        generations = {
            cell: self._answer_generation(
                generated[cell][1],
                generated[cell][2],
                gold_answer,
                stop_reason=generated[cell][3],
                stop_token_id=generated[cell][4],
            )
            for cell in CELL_ORDER
        }
        return CotSwapScan(
            sample_id=sample_id,
            input_uses=input_uses,
            generations=generations,
        )

    def provenance(self) -> dict[str, object]:
        torch = self._torch
        model_metadata_revision = getattr(self.model.config, "_commit_hash", None)
        model_revision = model_metadata_revision or self.revision
        tokenizer_metadata_revision = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        tokenizer_revision = tokenizer_metadata_revision or self.revision
        payload: dict[str, object] = {
            "operation": "cot-swap",
            "runtime": "HuggingFaceCotSwapRuntime",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "model": self.config.model,
            "requested_revision": self.revision,
            "model_revision": model_revision,
            "model_revision_source": (
                "model-config-metadata" if model_metadata_revision else "explicit-load-revision"
            ),
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_revision_source": (
                "tokenizer-init-metadata"
                if tokenizer_metadata_revision
                else "explicit-load-revision"
            ),
            "dtype": "bfloat16",
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "generation": {
                **_GENERATION,
                "eos_token_id": list(self.effective_eos_token_ids),
            },
            "implementation": _IMPLEMENTATION,
            "batching": dict(_BATCHING),
            "answer_span_decoding": dict(_ANSWER_SPAN_DECODING),
            "effective_eos_token_ids": list(self.effective_eos_token_ids),
            "effective_eos_token_ids_source": self.effective_eos_token_ids_source,
            "answer_extraction": _ANSWER_EXTRACTION,
            "text_intervention": dict(_TEXT_INTERVENTION),
        }
        device_count = torch.cuda.device_count()
        if device_count == 1:
            # Retain the established single-GPU manifest byte shape.
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        else:
            payload["model_parallel"] = True
            payload["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(device_count)
            ]
            payload["gpu_total_memory_bytes"] = [
                torch.cuda.get_device_properties(index).total_memory
                for index in range(device_count)
            ]
            raw_device_map = getattr(self.model, "hf_device_map", {})
            payload["model_device_map"] = {
                str(key): value if isinstance(value, (int, str)) else str(value)
                for key, value in sorted(raw_device_map.items())
            }
        return payload

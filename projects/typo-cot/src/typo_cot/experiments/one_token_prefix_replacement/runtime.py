"""Hugging Face GPU runtime for one-token profiling and generation."""

# Model/tokenizer payload shape violations are exposed as reproducibility-value errors.
# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from typo_cot.evaluation.fallback import extract_with_fallback
from typo_cot.experiments.clean_prefix_scan.planning import align_clean_cot_suffixes
from typo_cot.experiments.one_token_prefix_replacement.integrity import (
    implementation_code_identity,
)
from typo_cot.experiments.one_token_prefix_replacement.planning import (
    OneTokenInputPlan,
    OneTokenProfile,
)
from typo_cot.experiments.one_token_prefix_replacement.protocol import (
    ANSWER_DECODING,
    ANSWER_EXTRACTION,
    BENCHMARK_DATASET_NAMES,
    GENERATION,
    GENERATION_INPUT,
    IMPLEMENTATION,
    PRE_ANSWER_BOUNDARY,
    PROFILE_INPUT,
    TOKEN_ADMISSIBILITY,
)

if TYPE_CHECKING:
    from typo_cot.experiments.one_token_prefix_replacement.runner import (
        OneTokenGeneration,
        OneTokenPrefixReplacementConfig,
    )


_TRIGGER = re.compile(r"[Tt]he answer is")
_MARKER_RE = re.compile(str(TOKEN_ADMISSIBILITY["marker_regex"]), re.IGNORECASE)


class OneTokenBoundaryInvalid(ValueError):
    """Selection-eligible target that fails the exact prompt-prefix audit."""

    def __init__(self, message: str, *, clean_cot_ids: tuple[int, ...]) -> None:
        super().__init__(message)
        ids = tuple(clean_cot_ids)
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids):
            raise ValueError("clean_cot_ids must contain non-negative integer token IDs")
        self.clean_cot_ids = ids

    @property
    def cot_token_count(self) -> int:
        return len(self.clean_cot_ids)

    @property
    def eligible_length(self) -> bool:
        return 8 <= self.cot_token_count <= 512


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


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"tokenizer returned no token IDs for {field}")
    result = tuple(value)
    if not result or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result
    ):
        raise ValueError(f"tokenizer returned invalid token IDs for {field}")
    return result


def _tokenizer_candidate_pool(
    tokenizer: object,
    *,
    model_logit_size: int,
) -> tuple[frozenset[int], dict[str, object]]:
    """Reconstruct the submitted producer's real, non-special token pool."""

    if not isinstance(model_logit_size, int) or isinstance(model_logit_size, bool):
        raise ValueError("model logit size must be an integer")
    if model_logit_size <= 0:
        raise ValueError("model logit size must be positive")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer exposes no vocabulary")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("tokenizer vocabulary must be a non-empty mapping")
    if any(
        not isinstance(token, str) or not isinstance(token_id, int) or isinstance(token_id, bool)
        for token, token_id in vocab.items()
    ):
        raise ValueError("tokenizer vocabulary has invalid token entries")
    real_ids = {token_id for token_id in vocab.values() if 0 <= token_id < model_logit_size}
    added_decoder = getattr(tokenizer, "added_tokens_decoder", {})
    if not isinstance(added_decoder, Mapping):
        raise ValueError("tokenizer added-token decoder must be a mapping")
    added_special_ids = {
        token_id
        for token_id, token in added_decoder.items()
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and 0 <= token_id < model_logit_size
        and getattr(token, "special", False) is True
    }
    raw_special_ids = getattr(tokenizer, "all_special_ids", ())
    if not isinstance(raw_special_ids, (tuple, list, set)):
        raise ValueError("tokenizer all_special_ids must be a sequence")
    special_ids = {
        token_id
        for token_id in raw_special_ids
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and 0 <= token_id < model_logit_size
    } | added_special_ids
    marker_ids = {
        token_id
        for token, token_id in vocab.items()
        if 0 <= token_id < model_logit_size and _MARKER_RE.fullmatch(token)
    }
    admissible = frozenset(real_ids - special_ids - marker_ids)
    admissible_digest = hashlib.sha256()
    for token_id in sorted(admissible):
        admissible_digest.update(str(token_id).encode("ascii"))
        admissible_digest.update(b"\n")
    return admissible, {
        "implementation": TOKEN_ADMISSIBILITY["implementation"],
        "model_logit_size": model_logit_size,
        "tokenizer_vocab_entries": len(vocab),
        "n_real_tokenizer_ids_in_logits": len(real_ids),
        "n_special_ids": len(special_ids),
        "n_marker_ids": len(marker_ids),
        "n_admissible_ids": len(admissible),
        "admissible_token_ids_sha256_algorithm": TOKEN_ADMISSIBILITY[
            "admissible_token_ids_sha256_algorithm"
        ],
        "admissible_token_ids_sha256": admissible_digest.hexdigest(),
        "marker_regex": TOKEN_ADMISSIBILITY["marker_regex"],
    }


class HuggingFaceOneTokenPrefixReplacementRuntime:
    """Load one pinned causal LM and execute the final-PDF diagnostic."""

    def __init__(
        self,
        config: OneTokenPrefixReplacementConfig,
        *,
        revision: str,
    ) -> None:
        if not revision:
            raise ValueError("one-token diagnostic requires a pinned source revision")
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
            raise RuntimeError("one-token diagnostic requires the requested CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "one-token diagnostic requires exactly one visible GPU; set "
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
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer exposes no pad_token_id")
        self.device = next(self.model.parameters()).device
        output_embeddings = self.model.get_output_embeddings()
        weight = getattr(output_embeddings, "weight", None)
        shape = getattr(weight, "shape", ())
        if not isinstance(shape, (tuple, list)) and not hasattr(shape, "__len__"):
            raise ValueError("model output embeddings expose no vocabulary dimension")
        if len(shape) < 1:
            raise ValueError("model output embeddings expose no vocabulary dimension")
        self.admissible_token_ids, self.token_pool_stats = _tokenizer_candidate_pool(
            self.tokenizer,
            model_logit_size=int(shape[0]),
        )
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
            raise ValueError("one-token diagnostic does not support stop_strings")
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("one-token diagnostic does not support forced_eos_token_id")
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

    def prepare_pair(self, pair: Mapping[str, object]) -> OneTokenInputPlan:
        """Tokenize the exact clean pre-answer suffix under both prompts."""

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
        if clean.get("prompt_token_count") != len(clean_prompt_ids):
            raise ValueError("clean prompt token count differs from the prepared pair")
        if edited.get("prompt_token_count") != len(edited_prompt_ids):
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
            return OneTokenInputPlan(
                clean_prompt_ids=clean_prompt_ids,
                edited_prompt_ids=edited_prompt_ids,
                clean_full_ids=clean_full_ids,
                edited_full_ids=edited_full_ids,
                clean_cot_ids=alignment.clean_cot_ids,
            )
        except ValueError as exc:
            raise OneTokenBoundaryInvalid(
                str(exc),
                clean_cot_ids=alignment.clean_cot_ids,
            ) from exc

    def profile_pair(self, plan: OneTokenInputPlan) -> OneTokenProfile:
        """Compute float32 KL, target ranks, and edited-context top-1 IDs."""

        if not isinstance(plan, OneTokenInputPlan):
            raise TypeError("plan must be OneTokenInputPlan")
        clean_input = self._torch.tensor(
            [plan.profile_clean_input_ids], dtype=self._torch.long, device=self.device
        )
        edited_input = self._torch.tensor(
            [plan.profile_edited_input_ids], dtype=self._torch.long, device=self.device
        )
        with self._torch.inference_mode():
            clean_output = self.model(input_ids=clean_input).logits[0]
            clean_logits = clean_output[
                len(plan.clean_prompt_ids) - 1 : len(plan.clean_prompt_ids)
                - 1
                + plan.cot_token_count
            ].detach()
            del clean_output
            edited_output = self.model(input_ids=edited_input).logits[0]
            edited_logits = edited_output[
                len(plan.edited_prompt_ids) - 1 : len(plan.edited_prompt_ids)
                - 1
                + plan.cot_token_count
            ].detach()
            del edited_output
        if (
            clean_logits.shape != edited_logits.shape
            or int(clean_logits.shape[0]) != plan.cot_token_count
        ):
            raise ValueError("profile forward returned the wrong position grid")

        kl_values: list[float] = []
        clean_ranks: list[int] = []
        edited_ranks: list[int] = []
        edited_top1: list[int] = []
        targets = self._torch.tensor(
            plan.clean_cot_ids,
            dtype=self._torch.long,
            device=clean_logits.device,
        )
        chunk_size = 16
        with self._torch.inference_mode():
            for start in range(0, plan.cot_token_count, chunk_size):
                stop = min(start + chunk_size, plan.cot_token_count)
                clean = clean_logits[start:stop].float()
                edited = edited_logits[start:stop].float()
                target = targets[start:stop]
                clean_logp = self._torch.log_softmax(clean, dim=-1)
                edited_logp = self._torch.log_softmax(edited, dim=-1)
                clean_p = clean_logp.exp()
                kl = (clean_p * (clean_logp - edited_logp)).sum(dim=-1)
                kl_values.extend(max(0.0, float(value)) for value in kl.cpu().tolist())
                target_clean = clean.gather(1, target[:, None])
                target_edited = edited.gather(1, target[:, None])
                clean_ranks.extend(
                    int(value) for value in (1 + (clean > target_clean).sum(dim=-1)).cpu().tolist()
                )
                edited_ranks.extend(
                    int(value)
                    for value in (1 + (edited > target_edited).sum(dim=-1)).cpu().tolist()
                )
                edited_top1.extend(int(value) for value in edited.argmax(dim=-1).cpu().tolist())
        del clean_logits, edited_logits, clean_input, edited_input, targets
        return OneTokenProfile(
            clean_to_edited_kl=tuple(kl_values),
            clean_token_rank_under_clean=tuple(clean_ranks),
            clean_token_rank_under_edited=tuple(edited_ranks),
            edited_top1_ids=tuple(edited_top1),
            edited_top1_is_admissible=tuple(
                token_id in self.admissible_token_ids for token_id in edited_top1
            ),
        )

    def generate_arm(
        self,
        plan: OneTokenInputPlan,
        *,
        position: int,
        forced_token_id: int,
        gold_answer: str,
    ) -> OneTokenGeneration:
        """Generate from direct IDs and decode only the newly generated suffix."""

        from typo_cot.experiments.one_token_prefix_replacement.runner import (
            OneTokenGeneration,
        )

        if not isinstance(plan, OneTokenInputPlan):
            raise TypeError("plan must be OneTokenInputPlan")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("gold_answer must be a non-empty string")
        input_token_ids = plan.generation_input_ids(position, forced_token_id)
        input_ids = self._torch.tensor(
            [input_token_ids], dtype=self._torch.long, device=self.device
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
            raise ValueError("model.generate returned an invalid one-token batch")
        output_row = tuple(int(token) for token in output_ids[0].detach().cpu().tolist())
        if output_row[: len(input_token_ids)] != input_token_ids:
            raise ValueError("model.generate changed the fixed one-token input")
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
        return OneTokenGeneration(
            token_ids=generated_ids,
            text=text,
            value=extraction.value,
            is_extracted=extraction.is_extracted,
            is_correct=extraction.is_correct,
            method=extraction.method,
            primary_method=extraction.primary_method,
            stop_reason=stop_reason,  # type: ignore[arg-type]
            stop_token_id=stop_token_id,
        )

    def provenance(self) -> Mapping[str, object]:
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
            "profile_input": PROFILE_INPUT,
            "generation_input": GENERATION_INPUT,
            "token_admissibility": dict(self.token_pool_stats),
            "implementation_code_identity": implementation_code_identity(),
        }


__all__ = [
    "HuggingFaceOneTokenPrefixReplacementRuntime",
    "OneTokenBoundaryInvalid",
    "implementation_code_identity",
]

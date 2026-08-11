"""Hugging Face GPU runtime for teacher-forced multi-token KL patching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typo_cot.experiments.fixed_window_answer_patching.runtime import (
    HuggingFaceFixedWindowAnswerPatchingRuntime,
)
from typo_cot.experiments.rebuttal_runtime import manifest_runtime_pair


@dataclass(frozen=True, slots=True)
class _RuntimeConfig:
    model: str
    benchmark: str
    gpu_id: str


def _flat_token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value
    ):
        raise ValueError(f"{field} must be one token-ID sequence")
    return tuple(value)


def clean_continuation_target_ids(
    tokenizer: Any,
    *,
    clean_prompt: str,
    clean_continuation: str,
    count: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Derive targets only when prompt tokenization is an exact full-text prefix."""

    if not isinstance(clean_prompt, str) or not clean_prompt:
        raise ValueError("clean_prompt must be non-empty")
    if not isinstance(clean_continuation, str):
        raise ValueError("clean_continuation must be a string")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    kwargs = {"add_special_tokens": True, "return_attention_mask": False}
    prompt_encoding = tokenizer(clean_prompt, **kwargs)
    full_encoding = tokenizer(clean_prompt + clean_continuation, **kwargs)
    if not isinstance(prompt_encoding, Mapping) or not isinstance(full_encoding, Mapping):
        raise ValueError("tokenizer must return mapping-like encodings")
    prompt_ids = _flat_token_ids(prompt_encoding.get("input_ids"), field="prompt input_ids")
    full_ids = _flat_token_ids(full_encoding.get("input_ids"), field="full input_ids")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("clean prompt is not an exact token-ID prefix of prompt plus continuation")
    suffix = full_ids[len(prompt_ids) :]
    if len(suffix) < count:
        raise ValueError(f"clean continuation contains fewer than {count} token IDs")
    targets = suffix[:count]
    raw_token_text = tokenizer.convert_ids_to_tokens(list(targets))
    if not isinstance(raw_token_text, list) or len(raw_token_text) != len(targets):
        raise ValueError("tokenizer must return one vocabulary piece per target token")
    token_text = tuple(raw_token_text)
    if any(not isinstance(text, str) for text in token_text):
        raise ValueError("tokenizer vocabulary pieces must be strings")
    return targets, token_text


class HuggingFaceMultiTokenKLReadoutRuntime(HuggingFaceFixedWindowAnswerPatchingRuntime):
    """Run clean, typo, and patched teacher-forced forwards for one setting."""

    def __init__(self, *, model: str, task: str, revision: str, gpu_id: str) -> None:
        super().__init__(
            _RuntimeConfig(model=model, benchmark=task, gpu_id=gpu_id),  # type: ignore[arg-type]
            revision=revision,
        )

    def _append_targets(self, input_ids: Any, attention_mask: Any, targets: tuple[int, ...]) -> Any:
        prefix = self._torch.tensor(
            [targets[:-1]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        suffix_mask = self._torch.ones_like(prefix, dtype=attention_mask.dtype)
        return (
            self._torch.cat((input_ids, prefix), dim=1),
            self._torch.cat((attention_mask, suffix_mask), dim=1),
        )

    @staticmethod
    def _target_logits(output: Any, *, prompt_tokens: int, count: int) -> Any:
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("model output must contain [1, sequence, vocabulary] logits")
        start = prompt_tokens - 1
        selected = logits[0, start : start + count, :]
        if int(selected.shape[0]) != count:
            raise ValueError("model output does not cover every teacher-forced target")
        return selected.detach().float().cpu()

    def scan_pair(
        self,
        pair: Mapping[str, object],
        *,
        teacher_forced_tokens: int,
        layer_window: tuple[int, int],
    ) -> Any:
        from typo_cot.experiments.fixed_window_answer_patching.patching import (
            PrefillBlockOutputWindowPatch,
        )
        from typo_cot.experiments.layerwise_kl_patching.patching import (
            capture_block_outputs,
        )
        from typo_cot.experiments.multitoken_kl_readout.metrics import (
            kl_trajectory_from_logits,
        )
        from typo_cot.experiments.multitoken_kl_readout.runner import MultiTokenKLScan

        if teacher_forced_tokens != 16 or layer_window != (0, 6):
            raise ValueError("runtime arguments differ from the frozen multi-token protocol")
        if self.num_layers < layer_window[1]:
            raise ValueError("multi-token KL readout requires at least six decoder layers")
        runtime_pair = manifest_runtime_pair(pair)
        clean_ids, clean_mask, clean_positions = self._tokenize_and_validate(
            runtime_pair, side="clean"
        )
        typo_ids, typo_mask, typo_positions = self._tokenize_and_validate(
            runtime_pair, side="edited"
        )
        if len(clean_positions) != len(typo_positions):
            raise ValueError("clean and typo alignment cardinalities differ")
        clean_prompt = pair.get("clean_text")
        continuation = pair.get("clean_continuation")
        if not isinstance(clean_prompt, str) or not isinstance(continuation, str):
            raise ValueError("manifest clean prompt/continuation fields differ")
        target_ids, target_text = clean_continuation_target_ids(
            self.tokenizer,
            clean_prompt=clean_prompt,
            clean_continuation=continuation,
            count=teacher_forced_tokens,
        )
        clean_full_ids, clean_full_mask = self._append_targets(clean_ids, clean_mask, target_ids)
        typo_full_ids, typo_full_mask = self._append_targets(typo_ids, typo_mask, target_ids)
        clean_holder: dict[str, Any] = {}

        def clean_forward() -> Any:
            output = self.model(
                input_ids=clean_full_ids,
                attention_mask=clean_full_mask,
                use_cache=False,
            )
            clean_holder["output"] = output
            return output

        start, stop = layer_window
        with self._torch.inference_mode():
            donor_cache = capture_block_outputs(
                self.layers[start:stop],
                positions=clean_positions,
                forward=clean_forward,
            )
            clean_logits = self._target_logits(
                clean_holder["output"],
                prompt_tokens=int(clean_ids.shape[1]),
                count=teacher_forced_tokens,
            )
            typo_output = self.model(
                input_ids=typo_full_ids,
                attention_mask=typo_full_mask,
                use_cache=False,
            )
            typo_logits = self._target_logits(
                typo_output,
                prompt_tokens=int(typo_ids.shape[1]),
                count=teacher_forced_tokens,
            )
            with PrefillBlockOutputWindowPatch(
                self.layers,
                layer_indices=tuple(range(start, stop)),
                positions=typo_positions,
                donor_values=tuple(donor_cache),
            ):
                patched_output = self.model(
                    input_ids=typo_full_ids,
                    attention_mask=typo_full_mask,
                    use_cache=False,
                )
            patched_logits = self._target_logits(
                patched_output,
                prompt_tokens=int(typo_ids.shape[1]),
                count=teacher_forced_tokens,
            )

        return MultiTokenKLScan(
            target_token_ids=target_ids,
            target_token_text=target_text,
            untreated_kl=kl_trajectory_from_logits(
                clean_logits,
                typo_logits,
                negative_roundoff_tolerance=1e-12,
            ),
            patched_kl=kl_trajectory_from_logits(
                clean_logits,
                patched_logits,
                negative_roundoff_tolerance=1e-12,
            ),
        )

    def provenance(self) -> dict[str, object]:
        payload = super().provenance()
        payload.pop("generation", None)
        payload.pop("answer_extraction", None)
        payload.update(
            {
                "operation": "multitoken-kl-readout",
                "runtime": "HuggingFaceMultiTokenKLReadoutRuntime",
                "task": self.config.benchmark,
                "coordinate_source": "rebuttal-pair-manifest/v1",
                "layer_window": [0, 6],
                "teacher_forced_tokens": 16,
                "target_source": "clean-continuation-token-ids/v1",
                "prompt_prefix_validation": "exact-token-id-prefix/v1",
                "model_inputs": "prompt-plus-first-15-target-token-ids/v1",
                "divergence": "kl-clean-to-condition/v1",
                "negative_kl_roundoff_tolerance": 1e-12,
                "forward": {
                    "use_cache": False,
                    "logits_materialized_dtype": "float32",
                    "kl_dtype": "float64",
                },
            }
        )
        return payload


__all__ = ["HuggingFaceMultiTokenKLReadoutRuntime", "clean_continuation_target_ids"]

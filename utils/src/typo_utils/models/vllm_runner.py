"""vLLM 推論ラッパー（任意依存: extra ``vllm``）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]


@dataclass
class GenerationOutput:
    text: str
    token_logprobs: list[float] | None = None


class VLLMRunner:
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        gpu_ids: list[int] | None = None,
        max_model_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        if LLM is None:
            raise ImportError(
                "vllm が必要です。`uv sync --extra vllm` を実行してください。"
            )

        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            **kwargs,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)

    def __enter__(self) -> VLLMRunner:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1,
            **kwargs,
        )
        outputs = self._llm.generate(prompts, params)
        results = []
        for out in outputs:
            completion = out.outputs[0]
            logprobs = None
            if completion.logprobs:
                logprobs = [
                    lp[token_id].logprob
                    for lp, token_id in zip(completion.logprobs, completion.token_ids)
                    if lp
                ]
            results.append(GenerationOutput(text=completion.text, token_logprobs=logprobs))
        return results

    def score_log_likelihood(
        self,
        prompts: list[str],
        continuations: list[list[str]],
    ) -> list[list[float]]:
        flat_prompts: list[str] = []
        structure: list[int] = []
        for prompt, conts in zip(prompts, continuations):
            structure.append(len(conts))
            for cont in conts:
                flat_prompts.append(prompt + cont)

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        outputs = self._llm.generate(flat_prompts, params)

        flat_scores: list[float] = []
        for i, (out, (prompt, conts)) in enumerate(
            zip(outputs, ((p, c) for p, cs in zip(prompts, continuations) for c in cs))
        ):
            pass

        flat_scores = []
        idx = 0
        for prompt, conts in zip(prompts, continuations):
            for cont in conts:
                out = outputs[idx]
                prompt_logprobs = out.prompt_logprobs or []
                prompt_tokens = len(self._llm.get_tokenizer().encode(prompt))
                cont_logprobs = []
                for j in range(prompt_tokens, len(prompt_logprobs)):
                    lp = prompt_logprobs[j]
                    if lp:
                        cont_logprobs.append(max(lp.values(), key=lambda x: x.logprob).logprob)
                flat_scores.append(sum(cont_logprobs))
                idx += 1

        results: list[list[float]] = []
        offset = 0
        for n in structure:
            results.append(flat_scores[offset : offset + n])
            offset += n
        return results

    def compute_perplexity(
        self,
        texts: list[str],
    ) -> list[float]:
        import math

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        outputs = self._llm.generate(texts, params)
        perplexities: list[float] = []
        for out in outputs:
            prompt_logprobs = out.prompt_logprobs or []
            total = 0.0
            count = 0
            for lp in prompt_logprobs:
                if lp:
                    total += max(lp.values(), key=lambda x: x.logprob).logprob
                    count += 1
            if count == 0:
                perplexities.append(float("inf"))
            else:
                perplexities.append(math.exp(-total / count))
        return perplexities

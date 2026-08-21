"""Implementation of UCC-Inj experiment 6.

The reference repository compares every GSM8K test question with variants in
which one, two, or three random Unicode variation selectors are inserted after
each character.  It feeds each clean/noisy question through the same chat
template, extracts the hidden state at the *last model-input token*, then
reports clean--noisy cosine similarity for every hidden-state index.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .noise import inject_variation_selector_noise


@dataclass(frozen=True)
class Exp6Config:
    model: str
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "test"
    question_field: str = "question"
    seed: int = 42
    noise_levels: tuple[int, ...] = (1, 2, 3)
    limit: int | None = None
    max_length: int = 2048
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    add_generation_prompt: bool = False

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if not self.noise_levels or any(level <= 0 for level in self.noise_levels):
            raise ValueError("noise_levels must contain one or more positive integers")
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise_levels must not contain duplicates")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when supplied")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")


def stable_example_seed(*, root_seed: int, example_index: int, noise_level: int) -> int:
    """Derive a cross-process-stable seed; Python's hash is intentionally avoided."""
    payload = f"ucc-inj-exp6/v1:{root_seed}:{example_index}:{noise_level}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def last_non_padding_index(attention_mask: Any) -> int:
    """Return the final valid token index for a single-example attention mask."""
    values = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
    valid = np.flatnonzero(values)
    if valid.size == 0:
        raise ValueError("attention mask has no non-padding tokens")
    return int(valid[-1])


def cosine_per_layer(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for matching ``[layer, hidden]`` arrays."""
    clean_values = np.asarray(clean, dtype=np.float64)
    noisy_values = np.asarray(noisy, dtype=np.float64)
    if clean_values.shape != noisy_values.shape or clean_values.ndim != 2:
        raise ValueError("clean and noisy states must both have shape [layers, hidden]")
    denominator = np.linalg.norm(clean_values, axis=1) * np.linalg.norm(noisy_values, axis=1)
    result = np.full(clean_values.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = np.sum(clean_values[valid] * noisy_values[valid], axis=1) / denominator[valid]
    return result


def summarise_layer_cosines(records: Iterable[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Aggregate per-example cosine records into one row per noise-level/layer."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        level = int(record["noise_level"])
        for layer, value in enumerate(record["cosine_similarity"]):
            numeric = float(value)
            if np.isfinite(numeric):
                grouped.setdefault((level, layer), []).append(numeric)
    return [
        {
            "noise_level": level,
            "layer": layer,
            "n": len(values),
            "mean_cosine": float(np.mean(values)),
            "median_cosine": float(np.median(values)),
            "std_cosine": float(np.std(values, ddof=0)),
        }
        for (level, layer), values in sorted(grouped.items())
    ]


class HiddenStateExtractor:
    """Extract last-input-token hidden states without depending on padding side."""

    def __init__(self, *, tokenizer: Any, model: Any, config: Exp6Config, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.torch = torch_module

    def states(self, text: str) -> np.ndarray:
        messages = [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.config.add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model(**encoded, output_hidden_states=True, use_cache=False)
        index = last_non_padding_index(encoded["attention_mask"][0].detach().cpu().numpy())
        return np.stack(
            [hidden[0, index, :].float().cpu().numpy() for hidden in output.hidden_states], axis=0
        )


def _load_runtime(config: Exp6Config) -> HiddenStateExtractor:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    ).to(config.device).eval()
    return HiddenStateExtractor(tokenizer=tokenizer, model=model, config=config, torch_module=torch)


def load_gsm8k_questions(config: Exp6Config) -> list[str]:
    """Load the reference study's GSM8K split (1319 examples for ``test``)."""
    from datasets import load_dataset

    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split)
    questions = [str(row[config.question_field]) for row in dataset]
    if config.limit is not None:
        questions = questions[: config.limit]
    return questions


def run_exp6(
    config: Exp6Config,
    *,
    questions: Sequence[str] | None = None,
    extractor: HiddenStateExtractor | None = None,
    progress: Callable[[Iterable[Any]], Iterable[Any]] = lambda values: values,
) -> tuple[list[dict[str, Any]], list[dict[str, float | int]]]:
    """Run exp6 and return detailed records and per-layer summary statistics."""
    config.validate()
    active_questions = list(questions) if questions is not None else load_gsm8k_questions(config)
    if config.limit is not None:
        active_questions = active_questions[: config.limit]
    active_extractor = extractor if extractor is not None else _load_runtime(config)
    records: list[dict[str, Any]] = []
    for example_index, question in enumerate(progress(active_questions)):
        clean_states = active_extractor.states(question)
        for level in config.noise_levels:
            noisy_question = inject_variation_selector_noise(
                question,
                noise_level=level,
                seed=stable_example_seed(
                    root_seed=config.seed, example_index=example_index, noise_level=level
                ),
            )
            noisy_states = active_extractor.states(noisy_question)
            similarities = cosine_per_layer(clean_states, noisy_states)
            records.append(
                {
                    "schema_version": "ucc-inj-exp6/v1",
                    "example_index": example_index,
                    "noise_level": level,
                    "cosine_similarity": [float(value) for value in similarities],
                }
            )
    return records, summarise_layer_cosines(records)


def write_exp6_results(
    *,
    output_dir: Path,
    config: Exp6Config,
    records: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, float | int]],
) -> None:
    """Write self-describing JSONL/JSON results without replacing prior runs."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "layer_summary.json").write_text(
        json.dumps(list(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

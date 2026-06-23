"""Perplexity ベンチマーク (Wikitext-2, C4)。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class _PerplexityBase(Benchmark):
    eval_mode = EvalMode.PERPLEXITY
    num_few_shot = 0

    def _format_single(self, example: BenchmarkExample, include_answer: bool) -> str:
        return example.question

    def format_choices(self, example: BenchmarkExample) -> list[str]:
        return []


class Wikitext2Benchmark(_PerplexityBase):
    name = "wikitext2"

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-v1", split="test", trust_remote_code=False
        )
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if not text:
                continue
            examples.append(
                BenchmarkExample(
                    id=f"wikitext2_{i}",
                    question=text,
                    choices=[],
                    answer="",
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


class C4Benchmark(_PerplexityBase):
    name = "c4"

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/c4", "en", split="validation", streaming=True, trust_remote_code=False
        )
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if not text:
                continue
            examples.append(
                BenchmarkExample(
                    id=f"c4_{i}",
                    question=text,
                    choices=[],
                    answer="",
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


BENCHMARKS[Wikitext2Benchmark.name] = Wikitext2Benchmark
BENCHMARKS[C4Benchmark.name] = C4Benchmark

"""Github Typo Corpus ベンチマーク。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class GithubTypoBenchmark(Benchmark):
    name = "github_typo"
    eval_mode = EvalMode.PERPLEXITY
    num_few_shot = 0

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset(
            "chirunder/github_typo_corrections", split="train", trust_remote_code=False
        )
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            text = row["text"].strip()
            correction = row["correction"].strip()
            if not text or not correction:
                continue
            examples.append(
                BenchmarkExample(
                    id=f"github_typo_{i}",
                    question=text,
                    choices=[],
                    answer="",
                    metadata={"correction": correction},
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples

    def _format_single(self, example: BenchmarkExample, include_answer: bool) -> str:
        return example.question

    def format_choices(self, example: BenchmarkExample) -> list[str]:
        return []


BENCHMARKS[GithubTypoBenchmark.name] = GithubTypoBenchmark

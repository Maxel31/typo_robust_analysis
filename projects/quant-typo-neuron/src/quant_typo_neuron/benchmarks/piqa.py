"""PIQA ベンチマーク。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class PIQABenchmark(Benchmark):
    name = "piqa"
    eval_mode = EvalMode.LOG_LIKELIHOOD
    num_few_shot = 0

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("gimmaru/piqa", split="validation")
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    id=f"piqa_{i}",
                    question=row["goal"],
                    choices=[row["sol1"], row["sol2"]],
                    answer=int(row["label"]),
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


BENCHMARKS[PIQABenchmark.name] = PIQABenchmark

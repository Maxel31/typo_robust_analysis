"""MMLU ベンチマーク。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class MMLUBenchmark(Benchmark):
    name = "mmlu"
    eval_mode = EvalMode.LOG_LIKELIHOOD
    num_few_shot = 5

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=False)
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    id=f"mmlu_{i}",
                    question=row["question"],
                    choices=row["choices"],
                    answer=int(row["answer"]),
                    metadata={"subject": row.get("subject", "")},
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


BENCHMARKS[MMLUBenchmark.name] = MMLUBenchmark

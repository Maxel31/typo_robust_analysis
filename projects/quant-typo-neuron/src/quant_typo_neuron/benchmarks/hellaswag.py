"""HellaSwag ベンチマーク。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class HellaSwagBenchmark(Benchmark):
    name = "hellaswag"
    eval_mode = EvalMode.LOG_LIKELIHOOD
    num_few_shot = 10

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
        examples: list[BenchmarkExample] = []
        for row in ds:
            ctx = row["ctx"]
            if row.get("activity_label"):
                ctx = f"{row['activity_label']}: {ctx}"
            examples.append(
                BenchmarkExample(
                    id=str(row["ind"]),
                    question=ctx,
                    choices=row["endings"],
                    answer=int(row["label"]),
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


BENCHMARKS[HellaSwagBenchmark.name] = HellaSwagBenchmark

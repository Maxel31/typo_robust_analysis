"""ARC-Easy / ARC-Challenge ベンチマーク。"""

from __future__ import annotations

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class _ARCBase(Benchmark):
    eval_mode = EvalMode.LOG_LIKELIHOOD
    _subset: str

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", self._subset, split="test", trust_remote_code=False)
        examples: list[BenchmarkExample] = []
        for row in ds:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            answer_idx = labels.index(row["answerKey"]) if row["answerKey"] in labels else 0
            examples.append(
                BenchmarkExample(
                    id=row["id"],
                    question=row["question"],
                    choices=texts,
                    answer=answer_idx,
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


class ARCEasyBenchmark(_ARCBase):
    name = "arc_easy"
    num_few_shot = 0
    _subset = "ARC-Easy"


class ARCChallengeBenchmark(_ARCBase):
    name = "arc_challenge"
    num_few_shot = 25
    _subset = "ARC-Challenge"


BENCHMARKS[ARCEasyBenchmark.name] = ARCEasyBenchmark
BENCHMARKS[ARCChallengeBenchmark.name] = ARCChallengeBenchmark

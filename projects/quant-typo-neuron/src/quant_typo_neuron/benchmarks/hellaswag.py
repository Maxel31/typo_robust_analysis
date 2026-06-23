"""HellaSwag ベンチマーク。"""

from __future__ import annotations

import re

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


def _preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


class HellaSwagBenchmark(Benchmark):
    name = "hellaswag"
    eval_mode = EvalMode.LOG_LIKELIHOOD
    num_few_shot = 10

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
        examples: list[BenchmarkExample] = []
        for row in ds:
            ctx = _preprocess(row["ctx"])
            if row.get("activity_label"):
                ctx = f"{row['activity_label']}: {ctx}"
            endings = [_preprocess(e) for e in row["endings"]]
            examples.append(
                BenchmarkExample(
                    id=str(row["ind"]),
                    question=ctx,
                    choices=endings,
                    answer=int(row["label"]),
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples


    def _format_scoring_single(
        self, example: BenchmarkExample, include_answer: bool
    ) -> str:
        text = example.question
        if include_answer and example.choices:
            answer_text = example.choices[int(example.answer)]
            text += f" {answer_text}"
        return text


BENCHMARKS[HellaSwagBenchmark.name] = HellaSwagBenchmark

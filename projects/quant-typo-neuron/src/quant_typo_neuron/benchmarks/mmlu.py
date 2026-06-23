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


    def _format_scoring_single(
        self, example: BenchmarkExample, include_answer: bool
    ) -> str:
        subject = example.metadata.get("subject", "").replace("_", " ")
        header = (
            f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            if subject
            else ""
        )
        lines = [example.question]
        for i, choice in enumerate(example.choices):
            label = chr(ord("A") + i)
            lines.append(f"{label}. {choice}")
        lines.append("Answer:")
        text = header + "\n".join(lines)
        if include_answer:
            answer_label = chr(ord("A") + int(example.answer))
            text += f" {answer_label}"
        return text

    def format_scoring_choices(self, example: BenchmarkExample) -> list[str]:
        return [f" {chr(ord('A') + i)}" for i in range(len(example.choices))]


BENCHMARKS[MMLUBenchmark.name] = MMLUBenchmark

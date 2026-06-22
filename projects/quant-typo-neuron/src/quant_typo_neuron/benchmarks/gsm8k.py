"""GSM8K ベンチマーク。"""

from __future__ import annotations

import re

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    EvalMode,
)


class GSM8KBenchmark(Benchmark):
    name = "gsm8k"
    eval_mode = EvalMode.GENERATION
    num_few_shot = 5

    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=False)
        examples: list[BenchmarkExample] = []
        for i, row in enumerate(ds):
            answer_text = row["answer"]
            match = re.search(r"####\s*(.+)$", answer_text)
            final_answer = match.group(1).strip() if match else answer_text.strip()
            examples.append(
                BenchmarkExample(
                    id=f"gsm8k_{i}",
                    question=row["question"],
                    choices=[],
                    answer=final_answer,
                    metadata={"full_answer": answer_text},
                )
            )
            if max_samples and len(examples) >= max_samples:
                break
        return examples

    def _format_single(self, example: BenchmarkExample, include_answer: bool) -> str:
        lines = [f"Question: {example.question}"]
        if include_answer:
            lines.append(f"Answer: {example.answer}")
        return "\n".join(lines)

    def extract_answer(self, output: str) -> str:
        match = re.search(r"####\s*(.+?)(?:\s*$)", output.strip())
        if match:
            return match.group(1).strip()
        nums = re.findall(r"[\d,]+\.?\d*", output)
        return nums[-1].replace(",", "") if nums else output.strip()


BENCHMARKS[GSM8KBenchmark.name] = GSM8KBenchmark

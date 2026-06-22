"""ベンチマーク共通の抽象インターフェースとデータ型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto


class EvalMode(Enum):
    LOG_LIKELIHOOD = auto()
    GENERATION = auto()
    PERPLEXITY = auto()


@dataclass
class BenchmarkExample:
    id: str
    question: str
    choices: list[str]
    answer: int | str
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    example_id: str
    question_text: str
    typo_annotations: list
    model_output: str | float
    predicted: int | str
    correct: bool
    token_logprobs: list[float] | None = None


class Benchmark(ABC):
    name: str
    eval_mode: EvalMode
    num_few_shot: int

    @abstractmethod
    def load(self, max_samples: int | None = None) -> list[BenchmarkExample]:
        ...

    def format_prompt(
        self, example: BenchmarkExample, few_shot_examples: list[BenchmarkExample]
    ) -> str:
        parts: list[str] = []
        for fs in few_shot_examples:
            parts.append(self._format_single(fs, include_answer=True))
        parts.append(self._format_single(example, include_answer=False))
        return "\n\n".join(parts)

    def _format_single(self, example: BenchmarkExample, include_answer: bool) -> str:
        lines = [example.question]
        for i, choice in enumerate(example.choices):
            label = chr(ord("A") + i)
            lines.append(f"{label}. {choice}")
        if include_answer and example.choices:
            answer_label = chr(ord("A") + int(example.answer))
            lines.append(f"Answer: {answer_label}")
        return "\n".join(lines)

    def format_choices(self, example: BenchmarkExample) -> list[str]:
        return list(example.choices)

    def extract_answer(self, output: str) -> int | str:
        return output.strip()

    def score(self, predicted: int | str, gold: int | str) -> bool:
        return predicted == gold


BENCHMARKS: dict[str, type[Benchmark]] = {}

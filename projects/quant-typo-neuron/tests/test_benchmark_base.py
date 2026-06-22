"""ベンチマーク基底クラスのテスト。"""

from quant_typo_neuron.benchmarks.base import (
    BENCHMARKS,
    Benchmark,
    BenchmarkExample,
    BenchmarkResult,
    EvalMode,
)


def test_eval_mode_has_expected_members():
    assert hasattr(EvalMode, "LOG_LIKELIHOOD")
    assert hasattr(EvalMode, "GENERATION")
    assert hasattr(EvalMode, "PERPLEXITY")


def test_benchmark_example_fields():
    ex = BenchmarkExample(id="q1", question="What?", choices=["A", "B"], answer=0)
    assert ex.id == "q1"
    assert ex.metadata == {}


def test_benchmark_example_empty_choices():
    ex = BenchmarkExample(id="q2", question="Compute.", choices=[], answer="42")
    assert ex.choices == []
    assert ex.answer == "42"


def test_benchmark_result_fields():
    r = BenchmarkResult(
        example_id="q1",
        question_text="What?",
        typo_annotations=[],
        model_output="A",
        predicted=0,
        correct=True,
    )
    assert r.token_logprobs is None
    assert r.correct is True


class _DummyBenchmark(Benchmark):
    name = "dummy"
    eval_mode = EvalMode.LOG_LIKELIHOOD
    num_few_shot = 0

    def load(self, max_samples=None):
        return [BenchmarkExample(id="1", question="Q?", choices=["A"], answer=0)]


def test_benchmark_abc_can_be_subclassed():
    b = _DummyBenchmark()
    examples = b.load()
    assert len(examples) == 1
    assert examples[0].id == "1"


def test_benchmark_format_prompt():
    b = _DummyBenchmark()
    ex = BenchmarkExample(id="1", question="What color?", choices=["red", "blue"], answer=0)
    prompt = b.format_prompt(ex, [])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_benchmark_format_choices():
    b = _DummyBenchmark()
    ex = BenchmarkExample(id="1", question="Q?", choices=["A", "B", "C"], answer=1)
    choices = b.format_choices(ex)
    assert isinstance(choices, list)
    assert len(choices) == 3


def test_benchmark_score():
    b = _DummyBenchmark()
    assert b.score(0, 0) is True
    assert b.score(1, 0) is False


def test_benchmark_registry_is_dict():
    assert isinstance(BENCHMARKS, dict)

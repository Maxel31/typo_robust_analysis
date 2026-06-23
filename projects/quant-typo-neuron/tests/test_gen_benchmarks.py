"""生成/perplexity ベンチマークローダーのテスト。"""

import pytest

from quant_typo_neuron.benchmarks.base import BENCHMARKS, BenchmarkExample, EvalMode
from quant_typo_neuron.benchmarks.gsm8k import GSM8KBenchmark
from quant_typo_neuron.benchmarks.perplexity import Wikitext2Benchmark, C4Benchmark
from quant_typo_neuron.benchmarks.github_typo import GithubTypoBenchmark


class TestGSM8K:
    def test_eval_mode(self):
        assert GSM8KBenchmark.eval_mode == EvalMode.GENERATION

    def test_registered(self):
        assert GSM8KBenchmark.name in BENCHMARKS

    def test_load(self):
        b = GSM8KBenchmark()
        examples = b.load(max_samples=3)
        assert len(examples) <= 3
        for ex in examples:
            assert isinstance(ex, BenchmarkExample)
            assert isinstance(ex.answer, str)

    def test_extract_answer_hash_format(self):
        b = GSM8KBenchmark()
        assert b.extract_answer("some reasoning #### 42") == "42"

    def test_extract_answer_strips(self):
        b = GSM8KBenchmark()
        assert b.extract_answer("  #### 7.5  ") == "7.5"

    def test_format_prompt_is_string(self):
        b = GSM8KBenchmark()
        examples = b.load(max_samples=1)
        prompt = b.format_prompt(examples[0], [])
        assert isinstance(prompt, str)
        assert len(prompt) > 0


@pytest.mark.parametrize("cls", [Wikitext2Benchmark, C4Benchmark])
class TestPerplexityBenchmarks:
    def test_eval_mode(self, cls):
        assert cls.eval_mode == EvalMode.PERPLEXITY

    def test_registered(self, cls):
        assert cls.name in BENCHMARKS

    def test_load(self, cls):
        b = cls()
        examples = b.load(max_samples=3)
        assert len(examples) >= 1
        for ex in examples:
            assert isinstance(ex, BenchmarkExample)
            assert len(ex.question) > 0

    def test_choices_empty(self, cls):
        b = cls()
        examples = b.load(max_samples=1)
        assert examples[0].choices == []


class TestGithubTypo:
    def test_eval_mode(self):
        assert GithubTypoBenchmark.eval_mode == EvalMode.PERPLEXITY

    def test_registered(self):
        assert "github_typo" in BENCHMARKS

    def test_load(self):
        b = GithubTypoBenchmark()
        examples = b.load(max_samples=5)
        assert len(examples) <= 5
        for ex in examples:
            assert isinstance(ex, BenchmarkExample)

    def test_has_correction_metadata(self):
        b = GithubTypoBenchmark()
        examples = b.load(max_samples=3)
        for ex in examples:
            assert "correction" in ex.metadata

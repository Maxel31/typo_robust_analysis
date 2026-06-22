"""多肢選択ベンチマークローダーのテスト。"""

import pytest

from quant_typo_neuron.benchmarks.base import BENCHMARKS, BenchmarkExample, EvalMode
from quant_typo_neuron.benchmarks.arc import ARCChallengeBenchmark, ARCEasyBenchmark
from quant_typo_neuron.benchmarks.hellaswag import HellaSwagBenchmark
from quant_typo_neuron.benchmarks.mmlu import MMLUBenchmark
from quant_typo_neuron.benchmarks.piqa import PIQABenchmark

ALL_MC = [ARCEasyBenchmark, ARCChallengeBenchmark, HellaSwagBenchmark, MMLUBenchmark, PIQABenchmark]


@pytest.mark.parametrize("cls", ALL_MC)
def test_mc_benchmark_eval_mode(cls):
    assert cls.eval_mode == EvalMode.LOG_LIKELIHOOD


@pytest.mark.parametrize("cls", ALL_MC)
def test_mc_benchmark_registered(cls):
    b = cls()
    assert b.name in BENCHMARKS


@pytest.mark.parametrize("cls", ALL_MC)
def test_mc_benchmark_load_returns_examples(cls):
    b = cls()
    examples = b.load(max_samples=5)
    assert 0 < len(examples) <= 5
    for ex in examples:
        assert isinstance(ex, BenchmarkExample)
        assert len(ex.choices) >= 2


@pytest.mark.parametrize("cls", ALL_MC)
def test_mc_benchmark_format_prompt(cls):
    b = cls()
    examples = b.load(max_samples=1)
    prompt = b.format_prompt(examples[0], [])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.parametrize("cls", ALL_MC)
def test_mc_benchmark_format_choices(cls):
    b = cls()
    examples = b.load(max_samples=1)
    choices = b.format_choices(examples[0])
    assert isinstance(choices, list)
    assert all(isinstance(c, str) for c in choices)
    assert len(choices) == len(examples[0].choices)


def test_arc_easy_few_shot_count():
    assert ARCEasyBenchmark.num_few_shot == 0


def test_arc_challenge_few_shot_count():
    assert ARCChallengeBenchmark.num_few_shot == 25


def test_hellaswag_few_shot_count():
    assert HellaSwagBenchmark.num_few_shot == 10


def test_mmlu_few_shot_count():
    assert MMLUBenchmark.num_few_shot == 5


def test_piqa_few_shot_count():
    assert PIQABenchmark.num_few_shot == 0

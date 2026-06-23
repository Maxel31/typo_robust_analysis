"""実験パイプライン。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from typo_utils.data.typo import inject_typos_by_count
from typo_utils.models.vllm_runner import VLLMRunner
from typo_utils.seed import set_seed

from quant_typo_neuron.benchmarks import BENCHMARKS
from quant_typo_neuron.benchmarks.base import BenchmarkExample, BenchmarkResult, EvalMode
from quant_typo_neuron.few_shot import FewShotCache
from quant_typo_neuron.output import make_run_id, save_metrics, save_predictions


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    set_seed(cfg.seed)

    bench_cls = BENCHMARKS[cfg.benchmark.name]
    bench = bench_cls()
    examples = bench.load(max_samples=cfg.benchmark.get("max_samples"))

    few_shot_cache = FewShotCache(cache_dir=cfg.get("few_shot_dir", "data/few_shot"))
    few_shot_examples = few_shot_cache.get_or_create(
        benchmark=bench.name,
        num_shots=bench.num_few_shot,
        seed=cfg.seed,
        pool=examples,
    )

    typo_type = cfg.typo.get("type", "clean")
    num_typos = cfg.typo.get("num_typos", 0)

    if typo_type != "clean" and num_typos > 0:
        examples = _apply_typos(examples, typo_type, num_typos, cfg.seed)

    gpu_ids = cfg.get("gpu_ids")
    if gpu_ids is not None:
        gpu_ids = list(gpu_ids)

    with VLLMRunner(
        cfg.model.name,
        tensor_parallel_size=cfg.model.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.model.get("gpu_memory_utilization", 0.9),
        gpu_ids=gpu_ids,
    ) as runner:
        if bench.eval_mode == EvalMode.LOG_LIKELIHOOD:
            results = _eval_log_likelihood(runner, bench, examples, few_shot_examples)
        elif bench.eval_mode == EvalMode.GENERATION:
            results = _eval_generation(
                runner, bench, examples, few_shot_examples,
                max_tokens=cfg.get("max_tokens", 256),
            )
        elif bench.eval_mode == EvalMode.PERPLEXITY:
            results = _eval_perplexity(runner, bench, examples)
        else:
            raise ValueError(f"Unknown eval mode: {bench.eval_mode}")

    metrics = _compute_metrics(results, bench.eval_mode)
    calibration = cfg.model.get("calibration", "none")
    metrics.update({
        "model": cfg.model.name,
        "quantization_method": cfg.model.get("quantization_method", "none"),
        "bits": cfg.model.get("bits", 16),
        "calibration": calibration,
        "benchmark": bench.name,
        "typo_type": typo_type,
        "num_typos": num_typos,
    })

    method = cfg.model.get("quantization_method", "none")
    bits = cfg.model.get("bits", 16)
    quantization = f"{method}_w{bits}"
    run_id = make_run_id(
        model=cfg.model.name,
        quantization=quantization,
        benchmark=bench.name,
        typo_type=typo_type,
        num_typos=num_typos,
        calibration=calibration,
    )
    results_dir = Path(cfg.get("results_root", "results")) / run_id
    save_predictions(results, results_dir / "predictions.jsonl")
    save_metrics(metrics, results_dir / "metrics.json")

    return metrics


def _apply_typos(
    examples: list[BenchmarkExample],
    typo_type: str,
    num_typos: int,
    seed: int,
) -> list[BenchmarkExample]:
    modified = []
    for i, ex in enumerate(examples):
        text, annotations = inject_typos_by_count(
            ex.question, num_typos=num_typos, typo_type=typo_type, seed=seed + i
        )
        modified.append(BenchmarkExample(
            id=ex.id,
            question=text,
            choices=ex.choices,
            answer=ex.answer,
            metadata={**ex.metadata, "typo_annotations": [a.__dict__ for a in annotations]},
        ))
    return modified


def _eval_log_likelihood(runner, bench, examples, few_shot_examples):
    prompts = []
    all_choices = []
    for ex in examples:
        prompt = bench.format_scoring_prompt(ex, few_shot_examples)
        choices = bench.format_scoring_choices(ex)
        prompts.append(prompt)
        all_choices.append(choices)

    scores = runner.score_log_likelihood(prompts, all_choices, normalize=True)

    results = []
    for ex, choice_scores in zip(examples, scores):
        predicted = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        correct = bench.score(predicted, ex.answer)
        results.append(BenchmarkResult(
            example_id=ex.id,
            question_text=ex.question,
            typo_annotations=ex.metadata.get("typo_annotations", []),
            model_output=str(choice_scores),
            predicted=predicted,
            correct=correct,
            token_logprobs=choice_scores,
        ))
    return results


def _eval_generation(runner, bench, examples, few_shot_examples, max_tokens=256):
    prompts = [bench.format_prompt(ex, few_shot_examples) for ex in examples]
    outputs = runner.generate(prompts, max_tokens=max_tokens)

    results = []
    for ex, out in zip(examples, outputs):
        predicted = bench.extract_answer(out.text)
        correct = bench.score(predicted, ex.answer)
        results.append(BenchmarkResult(
            example_id=ex.id,
            question_text=ex.question,
            typo_annotations=ex.metadata.get("typo_annotations", []),
            model_output=out.text,
            predicted=predicted,
            correct=correct,
            token_logprobs=out.token_logprobs,
        ))
    return results


def _eval_perplexity(runner, bench, examples):
    texts = [ex.question for ex in examples]
    perplexities = runner.compute_perplexity(texts)

    results = []
    for ex, ppl in zip(examples, perplexities):
        results.append(BenchmarkResult(
            example_id=ex.id,
            question_text=ex.question,
            typo_annotations=ex.metadata.get("typo_annotations", []),
            model_output=ppl,
            predicted=ppl,
            correct=True,
        ))
    return results


def _compute_metrics(results: list[BenchmarkResult], eval_mode: EvalMode) -> dict[str, Any]:
    if eval_mode == EvalMode.PERPLEXITY:
        ppls = [r.model_output for r in results if isinstance(r.model_output, float)]
        return {
            "mean_perplexity": sum(ppls) / len(ppls) if ppls else float("inf"),
            "num_examples": len(results),
        }
    else:
        correct = [r.correct for r in results]
        n = len(correct)
        return {
            "accuracy": sum(correct) / n if n else 0.0,
            "num_examples": len(results),
            "num_correct": sum(correct),
        }

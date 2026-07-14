#!/usr/bin/env python3
"""Rebuttal 用: AttnLRP を省略した生成専用 Phase 3 ランナー.

実験④ (Matched-Random) は accuracy drop の比較のみが目的で CoT 帰属を必要としない。
既存 run_inference.py は AttnLRP 計算が必須 (スキップ不可) のため、
生成 (greedy) + 回答抽出 + results/summary 出力のみを行う軽量ランナーを提供する。

プロンプト構築・生成パラメータ・回答抽出・出力スキーマは run_inference.py と同一
(importance_scores/ が無い点のみ異なる)。

使用例:
  uv run --no-sync python scripts/rebuttal/run_generation_only.py \
    --model google/gemma-3-4b-it --benchmark mmlu \
    --perturbed_data datasets/rebuttal/gemma-3-4b-it_mmlu_k4_matched_random/perturbed_dataset.json \
    --batch_size 4 --gpu_id 3 --output_dir outputs/rebuttal/perturbed
"""

import argparse
import gc
import json
import logging
import time
from datetime import datetime
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generation_only")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成専用 Phase 3 (LRPなし)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--perturbed_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/rebuttal/perturbed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import torch

    from typo_cot.data.loader import Sample
    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper
    from typo_cot.perturbation.dataset import PerturbedDataset

    perturbed_dataset = PerturbedDataset.load(Path(args.perturbed_data))
    metadata = perturbed_dataset.metadata
    num_perturbations = metadata.get("num_perturbations", "unknown")
    perturbation_mode = metadata.get("perturbation_mode", "importance")

    model_short = args.model.split("/")[-1]
    experiment_name = (
        f"{model_short}_{args.benchmark}_k{num_perturbations}_{perturbation_mode}"
    )
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリ: {output_dir}")

    # config.json (run_inference.save_config 互換 + lrp_skipped フラグ)
    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "num_samples_per_subset": None,
        "batch_size": args.batch_size,
        "gpu_id": args.gpu_id,
        "top_k": None,
        "max_new_tokens": args.max_new_tokens,
        "seed": 42,
        "heatmap_interval": 0,
        "timestamp": datetime.now().isoformat(),
        "perturbed_data": str(args.perturbed_data),
        "phase": "phase3",
        "perturbed_metadata": metadata,
        "lrp_skipped": True,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # サンプル構築 (run_inference.py:378-394 と同一)
    samples = []
    ps_by_id = {}
    for ps in perturbed_dataset.samples:
        samples.append(
            Sample(
                sample_id=ps.sample_id,
                question=ps.perturbed_question,
                correct_answer=ps.correct_answer,
                subset=ps.subset,
                choices=ps.perturbed_choices,
                context=ps.context,
            )
        )
        ps_by_id[ps.sample_id] = ps
    if args.limit:
        samples = samples[: args.limit]
    logger.info(f"サンプル数: {len(samples)}")

    logger.info(f"モデルをロード: {args.model} (GPU {args.gpu_id})")
    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=False
    )
    template = create_prompt_template(args.benchmark)
    extractor = create_extractor(args.benchmark)

    results_list = []
    subset_stats: dict[str, dict[str, int]] = {}
    correct_count = 0
    processed_count = 0

    batch_size = args.batch_size
    num_batches = (len(samples) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(num_batches):
        batch_samples = samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        batch_prompts = []
        for sample in batch_samples:
            if args.benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
                pr = template.generate(
                    question=sample.question, choices=sample.choices,
                    subject=sample.subset,
                )
            elif args.benchmark in ["bbh", "math", "strategy_qa"]:
                pr = template.generate(question=sample.question, subject=sample.subset)
            else:
                pr = template.generate(question=sample.question)
            batch_prompts.append(pr.get_full_prompt())

        try:
            gen_results = wrapper.generate_batch(
                batch_prompts, max_new_tokens=args.max_new_tokens, temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"バッチ {batch_idx} 失敗、個別処理にフォールバック: {exc}")
            gen_results = []
            for p in batch_prompts:
                try:
                    gen_results.append(
                        wrapper.generate(p, max_new_tokens=args.max_new_tokens,
                                         temperature=0.0)
                    )
                except Exception as exc2:  # noqa: BLE001
                    logger.error(f"個別生成も失敗: {exc2}")
                    gen_results.append(None)

        for sample, gen in zip(batch_samples, gen_results, strict=True):
            if gen is None:
                continue
            extraction = extractor.extract(gen.generated_text)
            is_correct = extractor.is_correct(
                extraction.extracted_answer, sample.correct_answer
            )
            if is_correct:
                correct_count += 1
            processed_count += 1

            subset = sample.subset or "default"
            st = subset_stats.setdefault(subset, {"correct": 0, "total": 0})
            st["total"] += 1
            if is_correct:
                st["correct"] += 1

            ps = ps_by_id.get(sample.sample_id)
            entry = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "correct_answer": sample.correct_answer,
                "choices": sample.choices,
                "context": sample.context,
                "generated_text": gen.generated_text,
                "extracted_answer": extraction.extracted_answer,
                "is_correct": is_correct,
                "subset": subset,
                "question_top_k_words": [],
                "cot_top_k_words": [],
            }
            if ps is not None:
                entry["original_question"] = ps.original_question
                entry["perturbed_tokens"] = [
                    {
                        "token_index": pt.token_index,
                        "original_token": pt.original_token,
                        "perturbed_token": pt.perturbed_token,
                        "importance_score": pt.importance_score,
                        "perturbation_type": pt.perturbation_type,
                    }
                    for pt in ps.perturbed_tokens
                ]
            results_list.append(entry)

        if (batch_idx + 1) % 25 == 0:
            done = min((batch_idx + 1) * batch_size, len(samples))
            rate = (time.time() - t0) / done
            eta_min = rate * (len(samples) - done) / 60
            acc = correct_count / processed_count if processed_count else 0
            logger.info(
                f"{done}/{len(samples)} ({rate:.2f}s/sample, 残り約{eta_min:.0f}分, "
                f"正答率 {acc:.1%})"
            )
            tmp = output_dir / "results.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results_list, f, ensure_ascii=False)
            tmp.replace(output_dir / "results.json")

        del gen_results, batch_prompts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    summary = {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "num_samples_per_subset": None,
            "batch_size": args.batch_size,
            "total_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "overall_metrics": {
            "accuracy": correct_count / processed_count if processed_count else 0,
            "total_correct": correct_count,
            "total_samples": processed_count,
        },
        "per_subset_metrics": {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in subset_stats.items()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"完了: accuracy={summary['overall_metrics']['accuracy']:.4f} "
        f"({correct_count}/{processed_count}), {output_dir}"
    )


if __name__ == "__main__":
    main()

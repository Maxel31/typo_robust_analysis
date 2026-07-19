#!/usr/bin/env python3
"""実験10④: 自然typo摂動データセットの生成側推論 (LRPなし・生成のみ).

A/B 比較の評価軸は flip 率・精度低下 (生成テキストのみで計算可能) なので、
AttnLRP の重要度計算を省略した生成専用スクリプトとして run_inference.py から
分離する (他エージェントとの衝突回避のため既存スクリプトは変更しない)。

- モデルは run_inference.py と同じく LXT ラップ込みでロードし、
  greedy (temperature=0.0) 生成 → 回答抽出 → is_correct 判定のみ行う。
- 出力スキーマはアーカイブの perturbed results.json 互換
  (LRP 由来の question_top_k_words / cot_top_k_words は空リスト)。
- --start/--end のシャード実行と --merge を run_inference.py と同じ規約で提供。

使い方 (GPU ヘルパー経由):
  bash tmp/gpu-locks/run_with_gpu.sh uv run --no-sync python \
    scripts/exp10_natural_typo/run_generation.py \
    --model google/gemma-3-4b-it --benchmark gsm8k \
    --perturbed_data datasets/perturbed/gemma-3-4b-it_gsm8k_k4_natural_with_choices/perturbed_dataset.json \
    --batch_size 8 --start 0 --end 330
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from typo_cot.data.loader import Sample
from typo_cot.evaluation.extractor import create_extractor
from typo_cot.models.prompts import create_prompt_template
from typo_cot.models.wrapper import create_model_wrapper
from typo_cot.perturbation.dataset import PerturbedDataset
from typo_cot.sharding import (
    build_summary_from_results,
    load_shard_rows,
    merge_shard_results,
    shard_results_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True, choices=["gsm8k", "mmlu"])
    parser.add_argument("--perturbed_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="./outputs/perturbed")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def build_prompt(sample: Sample, template, benchmark: str) -> str:
    if benchmark == "mmlu":
        result = template.generate(
            question=sample.question,
            choices=sample.choices,
            subject=sample.subset,
        )
    else:
        result = template.generate(question=sample.question)
    return result.get_full_prompt()


def main() -> None:
    args = parse_args()

    model_short = args.model.split("/")[-1]
    perturbed_path = Path(args.perturbed_data)
    if not perturbed_path.exists():
        logger.error(f"摂動データセットが見つかりません: {perturbed_path}")
        sys.exit(1)

    dataset = PerturbedDataset.load(perturbed_path)
    metadata = dataset.metadata
    mode = metadata.get("perturbation_mode", "natural")
    k = metadata.get("num_perturbations", 4)
    experiment_name = f"{model_short}_{args.benchmark}_k{k}_{mode}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリ: {output_dir}")

    # シャード統合モード
    if args.merge:
        merged, covered = merge_shard_results(output_dir)
        if not merged:
            logger.error(f"統合対象のシャードが見つかりません: {output_dir / 'shards'}")
            sys.exit(1)
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        summary = build_summary_from_results(
            model=args.model,
            benchmark=args.benchmark,
            results=merged,
            batch_size=args.batch_size,
            merged_shards=covered,
        )
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(
            f"シャード統合完了: {len(merged)}件, "
            f"正答率={summary['overall_metrics']['accuracy']:.2%}"
        )
        return

    # 設定を保存
    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "perturbed_data": str(perturbed_path),
        "phase": "phase3_generation_only",
        "note": "実験10④ A/B 検証 B側 (自然typo分布)。AttnLRP なしの生成専用実行",
        "perturbed_metadata": metadata,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # サンプル構築 (run_inference.py の Phase 3 と同じ規約:
    # perturbed_choices=None なら選択肢は perturbed_question に含まれている)
    samples = [
        Sample(
            sample_id=ps.sample_id,
            question=ps.perturbed_question,
            correct_answer=ps.correct_answer,
            subset=ps.subset,
            choices=ps.perturbed_choices,
            context=ps.context,
        )
        for ps in dataset.samples
    ]
    total = len(samples)
    end = min(args.end, total) if args.end is not None else total
    if args.start >= end:
        logger.error(f"不正なシャード範囲: [{args.start}, {end})")
        sys.exit(1)

    shard_path = shard_results_path(output_dir, args.start, end)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_samples = samples[args.start : end]

    # シャード内復帰
    rows = load_shard_rows(shard_path)
    done_ids = {r["sample_id"] for r in rows}
    if done_ids:
        shard_samples = [s for s in shard_samples if s.sample_id not in done_ids]
        logger.info(f"シャード内復帰: {len(done_ids)}件処理済みをスキップ")
    if not shard_samples:
        logger.info(f"シャード [{args.start}, {end}) は完了済み")
        return
    logger.info(f"シャード [{args.start}, {end}) / 全{total}件, 残り{len(shard_samples)}")

    perturbed_by_id = {ps.sample_id: ps for ps in dataset.samples}

    logger.info(f"モデルをロード: {args.model}")
    wrapper = create_model_wrapper(
        model_name=args.model,
        gpu_id=args.gpu_id,
        wrap_for_lxt=True,  # A側 (run_inference.py) と同一のモデルラップで生成
    )
    template = create_prompt_template(args.benchmark)
    extractor = create_extractor(args.benchmark)

    def flush() -> None:
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    bs = args.batch_size
    n_batches = (len(shard_samples) + bs - 1) // bs
    pbar = tqdm(total=len(shard_samples), desc="自然typo生成", unit="sample")
    for bi in range(n_batches):
        batch = shard_samples[bi * bs : (bi + 1) * bs]
        prompts = [build_prompt(s, template, args.benchmark) for s in batch]
        try:
            gen_results = wrapper.generate_batch(
                prompts, max_new_tokens=args.max_new_tokens, temperature=0.0
            )
        except Exception as e:
            logger.warning(f"バッチ {bi} 失敗、個別処理にフォールバック: {e}")
            gen_results = []
            for p in prompts:
                try:
                    gen_results.append(
                        wrapper.generate(p, max_new_tokens=args.max_new_tokens, temperature=0.0)
                    )
                except Exception as e2:
                    logger.error(f"個別生成も失敗: {e2}")
                    gen_results.append(None)

        for sample, gen in zip(batch, gen_results, strict=True):
            if gen is None:
                continue
            extraction = extractor.extract(gen.generated_text)
            is_correct = extractor.is_correct(
                extraction.extracted_answer, sample.correct_answer
            )
            ps = perturbed_by_id[sample.sample_id]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "correct_answer": sample.correct_answer,
                    "choices": ps.choices,
                    "context": sample.context,
                    "generated_text": gen.generated_text,
                    "extracted_answer": extraction.extracted_answer,
                    "is_correct": is_correct,
                    "subset": sample.subset or "default",
                    "question_top_k_words": [],
                    "cot_top_k_words": [],
                    "original_question": ps.original_question,
                    "perturbed_tokens": [
                        {
                            "token_index": pt.token_index,
                            "original_token": pt.original_token,
                            "perturbed_token": pt.perturbed_token,
                            "importance_score": pt.importance_score,
                            "perturbation_type": pt.perturbation_type,
                        }
                        for pt in ps.perturbed_tokens
                    ],
                }
            )
            pbar.update(1)
        flush()

    pbar.close()
    n_correct = sum(1 for r in rows if r.get("is_correct"))
    accuracy = n_correct / len(rows) if rows else 0
    logger.info(f"シャード完了: {len(rows)}件, 正答率={accuracy:.2%}, 保存: {shard_path}")


if __name__ == "__main__":
    main()

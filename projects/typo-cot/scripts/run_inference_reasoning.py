#!/usr/bin/env python3
"""実験10③: reasoning モデル(DeepSeek-R1蒸留系)用の推論スクリプト.

run_inference.py との差分:
- ゼロショット+チャットテンプレート(DeepSeek公式推奨: few-shot は性能劣化)
- CoT は <think>...</think>、最終回答は閉じタグ以降のセクション
- 長大 CoT のため生成専用のバッチ経路(R_C の AttnLRP は非計算=計画書の制約)
- --compute_rq で R_Q(質問→CoT開始の寄与)のみ計算し、
  PerturbedDatasetCreator 互換の importance_scores/*.pt を保存
- シャード分割(--start/--end)+ --merge で results.json/summary.json を統合

出力スキーマはアーカイブ outputs/baseline/<model>_<bench>/ と互換
(results.json に reasoning 拡張フィールドを追加)。

使用例(シャード実行):
  uv run python scripts/run_inference_reasoning.py \
    --benchmark gsm8k --start 0 --end 200 --batch_size 8 --compute_rq
統合:
  uv run python scripts/run_inference_reasoning.py --benchmark gsm8k --merge
"""

import argparse
import gc
import json
import logging
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reasoning_inference")

SUPPORTED_BENCHMARKS = ["gsm8k", "mmlu", "math"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="reasoning モデル (R1蒸留) 推論",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument("--benchmark", type=str, required=True, choices=SUPPORTED_BENCHMARKS)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="MMLU: サブセットごとのサンプル数 (デフォルト100=アーカイブと同一)",
    )
    parser.add_argument("--start", type=int, default=0, help="シャード開始インデックス")
    parser.add_argument("--end", type=int, default=None, help="シャード終了インデックス(exclusive)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Noneの場合ベンチマーク別デフォルト (gsm8k/mmlu: 4096, math: 8192)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/baseline")
    parser.add_argument(
        "--perturbed_data", type=str, default=None, help="摂動データセットのパス(Phase 3)"
    )
    parser.add_argument(
        "--compute_rq",
        action="store_true",
        help="R_Q (質問→CoT開始) のAttnLRPを計算し importance_scores/*.pt を保存",
    )
    parser.add_argument("--merge", action="store_true", help="シャードを統合して summary を出力")
    parser.add_argument("--limit", type=int, default=None, help="スモーク用: 先頭N件のみ")
    return parser.parse_args()


def load_samples(args: argparse.Namespace):
    """ベンチマークサンプルをアーカイブと同一の集合でロードする."""
    from typo_cot.data.loader import Sample, create_loader
    from typo_cot.perturbation.dataset import PerturbedDataset

    if args.perturbed_data:
        ds = PerturbedDataset.load(Path(args.perturbed_data))
        samples = [
            Sample(
                sample_id=ps.sample_id,
                question=ps.perturbed_question,
                choices=ps.perturbed_choices,
                correct_answer=ps.correct_answer,
                subset=ps.subset,
                context=ps.context,
            )
            for ps in ds.samples
        ]
        return samples, ds
    # アーカイブ規約: mmlu=サブセットごと100件(seed42), gsm8k/math=全件
    samples_per_subset = args.num_samples if args.num_samples is not None else 100
    loader = create_loader(
        benchmark=args.benchmark,
        samples_per_subset=samples_per_subset,
        seed=args.seed,
        num_samples=args.num_samples,
    )
    return loader.load(), None


def experiment_dir(args: argparse.Namespace, perturbed_metadata: dict | None) -> Path:
    model_short = args.model.split("/")[-1]
    name = f"{model_short}_{args.benchmark}"
    if perturbed_metadata is not None:
        k = perturbed_metadata.get("num_perturbations", "unknown")
        mode = perturbed_metadata.get("perturbation_mode", "importance")
        name = f"{name}_k{k}_{mode}"
        if args.output_dir == "./outputs/baseline":
            args.output_dir = "./outputs/perturbed"
    return Path(args.output_dir) / name


def build_summary(args, results: list[dict], max_new_tokens: int) -> dict:
    """アーカイブ summary.json 互換のサマリーを構築(reasoning 拡張つき)."""
    n = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    extracted = sum(1 for r in results if r["extracted_answer"])
    think_closed = sum(1 for r in results if r.get("has_think_close"))
    truncated = sum(1 for r in results if r.get("truncated"))

    per_subset: dict[str, dict] = {}
    for r in results:
        s = r.get("subset") or "default"
        st = per_subset.setdefault(s, {"correct": 0, "total": 0})
        st["total"] += 1
        if r["is_correct"]:
            st["correct"] += 1
    per_subset_metrics = {
        s: {
            "accuracy": st["correct"] / st["total"] if st["total"] else 0,
            "correct": st["correct"],
            "total": st["total"],
        }
        for s, st in per_subset.items()
    }

    return {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "num_samples_per_subset": args.num_samples,
            "batch_size": args.batch_size,
            "total_samples": n,
            "timestamp": datetime.now().isoformat(),
        },
        "overall_metrics": {
            "accuracy": correct / n if n else 0,
            "total_correct": correct,
            "total_samples": n,
            "extraction_success_rate": extracted / n if n else 0,
            "think_close_rate": think_closed / n if n else 0,
            "truncation_rate": truncated / n if n else 0,
            "max_new_tokens": max_new_tokens,
        },
        "per_subset_metrics": per_subset_metrics,
    }


def merge_shards(args: argparse.Namespace, output_dir: Path, max_new_tokens: int) -> None:
    """shards/results_*.json を連結して results.json / summary.json を出力する."""
    shard_dir = output_dir / "shards"
    shard_files = sorted(shard_dir.glob("results_*.json"))
    if not shard_files:
        logger.error(f"シャードが見つかりません: {shard_dir}")
        return
    merged: dict[str, dict] = {}
    pat = re.compile(r"results_(\d+)_(\d+)\.json")
    covered: list[tuple[int, int]] = []
    for f in shard_files:
        m = pat.match(f.name)
        if m:
            covered.append((int(m.group(1)), int(m.group(2))))
        with open(f, encoding="utf-8") as fh:
            for row in json.load(fh):
                merged[row["sample_id"]] = row  # 後勝ち(再実行シャード優先)
    results = list(merged.values())
    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    summary = build_summary(args, results, max_new_tokens)
    summary["experiment_info"]["merged_shards"] = [list(c) for c in sorted(covered)]
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    om = summary["overall_metrics"]
    logger.info(
        f"統合完了: n={om['total_samples']}, accuracy={om['accuracy']:.2%}, "
        f"extraction={om['extraction_success_rate']:.2%}, "
        f"think_close={om['think_close_rate']:.2%}, trunc={om['truncation_rate']:.2%}"
    )


def main() -> None:
    args = parse_args()

    from typo_cot.models.reasoning import REASONING_MAX_NEW_TOKENS

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else REASONING_MAX_NEW_TOKENS[args.benchmark]
    )

    perturbed_metadata = None
    perturbed_ds = None
    if args.perturbed_data:
        from typo_cot.perturbation.dataset import PerturbedDataset

        perturbed_ds = PerturbedDataset.load(Path(args.perturbed_data))
        perturbed_metadata = perturbed_ds.metadata

    output_dir = experiment_dir(args, perturbed_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(args, output_dir, max_new_tokens)
        return

    import torch

    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.reasoning import (
        build_full_prompt,
        extract_reasoning_answer,
        split_reasoning_output,
        think_prefix_end,
    )
    from typo_cot.models.wrapper import create_model_wrapper

    torch.manual_seed(args.seed)

    samples, loaded_perturbed = load_samples(args)
    if loaded_perturbed is not None:
        perturbed_ds = loaded_perturbed
    total = len(samples)
    end = min(args.end if args.end is not None else total, total)
    shard_samples = samples[args.start : end]
    if args.limit:
        shard_samples = shard_samples[: args.limit]
        end = args.start + len(shard_samples)
    logger.info(f"シャード [{args.start}, {end}) / 全{total}件, {len(shard_samples)}サンプル")

    # config.json (アーカイブ互換 + reasoning 拡張)
    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "num_samples_per_subset": args.num_samples,
        "batch_size": args.batch_size,
        "gpu_id": "(run_with_gpu.sh)",
        "top_k": None,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "heatmap_interval": 0,
        "timestamp": datetime.now().isoformat(),
        "phase": "phase3" if args.perturbed_data else "phase1",
        "prompt_style": "zero_shot_chat_template",
        "cot_format": "think_tags",
        "rq_computed": bool(args.compute_rq),
        "rc_computed": False,
    }
    if args.perturbed_data:
        config["perturbed_data"] = str(args.perturbed_data)
        config["perturbed_metadata"] = perturbed_metadata
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # モデルロード(compute_rq 時は lxt ラップ = R_Q backward 用)
    wrapper = create_model_wrapper(
        model_name=args.model,
        gpu_id="0",  # run_with_gpu.sh の CUDA_VISIBLE_DEVICES を尊重(外部設定優先)
        wrap_for_lxt=args.compute_rq,
    )
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    extractor = create_extractor(args.benchmark)

    analyzer = None
    if args.compute_rq:
        from typo_cot.lrp.analyzer import create_analyzer

        analyzer = create_analyzer(
            model=model, tokenizer=tokenizer, top_k=None, device=wrapper.device
        )
        (output_dir / "importance_scores").mkdir(exist_ok=True)

    eos_strings = [tokenizer.eos_token or ""]
    if tokenizer.pad_token:
        eos_strings.append(tokenizer.pad_token)

    results: list[dict] = []
    n_batches = (len(shard_samples) + args.batch_size - 1) // args.batch_size
    correct_count = 0

    for bi in range(n_batches):
        batch = shard_samples[bi * args.batch_size : (bi + 1) * args.batch_size]
        prompts = [
            build_full_prompt(tokenizer, args.benchmark, s.question, s.choices)
            for s in batch
        ]

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            [p.full_prompt for p in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # チャットテンプレートが BOS を含むため
        )
        input_ids = inputs["input_ids"].to(wrapper.device)
        attention_mask = inputs["attention_mask"].to(wrapper.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, (sample, fp) in enumerate(zip(batch, prompts, strict=True)):
            gen_ids = output_ids[i, input_len:]
            # 末尾の pad を除去
            gen_list = gen_ids.tolist()
            while gen_list and gen_list[-1] == tokenizer.pad_token_id:
                gen_list.pop()
            num_generated = len(gen_list)
            truncated = num_generated >= max_new_tokens
            generated_text = tokenizer.decode(gen_list, skip_special_tokens=False)
            for e in eos_strings:
                if e:
                    generated_text = generated_text.replace(e, "")

            split = split_reasoning_output(generated_text)
            extraction = extract_reasoning_answer(extractor, split)
            is_correct = extractor.is_correct(
                extraction.extracted_answer, sample.correct_answer
            )
            if is_correct:
                correct_count += 1

            row = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "correct_answer": sample.correct_answer,
                "choices": sample.choices,
                "context": sample.context,
                "generated_text": generated_text,
                "extracted_answer": extraction.extracted_answer,
                "is_correct": is_correct,
                "subset": sample.subset or "default",
                "question_top_k_words": [],
                "cot_top_k_words": [],
                # reasoning 拡張
                "cot_text": split.cot_text,
                "answer_text": split.answer_text,
                "has_think_close": split.has_think_close,
                "truncated": truncated,
                "num_generated_tokens": num_generated,
                "extraction_method": extraction.extraction_method,
            }
            if perturbed_ds is not None:
                orig = next(
                    (p for p in perturbed_ds.samples if p.sample_id == sample.sample_id),
                    None,
                )
                if orig is not None:
                    row["original_question"] = orig.original_question
                    row["perturbed_tokens"] = [
                        {
                            "token_index": pt.token_index,
                            "original_token": pt.original_token,
                            "perturbed_token": pt.perturbed_token,
                            "importance_score": pt.importance_score,
                            "perturbation_type": pt.perturbation_type,
                        }
                        for pt in orig.perturbed_tokens
                    ]

            # R_Q 計算(プロンプト+<think>開始タグまでの短い入力で backward)
            if analyzer is not None:
                try:
                    rq_input = fp.full_prompt + generated_text[: think_prefix_end(generated_text)]
                    imp = analyzer.analyze(
                        input_text=rq_input,
                        target_position=-1,
                        question_char_start=fp.question_start,
                        question_char_end=fp.question_end,
                    )

                    def _filtered(scores, offsets, lo, hi):
                        out = []
                        for (tok, sc), (st, en) in zip(scores, offsets, strict=False):
                            keep = not (en <= lo or st >= hi)
                            out.append((tok, sc if keep else 0.0))
                        return out

                    raw_scores = [
                        (imp.tokens[j], imp.raw_relevance[j].item())
                        for j in range(len(imp.tokens))
                    ]
                    q_scores = _filtered(
                        raw_scores, imp.offset_mapping, fp.question_start, fp.question_end
                    )
                    qc_scores = _filtered(
                        raw_scores,
                        imp.offset_mapping,
                        fp.question_start,
                        fp.question_with_choices_end,
                    )
                    row["question_top_k_words"] = [
                        {"word": w.word, "score": w.score} for w in imp.top_k_words[:20]
                    ]
                    torch.save(
                        {
                            "type": "question",
                            "token_scores": q_scores,
                            "token_scores_with_choices": qc_scores,
                            "word_scores": [
                                {
                                    "word": w.word,
                                    "score": w.score,
                                    "token_indices": w.token_indices,
                                }
                                for w in imp.word_scores
                            ],
                            "top_k_words": [
                                {
                                    "word": w.word,
                                    "score": w.score,
                                    "token_indices": w.token_indices,
                                }
                                for w in imp.top_k_words
                            ],
                            "raw_relevance": imp.raw_relevance.cpu(),
                            "tokens": imp.tokens,
                            "offset_mapping": imp.offset_mapping,
                            "question_char_start": fp.question_start,
                            "question_char_end": fp.question_end,
                            "question_with_choices_end": fp.question_with_choices_end,
                        },
                        output_dir / "importance_scores" / f"{sample.sample_id}.pt",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"R_Q 計算失敗 {sample.sample_id}: {e}")

            results.append(row)

        del output_ids, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        done = len(results)
        logger.info(
            f"バッチ {bi + 1}/{n_batches} 完了 ({done}/{len(shard_samples)}), "
            f"正答率={correct_count / done:.1%}"
        )

        # シャード途中経過を毎バッチ上書き保存(中断復帰用)
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        shard_path = shard_dir / f"results_{args.start:05d}_{end:05d}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        f"シャード完了: {len(results)}件, 正答率={correct_count / max(1, len(results)):.2%}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""実験6-(i)〜(iii): 帰属ファミリー代替手法 (G×I / IG / rollout) の実行スクリプト.

アーカイブの run ディレクトリ (baseline / perturbed、results.json 必須) を
読み取り専用で入力し、各サンプルの CoT について指定手法の語ランキングを
計算して保存する。サンプル選定・3分割・語集約・R_C 比較の規約は
exp6-(iv) LOO (scripts/run_loo_scoring.py) と完全に共有する。

出力:
  {output_dir}/{model_short}_{benchmark}_{method}_{run_label}/
  ├── config.json    # 実行設定
  ├── results.json   # サンプルごとの手法ランキング (R_C 互換 word/score スキーマ)
  └── summary.json   # 集計 (手法 vs R_C の Jaccard@10 など)

使用例 (GPU ヘルパー経由):
  bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_attribution_family.py \\
    --run_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \\
    --model google/gemma-3-4b-it --benchmark gsm8k --method gxi \\
    --n 300 --seed 42 --output_dir results/attribution_family --run_label clean
"""

import argparse
import importlib.util
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("attribution_family")
logger.setLevel(logging.INFO)

METHODS = ["gxi", "ig", "rollout"]


def _load_loo_cli():
    """run_loo_scoring.py の CLI ヘルパーを importlib で共有する.

    サンプル選定 (select_sample_ids) / プロンプト再構築 (build_prompt) /
    R_C ローダー (load_rc_ranking、Mistral 再構築フォールバック込み) を
    LOO 本番と完全に同一実装で使うための隔離点。
    """
    path = Path(__file__).resolve().parent / "run_loo_scoring.py"
    spec = importlib.util.spec_from_file_location("run_loo_scoring", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_and_tokenizer(model_name: str, gpu_id: str, method: str):
    """手法に応じたモデルロード.

    - gxi / ig: 素の transformers モデル (lxt パッチなし、backward 用)
    - rollout: attn_implementation="eager" (output_attentions が実 attention を
      返すため必須。SDPA では attention が返らない)
    """
    import torch

    from typo_cot.models.wrapper import create_model_wrapper, setup_device

    if method == "rollout":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, _ = setup_device(gpu_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    wrapper = create_model_wrapper(
        model_name=model_name, gpu_id=gpu_id, wrap_for_lxt=False
    )
    return wrapper.model, wrapper.tokenizer


def compute_method_scores(
    method: str, model, prep, ig_steps: int, ig_step_batch: int
) -> tuple[list[float], dict]:
    """1サンプルのトークン別スコアと手法別メタデータを計算する."""
    from typo_cot.attribution_family.methods import (
        attention_rollout_token_scores,
        gradient_x_input_token_scores,
        integrated_gradients_token_scores,
    )

    if method == "gxi":
        scores, obj = gradient_x_input_token_scores(
            model, prep.input_ids, prep.target_start
        )
        return scores, {"objective_logprob": obj}
    if method == "ig":
        scores, info = integrated_gradients_token_scores(
            model,
            prep.input_ids,
            prep.target_start,
            steps=ig_steps,
            step_batch=ig_step_batch,
        )
        return scores, {"ig_completeness": info}
    if method == "rollout":
        scores = attention_rollout_token_scores(
            model, prep.input_ids, prep.target_start
        )
        return scores, {}
    raise ValueError(f"unknown method: {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribution-family CoT word ranking (exp6 i-iii)"
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="results.json を含む run ディレクトリ (読み取り専用)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="指定時: clean 正解サンプルから seed 固定で n 件選定 "
                             "(本番プロトコル seed=42、LOO 本番と同一サンプル)")
    parser.add_argument("--clean_run_dir", type=str, default=None,
                        help="サンプル選定に使う clean run (摂動 run では baseline)")
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="results/attribution_family")
    parser.add_argument("--run_label", type=str, default="clean")
    parser.add_argument("--jaccard_k", type=int, default=10)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--ig_step_batch", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="これを超える系列はスキップ (メモリ保護)")
    args = parser.parse_args()

    loo_cli = _load_loo_cli()
    run_dir = Path(args.run_dir)

    n_selected = None
    if args.seed is not None:
        clean_dir = Path(args.clean_run_dir) if args.clean_run_dir else run_dir
        selected = loo_cli.select_sample_ids(
            loo_cli.load_run_entries(clean_dir), args.n, args.seed
        )
        n_selected = len(selected)
        selected_set = set(selected)
        entries = [
            e for e in loo_cli.load_run_entries(run_dir)
            if e["sample_id"] in selected_set
        ]
        logger.info(
            f"サンプル選定: clean 正解から seed={args.seed} で {n_selected} 件 "
            f"(run_dir 内で一致 {len(entries)} 件)"
        )
    else:
        entries = loo_cli.load_run_entries(run_dir)[args.sample_offset:]

    model_short = args.model.split("/")[-1]
    out_dir = (
        Path(args.output_dir)
        / f"{model_short}_{args.benchmark}_{args.method}_{args.run_label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "method": args.method,
        "run_dir": str(run_dir),
        "n": args.n,
        "seed": args.seed,
        "clean_run_dir": args.clean_run_dir,
        "n_selected": n_selected,
        "jaccard_k": args.jaccard_k,
        "ig_steps": args.ig_steps if args.method == "ig" else None,
        "ig_step_batch": args.ig_step_batch if args.method == "ig" else None,
        "ig_baseline": "zero_embedding" if args.method == "ig" else None,
        "rollout_residual": 0.5 if args.method == "rollout" else None,
        "objective": "answer_token_sequence_logprob",
        "max_tokens": args.max_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    import os

    from typo_cot.attribution_family.methods import (
        decode_tokens_for_alignment,
        prepare_sample,
        token_scores_to_word_ranking,
    )
    from typo_cot.intervention.loo_scorer import loo_jaccard_topk
    from typo_cot.models.prompts import create_prompt_template

    gpu_id = loo_cli.resolve_gpu_id(args.gpu_id, os.environ)
    logger.info(f"モデルをロード: {args.model} (GPU {gpu_id}, method={args.method})")
    model, tokenizer = load_model_and_tokenizer(args.model, gpu_id, args.method)
    template = create_prompt_template(args.benchmark)

    stats = {
        "total_seen": 0,
        "scored": 0,
        "skip_no_answer_pattern": 0,
        "skip_too_long": 0,
        "align_failed": 0,
        "errors": 0,
        "rc_degenerate": 0,
    }
    out_results: list[dict] = []
    t0 = time.time()

    for entry in entries:
        if args.n is not None and stats["scored"] >= args.n:
            break
        stats["total_seen"] += 1
        sid = entry["sample_id"]

        try:
            prompt = loo_cli.build_prompt(template, args.benchmark, entry)
            prep = prepare_sample(tokenizer, prompt, entry["generated_text"])
            if prep is None:
                stats["skip_no_answer_pattern"] += 1
                continue
            if len(prep.input_ids) > args.max_tokens:
                stats["skip_too_long"] += 1
                continue
            token_scores, extra = compute_method_scores(
                args.method, model, prep, args.ig_steps, args.ig_step_batch
            )
            tokens = decode_tokens_for_alignment(tokenizer, prep.input_ids)
            ranking = token_scores_to_word_ranking(
                tokens,
                token_scores,
                prep.full_text,
                prep.cot_token_start,
                prep.cot_token_end,
            )
            if ranking is None:
                stats["align_failed"] += 1
                logger.warning(f"{sid}: トークン整合失敗 (語ランキング構築不可)")
                continue
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            logger.warning(f"{sid} 失敗: {exc}")
            continue

        # R_C (AttnLRP) の CoT 語ランキングとの Jaccard@k。full_text 配線により
        # Mistral の結合不良アーカイブでは token_scores 再構築ローダーが働く。
        jaccard_vs_rc = None
        rc_ranking, rc_degenerate = loo_cli.load_rc_ranking(
            run_dir, sid, full_text=prompt + entry["generated_text"]
        )
        if rc_ranking is not None:
            stats["rc_degenerate"] += int(rc_degenerate)
            jaccard_vs_rc = loo_jaccard_topk(
                ranking, rc_ranking, k=args.jaccard_k
            )

        out_results.append(
            {
                "sample_id": sid,
                "extracted_answer": entry.get("extracted_answer"),
                "is_correct": entry.get("is_correct"),
                "answer_text": prep.answer_text,
                "pattern_type": prep.pattern_type,
                "method": args.method,
                "n_tokens": len(prep.input_ids),
                "n_cot_tokens": prep.cot_token_end - prep.cot_token_start + 1,
                "n_ranked_words": len(ranking),
                # R_C ランキング (cot_top_k_words) と同一スキーマの完全ランキング
                "method_word_scores": ranking,
                f"vs_rc_jaccard_top{args.jaccard_k}": jaccard_vs_rc,
                **extra,
            }
        )
        stats["scored"] += 1

        if stats["scored"] % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / stats["scored"]
            logger.info(f"{stats['scored']} 件処理済 ({rate:.1f}s/sample)")
            tmp = out_dir / "results.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out_results, f, ensure_ascii=False)
            tmp.replace(out_dir / "results.json")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)

    jkey = f"vs_rc_jaccard_top{args.jaccard_k}"
    jaccards = [r[jkey] for r in out_results if r[jkey] is not None]
    n_words = [r["n_ranked_words"] for r in out_results]
    completeness = [
        r["ig_completeness"]["completeness_ratio"]
        for r in out_results
        if r.get("ig_completeness")
        and r["ig_completeness"].get("completeness_ratio") is not None
    ]
    summary = {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "method": args.method,
            "run_dir": str(run_dir),
            "run_label": args.run_label,
            "timestamp": datetime.now().isoformat(),
        },
        "stats": {**stats, "elapsed_sec": round(time.time() - t0, 1)},
        "metrics": {
            f"mean_{jkey}": statistics.mean(jaccards) if jaccards else None,
            f"median_{jkey}": statistics.median(jaccards) if jaccards else None,
            "n_with_rc": len(jaccards),
            "mean_n_ranked_words": statistics.mean(n_words) if n_words else None,
            "mean_ig_completeness_ratio": (
                statistics.mean(completeness) if completeness else None
            ),
            "median_ig_completeness_ratio": (
                statistics.median(completeness) if completeness else None
            ),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"完了: {out_dir}")
    logger.info(json.dumps(summary["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()

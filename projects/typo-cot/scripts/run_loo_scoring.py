#!/usr/bin/env python3
"""実験6-(iv): leave-one-out (LOO) 語重要度スコアリングの実行スクリプト.

アーカイブの run ディレクトリ（baseline / perturbed、results.json 必須）を
読み取り専用で入力し、各サンプルの clean(または摂動) CoT について
全語タイプの LOO ランキングを計算して保存する。

出力:
  {output_dir}/{model_short}_{benchmark}_{run_label}/
  ├── config.json    # 実行設定
  ├── results.json   # サンプルごとの LOO ランキング (R_C 互換 word/score スキーマ)
  └── summary.json   # 集計 (Jaccard@10 vs R_C など)

使用例 (GPU ヘルパー経由):
  bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_loo_scoring.py \\
    --run_dir /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/gemma-3-4b-it_gsm8k \\
    --model google/gemma-3-4b-it --benchmark gsm8k --n 16 \\
    --output_dir results/smoke
"""

import argparse
import json
import logging
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("loo_scoring")
logger.setLevel(logging.INFO)

NUMERIC_CHARS = set("0123456789")
OPERATOR_WORDS = {"=", "+", "-", "*", "/", "×", "÷", "%"}


def load_run_entries(run_dir: Path) -> list[dict]:
    """run ディレクトリの results.json を読む.

    Step 0 の master table が入手可能になったら、この関数だけを
    「condition で master table を引く」実装に差し替える（データアクセスの隔離点）。
    """
    with open(run_dir / "results.json", encoding="utf-8") as f:
        return json.load(f)


def select_sample_ids(
    entries: list[dict], n: int | None, seed: int
) -> list[str]:
    """clean 正解サンプルから決定論的に n 件の sample_id を選定する.

    - 対象は is_correct=True のエントリのみ（clean run の results.json）
    - random.Random(seed) で n 件を非復元抽出し、results.json の出現順で返す
    - n が None または正解数以上なら全正解 id を出現順で返す
    """
    correct_ids = [e["sample_id"] for e in entries if e.get("is_correct")]
    if n is None or n >= len(correct_ids):
        return correct_ids
    chosen = set(random.Random(seed).sample(correct_ids, n))
    return [sid for sid in correct_ids if sid in chosen]


def load_rc_ranking(
    run_dir: Path, sample_id: str, full_text: str | None = None
) -> tuple[list[dict] | None, bool]:
    """importance_scores/{sid}_cot.pt から R_C 語ランキングをロードする.

    full_text (prompt + generated_text) を渡すと、word_scores が結合不良の
    アーカイブ (Mistral: 全文が1語に潰れている) では token_scores からの
    再構築フォールバック (rc_word_ranking_from_cot_pt) が働く。

    Returns:
        (ranking | None, degenerate) — ranking はスコア降順 [{word, score}]、
        degenerate は word_scores 結合不良の検出フラグ (summary 集計用)。
    """
    path = run_dir / "importance_scores" / f"{sample_id}_cot.pt"
    if not path.exists():
        return None, False
    import torch

    from typo_cot.intervention.loo_scorer import (
        rc_word_ranking_from_cot_pt,
        word_scores_degenerate,
    )

    data = torch.load(path, map_location="cpu", weights_only=False)
    degenerate = word_scores_degenerate(data)
    return rc_word_ranking_from_cot_pt(data, full_text=full_text), degenerate


def build_prompt(template, benchmark: str, entry: dict) -> str:
    """run_inference / rebuttal スクリプトと同一のプロンプト再構築."""
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        prompt_result = template.generate(
            question=entry["question"],
            choices=entry.get("choices"),
            subject=entry.get("subset"),
        )
    elif benchmark == "gsm8k":
        prompt_result = template.generate(question=entry["question"])
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        prompt_result = template.generate(
            question=entry["question"], subject=entry.get("subset")
        )
    else:
        prompt_result = template.generate(question=entry["question"])
    return prompt_result.get_full_prompt()


def resolve_gpu_id(cli_gpu_id: str, env: dict) -> str:
    """使用する GPU ID を解決する.

    create_model_wrapper -> setup_device は CUDA_VISIBLE_DEVICES を gpu_id で
    無条件に上書きするため、run_with_gpu.sh などの外部ランチャーが設定した
    CUDA_VISIBLE_DEVICES があればそれを優先する（既存値と同じ値を再設定する
    だけになり、マスクが壊れない）。未設定なら CLI の --gpu_id を使う。
    """
    env_cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    return env_cvd if env_cvd else cli_gpu_id


def is_numeric_or_operator(word: str) -> bool:
    """数値・演算語の判定（スモーク基準(b)の集計用）."""
    return word in OPERATOR_WORDS or any(c in NUMERIC_CHARS for c in word)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOO word importance scoring (exp6-iv)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="results.json を含む run ディレクトリ (読み取り専用)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--n", type=int, default=None,
                        help="スコアリングするサンプル数上限 (None=全件)")
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="results/loo")
    parser.add_argument("--jaccard_k", type=int, default=10)
    parser.add_argument("--deletion-mode", dest="deletion_mode", type=str,
                        default="occurrence", choices=["occurrence", "type"],
                        help="occurrence=案B(出現ごと削除→タイプ平均、主定義) / "
                             "type=案A(全出現一括削除、感度分析)")
    parser.add_argument("--run_label", type=str, default="loo",
                        help="出力ディレクトリ名の接尾辞 (clean/perturbed の区別などに使う)")
    parser.add_argument("--seed", type=int, default=None,
                        help="指定時: clean 正解サンプルから seed 固定で n 件を"
                             "決定論的に選定して対象を絞る (本番プロトコル seed=42)")
    parser.add_argument("--clean_run_dir", type=str, default=None,
                        help="サンプル選定に使う clean run ディレクトリ "
                             "(省略時 run_dir。摂動 run では baseline を指定)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    n_selected = None
    if args.seed is not None:
        clean_dir = Path(args.clean_run_dir) if args.clean_run_dir else run_dir
        selected = select_sample_ids(load_run_entries(clean_dir), args.n, args.seed)
        n_selected = len(selected)
        selected_set = set(selected)
        entries = [
            e for e in load_run_entries(run_dir)
            if e["sample_id"] in selected_set
        ]
        logger.info(
            f"サンプル選定: clean 正解から seed={args.seed} で {n_selected} 件 "
            f"(run_dir 内で一致 {len(entries)} 件)"
        )
    else:
        entries = load_run_entries(run_dir)[args.sample_offset:]

    model_short = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) / f"{model_short}_{args.benchmark}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "run_dir": str(run_dir),
        "n": args.n,
        "sample_offset": args.sample_offset,
        "batch_size": args.batch_size,
        "jaccard_k": args.jaccard_k,
        "method": "loo",
        "deletion_mode": args.deletion_mode,
        "aggregation": "mean" if args.deletion_mode == "occurrence" else "whole_type",
        "seed": args.seed,
        "clean_run_dir": args.clean_run_dir,
        "n_selected": n_selected,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    from typo_cot.intervention.loo_scorer import (
        loo_jaccard_topk,
        score_sample_loo,
    )
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    import os

    gpu_id = resolve_gpu_id(args.gpu_id, os.environ)
    logger.info(f"モデルをロード: {args.model} (GPU {gpu_id})")
    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=gpu_id, wrap_for_lxt=False
    )
    model, tokenizer = wrapper.model, wrapper.tokenizer
    template = create_prompt_template(args.benchmark)

    stats = {
        "total_seen": 0,
        "scored": 0,
        "skip_no_answer_pattern": 0,
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
            prompt = build_prompt(template, args.benchmark, entry)
            loo = score_sample_loo(
                model, tokenizer, prompt, entry["generated_text"],
                batch_size=args.batch_size,
                deletion_mode=args.deletion_mode,
            )
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            logger.warning(f"{sid} 失敗: {exc}")
            continue

        if loo is None:
            stats["skip_no_answer_pattern"] += 1
            continue

        # R_C (AttnLRP) の CoT 語ランキングとの Jaccard@k。
        # full_text 配線により Mistral の結合不良アーカイブ (word_scores が
        # 全文1語) では token_scores からの再構築ローダーが自動で働く。
        jaccard_vs_rc = None
        rc_ranking, rc_degenerate = load_rc_ranking(
            run_dir, sid, full_text=prompt + entry["generated_text"]
        )
        if rc_ranking is not None:
            stats["rc_degenerate"] += int(rc_degenerate)
            jaccard_vs_rc = loo_jaccard_topk(
                loo["word_scores"], rc_ranking, k=args.jaccard_k
            )

        top10 = loo["word_scores"][:10]
        out_results.append(
            {
                "sample_id": sid,
                "extracted_answer": entry.get("extracted_answer"),
                "is_correct": entry.get("is_correct"),
                "answer_text": loo["answer_text"],
                "pattern_type": loo["pattern_type"],
                "base_logprob": loo["base_logprob"],
                "n_word_types": loo["n_word_types"],
                "n_variants": loo["n_variants"],
                "deletion_mode": loo["deletion_mode"],
                "aggregation": loo["aggregation"],
                # R_C ランキング (cot_top_k_words) と同一スキーマの完全ランキング
                "loo_word_scores": loo["word_scores"],
                # 付帯情報 (出現回数・変種 log-prob)
                "word_types": loo["word_types"],
                f"loo_vs_rc_jaccard_top{args.jaccard_k}": jaccard_vs_rc,
                "top10_numeric_or_operator": sum(
                    1 for ws in top10 if is_numeric_or_operator(ws["word"])
                ),
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

    jkey = f"loo_vs_rc_jaccard_top{args.jaccard_k}"
    jaccards = [r[jkey] for r in out_results if r[jkey] is not None]
    n_types = [r["n_word_types"] for r in out_results]
    n_variants = [r["n_variants"] for r in out_results]
    top1_scores = [
        r["loo_word_scores"][0]["score"] for r in out_results if r["loo_word_scores"]
    ]
    summary = {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "method": "loo",
            "deletion_mode": args.deletion_mode,
            "run_dir": str(run_dir),
            "timestamp": datetime.now().isoformat(),
        },
        "stats": {**stats, "elapsed_sec": round(time.time() - t0, 1)},
        "metrics": {
            f"mean_{jkey}": statistics.mean(jaccards) if jaccards else None,
            f"median_{jkey}": statistics.median(jaccards) if jaccards else None,
            "mean_n_word_types": statistics.mean(n_types) if n_types else None,
            "min_n_word_types": min(n_types) if n_types else None,
            "max_n_word_types": max(n_types) if n_types else None,
            "mean_n_variants": statistics.mean(n_variants) if n_variants else None,
            "total_n_variants": sum(n_variants) if n_variants else None,
            "mean_top1_loo_score": statistics.mean(top1_scores) if top1_scores else None,
            "mean_top10_numeric_or_operator": (
                statistics.mean(r["top10_numeric_or_operator"] for r in out_results)
                if out_results
                else None
            ),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"完了: {out_dir}")
    logger.info(json.dumps(summary["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()

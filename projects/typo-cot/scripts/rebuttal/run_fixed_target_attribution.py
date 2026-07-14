#!/usr/bin/env python3
"""Rebuttal 実験①: Fixed-Target Attribution (AxQH Weakness 2 / endogeneity 反証).

摂動後条件の CoT→Answer 帰属 (R_C) を、摂動前 (baseline) 生成の answer span 文字列を
固定ターゲットとして再計算する。

実装方式 (force-decode splice):
- 摂動後 generated_text 中の最終回答スパン ("The answer is (X)" 等) の選択肢/数値部分を、
  baseline 生成の回答文字列に置換した spliced_text を構築する。
- 既存パイプラインの analyze_combined と同一の手順・同一の compute_relevance 規約
  (回答トークン位置の max logit を backprop) で CoT→Answer relevance を再計算する。
- 回答が flip していないサンプルでは spliced_text == 元テキストとなり、標準 R_C と
  一致する (これを --sanity でサニティチェックとして利用)。

出力 (既存 run_analysis.py 互換):
  {output_dir}/{model_short}_{benchmark}_k{N}_fixed_target/
  ├── config.json                  # 摂動後 run の config + perturbation_mode=fixed_target
  ├── results.json                 # 摂動後 run のエントリのコピー (+ fixed_target メタ)
  ├── fixed_target_stats.json      # 処理統計
  └── importance_scores/
      ├── {id}.pt                  # 摂動後 run へのシンボリックリンク (Q側は不変)
      └── {id}_cot.pt              # 固定ターゲット版 R_C (既存スキーマ互換)

使用例:
  uv run --no-sync python scripts/rebuttal/run_fixed_target_attribution.py \
    --baseline_dir outputs/baseline/gemma-3-4b-it_mmlu \
    --perturbed_dir outputs/perturbed/gemma-3-4b-it_mmlu_k4_importance \
    --model google/gemma-3-4b-it --benchmark mmlu --gpu_id 2 \
    --output_dir outputs/rebuttal/fixed_target
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path


# ログ設定
logging.basicConfig(
    level=logging.WARNING,  # analyzer の INFO ログ (1サンプル毎) を抑制
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fixed_target")
logger.setLevel(logging.INFO)

# lrp/analyzer.py:550-566 の回答パターンをコピー (テキストマッチ部分のみ使用)。
# analyzer._find_answer_pattern と同一の正規表現・同一の「最後のマッチ採用」規約。
ANSWER_PATTERNS = [
    (r"[Tt]he\s+answer\s+is[:\s]*\(([A-Ja-j])\)", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"[Aa]nswer[:\s]+\(([A-Ja-j])\)", "choice"),
    (r"[Aa]nswer[:\s]+([A-Ja-j])(?:\.|,|\s|$)", "choice"),
    (r"\*\*\(([A-Ja-j])\)\*\*", "choice"),
    (r"\*\*([A-Ja-j])\*\*", "choice"),
    (r"(?:correct|right)\s+(?:answer|option)\s+is[:\s]*\(?([A-Ja-j])\)?", "choice"),
    (r"[Tt]he\s+answer\s+is[:\s]*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Tt]he\s+answer\s+is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"####\s*(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"[Aa]nswer[:\s]+(-?[\d,]+(?:\.\d+)?)", "number"),
    (r"(?:^|\n)\s*\(?([A-Ja-j])\)?\s*\.?\s*$", "choice"),
]


def find_answer_match(text: str):
    """analyzer._find_answer_pattern と同じ規約で最終回答のマッチを返す.

    Returns:
        (match, pattern_type) または (None, None)
    """
    for pattern, ptype in ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1], ptype
    return None, None


def build_prompt(template, benchmark: str, entry: dict) -> str:
    """run_inference.generate_prompt_for_sample と同一のプロンプト再構築."""
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        prompt_result = template.generate(
            question=entry["question"],
            choices=entry.get("choices"),  # Phase3 では None (選択肢は question に埋め込み済)
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


def analyze_cot_fixed(analyzer, prompt: str, generated_text: str) -> dict:
    """analyze_combined (analyzer.py:636-847) の CoT→Answer パスのみを同一手順で実行.

    質問側パス (Question→CoT) は省略する (固定ターゲット化の影響を受けないため)。
    """
    import torch

    full_text = prompt + generated_text
    prompt_length = len(prompt)

    prompt_inputs = analyzer.tokenizer(prompt, return_tensors="pt")
    prompt_token_count = prompt_inputs["input_ids"].shape[1]

    full_inputs = analyzer.tokenizer(full_text, return_tensors="pt")
    full_input_ids = full_inputs["input_ids"].to(analyzer.device)
    total_token_count = full_input_ids.shape[1]

    offset_list = analyzer._compute_offset_mapping_fallback(
        full_text, full_input_ids[0].tolist()
    )
    tokens = [analyzer.tokenizer.decode([tid]) for tid in full_input_ids[0].tolist()]

    answer_token_start, answer_token_end, answer_choice_position = (
        analyzer._find_answer_pattern(
            generated_text=generated_text,
            tokens=tokens,
            prompt_token_count=prompt_token_count,
            offset_list=offset_list,
            prompt_length=prompt_length,
        )
    )

    cot_token_start = prompt_token_count
    if answer_token_start is not None:
        cot_token_end = answer_token_start - 1
    else:
        cot_token_end = total_token_count - 2

    if answer_choice_position is not None:
        target_position = answer_choice_position
    else:
        target_position = -1

    cot_relevance = analyzer.compute_relevance(
        full_input_ids, target_position=target_position
    )

    # analyze_combined と同一のフィルタリング (プロンプト部と回答部を0に)
    cot_filtered_relevance = cot_relevance.clone()
    for i in range(prompt_token_count):
        cot_filtered_relevance[i] = 0.0
    if answer_token_start is not None:
        for i in range(answer_token_start, total_token_count):
            cot_filtered_relevance[i] = 0.0

    cot_word_scores = analyzer.tokens_to_words(tokens, cot_filtered_relevance)
    cot_token_scores = [
        (tokens[i], cot_relevance[i].item()) for i in range(len(tokens))
    ]

    result = {
        "type": "cot",
        "token_scores": cot_token_scores,
        "word_scores": [
            {"word": ws.word, "score": ws.score, "token_indices": ws.token_indices}
            for ws in cot_word_scores
        ],
        "raw_relevance": cot_filtered_relevance.cpu(),
        "cot_token_start": cot_token_start,
        "cot_token_end": cot_token_end,
        # 追加メタ情報 (既存スキーマには無いが analysis 側は未使用キーを無視する)
        "fixed_target_position": target_position,
        "fixed_answer_token_start": answer_token_start,
    }
    del cot_relevance, cot_filtered_relevance, full_input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def top_k_token_set(token_scores: list, k: int = 10) -> set:
    """analysis/metrics.py top_k_jaccard_by_token と同じ規約 (トークン文字列 dedup, max)."""
    best: dict[str, float] = {}
    for tok, score in token_scores:
        if tok not in best or score > best[tok]:
            best[tok] = score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return {t for t, _ in ranked[: min(k, len(ranked))]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-Target Attribution (rebuttal ①)")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--perturbed_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="outputs/rebuttal/fixed_target")
    parser.add_argument("--limit", type=int, default=None, help="パイロット用サンプル数上限")
    parser.add_argument(
        "--sanity", type=int, default=0,
        help="flip していないサンプル N 件で既存 _cot.pt との一致を検証",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    perturbed_dir = Path(args.perturbed_dir)

    with open(baseline_dir / "results.json", encoding="utf-8") as f:
        baseline_results = {r["sample_id"]: r for r in json.load(f)}
    with open(perturbed_dir / "results.json", encoding="utf-8") as f:
        perturbed_results = json.load(f)
    with open(perturbed_dir / "config.json", encoding="utf-8") as f:
        perturbed_config = json.load(f)

    num_perturbations = perturbed_config.get("perturbed_metadata", {}).get(
        "num_perturbations", "unknown"
    )
    model_short = args.model.split("/")[-1]
    out_dir = (
        Path(args.output_dir)
        / f"{model_short}_{args.benchmark}_k{num_perturbations}_fixed_target"
    )
    scores_dir = out_dir / "importance_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # config.json: 摂動後 run の config を継承しモードを書き換え
    config = dict(perturbed_config)
    config["fixed_target_source"] = str(perturbed_dir)
    config["baseline_source"] = str(baseline_dir)
    config["timestamp"] = datetime.now().isoformat()
    if "perturbed_metadata" in config:
        config["perturbed_metadata"] = dict(config["perturbed_metadata"])
        config["perturbed_metadata"]["perturbation_mode"] = "fixed_target"
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # モデル・分析器 (run_inference.py:354-371 と同一)
    from typo_cot.lrp.analyzer import create_analyzer
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    logger.info(f"モデルをロード: {args.model} (GPU {args.gpu_id})")
    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=True
    )
    template = create_prompt_template(args.benchmark)
    analyzer = create_analyzer(
        model=wrapper.model, tokenizer=wrapper.tokenizer, top_k=None,
        device=wrapper.device,
    )

    stats = {
        "total": 0,
        "processed": 0,
        "spliced": 0,           # flip 等で回答文字列を置換したサンプル
        "identical": 0,         # baseline 回答 == 摂動後回答 (splice 無し)
        "skip_no_baseline": 0,  # baseline 側に存在しない
        "skip_no_base_answer": 0,   # baseline 生成に回答パターン無し
        "skip_no_pert_answer": 0,   # 摂動後生成に回答パターン無し
        "errors": 0,
    }
    skipped_ids: dict[str, str] = {}
    sanity_report = []
    out_results = []

    samples = perturbed_results[: args.limit] if args.limit else perturbed_results
    t0 = time.time()

    for idx, entry in enumerate(samples):
        sid = entry["sample_id"]
        stats["total"] += 1

        base = baseline_results.get(sid)
        if base is None:
            stats["skip_no_baseline"] += 1
            skipped_ids[sid] = "no_baseline"
            continue

        base_match, base_type = find_answer_match(base["generated_text"])
        if base_match is None:
            stats["skip_no_base_answer"] += 1
            skipped_ids[sid] = "no_baseline_answer_pattern"
            continue
        base_answer_str = base_match.group(1)

        pert_gen = entry["generated_text"]
        pert_match, pert_type = find_answer_match(pert_gen)
        if pert_match is None:
            stats["skip_no_pert_answer"] += 1
            skipped_ids[sid] = "no_perturbed_answer_pattern"
            continue

        s, e = pert_match.span(1)
        pert_answer_str = pert_gen[s:e]
        if pert_answer_str == base_answer_str:
            spliced_text = pert_gen
            stats["identical"] += 1
            spliced = False
        else:
            spliced_text = pert_gen[:s] + base_answer_str + pert_gen[e:]
            stats["spliced"] += 1
            spliced = True

        try:
            prompt = build_prompt(template, args.benchmark, entry)
            cot_data = analyze_cot_fixed(analyzer, prompt, spliced_text)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            skipped_ids[sid] = f"error: {exc}"
            logger.warning(f"{sid} 失敗: {exc}")
            continue

        cot_data["fixed_target_baseline_answer"] = base_answer_str
        cot_data["fixed_target_perturbed_answer"] = pert_answer_str
        cot_data["fixed_target_spliced"] = spliced

        import torch

        torch.save(cot_data, scores_dir / f"{sid}_cot.pt")

        # 質問側 .pt は摂動後 run と同一 (ターゲット固定の影響なし) → シンボリックリンク
        src_q = (perturbed_dir / "importance_scores" / f"{sid}.pt").resolve()
        dst_q = scores_dir / f"{sid}.pt"
        if src_q.exists() and not dst_q.exists():
            dst_q.symlink_to(src_q)

        out_entry = dict(entry)
        out_entry["fixed_target"] = {
            "baseline_answer": base_answer_str,
            "perturbed_answer": pert_answer_str,
            "spliced": spliced,
            "baseline_pattern_type": base_type,
            "perturbed_pattern_type": pert_type,
        }
        out_results.append(out_entry)
        stats["processed"] += 1

        # サニティチェック: splice 無しサンプルは既存 _cot.pt と一致するはず
        if args.sanity and not spliced and len(sanity_report) < args.sanity:
            import torch

            ref_path = perturbed_dir / "importance_scores" / f"{sid}_cot.pt"
            if ref_path.exists():
                ref = torch.load(ref_path, map_location="cpu", weights_only=False)
                ref_set = top_k_token_set(ref["token_scores"], 10)
                new_set = top_k_token_set(cot_data["token_scores"], 10)
                overlap = len(ref_set & new_set) / max(1, len(ref_set | new_set))
                sanity_report.append(
                    {
                        "sample_id": sid,
                        "top10_jaccard_vs_existing": overlap,
                        "cot_range_match": (
                            ref.get("cot_token_start") == cot_data["cot_token_start"]
                            and ref.get("cot_token_end") == cot_data["cot_token_end"]
                        ),
                    }
                )
                logger.info(
                    f"[sanity] {sid}: top10 Jaccard vs 既存 = {overlap:.3f}, "
                    f"CoT範囲一致 = {sanity_report[-1]['cot_range_match']}"
                )

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta_min = rate * (len(samples) - idx - 1) / 60
            logger.info(
                f"{idx + 1}/{len(samples)} 処理済 ({rate:.2f}s/sample, 残り約{eta_min:.0f}分)"
            )
            # 中間保存 (中断保険)
            tmp = out_dir / "results.json.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out_results, f, ensure_ascii=False)
            tmp.replace(out_dir / "results.json")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["skipped_ids"] = skipped_ids
    stats["sanity_report"] = sanity_report
    with open(out_dir / "fixed_target_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # summary.json (run_analysis には不要だが一覧性のため)
    correct = sum(1 for r in out_results if r.get("is_correct"))
    summary = {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "mode": "fixed_target",
            "timestamp": datetime.now().isoformat(),
        },
        "overall_metrics": {
            "accuracy": correct / len(out_results) if out_results else 0,
            "total_correct": correct,
            "total_samples": len(out_results),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"完了: {out_dir}")
    logger.info(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("skipped_ids", "sanity_report")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

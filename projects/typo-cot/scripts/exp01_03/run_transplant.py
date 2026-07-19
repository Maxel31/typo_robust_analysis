#!/usr/bin/env python3
"""実験1 (CoT transplant 2×2) + 実験3 (--dump-divergence) 実行スクリプト.

アーカイブの baseline/perturbed 生成ログから PairRecord を構築し、
4 セル (A/B/C/D) の teacher-forcing 生成 → flip 表 / bootstrap / GLMM。
--dump-divergence 指定時は A セルと C セル (DE 条件) の forward を共有して
CoT 位置別の KL / log-prob / rank プロファイルを出力する。

GPU 実行は必ず run_with_gpu.sh 経由 (CUDA_VISIBLE_DEVICES はヘルパーが設定
するため、このスクリプトでは一切変更しない)。

例:
    bash <...>/run_with_gpu.sh uv run python scripts/exp01_03/run_transplant.py \
        --model google/gemma-3-4b-it --benchmark gsm8k \
        --baseline-dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
        --perturbed-dir <archive>/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
        --n 32 --dump-divergence --output-dir results/smoke/gemma3-4b_gsm8k_lxt4
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from typo_cot.intervention.analysis import (
    bootstrap_flip_cis,
    flip_table,
    glmm_decomposition,
)
from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import build_cell_inputs
from typo_cot.intervention.divergence import (
    align_cot_targets,
    divergence_onset,
    positionwise_divergence,
    precision_at_k,
    shuffle_null_precision,
)
from typo_cot.intervention.records import PairRecord
from typo_cot.intervention.reasoning_cells import (
    make_reasoning_extract_fn,
    make_reasoning_prompt_builder,
    truncate_reasoning_cot,
)
from typo_cot.intervention.runner import run_cells
from typo_cot.evaluation.extractor import create_extractor
from typo_cot.models.wrapper import ModelWrapper

# R1蒸留系 (reasoning モデル) の自動判定用マーカー
_REASONING_MARKERS = ("r1-distill", "deepseek-r1")


def is_reasoning_model(model_name: str) -> bool:
    """モデル名から R1蒸留系 (reasoning モデル) かを判定する."""
    low = model_name.lower()
    return any(m in low for m in _REASONING_MARKERS)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_transplant")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace モデル名")
    p.add_argument("--benchmark", required=True, help="ベンチマーク名 (gsm8k/mmlu/...)")
    p.add_argument("--baseline-dir", required=True, help="アーカイブ baseline ディレクトリ")
    p.add_argument("--perturbed-dir", required=True, help="アーカイブ perturbed ディレクトリ")
    p.add_argument("--output-dir", required=True, help="結果出力先")
    p.add_argument("--n", type=int, default=None, help="サンプル数上限 (スモーク・シャード分割用)")
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="結合済みペアの先頭 start 件を読み飛ばす (大設定のシャード分割用)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="答えスパン再生成の最大トークン (既定: 基底16 / reasoning 64)",
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        help=(
            "R1蒸留系 (DeepSeek-R1-Distill) モード: チャットテンプレート + <think> 構造対応の"
            "切断・抽出・生成を使う。モデル名から自動判定もされる"
        ),
    )
    p.add_argument(
        "--clean-correct-only",
        action="store_true",
        help="clean 条件正解のサンプルのみロード (主分析対象を先に絞る)",
    )
    p.add_argument("--dump-divergence", action="store_true", help="実験3 の位置別プロファイル出力")
    p.add_argument(
        "--divergence-rank-threshold", type=int, default=5, help="発散オンセットの順位しきい値"
    )
    p.add_argument("--precision-k", type=int, default=10)
    p.add_argument("--n-shuffles", type=int, default=1000, help="precision@k 帰無分布の B")
    p.add_argument(
        "--trigger-pattern",
        default=None,
        help="答え句の正規表現 (モデル別差し替え。既定: The answer is)",
    )
    p.add_argument(
        "--dedup-same-answer-triggers",
        action="store_true",
        help=(
            "同一答えを繰り返すだけの複数トリガー (Qwen の癖) を multi_trigger 除外"
            "しない。切断点は最初のトリガー直前で不変 (既定 False で従来挙動)"
        ),
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--strip-conclusion-mode",
        default=None,
        choices=["last_line", "last_sentence"],
        help=(
            "A2 (ii) 結論剥ぎ: C セル (typo質問+clean CoT) の強制 CoT の末尾行/文を"
            "除去して restore を再測定する。GSM8K の末尾読み上げ行に金答え数値が"
            "載る「丸写し」経路を潰す検証用 (既定 None で従来挙動)"
        ),
    )
    return p.parse_args()


def build_generate_fn(wrapper: ModelWrapper, max_new_tokens: int):
    """ModelWrapper.generate_batch を runner.GenerateFn に適合させる."""

    def generate_fn(prompts: list[str]) -> list[str]:
        results = wrapper.generate_batch(
            prompts, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False
        )
        return [r.generated_text for r in results]

    return generate_fn


def build_reasoning_generate_fn(wrapper: ModelWrapper, max_new_tokens: int):
    """R1蒸留系の teacher-forcing 生成関数.

    ModelWrapper.generate_batch は add_special_tokens=True + 文字列スライスで
    継続テキストを取り出すが、R1 はチャットテンプレートが BOS を含み特殊トークン
    (<｜Assistant｜> 等) が skip されると文字位置がずれる。そのため専用実装で
    (a) add_special_tokens=False でトークナイズ、(b) 新規トークン ID のみを
    skip_special_tokens=True でデコードして答えスパンを得る。greedy。
    """
    tok = wrapper.tokenizer

    @torch.no_grad()
    def generate_fn(prompts: list[str]) -> list[str]:
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(wrapper.device)
        attention_mask = enc["attention_mask"].to(wrapper.device)
        input_len = input_ids.shape[1]
        output_ids = wrapper.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        spans: list[str] = []
        for i in range(len(prompts)):
            gen = output_ids[i, input_len:].tolist()
            while gen and gen[-1] == tok.pad_token_id:
                gen.pop()
            spans.append(tok.decode(gen, skip_special_tokens=True))
        return spans

    return generate_fn


@torch.no_grad()
def dump_divergence_for_pair(
    wrapper: ModelWrapper,
    pair: PairRecord,
    trigger_pattern: str | None,
    rank_threshold: int,
    precision_k: int,
    n_shuffles: int,
    dedup_same_answer_triggers: bool = False,
    prompt_builder=None,
    truncator=None,
    add_special_tokens: bool = True,
) -> dict | None:
    """A セル / C セル (DE 条件) の forward で位置別 divergence を計算する.

    clean run = (clean質問, clean CoT), typo run = (typo質問, clean CoT)。
    CoT は同一文字列なので位置は 1:1 対応 (質問長差のオフセットのみ)。

    R1蒸留系はチャットテンプレート prompt_builder / <think> 対応 truncator を
    注入し、tokenize は add_special_tokens=False (テンプレートが BOS を内包)。
    """
    cells = build_cell_inputs(
        pair,
        trigger_pattern=trigger_pattern,
        dedup_same_answer_triggers=dedup_same_answer_triggers,
        prompt_builder=prompt_builder,
        truncator=truncator,
    )
    tok = wrapper.tokenizer

    ids_full: dict[str, list[int]] = {}
    prompt_lens: dict[str, int] = {}
    for run, cell in (("clean", "A"), ("typo", "C")):
        prompt_ids = tok(
            cells.prompts[cell], return_tensors=None, add_special_tokens=add_special_tokens
        )["input_ids"]
        full_ids = tok(
            cells.full_input(cell), return_tensors=None, add_special_tokens=add_special_tokens
        )["input_ids"]
        ids_full[run] = full_ids
        prompt_lens[run] = len(prompt_ids)

    aligned = align_cot_targets(
        ids_full["clean"], prompt_lens["clean"], ids_full["typo"], prompt_lens["typo"]
    )
    if not aligned.ok or len(aligned.cot_ids) < 2:
        return {
            "sample_id": pair.sample_id,
            "ok": False,
            "reason": "token_alignment_mismatch" if not aligned.ok else "cot_too_short",
        }

    t_len = len(aligned.cot_ids)
    logits_by_run = {}
    for run, start in (("clean", aligned.start_clean), ("typo", aligned.start_typo)):
        input_ids = torch.tensor([ids_full[run]], device=wrapper.device)
        out = wrapper.model(input_ids=input_ids, use_cache=False)
        # 位置 start-1+j の logits がトークン start+j (= cot_ids[j]) を予測
        logits_by_run[run] = out.logits[0, start - 1 : start - 1 + t_len, :]

    profiles = positionwise_divergence(
        logits_by_run["clean"], logits_by_run["typo"], aligned.cot_ids
    )
    del logits_by_run

    onset = divergence_onset(profiles.rank_typo, threshold=rank_threshold)

    cot_tokens = [tok.decode([tid]) for tid in aligned.cot_ids]
    kl_words = [(w.strip() or w, v) for w, v in zip(cot_tokens, profiles.kl, strict=True)]
    rc_words = [
        (d["word"], float(d["score"]))
        for d in pair.extra.get("rc_top_words", [])
        if d.get("word")
    ]

    precision = None
    null = None
    if rc_words:
        precision = precision_at_k(kl_words, rc_words, k=precision_k)
        null_res = shuffle_null_precision(kl_words, rc_words, k=precision_k, n_shuffles=n_shuffles)
        null = {
            "null_mean": null_res.null_mean,
            "null_std": null_res.null_std,
            "p_value": null_res.p_value,
        }

    return {
        "sample_id": pair.sample_id,
        "ok": True,
        "n_positions": t_len,
        "onset": onset,
        "kl_sum": float(sum(profiles.kl)),
        "kl_max": float(max(profiles.kl)) if profiles.kl else None,
        "precision_at_k": precision,
        "null": null,
        "kl": profiles.kl,
        "logp_clean": profiles.logp_clean,
        "logp_typo": profiles.logp_typo,
        "rank_clean": profiles.rank_clean,
        "rank_typo": profiles.rank_typo,
        "tokens": cot_tokens,
    }


def _count_reasons(outcomes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        for r in o.exclude_reasons:
            counts[r] = counts.get(r, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reasoning = args.reasoning or is_reasoning_model(args.model)
    if args.max_new_tokens is None:
        args.max_new_tokens = 64 if reasoning else 16
    logger.info(
        "mode=%s, max_new_tokens=%d",
        "reasoning (R1)" if reasoning else "base",
        args.max_new_tokens,
    )

    pairs = load_pair_records(
        args.baseline_dir,
        args.perturbed_dir,
        clean_correct_only=args.clean_correct_only,
        limit=args.n,
        start=args.start,
    )
    logger.info("PairRecord %d 件をロード", len(pairs))

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(model_name=args.model, device=device, dtype=dtype)
    _ = wrapper.model  # ロード

    # reasoning モード: R1蒸留系の注入部品 (チャットテンプレート・<think> 切断・
    # 抽出チェーン) を組み立てる。基底モデルは None のまま従来挙動。
    prompt_builder = None
    truncator = None
    extract_fn = None
    dedup = args.dedup_same_answer_triggers
    add_special_tokens = True
    if reasoning:
        prompt_builder = make_reasoning_prompt_builder(wrapper.tokenizer)
        truncator = truncate_reasoning_cot
        extract_fn = make_reasoning_extract_fn(create_extractor(args.benchmark))
        dedup = True  # R1 の反復宣言を良性の重複として扱う (感度分析で除外込みも報告)
        add_special_tokens = False  # チャットテンプレートが BOS を内包
        generate_fn = build_reasoning_generate_fn(wrapper, args.max_new_tokens)
    else:
        generate_fn = build_generate_fn(wrapper, args.max_new_tokens)

    logger.info("4 セル生成を開始 (batch_size=%d)", args.batch_size)
    outcomes = run_cells(
        pairs,
        generate_fn,
        batch_size=args.batch_size,
        trigger_pattern=args.trigger_pattern,
        dedup_same_answer_triggers=dedup,
        prompt_builder=prompt_builder,
        truncator=truncator,
        extract_fn=extract_fn,
        strip_conclusion_mode=args.strip_conclusion_mode,
    )

    table = flip_table(outcomes)
    cis = bootstrap_flip_cis(outcomes)
    glmm = glmm_decomposition(outcomes)

    te_mismatch_ids = [o.sample_id for o in outcomes if o.te_match is False]

    with open(out_dir / "outcomes.json", "w", encoding="utf-8") as f:
        json.dump([asdict(o) for o in outcomes], f, ensure_ascii=False, indent=1)

    divergence_summary = None
    if args.dump_divergence:
        div_dir = out_dir / "divergence"
        div_dir.mkdir(exist_ok=True)
        div_records = []
        outcome_by_id = {o.sample_id: o for o in outcomes}
        n_attempted = 0
        for pair in tqdm(pairs, desc="divergence"):
            o = outcome_by_id[pair.sample_id]
            if o.exclude:
                continue
            n_attempted += 1
            rec = dump_divergence_for_pair(
                wrapper,
                pair,
                args.trigger_pattern,
                args.divergence_rank_threshold,
                args.precision_k,
                args.n_shuffles,
                dedup_same_answer_triggers=dedup,
                prompt_builder=prompt_builder,
                truncator=truncator,
                add_special_tokens=add_special_tokens,
            )
            if rec is None:
                continue
            with open(div_dir / f"{pair.sample_id}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            if rec.get("ok"):
                rec_small = {
                    k: v
                    for k, v in rec.items()
                    if k
                    not in ("kl", "logp_clean", "logp_typo", "rank_clean", "rank_typo", "tokens")
                }
                rec_small["te_flip"] = o.answers["B"].strip() != o.answers["A"].strip()
                div_records.append(rec_small)

        n_ok = len(div_records)
        flip_recs = [r for r in div_records if r["te_flip"]]
        noflip_recs = [r for r in div_records if not r["te_flip"]]

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        divergence_summary = {
            "n_attempted": n_attempted,
            "n_ok": n_ok,
            "n_alignment_failed": n_attempted - n_ok,
            "mean_kl_sum": _mean([r["kl_sum"] for r in div_records]),
            "mean_precision_at_k": _mean([r["precision_at_k"] for r in div_records]),
            "mean_null_mean": _mean([r["null"]["null_mean"] for r in div_records if r["null"]]),
            "onset_rate": _mean([1 if r["onset"] is not None else 0 for r in div_records]),
            "flip_group": {
                "n": len(flip_recs),
                "mean_onset": _mean([r["onset"] for r in flip_recs]),
                "mean_kl_sum": _mean([r["kl_sum"] for r in flip_recs]),
            },
            "noflip_group": {
                "n": len(noflip_recs),
                "mean_onset": _mean([r["onset"] for r in noflip_recs]),
                "mean_kl_sum": _mean([r["kl_sum"] for r in noflip_recs]),
            },
        }

    summary = {
        "config": {
            "model": args.model,
            "benchmark": args.benchmark,
            "baseline_dir": str(args.baseline_dir),
            "perturbed_dir": str(args.perturbed_dir),
            "n": args.n,
            "start": args.start,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "clean_correct_only": args.clean_correct_only,
            "dump_divergence": args.dump_divergence,
            "reasoning": reasoning,
            "dedup_same_answer_triggers": dedup,
            "strip_conclusion_mode": args.strip_conclusion_mode,
            "timestamp": datetime.now().isoformat(),
        },
        "flip_table": table,
        "bootstrap_ci": {k: list(v) for k, v in cis.items()},
        "glmm": glmm,
        "te_mismatch_sample_ids": te_mismatch_ids,
        "exclusion_reasons": _count_reasons(outcomes),
        "divergence": divergence_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== flip table ===")
    logger.info(json.dumps(table, ensure_ascii=False, indent=1))
    logger.info("TE match rate: %s", table["te_match_rate"])
    if divergence_summary:
        logger.info("=== divergence ===")
        logger.info(json.dumps(divergence_summary, ensure_ascii=False, indent=1))
    logger.info("保存先: %s", out_dir)


if __name__ == "__main__":
    main()

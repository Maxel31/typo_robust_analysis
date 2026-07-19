#!/usr/bin/env python3
"""実験7: byte-identical 復元サンプルの within-run flip 正式検証.

校正済みデータセット (data/exp7/corrected/{model}_{bench}_k4_{mode}) から
「校正後プロンプトが clean プロンプトとバイト同一」のサンプルのみを抽出し、
clean 行と校正後行を同一バッチ (greedy, temperature=0.0) で生成して
答えの一致率を測る。本番評価生成のクロスラン比較に乗る再現性ノイズを排した
「byte-identical 復元 → flip 0%」の正式測定。

使用例:
  # CPU スキャン: 全設定の byte-identical 集合規模を確認
  uv run python scripts/exp7/within_run_flip.py --scan

  # 1 設定の within-run 生成 (1 シャード = 1 設定 × 1 校正器、GPU ヘルパー経由)
  bash tmp/gpu-locks/run_with_gpu.sh uv run python \
    scripts/exp7/within_run_flip.py \
    --config gemma-3-4b-it_gsm8k_k4_spellfix --gpu_id 0
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
logger = logging.getLogger("within_run_flip")


def load_config_data(config_dir: Path):
    with open(config_dir / "perturbed_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(config_dir / "restoration_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    flags = {p["sample_id"]: p["fully_restored"] for p in stats["per_sample"]}
    return data["metadata"], data["samples"], flags


def scan(corrected_dir: Path, output_dir: Path) -> None:
    """CPU のみ: 全設定の byte-identical 集合規模を index JSON に書き出す."""
    from typo_cot.defense.within_run import select_byte_identical
    from typo_cot.models.prompts import create_prompt_template

    templates: dict[str, object] = {}
    index: dict[str, dict] = {}
    for config_dir in sorted(corrected_dir.iterdir()):
        if not (config_dir / "perturbed_dataset.json").is_file():
            continue
        metadata, samples, flags = load_config_data(config_dir)
        bench = metadata["benchmark"]
        if bench not in templates:
            templates[bench] = create_prompt_template(bench)
        _, st = select_byte_identical(samples, flags, templates[bench], bench)
        index[config_dir.name] = {
            "model": metadata["source_model"],
            "benchmark": bench,
            "corrector_mode": metadata["perturbation_mode"],
            "n_samples": st["n_samples"],
            "n_byte_identical": st["n_byte_identical"],
            "byte_identical_rate": st["n_byte_identical"] / st["n_samples"],
            "n_fully_restored_flag": st["n_fully_restored_flag"],
            "n_flag_mismatch": len(st["flag_mismatch_ids"]),
            "flag_mismatch_ids": st["flag_mismatch_ids"][:20],
        }
        logger.info(
            f"{config_dir.name}: byte-identical "
            f"{st['n_byte_identical']}/{st['n_samples']} "
            f"({st['n_byte_identical'] / st['n_samples']:.1%}), "
            f"flag不一致 {len(st['flag_mismatch_ids'])}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "byte_identical_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"created_at": datetime.now().isoformat(), "configs": index},
            f, ensure_ascii=False, indent=1,
        )
    total = sum(v["n_byte_identical"] for v in index.values())
    logger.info(f"スキャン完了: {len(index)} 設定, byte-identical 総数 {total} → {out}")


def run_generation(args) -> None:
    import torch

    from typo_cot.defense.within_run import (
        aggregate_within_run,
        batch_rows,
        iter_pair_batches,
        select_byte_identical,
    )
    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    config_dir = Path(args.corrected_dir) / args.config
    exp_dir = Path(args.output_dir) / args.config
    out_path = exp_dir / "within_run_results.json"
    if out_path.is_file() and not args.force:
        logger.info(f"既存の結果があるためスキップ: {out_path}")
        print(f"=== WITHIN-RUN {args.config} SKIP (exists) ===")
        return

    metadata, samples, flags = load_config_data(config_dir)
    bench = metadata["benchmark"]
    model = metadata["source_model"]
    template = create_prompt_template(bench)
    extractor = create_extractor(bench)

    pairs, sel_stats = select_byte_identical(samples, flags, template, bench)
    if sel_stats["flag_mismatch_ids"]:
        logger.warning(
            f"fully_restored フラグとプロンプト byte 一致の不一致: "
            f"{len(sel_stats['flag_mismatch_ids'])} 件 (選別はプロンプト一致が正)"
        )
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info(
        f"{args.config}: byte-identical {sel_stats['n_byte_identical']}"
        f"/{sel_stats['n_samples']}, 生成対象 {len(pairs)} ペア "
        f"(model={model}, bench={bench})"
    )

    logger.info(f"モデルをロード: {model} (GPU {args.gpu_id})")
    wrapper = create_model_wrapper(
        model_name=model, gpu_id=args.gpu_id, wrap_for_lxt=False
    )

    exp_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failed: list[str] = []
    t0 = time.time()
    n_batches = 0

    for batch_pairs in iter_pair_batches(pairs, args.pairs_per_batch):
        rows = batch_rows(batch_pairs)
        try:
            gens = wrapper.generate_batch(
                rows, max_new_tokens=args.max_new_tokens, temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"バッチ失敗 (1回リトライ): {exc}")
            try:
                gens = wrapper.generate_batch(
                    rows, max_new_tokens=args.max_new_tokens, temperature=0.0
                )
            except Exception as exc2:  # noqa: BLE001
                logger.error(f"リトライも失敗、ペアを failed 記録: {exc2}")
                failed.extend(p["sample_id"] for p in batch_pairs)
                continue

        for i, p in enumerate(batch_pairs):
            text_clean = gens[2 * i].generated_text
            text_corr = gens[2 * i + 1].generated_text
            ans_clean = extractor.extract(text_clean).extracted_answer
            ans_corr = extractor.extract(text_corr).extracted_answer
            rec = {
                "sample_id": p["sample_id"],
                "ans_clean": ans_clean,
                "ans_corr": ans_corr,
                "flip": ans_corr != ans_clean,
                "correct_clean": extractor.is_correct(
                    ans_clean, p["correct_answer"]
                ),
                "correct_corr": extractor.is_correct(
                    ans_corr, p["correct_answer"]
                ),
                "gen_identical": text_clean == text_corr,
            }
            if rec["flip"]:
                # flip 事例の精査用に生成文を保持 (byte-identical 入力での
                # flip は想定外なので必ず原因分類する)
                rec["gen_text_clean"] = text_clean
                rec["gen_text_corr"] = text_corr
                logger.error(
                    f"{p['sample_id']}: byte-identical 入力で flip "
                    f"(clean={ans_clean!r} corr={ans_corr!r})"
                )
            records.append(rec)

        n_batches += 1
        if n_batches % 25 == 0:
            done = len(records)
            rate = (time.time() - t0) / max(done, 1)
            eta_min = rate * (len(pairs) - done) / 60
            n_flip = sum(r["flip"] for r in records)
            logger.info(
                f"{done}/{len(pairs)} ペア ({rate:.2f}s/pair, "
                f"残り約{eta_min:.0f}分, flip={n_flip})"
            )
            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"partial": True, "records": records}, f, ensure_ascii=False
                )

        del gens, rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = aggregate_within_run(records)
    agg["n_gen_identical"] = sum(r["gen_identical"] for r in records)
    payload = {
        "config": {
            "config_name": args.config,
            "model": model,
            "benchmark": bench,
            "corrector_mode": metadata["perturbation_mode"],
            "corrector_model": metadata.get("corrector_model"),
            "pairs_per_batch": args.pairs_per_batch,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "timestamp": datetime.now().isoformat(),
        },
        "selection_stats": sel_stats,
        "aggregate": agg,
        "failed_sample_ids": failed,
        "records": records,
    }
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    tmp.replace(out_path)
    logger.info(
        f"完了: {out_path} (n={agg['n']}, flip={agg['n_flip']}, "
        f"failed={len(failed)}, {(time.time() - t0) / 60:.0f}分)"
    )
    print(f"=== WITHIN-RUN {args.config} DONE flips={agg['n_flip']}/{agg['n']} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="byte-identical 復元サンプルの within-run flip 検証"
    )
    parser.add_argument("--corrected_dir", default="data/exp7/corrected")
    parser.add_argument("--output_dir", default="results/prod/exp7/within_run")
    parser.add_argument("--scan", action="store_true",
                        help="CPU スキャンのみ (byte-identical 集合規模の index 出力)")
    parser.add_argument("--config", default=None,
                        help="設定名 (例: gemma-3-4b-it_gsm8k_k4_spellfix)")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--pairs_per_batch", type=int, default=2,
                        help="1 バッチのペア数 (行数はこの2倍。本番評価の batch4 相当=2)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N ペアのみ (スモーク用)")
    parser.add_argument("--force", action="store_true",
                        help="既存結果があっても再実行する")
    args = parser.parse_args()

    if args.scan:
        scan(Path(args.corrected_dir), Path(args.output_dir))
        return
    if not args.config:
        parser.error("--scan か --config のいずれかが必要です")
    run_generation(args)


if __name__ == "__main__":
    main()

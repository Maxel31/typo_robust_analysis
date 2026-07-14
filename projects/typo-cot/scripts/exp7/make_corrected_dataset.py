#!/usr/bin/env python3
"""実験7: 校正器ラダーで摂動データセットを校正した「復元後」データセットを作成する.

rebuttal の make_spellfix_dataset.py を3段の校正器 (pyspell / neural / llm) に
一般化したもの。LXT-4 摂動済みデータセットの perturbed_question に校正器を適用し、
既存 Phase 3 (run_inference.py / run_generation_only.py --perturbed_data) 互換形式で
出力する。同時に語単位の復元判定 (restoration_stats.json) を保存する。

校正はテキスト処理でモデル非依存 = ベンチマーク×校正器のジョブで完結する。

使用例:
  uv run python scripts/exp7/make_corrected_dataset.py \
    --input /path/to/perturbed_dataset.json \
    --corrector pyspell \
    --output_dir data/exp7/corrected

  uv run python scripts/exp7/make_corrected_dataset.py \
    --input /path/to/perturbed_dataset.json \
    --corrector llm --llm_model Qwen/Qwen2.5-7B-Instruct --limit 16
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from typo_cot.defense.correctors import LLMCorrector, create_corrector
from typo_cot.defense.restoration import (
    build_reference,
    classify_restoration,
)

# 校正器 -> perturbation_mode 名 (pyspell は rebuttal 互換で spellfix)
MODE_NAMES = {"pyspell": "spellfix", "neural": "neuralfix", "llm": "llmfix"}


def main() -> None:
    parser = argparse.ArgumentParser(description="校正済みデータセット作成 (実験7)")
    parser.add_argument("--input", type=str, required=True,
                        help="摂動済み perturbed_dataset.json のパス")
    parser.add_argument("--corrector", type=str, required=True,
                        choices=["pyspell", "neural", "llm"])
    parser.add_argument("--output_dir", type=str, default="data/exp7/corrected")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--neural_model", type=str, default="ai-forever/T5-large-spell")
    parser.add_argument("--neural_prefix", type=str, default="grammar: ")
    parser.add_argument("--device", type=str, default=None,
                        help="校正モデルのデバイス (cuda/cpu、None=自動)")
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N サンプルのみ処理 (スモーク用)")
    parser.add_argument("--start", type=int, default=0,
                        help="処理開始サンプル位置 (シャード実行用)")
    parser.add_argument("--shard", action="store_true",
                        help="シャードモード: 出力を out_dir/shards/ に書き、"
                             "merge_corrected_shards.py で結合する")
    args = parser.parse_args()

    if args.shard and args.limit is None:
        parser.error("--shard には --limit (シャードサイズ) が必要です")

    if args.corrector == "pyspell":
        corrector = create_corrector("pyspell")
        corrector_model = "pyspellchecker==0.9.0"
    elif args.corrector == "neural":
        corrector = create_corrector(
            "neural", model_name=args.neural_model,
            prefix=args.neural_prefix, device=args.device,
        )
        corrector_model = args.neural_model
    else:
        corrector = create_corrector(
            "llm", model_name=args.llm_model, device=args.device
        )
        corrector_model = args.llm_model

    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    samples = data["samples"]
    shard_start = args.start
    if args.limit is not None:
        shard_end = min(shard_start + args.limit, len(samples))
    else:
        shard_end = len(samples)
    samples = samples[shard_start:shard_end]

    model_short = metadata.get("source_model", "unknown").split("/")[-1]
    benchmark = metadata.get("benchmark", "unknown")
    k = metadata.get("num_perturbations", "unknown")
    mode = MODE_NAMES[args.corrector]
    out_dir = Path(args.output_dir) / f"{model_short}_{benchmark}_k{k}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_samples = []
    per_sample_stats = []
    agg = {
        "n_samples": 0,
        "word_total": 0,          # difflib で同定できた摂動語数
        "word_restored": 0,       # 訂正後に原語へ戻った摂動語数
        "fully_restored": 0,      # 全文一致 (空白正規化)
        "perturbed_words_all_restored": 0,
        "collateral_changes": 0,  # 摂動されていない語への副作用的変更数
        "unalignable": 0,         # 語数変化などで対応付け不能だった摂動語数
        "llm_parse_failures": 0,
    }

    for i, s in enumerate(samples):
        reference = build_reference(s["original_question"], s.get("choices"))
        perturbed_q = s["perturbed_question"]

        llm_meta = None
        if isinstance(corrector, LLMCorrector):
            corrected_q, llm_meta = corrector.correct_with_meta(perturbed_q)
            if llm_meta["parse_failed"]:
                agg["llm_parse_failures"] += 1
        else:
            corrected_q = corrector.correct(perturbed_q)

        r = classify_restoration(reference, perturbed_q, corrected_q)

        agg["n_samples"] += 1
        agg["word_total"] += r.n_perturbed_words
        agg["word_restored"] += r.n_restored
        agg["unalignable"] += r.n_unalignable
        agg["collateral_changes"] += r.n_collateral
        if r.fully_restored:
            agg["fully_restored"] += 1
        if r.all_perturbed_restored:
            agg["perturbed_words_all_restored"] += 1

        stat = {
            "sample_id": s["sample_id"],
            "n_perturbed_words": r.n_perturbed_words,
            "n_restored": r.n_restored,
            "fully_restored": r.fully_restored,
            "all_perturbed_restored": r.all_perturbed_restored,
            "n_collateral_changes": r.n_collateral,
            "restored_flags": [list(t) for t in r.restored_flags],
        }
        if llm_meta is not None:
            stat["llm_parse_failed"] = llm_meta["parse_failed"]
            stat["llm_n_calls"] = llm_meta["n_calls"]
        per_sample_stats.append(stat)

        new_s = dict(s)
        new_s["perturbed_question"] = corrected_q
        new_samples.append(new_s)

        if (i + 1) % 50 == 0:
            rate = agg["word_restored"] / agg["word_total"] if agg["word_total"] else 0
            print(f"{i + 1}/{len(samples)} 処理済み (語復元率 {rate:.3f})", flush=True)

    if args.shard:
        # シャードモード: 部分結果のみ書き出し (merge_corrected_shards.py で結合)
        shard_dir = out_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "start": shard_start,
            "end": shard_end,
            "corrector": args.corrector,
            "corrector_model": corrector_model,
            "source": str(input_path),
            "created_at": datetime.now().isoformat(),
            "samples": new_samples,
            "per_sample": per_sample_stats,
            "aggregate": agg,
        }
        shard_path = shard_dir / f"{shard_start:05d}_{shard_end:05d}.json"
        tmp = shard_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(shard_path)
        print(f"シャード出力: {shard_path} ({agg['n_samples']} サンプル)")
        return

    new_metadata = dict(metadata)
    new_metadata["perturbation_mode"] = mode
    new_metadata["correction_source"] = str(input_path)
    new_metadata["corrector"] = args.corrector
    new_metadata["corrector_model"] = corrector_model
    new_metadata["created_at"] = datetime.now().isoformat()
    new_metadata["total_samples"] = len(new_samples)

    with open(out_dir / "perturbed_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": new_metadata, "samples": new_samples}, f,
                  ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)

    rates = {
        "word_restoration_rate": agg["word_restored"] / agg["word_total"]
        if agg["word_total"] else 0.0,
        "full_restoration_rate": agg["fully_restored"] / agg["n_samples"]
        if agg["n_samples"] else 0.0,
        "all_perturbed_restored_rate": agg["perturbed_words_all_restored"] / agg["n_samples"]
        if agg["n_samples"] else 0.0,
    }
    with open(out_dir / "restoration_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": agg, "rates": rates, "per_sample": per_sample_stats},
            f, ensure_ascii=False, indent=2,
        )

    print(f"出力: {out_dir}")
    print(json.dumps({"aggregate": agg, "rates": rates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

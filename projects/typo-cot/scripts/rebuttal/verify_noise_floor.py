#!/usr/bin/env python3
"""生成再現性ノイズフロアの独立検証 (spellfix 完全復元サブセットの flip 5.9% の切り分け).

対象: GSM8K で spellfix 後の質問が baseline とバイト同一だった 185 サンプル。

Test A (同一プロセス内決定性): batch_size=1 で各プロンプトを同一プロセス内で
2 回連続生成し、抽出回答が一致するかを確認する。
  - flip しない → 同一プロセス内は決定的。5.9% はバッチ構成/GPU 構成差由来。
  - flip する   → 実行内非決定性 (カーネル非決定 reduction 等) が存在。

Test C (クロス run フロア測定): 同じ 185 件を batch_size=4・シャッフル順で
まとめて生成し (baseline とはバッチ同居も GPU 構成も異なる)、baseline の
抽出回答との flip 率を測る。これが独立に測った再現性ノイズフロアであり、
spellfix 完全復元サブセットの flip 率 (5.9%) と一致するはずである。

使用例:
  uv run --no-sync python scripts/rebuttal/verify_noise_floor.py --gpu_id 2
"""

import argparse
import json
import random
import time
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser(description="ノイズフロア検証")
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument(
        "--baseline_dir", type=str, default="outputs/baseline/gemma-3-4b-it_gsm8k"
    )
    parser.add_argument(
        "--spellfix_analysis", type=str,
        default="outputs/rebuttal/spellfix_analysis_gemma-3-4b-it_gsm8k.json",
    )
    parser.add_argument(
        "--spellfix_dir", type=str,
        default="outputs/rebuttal/perturbed/gemma-3-4b-it_gsm8k_k4_spellfix",
    )
    parser.add_argument("--output", type=str,
                        default="outputs/rebuttal/noise_floor_verification.json")
    parser.add_argument("--limit_a", type=int, default=60,
                        help="Test A のサンプル数上限 (2回生成のため)")
    args = parser.parse_args()

    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    base = {r["sample_id"]: r
            for r in json.load(open(Path(args.baseline_dir) / "results.json"))}
    fixr = {r["sample_id"]: r
            for r in json.load(open(Path(args.spellfix_dir) / "results.json"))}
    per = json.load(open(args.spellfix_analysis))["per_sample"]
    ids = [r["sample_id"] for r in per if r["fully_restored"]]
    # バイト同一のみ (analyze 時に全185件が該当することを確認済みだが再チェック)
    ids = [s for s in ids if base[s]["question"] == fixr[s]["question"]]
    print(f"対象 (バイト同一の完全復元サンプル): {len(ids)} 件")

    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=False
    )
    template = create_prompt_template(args.benchmark)
    extractor = create_extractor(args.benchmark)

    prompts = {s: template.generate(question=base[s]["question"]).get_full_prompt()
               for s in ids}

    result = {"n_target_samples": len(ids)}

    # ---- Test A: 同一プロセス内 2 回連続生成 (batch_size=1) ----
    a_ids = ids[: args.limit_a]
    flips_a = 0
    diffs_a = []
    t0 = time.time()
    for s in a_ids:
        g1 = wrapper.generate(prompts[s], max_new_tokens=512, temperature=0.0)
        g2 = wrapper.generate(prompts[s], max_new_tokens=512, temperature=0.0)
        a1 = extractor.extract(g1.generated_text).extracted_answer
        a2 = extractor.extract(g2.generated_text).extracted_answer
        text_same = g1.generated_text == g2.generated_text
        if a1 != a2:
            flips_a += 1
            diffs_a.append(s)
        elif not text_same:
            diffs_a.append(f"{s} (text differs, answer same)")
    result["test_a_inprocess"] = {
        "n": len(a_ids),
        "answer_flips_run1_vs_run2": flips_a,
        "flip_rate": flips_a / len(a_ids) if a_ids else None,
        "notes": diffs_a[:20],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"Test A: {flips_a}/{len(a_ids)} flips (同一プロセス内 2 回実行)")

    # ---- Test C: クロス run フロア (batch 4, シャッフル順) ----
    order = list(ids)
    random.Random(123).shuffle(order)
    flips_c_vs_base = 0
    flips_c_vs_fix = 0
    answers = {}
    t0 = time.time()
    B = 4
    for i in range(0, len(order), B):
        chunk = order[i : i + B]
        gens = wrapper.generate_batch(
            [prompts[s] for s in chunk], max_new_tokens=512, temperature=0.0
        )
        for s, g in zip(chunk, gens, strict=True):
            answers[s] = extractor.extract(g.generated_text).extracted_answer
    for s in order:
        if answers[s] != base[s]["extracted_answer"]:
            flips_c_vs_base += 1
        if answers[s] != fixr[s]["extracted_answer"]:
            flips_c_vs_fix += 1
    result["test_c_crossrun"] = {
        "n": len(order),
        "flips_vs_baseline": flips_c_vs_base,
        "flip_rate_vs_baseline": flips_c_vs_base / len(order),
        "flips_vs_spellfix_run": flips_c_vs_fix,
        "flip_rate_vs_spellfix_run": flips_c_vs_fix / len(order),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(
        f"Test C: vs baseline {flips_c_vs_base}/{len(order)} "
        f"({flips_c_vs_base/len(order):.3f}), "
        f"vs spellfix run {flips_c_vs_fix}/{len(order)}"
    )

    result["environment"] = {
        "this_run": {"gpu_id": args.gpu_id, "batch_size": B, "order": "shuffled(seed123)"},
        "baseline_run": {"gpu_id": "0,1 (2-GPU device_map)", "batch_size": 4},
        "spellfix_run": {"gpu_id": "3", "batch_size": 4},
        "reference_flip_rate_spellfix_fully_restored": 11 / 185,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"出力: {args.output}")


if __name__ == "__main__":
    main()

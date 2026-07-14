#!/usr/bin/env python3
"""Rebuttal 実験② Option A: within-run 参照での flip 再測定.

各サンプルの 3 条件 (clean / LXT-4 摂動 / スペル訂正後) を 1 バッチ (3 行) で
同時生成し、flip を単一 forward ドメイン内で測定する。
- 比較対象が常に同一バッチ内にあるため、バッチ構成由来のクロス run ノイズが
  混入しない
- 訂正後プロンプトが clean とバイト同一のサンプルは同一バッチ内の同一行となり、
  ビット単位で同一計算 → flip=0 が厳密に成立 (assert で検証・記録)

これにより (ii)(iii) の flip 率がすべて「リバッタル実行内の clean 入力の回答」参照で
統一され、クロス run 再現性ノイズが混入しない。accuracy も同一構成での測定値になる。

使用例 (2 GPU シャード実行):
  uv run --no-sync python scripts/rebuttal/rerun_within_run_reference.py \
    --benchmark gsm8k --gpu_id 0 --shard 0 --num_shards 2
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("within_run")


def build_prompt(template, benchmark, question, choices, subset):
    """run_inference.generate_prompt_for_sample と同一のプロンプト構築."""
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        pr = template.generate(question=question, choices=choices, subject=subset)
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        pr = template.generate(question=question, subject=subset)
    else:
        pr = template.generate(question=question)
    return pr.get_full_prompt()


def main() -> None:
    parser = argparse.ArgumentParser(description="within-run 参照の 3 条件ペア生成")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/rebuttal/within_run")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import create_model_wrapper

    model_short = args.model.split("/")[-1]
    base_path = Path(f"outputs/baseline/{model_short}_{args.benchmark}/results.json")
    pert_path = Path(
        f"datasets/perturbed/{model_short}_{args.benchmark}_k4_with_choices/perturbed_dataset.json"
    )
    fix_path = Path(
        f"datasets/rebuttal/{model_short}_{args.benchmark}_k4_spellfix/perturbed_dataset.json"
    )

    with open(base_path, encoding="utf-8") as f:
        base = {r["sample_id"]: r for r in json.load(f)}
    with open(pert_path, encoding="utf-8") as f:
        pert = {s["sample_id"]: s for s in json.load(f)["samples"]}
    with open(fix_path, encoding="utf-8") as f:
        fix = {s["sample_id"]: s for s in json.load(f)["samples"]}

    ids = sorted(set(base) & set(pert) & set(fix))
    ids = ids[args.shard :: args.num_shards]
    if args.limit:
        ids = ids[: args.limit]
    logger.info(f"シャード {args.shard}/{args.num_shards}: {len(ids)} サンプル")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.benchmark}_shard{args.shard}.json"

    wrapper = create_model_wrapper(
        model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=False
    )
    template = create_prompt_template(args.benchmark)
    extractor = create_extractor(args.benchmark)

    results = []
    reused = 0
    identity_violations = 0
    t0 = time.time()

    for idx, sid in enumerate(ids):
        b, p, x = base[sid], pert[sid], fix[sid]
        clean_prompt = build_prompt(
            template, args.benchmark, b["question"], b.get("choices"), b.get("subset")
        )
        # Phase3 と同様、摂動/訂正後は choices 埋め込み済みのため choices=None
        pert_prompt = build_prompt(
            template, args.benchmark, p["perturbed_question"], None, b.get("subset")
        )
        fix_prompt = build_prompt(
            template, args.benchmark, x["perturbed_question"], None, b.get("subset")
        )

        try:
            # 3 条件を 1 バッチで生成: 比較が同一 forward ドメイン内で完結する。
            # clean と訂正後がバイト同一の場合、同一バッチ内の同一行となり
            # ビット単位で同一計算 → flip=0 が厳密に成立 (assert で検証)
            gens = wrapper.generate_batch(
                [clean_prompt, pert_prompt, fix_prompt],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            ans_clean = extractor.extract(gens[0].generated_text).extracted_answer
            ans_pert = extractor.extract(gens[1].generated_text).extracted_answer
            ans_corr = extractor.extract(gens[2].generated_text).extracted_answer
            corr_reused = fix_prompt == clean_prompt
            if corr_reused:
                reused += 1
                if ans_corr != ans_clean:
                    identity_violations += 1
                    logger.error(
                        f"{sid}: バイト同一入力で回答不一致 (想定外): "
                        f"clean={ans_clean!r} corr={ans_corr!r}"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{sid} 失敗: {exc}")
            continue

        results.append(
            {
                "sample_id": sid,
                "ans_clean": ans_clean,
                "ans_pert": ans_pert,
                "ans_corr": ans_corr,
                "correct_clean": extractor.is_correct(ans_clean, b["correct_answer"]),
                "correct_pert": extractor.is_correct(ans_pert, b["correct_answer"]),
                "correct_corr": extractor.is_correct(ans_corr, b["correct_answer"]),
                "corr_reused_clean": corr_reused,
            }
        )

        if (idx + 1) % 50 == 0:
            rate = (time.time() - t0) / (idx + 1)
            eta_min = rate * (len(ids) - idx - 1) / 60
            fp = sum(r["ans_pert"] != r["ans_clean"] for r in results) / len(results)
            fc = sum(r["ans_corr"] != r["ans_clean"] for r in results) / len(results)
            logger.info(
                f"{idx + 1}/{len(ids)} ({rate:.1f}s/sample, 残り約{eta_min:.0f}分, "
                f"flip_pert={fp:.3f}, flip_corr={fc:.3f}, reuse={reused})"
            )
            tmp = out_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False)
            tmp.replace(out_path)

    payload = {
        "config": {**vars(args), "timestamp": datetime.now().isoformat(),
                   "reused_clean_generation": reused,
                   "identity_violations": identity_violations},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    logger.info(
        f"完了: {out_path} (n={len(results)}, 再利用={reused}, "
        f"{(time.time() - t0) / 60:.0f}分)"
    )
    print(f"=== WITHIN-RUN {args.benchmark} SHARD {args.shard} DONE ===")


if __name__ == "__main__":
    main()

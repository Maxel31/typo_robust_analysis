#!/usr/bin/env python3
"""実験7 スモーク②③: 校正→復元判定→評価生成の1周 (n=16〜32).

アーカイブの LXT-4 摂動データセットから先頭 n サンプルを取り、
1) 校正器 (neural / llm / pyspell) で perturbed_question を校正
2) 語単位の復元判定 (fully_restored = byte-identical 検算対象)
3) 評価モデルで clean 質問と校正後質問の両方を greedy 生成 (batch_size=1)
4) accuracy (clean / corrected)・flip (対 clean・同一ラン内)・
   byte-identical 集合の flip 0% 検算
を1パスで実行し、結果 JSON を保存する。

LLM 校正器 (Qwen2.5-7B-Instruct) は評価モデル (Gemma) と別家族。
GPU ヘルパー経由での実行を想定:
  bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp7/smoke_correct_and_eval.py \
    --corrector llm --n 16 --benchmark gsm8k \
    --input /path/to/perturbed_dataset.json \
    --eval_model google/gemma-3-4b-it --output results/smoke/smoke_cycle_llm.json
"""

import argparse
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp7_smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description="実験7 スモーク: 校正→復元判定→評価1周")
    parser.add_argument("--input", required=True,
                        help="アーカイブの LXT-4 摂動 perturbed_dataset.json")
    parser.add_argument("--corrector", required=True,
                        choices=["pyspell", "neural", "llm"])
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--eval_model", default="google/gemma-3-4b-it")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--neural_model", default="ai-forever/T5-large-spell")
    parser.add_argument("--neural_prefix", default="grammar: ")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    from typo_cot.defense.correctors import LLMCorrector, create_corrector
    from typo_cot.defense.restoration import build_reference, classify_restoration
    from typo_cot.evaluation.extractor import create_extractor
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import ModelWrapper

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"][: args.n]
    logger.info(f"サンプル数: {len(samples)} (corrector={args.corrector})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) 校正 ---
    if args.corrector == "pyspell":
        corrector = create_corrector("pyspell")
        corrector_model = "pyspellchecker==0.9.0"
    elif args.corrector == "neural":
        corrector = create_corrector(
            "neural", model_name=args.neural_model, prefix=args.neural_prefix
        )
        corrector_model = args.neural_model
    else:
        corrector = create_corrector("llm", model_name=args.llm_model)
        corrector_model = args.llm_model

    records = []
    for s in samples:
        reference = build_reference(s["original_question"], s.get("choices"))
        perturbed = s["perturbed_question"]
        meta = None
        if isinstance(corrector, LLMCorrector):
            corrected, meta = corrector.correct_with_meta(perturbed)
        else:
            corrected = corrector.correct(perturbed)
        r = classify_restoration(reference, perturbed, corrected)
        records.append(
            {
                "sample_id": s["sample_id"],
                "reference": reference,
                "perturbed": perturbed,
                "corrected": corrected,
                "correct_answer": s["correct_answer"],
                "n_perturbed_words": r.n_perturbed_words,
                "n_restored": r.n_restored,
                "fully_restored": r.fully_restored,
                "all_perturbed_restored": r.all_perturbed_restored,
                "n_collateral": r.n_collateral,
                "llm_parse_failed": meta["parse_failed"] if meta else None,
            }
        )

    # 校正モデルの GPU メモリを解放 (評価モデルと同居させない)
    del corrector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_words = sum(r["n_perturbed_words"] for r in records)
    n_restored = sum(r["n_restored"] for r in records)
    n_full = sum(r["fully_restored"] for r in records)
    logger.info(
        f"校正完了: 語復元率 {n_restored}/{n_words}, byte-identical {n_full}/{len(records)}"
    )

    # --- 2) 評価生成 (clean と corrected を同一ラン・batch_size=1・greedy) ---
    logger.info(f"評価モデルをロード: {args.eval_model}")
    wrapper = ModelWrapper(model_name=args.eval_model, device=device)
    template = create_prompt_template(args.benchmark)
    extractor = create_extractor(args.benchmark)

    def answer_for(question_text: str, correct_answer: str) -> tuple[str | None, bool]:
        pr = template.generate(question=question_text)
        gen = wrapper.generate(
            pr.get_full_prompt(), max_new_tokens=args.max_new_tokens, temperature=0.0
        )
        ext = extractor.extract(gen.generated_text)
        return ext.extracted_answer, extractor.is_correct(
            ext.extracted_answer, correct_answer
        )

    for i, r in enumerate(records):
        ans_clean, ok_clean = answer_for(r["reference"], r["correct_answer"])
        ans_corr, ok_corr = answer_for(r["corrected"], r["correct_answer"])
        r["answer_clean"] = ans_clean
        r["answer_corrected"] = ans_corr
        r["correct_clean"] = ok_clean
        r["correct_corrected"] = ok_corr
        r["flip_vs_clean"] = ans_corr != ans_clean
        logger.info(
            f"[{i + 1}/{len(records)}] clean={ans_clean} corrected={ans_corr} "
            f"flip={r['flip_vs_clean']} byte_identical={r['fully_restored']}"
        )

    # --- 3) 集計 ---
    byte_identical = [r for r in records if r["fully_restored"]]
    flips_bi = sum(r["flip_vs_clean"] for r in byte_identical)
    summary = {
        "corrector": args.corrector,
        "corrector_model": corrector_model,
        "eval_model": args.eval_model,
        "benchmark": args.benchmark,
        "n": len(records),
        "input": args.input,
        "timestamp": datetime.now().isoformat(),
        "word_restoration_rate": n_restored / n_words if n_words else None,
        "byte_identical_rate": n_full / len(records),
        "collateral_sample_rate": sum(r["n_collateral"] > 0 for r in records) / len(records),
        "llm_parse_failures": sum(bool(r["llm_parse_failed"]) for r in records),
        "accuracy_clean": sum(r["correct_clean"] for r in records) / len(records),
        "accuracy_corrected": sum(r["correct_corrected"] for r in records) / len(records),
        "flip_rate_vs_clean": sum(r["flip_vs_clean"] for r in records) / len(records),
        "byte_identical_n": len(byte_identical),
        "byte_identical_flips": flips_bi,
        "byte_identical_flip_check": "PASS (0 flips)" if flips_bi == 0 else "FAIL",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim = [
        {k: v for k, v in r.items() if k not in ("reference", "perturbed")}
        for r in records
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": slim}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

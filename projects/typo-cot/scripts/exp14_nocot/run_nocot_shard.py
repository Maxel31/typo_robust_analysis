#!/usr/bin/env python3
"""実験14: no-CoT 答えスパン生成 (1 シャード = model×benchmark×condition).

指定した results.json (clean=baseline / typo=perturbed) から質問集合を読み、
no-CoT プロンプトで答えスパンのみを greedy 生成・抽出・採点して records.json に
保存する。CoT を通さない「質問→答えの直接読み出し」を測る。

GPU 実行は必ず run_with_gpu.sh 経由 (CUDA_VISIBLE_DEVICES はヘルパーが設定
するため、このスクリプトでは --gpu_id を setup_device に渡すが、環境変数が
既にあればそちらが優先される: models/wrapper.setup_device)。冪等: 完了済み
sample_id はスキップし、DONE マーカーがあれば即終了。

例:
    bash <...>/run_with_gpu.sh uv run python scripts/exp14_nocot/run_nocot_shard.py \
        --model google/gemma-3-4b-it --benchmark gsm8k --condition clean \
        --source-dir <archive>/outputs/baseline/gemma-3-4b-it_gsm8k \
        --output-dir results/exp14_nocot/gemma-3-4b-it_gsm8k_clean --n 16
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from typo_cot.models.wrapper import ModelWrapper, create_model_wrapper
from typo_cot.nocot.generate import build_nocot_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("nocot_shard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="実験14 no-CoT 答えスパン生成シャード")
    p.add_argument("--model", required=True, choices=ModelWrapper.ALLOWED_MODELS)
    p.add_argument(
        "--benchmark",
        required=True,
        choices=["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math"],
    )
    p.add_argument("--condition", required=True, choices=["clean", "importance", "random"])
    p.add_argument("--source-dir", required=True, help="results.json を含むディレクトリ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--start", type=int, default=0, help="サンプル列の先頭からスキップ")
    p.add_argument("--n", type=int, default=None, help="start から n 件のみ")
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--checkpoint-every", type=int, default=25, help="N バッチ毎に保存")
    return p.parse_args()


def load_samples(source_dir: Path, start: int, n: int | None) -> list[dict]:
    """source results.json から no-CoT 生成用サンプルを構築する.

    clean(baseline)/typo(perturbed) いずれも "question" が対象質問。摂動データは
    question に選択肢が内包され choices=None のため、そのまま渡せば良い。
    """
    with open(source_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    samples = []
    for r in results:
        samples.append(
            {
                "sample_id": r["sample_id"],
                "question": r["question"],
                "choices": r.get("choices"),
                "correct_answer": r["correct_answer"],
                "subset": r.get("subset"),
            }
        )
    # 決定的な順序 (sample_id ソート) にしてから start/n 切り出し
    samples.sort(key=lambda s: s["sample_id"])
    if n is not None:
        samples = samples[start : start + n]
    else:
        samples = samples[start:]
    return samples


def build_token_generate_fn(wrapper: ModelWrapper, max_new_tokens: int):
    """左パディング + 新規トークンのみデコードの堅牢な greedy 生成関数."""
    tok = wrapper.tokenizer

    @torch.no_grad()
    def generate_fn(prompts: list[str]) -> list[str]:
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.json"
    done_path = out_dir / "DONE"
    meta_path = out_dir / "meta.json"

    samples = load_samples(Path(args.source_dir), args.start, args.n)
    total = len(samples)
    logger.info("シャード: %s %s %s (n=%d)", args.model, args.benchmark, args.condition, total)

    # 既存 records を読み込み (冪等・再開)
    records: dict[str, dict] = {}
    if records_path.exists():
        try:
            records = json.load(open(records_path, encoding="utf-8"))
            logger.info("既存 records を読み込み: %d 件", len(records))
        except Exception as e:  # pragma: no cover
            logger.warning("既存 records の読み込みに失敗、最初から: %s", e)
            records = {}

    if done_path.exists() and len(records) >= total:
        logger.info("DONE 済み (%d/%d) — スキップ", len(records), total)
        return

    todo = [s for s in samples if s["sample_id"] not in records]
    logger.info("未処理: %d 件", len(todo))

    if todo:
        from typo_cot.evaluation.extractor import create_extractor

        extractor = create_extractor(args.benchmark)
        wrapper = create_model_wrapper(
            model_name=args.model, gpu_id=args.gpu_id, wrap_for_lxt=False
        )
        gen_fn = build_token_generate_fn(wrapper, args.max_new_tokens)

        prompts = [build_nocot_prompt(s, args.benchmark) for s in todo]
        bs = args.batch_size
        n_batches = (len(todo) + bs - 1) // bs
        for bi in range(n_batches):
            chunk = todo[bi * bs : (bi + 1) * bs]
            chunk_prompts = prompts[bi * bs : (bi + 1) * bs]
            spans = gen_fn(chunk_prompts)
            for s, gen in zip(chunk, spans, strict=True):
                extraction = extractor.extract(gen)
                answer = extraction.extracted_answer
                is_correct = bool(answer) and extractor.is_correct(
                    answer, s["correct_answer"]
                )
                records[s["sample_id"]] = {
                    "answer": answer,
                    "is_correct": is_correct,
                    "generated": gen,
                    "extraction_method": extraction.extraction_method,
                }
            if (bi + 1) % args.checkpoint_every == 0 or bi == n_batches - 1:
                json.dump(records, open(records_path, "w", encoding="utf-8"), ensure_ascii=False)
                logger.info("進捗 %d/%d バッチ (%d/%d 件)", bi + 1, n_batches, len(records), total)

    # 最終保存
    json.dump(records, open(records_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    n_correct = sum(1 for r in records.values() if r["is_correct"])
    meta = {
        "model": args.model,
        "benchmark": args.benchmark,
        "condition": args.condition,
        "source_dir": str(args.source_dir),
        "start": args.start,
        "n": args.n,
        "total": total,
        "n_records": len(records),
        "n_correct": n_correct,
        "accuracy": n_correct / len(records) if records else None,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    done_path.write_text(datetime.now().isoformat())
    logger.info(
        "完了: %d 件 acc=%.1f%%", len(records), 100 * (n_correct / len(records) if records else 0)
    )


if __name__ == "__main__":
    main()

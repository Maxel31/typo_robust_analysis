"""実験9: inner lexicon 修復スコアの計測ランナー.

clean/typo 質問対をエンコード (生成不要・forward のみ) し、
摂動語スパン末尾トークンの層別 cos 類似 (修復スコア) と
logit lens の復号ランクを語レベルで計測して JSONL に保存する。

使用例 (GPU ヘルパー経由):
    bash <gpu-locks>/run_with_gpu.sh uv run python scripts/exp9/run_inner_repair.py \
        --model gemma-3-4b-it --benchmarks gsm8k mmlu \
        --conditions lxt4 --n 32 --output-dir results/smoke/exp9

アーカイブ (configs/paths.yaml: jsai2026_root) は読み取り専用。
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from typo_cot.repair.archive_access import RepairInputRecord, load_condition_records
from typo_cot.repair.features import first_token_id, split_increment, zipf_frequency
from typo_cot.repair.lexicon_probe import (
    LogitLens,
    extract_span_hiddens,
    layerwise_cos,
)
from typo_cot.repair.pipeline import HF_MODEL_NAMES, build_prompt_pair, build_word_rows
from typo_cot.repair.span_align import align_typo_spans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp9.run")

_DEFAULT_PATHS = Path(__file__).parent.parent.parent / "configs" / "paths.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="実験9: 修復スコア計測")
    p.add_argument("--model", required=True, help="モデル短縮名 (例: gemma-3-4b-it)")
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu"])
    p.add_argument("--conditions", nargs="+", default=["lxt4", "random4"],
                   choices=["lxt4", "random4"])
    p.add_argument("--n", type=int, default=None, help="条件あたりの最大サンプル数")
    p.add_argument("--archive-root", default=None,
                   help="アーカイブルート (省略時 configs/paths.yaml の jsai2026_root)")
    p.add_argument("--output-dir", default="results/exp9")
    p.add_argument("--lens-top-k", type=int, default=5)
    p.add_argument("--clean-correct-only", action="store_true",
                   help="clean 正解サンプルに限定 (主推定量の規約)")
    p.add_argument("--dry-run", action="store_true",
                   help="forward を行わず整列成功率のみ検証 (CPU で完結)")
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    return p.parse_args()


def _archive_root(args: argparse.Namespace) -> Path:
    if args.archive_root:
        return Path(args.archive_root)
    with open(_DEFAULT_PATHS) as f:
        paths = yaml.safe_load(f)
    return Path(paths["jsai2026_root"])


def _iter_records(
    root: Path, model: str, benchmark: str, condition: str, args: argparse.Namespace
) -> list[RepairInputRecord]:
    records = load_condition_records(root, model, benchmark, condition)
    records = [r for r in records if r.span_extract_ok and r.flip is not None]
    if args.clean_correct_only:
        records = [r for r in records if r.clean_correct]
    if args.n is not None:
        records = records[: args.n]
    return records


def run_dry(root: Path, args: argparse.Namespace) -> None:
    """forward なしで整列成功率を検証する (CPU)."""
    report: dict = {}
    for benchmark in args.benchmarks:
        for condition in args.conditions:
            records = _iter_records(root, args.model, benchmark, condition, args)
            n_tokens = 0
            n_aligned = 0
            for rec in records:
                clean_prompt, typo_prompt = build_prompt_pair(rec)
                spans = align_typo_spans(clean_prompt, typo_prompt, rec.perturbed_tokens)
                n_tokens += len(rec.perturbed_tokens)
                n_aligned += len(spans)
            key = f"{benchmark}/{condition}"
            report[key] = {
                "n_samples": len(records),
                "n_perturbed_tokens": n_tokens,
                "n_aligned_spans": n_aligned,
                "align_rate": round(n_aligned / n_tokens, 4) if n_tokens else None,
            }
            logger.info("[dry] %s: %s", key, report[key])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"dry_run_{args.model}.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def sanity_clean_pair(model, tokenizer, prompt: str, span: tuple[int, int]) -> dict:
    """サニティ (a): 同一 clean プロンプトを 2 回 forward し cos≈1 を確認する."""
    h1, _ = extract_span_hiddens(model, tokenizer, prompt, [span])
    h2, _ = extract_span_hiddens(model, tokenizer, prompt, [span])
    cos = layerwise_cos(h1[:, 0], h2[:, 0])
    return {
        "min_cos": float(cos.min().item()),
        "mean_cos": float(cos.mean().item()),
        "pass": bool(cos.min().item() > 0.999),
    }


def main() -> None:
    args = parse_args()
    root = _archive_root(args)
    logger.info("アーカイブ: %s (読み取り専用)", root)

    if args.dry_run:
        run_dry(root, args)
        return

    hf_name = HF_MODEL_NAMES[args.model]
    logger.info("モデルロード: %s (local_files_only=%s)", hf_name, args.local_files_only)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name, local_files_only=args.local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=args.local_files_only,
    )
    model.eval()
    lens = LogitLens.from_model(model)
    logger.info("logit lens: softcap=%s", lens.softcap)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in args.benchmarks:
        for condition in args.conditions:
            t0 = time.time()
            records = _iter_records(root, args.model, benchmark, condition, args)
            logger.info("%s/%s: %d サンプル", benchmark, condition, len(records))

            all_rows: list[dict] = []
            n_skipped = 0
            sanity: dict | None = None
            for rec in records:
                clean_prompt, typo_prompt = build_prompt_pair(rec)
                spans = align_typo_spans(clean_prompt, typo_prompt, rec.perturbed_tokens)
                if not spans:
                    n_skipped += 1
                    continue
                clean_spans = [(s.clean_start, s.clean_end) for s in spans]
                typo_spans = [(s.typo_start, s.typo_end) for s in spans]
                clean_h, _ = extract_span_hiddens(model, tokenizer, clean_prompt, clean_spans)
                typo_h, _ = extract_span_hiddens(model, tokenizer, typo_prompt, typo_spans)
                cos_curves = layerwise_cos(clean_h, typo_h)  # [L+1, n]

                typo_ranks: list[list[int] | None] = []
                clean_ranks: list[list[int] | None] = []
                incs: list[int] = []
                zipfs: list[float] = []
                for j, s in enumerate(spans):
                    target = first_token_id(tokenizer, s.clean_word)
                    if torch.isnan(typo_h[:, j]).any() or torch.isnan(clean_h[:, j]).any():
                        typo_ranks.append(None)
                        clean_ranks.append(None)
                    else:
                        typo_ranks.append(lens.layer_ranks(typo_h[:, j], target))
                        clean_ranks.append(lens.layer_ranks(clean_h[:, j], target))
                    incs.append(split_increment(tokenizer, s.clean_word, s.typo_word))
                    zipfs.append(zipf_frequency(s.clean_word))

                all_rows.extend(
                    build_word_rows(
                        rec, spans, cos_curves, typo_ranks, clean_ranks, incs, zipfs,
                        lens_top_k=args.lens_top_k,
                    )
                )
                if sanity is None:
                    sanity = sanity_clean_pair(model, tokenizer, clean_prompt, clean_spans[0])
                    logger.info("サニティ (clean 同一対 cos): %s", sanity)

            tag = f"{args.model}_{benchmark}_{condition}"
            rows_path = out_dir / f"word_rows_{tag}.jsonl"
            with open(rows_path, "w") as f:
                for row in all_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            flips = [r for r in all_rows if r["flip"]]
            noflips = [r for r in all_rows if not r["flip"]]
            summary = {
                "model": args.model,
                "benchmark": benchmark,
                "condition": condition,
                "n_samples": len(records),
                "n_skipped_no_span": n_skipped,
                "n_word_rows": len(all_rows),
                "n_flip_rows": len(flips),
                "mean_repair_flip": (
                    sum(r["repair_score"] for r in flips) / len(flips) if flips else None
                ),
                "mean_repair_noflip": (
                    sum(r["repair_score"] for r in noflips) / len(noflips)
                    if noflips
                    else None
                ),
                "lens_hit_rate_typo": (
                    sum(1 for r in all_rows if r["lens_first_hit_layer_top5"] is not None)
                    / len(all_rows)
                    if all_rows
                    else None
                ),
                "lens_hit_rate_clean_self": (
                    sum(
                        1
                        for r in all_rows
                        if r["clean_self_first_hit_layer_top5"] is not None
                    )
                    / len(all_rows)
                    if all_rows
                    else None
                ),
                "sanity_clean_pair": sanity,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            with open(out_dir / f"summary_{tag}.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("完了 %s: %s", tag, summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""実験10④: 自然typo分布の B 側摂動データセットを作成する CLI (A/B 設計).

標的語は A 側 (アーカイブの LXT-4 = gemma-3-4b-it_{bench}_k4_importance の
results.json 内 perturbed_tokens) と同一に固定し、編集操作の分布のみ
GitHub Typo Corpus から推定した自然分布 (configs/natural_typo_distribution.json)
に差し替える。

出力: datasets/perturbed/gemma-3-4b-it_{bench}_k4_natural_with_choices/
      (perturbed_dataset.json + config.json; run_inference.py 互換スキーマ)

再現手順 (PYTHONHASHSEED=42 必須: トークン単位シードが hash() 由来):
  PYTHONHASHSEED=42 uv run --no-sync python \
    scripts/exp10_natural_typo/create_datasets.py
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from typo_cot.perturbation.dataset import PerturbedDataset, PerturbedSample
from typo_cot.perturbation.natural_typo import (
    OPERATIONS,
    NaturalTypoDistribution,
    apply_natural_typos_to_targets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS = ["gsm8k", "mmlu"]
MODEL_SHORT = "gemma-3-4b-it"
ARCHIVE = "/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs"
LETTERS = "ABCDEFGHIJ"


def build_text_with_choices(question: str, choices: list[str] | None) -> str:
    """dataset.py と同じ形式で質問文+選択肢テキストを構築."""
    if not choices:
        return question
    options_str = " ".join(
        f"({LETTERS[i]}) {choice}" for i, choice in enumerate(choices)
    )
    return f"{question}\n{options_str}"


def create_for_benchmark(
    bench: str,
    distribution: NaturalTypoDistribution,
    seed: int,
    output_dir: Path,
) -> None:
    baseline_dir = Path(ARCHIVE) / "baseline" / f"{MODEL_SHORT}_{bench}"
    aside_path = (
        Path(ARCHIVE) / "perturbed" / f"{MODEL_SHORT}_{bench}_k4_importance" / "results.json"
    )
    with open(baseline_dir / "results.json", encoding="utf-8") as f:
        baseline_rows = {r["sample_id"]: r for r in json.load(f)}
    with open(aside_path, encoding="utf-8") as f:
        aside_rows = json.load(f)
    logger.info(
        f"[{bench}] baseline {len(baseline_rows)}件 / A側 (LXT-4) {len(aside_rows)}件"
    )

    samples: list[PerturbedSample] = []
    all_warnings: list[str] = []
    n_targets = 0
    n_applied = 0

    for aside in aside_rows:
        sample_id = aside["sample_id"]
        base = baseline_rows.get(sample_id)
        if base is None:
            all_warnings.append(f"{sample_id}: baseline に存在しません")
            continue

        score_path = baseline_dir / "importance_scores" / f"{sample_id}.pt"
        importance = torch.load(score_path, map_location="cpu", weights_only=False)
        offset_mapping = {
            i: (int(s), int(e))
            for i, (s, e) in enumerate(importance.get("offset_mapping", []))
        }
        question_char_start = importance.get("question_char_start", 0)

        question = base["question"]
        choices = base.get("choices")
        text = build_text_with_choices(question, choices)

        targets = aside.get("perturbed_tokens", [])
        n_targets += len(targets)

        perturbed_text, entries, warnings = apply_natural_typos_to_targets(
            text=text,
            targets=targets,
            offset_mapping=offset_mapping,
            text_char_start=question_char_start,
            distribution=distribution,
            seed=seed,
            sample_id=sample_id,
        )
        n_applied += len(entries)
        all_warnings.extend(warnings)

        samples.append(
            PerturbedSample(
                sample_id=sample_id,
                original_question=question,
                perturbed_question=perturbed_text,
                perturbed_tokens=entries,
                choices=choices,
                correct_answer=base["correct_answer"],
                subset=base.get("subset"),
                context=None,
                original_context=None,
                perturbed_choices=None,  # dataset.py と同様、選択肢は本文に含める
            )
        )

    if all_warnings:
        logger.warning(f"[{bench}] 警告 {len(all_warnings)} 件 (先頭10件):")
        for w in all_warnings[:10]:
            logger.warning(f"  {w}")

    op_counter = Counter(pt.perturbation_type for s in samples for pt in s.perturbed_tokens)
    logger.info(
        f"[{bench}] 標的 {n_targets} 件中 {n_applied} 件に適用 / 操作分布: {dict(op_counter)}"
    )

    metadata = {
        "source_dir": str(baseline_dir),
        "source_model": f"google/{MODEL_SHORT}",
        "benchmark": bench,
        "num_perturbations": 4,
        "perturbation_mode": "natural",
        "include_choices": True,
        "seed": seed,
        "total_samples": len(samples),
        "skipped_samples": len(aside_rows) - len(samples),
        "created_at": datetime.now().isoformat(),
        "perturbation_types": list(OPERATIONS),
        "ab_design": {
            "targets_from": str(aside_path),
            "distribution": "configs/natural_typo_distribution.json",
            "distribution_counts": distribution.metadata.get("counts", {}),
            "targets_total": n_targets,
            "targets_applied": n_applied,
            "warnings": len(all_warnings),
        },
    }

    dataset = PerturbedDataset(metadata=metadata, samples=samples)
    dataset_dir = output_dir / f"{MODEL_SHORT}_{bench}_k4_natural_with_choices"
    dataset.save(dataset_dir / "perturbed_dataset.json")
    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"[{bench}] 保存: {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distribution", default="configs/natural_typo_distribution.json"
    )
    parser.add_argument("--output_dir", default="./datasets/perturbed")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "42":
        logger.error("PYTHONHASHSEED=42 を設定してください (再現性のため必須)")
        sys.exit(1)

    distribution = NaturalTypoDistribution.load(args.distribution)
    logger.info(f"自然typo分布を読み込み: {args.distribution}")
    logger.info(f"操作比率: {distribution.op_probs}")

    for bench in args.benchmarks:
        create_for_benchmark(
            bench=bench,
            distribution=distribution,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()

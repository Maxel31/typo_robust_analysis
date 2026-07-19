#!/usr/bin/env python3
"""実験5(双子語統制): Matched-Rnd-4 データセット作成 CLI.

LXT-4 の各標的語に対し 5 変数 (内容/機能語・文字長±1・Zipf頻度ビン・
サブワード分割増分・埋め込み中心性ビン) の層化マッチングで双子語を選び、
LXT と同一の抽選手続きで typo を注入した Matched-Rnd-4 データセットを作る。

出力 (既存 perturbed_dataset.json と完全互換):
  {output_dir}/{model}_{benchmark}_k{k}_matched_rnd/
    perturbed_dataset.json   摂動データセット (perturbation_mode="matched_rnd")
    config.json              メタデータ
    matched_stats.json       SMD バランス表・caliper 緩和率・per-target 記録

再現性: per-token seed と選択 rng は hash() を使うため PYTHONHASHSEED の固定が必要。

使用例:
  PYTHONHASHSEED=42 uv run python scripts/exp5/make_matched_twin_dataset.py \
    --baseline_dir /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/gemma-3-4b-it_gsm8k \
    -k 4 --output_dir data/exp5/matched_rnd
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("make_matched_twin_dataset")


def build_feature_extractor(
    model_name: str,
    seed: int,
    use_embedding: bool,
    embedding_model: str,
):
    """実依存 (HF tokenizer / wordfreq / spaCy / sentence-transformers) を配線する."""
    from typo_cot.perturbation.matched_sampler import (
        FeatureExtractor,
        function_word_class,
        make_spacy_classifier,
    )

    # トークナイザ (分割増分用): まずローカルキャッシュを試す
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        logger.info(f"トークナイザをローカルキャッシュからロード: {model_name}")
    except Exception:  # noqa: BLE001
        logger.info(f"ローカルキャッシュに無いためダウンロード: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Zipf 頻度
    from wordfreq import zipf_frequency

    def zipf_fn(word: str) -> float:
        return zipf_frequency(word, "en") if word else 0.0

    # POS 分類 (spaCy en_core_web_sm、失敗時は機能語リスト)
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        classify_fn = make_spacy_classifier(nlp)
        logger.info("POS 分類: spaCy en_core_web_sm")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"spaCy をロードできないため機能語リストにフォールバック: {exc}")
        classify_fn = function_word_class

    # 埋め込み中心性 (第2優先; 無効化可能)
    embed_fn = None
    if use_embedding:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(embedding_model, device="cpu")
            cache: dict[str, list[float]] = {}

            def embed_fn(text: str) -> list[float]:
                if text not in cache:
                    cache[text] = st_model.encode(
                        text, show_progress_bar=False, convert_to_numpy=True
                    ).tolist()
                return cache[text]

            logger.info(f"埋め込み中心性: {embedding_model}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"sentence-transformers 無効 (中心性ビンをスキップ): {exc}")
            embed_fn = None

    return FeatureExtractor(
        tokenizer=tokenizer,
        zipf_fn=zipf_fn,
        classify_fn=classify_fn,
        embed_fn=embed_fn,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-Rnd-4 摂動データセット作成")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--num_perturbations", "-k", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="data/exp5/matched_rnd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="サンプル数制限 (スモーク用)")
    parser.add_argument(
        "--no_embedding",
        action="store_true",
        help="埋め込み中心性 (第2優先変数) を無効化",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED が未設定です。token_seed / 選択 rng の再現には "
            "PYTHONHASHSEED=42 で実行してください。"
        )

    from typo_cot.perturbation.matched_dataset import MatchedTwinDatasetCreator
    from typo_cot.perturbation.matched_sampler import (
        MatchedTwinSampler,
        compute_smd_table,
    )

    baseline_dir = Path(args.baseline_dir)
    with open(baseline_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    model_name = config.get("model", "unknown")
    model_short = model_name.split("/")[-1]
    benchmark = config.get("benchmark", "unknown")

    extractor = build_feature_extractor(
        model_name=model_name,
        seed=args.seed,
        use_embedding=not args.no_embedding,
        embedding_model=args.embedding_model,
    )
    sampler = MatchedTwinSampler(
        extractor, num_perturbations=args.num_perturbations, seed=args.seed
    )

    creator = MatchedTwinDatasetCreator(
        baseline_dir=baseline_dir,
        num_perturbations=args.num_perturbations,
        sampler=sampler,
        seed=args.seed,
        include_choices=True,
    )
    if args.limit is not None:
        creator.results = creator.results[: args.limit]

    dataset = creator.create()

    dataset_name = f"{model_short}_{benchmark}_k{args.num_perturbations}_matched_rnd"
    dataset_dir = Path(args.output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "perturbed_dataset.json"
    dataset.save(dataset_path)
    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, ensure_ascii=False, indent=2)

    # --- マッチ品質統計 (SMD バランス表 + 緩和率) ---
    records = creator.match_records
    smd_table = compute_smd_table(records)

    # 実際に摂動されたトークンとマッチ双子語の一致率 (perturb() 失敗の充填を検出)
    matched_by_sample: dict[str, set[int]] = {}
    for r in records:
        if r.matched is not None:
            matched_by_sample.setdefault(r.sample_id, set()).add(r.matched.token_index)
    applied_total = 0
    applied_from_matched = 0
    for s in dataset.samples:
        matched_set = matched_by_sample.get(s.sample_id, set())
        for pt in s.perturbed_tokens:
            applied_total += 1
            if pt.token_index in matched_set:
                applied_from_matched += 1

    stats = {
        "dataset": dataset_name,
        "seed": args.seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "num_perturbations": args.num_perturbations,
        "embedding_enabled": not args.no_embedding,
        "created_at": datetime.now().isoformat(),
        "smd_table": smd_table,
        "applied_total": applied_total,
        "applied_from_matched": applied_from_matched,
        "applied_from_matched_rate": (
            applied_from_matched / applied_total if applied_total else 0.0
        ),
        "per_target": [r.to_dict() for r in records],
    }
    with open(dataset_dir / "matched_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"出力: {dataset_path}")
    summary = {k: v for k, v in stats.items() if k != "per_target"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

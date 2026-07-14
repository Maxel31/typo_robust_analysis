#!/usr/bin/env python3
"""推論 + AttnLRP重要度分析スクリプト.

Phase 1（ベースライン推論）およびPhase 3（摂動後推論）で共通して使用する。
質問文に対して推論を行い、AttnLRPで重要度スコアを計算する。

出力:
- 質問文の各単語の重要度（CoT最初のトークンに対する寄与）
- CoT推論過程の各単語の重要度（最終回答に対する寄与）
- PDFヒートマップ（オプション）
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


from typo_cot.data.loader import Sample, create_loader
from typo_cot.evaluation.extractor import create_extractor
from typo_cot.lrp.analyzer import create_analyzer
from typo_cot.models.prompts import create_prompt_template
from typo_cot.models.wrapper import ModelWrapper, create_model_wrapper
from typo_cot.perturbation.dataset import PerturbedDataset
from typo_cot.visualization.heatmap import (
    generate_cot_heatmap,
    generate_question_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_prompt_for_sample(
    sample: Sample,
    template,
    benchmark: str,
) -> tuple:
    """サンプルに対してプロンプトを生成.

    Args:
        sample: サンプルデータ
        template: プロンプトテンプレート
        benchmark: ベンチマーク名

    Returns:
        (prompt_result, full_prompt) のタプル
    """
    if benchmark in ["mmlu", "mmlu_pro", "arc", "commonsense_qa"]:
        prompt_result = template.generate(
            question=sample.question,
            choices=sample.choices,
            subject=sample.subset,
        )
    elif benchmark == "gsm8k":
        prompt_result = template.generate(question=sample.question)
    elif benchmark == "squad_v2":
        prompt_result = template.generate(
            question=sample.question,
            context=sample.context,
        )
    elif benchmark in ["bbh", "math", "strategy_qa"]:
        # 新規追加ベンチマーク: question のみ. BBH は subtask を subject に渡す.
        prompt_result = template.generate(
            question=sample.question,
            subject=sample.subset,
        )
    else:
        prompt_result = template.generate(question=sample.question)

    return prompt_result, prompt_result.get_full_prompt()


def clear_gpu_memory() -> None:
    """GPUメモリを積極的にクリアする."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース."""
    parser = argparse.ArgumentParser(
        description="Phase 1: ベースライン推論",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ModelWrapper.ALLOWED_MODELS,
        help="使用するモデル名",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=[
            "mmlu",
            "mmlu_pro",
            "gsm8k",
            "squad_v2",
            "arc",
            "commonsense_qa",
            "bbh",
            "math",
            "strategy_qa",
        ],
        help="ベンチマーク名",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="サンプル数（MMLU/MMLU-Pro: サブセットごと、SQuAD v2: 全体からランダムサンプリング）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="推論時のバッチサイズ（AttnLRP計算は1サンプルずつ実行）",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="使用するGPU ID（複数の場合はカンマ区切り: '0,1,2'）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/phase1",
        help="結果出力先ディレクトリ",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="保存する上位重要単語数（指定しない場合は全トークン保存）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="生成する最大トークン数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード",
    )
    parser.add_argument(
        "--heatmap_interval",
        type=int,
        default=0,
        help="ヒートマップを生成する間隔（n件に1回）。0の場合は生成しない",
    )
    parser.add_argument(
        "--no_heatmaps",
        action="store_true",
        help="ヒートマップ生成を無効化（--heatmap_interval=0と同等）",
    )
    parser.add_argument(
        "--perturbed_data",
        type=str,
        default=None,
        help="摂動データセットのパス（Phase 3で使用）。指定時は--benchmarkから推論",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="サンプル処理失敗時の最大リトライ回数",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=1.0,
        help="リトライ間の待機時間（秒）",
    )

    return parser.parse_args()


def save_config(
    args: argparse.Namespace, output_dir: Path, perturbed_metadata: dict | None = None
) -> None:
    """実験設定を保存."""
    heatmap_interval = 0 if args.no_heatmaps else args.heatmap_interval
    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "num_samples_per_subset": args.num_samples,
        "batch_size": args.batch_size,
        "gpu_id": args.gpu_id,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "heatmap_interval": heatmap_interval,
        "timestamp": datetime.now().isoformat(),
    }

    # 摂動データセットの情報を追加
    if args.perturbed_data:
        config["perturbed_data"] = str(args.perturbed_data)
        config["phase"] = "phase3"
        if perturbed_metadata:
            config["perturbed_metadata"] = perturbed_metadata
    else:
        config["phase"] = "phase1"

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"設定を保存: {config_path}")


def save_importance_scores(
    sample_id: str,
    importance_result: dict,
    output_dir: Path,
) -> None:
    """重要度スコアを保存."""
    scores_dir = output_dir / "importance_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    scores_path = scores_dir / f"{sample_id}.pt"
    torch.save(importance_result, scores_path)


def get_model_short_name(model_name: str) -> str:
    """モデル名から短縮名を取得.

    Args:
        model_name: フルモデル名（例: "meta-llama/Llama-3.2-1B-Instruct"）

    Returns:
        短縮名（例: "Llama-3.2-1B-Instruct"）
    """
    # "/" が含まれる場合は最後の部分を取得
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # ヒートマップ生成間隔を設定（0の場合は生成しない）
    heatmap_interval = 0 if args.no_heatmaps else args.heatmap_interval

    # モデル名とベンチマーク名から出力サブディレクトリを生成
    model_short = get_model_short_name(args.model)
    experiment_name = f"{model_short}_{args.benchmark}"

    # 摂動データセットの場合は摂動回数とモードも含める
    if args.perturbed_data:
        # 摂動データセットから情報を取得するため、先に読み込む
        perturbed_path = Path(args.perturbed_data)
        if perturbed_path.exists():
            # メタデータから摂動回数とモードを取得
            with open(perturbed_path, encoding="utf-8") as f:
                import json as json_loader

                temp_data = json_loader.load(f)
                metadata = temp_data.get("metadata", {})
                num_perturbations = metadata.get("num_perturbations", "unknown")
                perturbation_mode = metadata.get("perturbation_mode", "importance")
            experiment_name = f"{experiment_name}_k{num_perturbations}_{perturbation_mode}"

        # Phase 3（摂動推論）の場合、デフォルト出力先を outputs/perturbed に変更
        if args.output_dir == "./outputs/phase1":
            args.output_dir = "./outputs/perturbed"

    # 出力ディレクトリを作成（ベースディレクトリ/モデル名_ベンチマーク名/）
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ヒートマップ出力ディレクトリを作成
    heatmap_dir = output_dir / "heatmaps"
    cot_heatmap_dir = output_dir / "heatmaps_cot"
    if heatmap_interval > 0:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        cot_heatmap_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    if heatmap_interval > 0:
        logger.info(f"ヒートマップ生成: 有効（{heatmap_interval}件に1回）")

    # 摂動データセットの読み込み（Phase 3の場合）
    perturbed_dataset: PerturbedDataset | None = None
    perturbed_metadata: dict | None = None
    if args.perturbed_data:
        perturbed_path = Path(args.perturbed_data)
        if not perturbed_path.exists():
            logger.error(f"摂動データセットが見つかりません: {perturbed_path}")
            sys.exit(1)
        perturbed_dataset = PerturbedDataset.load(perturbed_path)
        perturbed_metadata = perturbed_dataset.metadata
        logger.info(f"摂動データセットを読み込み: {len(perturbed_dataset.samples)} サンプル")
        logger.info(f"摂動回数: {perturbed_metadata.get('num_perturbations', 'unknown')}")

    # 設定を保存
    save_config(args, output_dir, perturbed_metadata)

    # データローダーを作成（Phase 1の場合のみ）
    if perturbed_dataset is None:
        logger.info(f"データローダーを作成: {args.benchmark}")
        # ベンチマーク別のデフォルト値を設定
        if args.benchmark in ["mmlu", "mmlu_pro"]:
            samples_per_subset = args.num_samples if args.num_samples is not None else 100
            logger.info(f"サブセットごとのサンプル数: {samples_per_subset}")
        elif args.benchmark == "bbh":
            # BBH は 23 サブタスクなので samples_per_subset を使用
            samples_per_subset = args.num_samples if args.num_samples is not None else 50
            logger.info(f"BBH サブタスクごとのサンプル数: {samples_per_subset}")
        elif args.benchmark in ["squad_v2", "arc", "commonsense_qa", "math", "strategy_qa"]:
            num_samples = args.num_samples
            if num_samples is not None:
                logger.info(f"サンプル数: {num_samples}")
            else:
                logger.info(f"{args.benchmark}: 全件使用")
            samples_per_subset = 50  # MMLU用のデフォルト（使用されない）
        else:
            samples_per_subset = args.num_samples if args.num_samples is not None else 50
            logger.info("全件使用（GSM8Kでは--num_samplesは無視されます）")
        loader = create_loader(
            benchmark=args.benchmark,
            samples_per_subset=samples_per_subset,
            seed=args.seed,
            num_samples=args.num_samples,
        )
    else:
        loader = None

    # モデルラッパーを作成
    logger.info(f"モデルをロード: {args.model}")
    wrapper = create_model_wrapper(
        model_name=args.model,
        gpu_id=args.gpu_id,
        wrap_for_lxt=True,
    )

    # プロンプトテンプレートを作成
    template = create_prompt_template(args.benchmark)

    # AttnLRP分析器を作成
    analyzer = create_analyzer(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        top_k=args.top_k,
        device=wrapper.device,
    )

    # 回答抽出器を作成
    extractor = create_extractor(args.benchmark)

    # データをロード
    logger.info("データをロード中...")
    if perturbed_dataset is not None:
        # 摂動データセットからサンプルを作成
        samples = []
        for ps in perturbed_dataset.samples:
            # 摂動後の質問文・選択肢を使用してSampleオブジェクトを作成
            # perturbed_choicesがNoneの場合、questionに選択肢が含まれている
            # その場合はchoicesをNoneにしてプロンプトテンプレートで選択肢を追加しない
            choices = ps.perturbed_choices  # Noneの場合もそのまま
            sample = Sample(
                sample_id=ps.sample_id,
                question=ps.perturbed_question,  # 摂動後の質問文（選択肢含む場合あり）
                correct_answer=ps.correct_answer,
                subset=ps.subset,
                choices=choices,  # Noneの場合、questionに選択肢が含まれている
                context=ps.context,
            )
            samples.append(sample)
    else:
        samples = loader.load()
    total_samples = len(samples)
    logger.info(f"合計サンプル数: {total_samples}")

    # 結果を格納するリスト
    results_list: list[dict] = []

    # サブセットごとの統計
    subset_stats: dict[str, dict[str, int | float]] = {}
    heatmap_count = 0  # 生成したヒートマップ数

    correct_count = 0
    processed_count = 0

    # SQuAD用のEM/F1累積変数
    total_em_score = 0.0
    total_f1_score = 0.0

    # バッチ処理用にサンプルをグループ化
    batch_size = args.batch_size
    num_batches = (len(samples) + batch_size - 1) // batch_size

    # tqdmで進捗表示
    phase_desc = "Phase 3 推論" if perturbed_dataset else "Phase 1 推論"
    pbar = tqdm(total=len(samples), desc=phase_desc, unit="sample")

    for batch_idx in range(num_batches):
        # バッチ内のサンプルを取得
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # バッチ内の各サンプルのプロンプトを生成
        batch_prompts = []
        batch_prompt_results = []
        for sample in batch_samples:
            prompt_result, full_prompt = generate_prompt_for_sample(
                sample, template, args.benchmark
            )
            batch_prompts.append(full_prompt)
            batch_prompt_results.append(prompt_result)

        # バッチ推論を実行（失敗時は個別処理にフォールバック）
        batch_gen_results = None
        batch_inference_failed = False
        try:
            batch_gen_results = wrapper.generate_batch(
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"バッチ {batch_idx} の推論でエラー、個別処理にフォールバック: {e}")
            batch_inference_failed = True
            clear_gpu_memory()

        # 各サンプルに対してAttnLRP分析を実行（1サンプルずつ）
        for sample_idx, (sample, prompt_result, full_prompt) in enumerate(
            zip(batch_samples, batch_prompt_results, batch_prompts, strict=True)
        ):
            # このサンプルの処理をリトライ付きで実行
            sample_success = False
            last_error = None

            for retry_count in range(args.max_retries + 1):
                try:
                    # 推論結果を取得（バッチから取得 or 個別推論）
                    if batch_gen_results is not None and not batch_inference_failed:
                        gen_result = batch_gen_results[sample_idx]
                    else:
                        # 個別推論を実行
                        gen_result = wrapper.generate(
                            full_prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.0,
                        )
                    # 重要度を分析
                    if args.benchmark == "squad_v2":
                        # SQuAD: 全生成トークンをターゲットとした重要度計算（context + question）
                        combined_importance = analyzer.analyze_squad(
                            prompt=full_prompt,
                            generated_text=gen_result.generated_text,
                            context_char_start=prompt_result.context_start_in_full,
                            context_char_end=prompt_result.context_end_in_full,
                            question_char_start=prompt_result.question_start_in_full,
                            question_char_end=prompt_result.question_end_in_full,
                        )
                    else:
                        # 選択式/数値回答: 回答トークンをターゲットとした重要度計算
                        combined_importance = analyzer.analyze_combined(
                            prompt=full_prompt,
                            generated_text=gen_result.generated_text,
                            question_char_start=prompt_result.question_start_in_full,
                            question_char_end=prompt_result.question_end_in_full,  # 質問文のみ（top-k用）
                            question_with_choices_end=prompt_result.question_with_choices_end,  # 選択肢含む（ヒートマップ用）
                        )
                    # 質問文重要度を主要な結果として使用
                    importance = combined_importance.question_importance
                    cot_importance = combined_importance.cot_importance

                    # 回答を抽出
                    extraction = extractor.extract(gen_result.generated_text)

                    # 正解判定（SQuADはEM/F1、その他はis_correct）
                    if args.benchmark == "squad_v2":
                        scores = extractor.compute_scores(
                            extraction.extracted_answer,
                            sample.correct_answer,
                        )
                        em_score = scores["em"]
                        f1_score = scores["f1"]
                        is_correct = em_score == 1.0  # EMが1.0なら正解
                    else:
                        is_correct = extractor.is_correct(
                            extraction.extracted_answer,
                            sample.correct_answer,
                        )
                        em_score = None
                        f1_score = None

                    if is_correct:
                        correct_count += 1

                    # SQuAD用のEM/F1を累積
                    if (
                        args.benchmark == "squad_v2"
                        and em_score is not None
                        and f1_score is not None
                    ):
                        total_em_score += em_score
                        total_f1_score += f1_score

                    # サブセット統計を更新
                    subset = sample.subset or "default"
                    if subset not in subset_stats:
                        if args.benchmark == "squad_v2":
                            subset_stats[subset] = {
                                "correct": 0,
                                "total": 0,
                                "em_sum": 0.0,
                                "f1_sum": 0.0,
                            }
                        else:
                            subset_stats[subset] = {"correct": 0, "total": 0}
                    subset_stats[subset]["total"] += 1
                    if is_correct:
                        subset_stats[subset]["correct"] += 1
                    if (
                        args.benchmark == "squad_v2"
                        and em_score is not None
                        and f1_score is not None
                    ):
                        subset_stats[subset]["em_sum"] += em_score
                        subset_stats[subset]["f1_sum"] += f1_score

                    # n件に1回PDFヒートマップを生成
                    should_generate_heatmap = (
                        heatmap_interval > 0
                        and (processed_count + 1) % heatmap_interval == 0
                        and importance.tokens is not None
                        and importance.offset_mapping is not None
                    )
                    if should_generate_heatmap:
                        # 入力部分のヒートマップを生成
                        heatmap_path = heatmap_dir / f"{sample.sample_id}.pdf"
                        # SQuADの場合はcontext + questionの範囲
                        if args.benchmark == "squad_v2":
                            heatmap_start = prompt_result.context_start_in_full or 0
                            heatmap_end = prompt_result.question_end_in_full
                        else:
                            # その他のベンチマークは質問文 + 選択肢
                            heatmap_start = prompt_result.question_start_in_full
                            heatmap_end = prompt_result.question_with_choices_end
                        success = generate_question_heatmap(
                            tokens=importance.tokens,
                            relevance=importance.raw_relevance,
                            offset_mapping=importance.offset_mapping,
                            question_char_start=heatmap_start,
                            question_char_end=heatmap_end,
                            output_path=heatmap_path,
                        )
                        if success:
                            heatmap_count += 1
                            if args.benchmark == "squad_v2":
                                logger.info(f"context+questionヒートマップを生成: {heatmap_path}")
                            else:
                                logger.info(f"質問文+選択肢ヒートマップを生成: {heatmap_path}")

                        # CoTヒートマップを生成（回答部分も表示するが重要度は0）
                        if cot_importance is not None and cot_importance.tokens is not None:
                            cot_heatmap_path = cot_heatmap_dir / f"{sample.sample_id}_cot.pdf"
                            # 回答部分も含めてヒートマップを生成
                            answer_end = combined_importance.answer_token_end
                            if answer_end is None:
                                answer_end = combined_importance.cot_token_end
                            cot_success = generate_cot_heatmap(
                                tokens=cot_importance.tokens,
                                relevance=cot_importance.raw_relevance,
                                cot_token_start=combined_importance.cot_token_start,
                                cot_token_end=answer_end,  # 回答部分も含めて表示
                                output_path=cot_heatmap_path,
                            )
                            if cot_success:
                                logger.info(f"CoTヒートマップを生成: {cot_heatmap_path}")

                    # 結果をリストに追加
                    result_data = {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "correct_answer": sample.correct_answer,
                        "choices": sample.choices,  # Phase 2で使用
                        "context": sample.context,  # Phase 2で使用（SQuAD v2）
                        "generated_text": gen_result.generated_text,
                        "extracted_answer": extraction.extracted_answer,
                        "is_correct": is_correct,
                        "subset": subset,
                        "question_top_k_words": [
                            {"word": ws.word, "score": ws.score} for ws in importance.top_k_words
                        ],
                        "cot_top_k_words": [
                            {"word": ws.word, "score": ws.score}
                            for ws in cot_importance.top_k_words
                        ],
                    }
                    # 摂動データセットの場合は元の質問文も保存
                    if perturbed_dataset is not None:
                        # perturbed_datasetから元の質問文を取得
                        original_sample = next(
                            (
                                s
                                for s in perturbed_dataset.samples
                                if s.sample_id == sample.sample_id
                            ),
                            None,
                        )
                        if original_sample:
                            result_data["original_question"] = original_sample.original_question
                            result_data["perturbed_tokens"] = [
                                {
                                    "token_index": pt.token_index,
                                    "original_token": pt.original_token,
                                    "perturbed_token": pt.perturbed_token,
                                    "importance_score": pt.importance_score,
                                    "perturbation_type": pt.perturbation_type,
                                }
                                for pt in original_sample.perturbed_tokens
                            ]
                    # SQuADの場合はEM/F1スコアを追加
                    if (
                        args.benchmark == "squad_v2"
                        and em_score is not None
                        and f1_score is not None
                    ):
                        result_data["em_score"] = em_score
                        result_data["f1_score"] = f1_score
                    results_list.append(result_data)

                    # 重要度スコアを保存（質問文）
                    # raw_relevanceをCPUに移動してGPUメモリを解放
                    importance_data = {
                        "type": "question",
                        "token_scores": importance.token_scores,  # 質問文のみ
                        "token_scores_with_choices": importance.token_scores_with_choices,  # 選択肢含む
                        "word_scores": [
                            {
                                "word": ws.word,
                                "score": ws.score,
                                "token_indices": ws.token_indices,
                            }
                            for ws in importance.word_scores
                        ],
                        "top_k_words": [
                            {"word": ws.word, "score": ws.score, "token_indices": ws.token_indices}
                            for ws in importance.top_k_words
                        ],  # 質問文のみの上位k
                        "top_k_with_choices": [
                            {"word": ws.word, "score": ws.score, "token_indices": ws.token_indices}
                            for ws in (importance.top_k_with_choices or [])
                        ],  # 選択肢含む上位k
                        "raw_relevance": importance.raw_relevance.cpu(),
                        # Phase 2（摂動）用の追加情報
                        "tokens": importance.tokens,
                        "offset_mapping": importance.offset_mapping,
                        "question_char_start": prompt_result.question_start_in_full,
                        "question_char_end": prompt_result.question_end_in_full,
                        "question_with_choices_end": prompt_result.question_with_choices_end,
                    }
                    # SQuAD v2の場合はcontext範囲も保存（Phase 2で使用）
                    if args.benchmark == "squad_v2":
                        importance_data["context_char_start"] = prompt_result.context_start_in_full
                        importance_data["context_char_end"] = prompt_result.context_end_in_full
                    save_importance_scores(sample.sample_id, importance_data, output_dir)

                    # CoT重要度スコアを保存
                    cot_importance_data = {
                        "type": "cot",
                        "token_scores": cot_importance.token_scores,
                        "word_scores": [
                            {
                                "word": ws.word,
                                "score": ws.score,
                                "token_indices": ws.token_indices,
                            }
                            for ws in cot_importance.word_scores
                        ],
                        "raw_relevance": cot_importance.raw_relevance.cpu(),
                        "cot_token_start": combined_importance.cot_token_start,
                        "cot_token_end": combined_importance.cot_token_end,
                    }
                    save_importance_scores(
                        f"{sample.sample_id}_cot", cot_importance_data, output_dir
                    )

                    processed_count += 1

                    # メモリクリーンアップ（メモリリーク防止）
                    del combined_importance, importance, cot_importance
                    del extraction
                    clear_gpu_memory()

                    # 成功したのでリトライループを抜ける
                    sample_success = True
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"サンプル {sample.sample_id} 処理失敗 "
                        f"(試行 {retry_count + 1}/{args.max_retries + 1}): {e}"
                    )
                    clear_gpu_memory()
                    if retry_count < args.max_retries:
                        time.sleep(args.retry_delay)
                    continue

            # リトライループ終了後の処理
            if not sample_success:
                logger.error(
                    f"サンプル {sample.sample_id} は {args.max_retries + 1} 回試行後も失敗: {last_error}"
                )

            # tqdmの進捗表示を更新
            pbar.update(1)
            if processed_count > 0:
                accuracy = correct_count / processed_count
                pbar.set_postfix({"正答率": f"{accuracy:.1%}", "正答": correct_count})

        # バッチ終了後のメモリクリーンアップ
        if batch_gen_results is not None:
            del batch_gen_results
        del batch_prompts, batch_prompt_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    # 結果をJSONファイルに保存
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    logger.info(f"結果を保存: {results_path}")

    # サマリーを保存
    overall_metrics = {
        "accuracy": correct_count / processed_count if processed_count > 0 else 0,
        "total_correct": correct_count,
        "total_samples": processed_count,
    }

    # SQuADの場合はEM/F1スコアを追加
    if args.benchmark == "squad_v2" and processed_count > 0:
        overall_metrics["em_score"] = total_em_score / processed_count
        overall_metrics["f1_score"] = total_f1_score / processed_count

    # サブセットメトリクスを構築
    per_subset_metrics = {}
    for subset, stats in subset_stats.items():
        subset_metric = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        # SQuADの場合はサブセットごとのEM/F1も追加
        if args.benchmark == "squad_v2" and "em_sum" in stats and stats["total"] > 0:
            subset_metric["em_score"] = stats["em_sum"] / stats["total"]
            subset_metric["f1_score"] = stats["f1_sum"] / stats["total"]
        per_subset_metrics[subset] = subset_metric

    summary = {
        "experiment_info": {
            "model": args.model,
            "benchmark": args.benchmark,
            "num_samples_per_subset": args.num_samples,
            "batch_size": args.batch_size,
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "overall_metrics": overall_metrics,
        "per_subset_metrics": per_subset_metrics,
    }

    if heatmap_interval > 0:
        summary["heatmap_interval"] = heatmap_interval
        summary["heatmaps_generated"] = heatmap_count

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"サマリーを保存: {summary_path}")
    if args.benchmark == "squad_v2":
        logger.info(
            f"最終スコア: EM={overall_metrics.get('em_score', 0):.2%}, "
            f"F1={overall_metrics.get('f1_score', 0):.2%}"
        )
    else:
        logger.info(f"最終正答率: {overall_metrics['accuracy']:.2%}")
    if heatmap_interval > 0:
        logger.info(f"ヒートマップ生成数: {heatmap_count}件")
    phase_name = "Phase 3" if perturbed_dataset else "Phase 1"
    logger.info(f"{phase_name} 完了")


if __name__ == "__main__":
    main()

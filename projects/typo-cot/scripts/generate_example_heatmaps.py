#!/usr/bin/env python3
"""具体例サンプルのHeatmap生成スクリプト.

分析結果から抽出された具体例サンプルIDを用いて、
摂動前後のHeatmapを比較生成する。

使用方法:
1. 分析結果からexample_samples.jsonを読み込んで自動生成:
   python scripts/generate_example_heatmaps.py --analysis_dir outputs/analysis

2. 特定のサンプルIDを指定して生成:
   python scripts/generate_example_heatmaps.py \
       --sample_ids mmlu_abstract_algebra_0001 mmlu_abstract_algebra_0002 \
       --baseline_dir outputs/baseline/gemma-3-1b-it_mmlu \
       --perturbed_dir outputs/perturbed/gemma-3-1b-it_mmlu_k4_importance \
       --output_dir outputs/heatmaps

出力:
- 質問文+選択肢Heatmap (Question+Choices → CoT): {sample_id}_question_{before/after}.pdf
  ※ 選択肢部分も含めた重要度ハイライトを表示
- CoT Heatmap (CoT → Answer): {sample_id}_cot_{before/after}.pdf
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch


from typo_cot.visualization.heatmap import (
    generate_cot_heatmap,
    generate_question_heatmap,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_perturbed_dir_name(perturbed_dir: Path) -> dict[str, str] | None:
    """摂動後ディレクトリ名からモデル・データセット・摂動情報を抽出.

    ディレクトリ名のパターン: {model}_{dataset}_k{n}_{type}
    例: gemma-3-1b-it_mmlu_k4_importance

    Args:
        perturbed_dir: 摂動後ディレクトリのパス

    Returns:
        抽出された情報の辞書、パース失敗時はNone
    """
    dir_name = perturbed_dir.name
    # パターン: 末尾が _k{数字}_{importance|random} で終わる
    pattern = r"^(.+)_k(\d+)_(importance|random)$"
    match = re.match(pattern, dir_name)

    if not match:
        return None

    base_name = match.group(1)  # {model}_{dataset}
    k = match.group(2)
    pert_type = match.group(3)

    # base_nameをモデルとデータセットに分割
    # データセット名は既知のリストから推測、または最後の_区切りをデータセットとする
    known_datasets = ["mmlu", "gsm8k", "arc", "hellaswag", "truthfulqa"]
    model = base_name
    dataset = "unknown"

    for ds in known_datasets:
        if base_name.endswith(f"_{ds}"):
            model = base_name[: -(len(ds) + 1)]
            dataset = ds
            break

    return {
        "model": model,
        "dataset": dataset,
        "num_perturbations": k,
        "perturbation_type": pert_type,
    }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース."""
    parser = argparse.ArgumentParser(
        description="具体例サンプルのHeatmap生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 分析結果から自動生成
  python scripts/generate_example_heatmaps.py --analysis_dir outputs/analysis

  # 特定のサンプルIDを指定
  python scripts/generate_example_heatmaps.py \\
      --sample_ids mmlu_abstract_algebra_0001 \\
      --baseline_dir outputs/baseline/gemma-3-1b-it_mmlu \\
      --perturbed_dir outputs/perturbed/gemma-3-1b-it_mmlu_k4_importance
""",
    )

    # 自動モード（analysis_dirから読み込み）
    parser.add_argument(
        "--analysis_dir",
        type=str,
        help="分析結果ディレクトリ（example_samples.jsonを含む）",
    )

    # 手動モード（sample_idsを指定）
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        help="Heatmapを生成するサンプルID（スペース区切り）",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        help="ベースライン（摂動前）の結果ディレクトリ",
    )
    parser.add_argument(
        "--perturbed_dir",
        type=str,
        help="摂動後の結果ディレクトリ",
    )

    # 共通オプション
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/example_heatmaps",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="各タイプから生成する最大サンプル数",
    )
    parser.add_argument(
        "--latex_backend",
        type=str,
        default="pdflatex",
        choices=["pdflatex", "xelatex", "lualatex"],
        help="LaTeXバックエンド",
    )

    return parser.parse_args()


def load_importance_scores(scores_dir: Path, sample_id: str) -> dict | None:
    """重要度スコアを読み込む.

    Args:
        scores_dir: importance_scoresディレクトリ
        sample_id: サンプルID

    Returns:
        重要度スコアデータ（辞書）、見つからない場合はNone
    """
    # 質問文（Question→CoT）のスコア
    question_path = scores_dir / f"{sample_id}.pt"
    # CoT（CoT→Answer）のスコア
    cot_path = scores_dir / f"{sample_id}_cot.pt"

    if not question_path.exists():
        logger.warning(f"質問文スコアが見つかりません: {question_path}")
        return None

    if not cot_path.exists():
        logger.warning(f"CoTスコアが見つかりません: {cot_path}")
        return None

    question_data = torch.load(question_path, map_location="cpu", weights_only=False)
    cot_data = torch.load(cot_path, map_location="cpu", weights_only=False)

    return {
        "question": question_data,
        "cot": cot_data,
    }


def generate_heatmaps_for_sample(
    sample_id: str,
    baseline_dir: Path,
    perturbed_dir: Path,
    output_dir: Path,
    latex_backend: str = "pdflatex",
) -> bool:
    """1つのサンプルに対して摂動前後のHeatmapを生成.

    生成するHeatmap:
    - 質問文+選択肢 → CoT: 質問文と選択肢部分の重要度ハイライト
    - CoT → Answer: CoT推論過程の重要度ハイライト

    Args:
        sample_id: サンプルID
        baseline_dir: ベースライン結果ディレクトリ
        perturbed_dir: 摂動後結果ディレクトリ
        output_dir: 出力ディレクトリ
        latex_backend: LaTeXバックエンド

    Returns:
        成功した場合True
    """
    logger.info(f"Heatmap生成: {sample_id}")

    # 重要度スコアを読み込み
    baseline_scores_dir = baseline_dir / "importance_scores"
    perturbed_scores_dir = perturbed_dir / "importance_scores"

    baseline_data = load_importance_scores(baseline_scores_dir, sample_id)
    perturbed_data = load_importance_scores(perturbed_scores_dir, sample_id)

    if baseline_data is None or perturbed_data is None:
        logger.error(f"スコアデータの読み込みに失敗: {sample_id}")
        return False

    # 出力ディレクトリを作成
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # ========================================
    # 質問文+選択肢Heatmap（Question+Choices → CoT）
    # ========================================
    for phase, data in [("before", baseline_data), ("after", perturbed_data)]:
        q_data = data["question"]

        # 必要なデータを取得（キー名の互換性対応）
        tokens = q_data.get("tokens", [])
        relevance = q_data.get("relevance") or q_data.get("raw_relevance")
        offset_mapping = q_data.get("offset_mapping", [])
        question_char_start = q_data.get("question_char_start", 0)
        question_char_end = q_data.get("question_char_end", len("".join(tokens)))
        # 選択肢を含む終了位置（存在しない場合はquestion_char_endを使用）
        question_with_choices_end = q_data.get(
            "question_with_choices_end", question_char_end
        )

        if relevance is None or len(tokens) == 0:
            logger.warning(f"質問文データが不完全: {sample_id} ({phase})")
            continue

        # テンソルに変換
        if not isinstance(relevance, torch.Tensor):
            relevance = torch.tensor(relevance)

        output_path = sample_output_dir / f"question_{phase}.pdf"

        try:
            # 選択肢を含む範囲でヒートマップを生成
            result = generate_question_heatmap(
                tokens=tokens,
                relevance=relevance,
                offset_mapping=offset_mapping,
                question_char_start=question_char_start,
                question_char_end=question_with_choices_end,  # 選択肢を含む
                output_path=output_path,
                backend=latex_backend,
            )
            if result:
                logger.info(f"  質問文+選択肢Heatmap ({phase}): {output_path}")
            else:
                logger.warning(f"  質問文+選択肢Heatmap ({phase}) 生成失敗")
                success = False
        except Exception as e:
            logger.error(f"  質問文+選択肢Heatmap ({phase}) エラー: {e}")
            success = False

    # ========================================
    # CoT Heatmap（CoT → Answer）
    # ========================================
    for phase, data in [("before", baseline_data), ("after", perturbed_data)]:
        cot_data = data["cot"]
        q_data = data["question"]  # tokensはquestionデータから取得

        # 必要なデータを取得（キー名の互換性対応）
        # CoTデータにtokensがない場合はquestionデータから取得
        tokens = cot_data.get("tokens", []) or q_data.get("tokens", [])
        relevance = cot_data.get("relevance") or cot_data.get("raw_relevance")
        cot_token_start = cot_data.get("cot_token_start", 0)
        cot_token_end = cot_data.get("cot_token_end", len(tokens) - 1)

        if relevance is None or len(tokens) == 0:
            logger.warning(f"CoTデータが不完全: {sample_id} ({phase})")
            continue

        # テンソルに変換
        if not isinstance(relevance, torch.Tensor):
            relevance = torch.tensor(relevance)

        output_path = sample_output_dir / f"cot_{phase}.pdf"

        try:
            result = generate_cot_heatmap(
                tokens=tokens,
                relevance=relevance,
                cot_token_start=cot_token_start,
                cot_token_end=cot_token_end,
                output_path=output_path,
                backend=latex_backend,
            )
            if result:
                logger.info(f"  CoT Heatmap ({phase}): {output_path}")
            else:
                logger.warning(f"  CoT Heatmap ({phase}) 生成失敗")
                success = False
        except Exception as e:
            logger.error(f"  CoT Heatmap ({phase}) エラー: {e}")
            success = False

    return success


def find_analysis_pairs(analysis_dir: Path) -> list[dict]:
    """分析結果からベースライン-摂動ペアを検出.

    Args:
        analysis_dir: 分析結果ディレクトリ

    Returns:
        ペア情報のリスト
    """
    pairs = []

    # {dataset}/{model}/{pert_suffix}/example_samples.json を検索
    for example_file in analysis_dir.rglob("example_samples.json"):
        with open(example_file, encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})

        # ベースラインと摂動後のディレクトリパスを復元
        before_dir = metadata.get("before_dir", "")
        after_dir = metadata.get("after_dir", "")

        if before_dir and after_dir:
            pairs.append(
                {
                    "dataset": metadata.get("dataset", "unknown"),
                    "model": metadata.get("model", "unknown"),
                    "num_perturbations": metadata.get("num_perturbations", 0),
                    "perturbation_type": metadata.get("perturbation_type", "unknown"),
                    "baseline_dir": Path(before_dir),
                    "perturbed_dir": Path(after_dir),
                    "type1_samples": data.get("type1_samples", []),
                    "type2_samples": data.get("type2_samples", []),
                    "example_file": example_file,
                }
            )

    return pairs


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # 引数のバリデーション
    if args.analysis_dir is None and (
        args.sample_ids is None or args.baseline_dir is None or args.perturbed_dir is None
    ):
        logger.error(
            "--analysis_dir または (--sample_ids, --baseline_dir, --perturbed_dir) を指定してください"
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    if args.analysis_dir:
        # 自動モード：分析結果から読み込み
        analysis_dir = Path(args.analysis_dir)
        logger.info(f"分析結果ディレクトリ: {analysis_dir}")

        pairs = find_analysis_pairs(analysis_dir)

        if not pairs:
            logger.error("分析結果が見つかりません")
            sys.exit(1)

        logger.info(f"検出されたペア数: {len(pairs)}")

        total_success = 0
        total_fail = 0

        for pair in pairs:
            logger.info("=" * 60)
            logger.info(f"処理中: {pair['dataset']}/{pair['model']}/k{pair['num_perturbations']}_{pair['perturbation_type']}")

            # 出力サブディレクトリ
            pair_output_dir = (
                output_dir
                / pair["dataset"]
                / pair["model"]
                / f"k{pair['num_perturbations']}_{pair['perturbation_type']}"
            )

            # Type1とType2のサンプルを処理
            for sample_type, sample_ids in [
                ("type1_q_jaccard_low_rougel_low", pair["type1_samples"]),
                ("type2_rougel_high_cot_jaccard_low", pair["type2_samples"]),
            ]:
                if not sample_ids:
                    logger.info(f"  {sample_type}: サンプルなし")
                    continue

                logger.info(f"  {sample_type}: {len(sample_ids)}サンプル")

                for sample_id in sample_ids[: args.max_samples]:
                    type_output_dir = pair_output_dir / sample_type

                    success = generate_heatmaps_for_sample(
                        sample_id=sample_id,
                        baseline_dir=pair["baseline_dir"],
                        perturbed_dir=pair["perturbed_dir"],
                        output_dir=type_output_dir,
                        latex_backend=args.latex_backend,
                    )

                    if success:
                        total_success += 1
                    else:
                        total_fail += 1

        logger.info("=" * 60)
        logger.info(f"完了: 成功 {total_success}, 失敗 {total_fail}")

    else:
        # 手動モード：指定されたサンプルIDを処理
        baseline_dir = Path(args.baseline_dir)
        perturbed_dir = Path(args.perturbed_dir)

        logger.info(f"ベースライン: {baseline_dir}")
        logger.info(f"摂動後: {perturbed_dir}")
        logger.info(f"サンプル数: {len(args.sample_ids)}")

        # 摂動後ディレクトリ名から情報を抽出して出力ディレクトリを構築
        dir_info = parse_perturbed_dir_name(perturbed_dir)
        if dir_info:
            manual_output_dir = (
                output_dir
                / dir_info["dataset"]
                / dir_info["model"]
                / f"k{dir_info['num_perturbations']}_{dir_info['perturbation_type']}"
            )
            logger.info(
                f"出力構造: {dir_info['dataset']}/{dir_info['model']}/"
                f"k{dir_info['num_perturbations']}_{dir_info['perturbation_type']}"
            )
        else:
            # パース失敗時はディレクトリ名をそのまま使用
            manual_output_dir = output_dir / perturbed_dir.name
            logger.warning(
                f"ディレクトリ名のパースに失敗しました。"
                f"出力先: {manual_output_dir}"
            )

        success_count = 0
        fail_count = 0

        for sample_id in args.sample_ids:
            success = generate_heatmaps_for_sample(
                sample_id=sample_id,
                baseline_dir=baseline_dir,
                perturbed_dir=perturbed_dir,
                output_dir=manual_output_dir,
                latex_backend=args.latex_backend,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

        logger.info("=" * 60)
        logger.info(f"完了: 成功 {success_count}, 失敗 {fail_count}")
        logger.info(f"出力ディレクトリ: {manual_output_dir}")


if __name__ == "__main__":
    main()

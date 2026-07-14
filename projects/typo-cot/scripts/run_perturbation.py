#!/usr/bin/env python3
"""Phase 2: 摂動データセット作成スクリプト.

Phase 1の重要度スコアを参照して、重要なキーワードに摂動を適用した
データセットを作成する。

出力:
- perturbed_dataset.json: 摂動後のデータセット
- config.json: メタデータ
"""

import argparse
import logging
import sys
from pathlib import Path


from typo_cot.perturbation.dataset import create_perturbed_dataset

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース."""
    parser = argparse.ArgumentParser(
        description="Phase 2: 摂動データセットを作成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Phase 1の結果ディレクトリ（例: outputs/baseline/gemma-3-4b-it_mmlu）",
    )
    parser.add_argument(
        "--num_perturbations",
        "-k",
        type=int,
        required=True,
        help="摂動回数（重要度上位k個のトークンに摂動を適用）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/perturbed",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード",
    )
    parser.add_argument(
        "--random_perturbation",
        action="store_true",
        help="重要度上位k個を除外してランダムにトークンを選択して摂動",
    )
    parser.add_argument(
        "--bottom_k",
        action="store_true",
        help="重要度下位k個のトークンに摂動を適用（Anti-LRP）",
    )
    parser.add_argument(
        "--include_choices",
        action="store_true",
        default=True,
        help="選択肢も摂動対象に含める（デフォルト: True）",
    )
    parser.add_argument(
        "--no_include_choices",
        action="store_false",
        dest="include_choices",
        help="選択肢を摂動対象から除外",
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # 摂動モードを決定
    if args.random_perturbation:
        perturbation_mode = "ランダム"
    elif args.bottom_k:
        perturbation_mode = "重要度下位（Anti-LRP）"
    else:
        perturbation_mode = "重要度ベース"
    target_scope = "選択肢含む" if args.include_choices else "質問文のみ"

    logger.info("=" * 60)
    logger.info("Phase 2: 摂動データセット作成")
    logger.info("=" * 60)
    logger.info(f"入力ディレクトリ: {args.baseline_dir}")
    logger.info(f"摂動回数: {args.num_perturbations}")
    logger.info(f"摂動モード: {perturbation_mode}")
    logger.info(f"摂動対象: {target_scope}")
    logger.info(f"出力ディレクトリ: {args.output_dir}")
    logger.info(f"シード: {args.seed}")
    logger.info("=" * 60)

    # 入力ディレクトリの存在確認
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        logger.error(f"入力ディレクトリが存在しません: {baseline_dir}")
        sys.exit(1)

    # 摂動データセットを作成
    try:
        dataset_path = create_perturbed_dataset(
            baseline_dir=baseline_dir,
            num_perturbations=args.num_perturbations,
            output_dir=args.output_dir,
            seed=args.seed,
            random_perturbation=args.random_perturbation,
            include_choices=args.include_choices,
            bottom_k_perturbation=args.bottom_k,
        )

        logger.info("=" * 60)
        logger.info("完了")
        logger.info(f"データセット: {dataset_path}")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()

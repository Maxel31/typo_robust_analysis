#!/usr/bin/env python3
"""Appendix分析: トークン特性可視化スクリプト.

摂動前のimportance_scoresデータを使い、質問トークン・CoTトークンの
特性をC→C / C→I パターン別に可視化する。

使用方法:
1. 一括分析（推奨）:
   python scripts/run_appendix_analysis.py --outputs_dir outputs

2. 個別分析:
   python scripts/run_appendix_analysis.py \
       --analysis_dir outputs/analysis/mmlu/Llama-3.2-1B-Instruct/k4_importance \
       --output_dir outputs/appendix_analysis/mmlu/Llama-3.2-1B-Instruct/k4_importance

出力:
- wordcloud_A_{cc,ci}.png: 質問トークンのWordCloud
- wordcloud_B_{cc,ci}.png: CoTトークンのWordCloud
- pos_A_{cc,ci}.png: 質問トークンの品詞分布
- pos_B_{cc,ci}.png: CoTトークンの品詞分布
- position_A_{cc,ci}.png: 質問トークンのポジション散布図
- position_B_{cc,ci}.png: CoTトークンのポジション散布図
- results.json: 分析結果データ
"""

import argparse
import logging
import sys
from pathlib import Path


from typo_cot.analysis.appendix_analyzer import (
    AppendixAnalyzer,
    run_appendix_analysis,
)

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
        description="Appendix分析: トークン特性可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 一括分析（推奨）
  python scripts/run_appendix_analysis.py --outputs_dir outputs

  # top-kを変更
  python scripts/run_appendix_analysis.py --outputs_dir outputs --k 20

  # 個別分析
  python scripts/run_appendix_analysis.py \\
      --analysis_dir outputs/analysis/mmlu/Llama-3.2-1B-Instruct/k4_importance \\
      --output_dir outputs/appendix_analysis/mmlu/Llama-3.2-1B-Instruct/k4_importance
""",
    )

    # 一括実行オプション
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        help="outputsディレクトリ（一括分析時に使用）",
    )

    # 個別実行オプション
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        help="分析結果ディレクトリ（個別分析時に使用）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="出力ディレクトリ（個別分析時に使用）",
    )

    # 共通オプション
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="上位何トークンを使用するか（デフォルト: 10）",
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    if args.outputs_dir:
        # 一括分析モード
        logger.info(f"一括分析モード: {args.outputs_dir}")
        run_appendix_analysis(
            outputs_dir=args.outputs_dir,
            top_k=args.k,
        )
    elif args.analysis_dir:
        # 個別分析モード
        if args.output_dir is None:
            logger.error("個別分析時は --output_dir を指定してください")
            sys.exit(1)

        logger.info(f"個別分析モード: {args.analysis_dir}")
        analyzer = AppendixAnalyzer(
            analysis_dir=args.analysis_dir,
            output_dir=args.output_dir,
            top_k=args.k,
        )
        analyzer.analyze()
    else:
        logger.error("--outputs_dir または --analysis_dir を指定してください")
        sys.exit(1)

    logger.info("Appendix分析が完了しました")


if __name__ == "__main__":
    main()

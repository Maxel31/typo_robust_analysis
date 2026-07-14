#!/usr/bin/env python3
"""Phase 4: 摂動前後の分析スクリプト.

Phase 1（摂動前）とPhase 3（摂動後）の結果を比較し、
モデルの注意分布やCoT推論過程の変化を分析する。

使用方法:
1. 個別分析:
   python scripts/run_analysis.py --before_dir outputs/baseline/model_dataset \
                                   --after_dir outputs/perturbed/model_dataset_k4_importance

2. 一括分析（推奨）:
   python scripts/run_analysis.py --outputs_dir outputs

   この場合、outputs/baseline/ と outputs/perturbed/ を自動検索し、
   baseline-perturbedの全パターンの比較分析を一括実行する。

出力:
- analysis_results.json: 分析結果（パターン別メトリクス、個別サンプル結果）
"""

import argparse
import logging
import re
import sys
from pathlib import Path


from typo_cot.analysis import compute_unified_exclusion, run_analysis

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
        description="Phase 4: 摂動前後の分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 一括分析（推奨）
  python scripts/run_analysis.py --outputs_dir outputs

  # 個別分析
  python scripts/run_analysis.py --before_dir outputs/baseline/Llama-3.2-3B-Instruct_mmlu \\
                                 --after_dir outputs/perturbed/Llama-3.2-3B-Instruct_mmlu_k4_importance
""",
    )

    # 一括実行オプション
    parser.add_argument(
        "--outputs_dir",
        type=str,
        help="出力ルートディレクトリ（baseline/とperturbed/を含む）。"
        "指定すると全パターンを一括分析",
    )

    # 個別実行オプション
    parser.add_argument(
        "--before_dir",
        type=str,
        help="摂動前（Phase 1）の結果ディレクトリ",
    )
    parser.add_argument(
        "--after_dir",
        type=str,
        help="摂動後（Phase 3）の結果ディレクトリ",
    )

    # 共通オプション
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/analysis",
        help="分析結果の出力ディレクトリ",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="集計対象に含めるベンチマーク (例: mmlu mmlu_pro arc). "
             "指定しない場合は全ベンチを処理.",
    )

    return parser.parse_args()


def find_analysis_pairs(outputs_dir: Path) -> list[tuple[Path, Path, str]]:
    """baseline-perturbedのペアを自動検出する.

    Args:
        outputs_dir: 出力ルートディレクトリ（baseline/とperturbed/を含む）

    Returns:
        (baseline_dir, perturbed_dir, pair_name)のリスト
    """
    baseline_dir = outputs_dir / "baseline"
    perturbed_dir = outputs_dir / "perturbed"

    if not baseline_dir.exists():
        raise FileNotFoundError(f"baselineディレクトリが見つかりません: {baseline_dir}")
    if not perturbed_dir.exists():
        raise FileNotFoundError(f"perturbedディレクトリが見つかりません: {perturbed_dir}")

    pairs = []

    # perturbedディレクトリ名のパターン: {model}_{dataset}_k{n}_{type}
    perturbed_pattern = re.compile(r"^(.+)_k(\d+)_(importance|random)$")

    # baselineのサブディレクトリをリストアップ
    baselines = {d.name: d for d in baseline_dir.iterdir() if d.is_dir()}

    # perturbedのサブディレクトリをスキャン
    for perturbed_subdir in perturbed_dir.iterdir():
        if not perturbed_subdir.is_dir():
            continue

        match = perturbed_pattern.match(perturbed_subdir.name)
        if not match:
            continue

        base_name = match.group(1)  # {model}_{dataset}
        k = match.group(2)  # k値
        pert_type = match.group(3)  # importance or random

        # 対応するbaselineを検索
        if base_name in baselines:
            pair_name = f"{base_name}_k{k}_{pert_type}"
            pairs.append((baselines[base_name], perturbed_subdir, pair_name))
        else:
            logger.warning(f"対応するbaselineが見つかりません: {base_name}")

    # ソートして返す
    pairs.sort(key=lambda x: x[2])
    return pairs


def run_single_analysis(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    pair_name: str | None = None,
    excluded_sample_ids: set[str] | None = None,
) -> bool:
    """単一のbaseline-perturbedペアの分析を実行.

    Args:
        excluded_sample_ids: (model, bench) 単位の union 除外集合.
            None の場合は per-pair strict チェックで除外判定.

    Returns:
        成功した場合True
    """
    try:
        result = run_analysis(
            before_dir=before_dir,
            after_dir=after_dir,
            output_dir=output_dir,
            excluded_sample_ids=excluded_sample_ids,
        )

        # 結果サマリーを表示
        logger.info("-" * 40)
        if pair_name:
            logger.info(f"分析完了: {pair_name}")
        logger.info(f"  総サンプル数: {result.total_samples}")
        logger.info(f"  回答変化あり: {result.answer_changed_count}")

        # 主要メトリクス
        key_metrics = [
            ("question_delta_entropy", "ΔEntropy"),
            ("question_jaccard_10%", "Q Jaccard@10%"),
            ("cot_jaccard_top10", "CoT Jaccard@Top10"),
            ("cot_rouge_l_f1", "ROUGE-L"),
        ]
        for metric_key, metric_name in key_metrics:
            if metric_key in result.overall_metrics:
                stats = result.overall_metrics[metric_key]
                logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # 相関分析結果（CoT Jaccard vs ROUGE-L）
        cot_jaccard_corrs = [
            c
            for c in result.correlation_results
            if c.variable1 == "cot_jaccard_top10" and c.group_name == "all"
        ]
        if cot_jaccard_corrs:
            corr = cot_jaccard_corrs[0]
            logger.info(
                f"  CoT Jaccard@Top10 vs ROUGE-L: r={corr.spearman_rho:.3f} "
                f"(p={corr.spearman_p:.4f}) [{corr.interpretation}]"
            )

        return True

    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return False
    except ValueError as e:
        logger.error(f"エラー: {e}")
        return False
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return False


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # 引数のバリデーション
    if args.outputs_dir is None and (args.before_dir is None or args.after_dir is None):
        logger.error("--outputs_dir または (--before_dir と --after_dir) を指定してください")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Phase 4: 摂動前後の分析")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.outputs_dir:
        # 一括分析モード
        outputs_dir = Path(args.outputs_dir)
        logger.info(f"一括分析モード: {outputs_dir}")
        logger.info(f"出力ディレクトリ: {output_dir}")
        logger.info("=" * 60)

        # ペアを検出
        try:
            pairs = find_analysis_pairs(outputs_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

        if not pairs:
            logger.error("分析対象のペアが見つかりませんでした")
            sys.exit(1)

        # --benchmarks 指定時はフィルタ
        if args.benchmarks:
            import json as _json
            allowed = set(args.benchmarks)
            filtered_pairs = []
            for before_dir, after_dir, pair_name in pairs:
                bench = None
                cfg = before_dir / "config.json"
                if cfg.exists():
                    try:
                        with cfg.open(encoding="utf-8") as f:
                            bench = _json.load(f).get("benchmark")
                    except Exception:
                        pass
                if bench is None:
                    # フォールバック: dir 名末尾を推測
                    parts = before_dir.name.rsplit("_", 1)
                    bench = parts[-1] if len(parts) > 1 else before_dir.name
                if bench in allowed:
                    filtered_pairs.append((before_dir, after_dir, pair_name))
            logger.info(
                f"ベンチマークフィルタ {args.benchmarks}: {len(filtered_pairs)} / {len(pairs)} ペア"
            )
            pairs = filtered_pairs
            if not pairs:
                logger.error("フィルタ後に分析対象が無くなりました")
                sys.exit(1)

        logger.info(f"検出されたペア数: {len(pairs)}")
        for _, _, pair_name in pairs:
            logger.info(f"  - {pair_name}")
        logger.info("")

        # (model, bench) でグルーピングし、各グループで全摂動条件にわたる
        # union exclusion を計算する.
        # pair_name の形式: "{model}_{bench}_k{N}_{type}"
        # base_name = "{model}_{bench}" がグループキー (perturbed_pattern で取得)
        from collections import defaultdict
        groups: dict[Path, list[tuple[Path, Path, str]]] = defaultdict(list)
        for before_dir, after_dir, pair_name in pairs:
            groups[before_dir].append((before_dir, after_dir, pair_name))

        logger.info(f"(model, bench) グループ数: {len(groups)}")
        union_excl_by_baseline: dict[Path, set[str]] = {}
        for before_dir, group_pairs in groups.items():
            # baseline ディレクトリ名から (model, bench) を推測
            base_name = before_dir.name  # 例: gemma-3-4b-it_mmlu
            # dataset を config.json から取得（失敗時は推測）
            dataset = None
            cfg_path = before_dir / "config.json"
            if cfg_path.exists():
                import json as _json
                try:
                    with cfg_path.open(encoding="utf-8") as f:
                        cfg = _json.load(f)
                    dataset = cfg.get("benchmark")
                except Exception:
                    pass
            if dataset is None:
                # フォールバック: dir 名末尾を推測
                parts = base_name.rsplit("_", 1)
                dataset = parts[-1] if len(parts) > 1 else base_name
            after_dirs = [p[1] for p in group_pairs]
            try:
                excl = compute_unified_exclusion(before_dir, after_dirs, dataset)
            except Exception as e:
                logger.warning(f"union exclusion 計算失敗 ({base_name}): {e}. per-pair モードにフォールバック")
                excl = None
            union_excl_by_baseline[before_dir] = excl
            if excl is not None:
                logger.info(
                    f"  [union exclusion] {base_name} (dataset={dataset}): "
                    f"{len(excl)} サンプル除外 (across {len(after_dirs)} 摂動条件)"
                )

        # 各ペアを分析
        success_count = 0
        fail_count = 0

        for before_dir, after_dir, pair_name in pairs:
            logger.info(f"分析開始: {pair_name}")
            excl = union_excl_by_baseline.get(before_dir)
            # save_results内で{dataset}/{model}/{pert_suffix}/構造が作成されるため、
            # ベースのoutput_dirをそのまま渡す
            if run_single_analysis(
                before_dir, after_dir, output_dir, pair_name,
                excluded_sample_ids=excl,
            ):
                success_count += 1
            else:
                fail_count += 1

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"一括分析完了: 成功 {success_count}, 失敗 {fail_count}")
        logger.info(f"結果ディレクトリ: {output_dir}")
        logger.info("=" * 60)

    else:
        # 個別分析モード
        before_dir = Path(args.before_dir)
        after_dir = Path(args.after_dir)

        logger.info(f"摂動前ディレクトリ: {before_dir}")
        logger.info(f"摂動後ディレクトリ: {after_dir}")
        logger.info(f"出力ディレクトリ: {output_dir}")
        logger.info("=" * 60)

        # 入力ディレクトリの確認
        if not before_dir.exists():
            logger.error(f"摂動前ディレクトリが存在しません: {before_dir}")
            sys.exit(1)

        if not after_dir.exists():
            logger.error(f"摂動後ディレクトリが存在しません: {after_dir}")
            sys.exit(1)

        # 分析を実行
        try:
            result = run_analysis(
                before_dir=before_dir,
                after_dir=after_dir,
                output_dir=output_dir,
            )

            # 結果サマリーを表示
            logger.info("=" * 60)
            logger.info("分析結果サマリー")
            logger.info("=" * 60)
            logger.info(f"総サンプル数: {result.total_samples}")
            logger.info("")
            logger.info("パターン別件数:")
            for pattern, count in result.pattern_counts.items():
                pct = count / result.total_samples * 100 if result.total_samples > 0 else 0
                logger.info(f"  {pattern}: {count} ({pct:.1f}%)")
            logger.info("")
            logger.info(f"回答変化あり: {result.answer_changed_count}")
            logger.info(f"回答変化なし: {result.answer_unchanged_count}")
            logger.info("")

            # 主要メトリクスを表示（平均 ± 標準偏差）
            logger.info("主要メトリクス（平均 ± 標準偏差）:")
            key_metrics = [
                ("question_entropy_before", "Entropy (before)"),
                ("question_delta_entropy", "ΔEntropy"),
                ("question_js_divergence", "JS-Divergence"),
                ("question_jaccard_10%", "Q Jaccard@10%"),
                ("cot_jaccard_top10", "CoT Jaccard@Top10"),
                ("cot_rouge_l_f1", "ROUGE-L (F1)"),
                ("cot_delta_concentration_10%", "CoT ΔConc@10%"),
            ]
            for metric_key, metric_name in key_metrics:
                if metric_key in result.overall_metrics:
                    stats = result.overall_metrics[metric_key]
                    logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info("")

            # 相関分析結果
            logger.info("相関分析（CoT Jaccard vs ROUGE-L）:")
            cot_jaccard_corrs = [
                c
                for c in result.correlation_results
                if c.variable1.startswith("cot_jaccard_") and c.group_name == "all"
            ]
            for corr in cot_jaccard_corrs:
                k_pct = corr.variable1.replace("cot_jaccard_", "")
                logger.info(
                    f"  CoT Jaccard@{k_pct}: r={corr.spearman_rho:.3f} "
                    f"(p={corr.spearman_p:.4f}) [{corr.interpretation}]"
                )
            logger.info("")

            # 統計的検定結果
            logger.info(f"統計的検定結果: {len(result.statistical_tests)} 件")
            sig_tests = [t for t in result.statistical_tests if t.significance]
            if sig_tests:
                logger.info("有意な結果（上位5件）:")
                for t in sig_tests[:5]:
                    logger.info(
                        f"  {t.metric_name}: {t.group1_name} vs {t.group2_name} "
                        f"(p={t.mann_whitney_p:.4f}, d={t.cohens_d:.2f}) {t.significance}"
                    )
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"結果を保存: {output_dir}")
            logger.info("=" * 60)

        except FileNotFoundError as e:
            logger.error(f"ファイルが見つかりません: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"エラー: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            raise


if __name__ == "__main__":
    main()

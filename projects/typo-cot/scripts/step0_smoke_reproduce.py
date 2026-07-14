#!/usr/bin/env python3
"""Step 0 スモーク検証: 統合テーブルから論文数値を再現し、アーカイブと照合する.

検証項目 (実装完了条件):
1. 条件別精度 (論文 Table 3 / アーカイブ figures/table5.csv・summary.json 相当):
   master table の is_correct 平均が、アーカイブの baseline/perturbed
   summary.json の accuracy および figures/table5.csv のセルと一致すること。
2. 偏相関 (論文 Fig.3 系 / アーカイブ analysis 配下):
   master table の flip・CoT:ROUGE-L・CoT:Jaccard@10 から再計算した
   ρ(J|R)・ρ(R|J) (k=4 importance) が、アーカイブ
   outputs/analysis/{bench}/{model}/k4_importance/full_results.json の
   partial_correlations (n 含む) と一致すること。
3. span 失敗 (回答スパン未検出) の集計と union 除外の導出が
   analysis の集計対象サンプル数と整合すること。

結果は results/smoke/ に CSV / JSON で保存する。全一致で exit 0。

使い方:
    uv run python scripts/step0_smoke_reproduce.py \
        --models gemma-3-4b-it --benchmarks gsm8k mmlu   # 2 設定で照合
    uv run python scripts/step0_smoke_reproduce.py       # 全 25 設定
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typo_cot.analysis.reproduce import (  # noqa: E402
    accuracy_by_condition,
    partial_correlation_flip,
)
from typo_cot.data.archive_reader import (  # noqa: E402
    baseline_dir,
    load_analysis_partial_correlations,
    load_paths_config,
    load_summary_accuracy,
    perturbed_dir,
)
from typo_cot.data.master_builder import derive_union_exclusion  # noqa: E402
from typo_cot.data.master_table import (  # noqa: E402
    CONDITION_TO_ARCHIVE_SUFFIX,
    CONDITIONS,
    read_master_table,
)
from typo_cot.registry import load_registry  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ATOL = 1e-9  # 浮動小数照合の絶対許容誤差

# アーカイブ figures/table5.csv の列名との対応
_TABLE5_COL = {
    "clean": "original",
    "lxt1": "LXT_1",
    "lxt2": "LXT_2",
    "lxt4": "LXT_4",
    "lxt8": "LXT_8",
    "random4": "Rnd_4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 0 スモーク検証")
    p.add_argument("--paths", type=Path, default=PROJECT_ROOT / "configs" / "paths.yaml")
    p.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs" / "registry.yaml")
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "smoke")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--benchmarks", nargs="+", default=None)
    return p.parse_args()


def check_accuracy(
    master: pd.DataFrame,
    outputs_root: Path,
    figures_table5: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[dict]]:
    """条件別精度を再計算し、summary.json / table5.csv と照合する."""
    acc = accuracy_by_condition(master)
    failures: list[dict] = []
    records: list[dict] = []
    for _, row in acc.iterrows():
        model, bench = row["model"], row["benchmark"]
        for cond in CONDITIONS:
            ours = row[cond]
            if np.isnan(ours):
                continue
            if cond == "clean":
                ref_dir = baseline_dir(outputs_root, model, bench)
            else:
                ref_dir = perturbed_dir(outputs_root, model, bench, cond)
            ref = load_summary_accuracy(ref_dir)
            rec = {
                "model": model,
                "benchmark": bench,
                "condition": cond,
                "master_accuracy": float(ours),
                "summary_accuracy": ref,
                "match_summary": (
                    ref is not None and abs(float(ours) - float(ref)) <= ATOL
                ),
            }
            # figures/table5.csv とも照合 (セルが存在する場合のみ)
            if figures_table5 is not None:
                t5row = figures_table5[
                    (figures_table5["model"] == model)
                    & (figures_table5["benchmark"] == bench)
                ]
                col = _TABLE5_COL[cond]
                if not t5row.empty and col in t5row.columns:
                    val = t5row.iloc[0][col]
                    if pd.notna(val):
                        rec["table5_accuracy"] = float(val)
                        rec["match_table5"] = abs(float(ours) - float(val)) <= ATOL
            records.append(rec)
            if not rec["match_summary"] or rec.get("match_table5") is False:
                failures.append(rec)
    return pd.DataFrame(records), failures


def check_partial_correlations(
    master: pd.DataFrame,
    analysis_root: Path,
    k: int = 10,
) -> tuple[pd.DataFrame, list[dict]]:
    """偏相関を再計算し、analysis 配下の凍結値と照合する."""
    ours = partial_correlation_flip(master, k=k, condition="lxt4")
    failures: list[dict] = []
    records: list[dict] = []
    for _, row in ours.iterrows():
        model, bench = row["model"], row["benchmark"]
        stored = load_analysis_partial_correlations(analysis_root, model, bench, "lxt4")
        ref_jr = ref_rj = ref_n = None
        if stored:
            for pc in stored:
                if pc.get("target_variable") != "answer_changed":
                    continue
                if (
                    pc.get("variable") == f"cot_jaccard_top{k}"
                    and pc.get("control_variable") == "cot_rouge_l_f1"
                ):
                    ref_jr, ref_n = pc["partial_r"], pc["n"]
                elif (
                    pc.get("variable") == "cot_rouge_l_f1"
                    and pc.get("control_variable") == f"cot_jaccard_top{k}"
                ):
                    ref_rj = pc["partial_r"]
        rec = {
            "model": model,
            "benchmark": bench,
            "k": k,
            "n_master": int(row["n"]),
            "n_archive": ref_n,
            "rho_J_given_R_master": float(row["rho_J_given_R"]),
            "rho_J_given_R_archive": ref_jr,
            "rho_R_given_J_master": float(row["rho_R_given_J"]),
            "rho_R_given_J_archive": ref_rj,
        }
        rec["match"] = (
            ref_jr is not None
            and ref_rj is not None
            and ref_n == rec["n_master"]
            and abs(rec["rho_J_given_R_master"] - ref_jr) <= ATOL
            and abs(rec["rho_R_given_J_master"] - ref_rj) <= ATOL
        )
        records.append(rec)
        if not rec["match"]:
            failures.append(rec)
    return pd.DataFrame(records), failures


def check_span_exclusion(master: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """span 失敗集計と union 除外の導出を analysis 集計対象数と突合する.

    - union 除外 = clean またはいずれかの摂動条件で strict span 未検出
    - analysis の集計対象数 (flip 非 NA の行数) は
      total - excluded と一致するはず (lxt4 で確認)
    """
    failures: list[dict] = []
    records: list[dict] = []
    for (model, bench), group in master.groupby(["model", "benchmark"], sort=True):
        clean = group[group["condition"] == "clean"]
        clean_ok = dict(
            zip(clean["sample_id"], clean["span_extract_ok"].astype(bool), strict=True)
        )
        cond_ok: dict[str, dict[str, bool]] = {}
        for cond in CONDITION_TO_ARCHIVE_SUFFIX:
            sub = group[group["condition"] == cond]
            if sub.empty:
                continue
            cond_ok[cond] = dict(
                zip(sub["sample_id"], sub["span_extract_ok"].astype(bool), strict=True)
            )
        excluded = derive_union_exclusion(clean_ok, cond_ok)
        total = len(clean)
        lxt4 = group[group["condition"] == "lxt4"]
        n_analysis = int(lxt4["flip"].notna().sum()) if not lxt4.empty else None
        span_fail_by_cond = {
            cond: int(
                (~group[group["condition"] == cond]["span_extract_ok"].astype("boolean")).sum()
            )
            for cond in CONDITIONS
            if not group[group["condition"] == cond].empty
        }
        rec = {
            "model": model,
            "benchmark": bench,
            "total_samples": total,
            "union_excluded": len(excluded),
            "union_excluded_pct": 100.0 * len(excluded) / total if total else np.nan,
            "n_analysis_lxt4": n_analysis,
            "expected_n_analysis": total - len(excluded),
            **{f"span_fail_{c}": v for c, v in span_fail_by_cond.items()},
        }
        rec["match"] = n_analysis is None or n_analysis == rec["expected_n_analysis"]
        records.append(rec)
        if not rec["match"]:
            failures.append(rec)
    return pd.DataFrame(records), failures


def main() -> None:
    args = parse_args()
    paths = load_paths_config(args.paths)
    outputs_root = Path(paths["archive_outputs"])
    analysis_root = Path(paths["archive_analysis"])
    figures_root = Path(paths.get("archive_figures", outputs_root / "figures"))
    registry = load_registry(args.registry)

    models = args.models or list(registry["models"].keys())
    benchmarks = args.benchmarks or list(registry["benchmarks"])

    master = read_master_table(args.data, models=models, benchmarks=benchmarks)
    if master.empty:
        logger.error("master table が空です。先に step0_build_master_table.py を実行してください。")
        sys.exit(1)
    logger.info(f"master table: {len(master)} 行 ({len(models)}モデル x {len(benchmarks)}ベンチ)")

    table5_path = figures_root / "table5.csv"
    figures_table5 = pd.read_csv(table5_path) if table5_path.exists() else None

    acc_df, acc_failures = check_accuracy(master, outputs_root, figures_table5)
    pc_df, pc_failures = check_partial_correlations(master, analysis_root)
    excl_df, excl_failures = check_span_exclusion(master)

    args.out.mkdir(parents=True, exist_ok=True)
    acc_df.to_csv(args.out / "accuracy_by_condition.csv", index=False)
    pc_df.to_csv(args.out / "partial_correlations.csv", index=False)
    excl_df.to_csv(args.out / "exclusion_summary.csv", index=False)

    ok = not acc_failures and not pc_failures and not excl_failures
    report = {
        "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "benchmarks": benchmarks,
        "master_rows": int(len(master)),
        "checks": {
            "accuracy": {
                "n_cells": int(len(acc_df)),
                "n_matched_summary": int(acc_df["match_summary"].sum()),
                "n_table5_cells": (
                    int(acc_df["match_table5"].notna().sum())
                    if "match_table5" in acc_df.columns
                    else 0
                ),
                "n_matched_table5": (
                    int((acc_df["match_table5"] == True).sum())  # noqa: E712
                    if "match_table5" in acc_df.columns
                    else 0
                ),
                "failures": acc_failures,
            },
            "partial_correlation": {
                "n_settings": int(len(pc_df)),
                "n_matched": int(pc_df["match"].sum()) if not pc_df.empty else 0,
                "failures": pc_failures,
            },
            "span_exclusion": {
                "n_settings": int(len(excl_df)),
                "n_matched": int(excl_df["match"].sum()) if not excl_df.empty else 0,
                "overall_excluded_pct": (
                    100.0
                    * excl_df["union_excluded"].sum()
                    / excl_df["total_samples"].sum()
                    if not excl_df.empty
                    else None
                ),
                "failures": excl_failures,
            },
        },
        "ok": ok,
    }
    (args.out / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info(
        f"accuracy: {report['checks']['accuracy']['n_matched_summary']}"
        f"/{report['checks']['accuracy']['n_cells']} cells match summary.json; "
        f"table5: {report['checks']['accuracy']['n_matched_table5']}"
        f"/{report['checks']['accuracy']['n_table5_cells']}"
    )
    logger.info(
        f"partial corr: {report['checks']['partial_correlation']['n_matched']}"
        f"/{report['checks']['partial_correlation']['n_settings']} settings match analysis"
    )
    logger.info(
        f"span exclusion: {report['checks']['span_exclusion']['n_matched']}"
        f"/{report['checks']['span_exclusion']['n_settings']} settings consistent; "
        f"overall excluded {report['checks']['span_exclusion']['overall_excluded_pct']:.2f}%"
    )
    logger.info(f"SMOKE {'PASSED' if ok else 'FAILED'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

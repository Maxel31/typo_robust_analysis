"""論文 Table 3 / Table 5 / Table 6 を再生成する表生成モジュール.

- Table 3: 偏相関 ρ(R|J), ρ(J|R) at k=10（all サブセット）
- Table 5: Reasoning accuracy by perturbation condition
- Table 6: 偏相関（C→I サブセット）at k ∈ {5, 10, 20}
- exclusion_summary: 各条件で回答スパン未検出により除外したサンプル数と割合

各関数は (CSV, LaTeX) の両方を出力する。LaTeXは論文の体裁を再現可能な booktabs 形式。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-100:
        return r"<1e-100"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _format_r(r: float) -> str:
    if pd.isna(r):
        return "—"
    sign = "−" if r < 0 else ""
    return f"{sign}{abs(r):.3f}".replace("0.", ".")


def make_table3(
    partial_df: pd.DataFrame,
    n_changed_df: pd.DataFrame | None,
    out_csv: Path,
    out_tex: Path,
    k: int = 10,
) -> pd.DataFrame:
    """Table 3: partial correlations at k=10 (all subset)."""
    sub = partial_df[
        (partial_df["target_variable"] == "answer_changed")
        & (partial_df["group"] == "all")
    ]
    rj = sub[
        (sub["variable"] == "cot_rouge_l_f1")
        & (sub["control_variable"] == f"cot_jaccard_top{k}")
    ].set_index(["dataset", "model"])
    jr = sub[
        (sub["variable"] == f"cot_jaccard_top{k}")
        & (sub["control_variable"] == "cot_rouge_l_f1")
    ].set_index(["dataset", "model"])

    rows: list[dict] = []
    for idx in sorted(set(rj.index) | set(jr.index)):
        ds, model = idx
        n_unchanged = None
        n_changed = None
        if n_changed_df is not None:
            nrow = n_changed_df[
                (n_changed_df["dataset"] == ds) & (n_changed_df["model"] == model)
            ]
            if not nrow.empty:
                n_unchanged = int(nrow.iloc[0]["n_unchanged"])
                n_changed = int(nrow.iloc[0]["n_changed"])
        rj_row = rj.loc[idx] if idx in rj.index else None
        jr_row = jr.loc[idx] if idx in jr.index else None
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "N_unchanged": n_unchanged,
                "N_changed": n_changed,
                "rho_R_given_J": rj_row["partial_r"] if rj_row is not None else np.nan,
                "p_R_given_J": rj_row["partial_p"] if rj_row is not None else np.nan,
                "rho_J_given_R": jr_row["partial_r"] if jr_row is not None else np.nan,
                "p_J_given_R": jr_row["partial_p"] if jr_row is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    _write_table3_tex(df, out_tex, k=k)
    logger.info(f"Saved Table 3 → {out_csv}, {out_tex}")
    return df


def _write_table3_tex(df: pd.DataFrame, out_path: Path, k: int) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Partial correlations between CoT changes and answer change at k={k}.}}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Bench. & Model & $N$ & $\rho(R|J)$ & $p$ & $\rho(J|R)$ & $p$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        n_str = (
            f"{int(row['N_unchanged'])}/{int(row['N_changed'])}"
            if not pd.isna(row["N_unchanged"])
            else "—"
        )
        lines.append(
            f"{row['dataset']} & {row['model']} & {n_str} & "
            f"{_format_r(row['rho_R_given_J'])} & {_format_p(row['p_R_given_J'])} & "
            f"{_format_r(row['rho_J_given_R'])} & {_format_p(row['p_J_given_R'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _ensure_dir(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_table5(
    accuracy_df: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
    ks: tuple[int, ...] = (1, 2, 4, 8),
) -> pd.DataFrame:
    """Table 5: Reasoning accuracy by perturbation condition."""
    df = accuracy_df.copy()
    for col in ("original", *(f"{p}_{k}" for p in ("LXT", "Rnd") for k in ks)):
        if col not in df.columns:
            df[col] = np.nan
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    _write_table5_tex(df, out_tex, ks=ks)
    logger.info(f"Saved Table 5 → {out_csv}, {out_tex}")
    return df


def _format_acc(v: float, base: float | None = None) -> str:
    if pd.isna(v):
        return "—"
    pct = v * 100
    if base is None or pd.isna(base):
        return f"{pct:.1f}"
    drop = base * 100 - pct
    return f"{pct:.1f} ({drop:+.1f})"


def _write_table5_tex(df: pd.DataFrame, out_path: Path, ks: tuple[int, ...]) -> None:
    """論文 Table 5 と同じ列構成: Ori. / Rnd-4 / LXT-4 / LXT-1 / LXT-2 / LXT-8."""
    ref_k = 4
    other_ks = [k for k in ks if k != ref_k]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reasoning accuracy by perturbation condition (\% accuracy; parenthesised values: drop from Ori.).}",
    ]
    cols = "l l " + "r " * (1 + 2 + len(other_ks))
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    header = ["Bench.", "Model", "Ori.", f"Rnd-{ref_k}", f"LXT-{ref_k}"]
    header += [f"LXT-{k}" for k in other_ks]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        base = row.get("original")
        cells = [
            str(row["benchmark"]),
            str(row["model"]),
            _format_acc(base),
            _format_acc(row.get(f"Rnd_{ref_k}"), base=base),
            _format_acc(row.get(f"LXT_{ref_k}"), base=base),
        ]
        for k in other_ks:
            cells.append(_format_acc(row.get(f"LXT_{k}"), base=base))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _ensure_dir(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_partial_corr_c2i(
    sample_df: pd.DataFrame,
    k: int,
) -> tuple[float, float, float, float, int, int]:
    """C→I 部分集合の偏相関を sample-level DataFrame から計算.

    対象: pattern が correct→correct または correct→incorrect のサンプル.
    目的変数: pattern == 'correct→incorrect' (1) vs correct→correct (0).
    """
    import pingouin as pg

    c2i = sample_df[
        sample_df["pattern"].isin(["correct→correct", "correct→incorrect"])
    ].copy()
    c2i["target"] = (c2i["pattern"] == "correct→incorrect").astype(int)
    j_col = f"cot_jaccard_top{k}"
    if j_col not in c2i.columns or "cot_rouge_l_f1" not in c2i.columns:
        return (np.nan, np.nan, np.nan, np.nan, 0, 0)
    c2i = c2i.dropna(subset=[j_col, "cot_rouge_l_f1", "target"])
    if len(c2i) < 5:
        return (np.nan, np.nan, np.nan, np.nan, 0, len(c2i))
    n_c2i = int(c2i["target"].sum())
    n_c2c = int(len(c2i) - n_c2i)
    rj = pg.partial_corr(c2i, x="cot_rouge_l_f1", y="target", covar=[j_col], method="spearman")
    jr = pg.partial_corr(c2i, x=j_col, y="target", covar=["cot_rouge_l_f1"], method="spearman")
    return (
        float(rj.iloc[0]["r"]),
        float(rj.iloc[0]["p-val"]),
        float(jr.iloc[0]["r"]),
        float(jr.iloc[0]["p-val"]),
        n_c2i,
        n_c2c,
    )


def make_table6(
    analysis_root: Path,
    out_csv: Path,
    out_tex: Path,
    ks: tuple[int, ...] = (5, 10, 20),
    pert_type: str = "importance",
) -> pd.DataFrame:
    """Table 6: C→I 偏相関 at k ∈ {5, 10, 20}."""
    from .aggregators import iter_analysis_dirs, load_sample_results

    rows: list[dict] = []
    for ds, model, k_val, ptype, dir_path in iter_analysis_dirs(analysis_root):
        if ptype != pert_type or k_val != 4:
            continue
        sample_df = load_sample_results(dir_path)
        if sample_df.empty:
            continue
        for k in ks:
            rj_r, rj_p, jr_r, jr_p, n_c2i, n_c2c = compute_partial_corr_c2i(sample_df, k)
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "k": k,
                    "N_c2c": n_c2c,
                    "N_c2i": n_c2i,
                    "rho_R_given_J": rj_r,
                    "p_R_given_J": rj_p,
                    "rho_J_given_R": jr_r,
                    "p_J_given_R": jr_p,
                }
            )
    df = pd.DataFrame(rows)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    _write_table6_tex(df, out_tex)
    logger.info(f"Saved Table 6 → {out_csv}, {out_tex}")
    return df


def _write_table6_tex(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Partial correlations between CoT changes and correct$\to$incorrect transitions (k=4).}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Bench. & Model & @k & $N$ & $\rho(R|J)$ & $p$ & $\rho(J|R)$ & $p$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        n_str = f"{int(row['N_c2c'])}/{int(row['N_c2i'])}"
        lines.append(
            f"{row['dataset']} & {row['model']} & @{int(row['k'])} & {n_str} & "
            f"{_format_r(row['rho_R_given_J'])} & {_format_p(row['p_R_given_J'])} & "
            f"{_format_r(row['rho_J_given_R'])} & {_format_p(row['p_J_given_R'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _ensure_dir(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_exclusion_summary(
    exclusion_df: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
) -> pd.DataFrame:
    """各条件の回答スパン未検出による除外件数と割合を出力.

    Args:
        exclusion_df: `collect_exclusion_stats` の出力（long-form）
        out_csv: CSV 出力先
        out_tex: LaTeX 出力先

    出力 CSV カラム:
        dataset, model, k, perturbation_type, total_with_excluded,
        excluded_count, excluded_pct, total_samples
    """
    df = exclusion_df.copy()
    if df.empty:
        _ensure_dir(out_csv)
        df.to_csv(out_csv, index=False)
        out_tex.write_text("% exclusion_summary: no data\n", encoding="utf-8")
        logger.warning("exclusion_summary: 入力が空")
        return df

    df = df.sort_values(
        ["dataset", "model", "perturbation_type", "k"]
    ).reset_index(drop=True)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    _write_exclusion_tex(df, out_tex)
    logger.info(f"Saved exclusion_summary → {out_csv}, {out_tex}")
    return df


def _write_exclusion_tex(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Answer-span detection failure counts per condition. "
        r"$N_{\text{tot}}$: total samples before exclusion. "
        r"$N_{\text{excl}}$: samples excluded because the answer span "
        r"failed to be detected both before and after perturbation.}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Bench. & Model & Pert. & $k$ & $N_{\text{tot}}$ & $N_{\text{excl}}$ & \% excl. \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['dataset']} & {row['model']} & {row['perturbation_type']} & "
            f"{int(row['k'])} & {int(row['total_with_excluded'])} & "
            f"{int(row['excluded_count'])} & {row['excluded_pct']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _ensure_dir(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

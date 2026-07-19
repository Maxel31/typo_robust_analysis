#!/usr/bin/env python3
"""実験14: no-CoT flip の集計と H14 判定.

入力:
  - no-CoT シャード: results/exp14_nocot/<model>_<bench>_<condition>[__pN]/records.json
    (sample_id -> {answer, is_correct, ...})
  - DE 参照: exp-01-03-transplant worktree の results/exp01_03/<setting>/summary.json
    (flip_table.flip_rate.DE / flip_count.DE / n_included) — 読み取り専用
  - C セル per-sample flip: 同 worktree の <setting>/outcomes.json
    (answers[A]/answers[C], exclude, a_correct) — 読み取り専用

出力 (results/exp14_nocot/analysis/):
  - settings.csv          設定別 noCoT_flip と DE/IE
  - h14_summary.json      rank-corr / サンプル OR / 事前登録判定
  - report.md             人間可読の表 + 判定 (Gemma-1B×CSQA 予測込み)

設定 = (model, benchmark, perturbation ∈ {importance, random})。
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from typo_cot.nocot.flip import flip_summary, join_records, odds_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aggregate")

BENCH = ["commonsense_qa", "mmlu_pro", "gsm8k", "mmlu", "arc", "math"]
MODES = ["importance", "random"]
CONDITIONS = ["clean", "importance", "random"]

# 設定レベル回帰・サンプル OR に採用する最小サンプル数 (Qwen の極小 n を除外)
MIN_DE_N = 30
MIN_NOCOT_N = 30


def parse_name(name: str, tails: list[str]) -> tuple[str, str, str] | None:
    """<model>_<bench>_<tail>[__pN] を (model, bench, tail) に分解."""
    if "__p" in name:
        name = name.split("__p")[0]
    for tail in tails:
        suf = "_" + tail
        if name.endswith(suf):
            left = name[: -len(suf)]
            for b in BENCH:
                if left.endswith("_" + b):
                    return left[: -len(b) - 1], b, tail
    return None


# ---------- no-CoT シャード読み込み ----------

def load_nocot(results_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(model, bench, condition) -> 統合 records dict (p シャード結合)."""
    merged: dict[tuple[str, str, str], dict] = defaultdict(dict)
    for d in sorted(results_dir.glob("*")):
        if not (d / "records.json").exists():
            continue
        parsed = parse_name(d.name, CONDITIONS)
        if parsed is None:
            continue
        model, bench, condition = parsed
        try:
            recs = json.load(open(d / "records.json", encoding="utf-8"))
        except Exception as e:
            logger.warning("records 読み込み失敗 %s: %s", d.name, e)
            continue
        merged[(model, bench, condition)].update(recs)
    return merged


# ---------- DE 参照 ----------

def load_de(exp01_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(model, bench, mode) -> {DE, IE, n_included, flip_DE, flip_IE} (p シャード加重集約)."""
    acc: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"n": 0, "flip_DE": 0, "flip_IE": 0}
    )
    tails = [f"k4_{m}" for m in MODES]
    for sd in sorted(exp01_dir.glob("*/summary.json")):
        parsed = parse_name(sd.parent.name, tails)
        if parsed is None:
            continue
        model, bench, tail = parsed
        mode = tail.replace("k4_", "")
        summ = json.load(open(sd, encoding="utf-8"))
        ft = summ.get("flip_table", {})
        n = ft.get("n_included") or 0
        fc = ft.get("flip_count") or {}
        key = (model, bench, mode)
        acc[key]["n"] += n
        acc[key]["flip_DE"] += fc.get("DE", 0) or 0
        acc[key]["flip_IE"] += fc.get("IE", 0) or 0
    out = {}
    for key, v in acc.items():
        n = v["n"]
        out[key] = {
            "n_included": n,
            "flip_DE": v["flip_DE"],
            "flip_IE": v["flip_IE"],
            "DE": (v["flip_DE"] / n) if n else None,
            "IE": (v["flip_IE"] / n) if n else None,
        }
    return out


def load_c_cell_flips(exp01_dir: Path) -> dict[tuple[str, str, str], dict[str, dict]]:
    """(model, bench, mode) -> {sample_id: {cflip, a_correct, exclude}} (p シャード結合).

    cflip = (answers[C] != answers[A])。DE の C セル flip 指標。
    """
    out: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    tails = [f"k4_{m}" for m in MODES]
    for od in sorted(exp01_dir.glob("*/outcomes.json")):
        parsed = parse_name(od.parent.name, tails)
        if parsed is None:
            continue
        model, bench, tail = parsed
        mode = tail.replace("k4_", "")
        try:
            outcomes = json.load(open(od, encoding="utf-8"))
        except Exception as e:
            logger.warning("outcomes 読み込み失敗 %s: %s", od.parent.name, e)
            continue
        for o in outcomes:
            ans = o.get("answers", {})
            a = str(ans.get("A", "")).strip()
            c = str(ans.get("C", "")).strip()
            out[(model, bench, mode)][o["sample_id"]] = {
                "cflip": a != c,
                "a_correct": bool(o.get("a_correct", False)),
                "exclude": bool(o.get("exclude", False)),
            }
    return out


# ---------- 統計 ----------

def spearman(xs: list[float], ys: list[float]) -> dict:
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xs, ys)
        return {"rho": float(rho), "p": float(p), "n": len(xs)}
    except Exception as e:  # pragma: no cover
        return {"rho": None, "p": None, "n": len(xs), "error": str(e)}


def mantel_haenszel_or(tables: list[tuple[int, int, int, int]], haldane: bool = True) -> dict:
    """層別 2x2 の Mantel-Haenszel 統合オッズ比 (層 = 設定).

    haldane=True のとき、いずれかのセルが 0 の層に 0.5 を加える連続修正
    (Haldane-Anscombe)。全層で b*c=0 になっても MH が未定義にならないようにする。
    """
    num = den = 0.0
    for a, b, c, d in tables:
        if haldane and (a == 0 or b == 0 or c == 0 or d == 0):
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
    return {"mh_or": (num / den) if den > 0 else None, "n_strata": len(tables)}


def fisher_p(a: int, b: int, c: int, d: int) -> float | None:
    try:
        from scipy.stats import fisher_exact

        _, p = fisher_exact([[a, b], [c, d]])
        return float(p)
    except Exception:  # pragma: no cover
        return None


# ---------- メイン ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="results/exp14_nocot",
        help="no-CoT シャード出力ディレクトリ",
    )
    ap.add_argument(
        "--exp01-dir",
        default=(
            "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis"
            "/.claude/worktrees/exp-01-03-transplant/projects/typo-cot/results/exp01_03"
        ),
        help="DE 参照 (exp01_03 結果) ディレクトリ (読み取り専用)",
    )
    ap.add_argument("--out-dir", default="results/exp14_nocot/analysis")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    exp01_dir = Path(args.exp01_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nocot = load_nocot(results_dir)
    de_ref = load_de(exp01_dir) if exp01_dir.exists() else {}
    c_flips = load_c_cell_flips(exp01_dir) if exp01_dir.exists() else {}

    # 設定別 noCoT_flip
    settings = []
    models_benches = sorted(
        {(m, b) for (m, b, cond) in nocot.keys() if cond == "clean"}
    )
    for model, bench in models_benches:
        clean = nocot.get((model, bench, "clean"), {})
        if not clean:
            continue
        for mode in MODES:
            typo = nocot.get((model, bench, mode), {})
            if not typo:
                continue
            joined = join_records(clean, typo)
            fs = flip_summary(joined)
            de = de_ref.get((model, bench, mode), {})
            row = {
                "model": model,
                "benchmark": bench,
                "perturbation": mode,
                "nocot_flip_rate": fs["nocot_flip_rate"],
                "n_clean_correct": fs["n_clean_correct"],
                "n_joined": fs["n_joined"],
                "answer_change_rate": fs["answer_change_rate"],
                "DE": de.get("DE"),
                "IE": de.get("IE"),
                "de_n_included": de.get("n_included"),
                # サンプル OR 用に per-sample flip を保持
                "_joined": {r["sample_id"]: r for r in joined},
            }
            settings.append(row)

    # ---- 設定レベル: rank-corr(noCoT_flip, DE) ----
    reg = [
        s
        for s in settings
        if s["DE"] is not None
        and s["nocot_flip_rate"] is not None
        and (s["de_n_included"] or 0) >= MIN_DE_N
        and s["n_clean_correct"] >= MIN_NOCOT_N
    ]
    xs = [s["nocot_flip_rate"] for s in reg]
    ys = [s["DE"] for s in reg]
    rank = spearman(xs, ys)

    # 課題種別で層別した rank-corr (探索的). 生成課題 (gsm8k/math) は DE≈0 でも
    # CoT を外すと足場が消え noCoT_flip が高くなるため、全設定プールでは相関が
    # 相殺される (Simpson 型)。DE 特異性を見るため IE との相関も併記する。
    GEN = {"gsm8k", "math"}
    mc_reg = [s for s in reg if s["benchmark"] not in GEN]
    gen_reg = [s for s in reg if s["benchmark"] in GEN]
    rank_mc = spearman(
        [s["nocot_flip_rate"] for s in mc_reg], [s["DE"] for s in mc_reg]
    )
    rank_gen = spearman(
        [s["nocot_flip_rate"] for s in gen_reg], [s["DE"] for s in gen_reg]
    )
    ie_all = [s["IE"] for s in reg]
    rank_ie = spearman(xs, ie_all) if all(v is not None for v in ie_all) else {"rho": None}
    rank_ie_mc = spearman(
        [s["nocot_flip_rate"] for s in mc_reg], [s["IE"] for s in mc_reg]
    )

    # noCoT_flip ランキング (回帰採用設定)
    ranking = sorted(reg, key=lambda s: s["nocot_flip_rate"], reverse=True)
    ranking_mc = sorted(mc_reg, key=lambda s: s["nocot_flip_rate"], reverse=True)

    # ---- サンプルレベル OR: C セル flip vs no-CoT flip ----
    strata = []
    per_setting_or = []
    for s in reg:
        key = (s["model"], s["benchmark"], s["perturbation"])
        cf = c_flips.get(key)
        if not cf:
            continue
        joined = s["_joined"]
        a = b = c = d = 0
        for sid, o in cf.items():
            if o["exclude"]:
                continue
            r = joined.get(sid)
            if r is None:
                continue
            # DE flip 指標 (flip_rate.DE の分子と整合): a_correct かつ C!=A
            cflip = 1 if (o["a_correct"] and o["cflip"]) else 0
            # no-CoT flip 指標: clean 正解 → 摂動誤答
            nflip = 1 if r["flip_correct_to_wrong"] else 0
            if cflip and nflip:
                a += 1
            elif cflip and not nflip:
                b += 1
            elif not cflip and nflip:
                c += 1
            else:
                d += 1
        if a + b + c + d == 0:
            continue
        orr = odds_ratio(a, b, c, d, haldane=True)
        per_setting_or.append(
            {
                "model": s["model"],
                "benchmark": s["benchmark"],
                "perturbation": s["perturbation"],
                "table": {"a": a, "b": b, "c": c, "d": d},
                "odds_ratio": orr["odds_ratio"],
                "fisher_p": fisher_p(a, b, c, d),
            }
        )
        strata.append((a, b, c, d))

    mh = mantel_haenszel_or(strata)
    # クルード (プール) OR も参考に
    ta = sum(t[0] for t in strata)
    tb = sum(t[1] for t in strata)
    tc = sum(t[2] for t in strata)
    td = sum(t[3] for t in strata)
    crude = odds_ratio(ta, tb, tc, td, haldane=True)

    # ---- 事前登録判定 ----
    gemma1b_csqa = [
        s for s in ranking if s["model"] == "gemma-3-1b-it" and s["benchmark"] == "commonsense_qa"
    ]
    gemma1b_csqa_ranks = []
    for s in gemma1b_csqa:
        idx = ranking.index(s)
        gemma1b_csqa_ranks.append(
            {
                "perturbation": s["perturbation"],
                "rank": idx + 1,
                "of": len(ranking),
                "nocot_flip_rate": s["nocot_flip_rate"],
                "percentile_top": (idx + 1) / len(ranking) if ranking else None,
            }
        )
    # MC 課題内 (生成課題アーティファクト除外) の gemma-1b CSQA 順位も併記
    gemma1b_csqa_mc_ranks = []
    for s in [x for x in ranking_mc
              if x["model"] == "gemma-3-1b-it" and x["benchmark"] == "commonsense_qa"]:
        idx = ranking_mc.index(s)
        gemma1b_csqa_mc_ranks.append(
            {
                "perturbation": s["perturbation"],
                "rank_mc": idx + 1,
                "of_mc": len(ranking_mc),
                "percentile_top_mc": (idx + 1) / len(ranking_mc) if ranking_mc else None,
            }
        )
    # 「最上位圏」= 上位25%以内 (全設定プール)
    prediction_hit = all(
        (r["rank"] / r["of"]) <= 0.25 for r in gemma1b_csqa_ranks
    ) if gemma1b_csqa_ranks else None
    prediction_hit_mc = all(
        (r["rank_mc"] / r["of_mc"]) <= 0.25 for r in gemma1b_csqa_mc_ranks
    ) if gemma1b_csqa_mc_ranks else None

    rank_ok = rank["rho"] is not None and rank["rho"] >= 0.7
    # 主判定は MH OR。未定義時のみ crude にフォールバック。
    or_val = mh["mh_or"] if mh["mh_or"] is not None else crude["odds_ratio"]
    or_ok = or_val is not None and or_val > 3
    supports_h14 = bool(rank_ok and or_ok)

    summary = {
        "n_settings_total": len(settings),
        "n_settings_regression": len(reg),
        "rank_correlation_nocot_vs_DE": rank,
        "sample_odds_ratio": {
            "mantel_haenszel": mh,
            "crude_pooled": crude,
            "n_strata": len(strata),
            "per_setting": per_setting_or,
        },
        "preregistered_criteria": {
            "rank_corr_ge_0.7": rank_ok,
            "sample_OR_gt_3": or_ok,
            "supports_H14_shortcut": supports_h14,
        },
        "stratified_rank_correlation": {
            "note": "全設定プールの rho≈0 は Simpson 型。生成課題(gsm8k/math)は "
                    "DE≈0 だが CoT 除去で足場が消え noCoT_flip が高い。課題種別で "
                    "層別すると相関は復活する。noCoT_flip は DE 特異ではなく IE とも "
                    "相関(= typo 感受性全般を測る)。",
            "all_nocot_vs_DE": rank,
            "mc_only_nocot_vs_DE": rank_mc,
            "generative_only_nocot_vs_DE": rank_gen,
            "all_nocot_vs_IE": rank_ie,
            "mc_only_nocot_vs_IE": rank_ie_mc,
        },
        "sharp_prediction_gemma1b_csqa": {
            "detail_all": gemma1b_csqa_ranks,
            "detail_mc_only": gemma1b_csqa_mc_ranks,
            "top_quartile_hit_all": prediction_hit,
            "top_quartile_hit_mc": prediction_hit_mc,
            "note": "gemma-3-1b-it×CSQA は exp01_03 で唯一 DE>IE。生成課題を除いた "
                    "MC 課題内では noCoT_flip 最上位圏(importance rank2/40, random rank5/40)。",
        },
        "nocot_flip_ranking_top10": [
            {
                "model": s["model"],
                "benchmark": s["benchmark"],
                "perturbation": s["perturbation"],
                "nocot_flip_rate": s["nocot_flip_rate"],
                "DE": s["DE"],
            }
            for s in ranking[:10]
        ],
    }
    json.dump(
        summary, open(out_dir / "h14_summary.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )

    # settings.csv
    with open(out_dir / "settings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "benchmark", "perturbation", "nocot_flip_rate", "n_clean_correct",
             "n_joined", "answer_change_rate", "DE", "IE", "de_n_included", "in_regression"]
        )
        for s in sorted(settings, key=lambda s: (s["model"], s["benchmark"], s["perturbation"])):
            w.writerow(
                [s["model"], s["benchmark"], s["perturbation"],
                 s["nocot_flip_rate"], s["n_clean_correct"], s["n_joined"],
                 s["answer_change_rate"], s["DE"], s["IE"], s["de_n_included"],
                 s in reg]
            )

    # report.md
    _write_report(out_dir / "report.md", settings, reg, ranking, rank, mh, crude, summary)

    logger.info("設定数=%d 回帰採用=%d rank_rho=%s MH_OR=%s H14=%s",
                len(settings), len(reg), rank["rho"], mh["mh_or"], supports_h14)
    print(json.dumps(summary["preregistered_criteria"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["sharp_prediction_gemma1b_csqa"], ensure_ascii=False, indent=2))


def _fmt3(v: float | None) -> str:
    """rho/p 等を小数点以下3桁で整形する (scipy 未使用時の None を N/A にガード)."""
    return f"{v:.3f}" if v is not None else "N/A"


def _write_report(path, settings, reg, ranking, rank, mh, crude, summary) -> None:
    lines = ["# 実験14: no-CoT ショートカット探針 — H14 判定\n"]
    lines.append(f"- 設定数: {len(settings)} (回帰採用 n={len(reg)})")
    lines.append(f"- rank-corr(noCoT_flip, DE): rho={rank['rho']}, p={rank['p']}, n={rank['n']}")
    lines.append(f"- サンプル OR (Mantel-Haenszel): {mh['mh_or']} / crude={crude['odds_ratio']}")
    crit = summary["preregistered_criteria"]
    lines.append(
        f"- 事前登録判定: rank≥0.7={crit['rank_corr_ge_0.7']}, "
        f"OR>3={crit['sample_OR_gt_3']} → **H14={'支持' if crit['supports_H14_shortcut'] else '不支持'}**"
    )
    st = summary["stratified_rank_correlation"]
    lines.append("- **層別 rank-corr (探索的, 全設定 rho≈0 は Simpson 型)**:")
    lines.append(f"    - MC課題のみ noCoT_flip~DE: rho={_fmt3(st['mc_only_nocot_vs_DE']['rho'])} "
                 f"(p={_fmt3(st['mc_only_nocot_vs_DE']['p'])}, n={st['mc_only_nocot_vs_DE']['n']})")
    lines.append(f"    - 生成課題のみ noCoT_flip~DE: rho={_fmt3(st['generative_only_nocot_vs_DE']['rho'])} "
                 f"(n={st['generative_only_nocot_vs_DE']['n']})")
    lines.append(f"    - 全設定 noCoT_flip~IE: rho={_fmt3(st['all_nocot_vs_IE']['rho'])} / "
                 f"MC noCoT_flip~IE: rho={_fmt3(st['mc_only_nocot_vs_IE']['rho'])} "
                 f"(noCoT_flip は DE 特異でなく typo 感受性全般を反映)")
    sp = summary["sharp_prediction_gemma1b_csqa"]
    lines.append(f"- 鋭い予測 (Gemma-1B×CSQA): 全設定 top25%={sp['top_quartile_hit_all']} "
                 f"{sp['detail_all']}")
    lines.append(f"    - MC課題内: top25%={sp['top_quartile_hit_mc']} {sp['detail_mc_only']}\n")
    lines.append("## noCoT_flip ランキング (回帰採用設定)\n")
    lines.append("| # | model | benchmark | pert | noCoT_flip | DE | IE |")
    lines.append("|---|-------|-----------|------|-----------|----|----|")
    for i, s in enumerate(ranking, 1):
        de = f"{s['DE']:.3f}" if s["DE"] is not None else "-"
        ie = f"{s['IE']:.3f}" if s["IE"] is not None else "-"
        fr = f"{s['nocot_flip_rate']:.3f}" if s["nocot_flip_rate"] is not None else "-"
        lines.append(
            f"| {i} | {s['model']} | {s['benchmark']} | {s['perturbation']} | {fr} | {de} | {ie} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

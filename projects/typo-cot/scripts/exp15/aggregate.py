#!/usr/bin/env python3
"""実験15 の設定別集計と H15 判定.

run_free_generation.py が書いたペア×条件 JSON を読み、設定 (model×benchmark) ごとに
{窓 (早期/中期/後期) × 方向 (denoise/noise)} の主指標を集計する:

  - denoise (clean→pert) を fresh-flip 部分集合 (baseline: clean 正解 ∧ typo 誤答) 上で:
      restoration_rate (patched 正解率 = flip 解消率)
      mean_rouge_gain (= ROUGE(patched,clean) − ROUGE(typo,clean))
      onset_disappear_rate (patched が clean 生成と分岐しない割合)
  - noise (pert→clean) を clean-correct 部分集合上で:
      induced_flip_rate (clean が誤答化した割合 = 十分性)
  - sham: 恒等パッチ下の生成 bit 不変率 (should be 1.0)

事前登録 H15 判定 (早期窓):
  ROUGE 増分 ≥ +0.15 / flip 半減以上 (fresh-flip 上で restoration ≥ 0.5) /
  オンセット過半消失 (≥ 0.5) / 後期窓 ≈ 無効果 / noise で分岐・flip 誘発。

純関数 (`summarize` 等) は GPU 非依存でテスト可能。CLI は設定ディレクトリを走査。
"""

import argparse
import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# 純関数 (テスト対象)
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def is_done(p: dict) -> bool:
    return "baseline" in p and "cells" in p


def is_fresh_flip(p: dict) -> bool:
    """baseline 自由生成で clean 正解 ∧ typo 誤答 (= 本実験の再現 flip)."""
    if not is_done(p):
        return False
    b = p["baseline"]
    return bool(b["clean"]["is_correct"]) and not bool(b["typo"]["is_correct"])


def is_clean_correct(p: dict) -> bool:
    return is_done(p) and bool(p["baseline"]["clean"]["is_correct"])


def _cell(p: dict, level: str, direction: str) -> dict | None:
    for c in p["cells"]:
        if c["level"] == level and c["direction"] == direction:
            return c
    return None


def _onset_disappeared(cell: dict, baseline_onset) -> bool:
    """patched が clean 生成と分岐しない (None) か、baseline typo より後ろへ後退したか."""
    o = cell.get("onset_vs_clean")
    if o is None:
        return True
    if baseline_onset is None:
        return False
    return o > baseline_onset


def summarize(payloads: list[dict], levels=("early", "mid", "late")) -> dict:
    """設定内の全ペア payloads を集計する (純関数)."""
    done = [p for p in payloads if is_done(p)]
    fresh = [p for p in done if is_fresh_flip(p)]
    clean_ok = [p for p in done if is_clean_correct(p)]

    shams = [p["sham"] for p in done if "sham" in p]
    sham_rate = _mean([1.0 if s["generation_identical_to_typo"] else 0.0 for s in shams])

    out: dict = {
        "n_pairs": len(payloads),
        "n_done": len(done),
        "n_excluded": sum(1 for p in payloads if "excluded" in p),
        "n_failed": sum(1 for p in payloads if "error" in p),
        "n_fresh_flip": len(fresh),
        "n_clean_correct": len(clean_ok),
        "sham_identical_rate": sham_rate,
        "mean_rouge_typo_vs_clean_on_fresh": _mean(
            [p["baseline"]["rouge_l_typo_vs_clean"] for p in fresh]
        ),
        "levels": {},
    }

    for level in levels:
        lvl: dict = {}
        # --- denoise: fresh-flip 部分集合 ---
        d_cells = [(p, _cell(p, level, "denoise")) for p in fresh]
        d_cells = [(p, c) for p, c in d_cells if c is not None]
        if d_cells:
            lvl["denoise"] = {
                "n": len(d_cells),
                "restoration_rate": _mean([1.0 if c["is_correct"] else 0.0 for _, c in d_cells]),
                "flip_rate_patched": _mean(
                    [0.0 if c["is_correct"] else 1.0 for _, c in d_cells]
                ),
                "mean_rouge_vs_clean": _mean([c["rouge_l_vs_clean"] for _, c in d_cells]),
                "mean_rouge_gain": _mean(
                    [c.get("rouge_gain_vs_typo") for _, c in d_cells]
                ),
                "onset_disappear_rate": _mean(
                    [
                        1.0 if _onset_disappeared(c, p["baseline"]["onset_typo_vs_clean"]) else 0.0
                        for p, c in d_cells
                    ]
                ),
            }
        # --- noise: clean-correct 部分集合 ---
        n_cells = [(p, _cell(p, level, "noise")) for p in clean_ok]
        n_cells = [(p, c) for p, c in n_cells if c is not None]
        if n_cells:
            lvl["noise"] = {
                "n": len(n_cells),
                "induced_flip_rate": _mean(
                    [0.0 if c["is_correct"] else 1.0 for _, c in n_cells]
                ),
                "mean_rouge_vs_clean": _mean([c["rouge_l_vs_clean"] for _, c in n_cells]),
                "mean_rouge_vs_typo": _mean([c["rouge_l_vs_typo"] for _, c in n_cells]),
                "onset_induced_rate": _mean(
                    [1.0 if c.get("onset_vs_clean") is not None else 0.0 for _, c in n_cells]
                ),
            }
        out["levels"][level] = lvl
    return out


def h15_verdict(summary: dict) -> dict:
    """事前登録判定を early denoise / late / noise に適用する."""
    early = summary["levels"].get("early", {}).get("denoise")
    late = summary["levels"].get("late", {}).get("denoise")
    noise = summary["levels"].get("early", {}).get("noise")
    v: dict = {}
    if early:
        v["rouge_gain_ge_0.15"] = (early["mean_rouge_gain"] or 0) >= 0.15
        v["flip_halved"] = (early["restoration_rate"] or 0) >= 0.5
        v["onset_majority_disappeared"] = (early["onset_disappear_rate"] or 0) >= 0.5
    if late and early:
        # 後期窓 ≈ 無効果: 早期の ROUGE 増分の 1/3 未満
        v["late_near_null"] = (late["mean_rouge_gain"] or 0) < max(
            0.05, (early["mean_rouge_gain"] or 0) / 3
        )
    if noise:
        v["noise_induces_flip"] = (noise["induced_flip_rate"] or 0) >= 0.25
    v["sham_bit_identical"] = summary.get("sham_identical_rate") in (1.0, None)
    v["overall_backbone_closed"] = all(
        v.get(k, False)
        for k in [
            "rouge_gain_ge_0.15",
            "flip_halved",
            "onset_majority_disappeared",
            "late_near_null",
            "noise_induces_flip",
        ]
    )
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_payloads(setting_dir: Path) -> list[dict]:
    """設定ディレクトリ配下 (lxt4/ rnd4/) の全ペア JSON を読む."""
    payloads: list[dict] = []
    for cond in ("lxt4", "rnd4"):
        cdir = setting_dir / cond
        if not cdir.is_dir():
            continue
        for jf in sorted(cdir.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    payloads.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return payloads


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_md(setting: str, summary: dict, verdict: dict) -> str:
    lines = [f"## {setting}", ""]
    lines.append(
        f"- done={summary['n_done']} fresh_flip={summary['n_fresh_flip']} "
        f"clean_correct={summary['n_clean_correct']} excluded={summary['n_excluded']} "
        f"failed={summary['n_failed']} sham_identical={_fmt(summary['sham_identical_rate'])}"
    )
    lines.append(
        f"- baseline mean ROUGE(typo,clean) on fresh flips = "
        f"{_fmt(summary['mean_rouge_typo_vs_clean_on_fresh'])}"
    )
    lines.append("")
    lines.append(
        "| level | dir | n | restoration/induced_flip | ROUGE gain | ROUGE vs clean | onset disappear/induce |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for level in ("early", "mid", "late"):
        lvl = summary["levels"].get(level, {})
        d = lvl.get("denoise")
        if d:
            lines.append(
                f"| {level} | denoise | {d['n']} | {_fmt(d['restoration_rate'])} | "
                f"{_fmt(d['mean_rouge_gain'])} | {_fmt(d['mean_rouge_vs_clean'])} | "
                f"{_fmt(d['onset_disappear_rate'])} |"
            )
        n = lvl.get("noise")
        if n:
            lines.append(
                f"| {level} | noise | {n['n']} | {_fmt(n['induced_flip_rate'])} | "
                f"n/a | {_fmt(n['mean_rouge_vs_clean'])} | {_fmt(n['onset_induced_rate'])} |"
            )
    lines.append("")
    lines.append("**H15 判定:** " + json.dumps(verdict, ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, help="results/exp15 (設定ディレクトリの親)")
    ap.add_argument("--settings", nargs="+", default=None, help="設定名 (既定: 全 run_summary 持ち)")
    ap.add_argument("--out", default=None, help="集計 md の出力先 (既定: stdout)")
    args = ap.parse_args()

    root = Path(args.results_root)
    if args.settings:
        settings = args.settings
    else:
        settings = sorted(
            d.name for d in root.iterdir() if d.is_dir() and (d / "run_summary.json").exists()
        )

    all_summary: dict = {}
    md_parts: list[str] = ["# 実験15 集計 (patch保持 CoT 自由生成)", ""]
    for setting in settings:
        sdir = root / setting
        payloads = load_payloads(sdir)
        summary = summarize(payloads)
        verdict = h15_verdict(summary)
        all_summary[setting] = {"summary": summary, "verdict": verdict}
        md_parts.append(render_md(setting, summary, verdict))
        with open(sdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "verdict": verdict}, f, ensure_ascii=False, indent=2)

    md = "\n".join(md_parts)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        with open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(all_summary, f, ensure_ascii=False, indent=2)
    else:
        print(md)


if __name__ == "__main__":
    main()

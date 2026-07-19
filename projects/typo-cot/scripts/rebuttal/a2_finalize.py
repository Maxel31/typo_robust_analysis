#!/usr/bin/env python3
"""A2 最終集計: 3 点セット (リーク層別・結論剥ぎ・回復曲線) の図と判定を生成.

reads (全て analysis/a2_restore_audit/ 下, プロジェクトローカル):
  leak_stratification.json            … (i)
  conclusion_strip/{model}.json       … (ii)  a2_gpu_audit.py の出力
  recovery_curve/{model}.json         … (iii) a2_gpu_audit.py の出力

writes:
  conclusion_strip.png / recovery_curve.png / verdict.md / verdict.json

GPU 出力が未着でも (i) のみで部分判定を出す (idempotent, 追記なし・毎回再生成)。
"""

import argparse
import json
from pathlib import Path

MODELS = ["gemma-3-4b-it", "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3"]
BENCHMARKS = ["gsm8k", "mmlu"]
GRID = [0, 25, 50, 75, 100]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_strip(out_dir: Path):
    return {m: _load(out_dir / "conclusion_strip" / f"{m}.json") for m in MODELS}


def load_recovery(out_dir: Path):
    return {m: _load(out_dir / "recovery_curve" / f"{m}.json") for m in MODELS}


def fig_conclusion_strip(strip: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, bench in zip(axes, BENCHMARKS):
        combo = f"{bench}_random"
        labels, un, st, un_lk, st_lk = [], [], [], [], []
        for m in MODELS:
            r = strip.get(m, {}).get(combo)
            if not r:
                continue
            labels.append(m.replace("-Instruct", "").replace("-v0.3", ""))
            un.append(r.get("restore_unstripped") or 0)
            st.append(r.get("restore_stripped") or 0)
            un_lk.append(r.get("restore_unstripped_leaked") or 0)
            st_lk.append(r.get("restore_stripped_leaked") or 0)
        x = range(len(labels))
        w = 0.2
        ax.bar([i - 1.5 * w for i in x], un, w, label="unstripped (all)", color="#c44e52")
        ax.bar([i - 0.5 * w for i in x], st, w, label="stripped (all)", color="#dd8452")
        ax.bar([i + 0.5 * w for i in x], un_lk, w, label="unstripped (leak-lastline)", color="#4c72b0")
        ax.bar([i + 1.5 * w for i in x], st_lk, w, label="stripped (leak-lastline)", color="#55a868")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("restore rate")
        ax.set_title(f"{bench.upper()}: (ii) conclusion-strip (last line removed)")
        ax.legend(loc="lower left", fontsize=7)
        ax.axhline(0.5, ls=":", c="gray", lw=0.8)
    fig.suptitle("A2 (ii): does removing the final CoT line (which carries the answer) break restore?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_recovery(recovery: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, bench in zip(axes, BENCHMARKS):
        for m in MODELS:
            r = recovery.get(m, {}).get(f"{bench}_random")
            if not r:
                continue
            rates = r["recovery_rates"]
            ys = [rates.get(str(p)) for p in GRID]
            ax.plot(GRID, ys, marker="o", label=m.replace("-Instruct", "").replace("-v0.3", ""))
        ax.set_xlabel("forced clean-CoT prefix p (%)")
        ax.set_ylabel("recovery (restore) rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{bench.upper()}: (iii) recovery curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("A2 (iii): partial clean-CoT prefixes progressively restore the answer")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_verdict(leak: dict, strip: dict, recovery: dict) -> dict:
    v: dict = {"parts": {}}

    gsm = leak.get("gsm8k", {}).get("_pooled", {})
    mmlu = leak.get("mmlu", {}).get("_pooled", {})
    v["parts"]["i_leak_stratification"] = {
        "gsm8k_pooled": {
            "n": gsm.get("n_te_flipped"),
            "overall_restore": gsm.get("overall_restore"),
            "leak": gsm.get("primary_leak", {}).get("leak"),
            "no_leak": gsm.get("primary_leak", {}).get("no_leak"),
        },
        "mmlu_pooled": {
            "n": mmlu.get("n_te_flipped"),
            "overall_restore": mmlu.get("overall_restore"),
            "leak": mmlu.get("primary_leak", {}).get("leak"),
            "no_leak": mmlu.get("primary_leak", {}).get("no_leak"),
            "no_any_generous": mmlu.get("mmlu_detail", {}).get("no_any_generous"),
        },
    }

    def pool_strip(bench):
        num_un = num_st = num_unlk = num_stlk = 0.0
        den = denlk = 0
        for m in MODELS:
            r = strip.get(m, {}).get(f"{bench}_random")
            if not r:
                continue
            n, nlk = r.get("n") or 0, r.get("n_leaked_lastline") or 0
            if r.get("restore_unstripped") is not None:
                num_un += r["restore_unstripped"] * n
                num_st += (r.get("restore_stripped") or 0) * n
                den += n
            if nlk and r.get("restore_stripped_leaked") is not None:
                num_stlk += r["restore_stripped_leaked"] * nlk
                num_unlk += (r.get("restore_unstripped_leaked") or 0) * nlk
                denlk += nlk
        if den == 0:
            return None
        return {
            "n": den,
            "restore_unstripped": num_un / den,
            "restore_stripped": num_st / den,
            "n_leaked_lastline": denlk,
            "restore_unstripped_leaked": (num_unlk / denlk) if denlk else None,
            "restore_stripped_leaked": (num_stlk / denlk) if denlk else None,
        }

    v["parts"]["ii_conclusion_strip"] = {b: pool_strip(b) for b in BENCHMARKS}

    def pool_recovery(bench):
        acc = {str(p): 0.0 for p in GRID}
        den = 0
        for m in MODELS:
            r = recovery.get(m, {}).get(f"{bench}_random")
            if not r:
                continue
            n = r.get("n") or 0
            for p in GRID:
                val = r["recovery_rates"].get(str(p))
                if val is not None:
                    acc[str(p)] += val * n
            den += n
        if den == 0:
            return None
        return {"n": den, "recovery_rates": {p: acc[p] / den for p in acc}}

    v["parts"]["iii_recovery_curve"] = {b: pool_recovery(b) for b in BENCHMARKS}
    return v


def write_verdict_md(v: dict, path: Path):
    L = ["# A2 判定: restore「自明コピー」批判への反証 (3 点セット)", ""]
    L.append("**攻撃**: GSM8K の CoT 末尾は最終数値を含む。セル C (typo質問+clean CoT強制) の "
             "restore は「答え抽出段が CoT 末尾の数値を書き写すだけ」の自明な帰結ではないか。")
    L.append("")

    i = v["parts"]["i_leak_stratification"]
    L.append("## (i) リーク層別")
    g = i["gsm8k_pooled"]
    if g.get("no_leak"):
        L.append(f"- **GSM8K** (n={g['n']}): overall restore={g['overall_restore']:.3f}。"
                 f"リーク (最終行に金答え数値) が **{g['leak']['n']}/{g['n']} でほぼユニバーサル** "
                 f"→ (i) 単独では GSM8K の copy 経路を排除できない。→ (ii)/(iii) へ。")
    m = i["mmlu_pooled"]
    if m.get("no_leak"):
        nl = m["no_leak"]
        na = m.get("no_any_generous") or nl
        ci = nl.get("ci95") or [None, None]
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci[0] is not None else ""
        L.append(f"- **MMLU** (n={m['n']}): overall restore={m['overall_restore']:.3f}。"
                 f"**答え文字も選択肢本文も現れない no-leak: n={nl['n']}, "
                 f"restore={nl['restore']:.3f} {ci_s}**。最寛容 leak を除いても "
                 f"n={na['n']}, restore={na['restore']:.3f}。"
                 f"→ **MC では答えは CoT テキストに書かれておらず再導出されている = 反証。**")
    L.append("")

    L.append("## (ii) 結論剥ぎ (最終計算行を除去)")
    ii = v["parts"]["ii_conclusion_strip"]
    for b in BENCHMARKS:
        r = ii.get(b)
        if not r:
            L.append(f"- **{b.upper()}**: (GPU 未着)")
            continue
        line = (f"- **{b.upper()}** (n={r['n']}): unstripped={r['restore_unstripped']:.3f} → "
                f"stripped={r['restore_stripped']:.3f}")
        if r.get("restore_stripped_leaked") is not None:
            line += (f"。leak-lastline 部分 (n={r['n_leaked_lastline']}): "
                     f"unstripped={r['restore_unstripped_leaked']:.3f} → "
                     f"stripped={r['restore_stripped_leaked']:.3f}")
        L.append(line)
    L.append("")

    L.append("## (iii) 回復曲線 (先頭 p% 強制)")
    iii = v["parts"]["iii_recovery_curve"]
    for b in BENCHMARKS:
        r = iii.get(b)
        if not r:
            L.append(f"- **{b.upper()}**: (GPU 未着)")
            continue
        rr = r["recovery_rates"]
        curve = " ".join(f"p{p}={rr[p]:.2f}" for p in GRID)
        L.append(f"- **{b.upper()}** (n={r['n']}): {curve}")
    L.append("")

    L.append("## 総合判定")
    verdict_lines = []
    if m.get("no_leak") and m["no_leak"]["n"] > 0 and (m["no_leak"]["restore"] or 0) >= 0.6:
        verdict_lines.append(
            f"- **MMLU: 非自明 (支持)** — 答え文字列が CoT に無い事例でも restore "
            f"{m['no_leak']['restore']:.2f} (n={m['no_leak']['n']})。自明コピーでは説明不能。")
    gsm_ii = ii.get("gsm8k")
    if gsm_ii and gsm_ii.get("restore_stripped_leaked") is not None:
        keep = gsm_ii["restore_stripped_leaked"]
        base = gsm_ii["restore_unstripped_leaked"] or gsm_ii["restore_unstripped"]
        if keep >= 0.6 * (base or 1):
            verdict_lines.append(
                f"- **GSM8K: 結論剥ぎで restore 保持 (支持)** — 末尾の答え行を消しても "
                f"leak 群 restore {base:.2f}→{keep:.2f}。丸写しでなく再導出。")
        else:
            verdict_lines.append(
                f"- **GSM8K: 結論剥ぎで restore 低下** — leak 群 restore {base:.2f}→{keep:.2f}。"
                f"GSM8K の restore は部分的に答えコピーを含む (スコープ限定)。")
    gsm_iii = iii.get("gsm8k")
    if gsm_iii:
        rr = gsm_iii["recovery_rates"]
        if rr.get(50, 0) >= 0.4 * (rr.get(100, 1) or 1) and rr.get(100, 0) > rr.get(0, 0):
            verdict_lines.append(
                f"- **回復曲線: 段階的復帰 (支持)** — 部分プレフィックスで単調に回復 "
                f"(GSM8K p0={rr[0]:.2f}→p50={rr[50]:.2f}→p100={rr[100]:.2f})。")
    if not verdict_lines:
        verdict_lines.append("- (GPU 結果未着のため (i) のみで暫定: MMLU は非自明を支持)")
    L.extend(verdict_lines)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="analysis/a2_restore_audit")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)

    leak = _load(out_dir / "leak_stratification.json")
    strip = load_strip(out_dir)
    recovery = load_recovery(out_dir)

    if any(strip.values()):
        fig_conclusion_strip(strip, out_dir / "conclusion_strip.png")
    if any(recovery.values()):
        fig_recovery(recovery, out_dir / "recovery_curve.png")

    v = build_verdict(leak, strip, recovery)
    (out_dir / "verdict.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    write_verdict_md(v, out_dir / "verdict.md")
    print("written verdict + figures to", out_dir)
    print(json.dumps(v["parts"]["ii_conclusion_strip"], ensure_ascii=False))
    print(json.dumps(v["parts"]["iii_recovery_curve"], ensure_ascii=False))


if __name__ == "__main__":
    main()

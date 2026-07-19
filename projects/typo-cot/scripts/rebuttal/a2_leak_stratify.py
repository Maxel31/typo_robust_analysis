#!/usr/bin/env python3
"""A2 (i): restore のリーク層別 — 「自明コピー」批判への反証 (CPU のみ).

セル C (typo 質問 + clean CoT 強制) の restore 事例を「強制 clean CoT prefix 内に
最終答え文字列が明示的に現れるか否か (leak)」で層別し、restore 率を分けて報告する。

- GSM8K: 金答え数値が prefix に現れるか (anywhere / 最終行 lastline)。
- MMLU:  金答え選択肢文字が現れるか (marker / anywhere) + 選択肢本文 (option_text)。

リークなし事例でも restore が高ければ「答えは CoT に書かれておらず、再導出された」
ことになり自明コピー説を否定する。データは results/exp01_03/*/outcomes.json +
アーカイブ CoT テキスト (読み取り専用)。モデル・GPU 不要。

出力: analysis/a2_restore_audit/
  ├── leak_stratification.json  (全数値)
  ├── leak_stratification.md    (人間可読の表)
  └── leak_stratification.png   (restore by leak stratum の棒グラフ)
"""

import argparse
import json
import logging
import random as _random
from pathlib import Path

from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import truncate_before_answer
from typo_cot.intervention.leak_audit import answer_leak

logging.disable(logging.INFO)

MODELS = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}
BENCHMARKS = ("gsm8k", "mmlu")
PERTURBATIONS = ("random", "importance")


def _flipped(o: dict, cell: str) -> bool:
    return o["answers"][cell].strip() != o["answers"]["A"].strip()


def _restored(o: dict) -> bool:
    # restore = TE flip した事例が C セル (clean CoT 強制) で元の答えに戻った
    return not _flipped(o, "C")


def bootstrap_ci(values: list[int], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    n = len(values)
    if n == 0:
        return (None, None)
    rng = _random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int((alpha / 2) * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def _rate(vals: list[int]):
    return sum(vals) / len(vals) if vals else None


def collect_setting(model_key: str, benchmark: str, archive: Path, results_root: Path):
    """(model, benchmark) の全 perturbation・shard を横断して te_flipped 事例を集める.

    事例キーは (perturbation, shard, sample_id)。C セルの生成は typo 質問に依存する
    ため、同一 sample_id でも perturbation ごとに別事例として扱う。
    """
    base_dir = archive / "baseline" / f"{model_key}_{benchmark}"
    records: list[dict] = []
    seen: set = set()
    for pert in PERTURBATIONS:
        pert_dir = archive / "perturbed" / f"{model_key}_{benchmark}_k4_{pert}"
        if not pert_dir.exists():
            continue
        pairs = {p.sample_id: p for p in load_pair_records(str(base_dir), str(pert_dir))}
        shard_dirs = sorted(results_root.glob(f"{model_key}_{benchmark}_k4_{pert}*"))
        for sd in shard_dirs:
            op = sd / "outcomes.json"
            if not op.exists():
                continue
            outs = json.load(open(op, encoding="utf-8"))
            for o in outs:
                if o["exclude"] or not o["a_correct"]:
                    continue
                if not _flipped(o, "B"):  # te_flip のみ
                    continue
                pair = pairs.get(o["sample_id"])
                if pair is None:
                    continue
                key = (pert, sd.name, o["sample_id"])
                if key in seen:
                    continue
                seen.add(key)
                prefix = truncate_before_answer(pair.cot_clean, benchmark).prefix
                lk = answer_leak(prefix, pair.correct_answer, benchmark, choices=pair.choices_clean)
                records.append(
                    {
                        "sample_id": o["sample_id"],
                        "perturbation": pert,
                        "restore": int(_restored(o)),
                        "leaked": lk.leaked,
                        "numeric_leak": lk.numeric_leak,
                        "numeric_leak_lastline": lk.numeric_leak_lastline,
                        "letter_marker_leak": lk.letter_marker_leak,
                        "letter_anywhere_leak": lk.letter_anywhere_leak,
                        "option_text_leak": lk.option_text_leak,
                    }
                )
    return records


def _stratum(records, key_fn, want):
    vals = [r["restore"] for r in records if key_fn(r) == want]
    lo, hi = bootstrap_ci(vals)
    return {"n": len(vals), "restore": _rate(vals), "ci95": [lo, hi]}


def summarize(records: list[dict], benchmark: str) -> dict:
    overall = [r["restore"] for r in records]
    out = {
        "n_te_flipped": len(records),
        "overall_restore": _rate(overall),
        "overall_ci95": list(bootstrap_ci(overall)),
        "primary_leak": {
            "leak": _stratum(records, lambda r: r["leaked"], True),
            "no_leak": _stratum(records, lambda r: r["leaked"], False),
        },
    }
    if benchmark == "gsm8k":
        def gsm_class(r):
            if r["numeric_leak_lastline"]:
                return "lastline"
            if r["numeric_leak"]:
                return "earlier_only"
            return "absent"

        out["gsm8k_detail"] = {
            k: _stratum(records, gsm_class, k) for k in ("lastline", "earlier_only", "absent")
        }
    else:
        out["mmlu_detail"] = {
            "no_letter_marker": _stratum(records, lambda r: r["letter_marker_leak"], False),
            "no_letter_anywhere": _stratum(records, lambda r: r["letter_anywhere_leak"], False),
            "no_option_text": _stratum(records, lambda r: r["option_text_leak"], False),
            "no_any_generous": _stratum(
                records,
                lambda r: (r["letter_anywhere_leak"] or r["option_text_leak"]),
                False,
            ),
            "leak_rate_letter_marker": _rate([int(r["letter_marker_leak"]) for r in records]),
            "leak_rate_letter_anywhere": _rate([int(r["letter_anywhere_leak"]) for r in records]),
            "leak_rate_option_text": _rate([int(r["option_text_leak"]) for r in records]),
        }
    return out


def fmt_stratum(s: dict) -> str:
    if s["n"] == 0:
        return "n=0  —"
    lo, hi = s["ci95"]
    return f"n={s['n']:<4d} restore={s['restore']:.3f} [{lo:.2f},{hi:.2f}]"


def write_markdown(result: dict, path: Path):
    lines = ["# A2 (i) リーク層別 — restore「自明コピー」批判への反証", ""]
    lines.append(
        "セル C (typo質問+clean CoT強制) の restore を、強制 clean CoT prefix 内に"
        "最終答え文字列が現れるか (leak) で層別。**リークなし事例で restore が高いほど"
        "「答えは CoT に書かれておらず再導出された」= 自明コピー説の反証。**"
    )
    lines.append("")
    for bench in BENCHMARKS:
        lines.append(f"## {bench.upper()}")
        lines.append("")
        for model in MODELS:
            s = result[bench][model]
            if s["n_te_flipped"] == 0:
                continue
            lines.append(f"### {model} (n_te_flipped={s['n_te_flipped']})")
            lines.append(f"- overall restore = {s['overall_restore']:.3f}")
            lines.append(f"- **leak**:    {fmt_stratum(s['primary_leak']['leak'])}")
            lines.append(f"- **no-leak**: {fmt_stratum(s['primary_leak']['no_leak'])}")
            if bench == "gsm8k":
                d = s["gsm8k_detail"]
                lines.append(f"  - lastline:     {fmt_stratum(d['lastline'])}")
                lines.append(f"  - earlier-only: {fmt_stratum(d['earlier_only'])}")
                lines.append(f"  - absent:       {fmt_stratum(d['absent'])}")
            else:
                d = s["mmlu_detail"]
                lines.append(
                    f"  - leak rates: letter-marker={d['leak_rate_letter_marker']:.2f}"
                    f" letter-anywhere={d['leak_rate_letter_anywhere']:.2f}"
                    f" option-text={d['leak_rate_option_text']:.2f}"
                )
                lines.append(f"  - no-letter-marker:   {fmt_stratum(d['no_letter_marker'])}")
                lines.append(f"  - no-letter-anywhere: {fmt_stratum(d['no_letter_anywhere'])}")
                lines.append(f"  - no-option-text:     {fmt_stratum(d['no_option_text'])}")
                lines.append(f"  - no-any(generous):   {fmt_stratum(d['no_any_generous'])}")
            lines.append("")
        pooled = result[bench]["_pooled"]
        lines.append(f"### {bench.upper()} pooled over models")
        lines.append(f"- overall restore = {pooled['overall_restore']:.3f} (n={pooled['n_te_flipped']})")
        lines.append(f"- leak:    {fmt_stratum(pooled['primary_leak']['leak'])}")
        lines.append(f"- no-leak: {fmt_stratum(pooled['primary_leak']['no_leak'])}")
        if bench == "mmlu":
            lines.append(f"- no-any(generous): {fmt_stratum(pooled['mmlu_detail']['no_any_generous'])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(result: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, bench in zip(axes, BENCHMARKS):
        models = [m for m in MODELS if result[bench][m]["n_te_flipped"] > 0]
        x = range(len(models))
        leak_r = [result[bench][m]["primary_leak"]["leak"]["restore"] or 0 for m in models]
        noleak_r = [result[bench][m]["primary_leak"]["no_leak"]["restore"] or 0 for m in models]
        leak_n = [result[bench][m]["primary_leak"]["leak"]["n"] for m in models]
        noleak_n = [result[bench][m]["primary_leak"]["no_leak"]["n"] for m in models]
        w = 0.38
        ax.bar([i - w / 2 for i in x], leak_r, w, label="leak", color="#c44e52")
        ax.bar([i + w / 2 for i in x], noleak_r, w, label="no-leak", color="#4c72b0")
        for i, (bn, r) in enumerate(zip(leak_n, leak_r)):
            ax.text(i - w / 2, r + 0.02, f"n={bn}", ha="center", fontsize=8)
        for i, (bn, r) in enumerate(zip(noleak_n, noleak_r)):
            ax.text(i + w / 2, r + 0.02, f"n={bn}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.replace("-Instruct", "").replace("-v0.3", "") for m in models],
                           rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("restore rate")
        ax.set_title(f"{bench.upper()}: restore by answer-string leak")
        ax.legend(loc="lower right", fontsize=8)
        ax.axhline(0.5, ls=":", c="gray", lw=0.8)
    fig.suptitle("A2 (i): cell-C restore stratified by whether the gold answer leaks into the forced clean CoT")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive",
        default="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs",
    )
    ap.add_argument("--results-root", default="results/exp01_03")
    ap.add_argument("--output-dir", default="analysis/a2_restore_audit")
    args = ap.parse_args()

    archive = Path(args.archive)
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {}
    for bench in BENCHMARKS:
        result[bench] = {}
        pooled_records: list[dict] = []
        for model in MODELS:
            recs = collect_setting(model, bench, archive, results_root)
            result[bench][model] = summarize(recs, bench)
            pooled_records.extend(recs)
        result[bench]["_pooled"] = summarize(pooled_records, bench)

    (out_dir / "leak_stratification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(result, out_dir / "leak_stratification.md")
    make_figure(result, out_dir / "leak_stratification.png")

    for bench in BENCHMARKS:
        p = result[bench]["_pooled"]
        print(f"[{bench}] pooled n={p['n_te_flipped']} overall={p['overall_restore']:.3f} "
              f"leak(n={p['primary_leak']['leak']['n']},r={p['primary_leak']['leak']['restore']}) "
              f"no-leak(n={p['primary_leak']['no_leak']['n']},"
              f"r={p['primary_leak']['no_leak']['restore']})")
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()

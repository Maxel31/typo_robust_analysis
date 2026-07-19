#!/usr/bin/env python3
"""A2 (ii)+(iii): 結論剥ぎ・回復曲線 — restore「自明コピー」批判への反証 (GPU).

flip 事例 (typo 質問で答えが変わった included サンプル) について、セル C
(typo 質問 + clean CoT 強制) の変種を teacher-forcing 生成し restore を再測定する。
モデルは 1 回だけロードして 1 モデルの全 (benchmark, perturbation) を処理する
(GPU 取得回数を最小化)。GPU は必ず run_with_gpu.sh 経由で。

(ii) 結論剥ぎ: C セルの clean CoT 末尾行 (最終計算/読み上げ行) を除去して強制。
     GSM8K では末尾行に金答え数値が載る (leak) ので、これを消しても restore が
     保たれれば「テキストが運ぶのは結論の丸写しでなく再導出可能な推論内容」を支持。
(iii) 回復曲線: clean CoT prefix の先頭 p% (p=0/25/50/75/100) を強制し自由生成。
     部分プレフィックスで段階的に復帰するなら丸写しでは説明不能。

出力 (idempotent, combo 単位で追記・skip):
  analysis/a2_restore_audit/conclusion_strip/{model}.json
  analysis/a2_restore_audit/recovery_curve/{model}.json

使用例 (GPU ヘルパー経由):
  bash /.../tmp/gpu-locks/run_with_gpu.sh \\
    uv run python scripts/rebuttal/a2_gpu_audit.py --model gemma-3-4b-it
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.intervention.archive_loader import load_pair_records
from typo_cot.intervention.cell_builder import build_cell_inputs
from typo_cot.intervention.leak_audit import (
    RECOVERY_GRID,
    answer_leak,
    cut_prefix_by_fraction,
    strip_conclusion,
)
from typo_cot.models.wrapper import ModelWrapper

logging.disable(logging.INFO)

MODELS = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}
BENCHMARKS = ("gsm8k", "mmlu")


def _flipped(o: dict, cell: str) -> bool:
    return o["answers"][cell].strip() != o["answers"]["A"].strip()


def select_flip_cases(model_key, benchmark, perturbation, archive, results_root, n):
    """te_flip した included 事例を sample_id 昇順で最大 n 件返す (pair 付き)."""
    base_dir = archive / "baseline" / f"{model_key}_{benchmark}"
    pert_dir = archive / "perturbed" / f"{model_key}_{benchmark}_k4_{perturbation}"
    if not pert_dir.exists():
        return []
    pairs = {p.sample_id: p for p in load_pair_records(str(base_dir), str(pert_dir))}
    cases = []
    for sd in sorted(results_root.glob(f"{model_key}_{benchmark}_k4_{perturbation}*")):
        op = sd / "outcomes.json"
        if not op.exists():
            continue
        for o in json.load(open(op, encoding="utf-8")):
            if o["exclude"] or not o["a_correct"] or not _flipped(o, "B"):
                continue
            pair = pairs.get(o["sample_id"])
            if pair is None:
                continue
            cases.append((o, pair))
    cases.sort(key=lambda c: c[0]["sample_id"])
    return cases[:n]


def _gen(wrapper, prompts, max_new_tokens, batch_size):
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        res = wrapper.generate_batch(chunk, max_new_tokens=max_new_tokens,
                                     temperature=0.0, do_sample=False)
        out.extend(r.generated_text for r in res)
    return out


def run_conclusion_strip(wrapper, cases, benchmark, batch_size, max_new_tokens, mode):
    """(ii) C セル (unstripped) と C_strip (末尾行除去) を生成して restore を比較."""
    extractor = create_extractor(benchmark)
    tasks, meta = [], []
    for _o, pair in cases:
        cells = build_cell_inputs(pair)
        typo_prompt = cells.prompts["C"]
        clean_prefix = cells.forced_cots["C"]
        stripped = strip_conclusion(clean_prefix, mode=mode)
        lk = answer_leak(clean_prefix, pair.correct_answer, benchmark, choices=pair.choices_clean)
        leaked_lastline = (
            bool(lk.numeric_leak_lastline) if benchmark in ("gsm8k", "math") else bool(lk.leaked)
        )
        idx = len(meta)
        meta.append({
            "sample_id": pair.sample_id,
            "correct_answer": pair.correct_answer,
            "leaked_lastline": leaked_lastline,
            "stripped_empty": stripped.strip() == "",
        })
        tasks.append((idx, "C", typo_prompt + clean_prefix))
        tasks.append((idx, "Cs", typo_prompt + stripped))
    gens = _gen(wrapper, [t[2] for t in tasks], max_new_tokens, batch_size)
    ans = {}
    for (idx, var, _), g in zip(tasks, gens):
        ans[(idx, var)] = extractor.extract(g).extracted_answer.strip()

    per = []
    for i, m in enumerate(meta):
        ru = extractor.is_correct(ans[(i, "C")], m["correct_answer"])
        rs = extractor.is_correct(ans[(i, "Cs")], m["correct_answer"])
        per.append({**m, "ans_C": ans[(i, "C")], "ans_Cs": ans[(i, "Cs")],
                    "restore_unstripped": int(ru), "restore_stripped": int(rs)})

    def rate(items, key, filt=lambda r: True):
        v = [r[key] for r in items if filt(r)]
        return (sum(v) / len(v) if v else None), len(v)

    ru_all, n = rate(per, "restore_unstripped")
    rs_all, _ = rate(per, "restore_stripped")
    # leak (lastline) 部分集合での stripped restore が本命 (copy 経路を潰した群)
    rs_leak, n_leak = rate(per, "restore_stripped", lambda r: r["leaked_lastline"])
    ru_leak, _ = rate(per, "restore_unstripped", lambda r: r["leaked_lastline"])
    return {
        "n": n,
        "restore_unstripped": ru_all,
        "restore_stripped": rs_all,
        "n_leaked_lastline": n_leak,
        "restore_unstripped_leaked": ru_leak,
        "restore_stripped_leaked": rs_leak,
        "strip_mode": mode,
        "per_case": per,
    }


def run_recovery_curve(wrapper, cases, benchmark, batch_size, max_new_tokens, grid):
    """(iii) typo 質問下で clean CoT prefix の先頭 p% を強制し自由生成 → 回復率曲線."""
    extractor = create_extractor(benchmark)
    tasks, meta = [], []
    for _o, pair in cases:
        cells = build_cell_inputs(pair)
        typo_prompt = cells.prompts["C"]
        clean_prefix = cells.forced_cots["C"]
        idx = len(meta)
        meta.append({"sample_id": pair.sample_id, "correct_answer": pair.correct_answer})
        for p in grid:
            part = cut_prefix_by_fraction(clean_prefix, p)
            tasks.append((idx, p, typo_prompt + part))
    gens = _gen(wrapper, [t[2] for t in tasks], max_new_tokens, batch_size)
    rec = {}
    for (idx, p, _), g in zip(tasks, gens):
        rec[(idx, p)] = extractor.extract(g).extracted_answer.strip()

    per = []
    for i, m in enumerate(meta):
        curve = {str(p): int(extractor.is_correct(rec[(i, p)], m["correct_answer"])) for p in grid}
        per.append({**m, "curve": curve})
    recovery_rates = {
        str(p): (sum(r["curve"][str(p)] for r in per) / len(per) if per else None) for p in grid
    }
    return {"n": len(per), "grid": list(grid), "recovery_rates": recovery_rates, "per_case": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--archive",
                    default="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs")
    ap.add_argument("--results-root", default="results/exp01_03")
    ap.add_argument("--output-dir", default="analysis/a2_restore_audit")
    ap.add_argument("--perturbations", default="random",
                    help="カンマ区切り (random,importance)")
    ap.add_argument("--n", type=int, default=150, help="flip 事例数の上限/combo")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--strip-mode", default="last_line",
                    choices=["last_line", "last_sentence"])
    ap.add_argument("--max-new-tokens-strip", type=int, default=16)
    ap.add_argument("--max-new-tokens-recovery", type=int, default=256)
    ap.add_argument("--skip-strip", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    args = ap.parse_args()

    archive = Path(args.archive)
    results_root = Path(args.results_root)
    perts = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    model_id = MODELS[args.model]

    strip_path = Path(args.output_dir) / "conclusion_strip" / f"{args.model}.json"
    rec_path = Path(args.output_dir) / "recovery_curve" / f"{args.model}.json"
    strip_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    strip_out = json.loads(strip_path.read_text()) if strip_path.exists() else {}
    rec_out = json.loads(rec_path.read_text()) if rec_path.exists() else {}

    todo = []
    for bench in BENCHMARKS:
        for pert in perts:
            combo = f"{bench}_{pert}"
            need_strip = not args.skip_strip and combo not in strip_out
            need_rec = not args.skip_recovery and combo not in rec_out
            if need_strip or need_rec:
                todo.append((bench, pert, combo, need_strip, need_rec))
    if not todo:
        print(f"[{args.model}] nothing to do (all combos present)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model}] loading {model_id} on {device} for {len(todo)} combo(s)")
    wrapper = ModelWrapper(model_name=model_id, device=device, dtype=torch.bfloat16)
    _ = wrapper.model

    for bench, pert, combo, need_strip, need_rec in todo:
        cases = select_flip_cases(args.model, bench, pert, archive, results_root, args.n)
        print(f"[{args.model}] {combo}: {len(cases)} flip cases")
        if not cases:
            continue
        if need_strip:
            r = run_conclusion_strip(wrapper, cases, bench, args.batch_size,
                                     args.max_new_tokens_strip, args.strip_mode)
            r["timestamp"] = datetime.now().isoformat()
            strip_out[combo] = r
            strip_path.write_text(json.dumps(strip_out, ensure_ascii=False, indent=1))
            print(f"  strip: unstripped={r['restore_unstripped']} "
                  f"stripped={r['restore_stripped']} "
                  f"| leaked-lastline n={r['n_leaked_lastline']} "
                  f"stripped_restore={r['restore_stripped_leaked']}")
        if need_rec:
            r = run_recovery_curve(wrapper, cases, bench, args.batch_size,
                                   args.max_new_tokens_recovery, RECOVERY_GRID)
            r["timestamp"] = datetime.now().isoformat()
            rec_out[combo] = r
            rec_path.write_text(json.dumps(rec_out, ensure_ascii=False, indent=1))
            print(f"  recovery: {r['recovery_rates']}")

    print(f"[{args.model}] done -> {strip_path} , {rec_path}")


if __name__ == "__main__":
    main()

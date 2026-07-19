"""Qwen exp01_03 の per-sample dedup-on `exclude` を再生成する.

背景: Track C が multi_trigger 過剰除外バグを改修 (exp-01-03 commit 4052b2c,
build_cell_inputs(dedup_same_answer_triggers=True))。既存 outcomes.json の
`exclude` は改修前 (Qwen で n_included 崩壊)。実験11 の per-sample 連鎖媒介では
正しい included 判定が必要なため、dedup-on の exclude を再計算して上書きマップを
出力する。flip(answers B vs A)・a_correct は dedup で不変なので exclude のみ再計算。

出力: qwen_dedup_exclude.json = {setting_base: {sample_id: exclude_bool}}
  setting_base は __pN シャードを結合した pooled 設定名。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

EXP01 = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-01-03-transplant/projects/typo-cot"
sys.path.insert(0, EXP01 + "/src")
from typo_cot.intervention.archive_loader import load_pair_records  # noqa: E402
from typo_cot.intervention.cell_builder import build_cell_inputs  # noqa: E402

RESULTS = Path(EXP01 + "/results/exp01_03")
OUT = Path(__file__).resolve().parent / "qwen_dedup_exclude.json"
PREFIX = "Qwen"


def main():
    dirs = sorted(d for d in RESULTS.iterdir()
                  if d.is_dir() and d.name.startswith(PREFIX) and (d / "summary.json").exists())
    groups = defaultdict(list)
    for d in dirs:
        groups[d.name.split("__p")[0]].append(d)

    out = {}
    for base, ds in sorted(groups.items()):
        cfg = json.load(open(sorted(ds)[0] / "summary.json"))["config"]
        pairs = load_pair_records(cfg["baseline_dir"], cfg["perturbed_dir"])
        pair_by_id = {p.sample_id: p for p in pairs}
        excl = {}
        n_pre = n_post = 0
        for d in sorted(ds):
            raw = json.load(open(d / "outcomes.json"))
            for o in raw:
                sid = o["sample_id"]
                n_pre += int(o["exclude"])
                pair = pair_by_id.get(sid)
                if pair is None:
                    excl[sid] = bool(o["exclude"])  # ペア無しは据え置き
                else:
                    ci = build_cell_inputs(pair, dedup_same_answer_triggers=True)
                    excl[sid] = bool(ci.exclude)
                n_post += int(excl[sid])
        out[base] = excl
        print(f"{base:44s} n={len(excl):5d} excl_pre(sum)={n_pre:5d} excl_dedup_on(sum)={n_post:5d}")
    json.dump(out, open(OUT, "w"), indent=0)
    print("saved:", OUT)


if __name__ == "__main__":
    main()

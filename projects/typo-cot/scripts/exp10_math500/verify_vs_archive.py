#!/usr/bin/env python3
"""実験10②: 新規再生成した MATH-500 clean 結果をアーカイブ(2026-05-21生成)と照合する.

比較項目:
- 精度 (summary.overall_metrics.accuracy) の差
- 答え抽出成功率 (extracted_answer が非空の割合) の差
- sample_id 共通部分での抽出答え一致率 / is_correct 一致率 / 生成文完全一致率

greedy / seed=42 / 同一プロンプトなので高い一致が期待される。乖離が大きい場合は
原因調査してから VERIFY_OK を作成すること。

使い方:
  uv run --no-sync python scripts/exp10_math500/verify_vs_archive.py \
    --model_short gemma-3-1b-it [--acc_tol 0.03]
終了コード: 0=許容内 / 2=乖離大(調査要)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE_BASELINE = Path("/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_results(directory: Path) -> tuple[list[dict], dict]:
    with open(directory / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(directory / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    return results, summary


def extraction_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    ok = sum(1 for r in results if (r.get("extracted_answer") or "").strip() != "")
    return ok / len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_short", required=True, help="例: gemma-3-1b-it")
    parser.add_argument(
        "--new_dir",
        default=None,
        help="新規生成ディレクトリ (省略時: outputs/baseline/<model_short>_math)",
    )
    parser.add_argument("--acc_tol", type=float, default=0.03, help="精度差の許容幅")
    parser.add_argument("--extr_tol", type=float, default=0.03, help="抽出成功率差の許容幅")
    args = parser.parse_args()

    new_dir = (
        Path(args.new_dir)
        if args.new_dir
        else PROJECT_ROOT / "outputs" / "baseline" / f"{args.model_short}_math"
    )
    arch_dir = ARCHIVE_BASELINE / f"{args.model_short}_math"

    new_results, new_summary = load_results(new_dir)
    arch_results, arch_summary = load_results(arch_dir)

    new_acc = new_summary["overall_metrics"]["accuracy"]
    arch_acc = arch_summary["overall_metrics"]["accuracy"]
    new_extr = extraction_rate(new_results)
    arch_extr = extraction_rate(arch_results)

    new_by_id = {r["sample_id"]: r for r in new_results}
    arch_by_id = {r["sample_id"]: r for r in arch_results}
    common = sorted(set(new_by_id) & set(arch_by_id))

    same_answer = sum(
        1
        for sid in common
        if (new_by_id[sid].get("extracted_answer") or "")
        == (arch_by_id[sid].get("extracted_answer") or "")
    )
    same_correct = sum(
        1
        for sid in common
        if bool(new_by_id[sid].get("is_correct")) == bool(arch_by_id[sid].get("is_correct"))
    )
    same_text = sum(
        1
        for sid in common
        if (new_by_id[sid].get("generated_text") or "")
        == (arch_by_id[sid].get("generated_text") or "")
    )

    report = {
        "model_short": args.model_short,
        "new_dir": str(new_dir),
        "archive_dir": str(arch_dir),
        "timestamp": datetime.now().isoformat(),
        "n_new": len(new_results),
        "n_archive": len(arch_results),
        "n_common": len(common),
        "accuracy": {"new": new_acc, "archive": arch_acc, "delta": new_acc - arch_acc},
        "extraction_rate": {
            "new": new_extr,
            "archive": arch_extr,
            "delta": new_extr - arch_extr,
        },
        "per_sample_agreement": {
            "extracted_answer": same_answer / len(common) if common else 0,
            "is_correct": same_correct / len(common) if common else 0,
            "generated_text_exact": same_text / len(common) if common else 0,
        },
        "tolerances": {"acc_tol": args.acc_tol, "extr_tol": args.extr_tol},
    }

    ok = (
        abs(report["accuracy"]["delta"]) <= args.acc_tol
        and abs(report["extraction_rate"]["delta"]) <= args.extr_tol
        and len(new_results) == len(arch_results)
    )
    report["verdict"] = "OK" if ok else "DIVERGED"

    out_path = PROJECT_ROOT / "logs" / "exp10_math500" / f"verify_{args.model_short}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nverdict: {report['verdict']} (report: {out_path})", file=sys.stderr)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

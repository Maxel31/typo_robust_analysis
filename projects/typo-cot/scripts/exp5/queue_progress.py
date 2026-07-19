#!/usr/bin/env python3
"""実験5 生成キューの進捗 JSON を書き出すスキャナ.

run_generation_queue.sh がシャード完了ごとに呼ぶほか、手動での監視にも使える:

  uv run --no-sync python scripts/exp5/queue_progress.py --print

ステータス判定 (ステートレス; ディレクトリの実態から再構成):
  done            results/exp5/perturbed/<name>/summary.json が存在
  running         logs/exp5/queue_state/<name>.running が存在
  failed          attempts >= max_attempts かつ未完了
  queued          データセット構築済みで生成待ち
  dataset_pending データセット (matched_stats.json) 未構築
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]


def load_settings(path: Path) -> list[tuple[str, str]]:
    settings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        model, bench = line.split()
        settings.append((model, bench))
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="実験5 キュー進捗 JSON")
    parser.add_argument("--settings", type=Path, default=PROJ / "scripts/exp5/settings_25.txt")
    parser.add_argument("--data_dir", type=Path, default=PROJ / "data/exp5/matched_rnd")
    parser.add_argument("--out_dir", type=Path, default=PROJ / "results/exp5/perturbed")
    parser.add_argument("--state_dir", type=Path, default=PROJ / "logs/exp5/queue_state")
    parser.add_argument("--progress", type=Path, default=PROJ / "logs/exp5/queue_progress.json")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--print", dest="do_print", action="store_true")
    args = parser.parse_args()

    entries: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for model, bench in load_settings(args.settings):
        name = f"{model}_{bench}_k4_matched_rnd"
        summary_path = args.out_dir / name / "summary.json"
        stats_path = args.data_dir / name / "matched_stats.json"
        attempts_path = args.state_dir / f"{name}.attempts"
        running_path = args.state_dir / f"{name}.running"

        attempts = 0
        if attempts_path.exists():
            try:
                attempts = int(attempts_path.read_text().strip() or 0)
            except ValueError:
                attempts = 0

        entry: dict = {
            "model": model,
            "benchmark": bench,
            "dataset_ready": stats_path.exists(),
            "attempts": attempts,
            "log": str(PROJ / "logs/exp5/gen" / f"{name}.log"),
            "output_dir": str(args.out_dir / name),
            "accuracy": None,
        }
        if summary_path.exists():
            entry["status"] = "done"
            try:
                m = json.loads(summary_path.read_text())["overall_metrics"]
                entry["accuracy"] = m["accuracy"]
                entry["total_correct"] = m["total_correct"]
                entry["total_samples"] = m["total_samples"]
            except Exception:  # noqa: BLE001
                pass
        elif running_path.exists():
            entry["status"] = "running"
        elif not stats_path.exists():
            entry["status"] = "dataset_pending"
        elif attempts >= args.max_attempts:
            entry["status"] = "failed"
        else:
            entry["status"] = "queued"
        entries[name] = entry
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    progress = {
        "updated_at": datetime.now().isoformat(),
        "counts": counts,
        "settings": entries,
    }
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.progress.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.progress)
    if args.do_print:
        print(json.dumps({"updated_at": progress["updated_at"], "counts": counts}, ensure_ascii=False))
        for name, e in entries.items():
            acc = f" acc={e['accuracy']:.4f}" if e["accuracy"] is not None else ""
            print(f"  {e['status']:15s} {name}{acc}")


if __name__ == "__main__":
    main()

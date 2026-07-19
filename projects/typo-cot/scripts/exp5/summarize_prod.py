#!/usr/bin/env python3
"""実験5: 本番 25 設定の構築・生成結果の検証サマリ.

各設定について
- matched_stats.json の SMD 4 変数 / class_match 率 / 緩和ラダー / unmatched 率
- 生成結果 (results.json / summary.json) のスキーマ・件数・摂動トークン数
- アーカイブ clean / LXT-4 との精度比較 (共通 sample_id 上)
を検証し、25 行のサマリテーブルを results/prod/exp5/ に書き出す。

使用例:
  uv run --no-sync python scripts/exp5/summarize_prod.py \
    --archive /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026 \
    --output_dir results/prod/exp5
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMD_VARS = ["char_len", "zipf", "split_increment", "centrality"]
# バランス判定: |SMD| < 0.1 良好 / < 0.25 許容 (Cohen の慣行に準拠)
SMD_WARN = 0.10
SMD_ANOM = 0.25
EXPECTED_K = 4


def load_settings(path: Path) -> list[tuple[str, str]]:
    settings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        model, bench = line.split()
        settings.append((model, bench))
    return settings


def load_correct_map(path: Path) -> dict[str, bool]:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return {r["sample_id"]: bool(r.get("is_correct", False)) for r in results}


def verify_setting(
    model: str, bench: str, archive: Path
) -> tuple[dict, list[str]]:
    name = f"{model}_{bench}_k4_matched_rnd"
    flags: list[str] = []
    row: dict = {"setting": name, "model": model, "benchmark": bench}

    # --- 1) データセット構築 (matched_stats.json) ---
    stats_path = PROJECT_ROOT / "data/exp5/matched_rnd" / name / "matched_stats.json"
    stats = json.load(open(stats_path, encoding="utf-8"))
    smd_table = stats["smd_table"]
    row["embedding_enabled"] = stats.get("embedding_enabled")
    if not stats.get("embedding_enabled"):
        flags.append("embedding_disabled (5変数版でない)")
    row["n_targets"] = smd_table["n_targets"]
    row["n_matched"] = smd_table["n_matched"]
    for var in SMD_VARS:
        val = smd_table["smd"][var]
        row[f"smd_{var}"] = round(val, 4)
        if abs(val) >= SMD_ANOM:
            flags.append(f"smd_{var}={val:+.3f} (|SMD|>={SMD_ANOM})")
    row["class_match_rate"] = round(smd_table["class_match_rate"], 4)
    if smd_table["class_match_rate"] < 0.99:
        flags.append(f"class_match_rate={smd_table['class_match_rate']:.4f} (<0.99)")
    relax = smd_table["relaxation_rates"]
    row["unmatched_rate"] = round(relax.get("unmatched", 0.0), 6)
    if relax.get("unmatched", 0.0) > 0:
        flags.append(f"unmatched_rate={relax['unmatched']:.4f} (>0)")
    for level in ["exact", "no_centrality", "caliper", "class_len", "any"]:
        row[f"relax_{level}"] = round(relax.get(level, 0.0), 4)
    row["applied_from_matched_rate"] = round(
        stats.get("applied_from_matched_rate", float("nan")), 4
    )
    if stats.get("applied_from_matched_rate", 1.0) < 0.85:
        flags.append(
            f"applied_from_matched_rate={stats['applied_from_matched_rate']:.3f} (<0.85)"
        )

    # --- 2) 生成結果の検証 ---
    out_dir = PROJECT_ROOT / "results/exp5/perturbed" / name
    summary = json.load(open(out_dir / "summary.json", encoding="utf-8"))
    results = json.load(open(out_dir / "results.json", encoding="utf-8"))
    row["gen_n"] = len(results)
    row["acc_matched_rnd"] = round(summary["overall_metrics"]["accuracy"], 4)
    n_bad_k = sum(1 for r in results if len(r.get("perturbed_tokens", {})) != EXPECTED_K)
    row["n_samples_kne4"] = n_bad_k
    if n_bad_k:
        flags.append(f"{n_bad_k} samples with perturbed_tokens != {EXPECTED_K}")
    sids = [r["sample_id"] for r in results]
    if len(set(sids)) != len(sids):
        flags.append("duplicate sample_id in results.json")

    # --- 3) アーカイブ clean / LXT-4 との比較 ---
    clean_path = archive / "outputs/baseline" / f"{model}_{bench}" / "results.json"
    lxt_path = (
        archive / "outputs/perturbed" / f"{model}_{bench}_k4_importance" / "results.json"
    )
    clean = load_correct_map(clean_path)
    lxt = load_correct_map(lxt_path)
    matched = {r["sample_id"]: bool(r.get("is_correct", False)) for r in results}

    # スキーマ互換 (アーカイブ LXT-4 の results.json とキー一致)
    arch_entry_keys = set(json.load(open(lxt_path, encoding="utf-8"))[0].keys())
    new_entry_keys = set(results[0].keys())
    if arch_entry_keys != new_entry_keys:
        flags.append(
            f"schema keys differ from archive: +{sorted(new_entry_keys - arch_entry_keys)}"
            f" -{sorted(arch_entry_keys - new_entry_keys)}"
        )

    common = [sid for sid in clean if sid in lxt and sid in matched]
    row["n_common"] = len(common)
    if len(common) != len(results):
        flags.append(f"common ids {len(common)} != generated {len(results)}")

    def acc(m: dict[str, bool]) -> float:
        return sum(m[sid] for sid in common) / len(common) if common else float("nan")

    row["acc_clean"] = round(acc(clean), 4)
    row["acc_lxt4"] = round(acc(lxt), 4)
    row["drop_matched"] = round(row["acc_clean"] - acc(matched), 4)
    row["drop_lxt4"] = round(row["acc_clean"] - row["acc_lxt4"], 4)
    # 期待: LXT-4 (重要トークン摂動) の方が Matched-Rnd (双子語ランダム) より精度を下げる
    row["delta_drop"] = round(row["drop_lxt4"] - row["drop_matched"], 4)
    if acc(matched) < row["acc_lxt4"] - 0.02:
        flags.append(
            f"matched acc {acc(matched):.4f} < lxt4 acc {row['acc_lxt4']:.4f} - 0.02"
        )

    row["flags"] = "; ".join(flags)
    return row, flags


def main() -> None:
    parser = argparse.ArgumentParser(description="実験5 本番25設定の検証サマリ")
    parser.add_argument(
        "--archive",
        type=str,
        default="/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026",
    )
    parser.add_argument("--output_dir", type=str, default="results/prod/exp5")
    parser.add_argument(
        "--settings",
        type=str,
        default=str(PROJECT_ROOT / "scripts/exp5/settings_25.txt"),
    )
    args = parser.parse_args()

    archive = Path(args.archive)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_flags: dict[str, list[str]] = {}
    for model, bench in load_settings(Path(args.settings)):
        row, flags = verify_setting(model, bench, archive)
        rows.append(row)
        if flags:
            all_flags[row["setting"]] = flags
        status = "FLAG" if flags else "ok"
        print(f"[{status}] {row['setting']}: acc={row['acc_matched_rnd']}", flush=True)

    fieldnames = list(rows[0].keys())
    with open(out_dir / "matched_build_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "matched_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_settings": len(rows),
                "smd_warn_threshold": SMD_WARN,
                "smd_anomaly_threshold": SMD_ANOM,
                "flagged_settings": all_flags,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nwrote {out_dir / 'matched_build_summary.csv'} ({len(rows)} rows)")
    print(f"flagged settings: {len(all_flags)}")
    for name, flags in all_flags.items():
        for fl in flags:
            print(f"  {name}: {fl}")


if __name__ == "__main__":
    main()

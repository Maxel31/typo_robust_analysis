#!/usr/bin/env python3
"""Step 0: アーカイブ横断の統合テーブル (master table) 構築スクリプト.

アーカイブ (configs/paths.yaml の archive_outputs) から
25 設定 (5 モデル × 5 ベンチマーク) × 6 条件 (clean + lxt1/2/4/8 + random4) の
生成ログ・R_Q/R_C 帰属・flip 判定・CoT 指標を読み、
`data/{model}/{benchmark}/{condition}.parquet` に統合する。

- 移行元ファイルの sha256 を data/master_manifest.json に記録し、
  `--verify` で同一性を再検証できる (アーカイブには一切書き込まない)
- seed / prompt_id は configs/registry.yaml の凍結値を使用

使い方:
    uv run python scripts/step0_build_master_table.py \
        --models gemma-3-4b-it --benchmarks gsm8k mmlu   # スモーク
    uv run python scripts/step0_build_master_table.py    # 全 25 設定
    uv run python scripts/step0_build_master_table.py --verify
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typo_cot.data.archive_reader import (  # noqa: E402
    analysis_condition_dir,
    baseline_dir,
    load_analysis_sample_results,
    load_json,
    load_paths_config,
    perturbed_dir,
    sha256_file,
)
from typo_cot.data.master_builder import (  # noqa: E402
    build_condition_df,
    sample_metrics_from_analysis,
)
from typo_cot.data.master_table import (  # noqa: E402
    CONDITIONS,
    master_parquet_path,
    write_condition_parquet,
)
from typo_cot.registry import load_registry, prompt_id_for, validate_registry  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 0 統合テーブル構築")
    p.add_argument("--paths", type=Path, default=PROJECT_ROOT / "configs" / "paths.yaml")
    p.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs" / "registry.yaml")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--manifest", type=Path, default=None, help="既定: {out}/master_manifest.json")
    p.add_argument("--models", nargs="+", default=None, help="既定: レジストリの全モデル")
    p.add_argument("--benchmarks", nargs="+", default=None, help="既定: レジストリの全ベンチマーク")
    p.add_argument(
        "--verify",
        action="store_true",
        help="構築せず、manifest 記載の移行元ハッシュと行数を再検証する",
    )
    return p.parse_args()


def build_setting(
    outputs_root: Path,
    analysis_root: Path,
    out_root: Path,
    model: str,
    benchmark: str,
    seed: int,
    prompt_id: str,
) -> list[dict]:
    """1 設定 (model, benchmark) の全条件を parquet 化し、manifest エントリを返す."""
    entries: list[dict] = []
    bdir = baseline_dir(outputs_root, model, benchmark)
    baseline_path = bdir / "results.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline results.json not found: {baseline_path}")
    baseline_results = load_json(baseline_path)

    for condition in CONDITIONS:
        sources: list[dict] = [
            {"path": str(baseline_path), "sha256": sha256_file(baseline_path)}
        ]
        if condition == "clean":
            perturbed_results = None
            sample_metrics = None
            source_path = str(baseline_path)
        else:
            pdir = perturbed_dir(outputs_root, model, benchmark, condition)
            ppath = pdir / "results.json"
            if not ppath.exists():
                logger.warning(f"  [skip] {model} x {benchmark} x {condition}: {ppath} なし")
                continue
            perturbed_results = load_json(ppath)
            sources.append({"path": str(ppath), "sha256": sha256_file(ppath)})
            source_path = str(ppath)
            sample_results = load_analysis_sample_results(
                analysis_root, model, benchmark, condition
            )
            if sample_results is None:
                logger.warning(
                    f"  [warn] analysis full_results.json なし: "
                    f"{model} x {benchmark} x {condition} (flip/CoT指標は NA)"
                )
                sample_metrics = None
            else:
                sample_metrics = sample_metrics_from_analysis(sample_results)
                apath = (
                    analysis_condition_dir(analysis_root, model, benchmark, condition)
                    / "full_results.json"
                )
                sources.append({"path": str(apath), "sha256": sha256_file(apath)})

        df = build_condition_df(
            baseline_results=baseline_results,
            perturbed_results=perturbed_results,
            sample_metrics=sample_metrics,
            model=model,
            benchmark=benchmark,
            condition=condition,
            seed=seed,
            prompt_id=prompt_id,
            source_path=source_path,
        )
        path = write_condition_parquet(df, out_root)
        flip_nonnull = int(df["flip"].notna().sum())
        entry = {
            "model": model,
            "benchmark": benchmark,
            "condition": condition,
            "parquet": str(path),
            "rows": int(len(df)),
            "span_fail": int((~df["span_extract_ok"].astype("boolean")).sum()),
            "flip_true": int(df["flip"].astype("boolean").sum()) if flip_nonnull else 0,
            "flip_nonnull": flip_nonnull,
            "sources": sources,
        }
        entries.append(entry)
        logger.info(
            f"  {model} x {benchmark} x {condition}: rows={entry['rows']} "
            f"span_fail={entry['span_fail']} flip={entry['flip_true']}/{entry['flip_nonnull']}"
        )
    return entries


def verify_manifest(manifest_path: Path) -> bool:
    """manifest の移行元 sha256 と parquet 行数を再検証する."""
    import pandas as pd

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    n_files = 0
    for entry in manifest["entries"]:
        for src in entry["sources"]:
            actual = sha256_file(src["path"])
            n_files += 1
            if actual != src["sha256"]:
                logger.error(f"hash mismatch: {src['path']}")
                ok = False
        pq = Path(entry["parquet"])
        if not pq.exists():
            logger.error(f"parquet missing: {pq}")
            ok = False
            continue
        rows = len(pd.read_parquet(pq, columns=["sample_id"]))
        if rows != entry["rows"]:
            logger.error(f"row count mismatch: {pq} manifest={entry['rows']} actual={rows}")
            ok = False
    logger.info(
        f"verify: {len(manifest['entries'])} entries, {n_files} source files -> "
        f"{'OK' if ok else 'FAILED'}"
    )
    return ok


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.out / "master_manifest.json")

    if args.verify:
        sys.exit(0 if verify_manifest(manifest_path) else 1)

    paths = load_paths_config(args.paths)
    outputs_root = Path(paths["archive_outputs"])
    analysis_root = Path(paths["archive_analysis"])
    registry = load_registry(args.registry)
    validate_registry(registry)

    models = args.models or list(registry["models"].keys())
    benchmarks = args.benchmarks or list(registry["benchmarks"])
    seed = int(registry["seed"])

    logger.info(f"アーカイブ: {outputs_root}")
    logger.info(f"出力: {args.out}")
    logger.info(f"設定: {len(models)} models x {len(benchmarks)} benchmarks")

    all_entries: list[dict] = []
    for model in models:
        for benchmark in benchmarks:
            logger.info(f"[{model} x {benchmark}]")
            entries = build_setting(
                outputs_root,
                analysis_root,
                args.out,
                model,
                benchmark,
                seed=seed,
                prompt_id=prompt_id_for(registry, benchmark),
            )
            all_entries.extend(entries)

    # 既存 manifest とマージ (部分ビルドの積み上げを許す)
    manifest: dict = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "archive_outputs": str(outputs_root),
        "registry": str(args.registry),
        "entries": [],
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        new_keys = {(e["model"], e["benchmark"], e["condition"]) for e in all_entries}
        manifest["entries"] = [
            e
            for e in old.get("entries", [])
            if (e["model"], e["benchmark"], e["condition"]) not in new_keys
        ]
    manifest["entries"].extend(all_entries)
    manifest["entries"].sort(key=lambda e: (e["model"], e["benchmark"], e["condition"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_rows = sum(e["rows"] for e in manifest["entries"])
    logger.info(
        f"完了: {len(manifest['entries'])} parquet, 合計 {total_rows} 行 -> {manifest_path}"
    )
    # 参考: 期待ファイルの欠落チェック
    missing = [
        (m, b, c)
        for m in models
        for b in benchmarks
        for c in CONDITIONS
        if not master_parquet_path(args.out, m, b, c).exists()
    ]
    if missing:
        logger.warning(f"未生成の (model, benchmark, condition): {missing}")


if __name__ == "__main__":
    main()

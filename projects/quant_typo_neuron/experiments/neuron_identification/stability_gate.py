"""M0 再現ゲート②: seed/定義間の安定性ゲート (CLI).

README §5 の I/F:
    uv run python experiments/neuron_identification/stability_gate.py \
        --config configs/neuron_identification.yaml

役割:
    1. config から mask パスを収集（seed / responsibility 定義ごと）
    2. stability_report で平均ペアワイズ Jaccard・層分布 Spearman を算出
    3. stability_gate_decision でゲート合否を判定
    4. 結果を標準出力と results/<exp>/<run>/stability_gate.json に保存

設定ファイル（YAML）の読み取りキー（省略可能なものはデフォルト値あり）:
    exp_name: str                 # 実験名（結果ディレクトリ名）
    stability_gate:
        masks: list[str]          # mask JSON パスのリスト（明示指定）
        mask_glob: str            # または glob パターン（masks と排他）
        min_jaccard: float        # 合格閾値（デフォルト 0.5）
        min_rank_corr: float      # 合格閾値（デフォルト 0.7）
    results_dir: str              # 出力ルート（デフォルト "results"）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML config file.

    Tries PyYAML first; falls back to a minimal key:value parser for
    environments where PyYAML is not installed.
    """
    path = Path(path)
    try:
        import yaml  # type: ignore[import]
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Minimal fallback: only handles flat key: value pairs (not nested)
    cfg: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()
    return cfg


def _collect_mask_paths(cfg: dict, config_dir: Path) -> list[Path]:
    """Resolve mask file paths from config.

    Looks under cfg['stability_gate']['masks'] (list of paths) or
    cfg['stability_gate']['mask_glob'] (glob pattern relative to config_dir).
    Also accepts cfg['mask_paths'] as a flat-list alternative.

    Returns:
        Sorted list of resolved Path objects.
    """
    sg: dict = cfg.get("stability_gate", {}) or {}

    # Explicit list of paths
    explicit: list[str] | None = sg.get("masks") or cfg.get("mask_paths")
    if explicit:
        paths = [config_dir / p for p in explicit]
        return sorted(paths)

    # Glob pattern
    glob_pat: str | None = sg.get("mask_glob")
    if glob_pat:
        paths = sorted(config_dir.glob(glob_pat))
        return paths

    return []


def main(argv: list[str] | None = None) -> int:
    """Entry point for the stability gate CLI.

    Returns:
        0 if the gate passed, 1 if it failed, 2 on usage error.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file (configs/neuron_identification.yaml).",
    )
    parser.add_argument(
        "--mask",
        action="append",
        dest="masks",
        metavar="PATH",
        help=(
            "Path to a neuron_mask.json file. "
            "Repeat to supply multiple masks. "
            "If not given, mask paths are read from the config."
        ),
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=None,
        help="Override min_jaccard threshold (default: config or 0.5).",
    )
    parser.add_argument(
        "--min-rank-corr",
        type=float,
        default=None,
        help="Override min_rank_corr threshold (default: config or 0.7).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Override output JSON path. "
            "Default: results/<exp_name>/<timestamp>/stability_gate.json"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Root results directory (default: results/).",
    )

    args, _extra = parser.parse_known_args(argv)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = _load_yaml(config_path)
    config_dir = config_path.parent

    # ------------------------------------------------------------------
    # Resolve mask paths
    # ------------------------------------------------------------------
    if args.masks:
        mask_paths = [Path(p) for p in args.masks]
    else:
        mask_paths = _collect_mask_paths(cfg, config_dir)

    if len(mask_paths) < 2:
        print(
            "ERROR: stability_gate requires at least 2 mask files.\n"
            f"       Found: {mask_paths}\n"
            "       Supply via --mask <path> (repeatable) or "
            "config.stability_gate.masks / mask_glob.",
            file=sys.stderr,
        )
        return 2

    # Verify all mask paths exist
    missing = [p for p in mask_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: mask file not found: {p}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Load thresholds from config or CLI
    # ------------------------------------------------------------------
    sg_cfg: dict = cfg.get("stability_gate", {}) or {}
    min_jaccard: float = (
        args.min_jaccard
        if args.min_jaccard is not None
        else float(sg_cfg.get("min_jaccard", 0.5))
    )
    min_rank_corr: float = (
        args.min_rank_corr
        if args.min_rank_corr is not None
        else float(sg_cfg.get("min_rank_corr", 0.7))
    )

    # ------------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------------
    from quant_typo_neuron.neuron_identification.scoring import load_mask
    from quant_typo_neuron.neuron_identification.stability import (
        stability_report,
        stability_gate_decision,
    )

    print(f"Loading {len(mask_paths)} masks...")
    masks = []
    for p in mask_paths:
        m = load_mask(p)
        total_neurons = sum(len(dims) for dims in m.values())
        print(f"  {p}  ({total_neurons} neurons)")
        masks.append(m)

    # ------------------------------------------------------------------
    # Compute stability report
    # ------------------------------------------------------------------
    print("\nComputing pairwise stability metrics...")
    report = stability_report(masks)

    print(f"  mean pairwise Jaccard   : {report['mean_jaccard']:.4f}")
    print(f"  mean pairwise Spearman  : {report['mean_spearman']:.4f}")
    print(f"  num pairs evaluated     : {report['num_pairs']}")

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    decision = stability_gate_decision(
        report,
        min_jaccard=min_jaccard,
        min_rank_corr=min_rank_corr,
    )

    status_str = "PASSED" if decision["passed"] else "FAILED"
    print(f"\n[Gate ②] {status_str}")
    print(f"  Jaccard  {report['mean_jaccard']:.4f} >= {min_jaccard}  -> {'OK' if report['mean_jaccard'] >= min_jaccard else 'FAIL'}")
    print(f"  Spearman {report['mean_spearman']:.4f} >= {min_rank_corr}  -> {'OK' if report['mean_spearman'] >= min_rank_corr else 'FAIL'}")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        exp_name: str = str(cfg.get("exp_name", "neuron_identification"))
        results_root = Path(args.results_dir or cfg.get("results_dir", "results"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = results_root / exp_name / timestamp / "stability_gate.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gate": "stability_gate",
        "status": status_str,
        "decision": decision,
        "report": report,
        "mask_paths": [str(p) for p in mask_paths],
        "thresholds": {
            "min_jaccard": min_jaccard,
            "min_rank_corr": min_rank_corr,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResult saved: {out_path}")

    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

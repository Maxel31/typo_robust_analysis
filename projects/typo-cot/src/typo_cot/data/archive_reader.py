"""JSAI2026 アーカイブ (読み取り専用) への薄いアクセス層.

configs/paths.yaml が指すアーカイブのディレクトリ規約:
- baseline:  {outputs}/baseline/{model}_{benchmark}/results.json
- perturbed: {outputs}/perturbed/{model}_{benchmark}_{suffix}/results.json
  (suffix は master_table.CONDITION_TO_ARCHIVE_SUFFIX)
- analysis:  {outputs}/analysis/{benchmark}/{model}/{suffix}/full_results.json

本モジュールは読み取りとパス解決だけを行う。アーカイブへの書き込みは行わない。
master table 完成後は、このモジュール経由の直接読みを parquet 読みに
一行で差し替えられるよう、データアクセスをここに隔離する。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from typo_cot.data.master_table import CONDITION_TO_ARCHIVE_SUFFIX

# ---------------------------------------------------------------------------
# 統合テーブルのセル計画 (2026-07-18 wave2 取込で追加)
#
# v1 (JSAI2026 アーカイブ) の 25 設定 × 6 条件に加えて:
# - anti_lxt4 (k4_bottom_k): v1 25 設定、アーカイブ由来 (analysis は無い)
# - math (MATH-500 再生成): 6 モデル × 3 条件、exp-10-scope outputs 由来
# - Qwen2.5-7B: B5 × lxt4/random4 は exp-10-scope、clean はアーカイブ
# - DeepSeek-R1-Distill-Qwen-7B: 3 ベンチ × 3 条件、exp-10-scope (<think> 形式)
# ---------------------------------------------------------------------------

V1_MODELS: tuple[str, ...] = (
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
)
V1_BENCHMARKS: tuple[str, ...] = ("gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa")
V1_CELL_CONDITIONS: tuple[str, ...] = (
    "clean", "lxt1", "lxt2", "lxt4", "lxt8", "random4"
)
WAVE2_CONDITIONS: tuple[str, ...] = ("clean", "lxt4", "random4")
QWEN_MODEL = "Qwen2.5-7B-Instruct"
R1_MODEL = "DeepSeek-R1-Distill-Qwen-7B"
R1_BENCHMARKS: tuple[str, ...] = ("gsm8k", "math", "mmlu")


def build_cell_plan(
    paths_cfg: dict[str, Any], registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """統合テーブル全セルの取込元を列挙する (純粋なパス計画、io なし).

    Args:
        paths_cfg: configs/paths.yaml の dict
            (archive_outputs / archive_analysis / exp10_outputs を使用)
        registry: configs/registry.yaml の dict
            (prompts / reasoning_prompts の prompt_id を使用)

    Returns:
        セルの list。各セルは
        {model, benchmark, condition, baseline_path, perturbed_path,
         analysis_root, prompt_id}
        - baseline_path: clean 行の実体 + 各条件のプロベナンス用 results.json
        - perturbed_path: clean のとき None
        - analysis_root: flip/CoT 指標の full_results.json ルート (無ければ None)
    """
    arc = Path(paths_cfg["archive_outputs"])
    arc_analysis = Path(paths_cfg["archive_analysis"])
    exp10 = Path(paths_cfg["exp10_outputs"])

    def std_prompt(bench: str) -> str:
        return registry["prompts"][bench]["prompt_id"]

    def r1_prompt(bench: str) -> str:
        return registry["reasoning_prompts"][bench]["prompt_id"]

    def cell(
        model: str,
        bench: str,
        cond: str,
        baseline_root: Path,
        perturbed_root: Path | None,
        analysis_root: Path | None,
        prompt_id: str,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "benchmark": bench,
            "condition": cond,
            "baseline_path": baseline_dir(baseline_root, model, bench) / "results.json",
            "perturbed_path": (
                None
                if cond == "clean"
                else perturbed_dir(perturbed_root, model, bench, cond) / "results.json"
            ),
            "analysis_root": analysis_root,
            "prompt_id": prompt_id,
        }

    plan: list[dict[str, Any]] = []
    # v1 25 設定: 6 条件 (アーカイブ + analysis) + anti_lxt4 (アーカイブ, analysis なし)
    for model in V1_MODELS:
        for bench in V1_BENCHMARKS:
            for cond in V1_CELL_CONDITIONS:
                plan.append(
                    cell(model, bench, cond, arc, arc, arc_analysis, std_prompt(bench))
                )
            plan.append(
                cell(model, bench, "anti_lxt4", arc, arc, None, std_prompt(bench))
            )
        # MATH-500 再生成 (exp-10-scope)
        for cond in WAVE2_CONDITIONS:
            plan.append(cell(model, "math", cond, exp10, exp10, None, std_prompt("math")))
    # Qwen2.5-7B: B5 は clean=アーカイブ / 摂動=exp-10-scope、math は全て exp-10-scope
    for bench in V1_BENCHMARKS:
        for cond in WAVE2_CONDITIONS:
            plan.append(cell(QWEN_MODEL, bench, cond, arc, exp10, None, std_prompt(bench)))
    for cond in WAVE2_CONDITIONS:
        plan.append(cell(QWEN_MODEL, "math", cond, exp10, exp10, None, std_prompt("math")))
    # R1 蒸留: 3 ベンチ × 3 条件 (exp-10-scope, <think> 形式)
    for bench in R1_BENCHMARKS:
        for cond in WAVE2_CONDITIONS:
            plan.append(cell(R1_MODEL, bench, cond, exp10, exp10, None, r1_prompt(bench)))
    return plan


def load_paths_config(path: Path | str) -> dict[str, Any]:
    """configs/paths.yaml を読み込む."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path | str) -> Any:
    """JSON ファイルを読み込む."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """ファイルの sha256 hex digest を返す (移行同一性検証用)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def baseline_dir(outputs_root: Path | str, model: str, benchmark: str) -> Path:
    """baseline (clean) の結果ディレクトリ."""
    return Path(outputs_root) / "baseline" / f"{model}_{benchmark}"


def perturbed_dir(
    outputs_root: Path | str, model: str, benchmark: str, condition: str
) -> Path:
    """摂動条件の結果ディレクトリ. condition は master table の条件名."""
    suffix = CONDITION_TO_ARCHIVE_SUFFIX[condition]
    return Path(outputs_root) / "perturbed" / f"{model}_{benchmark}_{suffix}"


def analysis_condition_dir(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> Path:
    """analysis の (benchmark, model, condition) ディレクトリ."""
    suffix = CONDITION_TO_ARCHIVE_SUFFIX[condition]
    return Path(analysis_root) / benchmark / model / suffix


def load_analysis_sample_results(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> list[dict] | None:
    """full_results.json の sample_results を返す (無ければ None)."""
    path = analysis_condition_dir(analysis_root, model, benchmark, condition) / "full_results.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data.get("sample_results", [])


def load_analysis_partial_correlations(
    analysis_root: Path | str, model: str, benchmark: str, condition: str
) -> list[dict] | None:
    """full_results.json の partial_correlations を返す (無ければ None)."""
    path = analysis_condition_dir(analysis_root, model, benchmark, condition) / "full_results.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data.get("partial_correlations", [])


def load_summary_accuracy(result_dir: Path | str) -> float | None:
    """summary.json の overall accuracy を返す (無ければ None)."""
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        return None
    data = load_json(path)
    return (data.get("overall_metrics") or {}).get("accuracy")

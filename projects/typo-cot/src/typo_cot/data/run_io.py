"""アーカイブ run ディレクトリへの薄い読み取り層.

JSAI2026 アーカイブの {model}_{benchmark}[_k{N}_{mode}] ディレクトリ
(results.json / config.json / importance_scores/*.pt) を読む最小限の関数群。

Step 0 の master table (sample_id / model / benchmark / condition / ... / R_Q / R_C)
が完成したら、この層の関数を master table 参照に一行で差し替えられるよう、
データアクセスはすべてここを経由させる。
"""

import json
from pathlib import Path
from typing import Any


def load_results_list(run_dir: Path) -> list[dict[str, Any]]:
    """run ディレクトリの results.json をリストのまま読む."""
    with open(Path(run_dir) / "results.json", encoding="utf-8") as f:
        return json.load(f)


def load_results_by_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    """results.json を sample_id -> entry の辞書で読む."""
    return {r["sample_id"]: r for r in load_results_list(run_dir)}


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """run ディレクトリの config.json を読む."""
    with open(Path(run_dir) / "config.json", encoding="utf-8") as f:
        return json.load(f)


def question_scores_path(run_dir: Path, sample_id: str) -> Path:
    """Question→CoT relevance (R_Q) の .pt パス."""
    return Path(run_dir) / "importance_scores" / f"{sample_id}.pt"


def cot_scores_path(run_dir: Path, sample_id: str) -> Path:
    """CoT→Answer relevance (R_C) の .pt パス."""
    return Path(run_dir) / "importance_scores" / f"{sample_id}_cot.pt"


def link_reused_scores(
    src_run_dir: Path, dst_scores_dir: Path, sample_id: str
) -> dict[str, bool]:
    """摂動 run の .pt を fixed_target 側 scores ディレクトリへ symlink する.

    非flip サンプルの R_C^fixed は default と定義上同値なので、GPU 再計算せず
    default 側の `{sid}_cot.pt` / `{sid}.pt` を再利用する (実験4 の再利用トリック)。
    冪等 (既存リンクがあれば何もしない)。ソースが無いキーは False を返す。

    Returns:
        {"cot": bool, "question": bool} — 呼び出し後にリンク (または実体) が存在するか
    """
    dst_scores_dir = Path(dst_scores_dir)
    out = {}
    for key, src in [
        ("cot", cot_scores_path(src_run_dir, sample_id)),
        ("question", question_scores_path(src_run_dir, sample_id)),
    ]:
        dst = dst_scores_dir / src.name
        if not dst.exists():
            if src.exists():
                dst.symlink_to(src.resolve())
        out[key] = dst.exists()
    return out


def load_cot_scores(run_dir: Path, sample_id: str) -> dict[str, Any]:
    """R_C の .pt を CPU 上に読み込む (token_scores / cot_token_start 等)."""
    import torch

    return torch.load(
        cot_scores_path(run_dir, sample_id), map_location="cpu", weights_only=False
    )

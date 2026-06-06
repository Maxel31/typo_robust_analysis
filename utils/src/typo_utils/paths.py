"""リポジトリ内の共通パス解決。"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """workspace（モノレポ）ルートを探索して返す。

    `pyproject.toml` に `[tool.uv.workspace]` を持つ最上位ディレクトリをルートとみなす。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
            return parent
    # フォールバック: カレントワーキングディレクトリ
    return Path.cwd()


def datasets_dir() -> Path:
    """共有データセットディレクトリ。"""
    return repo_root() / "datasets"

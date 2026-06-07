"""データセット読み込み。共有 ``datasets/`` を参照する想定。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from typo_utils.paths import datasets_dir


def load_jsonl(path: str | Path) -> list[dict]:
    """JSONL を list[dict] で読み込む。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """JSONL を 1 行ずつ streaming で読む。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_dataset(name: str) -> Path:
    """共有 datasets ディレクトリ配下のデータセットパスを解決する。"""
    path = datasets_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    return path

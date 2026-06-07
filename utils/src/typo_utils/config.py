"""YAML 実験設定の読み込み（OmegaConf 薄ラッパ）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """YAML を読み込み、任意で CLI 風オーバーライド（``key=value``）を適用する。

    Example:
        cfg = load_config("configs/repro_baseline.yaml", ["typo.rate=0.2"])
    """
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """OmegaConf -> 素の dict（保存・ログ用）。"""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

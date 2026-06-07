"""YAML 実験設定の読み込み（OmegaConf 薄ラッパ）。

`_base_: <path>`（文字列 or リスト）を書くと、そのファイルを先に読み込んで
マージし、当該ファイルの値で上書きする（簡易継承）。パスは当該 YAML から
の相対。base 側がさらに `_base_` を持つ場合も再帰的に解決する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_with_base(path: Path) -> DictConfig:
    """1ファイルを読み、`_base_` があれば再帰的にマージして返す。"""
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        refs = [base_ref] if isinstance(base_ref, str) else list(base_ref)
        merged: DictConfig = OmegaConf.create({})
        for ref in refs:
            base_path = (path.parent / str(ref)).resolve()
            merged = OmegaConf.merge(merged, _load_with_base(base_path))
        # base を下地に、当該ファイルの値で上書き
        cfg = OmegaConf.merge(merged, cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """YAML を読み込み、`_base_` 継承を解決し、任意で CLI 風オーバーライド
    （``key=value``）を適用する。

    Example:
        cfg = load_config("configs/neuron_identification.yaml", ["dataset.n_samples=50"])
    """
    cfg = _load_with_base(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """OmegaConf -> 素の dict（保存・ログ用）。"""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

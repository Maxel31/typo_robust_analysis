"""実験記録: 常にローカル results/ へ保存し、加えて W&B へ log する。

W&B は optional extra（``tracking``）。未インストール / 未設定なら自動で no-op になり、
ローカル保存だけで実験が完結する（オフライン・可搬）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """1 回の実験 run を表すロガー。

    保存先: ``results/<exp_name>/<run_id>/``
      - config.yaml      実験設定
      - metrics.json     最終メトリクス
      - predictions.jsonl  予測（任意）
    """

    def __init__(
        self,
        exp_name: str,
        run_id: str,
        config: dict[str, Any] | None = None,
        results_root: str | Path = "results",
        use_wandb: bool = True,
    ) -> None:
        self.exp_name = exp_name
        self.run_id = run_id
        self.run_dir = Path(results_root) / exp_name / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, Any] = {}
        self._wandb_run = self._init_wandb(config) if use_wandb else None

        if config is not None:
            (self.run_dir / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _init_wandb(self, config: dict[str, Any] | None):
        # API キー未設定なら静かに無効化（offline 利用は WANDB_MODE=offline）
        if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
            return None
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "typo_robust_analysis"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=f"{self.exp_name}/{self.run_id}",
            group=self.exp_name,
            config=config,
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """メトリクスを記録（ローカル蓄積 + W&B）。"""
        self._metrics.update(metrics)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def log_predictions(self, predictions: list[dict[str, Any]]) -> None:
        """予測を predictions.jsonl に保存。"""
        path = self.run_dir / "predictions.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self) -> None:
        """metrics.json を書き出し、W&B run を閉じる。"""
        (self.run_dir / "metrics.json").write_text(
            json.dumps(self._metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()

"""再現実験のエントリポイント。

  uv run python experiments/reproduction/run.py --config configs/repro_baseline.yaml
"""

from __future__ import annotations

import argparse

from sample_project.runner import evaluate

from typo_utils.config import load_config, to_dict
from typo_utils.data.typo import TypoConfig
from typo_utils.seed import set_seed
from typo_utils.tracking import ExperimentLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="run", help="results/<exp>/<run_id>/ に保存")
    parser.add_argument("overrides", nargs="*", help="例: typo.rate=0.2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)

    typo = TypoConfig(rate=cfg.typo.rate, type=cfg.typo.type, seed=cfg.seed)

    with ExperimentLogger(
        exp_name=cfg.exp_name,
        run_id=args.run_id,
        config=to_dict(cfg),
        use_wandb=cfg.tracking.use_wandb,
    ) as logger:
        metrics = evaluate(typo)
        logger.log_metrics(metrics)
        print(f"[{cfg.exp_name}] {metrics}")


if __name__ == "__main__":
    main()

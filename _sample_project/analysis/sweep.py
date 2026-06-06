"""typo 率を振って頑健性カーブ用のデータを集め、results/ に保存するスイープ。

  uv run python analysis/sweep.py --config configs/repro_baseline.yaml
"""

from __future__ import annotations

import argparse

from sample_project.runner import evaluate

from typo_utils.config import load_config
from typo_utils.data.typo import TypoConfig
from typo_utils.tracking import ExperimentLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rates", default="0.0,0.1,0.2,0.3,0.4,0.5")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rates = [float(r) for r in args.rates.split(",")]

    with ExperimentLogger(
        exp_name=f"{cfg.exp_name}_sweep",
        run_id="sweep",
        config={"rates": rates, "typo_type": cfg.typo.type},
        use_wandb=cfg.tracking.use_wandb,
    ) as logger:
        rows = []
        for rate in rates:
            m = evaluate(TypoConfig(rate=rate, type=cfg.typo.type, seed=cfg.seed))
            acc = m.get("typo_acc", m["clean_acc"])
            rows.append({"typo_rate": rate, "accuracy": acc})
            logger.log_metrics({"typo_rate": rate, "accuracy": acc})
        logger.log_predictions(rows)
        print(rows)


if __name__ == "__main__":
    main()

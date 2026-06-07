"""提案手法の実験エントリポイント。

  uv run python experiments/proposed/run.py --config configs/proposed_method.yaml

再現実験との違いは config（method セクション）と、ここで提案手法を組み込む点。
テンプレートでは再現と同じ評価器を呼び、差し替えポイントだけ示している。
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
    parser.add_argument("--run-id", default="run")
    parser.add_argument("overrides", nargs="*")
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
        # TODO: ここで cfg.method に基づき提案手法を適用する
        metrics = evaluate(typo)
        logger.log_metrics(metrics)
        print(f"[{cfg.exp_name}] method={cfg.method.name} {metrics}")


if __name__ == "__main__":
    main()

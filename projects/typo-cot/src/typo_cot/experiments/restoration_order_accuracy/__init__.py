"""Public API for the Appendix E edited-word restoration-order experiment."""

from typo_cot.experiments.restoration_order_accuracy.aggregation import (
    BuildRestorationOrderTableConfig,
    RestorationOrderTableInputError,
    RestorationOrderTableResult,
    build_restoration_order_table,
)
from typo_cot.experiments.restoration_order_accuracy.protocol import (
    PAPER_BENCHMARKS,
    PAPER_BUDGETS,
    PAPER_MODELS,
    PAPER_ORDERS,
)
from typo_cot.experiments.restoration_order_accuracy.runner import (
    RestorationGeneration,
    RestorationOrderConfig,
    RestorationOrderResult,
    RestorationOrderRunError,
    run_restoration_order_accuracy,
)

__all__ = [
    "BuildRestorationOrderTableConfig",
    "PAPER_BENCHMARKS",
    "PAPER_BUDGETS",
    "PAPER_MODELS",
    "PAPER_ORDERS",
    "RestorationGeneration",
    "RestorationOrderConfig",
    "RestorationOrderResult",
    "RestorationOrderRunError",
    "RestorationOrderTableInputError",
    "RestorationOrderTableResult",
    "build_restoration_order_table",
    "run_restoration_order_accuracy",
]

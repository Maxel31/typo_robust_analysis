"""Component screening and causal validation consume disjoint diagnostic IDs."""

from __future__ import annotations

from pathlib import Path

from typo_robust_training.localization.component_config import (
    load_component_localization_config,
)
from typo_robust_training.localization.component_partition import partition_diagnostic_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gemma4b-component-localization.yaml"


def test_sha_partition_is_balanced_disjoint_and_input_order_invariant() -> None:
    protocol = load_component_localization_config(DEFAULT_CONFIG)
    records = tuple(
        {"record_id": f"{task}-{index:03d}", "task": task}
        for task in protocol.tasks
        for index in range(11)
    )
    first = partition_diagnostic_ids(records, protocol=protocol)
    second = partition_diagnostic_ids(tuple(reversed(records)), protocol=protocol)

    assert first == second
    assert set(first.screening).isdisjoint(first.causal_validation)
    assert set(first.screening) | set(first.causal_validation) == {
        str(record["record_id"]) for record in records
    }
    for task in protocol.tasks:
        assert (
            abs(len(first.screening_by_task[task]) - len(first.causal_validation_by_task[task]))
            <= 1
        )

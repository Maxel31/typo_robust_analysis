"""Frozen, outcome-blind fixtures for rebuttal v2 donor assignment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2.donors import assign_donors
from typo_cot.experiments.rebuttal_v2.identity import identity_hash
from typo_cot.experiments.rebuttal_v2.schemas import IntakeError


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/rebuttal_v2/protocol.json"


@pytest.fixture
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _row(
    pair_id: str,
    group: str,
    *,
    model_id: str = "model-a",
    task: str = "gsm8k",
    target_rule: str | None = "rule-a",
    aligned_word_count: int = 2,
    **extra: object,
) -> dict:
    return {
        "pair_id": pair_id,
        "original_problem_group_id": group,
        "model_id": model_id,
        "task": task,
        "target_rule": target_rule,
        "aligned_word_count": aligned_word_count,
        **extra,
    }


def _independent_order_key(row: dict, role: str) -> tuple[str, str]:
    payload = {
        "seed": 42,
        "role": role,
        "stratum": {
            "model_id": row["model_id"],
            "task": row["task"],
            "target_rule": row["target_rule"],
            "aligned_word_count": row["aligned_word_count"],
        },
        "pair_id": row["pair_id"],
    }
    raw = "rebuttal-donor-order/v2\n" + json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), row["pair_id"]


def test_fixed_hash_oracle_and_exact_one_to_one_assignment(protocol: dict) -> None:
    oracle_payload = {
        "seed": 42,
        "role": "recipient",
        "stratum": {
            "model_id": "model-a",
            "task": "gsm8k",
            "target_rule": "rule-a",
            "aligned_word_count": 2,
        },
        "pair_id": "recipient-a",
    }
    assert identity_hash("rebuttal-donor-order/v2", oracle_payload) == (
        "d2aeb7806779cba7e368478dd16f92e73252dbdbdbd42b13452c23609fad3b1b"
    )

    recipients = [_row(f"recipient-{letter}", f"rg-{letter}") for letter in "abc"]
    donors = [_row(f"donor-{letter}", f"dg-{letter}") for letter in "wxyz"]
    result = assign_donors(recipients, donors, protocol)

    expected_recipients = sorted(
        recipients, key=lambda row: _independent_order_key(row, "recipient")
    )
    expected_donors = sorted(donors, key=lambda row: _independent_order_key(row, "donor"))
    expected = dict(
        zip(
            (row["pair_id"] for row in expected_recipients),
            (row["pair_id"] for row in expected_donors[: len(expected_recipients)]),
            strict=True,
        )
    )
    actual = {row["pair_id"]: row["donor_pair_id"] for row in result["assignments"]}
    assert actual == expected
    assert result["schema_version"] == "rebuttal-donor-assignment/v2"
    assert result["donor_bank_available"] is True
    assert result["assignments"] == sorted(result["assignments"], key=lambda row: row["pair_id"])
    assert all(row["eligibility"] == "valid" for row in result["assignments"])
    assert all(row["reason_codes"] == [] for row in result["assignments"])
    assert len(set(actual.values())) == len(actual)
    assert result["unused_donor_pair_ids"] == sorted(
        set(row["pair_id"] for row in donors).difference(actual.values())
    )


def test_permutation_shard_and_outcome_metadata_cannot_change_assignment(protocol: dict) -> None:
    recipients = [
        _row("r1", "rg1", gold="A", correct=True, patch_outcome="success", shard=0),
        _row("r2", "rg2", gold="B", correct=False, patch_outcome="failure", shard=1),
        _row("r3", "rg3", gold="C", labels=["selected"], permutation=[2, 0, 1]),
    ]
    donors = [
        _row("d1", "dg1", gold="recipient-answer", donor_success=True, shard=0),
        _row("d2", "dg2", gold="other", donor_success=False, shard=5),
        _row("d3", "dg3", labels={"outcome": "best"}, permutation="late"),
    ]
    baseline = assign_donors(recipients, donors, protocol)

    changed_recipients = list(reversed(copy.deepcopy(recipients)))
    changed_donors = [copy.deepcopy(donors[index]) for index in (1, 2, 0)]
    for index, row in enumerate(changed_recipients):
        row.update(gold=f"changed-{index}", correct=None, patch_outcome="rewritten", shard=99)
    for index, row in enumerate(changed_donors):
        row.update(gold=f"leaked-{index}", donor_success="unknown", shard=-1)

    assert assign_donors(changed_recipients, changed_donors, protocol) == baseline


def test_short_bank_leaves_hash_late_recipients_invalid_without_reuse(protocol: dict) -> None:
    recipients = [_row(f"r{index}", f"rg{index}") for index in range(4)]
    donors = [_row("d0", "dg0"), _row("d1", "dg1")]
    result = assign_donors(recipients, donors, protocol)
    ordered = sorted(recipients, key=lambda row: _independent_order_key(row, "recipient"))
    expected_valid = {row["pair_id"] for row in ordered[:2]}

    by_id = {row["pair_id"]: row for row in result["assignments"]}
    assert {pair_id for pair_id, row in by_id.items() if row["eligibility"] == "valid"} == (
        expected_valid
    )
    assert all(by_id[pair_id]["reason_codes"] == [] for pair_id in expected_valid)
    for pair_id in set(by_id).difference(expected_valid):
        assert by_id[pair_id] == {
            "pair_id": pair_id,
            "donor_pair_id": None,
            "eligibility": "invalid",
            "reason_codes": ["donor_bank_insufficient"],
        }
    assigned = [row["donor_pair_id"] for row in by_id.values() if row["donor_pair_id"]]
    assert len(assigned) == len(set(assigned)) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "model-b"),
        ("task", "mmlu-pro"),
        ("target_rule", "rule-b"),
        ("aligned_word_count", 1),
    ],
)
def test_strata_are_exact_and_never_fall_back(protocol: dict, field: str, value: object) -> None:
    recipient = _row("r", "rg")
    donor = _row("d", "dg")
    donor[field] = value
    result = assign_donors([recipient], [donor], protocol)
    assert result["assignments"] == [
        {
            "pair_id": "r",
            "donor_pair_id": None,
            "eligibility": "invalid",
            "reason_codes": ["donor_bank_insufficient"],
        }
    ]
    assert result["unused_donor_pair_ids"] == ["d"]


def test_unknown_recipient_metadata_is_invalid_but_missing_bank_takes_precedence(
    protocol: dict,
) -> None:
    recipients = [
        _row("unknown-target", "rg1", target_rule=None),
        _row("zero-words", "rg2", aligned_word_count=0),
        _row("both", "rg3", target_rule=None, aligned_word_count=0),
    ]
    present = assign_donors(recipients, [], protocol)
    by_id = {row["pair_id"]: row for row in present["assignments"]}
    assert by_id["unknown-target"]["reason_codes"] == ["target_rule_unknown"]
    assert by_id["zero-words"]["reason_codes"] == ["aligned_word_count_zero"]
    assert by_id["both"]["reason_codes"] == [
        "aligned_word_count_zero",
        "target_rule_unknown",
    ]
    assert {row["eligibility"] for row in present["assignments"]} == {"invalid"}

    missing = assign_donors(recipients, None, protocol)
    assert missing["donor_bank_available"] is False
    assert missing["unused_donor_pair_ids"] == []
    assert {row["eligibility"] for row in missing["assignments"]} == {"not_available"}
    assert all("donor_bank_not_available" in row["reason_codes"] for row in missing["assignments"])


@pytest.mark.parametrize(
    "donor",
    [
        _row("d", "dg", target_rule=None),
        _row("d", "dg", aligned_word_count=0),
    ],
)
def test_unknown_donor_stratum_is_fatal(protocol: dict, donor: dict) -> None:
    with pytest.raises(IntakeError, match="unknown assignment stratum"):
        assign_donors([_row("r", "rg")], [donor], protocol)


def test_every_selected_recipient_group_is_excluded_from_bank(protocol: dict) -> None:
    recipient = _row("r", "current-group")
    with pytest.raises(IntakeError, match="overlap selected recipient groups"):
        assign_donors(
            [recipient],
            [_row("d", "other-selected-group")],
            protocol,
            recipient_groups={"current-group", "other-selected-group"},
        )

    # Passing an incomplete explicit set cannot waive the current recipient.
    with pytest.raises(IntakeError, match="overlap selected recipient groups"):
        assign_donors([recipient], [_row("d", "current-group")], protocol, recipient_groups=set())


def test_bank_variants_may_share_a_donor_group(protocol: dict) -> None:
    result = assign_donors(
        [_row("r1", "rg1"), _row("r2", "rg2")],
        [_row("d1", "shared-donor-group"), _row("d2", "shared-donor-group")],
        protocol,
    )
    assert all(row["eligibility"] == "valid" for row in result["assignments"])
    assert {row["donor_pair_id"] for row in result["assignments"]} == {"d1", "d2"}


@pytest.mark.parametrize(
    ("recipients", "donors", "error"),
    [
        ([_row("r", "g1"), _row("r", "g2")], [], "recipients contains duplicate"),
        ([_row("r", "g1")], [_row("d", "g2"), _row("d", "g3")], "donor_bank contains duplicate"),
        ([_row("same", "g1")], [_row("same", "g2")], "overlap recipient pair IDs"),
    ],
)
def test_duplicate_and_cross_role_pair_ids_are_fatal(
    protocol: dict, recipients: list[dict], donors: list[dict], error: str
) -> None:
    with pytest.raises(IntakeError, match=error):
        assign_donors(recipients, donors, protocol)


def test_required_group_and_row_types_and_frozen_protocol_are_enforced(protocol: dict) -> None:
    null_group = _row("r", "g")
    null_group["original_problem_group_id"] = None
    with pytest.raises(IntakeError, match="original_problem_group_id"):
        assign_donors([null_group], None, protocol)

    missing = _row("r", "g")
    del missing["task"]
    with pytest.raises(IntakeError, match="missing required fields: task"):
        assign_donors([missing], None, protocol)

    boolean_count = _row("r", "g")
    boolean_count["aligned_word_count"] = True
    with pytest.raises(IntakeError, match="not a boolean"):
        assign_donors([boolean_count], None, protocol)

    changed_protocol = copy.deepcopy(protocol)
    changed_protocol["seed"] = 43
    with pytest.raises(IntakeError, match="frozen revision"):
        assign_donors([_row("r", "g")], None, changed_protocol)


def test_assignment_is_pure_and_ignores_even_non_json_extra_fields(protocol: dict) -> None:
    marker = object()
    recipients = [_row("r", "rg", arbitrary=marker)]
    donors = [_row("d", "dg", arbitrary=marker)]
    recipients_before = [dict(recipients[0])]
    donors_before = [dict(donors[0])]

    result = assign_donors(recipients, donors, protocol)

    assert result["assignments"][0]["donor_pair_id"] == "d"
    assert recipients == recipients_before
    assert donors == donors_before
    assert recipients[0]["arbitrary"] is marker
    assert donors[0]["arbitrary"] is marker

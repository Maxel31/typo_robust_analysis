from __future__ import annotations

import pytest
import torch

from typo_robust_training.probe.subspace_kill_runtime import block_output_subspace_patch


class _IdentityBlock(torch.nn.Module):
    def forward(self, values):
        return values


def test_full_patch_changes_only_registered_token_and_runs_once() -> None:
    layer = _IdentityBlock()
    values = torch.zeros((1, 3, 4))
    donor = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with block_output_subspace_patch(
        layer, position=1, clean_donor=donor, row_basis=None
    ):
        result = layer(values)

    assert torch.equal(result[0, 1], donor)
    assert torch.equal(result[0, 0], values[0, 0])
    assert torch.equal(result[0, 2], values[0, 2])


def test_rank_patch_applies_qtq_without_dense_projector_or_basis_gradient() -> None:
    layer = _IdentityBlock()
    values = torch.zeros((1, 1, 4), requires_grad=True)
    donor = torch.ones(4)
    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    with block_output_subspace_patch(
        layer, position=0, clean_donor=donor, row_basis=basis
    ):
        result = layer(values)
    result.sum().backward()

    assert torch.equal(result.detach()[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert basis.grad is None
    assert donor.grad is None
    assert torch.equal(values.grad[0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_wrong_token_nonorthogonal_or_unused_patch_fails_closed() -> None:
    layer = _IdentityBlock()
    donor = torch.ones(4)
    with pytest.raises(RuntimeError, match="wrong count or coordinate"):
        with block_output_subspace_patch(
            layer, position=2, clean_donor=donor, row_basis=None
        ):
            layer(torch.zeros((1, 1, 4)))
    with pytest.raises(ValueError, match="not orthonormal"):
        with block_output_subspace_patch(
            layer,
            position=0,
            clean_donor=donor,
            row_basis=torch.ones((2, 4)),
        ):
            pass
    with pytest.raises(RuntimeError, match="did not run"):
        with block_output_subspace_patch(
            layer, position=0, clean_donor=donor, row_basis=None
        ):
            pass

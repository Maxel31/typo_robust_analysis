"""GPU usability verification.

MUST be run with ``CUDA_VISIBLE_DEVICES=2,3`` (only GPUs 2,3 are permitted for
this project). Per requirement, GPU availability is *verified*, not assumed:
these tests FAIL (not skip) when CUDA is unavailable.
"""
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA unavailable; run with CUDA_VISIBLE_DEVICES=2,3"


def test_at_least_one_visible_device():
    assert torch.cuda.device_count() >= 1


def test_matmul_runs_on_gpu():
    dev = "cuda:0"
    a = torch.randn(256, 256, device=dev)
    b = torch.randn(256, 256, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    assert torch.isfinite(c).all().item()

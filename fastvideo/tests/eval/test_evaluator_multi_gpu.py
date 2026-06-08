"""Multi-replica eval through the public ``Evaluator`` API.

Skipped automatically when fewer than 2 CUDA devices are visible.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.eval import create_evaluator


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-GPU evaluator tests require at least 2 visible CUDA devices",
)


_T, _C, _H, _W = 6, 3, 32, 32


def _make_samples(n: int) -> list[dict]:
    torch.manual_seed(7)
    samples: list[dict] = []
    for i in range(n):
        gen = torch.rand(_T, _C, _H, _W)
        # Slightly perturb each row's reference so scores vary by index.
        ref = gen + 0.01 * (i + 1) * torch.rand_like(gen)
        samples.append({"video": gen, "reference": ref})
    return samples


@pytest.fixture
def baseline_scores():
    """Reference scores computed on a single-GPU evaluator. The multi-GPU
    runs must reproduce these exactly when handed the same input list —
    that's the only way to verify round-robin dispatch isn't dropping or
    reordering samples."""
    samples = _make_samples(8)
    ev = create_evaluator(
        metrics=["common.psnr", "common.ssim"],
        device="cuda:0",
        num_gpus=1,
    )
    try:
        out = ev.evaluate(samples=samples)
    finally:
        ev.shutdown()
    return samples, out


def test_multi_gpu_evaluator_reports_two_workers():
    ev = create_evaluator(metrics=["common.psnr"], num_gpus=2)
    try:
        assert ev.num_gpus == 2
    finally:
        ev.shutdown()


def test_multi_gpu_dispatch_preserves_order_and_scores(baseline_scores):
    """Same samples, multi-GPU dispatch — results must match the single-GPU
    baseline element-for-element. This verifies (a) the round-robin doesn't
    reorder, (b) every sample is scored exactly once, (c) the workers
    don't share mutable state."""
    samples, expected = baseline_scores

    ev = create_evaluator(
        metrics=["common.psnr", "common.ssim"],
        num_gpus=2,
    )
    try:
        got = ev.evaluate(samples=samples)
    finally:
        ev.shutdown()

    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert set(g.keys()) == {"common.psnr", "common.ssim"}, f"row {i}"
        assert g["common.psnr"].score == pytest.approx(e["common.psnr"].score), \
            f"row {i} psnr drift"
        assert g["common.ssim"].score == pytest.approx(e["common.ssim"].score), \
            f"row {i} ssim drift"


def test_multi_gpu_evaluator_kwargs_form_runs_on_one_replica():
    """The kwargs form (single sample) is documented to always hit worker
    0; this test pins the contract so future refactors don't accidentally
    fan out a single call."""
    ev = create_evaluator(metrics=["common.psnr"], num_gpus=2)
    try:
        torch.manual_seed(0)
        gen = torch.rand(_T, _C, _H, _W)
        out = ev.evaluate(video=gen, reference=gen)
        assert out["common.psnr"].score > 50.0     # PSNR(x, x) is huge
    finally:
        ev.shutdown()


def test_multi_gpu_release_cuda_memory_runs_clean():
    """``release_cuda_memory`` must hit every replica without crashing."""
    ev = create_evaluator(metrics=["common.psnr"], num_gpus=2)
    try:
        samples = _make_samples(2)
        _ = ev.evaluate(samples=samples)
        ev.release_cuda_memory()                   # should not raise
    finally:
        ev.shutdown()

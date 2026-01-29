# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.distributed as dist

from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                  get_sp_group,
                                  maybe_init_distributed_environment_and_model_parallel)
from fastvideo.training.training_utils import shard_latents_across_sp


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


@pytest.fixture(scope="module")
def sp_dist():
    """
    These tests are intended to be run under torchrun with multiple GPUs, e.g.
      torchrun --nproc_per_node=2 -m pytest -q fastvideo/tests/distributed/test_sp_shard_latents_and_loss.py
      torchrun --nproc_per_node=4 -m pytest -q fastvideo/tests/distributed/test_sp_shard_latents_and_loss.py
      torchrun --nproc_per_node=8 -m pytest -q fastvideo/tests/distributed/test_sp_shard_latents_and_loss.py
    """
    ws = _world_size()
    if ws <= 1:
        pytest.skip("Requires torchrun with WORLD_SIZE>1")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    torch.manual_seed(1234)
    # Initialize FastVideo dist + model-parallel groups.
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=ws)
    assert get_sp_group().world_size == ws

    yield

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist_env_and_memory()


def _broadcast_tensor_(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=src)
    return t


def _all_gather_cat(t: torch.Tensor, dim: int) -> torch.Tensor:
    ws = _world_size()
    gathered = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=dim)


def test_shard_latents_t_not_divisible_but_thw_divisible_no_error(sp_dist):
    """
    Regression test for the commit "fix splitting on t":
    - t is NOT divisible by sp_size
    - (t*h*w) IS divisible by sp_size
    - sharding should NOT raise, and shards should round-trip to the original.
    """
    ws = _world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    b, c, t, h, w = 2, 3, 3, 2, 4  # t=3 (not divisible by 8), thw=24 (divisible by 8)
    assert (t * h * w) % ws == 0
    assert t % ws != 0

    latents = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    if _rank() == 0:
        latents.normal_()
    _broadcast_tensor_(latents)

    shard = shard_latents_across_sp(latents)
    assert shard.shape == (b, c, (t * h * w) // ws)

    gathered = _all_gather_cat(shard, dim=2)
    torch.testing.assert_close(gathered, latents.reshape(b, c, t * h * w))

    dist.barrier()


@pytest.mark.parametrize(
    "shape",
    [
        # No padding needed: thw divisible by 8, but t is not.
        (2, 1, 3, 2, 4),  # thw=24
        # Padding needed: thw not divisible by 8.
        (2, 1, 3, 2, 5),  # thw=30 -> pad to 32
    ],
)
def test_sharded_loss_gradient_matches_full_mse_after_rank_avg(sp_dist, shape):
    """
    Validates the math behind the new loss computation:

    If each rank computes:
      loss_rank = sp_world_size * local_sse / total_numel
    and the training stack averages gradients across ranks, then the resulting
    gradient should match the full MSE gradient.
    """
    ws = _world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    b, c, t, h, w = shape

    # Make inputs identical across ranks for deterministic comparison.
    init = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    x = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    y = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    if _rank() == 0:
        init.normal_()
        x.normal_()
        y.normal_()
    _broadcast_tensor_(init)
    _broadcast_tensor_(x)
    _broadcast_tensor_(y)

    # Full loss baseline (redundant compute; correct reference).
    w_full = torch.nn.Parameter(init.clone())
    pred_full = w_full * x
    full_loss = ((pred_full - y)**2).mean()
    full_loss.backward()
    grad_full = w_full.grad.detach().clone()

    # Sharded loss (compute on a shard only).
    w_shard = torch.nn.Parameter(init.clone())
    pred = w_shard * x
    sharded_pred = shard_latents_across_sp(pred)
    sharded_target = shard_latents_across_sp(y)
    local_sse = ((sharded_pred - sharded_target)**2).sum()
    shard_loss = ws * local_sse / pred.numel()
    shard_loss.backward()
    grad_shard = w_shard.grad.detach().clone()

    # Simulate "gradient averaging across ranks" (DDP/FSDP replicate-dim behavior).
    dist.all_reduce(grad_shard, op=dist.ReduceOp.SUM)
    grad_shard /= ws

    # (Optional) also average grad_full, for symmetry.
    dist.all_reduce(grad_full, op=dist.ReduceOp.SUM)
    grad_full /= ws

    torch.testing.assert_close(grad_shard, grad_full, rtol=1e-5, atol=1e-6)

    dist.barrier()


def test_padding_does_not_change_global_sse(sp_dist):
    """
    Padding-specific correctness test.

    When (t*h*w) is NOT divisible by sp_size, shard_latents_across_sp pads with
    zeros on the flattened axis. This test validates that:
    1) Summed SSE across SP ranks equals the full (unpadded) SSE.
    2) Any padded tokens that land on a rank are exactly zero for both pred/target.
    """
    from fastvideo.distributed.utils import compute_padding_for_sp

    ws = _world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    b, c, t, h = 2, 2, 1, 1
    # Force padding for any ws>1: seq_len = 2*ws + 1  -> remainder 1
    w = 2 * ws + 1
    seq_len = t * h * w
    assert seq_len % ws != 0

    pred = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    target = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    if _rank() == 0:
        pred.normal_()
        target.normal_()
    _broadcast_tensor_(pred)
    _broadcast_tensor_(target)

    # Full (unpadded) SSE reference.
    full_sse = ((pred - target)**2).sum()

    # Sharded local SSE (includes padded region, which should contribute 0).
    sharded_pred = shard_latents_across_sp(pred)
    sharded_target = shard_latents_across_sp(target)
    local_sse = ((sharded_pred - sharded_target)**2).sum()

    global_sse = local_sse.clone()
    dist.all_reduce(global_sse, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(global_sse, full_sse, rtol=1e-5, atol=1e-6)

    # Explicitly validate padded tokens are zeros on ranks that cover them.
    padded_seq_len, padding_amount = compute_padding_for_sp(seq_len, ws)
    assert padding_amount > 0
    elements_per_rank = padded_seq_len // ws
    start = _rank() * elements_per_rank
    end = (_rank() + 1) * elements_per_rank

    if end > seq_len:
        # There is padding on this rank: positions [max(seq_len, start), end)
        pad_start_global = max(seq_len, start)
        pad_end_global = end
        pad_start_local = pad_start_global - start
        pad_end_local = pad_end_global - start
        assert pad_start_local < pad_end_local

        pad_slice_pred = sharded_pred[:, :, pad_start_local:pad_end_local]
        pad_slice_target = sharded_target[:, :, pad_start_local:pad_end_local]
        torch.testing.assert_close(pad_slice_pred, torch.zeros_like(pad_slice_pred))
        torch.testing.assert_close(pad_slice_target, torch.zeros_like(pad_slice_target))

    dist.barrier()



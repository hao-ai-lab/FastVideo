# SPDX-License-Identifier: Apache-2.0
"""
Mismatch compatibility test (direction B):
  - WORLD_SIZE=4
  - SP: sp_size=2
  - HSDP/FSDP2 mesh: replicate_dim=1, shard_dim=4 (mesh_shape=(1,4))

We verify gradients from:
  (A) full MSE: mean((pred-target)^2)
  (B) SP-sharded loss: sp_size * local_sse / total_numel
match under this mismatch.

Run:
  source /home/hao_lab/miniconda3/etc/profile.d/conda.sh
  conda activate alexfv
  torchrun --standalone --nproc_per_node=4 -m pytest -q fastvideo/tests/distributed/test_hsdp_sp_mismatch_sp2_hsdp_shard4.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from fastvideo.distributed import cleanup_dist_env_and_memory
from fastvideo.distributed.parallel_state import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.training.training_utils import shard_latents_across_sp


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


@pytest.fixture(scope="module")
def dist_setup():
    if _world_size() != 4:
        pytest.skip("Designed for torchrun WORLD_SIZE=4")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=2)
    assert get_sp_group().world_size == 2
    yield
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist_env_and_memory()


class ScaleModule(torch.nn.Module):
    def __init__(self, c: int, init: torch.Tensor):
        super().__init__()
        assert init.shape == (c,)
        self.scale = torch.nn.Parameter(init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1, 1, 1, 1)


def _broadcast_(t: torch.Tensor, src: int = 0) -> None:
    dist.broadcast(t, src=src)


def _gather_full_vec_allranks(local_shard: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(local_shard) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_shard)
    return torch.cat(gathered, dim=0)


def test_sp_sharded_loss_matches_full_mse_under_mismatch(dist_setup):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.manual_seed(2027)

    b, c, t, h, w = 2, 8, 1, 1, 8  # thw=8 divisible by sp=2

    x = torch.empty((b, c, t, h, w), device=device, dtype=torch.float32)
    if _rank() == 0:
        x.normal_()
    _broadcast_(x)
    target = (0.25 * x).contiguous()

    init = torch.empty((c,), device=device, dtype=torch.float32)
    if _rank() == 0:
        init.normal_()
    _broadcast_(init)

    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(1, 4),
        mesh_dim_names=("replicate", "shard"),
    )

    mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )

    # (A) Full loss.
    m_full = ScaleModule(c=c, init=init).to(device)
    fully_shard(m_full, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
    pred_full = m_full(x)
    loss_full = ((pred_full - target) ** 2).mean()
    loss_full.backward()
    g_full = m_full.scale.grad
    g_full_local = g_full.to_local() if hasattr(g_full, "to_local") else g_full
    g_full_vec = _gather_full_vec_allranks(g_full_local.flatten())

    # (B) SP-sharded loss.
    m_sp = ScaleModule(c=c, init=init).to(device)
    fully_shard(m_sp, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
    pred = m_sp(x)
    sp = get_sp_group().world_size  # 2
    sharded_pred = shard_latents_across_sp(pred)
    sharded_target = shard_latents_across_sp(target)
    local_sse = ((sharded_pred - sharded_target) ** 2).sum()
    loss_sp = (sp * local_sse) / pred.numel()
    loss_sp.backward()
    g_sp = m_sp.scale.grad
    g_sp_local = g_sp.to_local() if hasattr(g_sp, "to_local") else g_sp
    g_sp_vec = _gather_full_vec_allranks(g_sp_local.flatten())

    torch.testing.assert_close(g_sp_vec, g_full_vec, rtol=1e-4, atol=1e-5)



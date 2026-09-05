import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from fastvideo.train.utils.lora import (
    _make_replicated_lora_parameter,
)


def _two_rank_replicated_lora_worker(
    rank,
    init_method,
    device_type,
):
    backend = "nccl" if device_type == "cuda" else "gloo"
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        mesh = init_device_mesh(
            device_type,
            (1, 2),
            mesh_dim_names=("replicate", "shard"),
        )
        local_initial_value = (
            torch.tensor([1.0, 2.0], device=device)
            if rank == 0 else torch.tensor([10.0, 20.0], device=device)
        )
        parameter = _make_replicated_lora_parameter(
            nn.Parameter(local_initial_value),
            mesh,
        )

        # A Replicate placement requires identical values. Creation broadcasts
        # the first mesh rank's value instead of merely assigning the label.
        assert torch.equal(
            parameter.to_local(),
            torch.tensor([1.0, 2.0], device=device),
        )

        optimizer = torch.optim.SGD([parameter], lr=0.1)
        coefficient = 1.0 if rank == 0 else 3.0
        loss = (parameter.to_local() * coefficient).sum()
        loss.backward()

        # Rank-local gradients [1, 1] and [3, 3] must be averaged before
        # gradient clipping and the optimizer step.
        assert isinstance(parameter.grad, DTensor)
        assert torch.equal(
            parameter.grad.to_local(),
            torch.tensor([2.0, 2.0], device=device),
        )

        # Repeating backward without zero_grad exercises the real gradient
        # accumulation path. The previously synchronized gradient must remain
        # intact while the new rank-local contribution is averaged.
        second_coefficient = 2.0 if rank == 0 else 4.0
        second_loss = (parameter.to_local() * second_coefficient).sum()
        second_loss.backward()
        assert torch.equal(
            parameter.grad.to_local(),
            torch.tensor([5.0, 5.0], device=device),
        )

        optimizer.step()
        assert torch.equal(
            parameter.to_local(),
            torch.tensor([0.5, 1.5], device=device),
        )

        gathered = [
            torch.empty_like(parameter.to_local())
            for _ in range(2)
        ]
        dist.all_gather(gathered, parameter.to_local())
        assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


def test_replicated_lora_parameter_stays_consistent_across_ranks(
    tmp_path,
):
    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    rendezvous = (tmp_path / "lora-gradient-rendezvous").resolve().as_uri()
    mp.spawn(
        _two_rank_replicated_lora_worker,
        args=(rendezvous, "cpu"),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_replicated_lora_parameter_stays_consistent_on_cuda(
    tmp_path,
):
    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    rendezvous = (
        tmp_path / "lora-gradient-cuda-rendezvous"
    ).resolve().as_uri()
    mp.spawn(
        _two_rank_replicated_lora_worker,
        args=(rendezvous, "cuda"),
        nprocs=2,
        join=True,
    )

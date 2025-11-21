# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py

import torch
import torch.distributed

from fastvideo.distributed.parallel_state import (get_sp_group,
                                                  get_sp_parallel_rank,
                                                  get_sp_world_size,
                                                  get_tp_group)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


# TODO: remove model, make it sequence_parallel
def sequence_model_parallel_all_to_all_4D(input_: torch.Tensor,
                                          scatter_dim: int = 2,
                                          gather_dim: int = 1) -> torch.Tensor:
    """All-to-all communication of 4D tensors (e.g. QKV matrices) across sequence parallel group."""
    return get_sp_group().all_to_all_4D(input_, scatter_dim, gather_dim)


def sequence_model_parallel_all_gather(input_: torch.Tensor,
                                       dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_sp_group().all_gather(input_, dim)


def sequence_model_parallel_shard(input_: torch.Tensor,
                                       dim: int = 1) -> torch.Tensor:
    """Shard the input tensor across model parallel group."""
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()
    assert input_.shape[dim] % sp_world_size == 0, f"input tensor dim={dim} must be divisible by sp_world_size={sp_world_size}"
    elements_per_rank = input_.shape[dim] // sp_world_size
    # sharding dim
    input_ = input_.movedim(dim, 0)
    input_ = input_[sp_rank*elements_per_rank:(sp_rank+1)*elements_per_rank]
    input_ = input_.movedim(0, dim)
    return input_
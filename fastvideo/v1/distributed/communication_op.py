# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py

import torch
import torch.distributed

from fastvideo.v1.distributed.parallel_state import get_sp_group, get_tp_group


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

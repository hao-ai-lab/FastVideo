# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py

import torch
import torch.distributed

from fastvideo.distributed.parallel_state import (get_sp_group,
                                                  get_sp_parallel_rank,
                                                  get_sp_world_size,
                                                  get_tp_group)
from fastvideo.distributed.utils import (unpad_sequence_tensor,
                                         compute_padding_for_sp,
                                         pad_sequence_tensor)


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


def sequence_model_parallel_all_gather_with_unpad(
        input_: torch.Tensor,
        original_seq_len: int,
        dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor and remove padding.
    
    Args:
        input_: Sharded (and possibly padded) tensor to gather
        original_seq_len: Original sequence length before padding
        dim: Dimension to gather along (default: -1)
        
    Returns:
        Tensor: Gathered and unpadded tensor
    """

    # First gather across all ranks
    gathered = get_sp_group().all_gather(input_, dim)

    current_seq_len = gathered.shape[dim]
    if current_seq_len > original_seq_len:
        gathered = unpad_sequence_tensor(gathered,
                                         original_seq_len,
                                         seq_dim=dim)

    return gathered


def sequence_model_parallel_shard(input_: torch.Tensor,
                                  dim: int = 1) -> tuple[torch.Tensor, int]:
    """Shard the input tensor across model parallel group with optional padding.
    
    Args:
        input_: Input tensor to shard
        dim: Dimension to shard along (default: 1)
        
    Returns:
        tuple: (sharded_tensor, original_seq_len)
            - sharded_tensor: The sharded (and possibly padded) tensor
            - original_seq_len: Original sequence length before padding
    """

    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    original_seq_len = input_.shape[dim]

    # Compute padding if needed
    padded_seq_len, padding_amount = compute_padding_for_sp(
        original_seq_len, sp_world_size)

    # Pad if necessary
    if padding_amount > 0:
        input_ = pad_sequence_tensor(input_, padded_seq_len, seq_dim=dim)

    elements_per_rank = padded_seq_len // sp_world_size

    # Sharding along dim
    input_ = input_.movedim(dim, 0)
    input_ = input_[sp_rank * elements_per_rank:(sp_rank + 1) *
                    elements_per_rank]
    input_ = input_.movedim(0, dim)

    return input_, original_seq_len

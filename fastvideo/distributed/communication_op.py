# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/communication_op.py

import torch
import torch.distributed

from fastvideo.distributed.parallel_state import (get_sp_group,
                                                  get_sp_parallel_rank,
                                                  get_sp_world_size,
                                                  get_tp_group,
                                                  model_parallel_is_initialized)
from fastvideo.distributed.utils import (unpad_sequence_tensor,
                                         compute_padding_for_sp,
                                         pad_sequence_tensor)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Track if SP communication has been warmed up
_sp_warmup_done = False


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

    # NCCL all_gather expects contiguous inputs.
    if not input_.is_contiguous():
        input_ = input_.contiguous()
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


def warmup_sequence_parallel_communication(
        device: torch.device | None = None) -> None:
    """Warmup NCCL communicators for sequence parallel all-to-all operations.
    
    The first NCCL collective operation is slow due to lazy communicator
    initialization. This function runs dummy all-to-all operations to
    trigger the initialization upfront, before the first real forward pass.
    
    Args:
        device: Device to use for warmup tensors. If None, uses CUDA device 0.
    """
    global _sp_warmup_done

    if _sp_warmup_done:
        return

    if not model_parallel_is_initialized():
        return

    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        _sp_warmup_done = True
        return

    if device is None:
        device = torch.device("cuda")

    logger.info("Warming up sequence parallel communication (SP=%d)...",
                sp_world_size)

    # Use small but representative tensor shapes for warmup
    # Shape: [batch, seq_len, num_heads, head_dim]
    # The all-to-all patterns used in attention:
    #   1. scatter_dim=2 (heads), gather_dim=1 (seq) - before attention
    #   2. scatter_dim=1 (seq), gather_dim=2 (heads) - after attention
    batch_size = 1
    seq_len_per_rank = 16  # Small sequence per rank
    num_heads = sp_world_size * 4  # Must be divisible by sp_world_size
    head_dim = 64

    # Create dummy tensor for warmup
    dummy = torch.zeros(batch_size,
                        seq_len_per_rank,
                        num_heads,
                        head_dim,
                        device=device,
                        dtype=torch.bfloat16)

    # Warmup pattern 1: scatter heads, gather sequence (before attention)
    _ = sequence_model_parallel_all_to_all_4D(dummy,
                                              scatter_dim=2,
                                              gather_dim=1)

    # Warmup pattern 2: scatter sequence, gather heads (after attention)
    dummy2 = torch.zeros(batch_size,
                         seq_len_per_rank * sp_world_size,
                         num_heads // sp_world_size,
                         head_dim,
                         device=device,
                         dtype=torch.bfloat16)
    _ = sequence_model_parallel_all_to_all_4D(dummy2,
                                              scatter_dim=1,
                                              gather_dim=2)

    # Warmup all-gather (used for replicated tokens)
    dummy3 = torch.zeros(batch_size,
                         8,
                         num_heads // sp_world_size,
                         head_dim,
                         device=device,
                         dtype=torch.bfloat16)
    _ = sequence_model_parallel_all_gather(dummy3, dim=2)

    # Synchronize to ensure warmup completes
    torch.cuda.synchronize(device)

    # Clean up
    del dummy, dummy2, dummy3

    _sp_warmup_done = True
    logger.info("Sequence parallel communication warmup complete.")

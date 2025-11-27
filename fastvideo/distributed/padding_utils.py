# SPDX-License-Identifier: Apache-2.0

"""Utilities for handling sequence padding in sequence parallel."""

import torch


def compute_padding_for_sp(seq_len: int, sp_world_size: int) -> tuple[int, int]:
    """
    Compute padding needed for sequence parallel.
    
    Args:
        seq_len: Original sequence length
        sp_world_size: Sequence parallel world size
        
    Returns:
        tuple: (padded_seq_len, padding_amount)
    """
    if seq_len % sp_world_size == 0:
        return seq_len, 0
    
    padding_amount = sp_world_size - (seq_len % sp_world_size)
    padded_seq_len = seq_len + padding_amount
    
    return padded_seq_len, padding_amount


def create_attention_mask_for_padding(
    seq_len: int,
    padded_seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor | None:
    """
    Create attention mask to ignore padded tokens.
    
    Args:
        seq_len: Original sequence length (before padding)
        padded_seq_len: Padded sequence length
        batch_size: Batch size
        device: Device to create mask on
        dtype: Data type for the mask (default: bool)
        
    Returns:
        Tensor: Boolean mask [B, padded_seq_len] where True = valid token,
                or None if no padding is needed
    """
    if seq_len == padded_seq_len:
        return None
    
    # Create mask: True for valid tokens, False for padding
    attention_mask = torch.ones(
        (batch_size, padded_seq_len),
        dtype=dtype,
        device=device,
    )
    
    # Mask out padding tokens
    attention_mask[:, seq_len:] = 0
    
    return attention_mask


def pad_sequence_tensor(
    tensor: torch.Tensor,
    target_seq_len: int,
    seq_dim: int = 1,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a tensor along the sequence dimension.
    
    Args:
        tensor: Input tensor to pad
        target_seq_len: Target sequence length after padding
        seq_dim: Dimension to pad along (default: 1)
        pad_value: Value to use for padding (default: 0.0)
        
    Returns:
        Tensor: Padded tensor
    """
    current_seq_len = tensor.shape[seq_dim]
    
    if current_seq_len >= target_seq_len:
        return tensor
    
    padding_amount = target_seq_len - current_seq_len
    
    # Create padding shape
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = padding_amount
    
    # Create padding tensor
    padding = torch.full(
        pad_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    
    # Concatenate along sequence dimension
    padded_tensor = torch.cat([tensor, padding], dim=seq_dim)
    
    return padded_tensor


def unpad_sequence_tensor(
    tensor: torch.Tensor,
    original_seq_len: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """
    Remove padding from a tensor along the sequence dimension.
    
    Args:
        tensor: Padded tensor
        original_seq_len: Original sequence length (before padding)
        seq_dim: Dimension to unpad along (default: 1)
        
    Returns:
        Tensor: Unpadded tensor
    """
    # Use slice to remove padding
    indices = [slice(None)] * tensor.dim()
    indices[seq_dim] = slice(0, original_seq_len)
    
    return tensor[tuple(indices)]

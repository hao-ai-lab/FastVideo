# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import torch.nn as nn

from fastvideo.v1.attention.selector import (backend_name_to_enum,
                                             get_attn_backend)
from fastvideo.v1.distributed.communication_op import (
    sequence_model_parallel_all_gather, sequence_model_parallel_all_to_all_4D)
from fastvideo.v1.distributed.parallel_state import (
    get_sequence_model_parallel_rank, get_sequence_model_parallel_world_size)
from fastvideo.v1.forward_context import ForwardContext, get_forward_context
from fastvideo.v1.platforms import _Backend


class DistributedAttention(nn.Module):
    """Distributed attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: Optional[int] = None,
                 softmax_scale: Optional[float] = None,
                 causal: bool = False,
                 supported_attention_backends: Optional[Tuple[_Backend,
                                                              ...]] = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads=num_heads,
                             head_size=head_size,
                             causal=causal,
                             softmax_scale=self.softmax_scale,
                             num_kv_heads=num_kv_heads,
                             prefix=f"{prefix}.impl",
                             **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: Optional[torch.Tensor] = None,
        replicated_k: Optional[torch.Tensor] = None,
        replicated_v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim(
        ) == 4, "Expected 4D tensors"
        # assert bs = 1
        assert q.shape[
            0] == 1, "Batch size must be 1, and there should be no padding tokens"
        batch_size, seq_len, num_heads, head_dim = q.shape
        local_rank = get_sequence_model_parallel_rank()
        world_size = get_sequence_model_parallel_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv,
                                                    scatter_dim=2,
                                                    gather_dim=1)
        # Apply backend-specific preprocess_qkv
        qkv = self.impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat(
                [replicated_q, replicated_k, replicated_v],
                dim=0)  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[:, :, local_rank *
                                            heads_per_rank:(local_rank + 1) *
                                            heads_per_rank]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        output = self.impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            replicated_output = output[:, seq_len * world_size:]
            output = output[:, :seq_len * world_size]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(
                replicated_output.contiguous(), dim=2)
        # Apply backend-specific postprocess_output
        output = self.impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(output,
                                                       scatter_dim=1,
                                                       gather_dim=2)
        return output, replicated_output


class LocalAttention(nn.Module):
    """Attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: Optional[int] = None,
                 softmax_scale: Optional[float] = None,
                 causal: bool = False,
                 supported_attention_backends: Optional[Tuple[_Backend,
                                                              ...]] = None,
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads=num_heads,
                             head_size=head_size,
                             softmax_scale=self.softmax_scale,
                             num_kv_heads=num_kv_heads,
                             causal=causal,
                             **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim] 
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim(
        ) == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        output = self.impl.forward(q, k, v, ctx_attn_metadata)
        return output

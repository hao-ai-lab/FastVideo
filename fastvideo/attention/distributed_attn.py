import torch
import torch.nn as nn
from typing import Optional
from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D, sequence_model_parallel_all_gather
from fastvideo.distributed.parallel_state import get_sequence_model_parallel_rank, get_sequence_model_parallel_world_size
from flash_attn import flash_attn_func


class DistributedAttention(nn.Module):
    """Distributed attention module that supports sequence parallelism.
    
    This class implements a minimal attention operation with support for distributed 
    processing across multiple GPUs using sequence parallelism. The implementation assumes
    batch_size=1 and no padding tokens for simplicity.
    
    The sequence parallelism strategy follows the Ulysses paper (https://arxiv.org/abs/2309.14509),
    which proposes redistributing attention heads across sequence dimension to enable efficient
    parallel processing of long sequences.
    
    Args:
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        softmax_scale (float, optional): Custom scaling factor for attention scores.
            If None, uses 1/sqrt(head_dim). Defaults to None.
    """
    def __init__(
        self,
        dropout_rate: float = 0.0,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.softmax_scale = softmax_scale
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
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"
        # assert bs = 1
        assert q.shape[0] == 1, "Batch size must be 1, and there should be no padding tokens"
        batch_size, seq_len, num_heads, head_dim = q.shape
        local_rank = get_sequence_model_parallel_rank()
        world_size = get_sequence_model_parallel_world_size()
        
        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0) # [3, seq_len, num_heads, head_dim]
        
        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
        
        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat([replicated_q, replicated_k, replicated_v], dim=0) # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[:, :, local_rank * heads_per_rank:(local_rank + 1) * heads_per_rank]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)
            
        q, k, v = qkv.chunk(3, dim=0)
        # Apply flash attention
        output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_rate,
            softmax_scale=self.softmax_scale,
            causal=self.causal
        )
        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            replicated_output = output[:, seq_len*world_size:]
            output = output[:, :seq_len*world_size]
            replicated_output = sequence_model_parallel_all_gather(replicated_output, dim=2)
        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
        return output, replicated_output


class LocalAttention(nn.Module):
    def __init__(self, dropout_rate: float = 0.0, causal: bool = False, softmax_scale: Optional[float] = None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.softmax_scale = softmax_scale
        
    def forward(self, q, k, v):
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
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"
        
        # Apply flash attention
        output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_rate,
            softmax_scale=self.softmax_scale,
            causal=self.causal
        )
        
        return output
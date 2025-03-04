import torch
import torch.nn as nn
from typing import Optional
from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
from flash_attn import flash_attn_qkvpacked_func


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
        
        # Stack QKV
        qkv = torch.stack([q, k, v], dim=2)  # [bs, seq_len, 3, num_heads, head_dim]
        
        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=3, gather_dim=1)
        
        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.stack([replicated_q, replicated_k, replicated_v], dim=2)
            qkv = torch.cat([qkv, replicated_qkv], dim=1)
            
        # Apply flash attention
        output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout_rate,
            softmax_scale=self.softmax_scale,
            causal=self.causal
        )
        
        # Redistribute back if using sequence parallelism
        o = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=3)
        replicated_o = None
        if replicated_q is not None:
            o, replicated_o = output.split([seq_len, seq_len], dim=1)
        return o, replicated_o

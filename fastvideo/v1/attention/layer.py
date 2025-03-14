import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Type

from fastvideo.v1.attention.flash_attn import LocalAttention, DistributedAttention


class AttentionBackend:
    """Base class for attention backends that provides implementation classes."""
    
    @staticmethod
    def get_impl_cls() -> Type:
        """Returns the implementation class for the attention backend."""
        raise NotImplementedError("Subclasses must implement get_impl_cls")


class LocalAttentionBackend(AttentionBackend):
    """Backend for local attention implementation."""
    
    @staticmethod
    def get_impl_cls() -> Type:
        """Returns the LocalAttention implementation class."""
        return LocalAttention


class DistributedAttentionBackend(AttentionBackend):
    """Backend for distributed attention implementation."""
    
    @staticmethod
    def get_impl_cls() -> Type:
        """Returns the DistributedAttention implementation class."""
        return DistributedAttention


class AttentionLayer(nn.Module):
    """Base attention layer that uses an implementation class from a backend."""
    
    def __init__(
        self,
        attn_backend: AttentionBackend,
        dropout_rate: float = 0.0,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        **extra_impl_args
    ):
        super().__init__()
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(
            dropout_rate=dropout_rate,
            causal=causal,
            softmax_scale=softmax_scale,
            **extra_impl_args
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass delegated to the implementation."""
        return self.impl(*args, **kwargs)


class LocalAttentionLayer(AttentionLayer):
    """Wrapper for local attention implementation."""
    
    def __init__(
        self,
        dropout_rate: float = 0.0,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        **extra_impl_args
    ):
        super().__init__(
            LocalAttentionBackend(),
            dropout_rate=dropout_rate,
            causal=causal,
            softmax_scale=softmax_scale,
            **extra_impl_args
        )
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply local attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            torch.Tensor: Output tensor after attention
        """
        return self.impl(q, k, v)


class DistributedAttentionLayer(AttentionLayer):
    """Wrapper for distributed attention implementation."""
    
    def __init__(
        self,
        dropout_rate: float = 0.0,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        **extra_impl_args
    ):
        super().__init__(
            DistributedAttentionBackend(),
            dropout_rate=dropout_rate,
            causal=causal,
            softmax_scale=softmax_scale,
            **extra_impl_args
        )
    
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
        return self.impl(q, k, v, replicated_q, replicated_k, replicated_v)


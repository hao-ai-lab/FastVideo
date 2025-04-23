# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func

try:
    from st_attn import sliding_tile_attention
except ImportError:
    print("Could not load Sliding Tile Attention.")
    sliding_tile_attention = None
    

from fastvideo.utils.communications import all_to_all_4D, all_gather
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from fastvideo.v1.layers.layernorm import (LayerNormScaleShift, #RMSNorm,
                                           ScaleResidual,
                                           ScaleResidualLayerNormScaleShift)
from fastvideo.v1.layers.linear import ReplicatedLinear, QKVParallelLinear
from fastvideo.v1.layers.visual_embedding import TimestepEmbedder
from fastvideo.v1.layers.rotary_embedding import (_apply_rotary_emb,
                                                  get_rotary_pos_embed)
from fastvideo.v1.attention import DistributedAttention, LocalAttention
from fastvideo.v1.platforms import _Backend
from fastvideo.v1.layers.mlp import MLP

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output

class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, head_dim, rope_split: list[int] = (64, 32, 32), bias: bool = False, with_rope: bool = True, with_qk_norm: bool = True, attn_type: str = "torch", supported_attention_backends=[_Backend.FLASH_ATTN, _Backend.TORCH_SDPA]):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.rope_split  = list(rope_split)
        self.n_heads = hidden_dim // head_dim

        self.wqkv = ReplicatedLinear(hidden_dim, hidden_dim * 3, bias=bias)
        self.wo = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)

        self.with_rope = with_rope
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)

        # self.core_attention = self.attn_processor(attn_type=attn_type)
        self.parallel = attn_type == 'parallel'
        self.attn = DistributedAttention(
            num_heads = self.n_heads,
            head_size = head_dim,
            causal    = False,
            supported_attention_backends=supported_attention_backends
        )


    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        x:   [B, S, H, D]
        cos: [S, D/2]  where D = head_dim = sum(self.rope_split)
        sin: [S, D/2]
        returns x with rotary applied exactly as v0 did
        """
        B, S, H, D = x.shape
        # 1) split cos/sin per chunk
        half_splits = [c // 2 for c in self.rope_split]  # [32,16,16] for [64,32,32]
        cos_splits = cos.split(half_splits, dim=1)
        sin_splits = sin.split(half_splits, dim=1)

        outs = []
        idx = 0
        for (chunk_size, cos_i, sin_i) in zip(self.rope_split, cos_splits, sin_splits):
            # slice the corresponding channels
            x_chunk = x[..., idx:idx + chunk_size]       # [B,S,H,chunk_size]
            idx += chunk_size

            # flatten to [S, B*H, chunk_size]
            x_flat = rearrange(x_chunk, 'b s h d -> s (b h) d')

            # apply rotary on *that* chunk
            out_flat = _apply_rotary_emb(x_flat, cos_i, sin_i, is_neox_style=True)

            # restore [B,S,H,chunk_size]
            out = rearrange(out_flat, 's (b h) d -> b s h d', b=B, h=H)
            outs.append(out)

        # concatenate back to [B,S,H,D]
        return torch.cat(outs, dim=-1)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, rope_positions=None, attn_mask=None, mask_strategy=None):
        
        B, S, _ = x.shape
        xqkv,_ = self.wqkv(x)
        xqkv = xqkv.view(*x.shape[:-1], self.n_heads, 3 * self.head_dim)
        q, k, v = torch.split(xqkv, [self.head_dim] * 3, dim=-1)  # [B,S,H,D]
        
        if self.with_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.with_rope:
            if rope_positions is not None:
                F, Ht, W = rope_positions
            assert F*Ht*W == S, "rope_positions mismatches sequence length"

            cos, sin = get_rotary_pos_embed(
                rope_sizes    = rope_positions,        # (F,H,W)
                hidden_size   = self.hidden_dim,
                heads_num     = self.n_heads,
                rope_dim_list = self.rope_split,
                rope_theta    = 1.0e4,
                dtype         = q.dtype,
            )  # each: [S, head_dim/2]
            cos = cos.to(x.device, dtype=x.dtype)
            sin = sin.to(x.device, dtype=x.dtype)
            
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        output,_ = self.attn(q, k, v)  # [B,heads,S,D]

        output = rearrange(output, 'b s h d -> b s (h d)')
        output,_ = self.wo(output)

        return output


class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, head_dim, bias=False, with_qk_norm=True, supported_attention_backends=[_Backend.FLASH_ATTN, _Backend.TORCH_SDPA]):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim

        self.wq = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)
        self.wkv = ReplicatedLinear(hidden_dim, hidden_dim * 2, bias=bias)
        self.wo = ReplicatedLinear(hidden_dim, hidden_dim, bias=bias)

        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)

        self.attn = LocalAttention(
            num_heads = self.n_heads,
            head_size = head_dim,
            causal    = False,
            supported_attention_backends=supported_attention_backends
        )

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, attn_mask=None):
        
        xq,_ = self.wq(x)
        xq = xq.view(*xq.shape[:-1], self.n_heads, self.head_dim)

        xkv,_ = self.wkv(encoder_hidden_states)
        xkv = xkv.view(*xkv.shape[:-1], self.n_heads, 2 * self.head_dim)

        xk, xv = torch.split(xkv, [self.head_dim] * 2, dim=-1)  ## seq_len, n, dim

        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        output = self.attn(xq, xk, xv)

        output = rearrange(output, 'b s h d -> b s (h d)')
        output,_ = self.wo(output)

        return output




class StepVideoTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(self,
                 dim: int,
                 attention_head_dim: int,
                 norm_eps: float = 1e-5,
                 ff_inner_dim: Optional[int] = None,
                 ff_bias: bool = False,
                 attention_type: str = 'torch'):
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNormScaleShift(dim, norm_type="layer", elementwise_affine=True, eps=norm_eps)
        self.attn1 = SelfAttention(dim,
                                   attention_head_dim,
                                   bias=False,
                                   with_rope=True,
                                   with_qk_norm=True,)

        self.norm2 = LayerNormScaleShift(dim, norm_type="layer", elementwise_affine=True, eps=norm_eps)
        self.attn2 = CrossAttention(dim, attention_head_dim, bias=False, with_qk_norm=True)

        self.ff = MLP(
                    input_dim      = dim,
                    mlp_hidden_dim = dim * 4 if ff_inner_dim is None else ff_inner_dim,
                    act_type       = "gelu_pytorch_tanh",
                    bias           = ff_bias
                )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    @torch.no_grad()
    def forward(self,
                q: torch.Tensor,
                kv: Optional[torch.Tensor] = None,
                t_expand: Optional[torch.LongTensor] = None,
                attn_mask=None,
                rope_positions: list = None,
                mask_strategy=None) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (torch.clone(chunk) for chunk in (
            self.scale_shift_table[None] + t_expand.reshape(-1, 6, self.dim)).chunk(6, dim=1))

        scale_shift_q = self.norm1(q, scale=scale_msa.squeeze(1), shift=shift_msa.squeeze(1))

        attn_q = self.attn1(scale_shift_q, rope_positions=rope_positions, mask_strategy=mask_strategy)

        q = attn_q * gate_msa + q

        attn_q = self.attn2(q, kv, attn_mask)

        q = attn_q + q

        scale_shift_q = self.norm2(q, scale=scale_mlp.squeeze(1), shift=shift_mlp.squeeze(1))

        ff_output = self.ff(scale_shift_q)

        q = ff_output * gate_mlp + q

        return q


class AdaLayerNormSingle(nn.Module):
    r"""
        Norm layer adaptive layer norm single (adaLN-single).

        As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

        Parameters:
            embedding_dim (`int`): The size of each embedding vector.
            use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, time_step_rescale=1000):
        super().__init__()

        self.emb = TimestepEmbedder(embedding_dim)

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(embedding_dim, 6 * embedding_dim, bias=True)

        self.time_step_rescale = time_step_rescale  ## timestep usually in [0, 1], we rescale it to [0,1000] for stability

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep * self.time_step_rescale)

        out,_ = self.linear(self.silu(embedded_timestep))

        return out, embedded_timestep

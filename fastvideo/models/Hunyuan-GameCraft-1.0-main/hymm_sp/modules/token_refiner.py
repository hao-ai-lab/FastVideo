from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn

from .activation_layers import get_activation_layer
from .attn_layers import attention
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, TextProjection
from .attn_layers import attention
from .mlp_layers import MLP
from .modulate_layers import apply_gate


class IndividualTokenRefinerBlock(nn.Module):
    """
    Transformer block for refining individual tokens with adaptive layer normalization.
    
    Combines self-attention and feed-forward network (FFN) layers with modulation
    based on conditional inputs (timestep and context embeddings). Supports query-key
    normalization for improved attention stability.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor, # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input token embeddings (batch_size, seq_len, hidden_size)
            c: Conditional embeddings (batch_size, hidden_size)
            attn_mask: Attention mask (batch_size, 1, seq_len, seq_len)
        
        Returns:
            Updated token embeddings after self-attention and FFN
        """
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(nn.Module):
    """
    Stack of IndividualTokenRefinerBlocks for sequential token refinement.
    
    Processes token sequences through multiple transformer blocks with
    attention masking support for handling variable-length sequences.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                act_type=act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            ) for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the stack of transformer blocks.
        
        Args:
            x: Input token embeddings (batch_size, seq_len, hidden_size)
            c: Conditional embeddings (batch_size, hidden_size)
            mask: Sequence mask indicating valid tokens (batch_size, seq_len)
        
        Returns:
            Refined token embeddings after all blocks
        """
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    Complete token refinement module with input embedding and conditional modulation.
    
    Integrates timestep embedding, context projection, and a stack of transformer
    blocks to refine token sequences based on both input data and conditional inputs.
    """
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer, **factory_kwargs)

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass of the complete token refiner.
        
        Args:
            x: Input features (batch_size, seq_len, in_channels)
            t: Timestep indices (batch_size,)
            mask: Sequence mask for variable-length inputs (batch_size, seq_len)
        
        Returns:
            Refined token embeddings (batch_size, seq_len, hidden_size)
        """
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)   # [b, s1, 1]
            context_aware_representations = (
                (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            )
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)
        
        x = self.individual_token_refiner(x, c, mask)

        return x

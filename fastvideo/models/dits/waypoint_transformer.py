# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Hugging Face Team and Overworld (original)
# Copyright (C) 2026 FastVideo Contributors (FastVideo port)
"""
Waypoint-1-Small World Model transformer for FastVideo.

This is a port of the Overworld Waypoint-1-Small model to FastVideo's
architecture, maintaining weight compatibility with the official checkpoint.

Reference: https://huggingface.co/Overworld/Waypoint-1-Small
"""

import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT

logger = init_logger(__name__)


# =============================================================================
# Control Input Dataclass
# =============================================================================

@dataclass
class CtrlInput:
    """Controller input for Waypoint world model.
    
    Attributes:
        button: Set of pressed button IDs (0-255). Uses Owl-Control keycodes.
        mouse: Tuple of (x, y) mouse velocity as floats.
        scroll: Scroll wheel value (-1, 0, or 1).
    """
    button: Set[int] = None
    mouse: Tuple[float, float] = (0.0, 0.0)
    scroll: float = 0.0
    
    def __post_init__(self):
        if self.button is None:
            self.button = set()
    
    def to_tensors(self, device: torch.device, dtype: torch.dtype, n_buttons: int = 256):
        """Convert to tensor format expected by the model.
        
        Returns:
            mouse: [1, 1, 2] tensor
            button: [1, 1, n_buttons] one-hot tensor
            scroll: [1, 1, 1] tensor
        """
        # Mouse velocity
        mouse = torch.tensor([[list(self.mouse)]], device=device, dtype=dtype)
        
        # One-hot button encoding
        button = torch.zeros(1, 1, n_buttons, device=device, dtype=dtype)
        for b in self.button:
            if 0 <= b < n_buttons:
                button[0, 0, b] = 1.0
        
        # Scroll
        scroll = torch.tensor([[[self.scroll]]], device=device, dtype=dtype)
        
        return mouse, button, scroll


# =============================================================================
# Helper Functions
# =============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization without learnable parameters."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def ada_rmsnorm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Adaptive RMS normalization with scale and bias."""
    x = rms_norm(x, eps)
    return x * (1 + scale.unsqueeze(1)) + bias.unsqueeze(1)


def ada_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply gating to output."""
    return x * gate.unsqueeze(1)


# =============================================================================
# Basic Building Blocks
# =============================================================================

class MLP(nn.Module):
    """Simple 2-layer MLP with SiLU activation."""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for output (produces scale + shift)."""
    
    def __init__(self, d_model: int):
        super().__init__()
        # Output 2*d_model: first half is scale, second half is shift
        self.fc = nn.Linear(d_model, 2 * d_model, bias=False)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, N, D] -> take first frame for modulation
        if cond.dim() == 3:
            cond = cond[:, 0:1, :]  # [B, 1, D]
        
        scale_shift = self.fc(cond)  # [B, 1, 2*D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each [B, 1, D]
        
        return rms_norm(x) * (1 + scale) + shift


class CFG(nn.Module):
    """Classifier-Free Guidance module with null embedding."""
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.null_emb = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(self, x: torch.Tensor, is_conditioned: Optional[bool] = None) -> torch.Tensor:
        B, L, _ = x.shape
        null = self.null_emb.expand(B, L, -1)
        
        if self.training or is_conditioned is None:
            if self.dropout == 0.0:
                return x
            drop = torch.rand(B, 1, 1, device=x.device) < self.dropout
            return torch.where(drop, null, x)
        
        return x if is_conditioned else null


# =============================================================================
# Controller Input Embedding
# =============================================================================

class ControllerInputEmbedding(nn.Module):
    """Embeds controller inputs (mouse + buttons + scroll) into model dimension."""
    
    def __init__(self, n_buttons: int, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        # Input: mouse (2) + buttons (n_buttons) + scroll (1) = n_buttons + 3
        self.mlp = MLP(n_buttons + 3, d_model * mlp_ratio, d_model)
    
    def forward(self, mouse: torch.Tensor, button: torch.Tensor, scroll: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mouse: [B, N, 2] mouse velocity
            button: [B, N, n_buttons] button states (one-hot or multi-hot)
            scroll: [B, N, 1] scroll wheel
        Returns:
            [B, N, d_model] control embedding
        """
        x = torch.cat((mouse, button, scroll), dim=-1)
        return self.mlp(x)


# =============================================================================
# Noise Conditioning (Wan-style)
# =============================================================================

class NoiseConditioner(nn.Module):
    """Timestep/noise level conditioner using sinusoidal embeddings."""
    
    def __init__(self, d_model: int, freq_dim: int = 512):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = MLP(freq_dim, d_model * 4, d_model)
    
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [B, N] noise levels
        Returns:
            [B, N, d_model] conditioning embedding
        """
        # Sinusoidal embedding
        half_dim = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=sigma.device, dtype=sigma.dtype) / half_dim)
        
        # sigma: [B, N] -> [B, N, 1]
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(-1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(0).unsqueeze(-1)
        
        # [B, N, half_dim]
        args = sigma * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return self.mlp(emb)


# =============================================================================
# MLPFusion for Control Conditioning
# =============================================================================

class MLPFusion(nn.Module):
    """Fuses per-frame control conditioning into tokens via MLP."""
    
    def __init__(self, d_model: int):
        super().__init__()
        # Input is concatenation of x and cond, each d_model
        self.mlp = MLP(2 * d_model, d_model, d_model)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] token features
            cond: [B, N, D] per-frame conditioning (N frames)
        Returns:
            [B, L, D] fused features
        """
        B, L, D = x.shape
        N = cond.shape[1]
        tokens_per_frame = L // N
        
        # Split fc1 weights for efficient computation
        Wx, Wc = self.mlp.fc1.weight.chunk(2, dim=1)
        
        # Reshape x to [B, N, tokens_per_frame, D]
        x_reshaped = x.view(B, N, tokens_per_frame, D)
        
        # Compute: fc1_x(x) + fc1_c(cond).unsqueeze(2)
        h = F.linear(x_reshaped, Wx) + F.linear(cond, Wc).unsqueeze(2)
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc2.weight)
        
        return y.flatten(1, 2)


# =============================================================================
# Conditioning Head (AdaLN-style modulation)
# =============================================================================

class CondHead(nn.Module):
    """Per-layer conditioning head producing 6 modulation vectors."""
    
    n_cond = 6  # scale0, bias0, gate0, scale1, bias1, gate1
    
    def __init__(self, d_model: int, noise_conditioning: str = "wan"):
        super().__init__()
        # Wan-style: add learnable bias before activation
        self.bias_in = nn.Parameter(torch.zeros(d_model)) if noise_conditioning == "wan" else None
        self.cond_proj = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_cond)
        ])
    
    def forward(self, cond: torch.Tensor):
        """
        Args:
            cond: [B, N, D] noise conditioning
        Returns:
            Tuple of 6 tensors, each [B, D] (using first frame's conditioning)
        """
        if self.bias_in is not None:
            cond = cond + self.bias_in
        
        h = F.silu(cond)
        
        # Take first frame's conditioning for modulation
        if h.dim() == 3:
            h = h[:, 0, :]  # [B, D]
        
        return tuple(proj(h) for proj in self.cond_proj)


# =============================================================================
# Attention Layers
# =============================================================================

class GateProj(nn.Module):
    """Simple wrapper to create gate_proj.weight naming in state_dict."""
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads, n_heads))
    
    def forward(self) -> torch.Tensor:
        """Return sigmoid of the gate weights."""
        return torch.sigmoid(self.weight)


class GatedSelfAttention(nn.Module):
    """Gated self-attention with GQA support."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        layer_idx: int,
        causal: bool = True,
        gated_attn: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.causal = causal
        self.gated_attn = gated_attn
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Per-head gating: learnable [n_heads, n_heads] matrix
        # Use a simple Linear-like wrapper to get gate_proj.weight naming
        if gated_attn:
            self.gate_proj = GateProj(n_heads)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        kv_cache=None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention: [B, L, n_heads, head_dim] -> [B, n_heads, L, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if pos_emb is not None:
            q = self._apply_rope(q.transpose(1, 2), pos_emb).transpose(1, 2)
            k = self._apply_rope(k.transpose(1, 2), pos_emb).transpose(1, 2)
        
        # GQA: repeat K, V heads if needed
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention using PyTorch SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            scale=self.scale,
        )
        # [B, n_heads, L, head_dim] -> [B, L, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        
        # Output projection with optional per-head gating
        out = self.out_proj(attn_out)
        
        if self.gated_attn:
            # Per-head gate: apply gate_proj [n_heads, n_heads] to attention output
            # Reshape to [B, L, n_heads, head_dim], apply gate, reshape back
            out_heads = out.view(B, L, self.n_heads, self.head_dim)
            # gate_proj.weight is [n_heads, n_heads], apply sigmoid and einsum
            gate = self.gate_proj()  # [n_heads, n_heads] with sigmoid
            # Weight contribution from each head
            out_heads = torch.einsum('blhd,gh->blgd', out_heads, gate)
            out = out_heads.reshape(B, L, D)
        
        return out
    
    def _apply_rope(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        # Simplified RoPE application
        # pos_emb: [B, L, head_dim] or similar
        # This is a placeholder - actual implementation depends on pos_emb format
        return x


class CrossAttention(nn.Module):
    """Cross-attention for prompt conditioning.
    
    Uses context_dim as inner dimension (not d_model), matching checkpoint structure.
    Uses a fixed head_dim=64 and calculates n_heads from context_dim.
    """
    
    def __init__(self, d_model: int, context_dim: int, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.context_dim = context_dim
        self.head_dim = head_dim
        # Calculate n_heads from context_dim and head_dim
        assert context_dim % head_dim == 0, f"context_dim {context_dim} must be divisible by head_dim {head_dim}"
        self.n_heads = context_dim // head_dim
        
        # Q: project from d_model to context_dim
        self.q_proj = nn.Linear(d_model, context_dim, bias=False)
        # K, V: project from context_dim to context_dim
        self.k_proj = nn.Linear(context_dim, context_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, context_dim, bias=False)
        # Out: project from context_dim back to d_model
        self.out_proj = nn.Linear(context_dim, d_model, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        _, S, _ = context.shape
        
        # [B, L, context_dim] -> [B, n_heads, L, head_dim]
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Create attention mask from padding mask if provided
        attn_mask = None
        if context_pad_mask is not None:
            # context_pad_mask: [B, S] where True = valid, False = pad
            # SDPA expects: [B, 1, 1, S] where -inf = masked, same dtype as q
            attn_mask = context_pad_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, 1, L, S)
            attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
        
        # Attention using PyTorch SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
        )
        # [B, n_heads, L, head_dim] -> [B, L, context_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.context_dim)
        
        return self.out_proj(attn_out)


# =============================================================================
# Transformer Block
# =============================================================================

class WaypointBlock(nn.Module):
    """Single Waypoint transformer block."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: int,
        layer_idx: int,
        prompt_conditioning: Optional[str],
        prompt_conditioning_period: int,
        prompt_embedding_dim: int,
        ctrl_conditioning_period: int,
        noise_conditioning: str,
        causal: bool = True,
        gated_attn: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Self-attention
        self.attn = GatedSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
            causal=causal,
            gated_attn=gated_attn,
        )
        
        # Feed-forward MLP
        self.mlp = MLP(d_model, d_model * mlp_ratio, d_model)
        
        # Conditioning head (6 modulation vectors)
        self.cond_head = CondHead(d_model, noise_conditioning)
        
        # Optional cross-attention for prompts (every prompt_conditioning_period layers)
        do_prompt_cond = (
            prompt_conditioning is not None
            and layer_idx % prompt_conditioning_period == 0
        )
        self.prompt_cross_attn = (
            CrossAttention(d_model, prompt_embedding_dim)  # Uses head_dim=64 internally
            if do_prompt_cond else None
        )
        
        # Optional MLPFusion for controls (every ctrl_conditioning_period layers)
        do_ctrl_cond = layer_idx % ctrl_conditioning_period == 0
        self.ctrl_mlpfusion = MLPFusion(d_model) if do_ctrl_cond else None
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] token features
            cond: [B, N, D] noise conditioning
            prompt_emb: [B, P, prompt_dim] prompt embeddings
            prompt_pad_mask: [B, P] prompt padding mask
            ctrl_emb: [B, N, D] control embeddings
            pos_emb: Position embeddings for RoPE
            kv_cache: KV cache for autoregressive generation
        """
        # Get modulation vectors
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)
        
        # Self-attention with AdaLN
        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x = self.attn(x, pos_emb, kv_cache)
        x = ada_gate(x, g0) + residual
        
        # Cross-attention for prompts
        if self.prompt_cross_attn is not None and prompt_emb is not None:
            x = self.prompt_cross_attn(
                rms_norm(x),
                context=rms_norm(prompt_emb),
                context_pad_mask=prompt_pad_mask,
            ) + x
        
        # MLPFusion for controls
        if self.ctrl_mlpfusion is not None and ctrl_emb is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctrl_emb)) + x
        
        # MLP with AdaLN
        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x
        
        return x


# =============================================================================
# Main Transformer Stack
# =============================================================================

class WaypointTransformer(nn.Module):
    """Stack of Waypoint transformer blocks."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.blocks = nn.ModuleList([
            WaypointBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                mlp_ratio=config.mlp_ratio,
                layer_idx=idx,
                prompt_conditioning=config.prompt_conditioning,
                prompt_conditioning_period=config.prompt_conditioning_period,
                prompt_embedding_dim=config.prompt_embedding_dim,
                ctrl_conditioning_period=config.ctrl_conditioning_period,
                noise_conditioning=config.noise_conditioning,
                causal=config.causal,
                gated_attn=config.gated_attn,
            )
            for idx in range(config.n_layers)
        ])
        
        # Share cond_head.cond_proj weights across all layers (Wan-style)
        if config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        ctrl_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        kv_cache=None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond, prompt_emb, prompt_pad_mask, ctrl_emb, pos_emb, kv_cache)
        return x


# =============================================================================
# Main WorldModel
# =============================================================================

def is_blocks(name: str) -> bool:
    """FSDP shard condition for transformer blocks."""
    return ".blocks." in name


class WaypointWorldModel(BaseDiT):
    """
    Waypoint World Model for interactive video generation.
    
    Denoises a frame given:
    - All previous frames (via KV cache)
    - The prompt embedding
    - The controller input embedding (mouse, buttons, scroll)
    - The current noise level
    """
    
    # Required class attributes for BaseDiT
    _fsdp_shard_conditions = [is_blocks]
    _compile_conditions = []
    param_names_mapping = {}  # Direct loading - checkpoint keys match model keys
    reverse_param_names_mapping = {}
    
    def __init__(self, config, hf_config: dict = None):
        # Skip BaseDiT.__init__ and call nn.Module.__init__ directly
        # because Waypoint uses a different config structure
        nn.Module.__init__(self)
        self.config = config
        self.hf_config = hf_config or {}
        
        # Required attributes for BaseDiT compatibility
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_channels_latents = config.channels
        
        # Timestep/noise conditioning
        self.denoise_step_emb = NoiseConditioner(config.d_model)
        
        # Controller input embedding
        self.ctrl_emb = ControllerInputEmbedding(
            config.n_buttons, config.d_model, config.mlp_ratio
        )
        
        # CFG modules
        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        else:
            self.ctrl_cfg = None
            
        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)
        else:
            self.prompt_cfg = None
        
        # Main transformer
        self.transformer = WaypointTransformer(config)
        
        # Patch embedding and output
        self.patch = tuple(config.patch)
        ph, pw = self.patch
        
        self.patchify = nn.Conv2d(
            config.channels,
            config.d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=False,
        )
        
        self.unpatchify = nn.Linear(
            config.d_model,
            config.channels * ph * pw,
            bias=True,
        )
        
        self.out_norm = AdaLN(config.d_model)
        
        # Position buffers for single frame
        T = config.tokens_per_frame
        idx = torch.arange(T, dtype=torch.long)
        self.register_buffer("_t_pos_1f", torch.empty(T, dtype=torch.long), persistent=False)
        self.register_buffer("_y_pos_1f", idx.div(config.width, rounding_mode="floor"), persistent=False)
        self.register_buffer("_x_pos_1f", idx.remainder(config.width), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        frame_timestamp: torch.Tensor,
        prompt_emb: Optional[torch.Tensor] = None,
        prompt_pad_mask: Optional[torch.Tensor] = None,
        mouse: Optional[torch.Tensor] = None,
        button: Optional[torch.Tensor] = None,
        scroll: Optional[torch.Tensor] = None,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C, H, W] - latent frames
            sigma: [B, N] - noise levels
            frame_timestamp: [B, N] - frame indices
            prompt_emb: [B, P, D] - prompt embeddings
            prompt_pad_mask: [B, P] - padding mask for prompts
            mouse: [B, N, 2] - mouse velocity
            button: [B, N, n_buttons] - button states
            scroll: [B, N, 1] - scroll wheel sign
            kv_cache: Optional KV cache for autoregressive generation
            
        Returns:
            [B, N, C, H, W] - denoised latent frames
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        
        assert H % ph == 0 and W % pw == 0, f"H={H}, W={W} must be divisible by patch={self.patch}"
        Hp, Wp = H // ph, W // pw
        
        # Noise conditioning
        cond = self.denoise_step_emb(sigma)  # [B, N, d_model]
        
        # Control embedding
        if button is not None:
            ctrl_emb = self.ctrl_emb(mouse, button, scroll)  # [B, N, d_model]
            if self.ctrl_cfg is not None:
                ctrl_emb = self.ctrl_cfg(ctrl_emb)
        else:
            ctrl_emb = None
        
        # Prompt CFG
        if prompt_emb is not None and self.prompt_cfg is not None:
            prompt_emb = self.prompt_cfg(prompt_emb)
        
        # Patchify: [B, N, C, H, W] -> [B, N*Hp*Wp, d_model]
        x = x.reshape(B * N, C, H, W)
        x = self.patchify(x)  # [B*N, d_model, Hp, Wp]
        x = x.view(B, N, self.config.d_model, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N * Hp * Wp, self.config.d_model)
        
        # Transformer
        x = self.transformer(
            x, cond, prompt_emb, prompt_pad_mask, ctrl_emb,
            pos_emb=None, kv_cache=kv_cache
        )
        
        # Output norm and unpatchify
        x = F.silu(self.out_norm(x, cond))
        x = self.unpatchify(x)  # [B, N*Hp*Wp, C*ph*pw]
        
        # Reshape back to image
        x = x.view(B, N, Hp, Wp, C, ph, pw)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, N, C, H, W)
        
        return x
    
    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(config)


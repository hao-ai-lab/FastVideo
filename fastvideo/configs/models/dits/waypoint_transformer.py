# SPDX-License-Identifier: Apache-2.0
"""Configuration for Waypoint-1-Small World Model transformer."""

from dataclasses import dataclass, field
from typing import List, Optional

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, _) -> bool:
    """Check if a parameter belongs to transformer blocks for FSDP sharding."""
    return "transformer.blocks" in n


@dataclass
class WaypointArchConfig(DiTArchConfig):
    """Architecture configuration for Waypoint World Model."""
    
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])
    
    # No param_names_mapping needed - model uses same names as checkpoint
    param_names_mapping: dict = field(default_factory=lambda: {})
    
    # Reverse mapping for saving checkpoints
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    
    # Model architecture
    d_model: int = 2560
    n_heads: int = 40
    n_kv_heads: int = 20  # GQA: 40 heads, 20 KV heads
    n_layers: int = 22
    mlp_ratio: int = 5
    
    # Input/output
    channels: int = 16  # VAE latent channels
    height: int = 16    # Patch grid height
    width: int = 16     # Patch grid width
    patch: tuple = (2, 2)  # Patch size
    tokens_per_frame: int = 256  # height * width after patching
    n_frames: int = 4096  # Max frames
    
    # Attention configuration
    local_window: int = 16
    global_window: int = 128
    global_attn_period: int = 4
    global_pinned_dilation: int = 8
    global_attn_offset: int = 0
    value_residual: bool = False
    gated_attn: bool = True
    causal: bool = True
    
    # Control conditioning
    n_buttons: int = 256
    ctrl_conditioning: Optional[str] = "mlp_fusion"
    ctrl_conditioning_period: int = 3
    ctrl_cond_dropout: float = 0.0
    
    # Prompt conditioning
    prompt_conditioning: Optional[str] = "cross_attention"
    prompt_conditioning_period: int = 3
    prompt_embedding_dim: int = 2048
    prompt_cond_dropout: float = 0.0
    
    # Noise conditioning (Wan-style)
    noise_conditioning: str = "wan"
    
    # Scheduler
    scheduler_sigmas: List[float] = field(default_factory=lambda: [
        1.0,
        0.8609585762023926,
        0.729332447052002,
        0.3205108940601349,
        0.0,
    ])
    
    # Misc
    base_fps: int = 60
    rope_impl: str = "ortho"
    
    def __post_init__(self):
        super().__post_init__()
        # Compute derived values
        self.hidden_size = self.d_model
        self.num_attention_heads = self.n_heads
        self.attention_head_dim = self.d_model // self.n_heads
        self.in_channels = self.channels
        self.out_channels = self.channels
        self.num_layers = self.n_layers


@dataclass  
class WaypointConfig(DiTConfig):
    """Full configuration for Waypoint World Model."""
    
    arch_config: DiTArchConfig = field(default_factory=WaypointArchConfig)
    prefix: str = "Waypoint"


# Alias for consistency with FastVideo naming conventions
WaypointWorldModelConfig = WaypointConfig


# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig

from fastvideo.configs.models.dits import CosmosVideoConfig
from fastvideo.configs.models.encoders import (BaseEncoderOutput,
                                                  T5LargeConfig)
from fastvideo.configs.models.vaes import CosmosVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_large_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess T5 Large text encoder outputs for Cosmos pipeline.
    
    Handles attention masks and sequence padding for the T5 Large model.
    """
    hidden_state = outputs.last_hidden_state
    
    if hidden_state is None:
        raise ValueError("T5 Large outputs missing last_hidden_state")
    
    mask = outputs.attention_mask
    
    # If no attention mask provided, assume all tokens are valid
    if mask is None:
        batch_size, seq_len = hidden_state.shape[:2]
        mask = torch.ones(batch_size, seq_len, device=hidden_state.device, dtype=torch.long)
    
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    # Check for NaN values and provide debugging info
    nan_count = torch.isnan(hidden_state).sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in T5 Large hidden states")
        print(f"Hidden state shape: {hidden_state.shape}")
        print(f"Hidden state dtype: {hidden_state.dtype}")
        print(f"Hidden state device: {hidden_state.device}")
        # Replace NaN values with zeros to avoid pipeline failure
        hidden_state = hidden_state.masked_fill(torch.isnan(hidden_state), 0.0)
    
    # Create list of tensors with proper sequence lengths
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    
    # Stack tensors with padding to fixed length (like wan.py implementation)
    prompt_embeds_tensor: torch.Tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in prompt_embeds
    ], dim=0)
    
    return prompt_embeds_tensor


@dataclass
class CosmosVideoConfigFixed(CosmosVideoConfig):
    """Fixed Cosmos Video Config that matches original Cosmos2 Video2World configuration."""
    
    def update_model_arch(self, config: dict) -> None:
        """Update model architecture config with HF config, but fix parameters to match original Cosmos2."""
        # First, apply the standard update
        super().update_model_arch(config)
        
        # CRITICAL FIXES to match original Cosmos2 Video2World configuration:
        
        # 1. Fix input channels: should be 16 (VAE) + 1 (condition mask) = 17
        setattr(self.arch_config, 'in_channels', 17)
        
        # 2. Fix output channels: should be 16 (VAE latent dimension)
        setattr(self.arch_config, 'out_channels', 16)
        
        # 3. Fix model architecture to match Cosmos2 2B model
        setattr(self.arch_config, 'num_attention_heads', 16)
        setattr(self.arch_config, 'attention_head_dim', 128)  # Fixed: should be 128, not 64
        setattr(self.arch_config, 'num_layers', 28)
        setattr(self.arch_config, 'hidden_size', 2048)  # 16 * 128 = 2048
        
        # 4. Fix patch size to match original
        setattr(self.arch_config, 'patch_size', (1, 2, 2))
        
        # 5. Fix max size to match original
        setattr(self.arch_config, 'max_size', (128, 240, 240))
        
        # 6. Fix text embedding dimension
        setattr(self.arch_config, 'text_embed_dim', 1024)
        
        # 7. Fix adaln lora dimension
        setattr(self.arch_config, 'adaln_lora_dim', 256)
        
        # 8. Fix rope scale to match original
        setattr(self.arch_config, 'rope_scale', (1.0, 3.0, 3.0))
        
        # 9. Enable concat padding mask
        setattr(self.arch_config, 'concat_padding_mask', True)
        
        # 10. Set num_channels_latents to 16 (VAE output dim)
        setattr(self.arch_config, 'num_channels_latents', 16)


@dataclass
class CosmosConfig(PipelineConfig):
    """Configuration for Cosmos2 Video2World pipeline matching original implementation."""

    # DiT configuration matching Cosmos2 2B model
    dit_config: DiTConfig = field(default_factory=CosmosVideoConfigFixed)
    
    # VAE configuration matching Cosmos2
    vae_config: VAEConfig = field(default_factory=CosmosVAEConfig)
    
    # Text encoding configuration
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5LargeConfig(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_large_postprocess_text, ))

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",))

    # Cosmos2 Video2World specific parameters
    conditioning_strategy: str = "frame_replace"  # Match original ConditioningStrategy.FRAME_REPLACE
    min_num_conditional_frames: int = 1
    max_num_conditional_frames: int = 2
    sigma_conditional: float = 0.0001
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder_class: str = "T5"
    
    # Denoising parameters
    embedded_cfg_scale: int = 6
    flow_shift: int = 7

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        
        # Store the VAE's latent dimension to use later
        self._vae_latent_dim = 16  # From CosmosVAEArchConfig.z_dim
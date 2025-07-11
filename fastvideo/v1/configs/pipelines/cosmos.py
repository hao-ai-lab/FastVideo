# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig

from fastvideo.v1.configs.models.dits import CosmosVideoConfig
from fastvideo.v1.configs.models.encoders import (BaseEncoderOutput,
                                                  T5LargeConfig)
from fastvideo.v1.configs.models.vaes import CosmosVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig


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
    """Fixed Cosmos Video Config that ensures out_channels matches VAE latent dimension."""
    
    def update_model_arch(self, config: dict) -> None:
        """Update model architecture config with HF config, but fix out_channels."""
        # First, apply the standard update
        super().update_model_arch(config)
        
        # CRITICAL FIX: Override out_channels to match VAE latent dimension
        # The cached config has out_channels=17, but VAE has z_dim=16
        # This ensures tensor compatibility in the scheduler
        setattr(self.arch_config, 'out_channels', 16)


@dataclass
class CosmosConfig(PipelineConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=CosmosVideoConfigFixed)
    # VAE
    vae_config: VAEConfig = field(default_factory=CosmosVAEConfig)
    # Denoising stage
    embedded_cfg_scale: int = 6
    flow_shift: int = 7

    # Text encoding stage
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

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        
        # CRITICAL FIX: Ensure transformer outputs match VAE latent dimensions
        # The cached config has out_channels=17, but VAE has z_dim=16
        # We need to override this after the model configuration is loaded
        # Store the VAE's latent dimension to use later
        self._vae_latent_dim = 16  # From CosmosVAEArchConfig.z_dim

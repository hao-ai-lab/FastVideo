# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan-GameCraft Transformer model.

This extends the HunyuanVideoTransformer3DModel with CameraNet support
for camera pose conditioning via Plücker coordinates.
"""

from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.hunyuangamecraft import (
    HunyuanGameCraftArchConfig,
    HunyuanGameCraftConfig,
)
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_shard,
    sequence_model_parallel_all_gather,
)
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import unpatchify
from fastvideo.models.cameranet import CameraNet
from fastvideo.models.dits.base import CachableDiT
from fastvideo.models.dits.hunyuanvideo import (
    HunyuanVideoTransformer3DModel,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    SingleTokenRefiner,
    FinalLayer,
    HunyuanRMSNorm,
)
from fastvideo.layers.mlp import MLP
from fastvideo.layers.visual_embedding import PatchEmbed, TimestepEmbedder


_DEFAULT_CONFIG = HunyuanGameCraftConfig()


class HunyuanGameCraftTransformer3DModel(CachableDiT):
    """
    HunyuanGameCraft Transformer with CameraNet support.
    
    This model extends the HunyuanVideo architecture with camera pose conditioning
    for controllable game video generation.
    
    The camera conditioning uses Plücker coordinate representation (6D ray format)
    which is processed through CameraNet and added to the image embeddings.
    """
    
    # Model registration
    _fsdp_shard_conditions = _DEFAULT_CONFIG.arch_config._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_CONFIG.arch_config._compile_conditions
    _supported_attention_backends = _DEFAULT_CONFIG.arch_config._supported_attention_backends
    param_names_mapping = _DEFAULT_CONFIG.arch_config.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_CONFIG.arch_config.reverse_param_names_mapping
    lora_param_names_mapping = getattr(_DEFAULT_CONFIG.arch_config, 'lora_param_names_mapping', {})
    
    def __init__(
        self, 
        config: HunyuanGameCraftArchConfig,
        hf_config: dict[str, Any],
    ):
        super().__init__(config=config, hf_config=hf_config)
        
        self.patch_size = [
            config.patch_size_t, config.patch_size, config.patch_size
        ]
        self.in_channels = config.in_channels
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = config.in_channels if config.out_channels is None else config.out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embeds = config.guidance_embeds
        self.rope_dim_list = list(config.rope_axes_dim)
        self.rope_theta = config.rope_theta
        self.text_states_dim = config.text_embed_dim
        self.text_states_dim_2 = config.pooled_projection_dim
        self.dtype = config.dtype
        
        pe_dim = config.hidden_size // config.num_attention_heads
        if sum(config.rope_axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.rope_axes_dim} but expected positional dim {pe_dim}"
            )
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        
        # Image projection
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            dtype=config.dtype,
            prefix=f"{config.prefix}.img_in"
        )
        
        # Text refinement
        self.txt_in = SingleTokenRefiner(
            self.text_states_dim,
            config.hidden_size,
            config.num_attention_heads,
            depth=config.num_refiner_layers,
            dtype=config.dtype,
            prefix=f"{config.prefix}.txt_in"
        )
        
        # Time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.time_in"
        )
        
        # Text modulation (pooled text embedding)
        self.vector_in = MLP(
            self.text_states_dim_2,
            self.hidden_size,
            self.hidden_size,
            act_type="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.vector_in"
        )
        
        # Guidance modulation (for CFG)
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size,
                act_layer="silu",
                dtype=config.dtype,
                prefix=f"{config.prefix}.guidance_in"
            )
            if self.guidance_embeds
            else None
        )
        
        # CameraNet for Plücker coordinate camera conditioning
        # Note: CameraNet outputs to VAE latent channels (16), not in_channels (33)
        camera_out_channels = 16  # VAE latent channels, same as out_channels
        self.camera_net = CameraNet(
            in_channels=config.camera_in_channels,
            downscale_coef=config.camera_downscale_coef,
            out_channels=camera_out_channels,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            dtype=config.dtype,
        )
        
        # Double stream blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                config.hidden_size,
                config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
                dtype=config.dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.double_blocks.{i}"
            )
            for i in range(config.num_layers)
        ])
        
        # Single stream blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                config.hidden_size,
                config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
                dtype=config.dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.single_blocks.{i+config.num_layers}"
            )
            for i in range(config.num_single_layers)
        ])
        
        # Final output layer
        self.final_layer = FinalLayer(
            config.hidden_size,
            self.patch_size,
            self.out_channels,
            dtype=config.dtype,
            prefix=f"{config.prefix}.final_layer"
        )
        
        self.__post_init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        camera_states: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the HunyuanGameCraft model.
        
        Args:
            hidden_states: Input image/video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D] or list
            timestep: Diffusion timestep
            encoder_hidden_states_image: Optional image embeddings
            guidance: Guidance scale for CFG
            camera_states: Camera pose tensor (Plücker coordinates) [B, F, 6, H, W]
            
        Returns:
            Denoised output tensor [B, C, T, H, W]
        """
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        enable_teacache = forward_batch is not None and forward_batch.enable_teacache
        
        if guidance is None:
            guidance = torch.tensor(
                [6016.0],
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
        
        img = x = hidden_states
        t = timestep
        
        # Split text embeddings
        if isinstance(encoder_hidden_states, torch.Tensor):
            txt = encoder_hidden_states[:, 1:]
            text_states_2 = encoder_hidden_states[:, 0, :self.text_states_dim_2]
        else:
            txt = encoder_hidden_states[0]
            text_states_2 = encoder_hidden_states[1]
        
        # Get spatial dimensions
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        
        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (tt, th, tw),
            self.hidden_size,
            self.num_attention_heads,
            self.rope_dim_list,
            self.rope_theta
        )
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)
        
        # Prepare modulation vectors
        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)
        
        # Add guidance modulation if needed
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)
        
        # Embed image
        img = self.img_in(img)
        
        # Add camera conditioning if provided (BEFORE sequence parallel sharding)
        if camera_states is not None:
            latent_len = ot  # Number of frames in latent space
            
            # Handle different frame configurations
            if latent_len == 18:
                # Two-segment generation: zero camera for first segment
                camera_latents = torch.cat([
                    self.camera_net(torch.zeros_like(camera_states)),
                    self.camera_net(camera_states)
                ], dim=1)
            elif latent_len == 9:
                # Single segment generation
                camera_latents = self.camera_net(camera_states)
            elif latent_len == 10:
                # Special case: partial zero camera
                camera_latents = torch.cat([
                    self.camera_net(torch.zeros_like(camera_states[:, 0:4, :, :, :])),
                    self.camera_net(camera_states)
                ], dim=1)
            else:
                # Default: just apply camera net
                camera_latents = self.camera_net(camera_states)
            
            # Add camera latents to image embeddings
            # Note: Both img and camera_latents should have shape [B, seq_len, hidden_size]
            img = img + camera_latents
        
        # Apply sequence parallel sharding (AFTER camera conditioning)
        img, _ = sequence_model_parallel_shard(img, dim=1)
        
        # Embed text
        txt = self.txt_in(txt, t)
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        should_skip_forward = self.should_skip_forward_for_cached_states(
            img=img, vec=vec
        )
        
        if should_skip_forward:
            img = self.retrieve_cached_states(img)
        else:
            if enable_teacache:
                original_img = img.clone()
            
            # Process through double stream blocks
            for block in self.double_blocks:
                img, txt = block(img, txt, vec, freqs_cis)
            
            # Merge txt and img for single stream blocks
            x = torch.cat((img, txt), 1)
            
            # Process through single stream blocks
            if len(self.single_blocks) > 0:
                for block in self.single_blocks:
                    x = block(x, vec, txt_seq_len, freqs_cis)
            
            # Extract image features
            img = x[:, :img_seq_len, ...]
            
            if enable_teacache:
                self.maybe_cache_states(img, original_img)
        
        # Final layer processing
        img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.final_layer(img, vec)
        
        # Unpatchify to get original shape
        img = unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)
        
        return img
    
    def maybe_cache_states(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor
    ) -> None:
        """Cache residual for TeaCache optimization."""
        self.previous_residual = hidden_states - original_hidden_states
    
    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        """Check if we should skip forward pass using cached states."""
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return False
        
        enable_teacache = forward_batch.enable_teacache
        if not enable_teacache:
            return False
        
        # TeaCache not yet fully implemented for this model
        return False
    
    def retrieve_cached_states(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve cached states for TeaCache optimization."""
        return hidden_states + self.previous_residual

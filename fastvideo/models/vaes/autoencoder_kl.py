# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from diffusers import AutoencoderKL as _DiffusersAutoencoderKL

from fastvideo.configs.models import VAEConfig


class AutoencoderKL(_DiffusersAutoencoderKL):
    
    def __init__(self, config: VAEConfig, **kwargs: Any) -> None:
        self.fastvideo_config = config
        arch = config.arch_config

        down_block_types = arch.down_block_types
        if isinstance(down_block_types, list):
            down_block_types = tuple(down_block_types)
        up_block_types = arch.up_block_types
        if isinstance(up_block_types, list):
            up_block_types = tuple(up_block_types)
        block_out_channels = arch.block_out_channels
        if isinstance(block_out_channels, list):
            block_out_channels = tuple(block_out_channels)

        latents_mean = arch.latents_mean
        if isinstance(latents_mean, list):
            latents_mean = tuple(latents_mean)
        latents_std = arch.latents_std
        if isinstance(latents_std, list):
            latents_std = tuple(latents_std)

        super().__init__(
            in_channels=arch.in_channels,
            out_channels=arch.out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=arch.layers_per_block,
            act_fn=arch.act_fn,
            latent_channels=arch.latent_channels,
            norm_num_groups=arch.norm_num_groups,
            sample_size=arch.sample_size,
            scaling_factor=arch.scaling_factor,
            shift_factor=arch.shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=arch.force_upcast,
            use_quant_conv=arch.use_quant_conv,
            use_post_quant_conv=arch.use_post_quant_conv,
            mid_block_add_attention=arch.mid_block_add_attention,
            **kwargs,
        )

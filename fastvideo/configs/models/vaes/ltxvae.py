# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LTXVAEArchConfig(VAEArchConfig):
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    decoder_causal: bool = False
    encoder_causal: bool = True
    in_channels: int = 3
    latent_channels: int = 128
    layers_per_block: tuple[int, ...] = (4, 3, 3, 3, 4)
    out_channels: int = 3
    patch_size: int = 4
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-06
    scaling_factor: float = 1.0
    spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False)

    # Additional fields that might be inherited from base class
    z_dim: int = 128  # Using latent_channels as z_dim
    is_residual: bool = False
    clip_output: bool = True

    def __post_init__(self):
        # Calculate compression ratios based on patch sizes and downsampling
        self.temporal_compression_ratio = self.patch_size_t
        # Spatial compression is usually patch_size * product of spatial downsampling
        self.spatial_compression_ratio = self.patch_size * (2**(
            len(self.block_out_channels) - 1))

        if isinstance(self.scaling_factor, int | float):
            self.scaling_factor_tensor: torch.Tensor = torch.tensor(
                self.scaling_factor)


@dataclass
class LTXVAEConfig(VAEConfig):
    arch_config: LTXVAEArchConfig = field(default_factory=LTXVAEArchConfig)
    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __post_init__(self):
        if hasattr(self, 'tile_sample_min_num_frames') and hasattr(
                self, 'tile_sample_stride_num_frames'):
            self.blend_num_frames = (self.tile_sample_min_num_frames -
                                     self.tile_sample_stride_num_frames) * 2

# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 VAE configuration for MAGI-2 I2V reference images."""

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig, WanVAEConfig


MAGI2_WAN_LATENTS_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)

MAGI2_WAN_LATENTS_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


@dataclass
class Magi2WanVAEArchConfig(WanVAEArchConfig):
    """Match the published Wan2.2 VAE encoder architecture and normalization."""

    in_channels: int = 12
    out_channels: int = 12
    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    is_residual: bool = True
    clip_output: bool = False
    latents_mean: tuple[float, ...] = MAGI2_WAN_LATENTS_MEAN
    latents_std: tuple[float, ...] = MAGI2_WAN_LATENTS_STD
    patch_size: int | None = 2
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 16


@dataclass
class Magi2WanVAEConfig(WanVAEConfig):
    """Load only the Wan encoder used by MAGI-2 image conditioning."""

    arch_config: WanVAEArchConfig = field(default_factory=Magi2WanVAEArchConfig)
    load_encoder: bool = True
    load_decoder: bool = False

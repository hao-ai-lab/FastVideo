# SPDX-License-Identifier: Apache-2.0
"""Configuration for the distilled MAGI-2 Turbo VAE decoder."""

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Magi2TurboVAEArchConfig(VAEArchConfig):
    """Describe the latent and output geometry used by Turbo VAE."""

    architectures: list[str] = field(default_factory=lambda: ["Magi2TurboVAEModel"])
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 48
    decoder_block_out_channels: tuple[int, ...] = (64, 128, 256, 512)
    decoder_causal: bool = False
    decoder_layers_per_block: tuple[int, ...] = (2, 2, 2, 3, 3)
    patch_size: int = 2
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-6
    scaling_factor: float = 1.0
    decoder_spatio_temporal_scaling: tuple[bool, ...] = (False, True, True, True)
    decoder_spatio_only: tuple[bool, ...] = (False, True, False, False)
    decoder_is_dw_conv: tuple[bool, ...] = (False, False, False, False, False)
    decoder_dw_kernel_size: int = 5
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16
    first_chunk_size: int = 7
    step_size: int = 7
    use_unpatchify: bool = True


@dataclass
class Magi2TurboVAEConfig(VAEConfig):
    """Provide the published Turbo VAE JSON and checkpoint paths."""

    arch_config: VAEArchConfig = field(default_factory=Magi2TurboVAEArchConfig)
    config_path: str = ""
    checkpoint_path: str = ""
    pretrained_dtype: str = "bfloat16"
    load_encoder: bool = False
    load_decoder: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class FluxVAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    scaling_factor: float = 0.18215
    scale_factor_spatial: int = 8
    scale_factor_temporal: int = 1
    vae_scale_factor: int = 8

    def __post_init__(self) -> None:
        self.spatial_compression_ratio = self.scale_factor_spatial
        self.temporal_compression_ratio = self.scale_factor_temporal
        if not getattr(self, "vae_scale_factor", None):
            self.vae_scale_factor = self.scale_factor_spatial


@dataclass
class FluxVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=FluxVAEArchConfig)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False


@dataclass
class Flux2VAEArchConfig(FluxVAEArchConfig):
    scaling_factor: float = 0.13025


@dataclass
class Flux2VAEConfig(FluxVAEConfig):
    arch_config: VAEArchConfig = field(default_factory=Flux2VAEArchConfig)

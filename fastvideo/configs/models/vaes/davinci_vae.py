# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig, WanVAEConfig


@dataclass
class DaVinciVAEArchConfig(WanVAEArchConfig):
    """Wan 2.2 VAE with z_dim=48 for daVinci-MagiHuman.

    The Wan2.2-TI2V-5B VAE uses the same architecture as WanVAE but with
    z_dim=48 instead of 16.  Per-channel normalization stats are not published
    for z_dim=48; these stubs must be replaced with calibrated values.
    TODO: calibrate real per-channel latent_mean / latent_std on a diverse set
          of videos encoded with the Wan2.2 VAE at z_dim=48.
    """
    z_dim: int = 48
    latents_mean: tuple[float, ...] = field(
        default_factory=lambda: tuple([0.0] * 48))
    latents_std: tuple[float, ...] = field(
        default_factory=lambda: tuple([1.0] * 48))


@dataclass
class DaVinciVAEConfig(WanVAEConfig):
    arch_config: DaVinciVAEArchConfig = field(
        default_factory=DaVinciVAEArchConfig)

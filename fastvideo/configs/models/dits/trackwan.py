# SPDX-License-Identifier: Apache-2.0
"""Config for the track-conditioned (MotionStream-style) Wan DiT.

Defaults target the Wan2.1-T2V-1.3B @ 480p bring-up teacher. The per-size architecture dims
(num_layers, num_attention_heads, ffn_dim, ...) are pulled from the checkpoint's ``config.json``
at load time (``ModelConfig.update_model_arch``), so only the track-specific fields and the widened
``in_channels`` are declared here.

``in_channels`` is widened to ``num_channels_latents + track_channels`` (the track conditioning is
channel-concatenated onto the latent). ``out_channels`` stays at the latent channel count because
the model still predicts the latent. The converted init checkpoint's ``config.json`` must carry
the widened ``in_channels`` (and ideally ``track_config``); these dataclass values are the fallback.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig


@dataclass
class TrackWanVideoArchConfig(WanVideoArchConfig):
    # I2V + track input layout, concatenated at the patch embed:
    #   16 (noisy latent) + 20 (I2V: 4 mask + 16 first-frame latent) + 16 (track) = 52.
    # out_channels stays 16 (the model still predicts the latent).
    in_channels: int = 52
    out_channels: int = 16

    # MotionStream-style track conditioning. FastVideo-specific; carried in the checkpoint
    # config.json so it survives load (update_model_arch copies matching fields).
    track_config: dict = field(
        default_factory=lambda: {
            "id_dim": 128,                  # sinusoidal track-ID embedding dim (the "d" in c_m)
            "track_channels": 16,           # track head output channels concatenated at patch embed
            "vae_spatial_compression": 8,   # Wan VAE spatial downsample (latent grid = H/8 x W/8)
            "vae_temporal_compression": 4,  # Wan VAE temporal downsample
            "max_track_id": 100_000,        # range of random per-track IDs
            "zero_init_head": True,         # zero-init final conv so step-0 == teacher
        })


@dataclass
class TrackWanVideoConfig(WanVideoConfig):
    arch_config: WanVideoArchConfig = field(default_factory=TrackWanVideoArchConfig)
    prefix: str = "Wan"

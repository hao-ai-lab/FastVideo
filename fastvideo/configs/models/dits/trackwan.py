# SPDX-License-Identifier: Apache-2.0
"""Configuration for track-conditioned Wan transformers."""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig


def _default_track_config() -> dict[str, int | bool]:
    return {
        "id_dim": 64,
        "track_channels": 16,
        "vae_spatial_compression": 8,
        "vae_temporal_compression": 4,
        "max_track_id": 100_000,
        "zero_init_head": False,
        "use_bias": False,
    }


@dataclass
class TrackWanVideoArchConfig(WanVideoArchConfig):
    """Wan I2V input widened with a latent-aligned point-track map.

    Channel order is fixed and checkpoint-visible:
    noisy latent (16), I2V mask (4), first-frame latent (16), track map (16).
    """

    in_channels: int = 52
    out_channels: int = 16
    image_dim: int = 1280
    added_kv_proj_dim: int | None = 5120
    track_config: dict[str, int | bool] = field(default_factory=_default_track_config)

    def __post_init__(self) -> None:
        super().__post_init__()
        track_channels = int(self.track_config.get("track_channels", 0))
        expected_in_channels = self.num_channels_latents + 20 + track_channels
        if track_channels <= 0:
            raise ValueError("track_config.track_channels must be positive")
        if self.in_channels != expected_in_channels:
            raise ValueError("TrackWan in_channels must equal latent channels + 20 I2V "
                             f"channels + track channels; got {self.in_channels}, expected "
                             f"{expected_in_channels}")


@dataclass
class TrackWanVideoConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=TrackWanVideoArchConfig)
    prefix: str = "Wan"


@dataclass
class CausalTrackWanVideoArchConfig(TrackWanVideoArchConfig):
    """Explicit config type for the causal TrackWan architecture."""


@dataclass
class CausalTrackWanVideoConfig(TrackWanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=CausalTrackWanVideoArchConfig)

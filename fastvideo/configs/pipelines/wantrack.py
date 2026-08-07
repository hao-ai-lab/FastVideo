# SPDX-License-Identifier: Apache-2.0
"""Pipeline configs for causal WanTrack Self-Forcing inference."""

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.models.dits.trackwan import CausalTrackWanVideoConfig
from fastvideo.configs.pipelines.wan import WanI2V480PConfig


@dataclass
class CausalTrackWanSFI2VConfig(WanI2V480PConfig):
    """Causal TrackWan I2V distilled with Self-Forcing (DMD timesteps)."""

    dit_config: DiTConfig = field(default_factory=CausalTrackWanVideoConfig)
    is_causal: bool = True
    flow_shift: float | None = 6.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 750, 500, 250])
    warp_denoising_step: bool = True
    context_noise: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Match the SF recipe used for Track-v0 / wantrack causal synth stage2.
        arch = self.dit_config.arch_config
        arch.local_attn_size = 6
        arch.sink_size = 1
        arch.rope_cache_policy = "relativistic"
        arch.num_frames_per_block = 3

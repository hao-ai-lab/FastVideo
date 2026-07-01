# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig


@dataclass
class DreamXWorldArchConfig(WanVideoArchConfig):
    """DreamX-World DiT config with camera PRoPE control fields."""

    add_control_adapter: bool = True
    cam_method: str | None = "prope"
    attn_compress: int = 1
    cam_self_attn_layers: tuple[int, ...] | None = None


@dataclass
class DreamXWorldConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=DreamXWorldArchConfig)

    prefix: str = "Wan"

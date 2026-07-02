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


@dataclass
class DreamXWorldARArchConfig(DreamXWorldArchConfig):
    """DreamX-World-5B autoregressive causal DiT config."""

    model_type: str = "ti2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    text_dim: int = 4096
    freq_dim: int = 256
    attn_compress: int = 4
    cam_self_attn_layers: tuple[int, ...] | None = tuple(range(30))
    local_attn_size: int = 12
    sink_size: int = 3
    num_frames_per_block: int = 3
    rope_cache_policy: str = "block_relativistic"
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)


@dataclass
class DreamXWorldARConfig(DreamXWorldConfig):
    arch_config: DiTArchConfig = field(default_factory=DreamXWorldARArchConfig)

    prefix: str = "Wan"

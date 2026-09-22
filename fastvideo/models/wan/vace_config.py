# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE transformer architecture config and checkpoint mappings."""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.models.wan.config import WanVideoArchConfig


@dataclass
class WanVACEArchConfig(WanVideoArchConfig):
    """Diffusers ``WanVACETransformer3DModel`` arch surface for Wan2.1-VACE."""

    vace_in_channels: int = 96
    vace_layers: list[int] = field(default_factory=lambda: [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
    ])

    param_names_mapping: dict = field(default_factory=lambda: {
        **WanVideoArchConfig().param_names_mapping,
        r"^vace_patch_embedding\.(.*)$": r"vace_patch_embedding.proj.\1",
        r"^vace_blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"vace_blocks.\1.to_q.\2",
        r"^vace_blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"vace_blocks.\1.to_k.\2",
        r"^vace_blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"vace_blocks.\1.to_v.\2",
        r"^vace_blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"vace_blocks.\1.to_out.\2",
        r"^vace_blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"vace_blocks.\1.norm_q.\2",
        r"^vace_blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"vace_blocks.\1.norm_k.\2",
        r"^vace_blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"vace_blocks.\1.attn2.to_out.\2",
        r"^vace_blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"vace_blocks.\1.ffn.fc_in.\2",
        r"^vace_blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"vace_blocks.\1.ffn.fc_out.\2",
        r"^vace_blocks\.(\d+)\.norm2\.(.*)$": r"vace_blocks.\1.self_attn_residual_norm.norm.\2",
        r"^vace_blocks\.(\d+)\.proj_in\.(.*)$": r"vace_blocks.\1.proj_in.\2",
        r"^vace_blocks\.(\d+)\.proj_out\.(.*)$": r"vace_blocks.\1.proj_out.\2",
    })

    def __post_init__(self) -> None:
        super().__post_init__()
        if max(self.vace_layers) >= self.num_layers:
            raise ValueError(f"VACE layers {self.vace_layers} exceed num_layers={self.num_layers}.")
        if 0 not in self.vace_layers:
            raise ValueError("VACE layers must include layer 0.")


@dataclass
class WanVACEVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanVACEArchConfig)

    prefix: str = "WanVACE"

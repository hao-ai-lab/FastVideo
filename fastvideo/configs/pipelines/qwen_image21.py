# SPDX-License-Identifier: Apache-2.0
"""Native Qwen-Image-2.1 inference configuration."""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.qwen_image21 import QwenImage21Config
from fastvideo.configs.models.encoders.qwen_image21 import QwenImage21Qwen3VLConfig
from fastvideo.configs.models.vaes.qwen_image21 import QwenImage21VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class QwenImage21PipelineConfig(PipelineConfig):
    dit_config: QwenImage21Config = field(default_factory=QwenImage21Config)
    vae_config: QwenImage21VAEConfig = field(default_factory=QwenImage21VAEConfig)
    text_encoder_configs: tuple[QwenImage21Qwen3VLConfig,
                                ...] = field(default_factory=lambda: (QwenImage21Qwen3VLConfig(), ))
    text_encoder_precisions: tuple[str, ...] = ("bf16", )
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_tiling: bool = True
    vae_sp: bool = False
    # Four output channels are required to preserve transparency through PNG export.
    output_channels: int = 4
    flow_shift: float | None = None
    scheduler_step_in_fp32: bool = True
    # Runtime placement policy. CPU keeps ten reference prefixes out of VRAM.
    kv_cache_device: str = "cpu"

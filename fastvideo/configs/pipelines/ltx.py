# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, VAEConfig
from fastvideo.configs.models.dits import LTXVideoConfig
from fastvideo.configs.models.vaes import LTXVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class LTXConfig(PipelineConfig):
    """Base configuration for LTX pipeline architecture."""

    # DiT
    dit_config: DiTConfig = field(default_factory=LTXVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=LTXVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "bf16"
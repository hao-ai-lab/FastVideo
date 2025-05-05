from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch

from fastvideo.v1.configs.models import DiTConfig, VAEConfig
from fastvideo.v1.configs.models.dits import StepVideoConfig
from fastvideo.v1.configs.models.vaes import StepVideoVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig



@dataclass
class StepVideoT2VConfig(PipelineConfig):
    """Base configuration for StepVideo pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=StepVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=StepVideoVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: int = 13

    # Text encoding stage
    # text_encoder_configs: Tuple[EncoderConfig, ...] = field(
    #     default_factory=lambda: (T5Config(), ))
    # postprocess_text_funcs: Tuple[Callable[[BaseEncoderOutput], torch.tensor],
    #                               ...] = field(default_factory=lambda:
    #                                            (t5_postprocess_text, ))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "bf16"
    # text_encoder_precisions: Tuple[str, ...] = field(
    #     default_factory=lambda: ("fp32", ))


    # def __post_init__(self):
    #     self.vae_config.load_encoder = False
    #     self.vae_config.load_decoder = True

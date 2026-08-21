# SPDX-License-Identifier: Apache-2.0
import html
from dataclasses import dataclass, field

import ftfy
import torch

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.models.dits.lingbotworld2 import LingBotWorld2CausalFastVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, LingBotWorld2UMT5Config
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.wan import Wan2_2_I2V_A14B_Config


def lingbotworld2_whitespace_preprocess(prompt: str) -> str:
    text = ftfy.fix_text(prompt)
    text = html.unescape(html.unescape(text))
    return " ".join(text.strip().split())


def lingbotworld2_t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    assert outputs.last_hidden_state is not None
    return outputs.last_hidden_state


@dataclass
class LingBotWorld2CausalFastI2V480PConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorld2CausalFastVideoConfig)
    vae_config: WanVAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (LingBotWorld2UMT5Config(), ))
    preprocess_text_funcs: tuple = field(default_factory=lambda: (lingbotworld2_whitespace_preprocess, ))
    postprocess_text_funcs: tuple = field(default_factory=lambda: (lingbotworld2_t5_postprocess_text, ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947
    vae_precision: str = "fp32"
    vae_decode_precision: str | None = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False
    dit_precision: str = "bf16"
    is_causal: bool = False

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

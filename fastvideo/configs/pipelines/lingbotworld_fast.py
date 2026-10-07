# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits.lingbotworld_fast import (
    LingBotWorldFastVideoConfig, )
from fastvideo.configs.models.encoders import T5Config
from fastvideo.configs.pipelines.lingbotworld2 import (
    LingBotWorld2CausalFastI2V480PConfig, )


@dataclass
class LingBotWorldFastI2V480PConfig(LingBotWorld2CausalFastI2V480PConfig):
    """Pipeline config for LingBot-World-Fast 480P image-to-video.

    Shares the LingBot World 2 causal-fast sampling loop and Wan VAE, but this
    checkpoint ships the stock ``UMT5EncoderModel`` text encoder (``d_model``
    fields) rather than LingBot World 2's custom one (``dim`` fields), so the
    standard ``T5Config`` is restored here alongside this DiT's arch config.
    """

    dit_config: DiTConfig = field(default_factory=LingBotWorldFastVideoConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))

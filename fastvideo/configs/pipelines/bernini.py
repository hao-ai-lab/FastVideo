# SPDX-License-Identifier: Apache-2.0
import html
import re
from dataclasses import dataclass, field

import ftfy

from fastvideo.configs.models.dits.bernini import BerniniRConfig
from fastvideo.configs.models.encoders.bernini import BerniniT5ArchConfig
from fastvideo.configs.models.encoders.t5 import T5Config
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig


def clean_bernini_prompt(prompt: str) -> str:
    """Match the official double HTML-unescape/whitespace cleanup before T5."""
    return re.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(prompt)))).strip()


@dataclass
class BerniniRPipelineConfig(WanT2V480PConfig):
    dit_config: BerniniRConfig = field(default_factory=BerniniRConfig)
    # Official reference encodes and decodes the Wan VAE in float32.
    vae_decode_precision: str = "fp32"
    # BerniniRendererModel defaults its UMT5 and DiT to bf16 (VAE stays fp32).
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    text_encoder_configs: tuple = field(default_factory=lambda: (T5Config(arch_config=BerniniT5ArchConfig()), ))
    preprocess_text_funcs: tuple = field(default_factory=lambda: (clean_bernini_prompt, ))

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig
from fastvideo.configs.models.encoders.reason1 import Reason1ArchConfig

@dataclass
class CosmosPredictTextEncoderArchConfig(Reason1ArchConfig):
    """Arch config for Cosmos Predict text encoder. It is basically Qwen2.5-VL-7B-Instruct."""
    pass

@dataclass
class CosmosPredictTextEncoderConfig(TextEncoderConfig):
    """Cosmos Predict text encoder config."""
    arch_config: CosmosPredictTextEncoderArchConfig = field(default_factory=CosmosPredictTextEncoderArchConfig)
    tokenizer_type: str = "Qwen/Qwen2.5-VL-7B-Instruct"

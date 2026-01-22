# SPDX-License-Identifier: Apache-2.0
"""SigLIP vision encoder configuration for FastVideo."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (ImageEncoderArchConfig,
                                                    ImageEncoderConfig)


@dataclass
class SiglipVisionArchConfig(ImageEncoderArchConfig):
    """Architecture configuration for SigLIP vision encoder.
    
    Fields match the config.json from HuggingFace SigLIP checkpoints.
    """

    # From config.json
    architectures: list[str] = field(
        default_factory=lambda: ["SiglipVisionModel"])
    attention_dropout: float = 0.0
    dtype: str | None = None
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 384
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14

    # FastVideo specific - QKV fusion mapping
    stacked_params_mapping: list = field(default_factory=lambda: [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ])


@dataclass
class SiglipVisionConfig(ImageEncoderConfig):
    """Configuration for SigLIP vision encoder."""

    arch_config: ImageEncoderArchConfig = field(
        default_factory=SiglipVisionArchConfig)

    # FastVideo specific
    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    enable_scale: bool = True
    is_causal: bool = False
    prefix: str = "siglip"

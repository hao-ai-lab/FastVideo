# SPDX-License-Identifier: Apache-2.0
"""Native DFN5B CLIP conditioner configurations for MMAudio."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.clip import (
    CLIPTextArchConfig,
    CLIPTextConfig,
    CLIPVisionArchConfig,
    CLIPVisionConfig,
)


@dataclass
class MMAudioDFNCLIPTextArchConfig(CLIPTextArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["MMAudioDFNCLIPTextEncoder"])
    vocab_size: int = 49408
    hidden_size: int = 1024
    intermediate_size: int = 4096
    projection_dim: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 77
    text_len: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 49406
    eos_token_id: int = 49407


@dataclass
class MMAudioDFNCLIPTextConfig(CLIPTextConfig):
    arch_config: CLIPTextArchConfig = field(default_factory=MMAudioDFNCLIPTextArchConfig)
    prefix: str = "mmaudio_dfn_clip_text"


@dataclass
class MMAudioDFNCLIPVisionArchConfig(CLIPVisionArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["MMAudioDFNCLIPVisionEncoder"])
    hidden_size: int = 1280
    intermediate_size: int = 5120
    projection_dim: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    image_size: int = 378
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5


@dataclass
class MMAudioDFNCLIPVisionConfig(CLIPVisionConfig):
    arch_config: CLIPVisionArchConfig = field(default_factory=MMAudioDFNCLIPVisionArchConfig)
    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = True
    enable_scale: bool = True
    is_causal: bool = False
    prefix: str = "mmaudio_dfn_clip_vision"

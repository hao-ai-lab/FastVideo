# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig


@dataclass
class DreamXWorldArchConfig(WanVideoArchConfig):
    """DreamX-World DiT config with camera PRoPE control fields."""

    add_control_adapter: bool = True
    cam_method: str | None = "prope"
    attn_compress: int = 1
    cam_self_attn_layers: tuple[int, ...] | None = None


@dataclass
class DreamXWorldConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=DreamXWorldArchConfig)

    prefix: str = "Wan"


@dataclass
class DreamXWorldARArchConfig(DreamXWorldArchConfig):
    """DreamX-World-5B autoregressive causal DiT config."""

    model_type: str = "ti2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    text_dim: int = 4096
    freq_dim: int = 256
    attn_compress: int = 4
    cam_self_attn_layers: tuple[int, ...] | None = tuple(range(30))
    local_attn_size: int = 12
    sink_size: int = 3
    num_frames_per_block: int = 3
    rope_cache_policy: str = "block_relativistic"
    # The official AR checkpoint (AMAP-ML/DreamX-World ``model.safetensors``)
    # already uses FastVideo's native key names and the converter copies the
    # tensors verbatim, so every rule is an identity. The rules enumerate the
    # full state-dict surface of ``DreamXWorldARTransformer3DModel`` (norm1 /
    # norm2 / head.norm are affine-free and have no parameters).
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.\1",
            r"^text_embedding\.([02])\.(.*)$": r"text_embedding.\1.\2",
            r"^time_embedding\.([02])\.(.*)$": r"time_embedding.\1.\2",
            r"^time_projection\.1\.(.*)$": r"time_projection.1.\1",
            r"^blocks\.(\d+)\.self_attn\.(q|k|v|o)\.(.*)$": r"blocks.\1.self_attn.\2.\3",
            r"^blocks\.(\d+)\.self_attn\.norm_(q|k)\.weight$": r"blocks.\1.self_attn.norm_\2.weight",
            r"^blocks\.(\d+)\.cross_attn\.(q|k|v|o)\.(.*)$": r"blocks.\1.cross_attn.\2.\3",
            r"^blocks\.(\d+)\.cross_attn\.norm_(q|k)\.weight$": r"blocks.\1.cross_attn.norm_\2.weight",
            r"^blocks\.(\d+)\.cam_self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(.*)$": r"blocks.\1.cam_self_attn.\2.\3",
            r"^blocks\.(\d+)\.cam_self_attn\.norm_(q|k)\.weight$": r"blocks.\1.cam_self_attn.norm_\2.weight",
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.norm3.\2",
            r"^blocks\.(\d+)\.ffn\.([02])\.(.*)$": r"blocks.\1.ffn.\2.\3",
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.modulation",
            r"^head\.head\.(.*)$": r"head.head.\1",
            r"^head\.modulation$": r"head.modulation",
        })
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)


@dataclass
class DreamXWorldARConfig(DreamXWorldConfig):
    arch_config: DiTArchConfig = field(default_factory=DreamXWorldARArchConfig)

    prefix: str = "Wan"

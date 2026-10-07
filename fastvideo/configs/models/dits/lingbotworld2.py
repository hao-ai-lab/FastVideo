# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LingBotWorld2CausalFastArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    model_type: str = "i2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 36
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

    local_attn_size: int = 18
    sink_size: int = 6
    chunk_size: int = 4
    sample_shift: float = 10.0
    num_train_timesteps: int = 1000
    timesteps_index: tuple[int, int, int, int] = (0, 250, 500, 750)
    max_area: int = 480 * 832

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_heads
        self.attention_head_dim = self.dim // self.num_heads
        self.in_channels = self.in_dim
        self.out_channels = self.out_dim
        self.num_channels_latents = self.out_dim


@dataclass
class LingBotWorld2CausalFastVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingBotWorld2CausalFastArchConfig)

    prefix: str = "Wan"

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def is_zimage_block(name: str, module) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in {"noise_refiner", "context_refiner", "layers"} and parts[-1].isdigit()


@dataclass
class ZImageDiTArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_zimage_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )

    all_patch_size: tuple[int, ...] = (2, )
    all_f_patch_size: tuple[int, ...] = (1, )
    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1536, 512, 512)

    adaln_embed_dim: int = 256
    frequency_embedding_size: int = 256
    timestep_mid_size: int = 1024
    max_period: int = 10000
    seq_multi_of: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.all_patch_size) != len(self.all_f_patch_size):
            raise ValueError("all_patch_size and all_f_patch_size must have equal length")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.dim // self.n_heads != sum(self.axes_dims):
            raise ValueError("attention head dimension must equal sum(axes_dims)")
        if len(self.axes_dims) != len(self.axes_lens) or any(dim % 2 for dim in self.axes_dims):
            raise ValueError("RoPE axes require matching lengths and even dimensions")

        self.hidden_size = self.dim
        self.num_attention_heads = self.n_heads
        self.num_channels_latents = self.in_channels
        self.out_channels = self.in_channels


@dataclass
class ZImageDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=ZImageDiTArchConfig)
    prefix: str = "ZImage"

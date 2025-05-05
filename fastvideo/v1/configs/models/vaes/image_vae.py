from dataclasses import dataclass, field
from typing import Tuple, Optional

from fastvideo.v1.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class ImageVAEArchConfig(VAEArchConfig):
    in_channels: int = 3,
    out_channels: int = 3,
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
    block_out_channels: Tuple[int] = (64,),
    layers_per_block: int = 1,
    act_fn: str = "silu",
    latent_channels: int = 4,
    norm_num_groups: int = 32,
    sample_size: int = 32,
    scaling_factor: float = 0.18215,
    shift_factor: Optional[float] = None,
    latents_mean: Optional[Tuple[float]] = None,
    latents_std: Optional[Tuple[float]] = None,
    force_upcast: float = True,
    use_quant_conv: bool = True,
    use_post_quant_conv: bool = True,
    mid_block_add_attention: bool = True,

    def __post_init__(self):
        self.spatial_compression_ratio: int = 2**(len(self.block_out_channels) -
                                                  1)
        self.temporal_compression_ratio = 1

@dataclass
class ImageVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=ImageVAEArchConfig)

    # overrides VAEConfig
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
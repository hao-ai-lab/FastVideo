from dataclasses import dataclass
from fastvideo.configs.models.upsamplers.base import UpsamplerConfig

@dataclass
class SRTo720pUpsamplerConfig(UpsamplerConfig):
    in_channels: int = 0
    out_channels: int = 0
    hidden_channels: int = 64
    num_blocks: int = 6
    global_residual: bool = False

@dataclass
class SRTo1080pUpsamplerConfig(UpsamplerConfig):
    z_channels: int = 0
    out_channels: int = 0
    block_out_channels: tuple[int, ...] = (0, 0)
    num_res_blocks: int = 2
    is_residual: bool = False
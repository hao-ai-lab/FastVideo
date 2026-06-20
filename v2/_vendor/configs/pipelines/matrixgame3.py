# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from v2._vendor.configs.models import DiTConfig
from v2._vendor.configs.models.dits.matrixgame3 import MatrixGame3WanVideoConfig
from v2._vendor.configs.pipelines.wan import WanT2V480PConfig


@dataclass
class MatrixGame3I2V720PConfig(WanT2V480PConfig):
    dit_config: DiTConfig = field(default_factory=MatrixGame3WanVideoConfig)
    flow_shift: float | None = 5.0
    vae_precision: str = "fp32"

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.vae_config.use_light_vae = True

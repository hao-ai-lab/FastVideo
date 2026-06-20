# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from v2._vendor.configs.models import DiTConfig
from v2._vendor.configs.pipelines.wan import Wan2_2_I2V_A14B_Config
from v2._vendor.configs.models.dits.lingbotworld import LingBotWorldVideoConfig


@dataclass
class LingBotWorldI2V480PConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947

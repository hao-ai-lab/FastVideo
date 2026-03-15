# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)


@dataclass
class WanGameVideoArchConfig(WanVideoArchConfig):
    """Wangame keeps WanVideo architecture defaults and checkpoint mappings."""


@dataclass
class WanGameVideoConfig(WanVideoConfig):
    arch_config: WanGameVideoArchConfig = field(
        default_factory=WanGameVideoArchConfig
    )
    prefix: str = "WanGame"

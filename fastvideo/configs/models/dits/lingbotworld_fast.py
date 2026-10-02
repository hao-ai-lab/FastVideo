# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.configs.models.dits.lingbotworld2 import (
    LingBotWorld2CausalFastArchConfig, )


@dataclass
class LingBotWorldFastArchConfig(LingBotWorld2CausalFastArchConfig):
    """Arch config for the released LingBot-World-Fast checkpoint.

    The tensor layout is identical to the LingBot World 2 causal-fast model, so
    the parent's shapes and ``param_names_mapping`` are reused verbatim. Only
    the sampling/attention-window values released with this checkpoint differ.
    """

    # The released `generate_fast.py` leaves `--local_attn_size` at -1, so
    # self-attention stays global and the KV cache never evicts. That also makes
    # `sink_size` unreachable, so it is deliberately not pinned here (the loader
    # overwrites it from the checkpoint config either way).
    local_attn_size: int = -1
    # `chunk_size` and `timesteps_index` are absent from the checkpoint config,
    # so these values are what the sampling loop actually runs with.
    chunk_size: int = 3
    timesteps_index: tuple[int, int, int, int] = (0, 179, 358, 679)


@dataclass
class LingBotWorldFastVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingBotWorldFastArchConfig)

    prefix: str = "Wan"

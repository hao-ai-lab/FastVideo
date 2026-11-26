# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex transformer configuration.

The released Pika adapters do not change MiniMax-H3's transformer: the
checkpoint keeps the same packed video/audio trunk, Qwen3-VL conditioner, and
VAEs, and applies a rank-128 PEFT adapter at inference.  This subclass exists so
the Pika pipeline names its own component config without duplicating the H3
architecture contract.
"""

from dataclasses import dataclass

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config


@dataclass
class PikaH3FlexDiTConfig(MiniMaxH3Config):
    """The released MiniMax-H3 transformer, sampled through Pika's adapters."""


__all__ = ["PikaH3FlexDiTConfig"]

# SPDX-License-Identifier: Apache-2.0
"""Bernini's released UMT5 tokenization contract."""

from dataclasses import dataclass

from fastvideo.configs.models.encoders.t5 import T5ArchConfig


@dataclass
class BerniniT5ArchConfig(T5ArchConfig):

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_kwargs["padding"] = "max_length"

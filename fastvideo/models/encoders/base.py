# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from fastvideo.configs.models.encoders import (BaseEncoderOutput, ImageEncoderConfig, TextEncoderConfig)
from fastvideo.platforms import AttentionBackendEnum

TextEncoderOutputT = TypeVar("TextEncoderOutputT")


class TextEncoder(nn.Module, ABC, Generic[TextEncoderOutputT]):
    """Base for native encoders with a model-specific forward output contract."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [])
    _stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = TextEncoderConfig()._supported_attention_backends
    supported_checkpoint_quantization_methods: frozenset[str] = frozenset()

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._fsdp_shard_conditions = config._fsdp_shard_conditions
        self._stacked_params_mapping = config.arch_config.stacked_params_mapping
        if not self.supported_attention_backends:
            raise ValueError(f"Subclass {self.__class__.__name__} must define _supported_attention_backends")

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> TextEncoderOutputT:
        pass

    @property
    def supported_attention_backends(self) -> tuple[AttentionBackendEnum, ...]:
        return self._supported_attention_backends


class ImageEncoder(nn.Module, ABC):
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = ImageEncoderConfig()._supported_attention_backends

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(f"Subclass {self.__class__.__name__} must define _supported_attention_backends")

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> tuple[AttentionBackendEnum, ...]:
        return self._supported_attention_backends

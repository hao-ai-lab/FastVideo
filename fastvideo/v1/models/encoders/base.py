# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import field
from typing import List, Optional, Tuple

import torch
from torch import nn

from fastvideo.v1.configs.models.encoders import (BaseEncoderOutput,
                                                  ImageEncoderConfig,
                                                  TextEncoderConfig)
from fastvideo.v1.platforms import AttentionBackendEnum


class TextEncoder(nn.Module, ABC):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [])
    _stacked_params_mapping: List[Tuple[str, str,
                                        str]] = field(default_factory=list)
    _supported_attention_backends: Tuple[
        AttentionBackendEnum,
        ...] = TextEncoderConfig()._supported_attention_backends

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._fsdp_shard_conditions = config._fsdp_shard_conditions
        self._stacked_params_mapping = config.arch_config.stacked_params_mapping
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                input_ids: Optional[torch.Tensor],
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = None,
                **kwargs) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> Tuple[AttentionBackendEnum, ...]:
        return self._supported_attention_backends


class ImageEncoder(nn.Module, ABC):
    _supported_attention_backends: Tuple[
        AttentionBackendEnum,
        ...] = ImageEncoderConfig()._supported_attention_backends

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor,
                **kwargs) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> Tuple[AttentionBackendEnum, ...]:
        return self._supported_attention_backends

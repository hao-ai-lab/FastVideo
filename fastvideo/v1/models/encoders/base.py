from abc import ABC, abstractmethod

import torch
from torch import nn

from fastvideo.v1.configs.models.encoders import (BaseEncoderOutput,
                                                  ImageEncoderConfig,
                                                  TextEncoderConfig)
from fastvideo.v1.platforms import _Backend


class TextEncoder(nn.Module, ABC):
    _supported_attention_backends: tuple[
        _Backend, ...] = TextEncoderConfig()._supported_attention_backends

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                input_ids: torch.Tensor | None,
                position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                output_hidden_states: bool | None = None,
                **kwargs) -> BaseEncoderOutput:
        pass

    @property
    def supported_attention_backends(self) -> tuple[_Backend, ...]:
        return self._supported_attention_backends


class ImageEncoder(nn.Module, ABC):
    _supported_attention_backends: tuple[
        _Backend, ...] = ImageEncoderConfig()._supported_attention_backends

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
    def supported_attention_backends(self) -> tuple[_Backend, ...]:
        return self._supported_attention_backends

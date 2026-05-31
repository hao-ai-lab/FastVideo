# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from fastvideo.configs.models import DiTConfig
from fastvideo.platforms import AttentionBackendEnum


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict
    reverse_param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: tuple[
        AttentionBackendEnum, ...] = DiTConfig()._supported_attention_backends

    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_fsdp_shard_conditions", "param_names_mapping",
            "_compile_conditions"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any],
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.hf_config = hf_config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        pass

    def __post_init__(self) -> None:
        required_attrs = [
            "hidden_size", "num_attention_heads", "num_channels_latents"
        ]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )

    @property
    def supported_attention_backends(self) -> tuple[AttentionBackendEnum, ...]:
        return self._supported_attention_backends
